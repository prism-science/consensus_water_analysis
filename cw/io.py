from __future__ import annotations

import json
from pathlib import Path

import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import numpy as np
import pandas as pd
from loguru import logger

from cw.metrics import (
    consensus_centers,
    consensus_water_mask,
    per_structure_consensus_pr,
)

# ── utilities ─────────────────────────────────────────────────────────────────


def normalize_ins_code(ins_code) -> str:
    if ins_code is None or (isinstance(ins_code, float) and pd.isna(ins_code)):
        return ""
    value = str(ins_code).strip()
    return "" if value in {"", "?"} else value


def read_cohort(txt_path: Path) -> list[str]:
    """Read PDB IDs from a cohort .txt file.

    Blank lines and lines starting with '#' are ignored.
    Returns lowercase 4-char PDB IDs in file order, de-duplicated (first
    occurrence kept). A cohort is a set of distinct structures, so a repeated ID
    would otherwise be loaded twice — double-counting it in every stage and, for
    clustering, doubling its water density and inflating the cluster_occupancy
    denominator.
    Each line is split on whitespace; only the first token is used (inline
    comments and extra fields are ignored). Any suffix after the first '_' is
    stripped (e.g. '5F14_final' → '5f14').
    """
    ids = []
    for line in txt_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        token = line.split()[0]
        ids.append(token.split("_")[0].lower())
    unique_ids = list(dict.fromkeys(ids))
    if len(unique_ids) != len(ids):
        logger.warning(
            f"{len(ids) - len(unique_ids)} duplicate PDB ID(s) in {txt_path.name}; "
            "keeping first of each"
        )
    return unique_ids


def cif_path_for(pdb_id: str, all_pdb_redo_dir: Path | str, cif_template: str) -> Path:
    """Resolve the CIF path for a PDB ID."""
    return Path(all_pdb_redo_dir) / cif_template.format(pdb_id=pdb_id)


def find_cohort_metadata(data_dir: Path | str, cohort: str) -> Path | None:
    """Path to a cohort directory's metadata.csv, walking up trailing `_<token>`
    segments of the directory name until one exists.

    Water-level subsets (`<cohort>_iso`, `<cohort>_bfactor…`) do not re-deposit
    per-structure metadata — resolution and R-free are per-PDB, shared across
    water filters — so the file lives in the parent cohort directory. Returns None
    when no ancestor name has one.
    """
    probe = cohort
    while True:
        path = Path(data_dir) / probe / "metadata.csv"
        if path.exists():
            return path
        if "_" not in probe:
            return None
        probe = probe.rsplit("_", 1)[0]


def load_cohort_frames(data_dir, cohort, cutoff, match_radius, suffix=""):
    """(per-water frame, per-structure frame, one-line summary) for one cohort.

    Both frames carry a `cohort` column and a `group` column holding the split
    label, resolved here because each cohort has its own cluster ids, occupancies
    and consensus centers. A water is consensus iff it is a within-cutoff member of
    a cluster whose occupancy clears `cutoff`. `suffix` selects a member-radius
    variant of the CSVs (e.g. "_0.5"); "" reads the default pair.
    """
    directory = Path(data_dir) / cohort
    clusters = pd.read_csv(directory / f"clusters{suffix}.csv")
    members = pd.read_csv(directory / f"cluster_members{suffix}.csv")

    is_consensus = consensus_water_mask(members, clusters, cutoff)
    waters = members.assign(
        group=np.where(is_consensus, "consensus", "nonconsensus"),
        cohort=cohort,
    )

    # Per-structure precision/recall/f1/num_water against the consensus centers,
    # joined to deposited metadata. The merge renames metadata's own num_water to
    # num_water_deposited so the clustered count stays the plain num_water;
    # metadata scalars can carry "<missing>" strings, so coerce to numeric.
    centers = consensus_centers(clusters, cutoff)
    pr = per_structure_consensus_pr(members, centers, match_radius)
    meta_path = find_cohort_metadata(data_dir, cohort)
    if meta_path is None:
        structures, note = pr, "no metadata.csv found"
    else:
        meta = pd.read_csv(meta_path)
        scalars = [c for c in meta.columns if c != "pdb_id"]
        meta[scalars] = meta[scalars].apply(pd.to_numeric, errors="coerce")
        structures = pr.merge(meta, on="pdb_id", how="left", suffixes=("", "_deposited"))
        matched = int(pr["pdb_id"].isin(meta["pdb_id"]).sum())
        note = f"metadata {meta_path.parent.name}/ ({matched}/{len(pr)} matched)"

    n_consensus = int((waters["group"] == "consensus").sum())
    summary = (
        f"{cohort}: {len(members)} waters, {n_consensus} consensus / "
        f"{len(members) - n_consensus} non-consensus, {len(clusters)} clusters, "
        f"{len(pr)} structures, {note}"
    )
    return waters, structures.assign(cohort=cohort), summary


def water_oxygen_mask(atoms: struc.AtomArray) -> np.ndarray:
    """Boolean mask selecting water oxygen atoms (HOH, hetero, element O)."""
    return (atoms.res_name == "HOH") & atoms.hetero & (atoms.element == "O")


def protein_heavy_mask(atoms: struc.AtomArray) -> np.ndarray:
    """Boolean mask selecting protein heavy atoms (non-hetero, non-hydrogen)."""
    return (~atoms.hetero) & (atoms.element != "H")


def count_water_oxygens(atoms: struc.AtomArray) -> int:
    """Count water oxygens (each altloc counted separately)."""
    return int(water_oxygen_mask(atoms).sum())


# ── CIF reading ───────────────────────────────────────────────────────────────


def load_edia(json_path: Path) -> dict[tuple[str, int, str], float] | None:
    """Load EDIAm scores from a per-structure JSON file.

    Deprecated for clustering use — prefer load_structure_waters, which adds
    altloc_id by pairing JSON entries with the CIF atom_site ordering.  This
    function silently overwrites duplicate keys for altloc waters.

    Keyed on (chain_id, res_id, ins_code) → EDIAm float.
    Returns None if the file is missing or unreadable.
    """
    if not json_path.exists():
        return None
    try:
        with open(json_path, encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None

    lookup: dict[tuple[str, int, str], float] = {}
    for entry in payload:
        if entry.get("compID") not in {"HOH", "WAT"}:
            continue
        pdb_info = entry.get("pdb", {})
        key = (
            str(pdb_info.get("strandID", "")),
            int(pdb_info.get("seqNum", 0)),
            normalize_ins_code(pdb_info.get("insCode", "")),
        )
        lookup[key] = float(entry.get("EDIAm", np.nan))
    return lookup


def load_edia_all_altlocs(json_path: Path) -> dict[tuple[str, int, str], list[float]]:
    """Like load_edia, but accumulates a list per residue instead of overwriting on
    duplicate keys. This is used to preserve waters with altlocs, but altloc id is not
    explicitly stored in the JSON file.

    Keyed on (chain_id, res_id, ins_code) → [EDIAm, ...] in JSON file order.
    Returns an empty dict on missing file or parse error.
    """
    if not json_path.exists():
        return {}
    try:
        with open(json_path, encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return {}

    scores: dict[tuple[str, int, str], list[float]] = {}
    for entry in payload:
        if entry.get("compID") not in {"HOH", "WAT"}:
            continue
        pdb_info = entry.get("pdb", {})
        key = (
            str(pdb_info.get("strandID", "")),
            int(pdb_info.get("seqNum", 0)),
            normalize_ins_code(pdb_info.get("insCode", "")),
        )
        scores.setdefault(key, []).append(float(entry.get("EDIAm", np.nan)))
    return scores


def edia_scores_in_order(
    keys: list[tuple[str, int, str]],
    edia_lists: dict[tuple[str, int, str], list[float]],
) -> np.ndarray:
    """Score each ordered (chain_id, res_id, ins_code) key against per-residue EDIAm lists.

    Single source of truth for the positional altloc-pairing contract: the EDIA JSON
    has no altloc field and emits one entry per altloc variant in file order, which
    matches the CIF atom_site order biotite preserves with altloc='all'. So the Nth
    occurrence of a key is paired with the Nth score in that key's list. Keys with no
    matching score (missing residue or position past the list end) become NaN.

    Both cw.io._attach_edia (DataFrame rows) and cw.filter.keep_by_edia (water-O atoms)
    build their keys and delegate here so the pairing rule lives in one place.
    """
    scores = np.full(len(keys), np.nan)
    seen: dict[tuple[str, int, str], int] = {}
    for i, key in enumerate(keys):
        pos = seen.get(key, 0)
        seen[key] = pos + 1
        vals = edia_lists.get(key)
        if vals is not None and pos < len(vals):
            scores[i] = vals[pos]
    return scores


def load_protein(
    src: Path | pdbx.CIFFile, altloc: str = "occupancy"
) -> tuple[struc.AtomArray, int]:
    """Load protein-only atoms (highest-occupancy altloc) and return (array, res_id_offset).

    The offset renumbers the first residue to 1 so BLOSUM62 alignment is
    robust across structures with different deposited numbering schemes.

    src may be a Path (CIF is read here) or an already-loaded CIFFile
    (caller's read is reused, avoiding a second disk hit).
    """
    cif_file = pdbx.CIFFile.read(src) if isinstance(src, Path) else src
    atoms = pdbx.get_structure(cif_file, model=1, altloc=altloc)

    protein_mask = struc.filter_amino_acids(atoms)
    protein = atoms[protein_mask]

    offset = int(protein.res_id[0]) - 1
    if offset > 0:
        protein.res_id = protein.res_id - offset

    return protein, offset


def _attach_edia(
    df: pd.DataFrame, edia_dict_of_lists: dict[tuple[str, int, str], list[float]]
) -> pd.DataFrame:
    """Attach EDIAm scores to df via the shared positional altloc-pairing contract.

    Builds one (chain_id, res_id, ins_code) key per CIF row in atom_site order and
    defers to edia_scores_in_order, which pairs the Nth row of a residue group with
    the Nth score in that group's list.

    df must still contain ins_code. Adds an 'edia' column.
    """
    keys = [
        (str(chain_id), int(res_id), ins_code)
        for chain_id, res_id, ins_code in zip(
            df["chain_id"], df["res_id"], df["ins_code"], strict=True
        )
    ]
    df["edia"] = edia_scores_in_order(keys, edia_dict_of_lists)
    return df


def load_structure_waters(
    cif_path: Path | pdbx.CIFFile,
    json_path: Path | None = None,
    pdb_id: str | None = None,
) -> pd.DataFrame:
    """Load water O records from one CIF, optionally attaching EDIA scores.

    Reads the CIF once with altloc='all'. EDIA is joined while ins_code is still
    present, then ins_code is dropped before returning.

    cif_path may be a Path or an already-loaded CIFFile; pass pdb_id explicitly
    in the latter case.

    Columns: pdb_id, chain_id, res_id, altloc, x, y, z, b_factor, occupancy, edia
    """
    if isinstance(cif_path, Path):
        pdb_id = pdb_id if pdb_id is not None else cif_path.stem.removesuffix("_final")
        cf = pdbx.CIFFile.read(cif_path)
    else:
        cf = cif_path
        pdb_id = pdb_id if pdb_id is not None else "structure"
    atoms = pdbx.get_structure(cf, model=1, altloc="all", extra_fields=["b_factor", "occupancy"])
    waters = atoms[water_oxygen_mask(atoms)]

    df = pd.DataFrame(
        {
            "pdb_id": pdb_id,
            "chain_id": waters.chain_id.astype(str),
            "res_id": waters.res_id.astype(int),
            "ins_code": [normalize_ins_code(ic) for ic in waters.ins_code],
            "altloc": waters.altloc_id.astype(str),
            "x": waters.coord[:, 0].astype(float),
            "y": waters.coord[:, 1].astype(float),
            "z": waters.coord[:, 2].astype(float),
            "b_factor": waters.b_factor.astype(float),
            "occupancy": waters.occupancy.astype(float),
        }
    )

    if json_path is not None:
        df = _attach_edia(df, load_edia_all_altlocs(json_path))
    else:
        df["edia"] = float("nan")

    return df.drop(columns=["ins_code"])


def resolve_aligned_water_inputs(
    member_ids: list[str],
    aligned_dir: Path,
    *,
    edia_dir: Path | str,
    edia_template: str,
) -> list[tuple[Path, Path | None]]:
    """Resolve (aligned_cif, edia_json) pairs for members with an aligned CIF.

    Members without an aligned CIF in aligned_dir are skipped, so the returned list length is
    the count of members actually found. The edia entry of a pair is None when that score
    file is absent for the member. The result feeds straight into collect_aligned_waters.
    """
    pairs: list[tuple[Path, Path | None]] = []
    for member_id in member_ids:
        cif = Path(aligned_dir) / f"{member_id}.cif"
        if not cif.exists():
            continue
        edia = Path(edia_dir) / edia_template.format(pdb_id=member_id)
        pairs.append((cif, edia if edia.exists() else None))
    return pairs


def collect_aligned_waters(
    cif_json_pairs: list[tuple[Path, Path | None]],
    *,
    out_path: Path | None = None,
) -> pd.DataFrame:
    """Concatenate water records from aligned CIFs into one DataFrame.

    Each element of cif_json_pairs is (aligned_cif, edia_json_path), where
    edia_json_path may be None if the EDIA score is unavailable for the
    structure. If out_path is given the result is also written to CSV before
    returning.
    """
    frames = [load_structure_waters(cif, json) for cif, json in cif_json_pairs]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
    return df


# ── CIF writing ───────────────────────────────────────────────────────────────


def write_water_points_cif(
    out_path: Path,
    coords: np.ndarray,
    *,
    chain_id: str | np.ndarray = "A",
    res_id: np.ndarray | None = None,
    occupancy: float | np.ndarray = 1.0,
    b_factor: float | np.ndarray = 0.0,
    data_block: str = "points",
) -> None:
    """Write 3D points as HOH water-oxygen atoms in an mmCIF file.

    Coordinates go through pdbx.set_structure, which writes Cartn_x/y/z at full
    float precision (no PDB 8.3f column-width limit) — safe for points in offset
    crystal frames. Shared by write_cluster_cif and by the batch point lists fed
    to phenix.map_value_at_point, where input atom order == output value order.
    """
    coords = np.asarray(coords, dtype=float)
    n = len(coords)
    atoms = struc.AtomArray(n)
    atoms.coord = coords
    atoms.chain_id = (
        np.full(n, chain_id, dtype="U4")
        if isinstance(chain_id, str)
        else np.asarray(chain_id).astype("U4")
    )
    atoms.res_id = (
        np.arange(1, n + 1, dtype=int) if res_id is None else np.asarray(res_id, dtype=int)
    )
    atoms.res_name[:] = "HOH"
    atoms.atom_name[:] = "O"
    atoms.element[:] = "O"
    atoms.hetero[:] = True
    atoms.add_annotation("b_factor", dtype=float)
    atoms.b_factor[:] = b_factor
    atoms.add_annotation("occupancy", dtype=float)
    atoms.occupancy[:] = occupancy

    cif_file = pdbx.CIFFile()
    pdbx.set_structure(cif_file, atoms, data_block=data_block)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cif_file.write(out_path)


def write_cluster_cif(
    out_path: Path,
    *,
    clusters_df: pd.DataFrame | None = None,
    members_df: pd.DataFrame | None = None,
    clusters_csv: Path | None = None,
    members_csv: Path | None = None,
    include_noise: bool = False,
) -> None:
    """Write cluster centers (and optionally noise points) as a CIF file.

    Cluster centers are written as HOH oxygens on chain 'A':
      res_id = cluster_id + 1, occupancy = cluster_occupancy,
      B-factor = RMS spread sqrt((std_x² + std_y² + std_z²) / 3).
    Noise waters (cluster_id == -1) are written on chain 'B' when include_noise is True.

    Pass DataFrames directly (from build_cluster_tables) or CSV paths to load from
    disk — useful when called independently after the pipeline has already run.
    """
    if clusters_df is None:
        if clusters_csv is None:
            raise ValueError("provide clusters_df or clusters_csv")
        clusters_df = pd.read_csv(clusters_csv)

    noise_df: pd.DataFrame | None = None
    noise_occupancy: float = 0.0
    if include_noise:
        if members_df is None:
            if members_csv is None:
                raise ValueError("include_noise=True requires members_df or members_csv")
            members_df = pd.read_csv(members_csv)
        n_structures = members_df["pdb_id"].nunique()
        noise_occupancy = 1.0 / n_structures if n_structures > 0 else 0.0
        noise_df = members_df[members_df["cluster_id"] == -1].reset_index(drop=True)

    centers = pd.DataFrame(
        {
            "chain_id": "A",
            "res_id": clusters_df["cluster_id"].to_numpy(dtype=int) + 1,
            "x": clusters_df["center_x"].to_numpy(dtype=float),
            "y": clusters_df["center_y"].to_numpy(dtype=float),
            "z": clusters_df["center_z"].to_numpy(dtype=float),
            "b_factor": np.sqrt(
                (clusters_df["std_x"] ** 2 + clusters_df["std_y"] ** 2 + clusters_df["std_z"] ** 2)
                / 3
            ).to_numpy(dtype=float),
            "occupancy": clusters_df["cluster_occupancy"].to_numpy(dtype=float),
        }
    )

    frames = [centers]

    if noise_df is not None and not noise_df.empty:
        noise = pd.DataFrame(
            {
                "chain_id": "B",
                "res_id": np.arange(1, len(noise_df) + 1, dtype=int),
                "x": noise_df["x"].to_numpy(dtype=float),
                "y": noise_df["y"].to_numpy(dtype=float),
                "z": noise_df["z"].to_numpy(dtype=float),
                "b_factor": noise_df["b_factor"].fillna(0.0).to_numpy(dtype=float),
                "occupancy": np.full(len(noise_df), noise_occupancy),
            }
        )
        frames.append(noise)

    rows = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

    write_water_points_cif(
        out_path,
        rows[["x", "y", "z"]].to_numpy(dtype=float),
        chain_id=rows["chain_id"].to_numpy().astype("U4"),
        res_id=rows["res_id"].to_numpy(dtype=int),
        occupancy=rows["occupancy"].to_numpy(dtype=float),
        b_factor=rows["b_factor"].to_numpy(dtype=float),
        data_block="clusters",
    )


def write_filtered_cif(
    src: Path | pdbx.CIFFile,
    keep_mask: np.ndarray,
    filtered_atoms: struc.AtomArray,
    out_path: Path,
) -> None:
    """Write a filtered CIF preserving all atom_site columns (including label_alt_id).

    Drops atom_site rows where keep_mask is False, then writes Cartn_x/y/z
    from filtered_atoms.coord so relocated waters are captured.
    pdbx.set_structure is not used because it regenerates atom_site from the
    AtomArray, overwriting label_alt_id with biotite's internal altloc_id values,
    which do not round-trip the original CIF '.'/'A'/'B' notation exactly.

    src may be a Path (CIF is read here) or an already-loaded CIFFile.
    """
    cif_file = pdbx.CIFFile.read(src) if isinstance(src, Path) else src
    atom_site = cif_file.block["atom_site"]

    filtered_data = {key: atom_site[key].as_array(str)[keep_mask] for key in atom_site}
    filtered_data["Cartn_x"] = filtered_atoms.coord[:, 0]
    filtered_data["Cartn_y"] = filtered_atoms.coord[:, 1]
    filtered_data["Cartn_z"] = filtered_atoms.coord[:, 2]
    # A new CIFCategory is required here rather than mutating the existing one's columns
    # in-place: CIFCategory caches _row_count the first time row_count is accessed
    # (e.g. via get_structure), and __setitem__ never resets it, so serialize() would
    # raise SerializationError comparing the stale original length against the filtered columns.
    cif_file.block["atom_site"] = pdbx.CIFCategory(filtered_data, "atom_site")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cif_file.write(out_path)


def write_transformed_cif(
    src: Path | pdbx.CIFFile,
    R: np.ndarray,
    t: np.ndarray,
    out_path: Path,
) -> None:
    """Apply a rigid transform to all atoms in a CIF and write to out_path.

    Modifies Cartn_x/y/z directly so every other column — including
    label_alt_id — is preserved unchanged.

    src may be a Path (CIF is read here) or an already-loaded CIFFile.
    """
    cif_file = pdbx.CIFFile.read(src) if isinstance(src, Path) else src
    atom_site = cif_file.block["atom_site"]

    x = atom_site["Cartn_x"].as_array(float)
    y = atom_site["Cartn_y"].as_array(float)
    z = atom_site["Cartn_z"].as_array(float)
    coords = np.column_stack([x, y, z])
    coords = (R @ coords.T).T + t

    atom_site["Cartn_x"] = coords[:, 0]
    atom_site["Cartn_y"] = coords[:, 1]
    atom_site["Cartn_z"] = coords[:, 2]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cif_file.write(out_path)
