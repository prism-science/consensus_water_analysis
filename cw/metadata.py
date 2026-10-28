from __future__ import annotations

import re
import time
from pathlib import Path

import gemmi
import numpy as np
import requests
from loguru import logger

_RCSB_ENTRY_URL = "https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
_WATER_COMPS = frozenset({"HOH", "WAT", "DOD"})
_PDB_CODE_RE = re.compile(r"\b([0-9][A-Za-z0-9]{3})\b")


def _pdb_codes(values: list[str]) -> list[str]:
    """Lowercased PDB codes found in free-text starting-model strings, de-duped in order."""
    return list(
        dict.fromkeys(code.lower() for v in values if v for code in _PDB_CODE_RE.findall(v))
    )


def max_cell_diff(cell_a: gemmi.UnitCell, cell_b: gemmi.UnitCell) -> float:
    """Largest relative percent difference across the six unit-cell parameters.

    For each of (a, b, c, alpha, beta, gamma), computes
    ``200 * |p1 - p2| / (p1 + p2)`` and returns the maximum.
    """
    p1 = np.array([cell_a.a, cell_a.b, cell_a.c, cell_a.alpha, cell_a.beta, cell_a.gamma])
    p2 = np.array([cell_b.a, cell_b.b, cell_b.c, cell_b.alpha, cell_b.beta, cell_b.gamma])
    return float(np.max(200.0 * np.abs(p1 - p2) / (p1 + p2)))


def _cif_first_float(block: gemmi.cif.Block, tag: str) -> float | str:
    """Return first parseable float for a loop column or scalar tag, else '<missing>'."""
    for v in block.find_loop(tag):
        s = gemmi.cif.as_string(v)
        if s and s not in (".", "?"):
            try:
                return float(s)
            except ValueError:
                pass
    val = block.find_value(tag)
    if val is not None:
        s = gemmi.cif.as_string(val)
        if s and s not in (".", "?"):
            try:
                return float(s)
            except ValueError:
                pass
    return "<missing>"


def _fetch_rcsb_entry(pdb_id: str) -> dict | None:
    """Fetch RCSB entry JSON. Returns None on 404 or persistent failure."""
    url = _RCSB_ENTRY_URL.format(pdb_id=pdb_id.lower())
    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=30)
        except requests.RequestException:
            if attempt == 2:
                return None
            time.sleep(2.0)
            continue
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 404:
            return None
        if attempt == 2:
            return None
        time.sleep(2.0)
    return None


def resolve_starting_model(entry: dict | None, pdb_id: str = "?") -> tuple[list[str], str, str]:
    """Resolve the starting model from an RCSB entry JSON.

    Returns ``(codes, status, csv_value)``:
      - ``codes``      resolved PDB codes (accession-preferred union), lowercased.
      - ``status``     ``'ok'`` (exactly one code) | ``'conflict'`` (>1 code, or the
                       two RCSB fields disagree) | ``'text'`` (free text, no code) |
                       ``'missing'`` (no entry / no field).
      - ``csv_value``  the exact string stored in the metadata.csv ``starting_model``
                       column: ``" | ".join(...)`` of codes (or lowercased free text),
                       or ``"<missing>"``.

    Starting model lives in one of two RCSB categories:
      _pdbx_initial_refinement_model.accession_code → bare PDB code(s)
      _refine.pdbx_starting_model                   → free text ("PDB entry 1CIL", "none", …)
    accession_code is the structured field and never omits a code the free-text field
    carries, so it is preferred; the free text is parsed for codes only as a fallback,
    and its raw value is kept when it carries no code ("none", "in house …").
    """
    if entry is None:
        return [], "missing", "<missing>"

    refine_raw = [
        v.strip()
        for r in (entry.get("refine") or [])
        for v in [r.get("pdbx_starting_model")]
        if v and v.strip()
    ]
    accession_raw = [
        v.strip()
        for m in (entry.get("pdbx_initial_refinement_model") or [])
        for v in [m.get("accession_code")]
        if v and v.strip()
    ]

    refine_codes = _pdb_codes(refine_raw)
    accession_codes = _pdb_codes(accession_raw)
    conflict = bool(refine_codes and accession_codes and set(refine_codes) != set(accession_codes))
    if conflict:
        logger.warning(
            f"{pdb_id}: starting model disagreement — "
            f"_refine.pdbx_starting_model={refine_codes} vs "
            f"_pdbx_initial_refinement_model.accession_code={accession_codes} "
            f"(keeping both)"
        )

    if accession_codes or refine_codes:
        # Union, accession first (preferred). When they agree this collapses to the
        # single code; when they disagree both are kept so the conflict is visible
        # in the CSV, not just the terminal warning.
        codes = list(dict.fromkeys(accession_codes + refine_codes))
        status = "conflict" if (conflict or len(codes) > 1) else "ok"
        return codes, status, " | ".join(codes)
    if refine_raw:
        texts = list(dict.fromkeys(v.lower() for v in refine_raw))
        return [], "text", " | ".join(texts)
    return [], "missing", "<missing>"


def resolve_deposited_r_factors(entry: dict | None) -> tuple[float | str, float | str]:
    """Deposited R-work and R-free from an RCSB entry JSON's ``refine`` array.

    These are the *deposited* values from RCSB, distinct from the local re-refined
    ``r_work`` / ``r_free`` read out of the on-disk CIF. Returns the first parseable
    value across ``refine`` blocks for each factor, else ``'<missing>'``.
    """
    if entry is None:
        return "<missing>", "<missing>"
    refine_blocks = entry.get("refine") or []

    def _first(key: str) -> float | str:
        for r in refine_blocks:
            v = r.get(key)
            if v is not None:
                try:
                    return float(v)
                except (TypeError, ValueError):
                    pass
        return "<missing>"

    return _first("ls_R_factor_R_work"), _first("ls_R_factor_R_free")


def resolve_diffrn_temp(entry: dict | None) -> float | str:
    """Lowest ``_diffrn.ambient_temp`` (K) on an RCSB entry, or ``'<missing>'``.

    The re-refined local CIFs mostly drop this category, so it comes from the same
    entry JSON already fetched for experiment_condition. Lowest wins when an entry
    lists several datasets: it is the temperature that determines whether the
    solvent is frozen, which is what a water analysis cares about.
    """
    if entry is None:
        return "<missing>"
    temps = []
    for block in entry.get("diffrn") or []:
        value = block.get("ambient_temp")
        if value is None:
            continue
        try:
            temps.append(float(value))
        except (TypeError, ValueError):
            continue
    return min(temps) if temps else "<missing>"


def resolve_crystal_grow(entry: dict | None) -> tuple[float | str, float | str]:
    """``(ph, crystal_grow_temp)`` from an RCSB entry's ``exptl_crystal_grow`` array.

    These are crystallization conditions — the same category ``experiment_condition``
    is parsed from — and are distinct from ``diffrn_temp``, which is the temperature
    the diffraction data were collected at. Returns the first parseable value across
    blocks for each field, else ``'<missing>'``.

    ``ph`` prefers the scalar ``pH``; when an entry gives only ``pdbx_pH_range`` the
    raw range string is passed through verbatim ('4.4-4.8'). Both fields are optional
    and roughly half of a typical cohort supplies neither, so the column holds floats
    and range strings side by side — read it as object dtype, not float.
    """
    if entry is None:
        return "<missing>", "<missing>"
    blocks = entry.get("exptl_crystal_grow") or []

    def _first(key: str) -> float | str:
        for block in blocks:
            value = block.get(key)
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    pass
        return "<missing>"

    ph = _first("pH")
    if ph == "<missing>":
        for block in blocks:
            ph_range = str(block.get("pdbx_pH_range") or "").strip()
            if ph_range:
                ph = ph_range
                break
    return ph, _first("temp")


def metadata_row(cif_path: Path, pdb_id: str) -> dict:
    """Extract one metadata CSV row from a local mmCIF file + RCSB API.

    Library split:
      - gemmi.read_structure → space_group, unit_cell, resolution, num_water
      - gemmi.cif.read       → r_work, r_free (local re-refined CIF has these)
                               ligand_names (from _pdbx_entity_nonpoly.comp_id loop)
      - RCSB Data API        → experiment_condition, starting_model,
                               deposited_r_work, deposited_r_free, ph,
                               crystal_grow_temp, diffrn_temp
                               (absent from re-refined local CIFs)
    """
    # --- gemmi structure: space group, unit cell, resolution ---
    st = gemmi.read_structure(str(cif_path))
    sg = st.find_spacegroup()
    space_group = sg.hm if sg is not None else "<missing>"
    cell = st.cell
    resolution = st.resolution if st.resolution > 0.0 else "<missing>"

    # --- biotite: water oxygen count across all altlocs ---
    # altloc="all" returns a flat AtomArray with one entry per altloc variant,
    # so a water with altlocs A and B contributes two oxygen atoms.
    import biotite.structure.io.pdbx as pdbx

    from cw.io import count_water_oxygens

    _cif = pdbx.CIFFile.read(cif_path)
    _atoms = pdbx.get_structure(_cif, model=1, altloc="all", use_author_fields=True)
    num_water = count_water_oxygens(_atoms)

    # --- gemmi raw CIF: R-work, R-free, ligand names ---
    block = gemmi.cif.read(str(cif_path)).sole_block()

    r_work = _cif_first_float(block, "_refine.ls_R_factor_R_work")
    r_free = _cif_first_float(block, "_refine.ls_R_factor_R_free")

    nonpoly_comp_ids = {
        gemmi.cif.as_string(v)
        for v in block.find_loop("_pdbx_entity_nonpoly.comp_id")
        if v not in (".", "?")
    } - _WATER_COMPS
    ligand_names = "|".join(sorted(nonpoly_comp_ids)) if nonpoly_comp_ids else ""

    # --- RCSB Data API: experiment condition, starting model, deposited R-factors ---
    entry = _fetch_rcsb_entry(pdb_id) if _PDB_CODE_RE.fullmatch(pdb_id) else None
    _codes, _status, starting_model = resolve_starting_model(entry, pdb_id)
    deposited_r_work, deposited_r_free = resolve_deposited_r_factors(entry)
    ph, crystal_grow_temp = resolve_crystal_grow(entry)
    if entry is not None:
        grow_blocks = entry.get("exptl_crystal_grow") or []
        grow_details = [
            " ".join(g["pdbx_details"].split()) for g in grow_blocks if g.get("pdbx_details")
        ]
        experiment_condition = "; ".join(grow_details) if grow_details else "<missing>"
    else:
        experiment_condition = "<missing>"

    return {
        "pdb_id": pdb_id,
        "space_group": space_group,
        "cell_a": cell.a,
        "cell_b": cell.b,
        "cell_c": cell.c,
        "cell_alpha": cell.alpha,
        "cell_beta": cell.beta,
        "cell_gamma": cell.gamma,
        "unit_cell_volume": cell.volume,
        "resolution": resolution,
        "r_work": r_work,
        "r_free": r_free,
        "deposited_r_work": deposited_r_work,
        "deposited_r_free": deposited_r_free,
        "num_water": num_water,
        "ligand_names": ligand_names,
        "experiment_condition": experiment_condition,
        "ph": ph,
        "crystal_grow_temp": crystal_grow_temp,
        "diffrn_temp": resolve_diffrn_temp(entry),
        "starting_model": starting_model,
    }
