"""Sample each structure's density map at the consensus cluster positions.

For every cluster center (and optionally the HDBSCAN noise waters), read the map
value from each cohort structure's deposited ``<pdb>_final.mtz`` using
``phenix.map_value_at_point``, then aggregate a median per cluster for plotting
against ``cluster_occupancy``.

Two approaches are computed:
  B (inverse-transform, main): each center is mapped back into a structure's own
    deposited frame before sampling — valid for all well-aligned structures.
  A (naive, comparison): aligned coordinates are sampled directly, valid only for
    structures already deposited in the reference frame (small ``rmsd_before``).

Usage:
    uv run scripts/mapvalue/map_value_at_clusters.py <cohort.txt> [options]

Reads clusters.csv / cluster_members.csv and aligned_pdbs/ from
data/<cohort_id>/ by default; writes map_values_raw.csv.gz and
cluster_map_summary.csv there.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

import numpy as np
import pandas as pd
from loguru import logger

import config
from cw.io import cif_path_for, read_cohort, write_water_points_cif
from cw.mapvalue import (
    phenix_env,
    recover_transform,
    run_map_value_at_point,
    select_eligible_naive,
    select_legit,
    to_deposited_frame,
)


def mtz_path_for(pdb_id: str) -> Path:
    return Path(config.ALL_PDB_REDO_DIR) / f"{pdb_id}/{pdb_id}_final.mtz"


def build_points(
    clusters_df: pd.DataFrame,
    members_df: pd.DataFrame | None,
):
    """Return (center_xyz, center_ids, noise_xyz, noise_ids, noise_pdb).

    center_ids are cluster_id values (clusters.csv order). noise_* describe the
    cluster_id == -1 waters from cluster_members.csv (empty arrays when members_df
    is None); noise_pdb gives each noise water's source pdb_id for own-map mode.
    """
    center_xyz = clusters_df[["center_x", "center_y", "center_z"]].to_numpy(dtype=float)
    center_ids = clusters_df["cluster_id"].to_numpy(dtype=int)

    if members_df is None:
        empty = np.empty((0, 3), dtype=float)
        return center_xyz, center_ids, empty, np.empty(0, dtype=int), np.empty(0, dtype=object)

    noise = members_df[members_df["cluster_id"] == -1].reset_index(drop=True)
    noise_xyz = noise[["x", "y", "z"]].to_numpy(dtype=float)
    noise_ids = np.arange(len(noise), dtype=int)
    noise_pdb = noise["pdb_id"].str.lower().to_numpy(dtype=object)
    return center_xyz, center_ids, noise_xyz, noise_ids, noise_pdb


def sample_one(
    pdb_id: str,
    *,
    approach: str,
    aligned_dir: Path,
    tmp_dir: Path,
    center_xyz: np.ndarray,
    center_ids: np.ndarray,
    noise_xyz: np.ndarray,
    noise_ids: np.ndarray,
    labels: str,
    scale: str,
    env: dict,
) -> pd.DataFrame | None:
    """Sample one structure's map at the supplied points; return long-format rows.

    approach 'B' inverse-transforms the points into the deposited frame; 'A'
    samples the aligned coordinates directly (naive). noise_xyz/noise_ids are the
    noise points to include for THIS structure (already selected by the caller).
    """
    types = np.concatenate([np.full(len(center_ids), "cluster"), np.full(len(noise_ids), "noise")])
    ids = np.concatenate([center_ids, noise_ids])
    pts_aligned = np.vstack([center_xyz, noise_xyz]) if len(noise_xyz) else center_xyz

    if approach == "B":
        try:
            R, t = recover_transform(
                cif_path_for(pdb_id, config.ALL_PDB_REDO_DIR, config.CIF_TEMPLATE),
                aligned_dir / f"{pdb_id}.cif",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"{pdb_id}: transform recovery failed ({exc}) — skipping")
            return None
        pts = to_deposited_frame(pts_aligned, R, t)
    else:  # naive: sample aligned coords directly
        pts = pts_aligned

    cif = tmp_dir / f"{pdb_id}_{approach}.cif"
    write_water_points_cif(cif, pts, data_block="points")
    vals, proc = run_map_value_at_point(
        mtz_path_for(pdb_id), cif, labels=labels, scale=scale, env=env
    )
    cif.unlink(missing_ok=True)

    if len(vals) != len(ids):
        logger.warning(
            f"{pdb_id} [{approach}]: got {len(vals)} values for {len(ids)} points "
            f"— recording NaN. stderr tail: {proc.stderr.strip()[-200:]}"
        )
        vals = np.full(len(ids), np.nan)

    return pd.DataFrame(
        {
            "pdb_id": pdb_id,
            "approach": approach,
            "point_type": types,
            "point_id": ids,
            "map_value": vals,
        }
    )


def run_pass(
    pdb_ids: list[str],
    approach: str,
    *,
    n_jobs: int,
    **kw,
) -> list[pd.DataFrame]:
    """Sample a list of structures concurrently; returns the non-empty chunks."""
    out: list[pd.DataFrame] = []
    done = 0
    with ThreadPoolExecutor(max_workers=n_jobs) as ex:
        futures = {ex.submit(sample_one, p, approach=approach, **kw): p for p in pdb_ids}
        for fut in as_completed(futures):
            chunk = fut.result()
            if chunk is not None:
                out.append(chunk)
            done += 1
            if done % 50 == 0 or done == len(pdb_ids):
                logger.info(f"  [{approach}] {done}/{len(pdb_ids)} structures sampled")
    return out


def summarize(raw: pd.DataFrame, clusters_df: pd.DataFrame, n_total: int) -> pd.DataFrame:
    """Per-cluster median/mean/std/n of map value for each approach, joined to occupancy."""
    clu = raw[raw["point_type"] == "cluster"]
    agg = (
        clu.groupby(["point_id", "approach"])["map_value"]
        .agg(median="median", mean="mean", std="std", n="count")
        .unstack("approach")
    )
    agg.columns = [f"{stat}_map_{appr}" for stat, appr in agg.columns]
    agg = agg.reset_index().rename(columns={"point_id": "cluster_id"})
    summary = clusters_df[["cluster_id", "cluster_occupancy"]].merge(
        agg, on="cluster_id", how="left"
    )
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Sample map values at consensus cluster positions.")
    p.add_argument("cohort", type=Path, help="Cohort .txt file (one member ID per line)")
    p.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Directory of aligned CIFs (default: DATA_DIR/<cohort_id>/aligned_pdbs/)",
    )
    p.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Output dir for results (default: DATA_DIR/<cohort_id>/)",
    )
    p.add_argument(
        "--clusters-dir",
        type=Path,
        default=None,
        help="Dir holding clusters.csv / cluster_members.csv (default: DATA_DIR/<cohort_id>/)",
    )
    p.add_argument(
        "--ref-pdb-id",
        default=config.REF_PDB_ID,
        help=f"Reference PDB for alignment_report_<ref>.csv (default: {config.REF_PDB_ID})",
    )
    p.add_argument(
        "--include-noise",
        action="store_true",
        help="Also sample at HDBSCAN noise waters (cluster_id == -1)",
    )
    p.add_argument(
        "--noise-mode",
        choices=["all-cohort", "own-map"],
        default="all-cohort",
        help="all-cohort: sample every structure at each noise point (option 2); "
        "own-map: sample each noise water only in its own structure (option 1)",
    )
    p.add_argument("--map-labels", default="FWT,PHWT", help="MTZ map-coefficient labels (2mFo-DFc)")
    p.add_argument("--scale", default="sigma", choices=["sigma", "volume"], help="Map scaling")
    p.add_argument(
        "--rmsd-after-max", type=float, default=1.0, help="Approach-B alignment gate (Å)"
    )
    p.add_argument("--n-common-ca-min", type=int, default=200, help="Approach-B min paired Cα")
    p.add_argument(
        "--rmsd-before-max", type=float, default=1.0, help="Approach-A naive-eligibility gate (Å)"
    )
    p.add_argument(
        "--no-naive", action="store_true", help="Skip the Approach-A (naive) comparison pass"
    )
    p.add_argument("--keep-points", action="store_true", help="Keep temporary point CIFs (debug)")
    p.add_argument(
        "-j", "--n-jobs", type=int, default=4, help="Concurrent phenix jobs (default: 4)"
    )
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--verbose", action="store_true")
    grp.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="WARNING" if args.quiet else "DEBUG" if args.verbose else "INFO")

    if not args.cohort.exists():
        logger.error(f"Cohort file not found: {args.cohort}")
        sys.exit(1)

    cohort_id = args.cohort.stem
    data_dir = Path(config.DATA_DIR) / cohort_id
    aligned_dir = args.input_dir or data_dir / "aligned_pdbs"
    out_dir = args.output_dir or data_dir
    clusters_dir = args.clusters_dir or data_dir
    clusters_csv = clusters_dir / "clusters.csv"
    members_csv = clusters_dir / "cluster_members.csv"
    align_csv = aligned_dir / f"alignment_report_{args.ref_pdb_id}.csv"

    for required in (clusters_csv, align_csv):
        if not required.exists():
            logger.error(f"Required input not found: {required}")
            sys.exit(1)

    member_ids = set(read_cohort(args.cohort))
    clusters_df = pd.read_csv(clusters_csv)
    members_df = pd.read_csv(members_csv) if (args.include_noise and members_csv.exists()) else None
    if args.include_noise and members_df is None:
        logger.error(f"--include-noise requires {members_csv}")
        sys.exit(1)
    align_df = pd.read_csv(align_csv)
    align_df = align_df[align_df["pdb_id"].str.lower().isin(member_ids)].reset_index(drop=True)

    has_mtz = {
        r["pdb_id"].lower()
        for _, r in align_df.iterrows()
        if mtz_path_for(r["pdb_id"].lower()).exists()
        and (aligned_dir / f"{r['pdb_id'].lower()}.cif").exists()
    }
    legit_mask = select_legit(
        align_df,
        rmsd_after_max=args.rmsd_after_max,
        n_common_ca_min=args.n_common_ca_min,
        has_mtz=has_mtz,
    )
    eligible_mask = (
        select_eligible_naive(align_df, rmsd_before_max=args.rmsd_before_max) & legit_mask
    )
    legit_ids = align_df.loc[legit_mask, "pdb_id"].str.lower().tolist()
    eligible_ids = align_df.loc[eligible_mask, "pdb_id"].str.lower().tolist()

    center_xyz, center_ids, noise_xyz, noise_ids, noise_pdb = build_points(clusters_df, members_df)

    logger.info(f"Cohort:        {cohort_id}  (ref {args.ref_pdb_id})")
    logger.info(
        f"Clusters:      {len(center_ids)} centers; occupancy "
        f"{clusters_df['cluster_occupancy'].min():.3f}–{clusters_df['cluster_occupancy'].max():.3f}"
    )
    if args.include_noise:
        logger.info(f"Noise waters:  {len(noise_ids)} ({args.noise_mode})")
    logger.info(
        f"Legit (B):     {len(legit_ids)}/{len(align_df)} "
        f"(rmsd_after≤{args.rmsd_after_max}, n_ca≥{args.n_common_ca_min}, mtz+cif present)"
    )
    logger.info(
        f"Eligible (A):  {len(eligible_ids)} (rmsd_before≤{args.rmsd_before_max})"
        + ("  [skipped]" if args.no_naive else "")
    )
    logger.info(f"Map:           {args.map_labels}  scale={args.scale}   jobs={args.n_jobs}")

    if not legit_ids:
        logger.error("No legitimate structures to sample.")
        sys.exit(1)

    tmp_dir = out_dir / "_points_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    env = phenix_env()
    common = dict(
        aligned_dir=aligned_dir,
        tmp_dir=tmp_dir,
        center_xyz=center_xyz,
        center_ids=center_ids,
        labels=args.map_labels,
        scale=args.scale,
        env=env,
    )

    chunks: list[pd.DataFrame] = []

    # Approach B: inverse-transform, all legit structures (centers + noise).
    if args.noise_mode == "own-map" and args.include_noise:
        # noise sampled only in its own structure; run B per structure with its own noise subset
        logger.info("Sampling map values (B, inverse-transform, own-map noise)...")

        def b_args(pid):
            sel = noise_pdb == pid
            return dict(noise_xyz=noise_xyz[sel], noise_ids=noise_ids[sel])

        out = []
        with ThreadPoolExecutor(max_workers=args.n_jobs) as ex:
            futs = {
                ex.submit(sample_one, pid, approach="B", **common, **b_args(pid)): pid
                for pid in legit_ids
            }
            for done, fut in enumerate(as_completed(futs), start=1):
                c = fut.result()
                if c is not None:
                    out.append(c)
                if done % 50 == 0 or done == len(legit_ids):
                    logger.info(f"  [B] {done}/{len(legit_ids)} structures sampled")
        chunks += out
    else:
        logger.info("Sampling map values (B, inverse-transform)...")
        chunks += run_pass(
            legit_ids, "B", n_jobs=args.n_jobs, noise_xyz=noise_xyz, noise_ids=noise_ids, **common
        )

    # Approach A: naive (aligned coords), eligible subset, centers only.
    if not args.no_naive and eligible_ids:
        logger.info("Sampling map values (A, naive aligned coords)...")
        chunks += run_pass(
            eligible_ids,
            "A",
            n_jobs=args.n_jobs,
            noise_xyz=np.empty((0, 3)),
            noise_ids=np.empty(0, dtype=int),
            **common,
        )

    if not args.keep_points:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    raw = pd.concat(chunks, ignore_index=True)
    raw = raw.merge(
        align_df[["pdb_id", "rmsd_before", "rmsd_after", "n_common_ca"]].assign(
            pdb_id=lambda d: d["pdb_id"].str.lower()
        ),
        on="pdb_id",
        how="left",
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "map_values_raw.csv.gz"
    summary_path = out_dir / "cluster_map_summary.csv"
    raw.to_csv(raw_path, index=False)
    summary = summarize(raw, clusters_df, n_total=len(legit_ids))
    summary.to_csv(summary_path, index=False)

    logger.info(f"map_values_raw.csv.gz:    {len(raw):,} rows  →  {raw_path}")
    logger.info(f"cluster_map_summary.csv:  {len(summary):,} clusters  →  {summary_path}")

    # Noise points: median over structures (approach B), plotted at occupancy 1/N.
    noise_rows = raw[(raw["point_type"] == "noise") & (raw["approach"] == "B")]
    if not noise_rows.empty:
        noise_summary = (
            noise_rows.groupby("point_id")["map_value"]
            .agg(median_map_B="median", n_B="count")
            .reset_index()
            .rename(columns={"point_id": "noise_id"})
        )
        noise_summary["cluster_occupancy"] = 1.0 / max(len(legit_ids), 1)
        noise_path = out_dir / "noise_map_summary.csv"
        noise_summary.to_csv(noise_path, index=False)
        logger.info(
            f"noise_map_summary.csv:    {len(noise_summary):,} noise points  →  {noise_path}"
        )

    if "median_map_B" in summary:
        logger.info(
            f"  median map value (B): {summary['median_map_B'].median():.2f} σ (cohort median of cluster medians)"
        )


if __name__ == "__main__":
    main()
