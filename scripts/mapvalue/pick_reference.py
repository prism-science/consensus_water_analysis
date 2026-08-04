"""Pick a reference dataset for Wilson-B matching before map_value sampling.

Reads each structure's PDB-REDO ``data.json`` (``properties.BWILS`` = isotropic
Wilson B, ``DATARESH`` = high-resolution limit, ``ISTWIN`` = twinning flag),
prints a table sorted by Wilson B, reports the spread, and recommends a
reference so that map coefficients can be put on a common effective B via
``scripts/mapvalue/scale_mtz.py``.

Decision rule (see plan / methodology):
  * The oversharpening risk lives on datasets that receive a *negative* delta_B
    (delta_B = B_ref - B_dataset), i.e. datasets softer/lower-res than the
    reference get sharpened. Blurring (delta_B > 0) never makes ripples.
  * Small Wilson-B spread → reference near the *median* (minimises max|delta_B|,
    so both blurring and sharpening stay mild).
  * Large spread (or --strategy maxb) → reference = *highest* Wilson B
    (lowest-res) clean dataset, so every delta_B >= 0 → blur-only, ripple-free.
  * Twinned datasets are excluded from reference candidates.

Usage:
    uv run scripts/mapvalue/pick_reference.py data/hewls_65_subsampled_similar.txt
    uv run scripts/mapvalue/pick_reference.py 6ybf 5f14 5f16 --data-root tests/fixtures
    uv run scripts/mapvalue/pick_reference.py <cohort.txt> --strategy maxb
"""

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from loguru import logger

import config
from cw.io import read_cohort


def load_props(pdb_id: str, data_root: Path, template: str) -> dict | None:
    """Return the ``properties`` block from a structure's data.json, or None."""
    path = Path(data_root) / template.format(pdb_id=pdb_id)
    if not path.exists():
        logger.warning(f"  {pdb_id}: data.json not found at {path}")
        return None
    try:
        return json.loads(path.read_text()).get("properties", {})
    except Exception as exc:
        logger.warning(f"  {pdb_id}: could not parse {path} ({exc})")
        return None


def edge_factor(delta_b: float, d_min: float) -> float:
    """Debye-Waller factor at the dataset's resolution edge: exp(-delta_B / (4 d_min²)).

    >1 means the highest-resolution terms are amplified (sharpening) — the ripple risk.
    """
    return math.exp(-delta_b / (4.0 * d_min * d_min))


def main() -> None:
    p = argparse.ArgumentParser(description="Recommend a reference dataset for Wilson-B matching.")
    p.add_argument("members", nargs="+", help="A cohort .txt file, or one or more PDB IDs")
    p.add_argument(
        "--data-root", default=config.ALL_PDB_REDO_DIR, help="Root holding <pdb>/data.json"
    )
    p.add_argument(
        "--data-template",
        default="{pdb_id}/data.json",
        help="data.json path template under --data-root",
    )
    p.add_argument(
        "--strategy",
        choices=["auto", "median", "maxb", "percentile", "fixed"],
        default="auto",
        help="auto: median if spread<=--max-spread else maxb; median: nearest-median clean; maxb: highest Wilson B (blur-only); percentile: target = --percentile of clean BWILS; fixed: target = --target-b",
    )
    p.add_argument(
        "--percentile",
        type=float,
        default=95.0,
        help="Percentile of clean Wilson B used as target (strategy=percentile)",
    )
    p.add_argument(
        "--target-b",
        type=float,
        default=None,
        help="Explicit target Wilson B in Å² (strategy=fixed)",
    )
    p.add_argument(
        "--drop-above-b",
        type=float,
        default=None,
        help="Exclude datasets with Wilson B above this (Å²) — trims the weakly-ordered tail that sets the blur ceiling",
    )
    p.add_argument(
        "--max-dmin",
        type=float,
        default=None,
        help="Exclude datasets with d_min worse (larger) than this (Å)",
    )
    p.add_argument(
        "--max-spread",
        type=float,
        default=8.0,
        help="Wilson-B spread (Å²) below which 'auto' picks the median",
    )
    p.add_argument(
        "--sharpen-warn",
        type=float,
        default=1.5,
        help="Warn when a dataset's edge sharpening factor exceeds this",
    )
    p.add_argument(
        "--name", default="cohort", help="Label for this cohort (used in the plan-CSV filename)"
    )
    p.add_argument(
        "--out", default=None, help="Plan CSV path (default: <name>_bmatch_plan.csv in CWD)"
    )
    p.add_argument(
        "--max-table-rows",
        type=int,
        default=70,
        help="Print the full per-dataset table only up to this many rows",
    )
    args = p.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{message}")

    # Resolve member list: a single existing .txt is treated as a cohort file.
    if (
        len(args.members) == 1
        and Path(args.members[0]).suffix == ".txt"
        and Path(args.members[0]).exists()
    ):
        member_ids = read_cohort(Path(args.members[0]))
    else:
        member_ids = [m.split("_")[0].lower() for m in args.members]

    rows = []
    for pdb_id in member_ids:
        props = load_props(pdb_id, args.data_root, args.data_template)
        if props is None:
            continue
        bwils = props.get("BWILS")
        d_min = props.get("DATARESH") or props.get("RESOLUTION")
        if bwils is None or d_min is None:
            logger.warning(f"  {pdb_id}: missing BWILS or resolution in data.json")
            continue
        rows.append(
            {
                "pdb_id": pdb_id,
                "d_min": float(d_min),
                "bwils": float(bwils),
                "baver": props.get("BAVER"),
                "breftype": props.get("BREFTYPE"),
                "istwin": bool(props.get("ISTWIN")),
            }
        )

    if not rows:
        sys.exit("No datasets with usable BWILS found.")

    # --- optional quality cuts (exclude before target selection AND from the plan) ---
    if args.max_dmin is not None:
        before = len(rows)
        rows = [r for r in rows if r["d_min"] <= args.max_dmin]
        logger.info(f"Dropped {before - len(rows)} dataset(s) with d_min > {args.max_dmin} Å")
    if args.drop_above_b is not None:
        before = len(rows)
        rows = [r for r in rows if r["bwils"] <= args.drop_above_b]
        logger.info(
            f"Dropped {before - len(rows)} dataset(s) with Wilson B > {args.drop_above_b} Å²"
        )
    if not rows:
        sys.exit("No datasets remain after quality cuts.")

    rows.sort(key=lambda r: r["bwils"])
    bwils_vals = [r["bwils"] for r in rows]
    b_min, b_max = bwils_vals[0], bwils_vals[-1]
    b_med = statistics.median(bwils_vals)
    spread = b_max - b_min

    def pct(vals: list[float], p: float) -> float:
        """Linear-interpolated percentile of a sorted-or-unsorted list."""
        s = sorted(vals)
        if len(s) == 1:
            return s[0]
        k = (len(s) - 1) * (p / 100.0)
        lo = int(math.floor(k))
        hi = min(lo + 1, len(s) - 1)
        return s[lo] + (s[hi] - s[lo]) * (k - lo)

    # --- distribution summary (table only for small cohorts) ---
    logger.info("")
    if len(rows) <= args.max_table_rows:
        logger.info(f"{'pdb_id':<8}{'d_min':>7}{'BWILS':>9}{'BAVER':>8}  {'Bref':>6}  twin")
        logger.info("-" * 48)
        for r in rows:
            baver = f"{r['baver']:.1f}" if r["baver"] is not None else "n/a"
            twin = "TWIN" if r["istwin"] else ""
            logger.info(
                f"{r['pdb_id']:<8}{r['d_min']:>7.2f}{r['bwils']:>9.2f}{baver:>8}  {str(r['breftype']):>6}  {twin}"
            )
        logger.info("-" * 48)
    else:
        logger.info(
            f"{len(rows)} datasets (table suppressed; full per-dataset plan written to CSV)."
        )
    logger.info(
        f"Wilson B (Å²): min={b_min:.1f}  p25={pct(bwils_vals, 25):.1f}  median={b_med:.1f}  "
        f"p75={pct(bwils_vals, 75):.1f}  p95={pct(bwils_vals, 95):.1f}  max={b_max:.1f}  spread={spread:.1f}"
    )
    d_mins = [r["d_min"] for r in rows]
    logger.info(f"Resolution (Å): best={min(d_mins):.2f}  worst={max(d_mins):.2f}")

    # --- choose target B ---
    clean = [r for r in rows if not r["istwin"]]
    if not clean:
        sys.exit("Every candidate is flagged as twinned; inspect the data manually.")
    n_twin = len(rows) - len(clean)
    if n_twin:
        logger.warning(f"Excluding {n_twin} twinned dataset(s) from reference candidates.")
    clean_b = [r["bwils"] for r in clean]

    strategy = args.strategy
    if strategy == "auto":
        strategy = "median" if spread <= args.max_spread else "maxb"
        logger.info(
            f"auto → '{strategy}' (spread {spread:.1f} {'<=' if spread <= args.max_spread else '>'} --max-spread {args.max_spread})"
        )

    if strategy == "fixed":
        if args.target_b is None:
            sys.exit("strategy=fixed requires --target-b")
        b_ref = args.target_b
    elif strategy == "percentile":
        b_ref = pct(clean_b, args.percentile)
    elif strategy == "maxb":
        b_ref = max(clean_b)
    else:  # median
        b_ref = b_med

    # Nearest clean dataset to the target B — this is the concrete "reference" structure.
    ref = min(clean, key=lambda r: abs(r["bwils"] - b_ref))

    # --- per-dataset plan ---
    plan = []
    for r in rows:
        d_b = b_ref - r["bwils"]
        plan.append(
            {
                "pdb_id": r["pdb_id"],
                "d_min": r["d_min"],
                "bwils": r["bwils"],
                "delta_b": d_b,
                "action": "blur" if d_b > 1e-6 else "sharpen" if d_b < -1e-6 else "identity",
                "edge_factor": edge_factor(d_b, r["d_min"]),
                "in_mtz": f"{args.data_root}/{r['pdb_id']}/{r['pdb_id']}_final.mtz",
                "out_mtz": f"{r['pdb_id']}_scaled.mtz",
            }
        )

    sharpened = [p for p in plan if p["action"] == "sharpen"]
    max_blur = max((p["delta_b"] for p in plan), default=0.0)
    worst_blur = min((p["edge_factor"] for p in plan if p["action"] == "blur"), default=1.0)

    # --- report ---
    logger.info("")
    logger.info(f"TARGET Wilson B = {b_ref:.1f} Å²  (strategy '{strategy}')")
    logger.info(
        f"  nearest real dataset: {ref['pdb_id']} (BWILS {ref['bwils']:.1f}, d_min {ref['d_min']:.2f} Å) — use as the reference, or pass --target-b {b_ref:.1f} to scale_mtz.py"
    )
    logger.info(
        f"  blur: {sum(1 for p in plan if p['action'] == 'blur')} datasets, up to ΔB +{max_blur:.1f} Å² "
        f"(sharpest dataset's high-res edge scaled to ×{worst_blur:.2f})"
    )
    if sharpened:
        worst_sharp = max(sharpened, key=lambda p: p["edge_factor"])
        logger.warning(
            f"  sharpen: {len(sharpened)} datasets get ΔB<0 (ripple risk). Worst: {worst_sharp['pdb_id']} "
            f"ΔB {worst_sharp['delta_b']:+.1f} Å² → edge ×{worst_sharp['edge_factor']:.2f}"
        )
        flagged = [p["pdb_id"] for p in sharpened if p["edge_factor"] > args.sharpen_warn]
        if flagged:
            logger.warning(
                f"  QC these sharpened maps for ripples (edge ×>{args.sharpen_warn}): {', '.join(flagged)}"
            )
    else:
        logger.info("  sharpen: none — every ΔB ≥ 0, ripple-free by construction (blur-only).")

    # --- write plan CSV ---
    out_csv = Path(args.out) if args.out else Path(f"{args.name}_bmatch_plan.csv")
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "pdb_id",
                "d_min",
                "bwils",
                "delta_b",
                "action",
                "edge_factor",
                "in_mtz",
                "out_mtz",
            ],
        )
        w.writeheader()
        for p in plan:
            w.writerow(
                {**p, "delta_b": round(p["delta_b"], 3), "edge_factor": round(p["edge_factor"], 4)}
            )
    logger.info("")
    logger.info(f"Wrote per-dataset scaling plan → {out_csv}  ({len(plan)} datasets)")
    logger.info("Scale every dataset (run under phenix.python) with:")
    logger.info(
        f"  tail -n +2 {out_csv} | awk -F, '{{print $7, $8, $4}}' | "
        f"while read IN OUT DB; do phenix.python scripts/mapvalue/scale_mtz.py $IN $OUT --delta-b $DB; done"
    )


if __name__ == "__main__":
    main()
