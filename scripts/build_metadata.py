"""Build metadata CSV for every structure in a cohort .txt.

Usage:
    uv run scripts/build_metadata.py <cohort.txt> [-o metadata.csv]

Fetches experiment_condition, starting_model, the deposited R-factors, and the
crystallization/data-collection conditions (ph, crystal_grow_temp, diffrn_temp)
from the RCSB Data API; all other fields come from local mmCIF files.
Writes to data/<cohort_id>/metadata.csv by default.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from loguru import logger

import config
from cw.io import cif_path_for, read_cohort
from cw.metadata import metadata_row


def main() -> None:
    parser = argparse.ArgumentParser(description="Build metadata CSV for a cohort.")
    parser.add_argument("cohort", type=Path, help="Cohort .txt file (one member ID per line)")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: config.DATA_DIR/<cohort_id>/metadata.csv)",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Directory of input CIFs, resolved with config.CIF_TEMPLATE "
        "(default: config.ALL_PDB_REDO_DIR)",
    )
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("--verbose", action="store_true", help="Show debug output")
    verbosity.add_argument("--quiet", action="store_true", help="Show warnings and errors only")
    args = parser.parse_args()

    logger.remove()
    if args.quiet:
        logger.add(sys.stderr, level="WARNING")
    elif args.verbose:
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.add(sys.stderr, level="INFO")

    cohort_path: Path = args.cohort
    if not cohort_path.exists():
        logger.error(f"Cohort file not found: {cohort_path}")
        sys.exit(1)

    cohort_id = cohort_path.stem
    out_path: Path = args.output or Path(config.DATA_DIR) / cohort_id / "metadata.csv"

    source_dir = args.source_dir or Path(config.ALL_PDB_REDO_DIR)
    member_ids = read_cohort(cohort_path)
    cif_paths = {m: cif_path_for(m, source_dir, config.CIF_TEMPLATE) for m in member_ids}
    missing = [m for m, p in cif_paths.items() if not p.exists()]
    found = [m for m, p in cif_paths.items() if p.exists()]

    logger.info(f"Cohort:  {cohort_id}")
    logger.info(f"Members: {len(member_ids)} total — {len(found)} found, {len(missing)} missing")
    if missing:
        logger.warning(f"Missing CIFs: {', '.join(missing)}")
    logger.info(f"Source:  {source_dir}")
    logger.info(f"Output:  {out_path}")

    rows = []
    n_err = 0
    for member_id in found:
        cif_path = cif_paths[member_id]
        try:
            row = metadata_row(cif_path, member_id)
            rows.append(row)
            logger.debug(f"  {member_id}: ok")
        except Exception as exc:
            n_err += 1
            logger.error(f"  {member_id}: {exc}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.sort_values(by="resolution", inplace=True)
    df.to_csv(out_path, index=False)
    logger.info(f"Wrote {len(rows)} rows to {out_path}")
    if n_err:
        logger.warning(f"{n_err} errors (see above)")


if __name__ == "__main__":
    main()
