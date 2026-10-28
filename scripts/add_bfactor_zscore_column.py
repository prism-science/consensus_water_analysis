"""Append a per-structure B-factor z-score column to a cluster_members.csv.

For each pdb_id in the table, the z-score standardizes each water's B-factor
against a reference population of B-factors read from the original CIF —
"water" (water O atoms), "protein" (protein heavy atoms) or "all" (every atom),
the same reference `cw.filter.keep_by_bfactor(mode="zscore")` uses. This means
`b_factor_zscore <= X` selects exactly the waters the B-factor z-score filter at
cutoff X would keep, for the same population. A degenerate structure (population
std 0) gets z = 0.0, matching keep_by_bfactor's "keep every water" behavior.

The population defaults to config.BFACTOR_POPULATION (itself "water" by default)
so the column agrees with Stage 2 without being told twice.

Usage:
    uv run scripts/add_bfactor_zscore_column.py <cluster_members.csv> [-o OUT]
                                                [--population water|protein|all]
"""

import argparse
import sys
from pathlib import Path

import biotite.structure.io.pdbx as pdbx
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from cw.filter import bfactor_reference
from cw.io import cif_path_for, water_oxygen_mask


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("members_csv", type=Path, help="cluster_members.csv to augment")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output CSV (default: overwrite the input in place)",
    )
    parser.add_argument(
        "--population",
        choices=["water", "protein", "all"],
        default=None,
        help="Z-score reference population (default: config.BFACTOR_POPULATION)",
    )
    args = parser.parse_args()

    population = args.population or getattr(config, "BFACTOR_POPULATION", "water")

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    df = pd.read_csv(args.members_csv)

    zscore = pd.Series(float("nan"), index=df.index)
    for pdb_id in df["pdb_id"].unique():
        cif_path = cif_path_for(pdb_id, config.ALL_PDB_REDO_DIR, config.CIF_TEMPLATE)
        if not cif_path.exists():
            logger.warning(f"{pdb_id}: CIF not found ({cif_path}) — leaving z-score NaN")
            continue
        atoms = pdbx.get_structure(
            pdbx.CIFFile.read(cif_path),
            model=1,
            altloc="all",
            extra_fields=["b_factor", "occupancy"],
            use_author_fields=True,
        )
        ref = bfactor_reference(atoms, water_oxygen_mask(atoms), population)
        std = ref.std()
        rows = df["pdb_id"] == pdb_id
        if std == 0:
            zscore.loc[rows] = 0.0
        else:
            zscore.loc[rows] = (df.loc[rows, "b_factor"] - ref.mean()) / std

    df["b_factor_zscore"] = zscore

    out_path = args.output or args.members_csv
    df.to_csv(out_path, index=False)
    n_missing = int(zscore.isna().sum())
    logger.info(
        f"Wrote {out_path} ({len(df)} rows, population={population})"
        + (f"; {n_missing} rows left NaN (missing CIF)" if n_missing else "")
    )


if __name__ == "__main__":
    main()
