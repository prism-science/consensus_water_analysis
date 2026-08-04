"""Sample electron-density map values at consensus cluster positions.

Cluster centers live in the aligned reference frame, but each structure's map
(from its deposited ``<pdb>_final.mtz``) lives in that crystal's own frame. To
read a consensus position from structure X's map we must map the point back into
X's deposited frame (Approach B, inverse-transform); feeding aligned coordinates
straight in (Approach A, naive) is only valid when X was deposited in the
reference frame already (small ``rmsd_before``).

The per-structure rigid transform is not persisted by the alignment pipeline, but
the aligned CIF is just the deposited model with one rigid transform applied to
all atoms, so superposing matched Cα recovers it exactly (residual ~0) — no
re-alignment needed. Sampling itself is delegated to ``phenix.map_value_at_point``.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from cw.align import kabsch
from cw.io import load_protein

# phenix.map_value_at_point prints one "... Map value: X" line per input point,
# in input order. There is no JSON output, so we parse stdout.
_MAP_VALUE_RE = re.compile(r"Map value:\s*([-+0-9.eE]+)")


def phenix_env() -> dict[str, str]:
    """Environment for the phenix subprocess with PYTHONPATH/PYTHONHOME stripped.

    Mirrors the isolation in scripts/phenix/re-refine.sh so phenix uses its own
    bundled python rather than the project venv that launched the orchestrator.
    """
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    return env


def _ca_lookup(cif_path: Path) -> dict[tuple[str, int], np.ndarray]:
    """Map (chain_id, res_id) → Cα coordinate for a structure's protein."""
    protein, _ = load_protein(cif_path)
    ca = protein[protein.atom_name == "CA"]
    return {
        (str(c), int(r)): xyz for c, r, xyz in zip(ca.chain_id, ca.res_id, ca.coord, strict=True)
    }


def recover_transform(deposited_cif: Path, aligned_cif: Path) -> tuple[np.ndarray, np.ndarray]:
    """Recover the rigid transform mapping the deposited frame to the aligned frame.

    Returns (R, t) with ``aligned ≈ R @ deposited + t`` (row-vector convention),
    obtained by Kabsch-superposing matched Cα. Because the aligned CIF is the
    deposited model under a single rigid transform, the fit is exact.
    Raises ValueError if fewer than 3 Cα are shared.
    """
    dep = _ca_lookup(deposited_cif)
    ali = _ca_lookup(aligned_cif)
    keys = sorted(set(dep) & set(ali))
    if len(keys) < 3:
        raise ValueError(f"too few common Cα ({len(keys)}) to recover transform")
    P = np.array([dep[k] for k in keys])
    Q = np.array([ali[k] for k in keys])
    R, t = kabsch(P, Q)  # Q ≈ R @ P + t
    return R, t


def to_deposited_frame(points_aligned: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Inverse-transform aligned-frame points into the deposited frame: Rᵀ (P − t)."""
    pts = np.asarray(points_aligned, dtype=float)
    return (R.T @ (pts - t).T).T


def run_map_value_at_point(
    mtz: Path,
    points_cif: Path,
    *,
    labels: str = "FWT,PHWT",
    scale: str = "sigma",
    env: dict[str, str] | None = None,
) -> tuple[np.ndarray, subprocess.CompletedProcess]:
    """Sample a map at every point in ``points_cif`` via phenix.map_value_at_point.

    ``labels`` selects the map-coefficient arrays (required — final.mtz carries
    several complex arrays; FWT/PHWT is the 2mFo-DFc map). Returns a float array
    of one value per input point in order, plus the CompletedProcess for
    diagnostics. The returned array length is checked against the points file by
    the caller; a phenix failure yields an empty array.
    """
    cmd = [
        "phenix.map_value_at_point",
        str(mtz),
        str(points_cif),
        f"data_manager.map_coefficients.user_selected_labels={labels}",
        f"scale={scale}",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    vals = np.array([float(m) for m in _MAP_VALUE_RE.findall(proc.stdout)], dtype=float)
    return vals, proc


def select_legit(
    alignment_df: pd.DataFrame,
    *,
    rmsd_after_max: float,
    n_common_ca_min: int,
    has_mtz: set[str] | None = None,
) -> pd.Series:
    """Approach-B legitimacy mask (boolean Series indexed like alignment_df).

    A structure is legitimate when it aligns well to the reference and a map
    exists; rmsd_before is irrelevant because we inverse-transform each point.
    """
    ok = (alignment_df["rmsd_after"] <= rmsd_after_max) & (
        alignment_df["n_common_ca"] >= n_common_ca_min
    )
    if has_mtz is not None:
        ok &= alignment_df["pdb_id"].str.lower().isin(has_mtz)
    return ok


def select_eligible_naive(alignment_df: pd.DataFrame, *, rmsd_before_max: float) -> pd.Series:
    """Approach-A subset mask: deposited frame already ≈ reference frame."""
    return alignment_df["rmsd_before"] <= rmsd_before_max
