import biotite.structure.io.pdbx as pdbx
import numpy as np
import pytest

from cw.align import align_to_reference, kabsch
from cw.io import load_protein

# ── kabsch (pure math, no fixtures) ───────────────────────────────────────────


def test_kabsch_identity():
    rng = np.random.default_rng(0)
    pts = rng.standard_normal((10, 3))
    R, t = kabsch(pts, pts)
    np.testing.assert_allclose((R @ pts.T).T + t, pts, atol=1e-10)


def test_kabsch_pure_translation():
    rng = np.random.default_rng(1)
    fixed = rng.standard_normal((10, 3))
    mobile = fixed + np.array([3.0, -1.5, 2.0])
    R, t = kabsch(mobile, fixed)
    np.testing.assert_allclose((R @ mobile.T).T + t, fixed, atol=1e-10)


def test_kabsch_rotation_and_translation():
    rng = np.random.default_rng(2)
    fixed = rng.standard_normal((20, 3))
    theta = np.pi / 3
    R_true = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(theta), -np.sin(theta)],
            [0.0, np.sin(theta), np.cos(theta)],
        ]
    )
    mobile = (R_true @ fixed.T).T + np.array([1.0, -2.0, 0.5])
    R, t = kabsch(mobile, fixed)
    np.testing.assert_allclose((R @ mobile.T).T + t, fixed, atol=1e-10)


def test_kabsch_no_reflection():
    rng = np.random.default_rng(3)
    for _ in range(20):
        R, _ = kabsch(rng.standard_normal((12, 3)), rng.standard_normal((12, 3)))
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-10)


# ── align_to_reference integration (5f14 → 5f16) ─────────────────────────────


def _atom_counts(cif_path):
    atoms = pdbx.get_structure(
        pdbx.CIFFile.read(str(cif_path)),
        model=1,
        altloc="all",
        use_author_fields=True,
    )
    n_protein = int((~atoms.hetero).sum())
    n_water = int(((atoms.res_name == "HOH") & atoms.hetero & (atoms.element == "O")).sum())
    return n_protein, n_water


def test_align_5f16_to_6ybf(cif_path_6ybf, cif_path_5f16, tmp_path):
    ref_protein, _ = load_protein(cif_path_6ybf)
    out_path = tmp_path / "5f16_aligned.cif"

    report = align_to_reference(cif_path_5f16, ref_protein, out_path=out_path)

    assert report is not None, "alignment was skipped — too few common Cα"
    assert report["rmsd_after"] < report["rmsd_before"], (
        f"RMSD did not improve: {report['rmsd_before']} → {report['rmsd_after']}"
    )

    prot_before, water_before = _atom_counts(cif_path_5f16)
    prot_after, water_after = _atom_counts(out_path)

    assert prot_after == prot_before, f"protein atom count changed: {prot_before} → {prot_after}"
    assert water_after == water_before, f"water count changed: {water_before} → {water_after}"
