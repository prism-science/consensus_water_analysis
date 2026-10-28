import biotite.structure.io.pdbx as pdbx
import numpy as np
import pytest

from cw.io import (
    find_cohort_metadata,
    load_structure_waters,
    normalize_ins_code,
    write_filtered_cif,
)

# ── normalize_ins_code ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "inp,expected",
    [
        (None, ""),
        (float("nan"), ""),  # pandas reads an empty CSV insertion_code as NaN
        ("", ""),
        ("?", ""),
        ("  ", ""),
        ("B", "B"),
        (" A ", "A"),
    ],
)
def test_normalize_ins_code(inp, expected):
    assert normalize_ins_code(inp) == expected


# ── load_structure_waters ─────────────────────────────────────────────────────


def test_load_structure_waters_no_json(cif_path_6ybf):
    # Without a JSON path all edia values are NaN; water rows are still produced.
    df = load_structure_waters(cif_path_6ybf)
    assert len(df) == 90
    assert df["edia"].isna().all()


def test_load_structure_waters_6ybf(cif_path_6ybf, edia_path_6ybf):
    # 6ybf has 90 HOH entries in both the CIF (altloc=all) and the EDIA JSON,
    # including 6 residues with altloc A/B variants (12 altloc water O total).
    df = load_structure_waters(cif_path_6ybf, edia_path_6ybf)

    # Row count matches both the CIF water-O count and the JSON HOH entry count.
    assert len(df) == 90

    expected_cols = {
        "pdb_id",
        "chain_id",
        "res_id",
        "altloc",
        "x",
        "y",
        "z",
        "b_factor",
        "occupancy",
        "edia",
    }
    assert set(df.columns) == expected_cols

    # EDIA is populated for all 90 waters — JSON entry count matches CIF water count.
    assert df["edia"].notna().sum() == 90

    # Altloc variants at res_id 345 carry distinct, correctly ordered EDIA scores.
    row_a = df[(df["res_id"] == 345) & (df["altloc"] == "A")].iloc[0]
    row_b = df[(df["res_id"] == 345) & (df["altloc"] == "B")].iloc[0]
    assert row_a["edia"] == pytest.approx(0.932854, rel=1e-4)
    assert row_b["edia"] == pytest.approx(0.691114, rel=1e-4)


# ── write_filtered_cif ────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def written_5f14(cif_path_5f14, tmp_path_factory):
    orig_atom_site = pdbx.CIFFile.read(cif_path_5f14).block["atom_site"]
    orig_n_atoms = len(orig_atom_site["id"].as_array(str))
    orig_altloc = orig_atom_site["label_alt_id"].as_array(str)

    # Simulate the script: parse once, pass the live CIFFile through get_structure
    # (which caches _row_count internally), then write with atoms removed — the
    # scenario that triggered the SerializationError.
    cif_file = pdbx.CIFFile.read(cif_path_5f14)
    atoms = pdbx.get_structure(
        cif_file,
        model=1,
        altloc="all",
        extra_fields=["b_factor", "occupancy"],
        use_author_fields=True,
    )

    water_O_idx = np.where((atoms.res_name == "HOH") & atoms.hetero & (atoms.element == "O"))[0]
    keep_mask = np.ones(len(atoms), dtype=bool)
    keep_mask[water_O_idx[-10:]] = False

    out_path = tmp_path_factory.mktemp("filter") / "5f14_filtered.cif"
    write_filtered_cif(cif_file, keep_mask, atoms[keep_mask], out_path)

    return orig_n_atoms, orig_altloc, pdbx.CIFFile.read(out_path), keep_mask


def test_write_filtered_cif_no_serialization_error(written_5f14):
    # Fixture completing without error is the assertion; atom count < original
    # confirms the removal path was exercised (not a silent no-op).
    orig_n_atoms, _, out_cif, _ = written_5f14
    assert len(out_cif.block["atom_site"]["id"].as_array(str)) < orig_n_atoms


def test_write_filtered_cif_atom_count_updated(written_5f14):
    _, _, out_cif, keep_mask = written_5f14
    assert len(out_cif.block["atom_site"]["id"].as_array(str)) == keep_mask.sum()


def test_write_filtered_cif_altloc_preserved(written_5f14):
    # label_alt_id must be carried through unchanged; set_structure would clobber it.
    _, orig_altloc, out_cif, keep_mask = written_5f14
    out_altloc = out_cif.block["atom_site"]["label_alt_id"].as_array(str)
    np.testing.assert_array_equal(out_altloc, orig_altloc[keep_mask])


# ── find_cohort_metadata ──────────────────────────────────────────────────────


def _write_metadata(root, cohort):
    directory = root / cohort
    directory.mkdir(parents=True)
    path = directory / "metadata.csv"
    path.write_text("pdb_id\n")
    return path


def test_find_cohort_metadata_prefers_the_cohorts_own_file(tmp_path):
    own = _write_metadata(tmp_path, "hewl_65_iso")
    _write_metadata(tmp_path, "hewl_65")
    assert find_cohort_metadata(tmp_path, "hewl_65_iso") == own


def test_find_cohort_metadata_walks_up_to_the_parent_cohort(tmp_path):
    parent = _write_metadata(tmp_path, "hewl_65")
    (tmp_path / "hewl_65_iso_bfactor").mkdir()
    assert find_cohort_metadata(tmp_path, "hewl_65_iso_bfactor") == parent


def test_find_cohort_metadata_returns_none_when_no_ancestor_has_one(tmp_path):
    _write_metadata(tmp_path, "unrelated_cohort")
    assert find_cohort_metadata(tmp_path, "hewl_65_iso") is None
