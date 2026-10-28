"""
Fill in GROUND_TRUTH from notebooks/00_metadata_filter_align_demo.py output for TEST_PDB_ID.
Once set, these become regression guards for the CIF parsing logic.
"""

import biotite.structure.io.pdbx as pdbx
import gemmi
import pytest

from cw.metadata import metadata_row

# fmt: off
GROUND_TRUTH_METADATA_5F16 = {
    "space_group": "P 43 21 2",
    "starting_model": "5f14",
    "experiment_condition": "2M sodium chloride, 12% (w/v) PEG 6000",
    "ligand_names": "CL|NA"
}
# fmt: on


@pytest.fixture(scope="module")
def row(cif_path_5f16):
    return metadata_row(cif_path_5f16)


def test_num_water_biotite_matches_gemmi(cif_path_5f16):
    st = gemmi.read_structure(str(cif_path_5f16))
    gemmi_count = sum(
        1
        for chain in st[0]
        for res in chain
        if res.is_water()
        for atom in res
        if atom.element == gemmi.Element("O")
    )
    atoms = pdbx.get_structure(
        pdbx.CIFFile.read(cif_path_5f16),
        model=1,
        altloc="all",
        use_author_fields=True,
    )
    biotite_count = int(((atoms.res_name == "HOH") & atoms.hetero & (atoms.element == "O")).sum())
    assert biotite_count == gemmi_count


def test_edia_count_matches_num_water(cif_path_5f16, edia_path_5f16, row):
    from cw.io import load_edia

    edia = load_edia(edia_path_5f16)
    assert edia is not None
    assert len(edia) == row["num_water"]


@pytest.mark.parametrize(
    "field", ["space_group", "starting_model", "experiment_condition", "ligand_names"]
)
def test_fields(row, field):
    assert row[field] == GROUND_TRUTH_METADATA_5F16[field]
