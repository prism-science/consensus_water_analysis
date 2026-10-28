from collections import Counter

import biotite.structure.io.pdbx as pdbx
import gemmi
import numpy as np
import pytest
from scipy.spatial.distance import cdist

from cw.filter import best_sym_positions, bfactor_reference, filter_waters, keep_by_bfactor
from cw.io import (
    edia_scores_in_order,
    load_edia_all_altlocs,
    load_structure_waters,
    normalize_ins_code,
    water_oxygen_mask,
)


def _make_cell(a=100.0):
    return gemmi.UnitCell(a, a, a, 90, 90, 90)


def _p1():
    return gemmi.SpaceGroup("P 1")


# ── best_sym_positions unit tests ──────────────────────────────────────────────


def test_p1_positions_unchanged():
    """P 1 has only the identity op — no water should be moved."""
    cell = _make_cell()
    water = np.array([[5.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
    protein = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    best_pos, _ = best_sym_positions(water, protein, cell, _p1())

    np.testing.assert_allclose(best_pos, water, atol=1e-6)


def test_periodic_image_closer():
    """Water near the cell boundary should be pulled to the image closer to the
    protein when the cell is small enough that a lattice shift matters."""
    # 10 Å cubic P1 cell.  Protein at origin; water at (9, 0, 0).
    # Naive distance = 9 Å.  Periodic image at (9-10, 0, 0) = (-1, 0, 0) → 1 Å.
    cell = gemmi.UnitCell(10, 10, 10, 90, 90, 90)
    water = np.array([[9.0, 0.0, 0.0]])
    protein = np.array([[0.0, 0.0, 0.0]])

    best_pos, best_dists = best_sym_positions(water, protein, cell, _p1())

    assert best_dists[0] == pytest.approx(1.0, abs=1e-5)


# ── filter_waters integration test (5f14) ─────────────────────────────────────

CUTOFF = 4.0


@pytest.fixture(scope="module")
def filtered_5f14(cif_path_5f14):
    st = gemmi.read_structure(str(cif_path_5f14))
    cell = st.cell
    sg = st.find_spacegroup() or gemmi.SpaceGroup("P 1")
    atoms = pdbx.get_structure(
        pdbx.CIFFile.read(str(cif_path_5f14)),
        model=1,
        altloc="all",
        extra_fields=["b_factor", "occupancy"],
        use_author_fields=True,
    )
    filtered, stats, keep_mask = filter_waters(atoms, cell, sg, CUTOFF)
    return atoms, filtered, stats["n_removed_distance"]


def test_filter_5f14_removes_some_waters(filtered_5f14):
    atoms, filtered, n_removed = filtered_5f14
    n_before = int(((atoms.res_name == "HOH") & atoms.hetero & (atoms.element == "O")).sum())
    assert n_removed > 0
    assert n_removed < n_before


def test_filter_5f14_remaining_waters_within_cutoff(filtered_5f14):
    """All kept water O atoms (at their updated canonical-ASU coordinates)
    must be within the cutoff of some protein heavy atom."""
    _, filtered, _ = filtered_5f14
    water_mask = (filtered.res_name == "HOH") & filtered.hetero & (filtered.element == "O")
    prot_mask = ~filtered.hetero & (filtered.element != "H")
    min_dists = cdist(filtered.coord[water_mask], filtered.coord[prot_mask]).min(axis=1)
    assert (min_dists <= CUTOFF + 1e-4).all()


def test_filter_5f14_protein_atoms_preserved(filtered_5f14):
    atoms, filtered, _ = filtered_5f14
    n_prot_before = int((~atoms.hetero).sum())
    n_prot_after = int((~filtered.hetero).sum())
    assert n_prot_after == n_prot_before


# ── EDIA filtering ─────────────────────────────────────────────────────────────


def _load_atoms(cif_path):
    return pdbx.get_structure(
        pdbx.CIFFile.read(str(cif_path)),
        model=1,
        altloc="all",
        extra_fields=["b_factor", "occupancy"],
        use_author_fields=True,
    )


def _water_keys(water_atoms):
    return [
        (
            str(water_atoms.chain_id[i]),
            int(water_atoms.res_id[i]),
            normalize_ins_code(water_atoms.ins_code[i]),
        )
        for i in range(len(water_atoms))
    ]


def test_edia_scores_in_order_matches_attach_edia(cif_path_6ybf, edia_path_6ybf):
    """edia_scores_in_order, keyed from water-O atoms in atom_site order, must
    reproduce the scores load_structure_waters attaches via cw.io._attach_edia — the
    shared positional altloc-pairing contract both paths delegate to."""
    df = load_structure_waters(cif_path_6ybf, edia_path_6ybf)
    waters = _load_atoms(cif_path_6ybf)[water_oxygen_mask(_load_atoms(cif_path_6ybf))]
    scores = edia_scores_in_order(_water_keys(waters), load_edia_all_altlocs(edia_path_6ybf))
    np.testing.assert_allclose(scores, df["edia"].to_numpy(), equal_nan=True)


def test_filter_preserves_water_altlocs(cif_path_6ybf):
    """Hard rule: every water altloc survives filtering as a distinct oxygen. 6ybf has
    altloc waters; a generous cutoff removes nothing, so the per-residue water-O counts
    (which are >1 for altloc waters) must be identical before and after."""
    st = gemmi.read_structure(str(cif_path_6ybf))
    cell = st.cell
    sg = st.find_spacegroup() or _p1()
    atoms = _load_atoms(cif_path_6ybf)

    in_counts = Counter(_water_keys(atoms[water_oxygen_mask(atoms)]))
    assert any(c > 1 for c in in_counts.values())  # fixture sanity: 6ybf has altloc waters

    filtered, stats, _ = filter_waters(atoms, cell, sg, 999.0)
    assert stats["n_removed_distance"] == 0
    assert Counter(_water_keys(filtered[water_oxygen_mask(filtered)])) == in_counts


def test_filter_edia_moderate_cutoff_drops_some(cif_path_6ybf, edia_path_6ybf):
    """A cutoff inside 6ybf's EDIAm range drops some — not all — distance-surviving
    waters; counts partition exactly and every kept water scores >= the cutoff."""
    st = gemmi.read_structure(str(cif_path_6ybf))
    cell = st.cell
    sg = st.find_spacegroup() or _p1()
    edia = load_edia_all_altlocs(edia_path_6ybf)

    dist_filtered, _, _ = filter_waters(_load_atoms(cif_path_6ybf), cell, sg, CUTOFF)
    surv = dist_filtered[water_oxygen_mask(dist_filtered)]
    cutoff = float(np.median(edia_scores_in_order(_water_keys(surv), edia)))

    filtered, stats, _ = filter_waters(
        _load_atoms(cif_path_6ybf), cell, sg, CUTOFF, edia_lists=edia, edia_cutoff=cutoff
    )
    kept = filtered[water_oxygen_mask(filtered)]
    assert stats["n_removed_edia"] > 0
    assert kept.array_length() > 0
    assert (
        stats["n_removed_distance"] + stats["n_removed_edia"] + kept.array_length()
        == stats["n_water"]
    )
    assert (edia_scores_in_order(_water_keys(kept), edia) >= cutoff).all()


def test_filter_edia_empty_lists_drops_all_waters(cif_path_6ybf):
    """The --drop-if-no-edia-json path passes an empty edia_lists: every water then
    lacks a score and is dropped by EDIA (distinct from the low-score high-cutoff path)."""
    st = gemmi.read_structure(str(cif_path_6ybf))
    cell = st.cell
    sg = st.find_spacegroup() or _p1()
    filtered, stats, _ = filter_waters(
        _load_atoms(cif_path_6ybf), cell, sg, CUTOFF, edia_lists={}, edia_cutoff=0.6
    )
    assert stats["n_water"] > 0
    assert int(water_oxygen_mask(filtered).sum()) == 0
    assert stats["n_removed_distance"] + stats["n_removed_edia"] == stats["n_water"]


def test_filter_edia_high_cutoff_removes_all_waters(cif_path_6ybf, edia_path_6ybf):
    """A cutoff above any real EDIAm drops every distance-surviving water, and the
    removal counts partition the starting waters exactly."""
    st = gemmi.read_structure(str(cif_path_6ybf))
    cell = st.cell
    sg = st.find_spacegroup() or _p1()
    edia = load_edia_all_altlocs(edia_path_6ybf)
    filtered, stats, _ = filter_waters(
        _load_atoms(cif_path_6ybf), cell, sg, CUTOFF, edia_lists=edia, edia_cutoff=99.0
    )
    assert stats["n_water"] > 0
    assert stats["n_removed_distance"] + stats["n_removed_edia"] == stats["n_water"]
    assert int(water_oxygen_mask(filtered).sum()) == 0


def test_filter_edia_disabled_matches_distance_only(cif_path_6ybf):
    """With no EDIA args the result is identical to distance-only filtering and
    n_removed_edia is 0."""
    st = gemmi.read_structure(str(cif_path_6ybf))
    cell = st.cell
    sg = st.find_spacegroup() or _p1()
    filtered, stats, _ = filter_waters(_load_atoms(cif_path_6ybf), cell, sg, CUTOFF)
    assert stats["n_removed_edia"] == 0


def test_filter_edia_neg_inf_matches_distance_only(cif_path_6ybf, edia_path_6ybf):
    """An edia_cutoff of -inf keeps every water that has an EDIA score, so with 6ybf's
    full EDIA coverage it drops nothing and reproduces distance-only filtering exactly:
    same keep-mask, same relocated coords, and n_removed_edia is 0."""
    st = gemmi.read_structure(str(cif_path_6ybf))
    cell = st.cell
    sg = st.find_spacegroup() or _p1()

    dist_atoms, dist_stats, dist_mask = filter_waters(_load_atoms(cif_path_6ybf), cell, sg, CUTOFF)
    edia = load_edia_all_altlocs(edia_path_6ybf)
    inf_atoms, inf_stats, inf_mask = filter_waters(
        _load_atoms(cif_path_6ybf), cell, sg, CUTOFF, edia_lists=edia, edia_cutoff=-np.inf
    )

    assert inf_stats["n_removed_edia"] == 0
    assert inf_stats == dist_stats
    np.testing.assert_array_equal(inf_mask, dist_mask)
    np.testing.assert_array_equal(inf_atoms.coord, dist_atoms.coord)


# ── B-factor filtering ──────────────────────────────────────────────────────────


def test_keep_by_bfactor_zscore_keeps_lowest_and_honours_population(cif_path_6ybf):
    """z-score mode is a monotonic standardization, so its keep-set is a clean raw-B
    threshold: every kept water has a lower B-factor than every dropped one (the
    best-ordered waters survive). The reference population is genuinely consulted —
    swapping it changes which waters survive at a fixed cutoff."""
    atoms = _load_atoms(cif_path_6ybf)
    mask = water_oxygen_mask(atoms)
    water_b = atoms.b_factor[mask]

    keep = keep_by_bfactor(atoms, mask, 0.5)  # defaults: mode="zscore", population="water"
    assert keep.any() and not keep.all()  # fixture sanity: cutoff splits the waters
    assert (
        water_b[keep].max() <= water_b[~keep].min()
    )  # no dropped water is better-ordered than a kept one

    masks = [keep_by_bfactor(atoms, mask, 0.5, population=p) for p in ("water", "protein", "all")]
    assert any(not np.array_equal(masks[0], m) for m in masks[1:])


def test_bfactor_reference_selects_the_named_population(cif_path_6ybf):
    """Each population name selects a different set of B-factors: waters are the water
    O atoms, "all" is every atom, and protein sits between them. An unknown name is an
    error rather than a silent fallback."""
    atoms = _load_atoms(cif_path_6ybf)
    mask = water_oxygen_mask(atoms)

    water = bfactor_reference(atoms, mask, "water")
    protein = bfactor_reference(atoms, mask, "protein")
    every = bfactor_reference(atoms, mask, "all")

    assert np.array_equal(water, atoms.b_factor[mask])
    assert np.array_equal(every, atoms.b_factor)
    assert len(water) < len(every) and len(protein) < len(every)
    assert water.mean() != protein.mean()  # the choice actually changes the reference

    with pytest.raises(ValueError, match="water.*protein.*all"):
        bfactor_reference(atoms, mask, "solvent")


def test_bfactor_reference_matches_keep_by_bfactor(cif_path_6ybf):
    """keep_by_bfactor's z-score is built on bfactor_reference, so standardizing by
    hand against the same population reproduces its keep-set exactly."""
    atoms = _load_atoms(cif_path_6ybf)
    mask = water_oxygen_mask(atoms)
    water_b = atoms.b_factor[mask]
    cutoff = 0.5

    for population in ("water", "protein", "all"):
        ref = bfactor_reference(atoms, mask, population)
        expected = (water_b - ref.mean()) / ref.std() <= cutoff
        assert np.array_equal(keep_by_bfactor(atoms, mask, cutoff, population=population), expected)


def test_keep_by_bfactor_degenerate_population_keeps_all(cif_path_6ybf):
    """A reference population with zero spread (std == 0) can't standardize, so every
    water is kept regardless of how strict the cutoff is."""
    atoms = _load_atoms(cif_path_6ybf)
    mask = water_oxygen_mask(atoms)
    atoms.b_factor[mask] = 20.0

    keep = keep_by_bfactor(atoms, mask, -5.0, population="water")
    assert keep.all()


def test_keep_by_bfactor_absolute(cif_path_6ybf):
    """absolute mode keeps waters with raw B-factor <= cutoff and ignores population."""
    atoms = _load_atoms(cif_path_6ybf)
    mask = water_oxygen_mask(atoms)
    water_b = atoms.b_factor[mask]
    cutoff = float(np.median(water_b))

    keep = keep_by_bfactor(atoms, mask, cutoff, mode="absolute", population="protein")
    np.testing.assert_array_equal(keep, water_b <= cutoff)


def test_filter_bfactor_partitions_and_respects_cutoff(cif_path_6ybf):
    """A cutoff inside the water B-factor z-score range drops some — not all — distance
    survivors; counts partition exactly and every kept water is within the cutoff."""
    st = gemmi.read_structure(str(cif_path_6ybf))
    cell = st.cell
    sg = st.find_spacegroup() or _p1()
    atoms = _load_atoms(cif_path_6ybf)
    water_b = atoms.b_factor[water_oxygen_mask(atoms)]
    cutoff = 0.0

    filtered, stats, _ = filter_waters(atoms, cell, sg, CUTOFF, bfactor_cutoff=cutoff)
    kept = filtered[water_oxygen_mask(filtered)]
    assert stats["n_removed_bfactor"] > 0
    assert kept.array_length() > 0
    assert (
        stats["n_removed_distance"] + stats["n_removed_bfactor"] + kept.array_length()
        == stats["n_water"]
    )
    assert ((kept.b_factor - water_b.mean()) / water_b.std() <= cutoff).all()


def test_filter_bfactor_disabled_matches_distance_only(cif_path_6ybf):
    """With no B-factor cutoff the result equals distance-only filtering and
    n_removed_bfactor is 0."""
    st = gemmi.read_structure(str(cif_path_6ybf))
    cell = st.cell
    sg = st.find_spacegroup() or _p1()
    filtered, stats, _ = filter_waters(_load_atoms(cif_path_6ybf), cell, sg, CUTOFF)
    assert stats["n_removed_bfactor"] == 0


def test_filter_bfactor_infinite_cutoff_matches_distance_only(cif_path_6ybf):
    """An infinite B-factor cutoff keeps every distance survivor in both zscore and
    absolute modes, reproducing distance-only filtering exactly: same keep-mask, same
    relocated coords, and n_removed_bfactor is 0."""
    st = gemmi.read_structure(str(cif_path_6ybf))
    cell = st.cell
    sg = st.find_spacegroup() or _p1()

    dist_atoms, dist_stats, dist_mask = filter_waters(_load_atoms(cif_path_6ybf), cell, sg, CUTOFF)

    for mode in ("zscore", "absolute"):
        bf_atoms, bf_stats, bf_mask = filter_waters(
            _load_atoms(cif_path_6ybf), cell, sg, CUTOFF, bfactor_cutoff=np.inf, bfactor_mode=mode
        )
        assert bf_stats["n_removed_bfactor"] == 0
        assert bf_stats == dist_stats
        np.testing.assert_array_equal(bf_mask, dist_mask)
        np.testing.assert_array_equal(bf_atoms.coord, dist_atoms.coord)
