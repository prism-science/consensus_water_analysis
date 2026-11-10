import numpy as np
import pandas as pd
import pytest

from cw.metrics import (
    compare_halves,
    consensus_water_mask,
    effect_size,
)

# ── consensus_water_mask ──────────────────────────────────────────────────────


def test_consensus_water_mask_rejects_every_non_consensus_case():
    # cluster 0 clears the cutoff, cluster 1 does not.
    clusters = pd.DataFrame({"cluster_id": [0, 1], "cluster_occupancy": [0.8, 0.1]})
    members = pd.DataFrame(
        {
            "cluster_id": [0, 0, 1, -1],
            "within_cutoff": [True, False, True, False],
        }
    )
    # consensus / radius-rejected / low-occupancy cluster / noise
    assert list(consensus_water_mask(members, clusters, 0.3)) == [
        True,
        False,
        False,
        False,
    ]


# ── compare_halves ────────────────────────────────────────────────────────────


def test_effect_size_rescales_scipys_u_to_cliffs_delta():
    # scipy supplies U; the [-1, 1] rescaling and its orientation are ours, and a
    # flipped sign would invert the `r` column of every reported table.
    low, high = np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])
    assert effect_size(high, low, "mannwhitney") == 1.0
    assert effect_size(low, high, "mannwhitney") == -1.0
    assert effect_size(low, low, "mannwhitney") == 0.0


def test_compare_halves_unit_resampling_ignores_duplicated_rows():
    # Pseudo-replication: duplicating every water within a pdb_id adds no
    # independent information. Row resampling is fooled — the interval shrinks and
    # p collapses — while resampling whole pdb_ids is invariant to the copies.
    # One structure's left value is NaN throughout, so the unit labels have to be
    # masked in step with the values and that unit contributes to one half only.
    rng = np.random.default_rng(0)
    per_unit = rng.normal(0.0, 1.0, 30)
    left_per_unit = per_unit + 1.0
    left_per_unit[7] = np.nan
    n_boot = 400

    def compare(reps):
        units = np.repeat(np.arange(30), reps)
        left, right = np.repeat(left_per_unit, reps), np.repeat(per_unit, reps)
        return (
            compare_halves(left, right, n_boot=n_boot),
            compare_halves(left, right, n_boot=n_boot, left_units=units, right_units=units),
        )

    def width(result):
        return result["ci_high"] - result["ci_low"]

    rows_once, clustered_once = compare(1)
    rows_duplicated, clustered_duplicated = compare(16)

    assert clustered_once["n_left"] == 29  # the NaN structure dropped
    assert clustered_once["n_right"] == 30

    assert width(clustered_duplicated) == pytest.approx(width(clustered_once))
    assert clustered_duplicated["p"] == clustered_once["p"]
    # p from an inverted bootstrap is a bound, not a measurement: it floors here.
    assert clustered_once["p"] == pytest.approx(1 / n_boot)

    assert width(rows_duplicated) < width(rows_once) / 2
    assert rows_duplicated["p"] < rows_once["p"]


# ── match_consensus_sites ─────────────────────────────────────────────────────


def _sites(ids, occupancies, xs):
    return pd.DataFrame(
        {
            "cluster_id": ids,
            "cluster_occupancy": occupancies,
            "center_x": xs,
            "center_y": 0.0,
            "center_z": 0.0,
        }
    )


def test_match_consensus_sites_one_to_one_lost_and_new():
    from cw.metrics import match_consensus_sites

    # baseline: sites at x=0 and x=0.6 (both consensus) plus a sub-cutoff site at x=10
    baseline = _sites([0, 1, 2], [0.9, 0.5, 0.1], [0.0, 0.6, 10.0])
    # other: one site at x=0.3 (equidistant-ish to both), one far away at x=5
    other = _sites([7, 8], [0.8, 0.4], [0.3, 5.0])

    matched = match_consensus_sites(baseline, other, occupancy_cutoff=0.3, radius=1.0)

    pairs = matched.dropna(subset=["baseline_cluster_id", "other_cluster_id"])
    assert len(pairs) == 1  # site 7 can serve only one baseline site
    assert pairs["other_cluster_id"].iloc[0] == 7
    assert pairs["distance"].iloc[0] == pytest.approx(0.3)

    lost = matched[matched["other_cluster_id"].isna()]
    assert len(lost) == 1 and lost["distance"].isna().all()
    new = matched[matched["baseline_cluster_id"].isna()]
    assert new["other_cluster_id"].tolist() == [8]
    # the sub-cutoff baseline site is ignored entirely
    assert 2 not in set(matched["baseline_cluster_id"].dropna())


def test_match_consensus_sites_prefers_minimum_total_distance():
    from cw.metrics import match_consensus_sites

    # greedy nearest-first would pair a0-b0 (0.1) and strand a1; the assignment
    # pairs a0-b1 and a1-b0 so both baseline sites survive
    baseline = _sites([0, 1], [0.9, 0.9], [0.0, 0.8])
    other = _sites([0, 1], [0.9, 0.9], [0.1, -0.5])
    matched = match_consensus_sites(baseline, other, occupancy_cutoff=0.3, radius=1.0)
    assert matched[["baseline_cluster_id", "other_cluster_id"]].notna().all(axis=1).sum() == 2


def test_match_consensus_sites_empty_side():
    from cw.metrics import match_consensus_sites

    baseline = _sites([0], [0.9], [0.0])
    other = _sites([0], [0.1], [0.0])  # nothing clears the cutoff
    matched = match_consensus_sites(baseline, other, occupancy_cutoff=0.3, radius=1.0)
    assert len(matched) == 1 and matched["other_cluster_id"].isna().all()
