from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.spatial import cKDTree  # ty: ignore[unresolved-import] (compiled re-export)


def precision_recall(
    reference_coords: np.ndarray,
    predicted_coords: np.ndarray,
    dist_cutoff: float,
) -> dict[str, float]:
    """Precision and recall for predicted water positions against a reference set.

    recall    = fraction of reference points with ≥1 predicted point within dist_cutoff
    precision = fraction of predicted points within dist_cutoff of any reference point
    f1        = harmonic mean of precision and recall (0.0 when both are 0)
    """
    ref_tree = cKDTree(reference_coords)
    pred_tree = cKDTree(predicted_coords)

    covered_refs = set(
        idx
        for matches in ref_tree.query_ball_point(predicted_coords, r=dist_cutoff)
        for idx in matches
    )
    recall = len(covered_refs) / len(reference_coords)

    covered_preds = set(
        idx
        for matches in pred_tree.query_ball_point(reference_coords, r=dist_cutoff)
        for idx in matches
    )
    precision = len(covered_preds) / len(predicted_coords)

    denom = precision + recall
    f1 = 2 * precision * recall / denom if denom > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def matched_precision_recall(
    reference_coords: np.ndarray,
    predicted_coords: np.ndarray,
    dist_cutoff: float,
) -> dict[str, float]:
    """One-to-one matched precision/recall — each water matched to at most one counterpart.

    Unlike precision_recall, a single predicted water cannot be credited for
    covering multiple references: maximum bipartite matching pairs references and
    predictions that are within dist_cutoff with no vertex reused. n_matched is
    the size of that matching, so precision = recall = 1 only when both sets pair
    up exactly.
    """
    ref_tree = cKDTree(reference_coords)
    pred_tree = cKDTree(predicted_coords)

    dmat = ref_tree.sparse_distance_matrix(pred_tree, dist_cutoff, output_type="coo_matrix")
    adj = csr_matrix((np.ones(dmat.nnz, dtype=bool), (dmat.row, dmat.col)), shape=dmat.shape)

    matching = maximum_bipartite_matching(adj, perm_type="column")
    n_matched = int((matching >= 0).sum())

    return {
        "precision": n_matched / len(predicted_coords),
        "recall": n_matched / len(reference_coords),
    }


def chamfer_distance(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    """Symmetric mean nearest-neighbor distance between two water sets (Å).

    Continuous cousin of precision_recall: instead of thresholding the NN
    distance at a cutoff and counting, it averages the raw NN distance in each
    direction and sums them. This is the mean-NN variant (not summed-squared).
    Lower is closer.
    """
    tree_a = cKDTree(coords_a)
    tree_b = cKDTree(coords_b)
    d_ab, _ = tree_b.query(coords_a)
    d_ba, _ = tree_a.query(coords_b)
    return float(d_ab.mean() + d_ba.mean())


def consensus_centers(clusters: pd.DataFrame, occupancy_cutoff: float) -> np.ndarray:
    """Center coordinates of clusters with cluster_occupancy ≥ occupancy_cutoff."""
    mask = clusters["cluster_occupancy"] >= occupancy_cutoff
    return clusters.loc[mask, ["center_x", "center_y", "center_z"]].to_numpy()


def consensus_water_mask(
    cluster_members: pd.DataFrame,
    clusters: pd.DataFrame,
    occupancy_cutoff: float,
) -> pd.Series:
    """Boolean mask over cluster_members rows: True for a consensus water.

    A water is consensus iff it is a within-cutoff member of a cluster whose
    cluster_occupancy clears occupancy_cutoff. Everything else is False — noise,
    radius-rejected members, and members of clusters below the cutoff. Indexed
    like cluster_members, so it can be assigned straight back onto it.
    """
    consensus_ids = clusters.loc[clusters["cluster_occupancy"] >= occupancy_cutoff, "cluster_id"]
    return cluster_members["within_cutoff"] & cluster_members["cluster_id"].isin(set(consensus_ids))


def pareto_front(
    pr_df: pd.DataFrame,
    x_col: str = "recall",
    y_col: str = "precision",
) -> pd.DataFrame:
    """Non-dominated subset of pr_df, maximizing both x_col and y_col.

    A row is dominated when another row has x' ≥ x and y' ≥ y with at least one
    strict; the survivors form the upper-right (Pareto-optimal) envelope of the
    point cloud. Returned sorted ascending by x_col. O(n²) — fine for the few
    hundred structures in a cohort.
    """
    pts = pr_df[[x_col, y_col]].to_numpy()
    keep = np.ones(len(pts), dtype=bool)
    for i in range(len(pts)):
        dominated = (
            (pts[:, 0] >= pts[i, 0])
            & (pts[:, 1] >= pts[i, 1])
            & ((pts[:, 0] > pts[i, 0]) | (pts[:, 1] > pts[i, 1]))
        )
        if dominated.any():
            keep[i] = False
    return pr_df[keep].sort_values(x_col).reset_index(drop=True)


def per_structure_consensus_pr(
    cluster_members: pd.DataFrame,
    center_coords: np.ndarray,
    dist_cutoff: float,
) -> pd.DataFrame:
    """Precision/recall of each structure's waters against the consensus centers.

    For every pdb_id, treats the consensus cluster centers as the reference set
    and that structure's waters as the prediction, then applies precision_recall.
    Returns one row per structure: pdb_id, precision, recall, f1, num_water.

    num_water is that structure's pooled water count — the size of its prediction
    set — taken straight from cluster_members, so it reflects the waters actually
    clustered for this cohort (e.g. after an EDIA/B-factor filter) rather than any
    external metadata total, and needs no metadata.csv.
    """
    rows = []
    for pdb_id, group in cluster_members.groupby("pdb_id"):
        water_coords = group[["x", "y", "z"]].to_numpy()
        pr = precision_recall(center_coords, water_coords, dist_cutoff)
        rows.append({"pdb_id": pdb_id, **pr, "num_water": len(group)})
    return pd.DataFrame(rows)


def clustering_summary(
    clusters: pd.DataFrame,
    cluster_members: pd.DataFrame,
    occupancy_cutoff: float,
    match_radius: float,
) -> dict[str, float | int | None]:
    """Cohort-level headline metrics for one clustering result.

    Reads a (clusters, cluster_members) pair — the artifacts of the clustering
    stage — and returns a flat dict:

      num_water               pooled water oxygens (every cluster_members row)
      num_conserved_clusters  clusters with cluster_occupancy >= occupancy_cutoff
      conserved_clusters_frac that count over the total number of clusters
      conserved_water_frac    within-radius members of conserved clusters over all
                              pooled waters — noise and radius-rejected waters are
                              excluded from the numerator, not the denominator
      knee_precision/recall/f1  the max-F1 per-structure point (the Pareto knee /
                              p-r-panel star)

    The consensus-dependent fields are None when there are no clusters, no pooled
    waters, or no clusters clear occupancy_cutoff.
    """
    num_water = len(cluster_members)
    n_clusters = len(clusters)
    summary: dict[str, float | int | None] = {
        "num_water": num_water,
        "num_conserved_clusters": None,
        "conserved_clusters_frac": None,
        "conserved_water_frac": None,
        "knee_precision": None,
        "knee_recall": None,
        "knee_f1": None,
    }
    if n_clusters == 0 or num_water == 0:
        return summary

    conserved_ids = clusters.loc[clusters["cluster_occupancy"] >= occupancy_cutoff, "cluster_id"]
    in_conserved = consensus_water_mask(cluster_members, clusters, occupancy_cutoff)
    summary["num_conserved_clusters"] = int(len(conserved_ids))
    summary["conserved_clusters_frac"] = len(conserved_ids) / n_clusters
    summary["conserved_water_frac"] = int(in_conserved.sum()) / num_water

    centers = consensus_centers(clusters, occupancy_cutoff)
    if len(centers):
        pr_df = per_structure_consensus_pr(cluster_members, centers, match_radius)
        knee = pr_df.loc[pr_df["f1"].idxmax()]
        summary["knee_precision"] = float(knee["precision"])
        summary["knee_recall"] = float(knee["recall"])
        summary["knee_f1"] = float(knee["f1"])
    return summary


def per_structure_consensus_chamfer(
    cluster_members: pd.DataFrame,
    center_coords: np.ndarray,
) -> pd.DataFrame:
    """Chamfer distance of each structure's waters against the consensus centers.

    Continuous counterpart to per_structure_consensus_pr. Returns one row per
    structure: pdb_id, chamfer (Å, lower is closer).
    """
    rows = []
    for pdb_id, group in cluster_members.groupby("pdb_id"):
        water_coords = group[["x", "y", "z"]].to_numpy()
        rows.append({"pdb_id": pdb_id, "chamfer": chamfer_distance(center_coords, water_coords)})
    return pd.DataFrame(rows)


def match_consensus_sites(
    baseline_clusters: pd.DataFrame,
    other_clusters: pd.DataFrame,
    occupancy_cutoff: float,
    radius: float,
) -> pd.DataFrame:
    """One-to-one match of two cohorts' consensus sites (same coordinate frame).

    Consensus sites are clusters with cluster_occupancy ≥ occupancy_cutoff. Pairs
    are chosen by minimum total distance (Hungarian assignment) among pairs no
    farther apart than `radius`, so one site never absorbs two. Returns one row
    per site, outer-joined: baseline_cluster_id / other_cluster_id (NaN when that
    side has no partner), their occupancies, and the center distance in Å (NaN
    when unmatched). A row with only a baseline id is a lost site; only an other
    id, a new site.
    """
    from scipy.optimize import linear_sum_assignment

    columns = ["cluster_id", "cluster_occupancy", "center_x", "center_y", "center_z"]
    base = baseline_clusters.loc[
        baseline_clusters["cluster_occupancy"] >= occupancy_cutoff, columns
    ].reset_index(drop=True)
    other = other_clusters.loc[
        other_clusters["cluster_occupancy"] >= occupancy_cutoff, columns
    ].reset_index(drop=True)

    base_xyz = base[["center_x", "center_y", "center_z"]].to_numpy(dtype=float)
    other_xyz = other[["center_x", "center_y", "center_z"]].to_numpy(dtype=float)

    pairs: list[tuple[int, int, float]] = []
    if len(base) and len(other):
        distance = np.linalg.norm(base_xyz[:, None, :] - other_xyz[None, :, :], axis=-1)
        # pairs beyond the radius get a cost larger than any feasible total, so the
        # assignment only uses them when forced, and they are dropped afterwards
        cost = np.where(distance <= radius, distance, radius * (len(base) + len(other) + 1))
        rows, cols = linear_sum_assignment(cost)
        pairs = [(r, c, distance[r, c]) for r, c in zip(rows, cols) if distance[r, c] <= radius]

    matched_base = {r for r, _, _ in pairs}
    matched_other = {c for _, c, _ in pairs}
    records = [
        {
            "baseline_cluster_id": base.at[r, "cluster_id"],
            "other_cluster_id": other.at[c, "cluster_id"],
            "baseline_occupancy": base.at[r, "cluster_occupancy"],
            "other_occupancy": other.at[c, "cluster_occupancy"],
            "distance": float(d),
        }
        for r, c, d in pairs
    ]
    records += [
        {"baseline_cluster_id": base.at[r, "cluster_id"],
         "baseline_occupancy": base.at[r, "cluster_occupancy"]}
        for r in range(len(base)) if r not in matched_base
    ]
    records += [
        {"other_cluster_id": other.at[c, "cluster_id"],
         "other_occupancy": other.at[c, "cluster_occupancy"]}
        for c in range(len(other)) if c not in matched_other
    ]
    return pd.DataFrame(
        records,
        columns=["baseline_cluster_id", "other_cluster_id", "baseline_occupancy",
                 "other_occupancy", "distance"],
    )


# ── two-sample comparison of a split's two halves ─────────────────────────────


def effect_size(a: np.ndarray, b: np.ndarray, test: str) -> float:
    """Signed effect size matching `test`; positive = `a` sits higher than `b`.

    mannwhitney  Cliff's delta = P(a>b) - P(a<b), which is scipy's Mann-Whitney U
                 rescaled to [-1, 1] (and, for two independent samples, identical
                 to the rank-biserial correlation). 0 = the halves overlap
                 completely, ±1 = they separate completely.
    ks           the KS D statistic (unsigned, in [0, 1])
    welch        Cohen's d
    """
    if test == "ks":
        return float(stats.ks_2samp(a, b).statistic)
    if test == "welch":
        pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        return float((a.mean() - b.mean()) / pooled) if pooled > 0 else float("nan")
    # method="asymptotic" rather than the default "auto": only the p-value differs
    # between methods and it is discarded here, while "auto" warns once per call on
    # small tied samples — thousands of times inside a bootstrap.
    u = stats.mannwhitneyu(a, b, method="asymptotic").statistic
    return 2 * u / (a.size * b.size) - 1


def group_values(values: np.ndarray, units: np.ndarray) -> dict:
    """Split `values` into one array per unit label."""
    order = np.argsort(units, kind="stable")
    values, units = values[order], units[order]
    edges = np.flatnonzero(np.r_[True, units[1:] != units[:-1], True])
    return {units[i]: values[i:j] for i, j in zip(edges[:-1], edges[1:], strict=True)}


def compare_halves(
    left_values,
    right_values,
    test: str = "mannwhitney",
    n_boot: int = 2000,
    seed: int = 0,
    left_units=None,
    right_units=None,
) -> dict:
    """Two-sample comparison of a split's two halves, NaNs dropped.

    Returns p / effect / ci_low / ci_high / n_left / n_right. The effect size is
    the headline number and the p-value secondary: at large n a difference far too
    small to matter still clears p < 0.001.

    `n_boot` percentile-bootstrap resamples (0 to skip) give the 95% CI on the
    effect size. Pass `*_units` when rows are not independent (waters sharing a
    pdb_id): the bootstrap then resamples whole units, one shared draw per
    iteration so the within-unit split stays paired, and p is inverted from that
    distribution rather than from a test that would count correlated rows as
    independent. An inverted p is a bound, not a measurement: it floors at 1 / n_boot.
    """
    a, b = np.asarray(left_values, dtype=float), np.asarray(right_values, dtype=float)
    keep_a, keep_b = np.isfinite(a), np.isfinite(b)
    a, b = a[keep_a], b[keep_b]
    if a.size < 2 or b.size < 2:
        return dict(
            p=float("nan"),
            effect=float("nan"),
            ci_low=float("nan"),
            ci_high=float("nan"),
            n_left=a.size,
            n_right=b.size,
        )
    clustered = left_units is not None

    p = float("nan")
    if not clustered:
        if test == "mannwhitney":
            p = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
        elif test == "ks":
            p = stats.ks_2samp(a, b).pvalue
        else:
            p = stats.ttest_ind(a, b, equal_var=False).pvalue

    ci_low = ci_high = float("nan")
    if n_boot:
        rng = np.random.default_rng(seed)
        if clustered:
            by_unit_a = group_values(a, np.asarray(left_units)[keep_a])
            by_unit_b = group_values(b, np.asarray(right_units)[keep_b])
            names = np.array(sorted(set(by_unit_a) | set(by_unit_b)))
            empty = np.empty(0)

            def resample():
                drawn = rng.choice(names, names.size)
                return (
                    np.concatenate([by_unit_a.get(u, empty) for u in drawn]),
                    np.concatenate([by_unit_b.get(u, empty) for u in drawn]),
                )
        else:

            def resample():
                return rng.choice(a, a.size), rng.choice(b, b.size)

        boot = []
        for _ in range(int(n_boot)):
            draw_a, draw_b = resample()
            boot.append(
                effect_size(draw_a, draw_b, test) if draw_a.size and draw_b.size else np.nan
            )
        boot = np.array(boot)
        ci_low, ci_high = np.nanpercentile(boot, [2.5, 97.5])
        if clustered:
            side = min(np.nanmean(boot <= 0), np.nanmean(boot >= 0))
            p = float(np.clip(2 * side, 1 / int(n_boot), 1.0))
    return dict(
        p=float(p),
        effect=effect_size(a, b, test),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        n_left=a.size,
        n_right=b.size,
    )


def spearman(df: pd.DataFrame, x_col: str, y_col: str) -> tuple[float, float]:
    """Rank correlation of two columns, NaNs dropped; (nan, nan) when undefined
    (the same column twice, or fewer than 3 complete pairs)."""
    if x_col == y_col:
        return float("nan"), float("nan")
    pair = df[[x_col, y_col]].dropna()
    if len(pair) < 3:
        return float("nan"), float("nan")
    rho, p = stats.spearmanr(pair[x_col], pair[y_col])
    return float(rho), float(p)
