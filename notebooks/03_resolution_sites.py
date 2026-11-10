import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # 03 — Consensus sites across resolution cutoffs

    Matches each cohort's consensus sites to a **baseline** cohort's (the first
    one entered) and asks which baseline sites survive. Built for one set of
    datasets re-refined at several high-resolution cutoffs, each run through the
    pipeline as its own cohort.

    All cohorts must share one coordinate frame: align every cohort to the **same
    reference CIF** (config on the baseline tree, `--source-dir` for the others).
    Sites are matched one-to-one by minimum total distance, within the cluster
    member radius; a baseline site with no partner is *lost*, a partner-less site
    in the other cohort is *new*.
    """)
    return


@app.cell
def _():
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(".")))

    import config
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from cw.metrics import match_consensus_sites

    return Path, config, match_consensus_sites, np, pd, plt


@app.cell
def _(mo):
    cohorts_input = mo.ui.text(
        value=(
            "ucsf_nsp3_mac1_P43_d1.0,ucsf_nsp3_mac1_P43_d1.25,"
            "ucsf_nsp3_mac1_P43_d1.5,ucsf_nsp3_mac1_P43_d2.0"
        ),
        label="cohorts (comma-separated; first = baseline)",
        full_width=True,
    )
    cohort_labels_input = mo.ui.text(
        value="1.0 Å,1.25 Å,1.5 Å,2.0 Å",
        label="labels (comma-separated, same order; blank = cohort names)",
        full_width=True,
    )
    cutoff_input = mo.ui.number(
        start=0.0, stop=1.0, step=0.05, value=0.3, label="consensus cutoff",
    )
    mo.vstack([cohorts_input, cohort_labels_input, cutoff_input])
    return cohort_labels_input, cohorts_input, cutoff_input


@app.cell
def _(Path, cohort_labels_input, cohorts_input, config, cutoff_input, mo, pd):
    COHORTS = [c.strip() for c in cohorts_input.value.split(",") if c.strip()]
    mo.stop(
        len(COHORTS) < 2,
        mo.md("**Enter at least two cohorts above** — the first is the baseline."),
    )
    _labels = [s.strip() for s in cohort_labels_input.value.split(",") if s.strip()]
    LABELS = _labels if len(_labels) == len(COHORTS) else COHORTS
    BASELINE = COHORTS[0]

    occupancy_cutoff = float(cutoff_input.value)
    match_radius = config.CLUSTER_MEMBER_RADIUS

    clusters_by_cohort = {
        _cohort: pd.read_csv(Path(config.DATA_DIR) / _cohort / "clusters.csv")
        for _cohort in COHORTS
    }
    PLOTS_DIR = Path("data") / "plots" / f"{BASELINE}_resolution_sites"

    print(f"baseline:     {BASELINE}")
    print(f"match radius: {match_radius} Å; consensus cutoff: {occupancy_cutoff}")
    return (
        BASELINE,
        COHORTS,
        LABELS,
        PLOTS_DIR,
        clusters_by_cohort,
        match_radius,
        occupancy_cutoff,
    )


@app.cell
def _(
    BASELINE,
    COHORTS,
    LABELS,
    clusters_by_cohort,
    match_consensus_sites,
    match_radius,
    occupancy_cutoff,
    pd,
):
    # One matched table per non-baseline cohort, stacked with a `cohort` column.
    # The baseline matched against itself is the trivial all-survive row, kept so
    # the summary and plots start from 100%.
    matches = pd.concat(
        [
            match_consensus_sites(
                clusters_by_cohort[BASELINE], clusters_by_cohort[_cohort],
                occupancy_cutoff, match_radius,
            ).assign(cohort=_cohort, label=_label)
            for _cohort, _label in zip(COHORTS, LABELS)
        ],
        ignore_index=True,
    )

    def _summarise(group):
        is_baseline = group["baseline_cluster_id"].notna()
        is_other = group["other_cluster_id"].notna()
        matched = is_baseline & is_other
        return pd.Series({
            "n_sites": int(is_other.sum()),
            "n_baseline_kept": int(matched.sum()),
            "baseline_kept_fraction": matched.sum() / is_baseline.sum(),
            "n_baseline_lost": int((is_baseline & ~is_other).sum()),
            "n_new": int((is_other & ~is_baseline).sum()),
            "median_shift": group.loc[matched, "distance"].median(),
            "p90_shift": group.loc[matched, "distance"].quantile(0.9),
        })

    site_summary = (
        matches.groupby("label", sort=False)[
            ["baseline_cluster_id", "other_cluster_id", "distance"]
        ]
        .apply(_summarise)
        .round(3)
    )
    site_summary
    return (matches,)


@app.cell
def _(mo):
    mo.md("""
    ## Which baseline sites survive?

    Left: fraction of baseline consensus sites kept at each cutoff, split by how
    strong the site was at baseline (its consensus score). Strong sites should
    hold up longest. Right: how far the surviving site centers moved.
    """)
    return


@app.cell
def _(PLOTS_DIR, mo):
    survival_save_path = mo.ui.text(
        value=str(PLOTS_DIR / "site_survival.png"), label="save path", full_width=True,
    )
    survival_save_button = mo.ui.run_button(label="save figure")
    mo.hstack([survival_save_path, survival_save_button], justify="start")
    return survival_save_button, survival_save_path


@app.cell
def _(
    LABELS,
    Path,
    matches,
    np,
    occupancy_cutoff,
    pd,
    plt,
    survival_save_button,
    survival_save_path,
):
    _baseline_rows = matches[matches["baseline_cluster_id"].notna()].copy()
    _baseline_rows["kept"] = _baseline_rows["other_cluster_id"].notna()
    _edges = [occupancy_cutoff, 0.5, 0.7, 0.9, 1.0 + 1e-9]
    _baseline_rows["score_bin"] = pd.cut(
        _baseline_rows["baseline_occupancy"], bins=_edges, right=False,
    )
    _survival = (
        _baseline_rows.groupby(["score_bin", "label"], observed=True, sort=False)["kept"]
        .mean()
        .unstack("label")
        .reindex(columns=LABELS)
    )

    _fig, (_ax_survival, _ax_shift) = plt.subplots(1, 2, figsize=(9, 3.5))
    _x = np.arange(len(LABELS))
    _overall = _baseline_rows.groupby("label", sort=False)["kept"].mean().reindex(LABELS)
    _ax_survival.plot(_x, _overall.to_numpy(), color="k", lw=2.5, marker="o", label="all")
    for _bin, _row in _survival.iterrows():
        _ax_survival.plot(
            _x, _row.to_numpy(), marker="o", lw=1.2,
            label=f"score {_bin.left:.1f}–{min(_bin.right, 1.0):.1f}",
        )
    _ax_survival.set_xticks(_x, LABELS)
    _ax_survival.set_ylim(0, 1.02)
    _ax_survival.set_ylabel("baseline sites kept")
    _ax_survival.legend(fontsize=8, frameon=False)

    _matched = matches.dropna(subset=["baseline_cluster_id", "other_cluster_id"])
    _shift_labels = LABELS[1:]
    _ax_shift.boxplot(
        [_matched.loc[_matched["label"] == _l, "distance"].to_numpy() for _l in _shift_labels],
        tick_labels=_shift_labels, showfliers=False,
    )
    _ax_shift.set_ylabel("center shift vs baseline (Å)")

    for _axis in (_ax_survival, _ax_shift):
        _axis.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    if survival_save_button.value:
        _out = Path(survival_save_path.value)
        _out.parent.mkdir(parents=True, exist_ok=True)
        _fig.savefig(_out, dpi=300, bbox_inches="tight")
        print(f"saved to {_out}")
    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    ## Consensus score: baseline vs each cutoff

    One point per site. Lost baseline sites sit on the x-axis (score 0 at the
    cutoff); new sites sit on the y-axis. Points on the diagonal kept their score.
    """)
    return


@app.cell
def _(PLOTS_DIR, mo):
    score_save_path = mo.ui.text(
        value=str(PLOTS_DIR / "site_score_vs_baseline.png"), label="save path", full_width=True,
    )
    score_save_button = mo.ui.run_button(label="save figure")
    mo.hstack([score_save_path, score_save_button], justify="start")
    return score_save_button, score_save_path


@app.cell
def _(LABELS, Path, matches, plt, score_save_button, score_save_path):
    _panels = LABELS[1:]
    _fig, _axes = plt.subplots(
        1, len(_panels), figsize=(3.2 * len(_panels), 3.2), sharex=True, sharey=True,
        squeeze=False,
    )
    for _axis, _label in zip(_axes[0], _panels):
        _rows = matches[matches["label"] == _label]
        _x = _rows["baseline_occupancy"].fillna(0.0)
        _y = _rows["other_occupancy"].fillna(0.0)
        _axis.scatter(_x, _y, s=10, alpha=0.5, edgecolor="none")
        _axis.plot([0, 1], [0, 1], color="gray", lw=1, ls="--")
        _axis.set_title(_label)
        _axis.set_xlabel(f"score at {LABELS[0]}")
        _axis.spines[["top", "right"]].set_visible(False)
    _axes[0][0].set_ylabel("score at cutoff")
    plt.tight_layout()

    if score_save_button.value:
        _out = Path(score_save_path.value)
        _out.parent.mkdir(parents=True, exist_ok=True)
        _fig.savefig(_out, dpi=300, bbox_inches="tight")
        print(f"saved to {_out}")
    _fig
    return


if __name__ == "__main__":
    app.run()
