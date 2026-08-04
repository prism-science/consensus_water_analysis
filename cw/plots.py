from __future__ import annotations

from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

METRIC_LABELS = {
    "edia": "EDIA",
    "b_factor_zscore": "B-factor z-score",
    "b_factor": "B-factor",
    "occupancy": "occupancy",
    "resolution": "resolution (Å)",
    "deposited_r_free": "deposited R-free",
    "deposited_r_work": "deposited R-work",
    "r_free": "R-free",
    "r_work": "R-work",
    "num_water": "#water (clustered)",
    "num_water_deposited": "#water (deposited)",
    "unit_cell_volume": "unit-cell volume (Å³)",
    "f1": "F1",
    "precision": "precision",
    "recall": "recall",
}

# A split spec — {col, left, right} with each side carrying its `value` in that
# column plus its label and colors — defines a comparison once so the violin, Q-Q
# and statistics views of it cannot drift apart.
WATER_SPLIT = dict(
    col="group",
    left=dict(value="consensus", label="consensus", fill="r", edge="r", mark="darkred"),
    right=dict(
        value="nonconsensus", label="non-consensus", fill="grey", edge="dimgrey", mark="black"
    ),
)


def structure_split_spec(split_metric, metric_labels=None):
    """Split spec for the per-structure good/poor comparison, labeled by the
    metric the split was cut on. Blue accent instead of the per-water figure's red,
    so the two levels are never confused; red stays reserved for consensus waters."""
    labels = METRIC_LABELS if metric_labels is None else metric_labels
    return dict(
        col="group",
        left=dict(
            value="good",
            label=f"≥ {labels.get(split_metric, split_metric)} cutoff",
            fill="mediumblue",
            edge="mediumblue",
            mark="darkblue",
        ),
        right=dict(
            value="poor",
            label="below cutoff",
            fill="saddlebrown",
            edge="dimgrey",
            mark="saddlebrown",
        ),
    )


def quantile_boundaries(values, n_bins, quantile_range=(0.01, 0.99)):
    """Unique quantile bin edges over the inner quantile_range of values (default
    1–99%, so the extreme tails are capped). Integer edges when values are
    integer-valued. plot_pr_scatter builds its color bins from these; callers can
    reuse the same edges to bin a related series (e.g. a num_water tradeoff curve)
    onto bands identical to the colorbar."""
    values = np.asarray(values)
    q_lo, q_hi = quantile_range
    edges = np.unique(np.quantile(values, np.linspace(q_lo, q_hi, n_bins + 1)))
    if np.allclose(values, np.round(values)):
        edges = np.unique(np.rint(edges))
    return edges


def noise_stats(cluster_members):
    """(n_structures, noise_occupancy, noise_count) for a cluster_members table.
    noise_occupancy is the occupancy a single-structure water would carry
    (1 / n_structures); noise_count is the number of unclustered (cluster_id ==
    -1) waters. Used to place the noise pile in occupancy histograms."""
    n_structures = cluster_members["pdb_id"].nunique()
    noise_occupancy = 1.0 / n_structures if n_structures else 0.0
    noise_count = int((cluster_members["cluster_id"] == -1).sum())
    return n_structures, noise_occupancy, noise_count


def broken_y_axis(ax_top, ax_bot, *, d=0.015, left=True, right=True, linewidth=1, color="k"):
    """Turn two vertically stacked axes into a broken y-axis: hide the facing
    spines/ticks and draw the diagonal break marks across the gap. Assumes
    ax_top sits above ax_bot and both share the x-axis. left/right select which
    corners get break marks — set right=False when a continuous overlay axis
    (e.g. a cumulative curve) sits on the right and must not look broken.
    linewidth and color set the thickness and color of the diagonal break marks."""
    ax_top.spines["bottom"].set_visible(False)
    ax_bot.spines["top"].set_visible(False)
    ax_top.tick_params(bottom=False)
    break_kw = dict(color=color, clip_on=False, linewidth=linewidth, transform=ax_top.transAxes)
    if left:
        ax_top.plot((-d, +d), (-d, +d), **break_kw)
    if right:
        ax_top.plot((1 - d, 1 + d), (-d, +d), **break_kw)
    break_kw["transform"] = ax_bot.transAxes
    if left:
        ax_bot.plot((-d, +d), (1 - d, 1 + d), **break_kw)
    if right:
        ax_bot.plot((1 - d, 1 + d), (1 - d, 1 + d), **break_kw)


def plot_pr_scatter(
    pr_df: pd.DataFrame,
    *,
    color=None,
    color_label: str | None = None,
    precision_col: str = "precision",
    recall_col: str = "recall",
    ax=None,
    title: str | None = None,
    cmap: str = "viridis",
    style: str = "scatter",
    marker_size: float = 20,
    alpha: float = 0.8,
    lim: tuple[float, float] | None = (0.0, 1.01),
    n_color_bins: int | None = None,
    color_quantile_range: tuple[float, float] = (0.01, 0.99),
    fontsize: float | None = None,
    cbar_kwargs: dict | None = None,
):
    """Precision vs recall for per-structure rows, on an equal-aspect square.

    color selects point coloring: a column name in pr_df, an array aligned to its
    rows, or None for a flat color. color_label labels the colorbar and defaults to
    the column name.

    n_color_bins, when set, discretizes the color scale into that many quantile bins
    over color_quantile_range (default inner 1–99%) — each bin holds ~equal counts and
    the extreme tails are capped, so the colorbar grows up/down triangles for the
    capped values. cmap is resampled to one distinct color per bin (with the two
    triangle colors taken from its ends), so qualitative palettes (e.g. "tab10")
    render as cleanly separated bands rather than collapsing onto one color — as long
    as the palette carries at least n_color_bins + 2 colors. Integer-valued colors get
    integer bin edges. Leave None for a plain continuous scale.

    style is "scatter" (alpha honored) or "hexbin" (hexes colored by the mean color
    value per cell). lim sets both axes to the same range (equal x/y); None autoscales.
    fontsize, when set, sizes every text element (axis labels, title, tick labels, and
    the colorbar label/ticks) so the caller can match it to a legend/annotation drawn
    on the returned ax; None keeps matplotlib defaults.

    cbar_kwargs is merged into the fig.colorbar call (over the {"pad": 0.02} default),
    so a caller can shrink the bar (e.g. {"fraction": 0.046}) when the square-aspect
    scatter leaves it towering over a small panel. Returns (fig, ax).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3))
    else:
        fig = ax.figure

    color_values = None
    if color is not None:
        if isinstance(color, str):
            color_values = pr_df[color].to_numpy()
            color_label = color_label or color
        else:
            color_values = np.asarray(color)

    norm, boundaries, plot_cmap = None, None, cmap
    if color_values is not None and n_color_bins:
        edges = quantile_boundaries(color_values, n_color_bins, color_quantile_range)
        if len(edges) >= 2:
            boundaries = edges
            n_bands = len(edges) - 1
            # One distinct color per band. Sample n_bands + 2 colors and build a
            # ListedColormap of exactly n_bands entries with the outer two as the
            # under/over (extend-triangle) colors, then size the norm to that band
            # count. Handing the norm the raw cmap with ncolors=256 instead lets a
            # small qualitative colormap (e.g. Dark2's 8 entries) overflow every
            # band past the first onto its last color — the "all grey" bug.
            picks = plt.get_cmap(cmap)(np.linspace(0, 1, n_bands + 2))
            plot_cmap = mcolors.ListedColormap(picks[1:-1]).with_extremes(
                under=picks[0], over=picks[-1]
            )
            norm = mcolors.BoundaryNorm(boundaries, ncolors=plot_cmap.N)

    if style == "hexbin":
        sc = ax.hexbin(
            pr_df[recall_col],
            pr_df[precision_col],
            C=color_values,
            reduce_C_function=np.mean,
            gridsize=20,
            cmap=plot_cmap,
            norm=norm,
            mincnt=1,
        )
    elif color_values is None:
        ax.scatter(
            pr_df[recall_col],
            pr_df[precision_col],
            alpha=alpha,
            edgecolors="k",
            s=marker_size,
            color="steelblue",
        )
        sc = None
    else:
        sc = ax.scatter(
            pr_df[recall_col],
            pr_df[precision_col],
            c=color_values,
            cmap=plot_cmap,
            norm=norm,
            alpha=alpha,
            edgecolors="k",
            s=marker_size,
        )

    if sc is not None and color_values is not None:
        cbar_opts: dict[str, Any] = {"pad": 0.02, **(cbar_kwargs or {})}
        cbar = (
            fig.colorbar(
                sc,
                ax=ax,
                extend="both",
                spacing="uniform",
                ticks=boundaries,
                format="%.3g",
                **cbar_opts,
            )
            if boundaries is not None
            else fig.colorbar(sc, ax=ax, **cbar_opts)
        )
        if color_label:
            cbar.set_label(color_label, fontsize=fontsize)
        cbar.ax.tick_params(labelsize=fontsize)

    ax.set_xlabel("recall", fontsize=fontsize)
    ax.set_ylabel("precision", fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    if lim is not None:
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        # equal range → force identical ticks on both axes (set_yticks can widen
        # the view to fit out-of-range ticks, so re-pin ylim after)
        ax.set_yticks(ax.get_xticks())
        ax.set_ylim(*lim)
    ax.set_box_aspect(1)
    if title:
        ax.set_title(title, fontsize=fontsize)
    return fig, ax


def plot_map_value_vs_occupancy(
    summary_df: pd.DataFrame,
    *,
    noise_df: pd.DataFrame | None = None,
    stat: str = "median",
    ax=None,
    title: str | None = None,
):
    """Scatter of per-cluster map value vs cluster occupancy, B vs A overlaid.

    summary_df is cluster_map_summary.csv: one row per cluster with
    cluster_occupancy and {stat}_map_B / {stat}_map_A columns (B = inverse-
    transform over all legitimate structures; A = naive over the rmsd_before-
    eligible subset). noise_df (noise_map_summary.csv) is drawn as a faint
    low-occupancy series when given. Returns (fig, ax).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.5, 4))
    else:
        fig = ax.figure

    if noise_df is not None and not noise_df.empty and f"{stat}_map_B" in noise_df:
        ax.scatter(noise_df["cluster_occupancy"], noise_df[f"{stat}_map_B"],
                   s=8, alpha=0.25, color="grey", label="noise (B)", zorder=1)

    b_col, a_col = f"{stat}_map_B", f"{stat}_map_A"
    if a_col in summary_df:
        ax.scatter(summary_df["cluster_occupancy"], summary_df[a_col],
                   s=18, alpha=0.6, color="tab:orange", edgecolors="none",
                   label="A: naive (rmsd_before-eligible)", zorder=2)
    if b_col in summary_df:
        ax.scatter(summary_df["cluster_occupancy"], summary_df[b_col],
                   s=18, alpha=0.8, color="tab:blue", edgecolors="k", linewidths=0.3,
                   label="B: inverse-transform (all legit)", zorder=3)

    ax.axhline(0, color="k", lw=0.6, ls=":")
    ax.set_xlabel("cluster occupancy")
    ax.set_ylabel(f"{stat} map value across structures (σ)")
    ax.set_xlim(0, 1.01)
    ax.legend(fontsize=8, framealpha=0.9)
    ax.set_title(title or "Map value at consensus position vs occupancy")
    return fig, ax


def finite(values):
    """`values` as a float array with NaN/inf dropped."""
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def binned_density(values, edges):
    """Raw binned density (area = 1, no kernel smoothing) aligned to bin
    centers, so a violin silhouette is the actual distribution rather than a
    KDE. All-zeros for empty input."""
    values = finite(values)
    if values.size == 0:
        return np.zeros(len(edges) - 1)
    return np.histogram(values, bins=edges, density=True)[0]


def metric_limits(values, clamp):
    """Non-degenerate (lo, hi) display range: the full range, or 0–99.9 pct
    when clamped."""
    values = finite(values)
    if values.size == 0:
        return 0.0, 1.0
    lo, hi = (
        (float(np.quantile(values, 0.0)), float(np.quantile(values, 0.999)))
        if clamp
        else (float(values.min()), float(values.max()))
    )
    return lo, hi if hi > lo else lo + 1.0


def panel_grid(n, panel_w, panel_h, gutters, vertical):
    """`n` panels of a fixed data-rectangle size in one column (`vertical`) or
    one row, so a figure is identical across runs. `gutters` is
    (left, right, top, bottom, gap) in inches. Returns (fig, axes)."""
    left, right, top, bottom, gap = gutters
    if vertical:
        fig_w = left + panel_w + right
        fig_h = bottom + n * panel_h + (n - 1) * gap + top
        spacing = {"hspace": gap / panel_h}
    else:
        fig_w = left + n * panel_w + (n - 1) * gap + right
        fig_h = bottom + panel_h + top
        spacing = {"wspace": gap / panel_w}
    fig, axes = plt.subplots(
        *((n, 1) if vertical else (1, n)),
        figsize=(fig_w, fig_h),
        squeeze=False,
    )
    fig.subplots_adjust(
        left=left / fig_w,
        right=1 - right / fig_w,
        bottom=bottom / fig_h,
        top=1 - top / fig_h,
        **spacing,
    )
    return fig, (axes[:, 0] if vertical else axes[0])


def despine_axis(ax, despine):
    if despine:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


def make_violin_figure(
    df,
    metrics,
    cohorts,
    cohort_labels,
    split,
    metric_labels,
    n_bins,
    clamp,
    center_mode,
    show_iqr,
    despine,
    vertical,
    font_size,
    legend_loc="upper left",
):
    """One panel per metric; within a panel the x-axis is the cohorts and each
    cohort is a split violin, left half = the split's left side.

    Each half is normalized to its own area (the groups differ hugely in size),
    then both are scaled by a single per-cohort factor so the taller peak just
    fills the half-slot — relative peak height between the halves is kept.
    """
    metrics = list(metrics)
    fs = font_size
    halfwidth = 0.4
    fig, axes = panel_grid(
        max(len(metrics), 1),
        panel_w=1.7 * len(cohorts),
        panel_h=2.5,
        # left = y-label + tick digits, bottom = x tick labels; both scale mildly
        # with the font so they hug the labels but don't clip when it is bumped.
        gutters=(0.55 + fs * 0.02, 0.2, 0.3, 0.30 + fs * 0.02, 1.2),
        vertical=vertical,
    )

    def draw_marks(ax, values, x_center, sign, color, density, edges, scale):
        # Short horizontal marks on one half: solid = median, diamond = mean,
        # dotted = Q1/Q3, colored by group. Each spans only the violin's width
        # at its own y (the scaled density of the bin it lands in), so it never
        # overshoots the silhouette. Marks use unclamped values, so one can sit
        # just outside the y-limits.
        values = finite(values)
        if values.size == 0:
            return

        def edge_at(y):
            b = np.searchsorted(edges, y, side="right") - 1
            return x_center + sign * density[int(np.clip(b, 0, len(density) - 1))] * scale

        if center_mode in ("median", "both"):
            m = np.median(values)
            ax.plot(
                [x_center, edge_at(m)], [m, m], color=color, lw=2, solid_capstyle="butt", zorder=4
            )
        if center_mode in ("mean", "both"):
            mu = values.mean()
            # diamond at the midpoint of the violin's width at the mean's height
            ax.plot(
                [(x_center + edge_at(mu)) / 2],
                [mu],
                marker="D",
                ms=6,
                mfc="white",
                mec=color,
                mew=1.5,
                zorder=5,
            )
        if show_iqr:
            for q in np.percentile(values, [25, 75]):
                ax.plot([x_center, edge_at(q)], [q, q], color=color, lw=2.0, ls=":", zorder=4)

    # strict=False: panel_grid allocates max(len(metrics), 1) axes, so an empty
    # metrics list leaves one unused axis.
    for ax, metric in zip(axes, metrics, strict=False):
        lo, hi = metric_limits(df[metric], clamp)
        # Shared bin edges across cohorts within a metric → the violins are
        # directly comparable along this panel's y-axis.
        edges = np.linspace(lo, hi, int(n_bins) + 1)
        centers = (edges[:-1] + edges[1:]) / 2

        for i, cohort in enumerate(cohorts):
            sub = df[df["cohort"] == cohort]
            halves = []
            for side, sign in (("left", -1), ("right", +1)):
                spec = split[side]
                values = sub.loc[sub[split["col"]] == spec["value"], metric]
                halves.append((spec, sign, values, binned_density(values, edges)))
            peak = max(density.max() for *_, density in halves)
            if peak <= 0:
                continue
            scale = halfwidth / peak
            for spec, sign, values, density in halves:
                ax.fill_betweenx(
                    centers,
                    i,
                    i + sign * density * scale,
                    step="mid",
                    color=spec["fill"],
                    alpha=0.5,
                    edgecolor=spec["edge"],
                    linewidth=0.8,
                )
                draw_marks(ax, values, i, sign, spec["mark"], density, edges, scale)

        ax.set_xticks(range(len(cohorts)))
        ax.set_xticklabels(cohort_labels, fontsize=fs, ha="center")
        ax.set_xlim(-0.6, len(cohorts) - 0.4)
        ax.set_ylim(lo, hi)
        ax.set_ylabel(metric_labels.get(metric, metric), fontsize=fs)
        ax.tick_params(axis="y", labelsize=fs)
        despine_axis(ax, despine)

    axes[0].legend(
        handles=[
            Patch(
                facecolor=split[side]["fill"],
                alpha=0.5,
                edgecolor=split[side]["edge"],
                label=split[side]["label"],
            )
            for side in ("left", "right")
        ],
        fontsize=fs - 2,
        loc=legend_loc,
        ncol=2,
        columnspacing=1.0,
        framealpha=0.9,
    )
    return fig
