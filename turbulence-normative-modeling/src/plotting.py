"""
Figures.

Two main figures are produced:

  Phase 1  For each TD feature, the trajectory across spatial scales split by age
           group, sex, and head-motion tertile, each paired with a bar panel
           showing the standardised per-scale slope or group difference.

  Phase 2  ROC curves comparing the classification of depressive symptoms from
           deviation scores against raw features, and the feature trajectories of
           the depressed and control groups before and after normative correction.
"""

from __future__ import annotations

from typing import Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import config
from .phase1_lmm import (
    FEATURES,
    Feature,
    scale_wise_group_difference,
    scale_wise_slopes,
    to_long_format,
)
from .prediction import CVResult, roc_points

AGE_BIN_EDGES = [44, 55, 65, 75, np.inf]
AGE_BIN_LABELS = ["45-55", "56-65", "66-75", "76+"]

MOTION_BIN_EDGES = [0, 0.1, 0.2, np.inf]
MOTION_BIN_LABELS = ["Low (<0.1)", "Medium (0.1-0.2)", "High (>0.2)"]

GROUP_COLOURS = {"Control": "seagreen", "Depressive symptoms": "orangered"}


def _add_covariate_bins(long: pd.DataFrame) -> pd.DataFrame:
    """Bin age and head motion for the group trajectory panels."""
    long = long.copy()
    long["age_group"] = pd.cut(
        long[config.COL_AGE], bins=AGE_BIN_EDGES, labels=AGE_BIN_LABELS
    )
    long["motion_group"] = pd.cut(
        long[config.COL_MOTION], bins=MOTION_BIN_EDGES, labels=MOTION_BIN_LABELS
    )
    return long


def _format_scale_axis(ax, scales: Sequence[float], show_labels: bool) -> None:
    """Label the x-axis with lambda values, only on the bottom row."""
    ax.set_xticks(range(len(scales)))
    if show_labels:
        ax.set_xticklabels([f"{s:.3f}" for s in scales], rotation=45, fontsize=10)
        ax.set_xlabel(r"Spatial scale $\lambda$", fontsize=14, fontweight="bold")
    else:
        ax.set_xticklabels([])
        ax.set_xlabel("")


def plot_phase1(
    df: pd.DataFrame,
    features: Sequence[Feature] = FEATURES,
    output_stem: str = "phase1_scale_sensitivity",
):
    """
    Build the Phase 1 figure: three features by three covariates.

    Rows alternate between the trajectory across scales and the standardised
    per-scale effect, one pair of rows per feature. Line plots share a y-range
    within a feature and bar plots share one within a covariate, so panels are
    directly comparable.
    """
    sns.set_style("whitegrid")

    n_rows = 2 * len(features)
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=3,
        figsize=(16, 6.5 * len(features)),
        gridspec_kw={"height_ratios": [1.5, 1] * len(features)},
    )

    bar_axes_by_column: Dict[int, list] = {0: [], 1: [], 2: []}
    scales: list = []

    for feature_index, feature in enumerate(features):
        line_row, bar_row = feature_index * 2, feature_index * 2 + 1

        long = _add_covariate_bins(to_long_format(df, feature))
        scales = sorted(long["scale"].unique())
        positions = list(range(len(scales)))

        # Plot against the scale's position rather than its value so that the
        # panels line up with the categorical bar plots beneath them.
        long["position"] = long["scale"].map({s: i for i, s in enumerate(scales)})

        panels = [
            (0, "age_group", "viridis", "Age", scale_wise_slopes(long, config.COL_AGE),
             "slope", "Slope (SD / year)"),
            (1, config.COL_SEX, "vlag", "Sex",
             scale_wise_group_difference(long, config.COL_SEX),
             "difference", "Difference (SD)"),
            (2, "motion_group", "magma", "Head motion",
             scale_wise_slopes(long, config.COL_MOTION),
             "slope", "Slope (SD / mm)"),
        ]

        for column, hue, palette, title, effects, effect_column, effect_label in panels:
            line_ax = axes[line_row, column]
            sns.lineplot(
                data=long,
                x="position",
                y="value",
                hue=hue,
                palette=palette,
                marker="o",
                errorbar=("ci", 95),
                ax=line_ax,
                zorder=3,
            )
            if feature_index == 0:
                line_ax.set_title(title, fontweight="bold", fontsize=18)
            line_ax.grid(True, linestyle="--", alpha=0.6, zorder=0)
            line_ax.set_ylabel("")
            line_ax.legend(fontsize=10, title=None)

            bar_ax = axes[bar_row, column]
            sns.barplot(
                x=positions,
                y=effects[effect_column].to_numpy(),
                hue=positions,
                palette=palette,
                legend=False,
                ax=bar_ax,
            )
            bar_ax.set_title(effect_label, fontsize=12)
            bar_ax.axhline(0, color="black", lw=1.5)
            bar_ax.set_ylabel("")
            bar_axes_by_column[column].append(bar_ax)

        # Name the feature once, along the left edge of its pair of rows.
        axes[line_row, 0].annotate(
            feature.label,
            xy=(-0.13, -0.15),
            xycoords="axes fraction",
            fontsize=16,
            fontweight="bold",
            rotation=90,
            ha="center",
            va="center",
            annotation_clip=False,
        )

        line_axes = [axes[line_row, c] for c in range(3)]
        low = min(ax.get_ylim()[0] for ax in line_axes)
        high = max(ax.get_ylim()[1] for ax in line_axes)
        for ax in line_axes:
            ax.set_ylim(low, high)

    for column, column_axes in bar_axes_by_column.items():
        low = min(ax.get_ylim()[0] for ax in column_axes)
        high = max(ax.get_ylim()[1] for ax in column_axes)
        for ax in column_axes:
            ax.set_ylim(low, high)
            ax.grid(axis="y", linestyle="--", alpha=0.5)

    for row in range(n_rows):
        for column in range(3):
            _format_scale_axis(axes[row, column], scales, show_labels=row == n_rows - 1)

    plt.subplots_adjust(
        top=0.97, bottom=0.08, left=0.07, right=0.97, hspace=0.15, wspace=0.15
    )

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(
            config.FIGURES_DIR / f"{output_stem}.{extension}",
            dpi=300,
            bbox_inches="tight",
        )
    return fig


def plot_roc_comparison(
    scenarios: Dict[str, tuple],
    output_stem: str = "phase2_roc_comparison",
):
    """
    Plot ROC curves for the deviation and raw feature sets, one panel per scenario.

    `scenarios` maps a panel title to a (deviation_result, raw_result) pair, which
    lets the age/sex-matched and age/sex/motion-matched analyses sit side by side.
    """
    n = len(scenarios)
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 7), squeeze=False)

    for ax, (title, (deviation, raw)) in zip(axes[0], scenarios.items()):
        for result, colour, style, label in (
            (deviation, "tab:blue", "-", "Deviation scores"),
            (raw, "tab:orange", "--", "Uncorrected features"),
        ):
            fpr, tpr = roc_points(result)
            mean_auc = np.mean(result.auc)
            ax.plot(
                fpr, tpr, color=colour, linestyle=style, linewidth=2,
                label=f"{label} (AUC = {mean_auc:.3f})",
            )

        ax.plot([0, 1], [0, 1], "k:", alpha=0.5)
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("False positive rate (1 - specificity)")
        ax.set_ylabel("True positive rate (sensitivity)")
        ax.legend(loc="lower right", fontsize=10)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)

    plt.tight_layout()

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(
            config.FIGURES_DIR / f"{output_stem}.{extension}",
            dpi=300,
            bbox_inches="tight",
        )
    return fig


def plot_group_trajectories(
    df: pd.DataFrame,
    group_column: str,
    features: Sequence[Feature] = FEATURES,
    output_stem: str = "phase2_group_trajectories",
):
    """
    Compare the two groups across spatial scales, before and after correction.

    The top row shows the raw features and the bottom row the normative deviation
    scores, so any group separation that only emerges after covariate correction
    is visible.
    """
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, len(features), figsize=(8 * len(features), 12),
                             constrained_layout=True)

    for column, feature in enumerate(features):
        raw_long = to_long_format(
            df, feature, id_columns=[config.COL_SUBJECT, config.COL_AGE,
                                     config.COL_SEX, config.COL_MOTION, group_column]
        )
        scales = sorted(raw_long["scale"].unique())
        raw_long["position"] = raw_long["scale"].map(
            {s: i for i, s in enumerate(scales)}
        )

        deviation_columns = sorted(
            [
                c for c in df.columns
                if c.startswith(f"{feature.name}_") and c.endswith("_deviation")
            ]
        )

        for row, (data, title) in enumerate(
            [
                (raw_long, f"{feature.label}\nraw feature"),
                (None, f"{feature.label}\nnormative deviation"),
            ]
        ):
            ax = axes[row, column]

            if row == 1:
                if not deviation_columns:
                    ax.set_visible(False)
                    continue
                data = pd.melt(
                    df,
                    id_vars=[config.COL_SUBJECT, group_column],
                    value_vars=deviation_columns,
                    var_name="column",
                    value_name="value",
                ).dropna(subset=["value"])
                order = {c: i for i, c in enumerate(deviation_columns)}
                data["position"] = data["column"].map(order)

            sns.lineplot(
                data=data,
                x="position",
                y="value",
                hue=group_column,
                palette=GROUP_COLOURS,
                marker="o",
                errorbar=("ci", 95),
                ax=ax,
                legend=(column == len(features) - 1),
            )
            ax.set_title(title, fontsize=14, fontweight="bold")
            ax.set_ylabel("Mean ± 95% CI" if column == 0 else "")
            ax.grid(True, linestyle="--", alpha=0.5)
            _format_scale_axis(ax, scales, show_labels=row == 1)

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(
            config.FIGURES_DIR / f"{output_stem}.{extension}",
            dpi=300,
            bbox_inches="tight",
        )
    return fig
