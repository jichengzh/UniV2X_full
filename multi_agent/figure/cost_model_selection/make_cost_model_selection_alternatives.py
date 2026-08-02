#!/usr/bin/env python3
"""Render scatter and heatmap alternatives for cost-model selection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

from make_cost_model_selection_figure import (
    FAMILY_COLORS,
    SOURCE_PATH,
    TARGET_ORDER,
    TARGET_TITLES,
    _candidate_label,
    _configure_matplotlib,
    _point_size,
    export_figure,
    load_rows,
    validate_rows,
)


FIGURE_DIR = Path(__file__).resolve().parent
SCATTER_STEM = FIGURE_DIR / "cost_model_selection_scatter"
HEATMAP_STEM = FIGURE_DIR / "cost_model_selection_heatmap"

QUALITY_CMAP = LinearSegmentedColormap.from_list(
    "within_target_quality",
    ["#F4F4F4", "#DCEAF3", "#75A5C7", "#173F5F"],
)


def normalized_mae_by_identity(
    rows: list[dict[str, object]],
) -> dict[tuple[str, str], float]:
    """Return min-max normalized OOF MAE independently for each target."""
    validate_rows(rows)
    normalized: dict[tuple[str, str], float] = {}
    for target in TARGET_ORDER:
        target_rows = [row for row in rows if row["target"] == target]
        values = [float(row["oof_mae"]) for row in target_rows]
        low, high = min(values), max(values)
        span = high - low
        if span <= 0:
            raise ValueError(f"{target} OOF MAE must have non-zero range")
        normalized.update(
            {
                (target, str(row["candidate"])): (
                    float(row["oof_mae"]) - low
                )
                / span
                for row in target_rows
            }
        )
    return normalized


def _short_label(row: dict[str, object]) -> str:
    return (
        _candidate_label(row)
        .replace(" · ", "–")
        .replace("residual", "res.")
        .replace("log1p", "log")
    )


def annotation_offset(
    x_value: float,
    y_value: float,
    x_minimum: float,
    x_maximum: float,
    index: int,
) -> tuple[int, int]:
    """Place scatter labels toward the plot interior with vertical staggering."""
    x_offset = 5 if x_value <= (x_minimum + x_maximum) / 2.0 else -5
    if y_value >= 0.92:
        y_offset = -7
    elif y_value <= 0.08:
        y_offset = 6
    else:
        y_offset = 5 if index % 2 == 0 else -8
    return x_offset, y_offset


def build_scatter_figure(rows: list[dict[str, object]]):
    """Plot rank fidelity against normalized error for each target."""
    validate_rows(rows)
    _configure_matplotlib()
    normalized_mae = normalized_mae_by_identity(rows)
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(183.0 / 25.4, 76.0 / 25.4),
        constrained_layout=True,
    )
    figure.get_layout_engine().set(rect=(0.0, 0.16, 1.0, 1.0))

    for axis, target in zip(axes, TARGET_ORDER):
        target_rows = sorted(
            (dict(row) for row in rows if row["target"] == target),
            key=lambda row: float(row["oof_mae"]),
        )
        spearman_values = [float(row["oof_spearman"]) for row in target_rows]
        x_minimum, x_maximum = min(spearman_values), max(spearman_values)
        for index, row in enumerate(target_rows):
            identity = (target, str(row["candidate"]))
            x_value = float(row["oof_spearman"])
            y_value = normalized_mae[identity]
            retained = bool(row["retained"])
            selected_folds = int(row["selected_folds"])
            color = FAMILY_COLORS[str(row["family"])]

            axis.scatter(
                x_value,
                y_value,
                s=_point_size(selected_folds, retained),
                marker="*" if retained else "o",
                facecolor=color,
                edgecolor="#8C2D2D" if retained else "white",
                linewidth=1.2 if retained else 0.6,
                zorder=3,
            )
            if selected_folds:
                x_offset, y_offset = annotation_offset(
                    x_value,
                    y_value,
                    x_minimum,
                    x_maximum,
                    index,
                )
                axis.annotate(
                    f"{_short_label(row)} · {selected_folds}/5",
                    xy=(x_value, y_value),
                    xytext=(x_offset, y_offset),
                    textcoords="offset points",
                    ha="left" if x_offset > 0 else "right",
                    va="bottom" if y_offset > 0 else "top",
                    fontsize=4.8,
                    fontweight="bold" if retained else "normal",
                    color="#7A1F1F" if retained else "#4A4A4A",
                    clip_on=False,
                )

        axis.set_title(TARGET_TITLES[target], loc="left", fontweight="bold")
        axis.set_xlabel("OOF Spearman ↑")
        axis.set_ylabel("Within-target normalized OOF MAE ↓")
        x_margin = max((x_maximum - x_minimum) * 0.05, 0.001)
        axis.set_xlim(x_minimum - x_margin, x_maximum + x_margin)
        axis.set_ylim(-0.08, 1.08)
        axis.grid(color="#E8E8E8", linewidth=0.6)
        axis.tick_params(length=2.5, width=0.6)

    figure.text(
        0.5,
        0.058,
        (
            "Blue = ExtraTrees; orange = LightGBM; stars = retained heads; "
            "point area indicates selected outer folds."
        ),
        ha="center",
        va="bottom",
        fontsize=5.5,
        color="#4A4A4A",
    )
    figure.text(
        0.5,
        0.033,
        (
            "Inner score = MAE/(P90–P10) + 0.25 × (1 − Spearman), evaluated by "
            "5 outer × 3 inner grouped CV; lower is better."
        ),
        ha="center",
        va="bottom",
        fontsize=5.2,
        color="#555555",
    )
    figure.text(
        0.5,
        0.008,
        (
            "OOF metrics are fixed-candidate summaries; within-target normalized MAE "
            "is not comparable across panels. Retained heads were selected by the "
            "inner joint score, not by these OOF axes."
        ),
        ha="center",
        va="bottom",
        fontsize=5.3,
        color="#666666",
    )
    return figure


def _desirability(values: list[float], higher_is_better: bool) -> list[float]:
    low, high = min(values), max(values)
    span = high - low
    if span <= 0:
        return [0.5 for _ in values]
    scaled = [(value - low) / span for value in values]
    return scaled if higher_is_better else [1.0 - value for value in scaled]


def heatmap_matrix(
    target_rows: list[dict[str, object]],
) -> tuple[list[list[float]], list[list[str]]]:
    mae = [float(row["oof_mae"]) for row in target_rows]
    spearman = [float(row["oof_spearman"]) for row in target_rows]
    score = [float(row["inner_score"]) for row in target_rows]
    folds = [int(row["selected_folds"]) for row in target_rows]
    matrix_columns = (
        _desirability(mae, higher_is_better=False),
        _desirability(spearman, higher_is_better=True),
        _desirability(score, higher_is_better=False),
        [value / 5.0 for value in folds],
    )
    matrix = [
        [column[row_index] for column in matrix_columns]
        for row_index in range(len(target_rows))
    ]
    labels = [
        [
            f"{mae[index]:.4f}",
            f"{spearman[index]:.3f}",
            f"{score[index]:.3f}",
            f"{folds[index]}/5",
        ]
        for index in range(len(target_rows))
    ]
    return matrix, labels


def build_heatmap_figure(rows: list[dict[str, object]]):
    """Plot a compact within-target comparison matrix for all candidates."""
    validate_rows(rows)
    _configure_matplotlib()
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(183.0 / 25.4, 90.0 / 25.4),
        constrained_layout=True,
    )
    figure.get_layout_engine().set(rect=(0.0, 0.105, 1.0, 1.0))

    for axis, target in zip(axes, TARGET_ORDER):
        target_rows = sorted(
            (dict(row) for row in rows if row["target"] == target),
            key=lambda row: float(row["inner_score"]),
        )
        matrix, cell_labels = heatmap_matrix(target_rows)
        axis.imshow(matrix, cmap=QUALITY_CMAP, vmin=0.0, vmax=1.0, aspect="auto")

        for row_index, row_values in enumerate(matrix):
            for column_index, desirability in enumerate(row_values):
                axis.text(
                    column_index,
                    row_index,
                    cell_labels[row_index][column_index],
                    ha="center",
                    va="center",
                    fontsize=5.2,
                    color="white" if desirability >= 0.62 else "#222222",
                    fontweight=(
                        "bold" if bool(target_rows[row_index]["retained"]) else "normal"
                    ),
                )

        retained_index = next(
            index for index, row in enumerate(target_rows) if bool(row["retained"])
        )
        axis.add_patch(
            Rectangle(
                (-0.49, retained_index - 0.49),
                3.98,
                0.98,
                fill=False,
                edgecolor="#8C2D2D",
                linewidth=1.4,
            )
        )
        axis.set_xticks(range(4))
        axis.set_xticklabels(
            ["OOF MAE ↓", "Spearman ↑", "Inner score ↓", "Selected"],
            rotation=28,
            ha="right",
            rotation_mode="anchor",
        )
        axis.set_yticks(range(len(target_rows)))
        axis.set_yticklabels([_candidate_label(row) for row in target_rows])
        axis.set_title(TARGET_TITLES[target], loc="left", fontweight="bold")
        axis.tick_params(axis="both", length=0, pad=2)
        for spine in axis.spines.values():
            spine.set_visible(False)
        for label, row in zip(axis.get_yticklabels(), target_rows):
            if bool(row["retained"]):
                label.set_fontweight("bold")
                label.set_color("#7A1F1F")

    figure.text(
        0.5,
        0.035,
        (
            "Inner score = MAE/(P90–P10) + 0.25 × (1 − Spearman), evaluated by "
            "5 outer × 3 inner grouped CV; lower is better."
        ),
        ha="center",
        va="bottom",
        fontsize=5.2,
        color="#555555",
    )
    figure.text(
        0.5,
        0.010,
        (
            "Cell text reports raw values; darker cells indicate better within-target "
            "relative performance. Red outlines mark retained heads."
        ),
        ha="center",
        va="bottom",
        fontsize=5.4,
        color="#555555",
    )
    return figure


def _export_and_close(figure: Any, output_stem: Path) -> None:
    export_figure(figure, output_stem)
    plt.close(figure)


def main() -> int:
    rows = load_rows(SOURCE_PATH)
    validate_rows(rows)
    _export_and_close(build_scatter_figure(rows), SCATTER_STEM)
    _export_and_close(build_heatmap_figure(rows), HEATMAP_STEM)
    print(f"Saved scatter and heatmap bundles to {FIGURE_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
