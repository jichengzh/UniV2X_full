#!/usr/bin/env python3
"""Render the Gold176 cost-model selection summary figure."""

from __future__ import annotations

from collections import Counter
import csv
import math
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt


FIGURE_DIR = Path(__file__).resolve().parent
SOURCE_PATH = FIGURE_DIR / "cost_model_selection_source.csv"
OUTPUT_STEM = FIGURE_DIR / "cost_model_selection"
SUMMARY_PATH = FIGURE_DIR / "cost_model_selection_summary.csv"

TARGET_ORDER = ("latency_ms", "energy_j", "ap70")
TARGET_TITLES = {
    "latency_ms": "(a) Latency",
    "energy_j": "(b) Energy",
    "ap70": r"(c) AP$_{70}$",
}
FAMILY_COLORS = {
    "extra_trees": "#4C78A8",
    "lightgbm": "#E08B3E",
}
OBJECTIVE_NAMES = {
    "squared_error": "SE",
    "mae": "MAE",
    "huber": "Huber",
    "quantile_p50": "Q50",
}
ENCODING_NAMES = {
    "raw": "raw",
    "log1p": "log1p",
    "model_anchor_residual": "residual",
}
REQUIRED_COLUMNS = {
    "target",
    "candidate",
    "family",
    "objective",
    "encoding",
    "oof_mae",
    "oof_spearman",
    "inner_score",
    "selected_folds",
    "retained",
}
EXPECTED_CANDIDATES = {
    "latency_ms": frozenset(
        {
            "extra_trees_raw",
            "extra_trees_log",
            "lgbm_l1_raw",
            "lgbm_l1_log",
            "lgbm_huber_raw",
            "lgbm_huber_log",
            "lgbm_quantile_raw",
            "lgbm_quantile_log",
        }
    ),
    "energy_j": frozenset(
        {
            "extra_trees_raw",
            "extra_trees_log",
            "lgbm_l1_raw",
            "lgbm_l1_log",
            "lgbm_huber_raw",
            "lgbm_huber_log",
            "lgbm_quantile_raw",
            "lgbm_quantile_log",
        }
    ),
    "ap70": frozenset(
        {
            "extra_trees_raw",
            "extra_trees_residual",
            "lgbm_l1_raw",
            "lgbm_l1_residual",
            "lgbm_huber_raw",
            "lgbm_huber_residual",
            "lgbm_quantile_raw",
            "lgbm_quantile_residual",
        }
    ),
}
CANDIDATE_SCHEMA = {
    "extra_trees_raw": ("extra_trees", "squared_error", "raw"),
    "extra_trees_log": ("extra_trees", "squared_error", "log1p"),
    "extra_trees_residual": (
        "extra_trees",
        "squared_error",
        "model_anchor_residual",
    ),
    "lgbm_l1_raw": ("lightgbm", "mae", "raw"),
    "lgbm_l1_log": ("lightgbm", "mae", "log1p"),
    "lgbm_l1_residual": ("lightgbm", "mae", "model_anchor_residual"),
    "lgbm_huber_raw": ("lightgbm", "huber", "raw"),
    "lgbm_huber_log": ("lightgbm", "huber", "log1p"),
    "lgbm_huber_residual": (
        "lightgbm",
        "huber",
        "model_anchor_residual",
    ),
    "lgbm_quantile_raw": ("lightgbm", "quantile_p50", "raw"),
    "lgbm_quantile_log": ("lightgbm", "quantile_p50", "log1p"),
    "lgbm_quantile_residual": (
        "lightgbm",
        "quantile_p50",
        "model_anchor_residual",
    ),
}


def _configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def load_rows(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if set(reader.fieldnames or ()) != REQUIRED_COLUMNS:
            raise ValueError(
                f"source columns must equal {sorted(REQUIRED_COLUMNS)}, "
                f"got {sorted(reader.fieldnames or ())}"
            )
        return [
            {
                **row,
                "oof_mae": float(row["oof_mae"]),
                "oof_spearman": float(row["oof_spearman"]),
                "inner_score": float(row["inner_score"]),
                "selected_folds": int(row["selected_folds"]),
                "retained": bool(int(row["retained"])),
            }
            for row in reader
        ]


def validate_rows(rows: list[dict[str, object]]) -> None:
    identities = [(str(row["target"]), str(row["candidate"])) for row in rows]
    if len(rows) != 24 or len(set(identities)) != 24:
        raise ValueError("source must contain 24 unique target-candidate rows")

    target_counts = Counter(str(row["target"]) for row in rows)
    if target_counts != Counter({target: 8 for target in TARGET_ORDER}):
        raise ValueError(f"each target must contain eight candidates: {target_counts}")

    for target in TARGET_ORDER:
        actual_candidates = frozenset(
            str(row["candidate"]) for row in rows if row["target"] == target
        )
        if actual_candidates != EXPECTED_CANDIDATES[target]:
            raise ValueError(f"{target} has an invalid candidate schema")

    for row in rows:
        candidate = str(row["candidate"])
        actual_schema = (
            str(row["family"]),
            str(row["objective"]),
            str(row["encoding"]),
        )
        if actual_schema != CANDIDATE_SCHEMA.get(candidate):
            raise ValueError(f"invalid candidate schema for {candidate}: {actual_schema}")
        numeric = (
            float(row["oof_mae"]),
            float(row["oof_spearman"]),
            float(row["inner_score"]),
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError(f"non-finite metric in {row['candidate']}")
        selected_folds = int(row["selected_folds"])
        if not 0 <= selected_folds <= 5:
            raise ValueError(f"selected_folds must be within [0, 5]: {row}")

    for target in TARGET_ORDER:
        target_rows = [row for row in rows if row["target"] == target]
        retained = [row for row in target_rows if row["retained"]]
        if len(retained) != 1:
            raise ValueError(f"{target} must contain exactly one retained head")
        winner = retained[0]
        if float(winner["inner_score"]) != min(
            float(row["inner_score"]) for row in target_rows
        ):
            raise ValueError(f"{target} retained head must minimize inner_score")
        if int(winner["selected_folds"]) != max(
            int(row["selected_folds"]) for row in target_rows
        ):
            raise ValueError(f"{target} retained head must maximize selected_folds")


def retained_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    validate_rows(rows)
    retained_by_target = {
        str(row["target"]): dict(row) for row in rows if bool(row["retained"])
    }
    return [retained_by_target[target] for target in TARGET_ORDER]


def _candidate_label(row: dict[str, object]) -> str:
    family = "ET" if row["family"] == "extra_trees" else "LGBM"
    objective = OBJECTIVE_NAMES[str(row["objective"])]
    encoding = ENCODING_NAMES[str(row["encoding"])]
    return f"{family} · {objective} · {encoding}"


def _point_size(selected_folds: int, retained: bool) -> float:
    base = 28.0 + 18.0 * selected_folds
    return max(base, 120.0) if retained else base


def build_figure(rows: list[dict[str, object]]):
    validate_rows(rows)
    _configure_matplotlib()
    width_inches = 183.0 / 25.4
    height_inches = 72.0 / 25.4
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(width_inches, height_inches),
        constrained_layout=True,
    )
    figure.get_layout_engine().set(rect=(0.0, 0.075, 1.0, 1.0))

    for axis, target in zip(axes, TARGET_ORDER):
        target_rows = sorted(
            (dict(row) for row in rows if row["target"] == target),
            key=lambda row: float(row["inner_score"]),
        )
        scores = [float(row["inner_score"]) for row in target_rows]
        y_positions = list(range(len(target_rows)))
        score_span = max(scores) - min(scores)
        margin = max(score_span * 0.12, 0.004)
        left = min(scores) - margin

        for y_position, row in zip(y_positions, target_rows):
            score = float(row["inner_score"])
            selected_folds = int(row["selected_folds"])
            is_retained = bool(row["retained"])
            color = FAMILY_COLORS[str(row["family"])]

            axis.hlines(
                y_position,
                left,
                score,
                color="#D8D8D8",
                linewidth=0.8,
                zorder=1,
            )
            axis.scatter(
                score,
                y_position,
                s=_point_size(selected_folds, is_retained),
                marker="*" if is_retained else "o",
                facecolor=color,
                edgecolor="#8C2D2D" if is_retained else "white",
                linewidth=1.2 if is_retained else 0.6,
                zorder=3,
            )
            if selected_folds:
                axis.annotate(
                    f"{selected_folds}/5",
                    xy=(score, y_position),
                    xytext=(4, 0),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                    fontsize=6,
                    fontweight="bold" if is_retained else "normal",
                    color="#7A1F1F" if is_retained else "#4A4A4A",
                )

        axis.set_yticks(y_positions)
        axis.set_yticklabels([_candidate_label(row) for row in target_rows])
        axis.invert_yaxis()
        axis.set_xlim(left, max(scores) + 2.5 * margin)
        axis.set_title(TARGET_TITLES[target], loc="left", fontweight="bold")
        axis.set_xlabel("Mean inner-validation score ↓")
        axis.grid(axis="x", color="#E8E8E8", linewidth=0.6)
        axis.tick_params(axis="y", length=0, pad=2)
        axis.tick_params(axis="x", length=2.5, width=0.6)

        for label, row in zip(axis.get_yticklabels(), target_rows):
            if row["retained"]:
                label.set_fontweight("bold")
                label.set_color("#7A1F1F")

    figure.text(
        0.5,
        0.012,
        (
            "Points show five-fold means (no repeated-seed uncertainty); "
            "scores are normalized within each target and are not comparable across panels."
        ),
        ha="center",
        va="bottom",
        fontsize=5.5,
        color="#555555",
    )

    return figure


def write_summary(rows: list[dict[str, object]], path: Path) -> None:
    summary = retained_rows(rows)
    fieldnames = [
        "target",
        "candidate",
        "family",
        "objective",
        "encoding",
        "oof_mae",
        "oof_spearman",
        "inner_score",
        "selected_folds",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {name: row[name] for name in fieldnames}
            for row in summary
        )


def build_bar_figure(rows: list[dict[str, object]]):
    """Render categorical model-selection scores as horizontal bars."""
    validate_rows(rows)
    _configure_matplotlib()
    width_inches = 181.0 / 25.4
    height_inches = 108.0 / 25.4
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(width_inches, height_inches),
        constrained_layout=True,
    )
    figure.get_layout_engine().set(rect=(0.0, 0.08, 1.0, 0.88))

    for axis, target in zip(axes, TARGET_ORDER):
        target_rows = sorted(
            (dict(row) for row in rows if row["target"] == target),
            key=lambda row: float(row["inner_score"]),
            reverse=True,
        )
        scores = [float(row["inner_score"]) for row in target_rows]
        y_positions = list(range(len(target_rows)))
        colors = [FAMILY_COLORS[str(row["family"])] for row in target_rows]
        score_min = min(scores)
        score_max = max(scores)
        score_span = max(score_max - score_min, 1e-6)
        axis_min = max(0.0, score_min - 0.18 * score_span)
        edge_colors = [
            "#8C2D2D" if bool(row["retained"]) else "white"
            for row in target_rows
        ]
        line_widths = [1.4 if bool(row["retained"]) else 0.5 for row in target_rows]
        axis.barh(
            y_positions,
            [score - axis_min for score in scores],
            left=axis_min,
            color=colors,
            edgecolor=edge_colors,
            linewidth=line_widths,
            height=0.68,
            zorder=3,
        )

        axis.set_yticks(y_positions)
        axis.set_yticklabels([_candidate_label(row) for row in target_rows])
        axis.set_title(TARGET_TITLES[target], loc="left", fontweight="bold")
        axis.set_xlim(axis_min, score_max + 0.18 * score_span)
        axis.set_xticks([])
        axis.grid(axis="x", color="#E8E8E8", linewidth=0.6, zorder=0)
        axis.tick_params(axis="y", length=0, pad=2)
        axis.tick_params(axis="x", length=2.5, width=0.6)

        for position, row, score in zip(y_positions, target_rows, scores):
            if bool(row["retained"]):
                axis.get_yticklabels()[position].set_fontweight("bold")
                axis.get_yticklabels()[position].set_color("#7A1F1F")
            axis.text(
                score + 0.025 * score_span,
                position,
                f"{score:.2f}",
                va="center",
                fontsize=9.5,
                fontweight="bold" if bool(row["retained"]) else "normal",
                color="#7A1F1F" if bool(row["retained"]) else "#4A4A4A",
            )

    legend_handles = [
        mpl.patches.Patch(color=FAMILY_COLORS["extra_trees"], label="ExtraTrees"),
        mpl.patches.Patch(color=FAMILY_COLORS["lightgbm"], label="LightGBM"),
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=9.5,
        bbox_to_anchor=(0.5, 1.02),
    )
    figure.supxlabel("Mean inner-validation score ↓", fontsize=9.5, y=0.01)
    return figure


def export_figure(figure: Any, output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(
        output_stem.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    figure.savefig(
        output_stem.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )


def main() -> int:
    rows = load_rows(SOURCE_PATH)
    validate_rows(rows)
    write_summary(rows, SUMMARY_PATH)
    figure = build_bar_figure(rows)
    export_figure(figure, OUTPUT_STEM)
    plt.close(figure)

    retained = ", ".join(
        f"{row['target']}={row['candidate']} "
        f"({row['inner_score']:.4f}, {row['selected_folds']}/5)"
        for row in retained_rows(rows)
    )
    print(f"Saved figure bundle to {FIGURE_DIR}")
    print(f"Retained heads: {retained}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
