#!/usr/bin/env python3
"""Draw background evidence figures from the 180-row latency summary table."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
DATA = (
    ROOT
    / "data/stage2_lut_generation_v1/generated/original60_quant_20260627/"
    / "exports/original60_quant_three_metric_summary_latest.csv"
)
OUT_DIR = Path(__file__).resolve().parent
BASE_WIDTH = np.array([64, 128, 256], dtype=float)


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "sans-serif"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7.5,
        "legend.fontsize": 6.4,
        "xtick.labelsize": 6.4,
        "ytick.labelsize": 6.4,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.unicode_minus": False,
    }
)

COLORS = {
    "fp32": "#4A5568",
    "fp16": "#2B6CB0",
    "int8": "#D97706",
    "grid": "#D7DCE2",
    "ink": "#1F2933",
    "muted": "#5E6B78",
    "accent": "#B42318",
}

PRECISION_LABELS = {"fp32": "FP32", "fp16": "FP16", "int8": "INT8"}


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    widths = df["width"].str.extract(r"(?P<s0>\d+)x(?P<s1>\d+)x(?P<s2>\d+)").astype(int)
    df = df.join(widths)
    df["channel_sum"] = df[["s0", "s1", "s2"]].sum(axis=1)
    df["eq_prune_rate"] = 1.0 - df["channel_sum"] / BASE_WIDTH.sum()
    df["precision"] = df["precision"].str.lower()
    return df.sort_values(["precision", "eq_prune_rate", "width"]).reset_index(drop=True)


def save_all(fig: plt.Figure, stem: str) -> None:
    for suffix, kwargs in {
        ".pdf": {},
        ".svg": {},
        ".png": {"dpi": 400},
        ".tiff": {"dpi": 600},
    }.items():
        fig.savefig(
            OUT_DIR / f"{stem}{suffix}",
            bbox_inches="tight",
            facecolor="white",
            **kwargs,
        )


def style_axes(ax: plt.Axes) -> None:
    ax.grid(True, color=COLORS["grid"], linewidth=0.45, alpha=0.85)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#9AA4B2")
        ax.spines[spine].set_linewidth(0.6)
    ax.tick_params(colors=COLORS["ink"], width=0.55, length=3)


def draw_latency_curves(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(3.2, 1.9))
    for precision in ["fp32", "fp16", "int8"]:
        sub = df[df["precision"] == precision]
        summary = (
            sub.groupby("eq_prune_rate", as_index=False)
            .agg(latency_ms=("latency_ms", "median"))
            .sort_values("eq_prune_rate")
        )
        ax.scatter(
            sub["eq_prune_rate"] * 100,
            sub["latency_ms"],
            s=10,
            color=COLORS[precision],
            alpha=0.32,
            edgecolors="none",
        )
        ax.plot(
            summary["eq_prune_rate"] * 100,
            summary["latency_ms"],
            color=COLORS[precision],
            linewidth=1.35,
            marker="o",
            markersize=2.4,
            label=PRECISION_LABELS[precision],
        )

    ax.set_xlabel("Equivalent channel pruning rate (%)")
    ax.set_ylabel("Measured latency (ms)")
    ax.set_xlim(0, 76)
    ax.set_ylim(0, 58)
    ax.legend(frameon=False, ncol=3, loc="upper right", handlelength=1.5, columnspacing=0.9)
    style_axes(ax)
    fig.tight_layout(pad=0.25)
    save_all(fig, "fig_background_latency_prune_curves")
    plt.close(fig)


def main() -> None:
    df = load_data()
    draw_latency_curves(df)


if __name__ == "__main__":
    main()
