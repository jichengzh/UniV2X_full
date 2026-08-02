#!/usr/bin/env python3
"""Draw a two-panel background evidence figure for the paper."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


PAPER_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent
BASE_WIDTH_SUM = 64 + 128 + 256

LATENCY_DATA = (
    PROJECT_ROOT
    / "data/stage2_lut_generation_v1/generated/original60_quant_20260627/"
    / "exports/original60_quant_three_metric_summary_latest.csv"
)
DS_DATA = PROJECT_ROOT / "figure/data/percinj_curve_full.csv"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "sans-serif"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7.5,
        "legend.fontsize": 6.2,
        "xtick.labelsize": 6.2,
        "ytick.labelsize": 6.2,
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
    "ds": "#2B6CB0",
    "collision": "#B42318",
    "catastrophe": "#7E22CE",
    "grid": "#D7DCE2",
    "ink": "#1F2933",
    "muted": "#5E6B78",
    "safe": "#DCEEDB",
    "warning": "#FFF2A6",
    "danger": "#F4C8C8",
    "threshold": "#B42318",
}

PRECISION_LABELS = {"fp32": "FP32", "fp16": "FP16", "int8": "INT8"}


def style_axes(ax: plt.Axes) -> None:
    ax.grid(True, color=COLORS["grid"], linewidth=0.45, alpha=0.85)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#9AA4B2")
        ax.spines[spine].set_linewidth(0.6)
    ax.tick_params(colors=COLORS["ink"], width=0.55, length=3)


def load_latency_data() -> pd.DataFrame:
    df = pd.read_csv(LATENCY_DATA)
    widths = df["width"].str.extract(r"(?P<s0>\d+)x(?P<s1>\d+)x(?P<s2>\d+)").astype(int)
    df = df.join(widths)
    df["channel_sum"] = df[["s0", "s1", "s2"]].sum(axis=1)
    df["eq_prune_rate"] = 1.0 - df["channel_sum"] / BASE_WIDTH_SUM
    df["precision"] = df["precision"].str.lower()
    return df.sort_values(["precision", "eq_prune_rate", "width"]).reset_index(drop=True)


def load_ds_data() -> pd.DataFrame:
    df = pd.read_csv(DS_DATA)
    df["completed_collision_rate_pct"] = df["completed_collision_rate"] * 100.0
    return df.sort_values("tau_perc_ms").reset_index(drop=True)


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


def draw_panel_a(ax_ds: plt.Axes, ax_rate: plt.Axes, df: pd.DataFrame) -> None:
    threshold_ms = 500
    regions = [
        (0, threshold_ms, COLORS["safe"]),
        (threshold_ms, 800, COLORS["warning"]),
        (800, 1000, COLORS["danger"]),
    ]

    for start, end, color in regions:
        ax_ds.axvspan(start, end, color=color, alpha=0.72, linewidth=0, zorder=0)
    ax_ds.axvline(
        threshold_ms,
        color=COLORS["threshold"],
        linewidth=0.9,
        linestyle="--",
        zorder=4,
    )

    ds_line = ax_ds.errorbar(
        df["tau_perc_ms"],
        df["honest_meanDS"],
        yerr=df["ci95"],
        fmt="o-",
        color=COLORS["ds"],
        linewidth=1.15,
        markersize=2.45,
        capsize=2.0,
        capthick=0.7,
        elinewidth=0.7,
        label="Honest DS",
        zorder=3,
    )
    collision_line = ax_rate.plot(
        df["tau_perc_ms"],
        df["completed_collision_rate_pct"],
        "s-",
        color=COLORS["collision"],
        linewidth=1.05,
        markersize=2.35,
        label="Collision",
        zorder=3,
    )[0]
    timeout_line = ax_rate.plot(
        df["tau_perc_ms"],
        df["catastrophe_rate_pct"],
        "^--",
        color=COLORS["catastrophe"],
        linewidth=0.95,
        markersize=2.45,
        label="Timeout",
        zorder=3,
    )[0]

    ax_ds.set_xlabel(r"Perception latency $\tau_{\mathrm{perc}}$ (ms)")
    ax_ds.set_ylabel("Honest DS", color=COLORS["ds"])
    ax_rate.set_ylabel("Rate (%)", color=COLORS["collision"])
    ax_ds.set_xlim(-20, 1020)
    ax_ds.set_ylim(38, 103)
    ax_rate.set_ylim(0, 64)
    ax_ds.set_xticks([0, 200, 400, 500, 600, 800, 1000])
    ax_ds.tick_params(axis="y", colors=COLORS["ds"])
    ax_rate.tick_params(axis="y", colors=COLORS["collision"])
    ax_rate.spines["right"].set_color(COLORS["collision"])
    ax_rate.spines["right"].set_linewidth(0.6)
    ax_rate.spines["top"].set_visible(False)
    ax_rate.grid(False)

    handles = [ds_line.lines[0], collision_line, timeout_line]
    ax_ds.legend(
        handles=handles,
        labels=["Honest DS", "Collision", "Timeout"],
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.53, 1.04),
        handlelength=1.3,
        columnspacing=0.8,
    )
    ax_ds.text(-0.12, 1.04, "(a)", transform=ax_ds.transAxes, fontsize=8.5, fontweight="bold")
    style_axes(ax_ds)


def draw_panel_b(ax: plt.Axes, df: pd.DataFrame) -> None:
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
            s=8,
            color=COLORS[precision],
            alpha=0.28,
            edgecolors="none",
        )
        ax.plot(
            summary["eq_prune_rate"] * 100,
            summary["latency_ms"],
            color=COLORS[precision],
            linewidth=1.25,
            marker="o",
            markersize=2.2,
            label=PRECISION_LABELS[precision],
        )

    ax.set_xlabel("Equivalent channel pruning rate (%)")
    ax.set_ylabel("Measured latency (ms)")
    ax.set_xlim(0, 76)
    ax.set_ylim(0, 58)
    ax.legend(frameon=False, ncol=3, loc="upper right", handlelength=1.4, columnspacing=0.7)
    ax.text(-0.12, 1.04, "(b)", transform=ax.transAxes, fontsize=8.5, fontweight="bold")
    style_axes(ax)


def main() -> None:
    latency_df = load_latency_data()
    ds_df = load_ds_data()

    fig = plt.figure(figsize=(7.05, 2.15))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.18, 1.0], wspace=0.36)

    ax_ds = fig.add_subplot(outer[0, 0])
    ax_rate = ax_ds.twinx()
    ax_latency = fig.add_subplot(outer[0, 1])

    draw_panel_a(ax_ds, ax_rate, ds_df)
    draw_panel_b(ax_latency, latency_df)

    fig.subplots_adjust(left=0.068, right=0.94, bottom=0.21, top=0.93)
    save_all(fig, "fig_background_joint_evidence")
    plt.close(fig)


if __name__ == "__main__":
    main()
