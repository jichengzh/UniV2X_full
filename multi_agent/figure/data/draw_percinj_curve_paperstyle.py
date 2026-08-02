#!/usr/bin/env python3
"""Draw a compact paper-style latency-to-driving-degradation figure."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "figure/data/percinj_curve_full.csv"
OUT_DIR = ROOT / "figure/data"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "sans-serif"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7.5,
        "legend.fontsize": 6.3,
        "xtick.labelsize": 6.2,
        "ytick.labelsize": 6.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.unicode_minus": False,
    }
)


COLORS = {
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


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    df["completed_collision_rate_pct"] = df["completed_collision_rate"] * 100.0
    return df.sort_values("tau_perc_ms").reset_index(drop=True)


def style_axes(ax: plt.Axes) -> None:
    ax.grid(True, color=COLORS["grid"], linewidth=0.45, alpha=0.85)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#9AA4B2")
        ax.spines[spine].set_linewidth(0.6)
    ax.tick_params(colors=COLORS["ink"], width=0.55, length=3)


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


def draw(df: pd.DataFrame) -> None:
    threshold_ms = 500
    regions = [
        (0, threshold_ms, COLORS["safe"], "stable"),
        (threshold_ms, 800, COLORS["warning"], "transition"),
        (800, 1000, COLORS["danger"], "failure-prone"),
    ]

    fig, (ax_ds, ax_rate) = plt.subplots(
        2,
        1,
        figsize=(3.35, 2.55),
        sharex=True,
        gridspec_kw={"height_ratios": [1.45, 1.0], "hspace": 0.11},
    )

    for ax in [ax_ds, ax_rate]:
        for start, end, color, _ in regions:
            ax.axvspan(start, end, color=color, alpha=0.72, linewidth=0, zorder=0)
        ax.axvline(
            threshold_ms,
            color=COLORS["threshold"],
            linewidth=0.9,
            linestyle="--",
            zorder=4,
        )
        style_axes(ax)

    ax_ds.errorbar(
        df["tau_perc_ms"],
        df["honest_meanDS"],
        yerr=df["ci95"],
        fmt="o-",
        color=COLORS["ds"],
        linewidth=1.25,
        markersize=2.8,
        capsize=2.2,
        capthick=0.75,
        elinewidth=0.75,
        label="Honest DS",
        zorder=3,
    )
    ax_ds.set_ylabel("Honest DS")
    ax_ds.set_ylim(38, 103)
    ax_ds.text(
        threshold_ms + 10,
        41.0,
        "threshold\n500 ms",
        color=COLORS["threshold"],
        fontsize=5.9,
        ha="left",
        va="bottom",
    )
    ax_ds.text(
        940,
        96,
        "Honest DS",
        color=COLORS["ink"],
        fontsize=6.2,
        ha="right",
        va="center",
    )

    ax_rate.plot(
        df["tau_perc_ms"],
        df["completed_collision_rate_pct"],
        "s-",
        color=COLORS["collision"],
        linewidth=1.15,
        markersize=2.6,
        label="Collision rate",
        zorder=3,
    )
    ax_rate.plot(
        df["tau_perc_ms"],
        df["catastrophe_rate_pct"],
        "^--",
        color=COLORS["catastrophe"],
        linewidth=1.05,
        markersize=2.7,
        label="Timeout rate",
        zorder=3,
    )
    ax_rate.set_ylabel("Rate (%)")
    ax_rate.set_xlabel(r"Perception latency $\tau_{\mathrm{perc}}$ (ms)")
    ax_rate.set_ylim(0, 64)
    ax_rate.set_xlim(-20, 1020)
    ax_rate.set_xticks([0, 200, 400, 500, 600, 800, 1000])
    ax_rate.legend(loc="upper left", frameon=False, handlelength=1.5, ncol=1)

    for x, label in [(250, "stable"), (650, "warning"), (900, "danger")]:
        ax_rate.text(
            x,
            2.0,
            label,
            ha="center",
            va="bottom",
            fontsize=5.8,
            color=COLORS["muted"],
        )

    fig.subplots_adjust(left=0.13, right=0.985, bottom=0.16, top=0.985)
    save_all(fig, "percinj_curve_final_paperstyle_500ms")
    plt.close(fig)


def main() -> None:
    draw(load_data())


if __name__ == "__main__":
    main()
