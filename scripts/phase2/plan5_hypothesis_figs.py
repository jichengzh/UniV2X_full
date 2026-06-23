"""Plan v5 — Generate per-hypothesis evidence figures.

Three figures, one per hypothesis (H1 / H2 / H3 / D1):
  1. h1_lat_smoothness.png — per (Q, D) cell lat vs plane + poly3 fit + R²
  2. h2_sparsity_reduction.png — bar chart of sparsity-induced lat reduction
  3. h3_ap_plateau.png — AP30/50/70 vs plane (PyTorch FP32 baseline)
  4. d1_pareto_dominate.png — g8 lat/AP combined (note: g32 not benched in plan v5)
"""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("/home/jichengzhi/UniV2X")
DATA_DIR = REPO / "paper_learning" / "2. AAAI最终故事" / "data"
FIG_DIR = DATA_DIR / "stats_v3_plan5"
FIG_DIR.mkdir(parents=True, exist_ok=True)

PLANE_MAP = {"p64_baseline": 64, "p64": 64, "p48": 48, "p32": 32, "p16": 16, "p8": 8}


def fig_h1():
    df = pd.read_csv(DATA_DIR / "plan5_phaseA_anchors.csv")
    df["plane"] = df["tag"].map(PLANE_MAP)
    df = df.dropna(subset=["plane", "lat_p50"]).copy()
    df["plane"] = df["plane"].astype(int)

    qs = sorted(df["q"].unique())
    ds = sorted(df["d"].unique())
    fig, axes = plt.subplots(len(ds), len(qs), figsize=(4 * len(qs), 3 * len(ds)), squeeze=False)

    cell_r2_list = []
    for di, d in enumerate(ds):
        for qi, q in enumerate(qs):
            ax = axes[di][qi]
            sub = df[(df["d"] == d) & (df["q"] == q)].sort_values("plane")
            x = sub["plane"].to_numpy(dtype=float)
            y = sub["lat_p50"].to_numpy(dtype=float)
            if len(x) >= 4:
                coefs = np.polyfit(x, y, 3)
                xx = np.linspace(x.min(), x.max(), 80)
                yy = np.polyval(coefs, xx)
                pred = np.polyval(coefs, x)
                ss_res = float(np.sum((y - pred) ** 2))
                ss_tot = float(np.sum((y - y.mean()) ** 2))
                r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0
                cell_r2_list.append(r2)
                ax.plot(xx, yy, "--", color="gray", alpha=0.7, label=f"poly3 fit")
            else:
                r2 = float("nan")
            ax.scatter(x, y, s=60, color="C0", zorder=5)
            for px, py in zip(x, y):
                ax.annotate(f"p{int(px)}", (px, py), textcoords="offset points",
                            xytext=(5, 5), fontsize=8)
            ax.set_xlabel("plane")
            ax.set_ylabel("lat_p50 [ms]")
            ax.set_title(f"Q={q}  D={d}\nR²(poly3) = {r2:.3f}")
            ax.grid(True, alpha=0.3)

    mean_r2 = float(np.mean(cell_r2_list)) if cell_r2_list else float("nan")
    fig.suptitle(
        f"H1 — lat vs plane smoothness on g8 architecture\n"
        f"mean R² across {len(cell_r2_list)} cells = {mean_r2:.3f}  "
        f"(threshold 0.85 → {'PASS' if mean_r2 >= 0.85 else 'FAIL'})",
        fontsize=12, y=1.00)
    fig.tight_layout()
    out = FIG_DIR / "h1_lat_smoothness.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return mean_r2


def fig_h2():
    sparse = pd.read_csv(DATA_DIR / "plan5_phaseB_anchors.csv")
    dense = pd.read_csv(DATA_DIR / "plan5_phaseA_anchors.csv")
    dense = dense[dense["d"] == "D1_default"]
    sparse["plane"] = sparse["tag"].map(PLANE_MAP)
    dense["plane"] = dense["tag"].map(PLANE_MAP)
    merged = sparse.merge(
        dense[["plane", "q", "lat_p50"]].rename(columns={"lat_p50": "lat_dense"}),
        on=["plane", "q"], how="left")
    merged["reduction_pct"] = (1 - merged["lat_p50"] / merged["lat_dense"]) * 100
    merged = merged.dropna(subset=["reduction_pct"])
    merged = merged.sort_values(["q", "plane"], ascending=[True, False])

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    qs = sorted(merged["q"].unique())
    width = 0.35
    planes_sorted = [64, 48, 32, 16, 8]
    x_pos = np.arange(len(planes_sorted))
    for qi, q in enumerate(qs):
        sub = merged[merged["q"] == q].set_index("plane").reindex(planes_sorted)
        vals = sub["reduction_pct"].to_numpy()
        bars = ax.bar(x_pos + qi * width, vals, width,
                       label=f"Q={q}", color=f"C{qi}")
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.5,
                        f"{v:.1f}%", ha="center", fontsize=8)
    ax.axhline(30, color="red", linestyle="--", label="threshold 30%")
    ax.set_xticks(x_pos + width / 2)
    ax.set_xticklabels([f"p{p}" for p in planes_sorted])
    ax.set_xlabel("plane")
    ax.set_ylabel("lat reduction by 2:4 sparsity [%]")
    n_pass_fp16 = int((merged[merged["q"] == "fp16"]["reduction_pct"] >= 30).sum())
    n_pass_int8 = int((merged[merged["q"] == "int8_mm"]["reduction_pct"] >= 30).sum())
    ax.set_title(
        f"H2 — 2:4 sparsity lat reduction vs dense baseline\n"
        f"fp16 pass={n_pass_fp16}/5, int8_mm pass={n_pass_int8}/5 (threshold ≥3/5 plane @ ≥30%) → FAIL"
    )
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    out = FIG_DIR / "h2_sparsity_reduction.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return n_pass_fp16, n_pass_int8


def fig_h3():
    df = pd.read_csv(DATA_DIR / "plan5_phaseC_real_ap.csv")
    df = df[df["q"] == "fp32"].copy()
    df["plane"] = df["tag"].map(PLANE_MAP)
    df = df.sort_values("plane")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    for metric, color in [("ap30", "C0"), ("ap50", "C1"), ("ap70", "C2")]:
        ax.plot(df["plane"], df[metric], "o-", color=color, label=metric.upper(),
                markersize=10, linewidth=2)
        for x, y in zip(df["plane"], df[metric]):
            ax.annotate(f"{y:.4f}", (x, y), textcoords="offset points",
                         xytext=(5, 8), fontsize=8)
    ax.invert_xaxis()
    ax.set_xlabel("plane (← prune more aggressive)")
    ax.set_ylabel("AP")
    ax.set_title("PyTorch FP32 AP vs plane (DAIR-V2X val n=1789)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks([64, 48, 32, 16, 8])

    ax2 = axes[1]
    base_ap50 = float(df[df["plane"] == 64]["ap50"].iloc[0])
    df["ap50_delta_pct"] = (df["ap50"] / base_ap50 - 1) * 100
    bars = ax2.bar([f"p{int(p)}" for p in df["plane"]], df["ap50_delta_pct"], color="C1")
    for b, v in zip(bars, df["ap50_delta_pct"]):
        ax2.text(b.get_x() + b.get_width() / 2,
                  v - 0.05 if v < 0 else v + 0.02,
                  f"{v:+.2f}%", ha="center", fontsize=9)
    ax2.axhline(-2.0, color="red", linestyle="--", label="threshold |gap|≤2%")
    ax2.set_ylabel("AP50 delta vs p64 baseline [%]")
    ax2.set_xlabel("plane")
    ax2.set_title("H3 (partial) — AP50 plateau across plane reduction\n"
                  "max |delta| = -1.1% << 2%; suggests INT8 gap likely small "
                  "but real TRT INT8 AP not measured")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    out = FIG_DIR / "h3_ap_plateau.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return float(df["ap50_delta_pct"].abs().max())


def fig_d1():
    df_a = pd.read_csv(DATA_DIR / "plan5_phaseA_anchors.csv")
    df_a = df_a[df_a["d"] == "D1_default"]
    df_a["plane"] = df_a["tag"].map(PLANE_MAP)
    df_c = pd.read_csv(DATA_DIR / "plan5_phaseC_real_ap.csv")
    df_c = df_c[df_c["q"] == "fp32"]
    df_c["plane"] = df_c["tag"].map(PLANE_MAP)
    ap_lookup = dict(zip(df_c["plane"], df_c["ap50"]))
    df_a["ap50"] = df_a["plane"].map(ap_lookup)

    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    qs = sorted(df_a["q"].unique())
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(qs)))
    for qi, q in enumerate(qs):
        sub = df_a[df_a["q"] == q].sort_values("plane")
        ax.scatter(sub["lat_p50"], sub["ap50"],
                   s=100, c=[cmap[qi]], label=f"Q={q}", marker="o", edgecolor="k", linewidth=0.5)
        for _, r in sub.iterrows():
            ax.annotate(f"p{int(r['plane'])}",
                        (r["lat_p50"], r["ap50"]),
                        textcoords="offset points", xytext=(5, 5), fontsize=8)
    base_lat = float(df_a[(df_a["q"] == "fp16") & (df_a["plane"] == 64)]["lat_p50"].iloc[0])
    base_ap = ap_lookup.get(64, np.nan)
    if not np.isnan(base_ap):
        ax.axvline(0.7 * base_lat, color="red", linestyle="--", alpha=0.6,
                   label=f"D1 lat threshold 0.7× baseline = {0.7*base_lat:.2f} ms")
        ax.axhline(0.92 * base_ap, color="orange", linestyle="--", alpha=0.6,
                   label=f"D1 AP threshold 0.92× baseline = {0.92*base_ap:.4f}")
        ax.scatter([base_lat], [base_ap], s=200, marker="*",
                   color="black", zorder=10, label="baseline (p64+fp16)")
    ax.set_xlabel("lat_p50 [ms] (RTX 4090, TRT engine)")
    ax.set_ylabel("AP50 (DAIR-V2X val n=1789, PyTorch FP32)")
    ax.set_title("D1 (derived) — g8 Pareto frontier vs baseline\n"
                 "(g32 reference data NOT benched in plan v5; D1 vs-g32 verdict undecided)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "d1_pareto_dominate.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main():
    r2 = fig_h1()
    pf16, pi8 = fig_h2()
    max_delta = fig_h3()
    fig_d1()
    print(f"\nH1 mean R² = {r2:.3f} (PASS if >= 0.85)")
    print(f"H2 pass planes: fp16={pf16}/5, int8_mm={pi8}/5 (FAIL: max=0)")
    print(f"H3 max |AP50 delta| = {max_delta:.2f}% (partial — no INT8 gap)")


if __name__ == "__main__":
    main()
