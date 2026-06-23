"""Q轴描述性统计图 (T1 P×Q×S) — 可复跑脚本.

Produces:
  figure/fig_q_axis_speedup.png    — INT8/FP16 speedup by width, annotated with alignment trap
  figure/fig_q_axis_ap_delta.png   — INT8 AP delta per width (e2e pipeline)
  figure/fig_q_axis_pareto.png     — (latency, AP) Pareto coverage FP16 vs INT8

Data source: results/latency_lut_pyramid_q.json
Run: python multi_agent/figure/make_q_axis_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

REPO = Path(__file__).resolve().parents[2]
Q_LUT = REPO / "results/latency_lut_pyramid_q.json"
OUT_DIR = Path(__file__).parent

# H800 TVM LUT for latency axis (us) — needed for Pareto plot
H800_LUT = REPO / "results/latency_lut_pyramid.json"


def load_q_lut():
    data = json.loads(Q_LUT.read_text())
    widths = data["widths"]
    complete = [w for w in widths if w.get("fp16_p50_ms") and w.get("int8_p50_ms")]
    pending = [w for w in widths if not w.get("int8_p50_ms")]
    return complete, pending, data


def load_h800_lut():
    if not H800_LUT.exists():
        return {}
    data = json.loads(H800_LUT.read_text())
    rows = data.get("widths") or data.get("grid") or []
    return {tuple(r["num_filters"]): r for r in rows}


def fig1_speedup(complete, pending):
    """INT8/FP16 backbone speedup by width, annotated."""
    fig, ax = plt.subplots(figsize=(9, 5))

    labels, speedups, colors = [], [], []
    ALIGNED_COLOR = "#2196F3"   # blue — aligned channels (s0 multiple of 32/64)
    TRAP_COLOR    = "#F44336"   # red  — alignment trap (s0=48 etc.)

    for w in complete:
        nf = w["num_filters"]
        sp = w["fp16_p50_ms"] / w["int8_p50_ms"]
        label = f"{w['label']}\n{nf}"
        labels.append(label)
        speedups.append(sp)
        # color by s0 alignment: s0=48 → trap
        s0 = nf[0]
        colors.append(TRAP_COLOR if s0 % 32 != 0 else ALIGNED_COLOR)

    for w in pending:
        nf = w["num_filters"]
        label = f"{w['label']}\n{nf}"
        labels.append(label + " [pending]")
        speedups.append(1.0)  # placeholder
        s0 = nf[0]
        colors.append("#BDBDBD")  # grey for pending

    x = np.arange(len(labels))
    bars = ax.bar(x, speedups, color=colors, edgecolor="black", linewidth=0.7, alpha=0.85, zorder=3)

    # Add value labels
    for bar, sp in zip(bars, speedups):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{sp:.3f}×", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Reference lines
    ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--", label="FP16 baseline")
    ax.axhline(1.2, color="#2196F3", linewidth=0.8, linestyle=":", alpha=0.7, label="1.2× threshold")

    # Annotation: alignment trap explanation
    trap_indices = [i for i, w in enumerate(complete) if w["num_filters"][0] % 32 != 0]
    for ti in trap_indices:
        ax.annotate("alignment\ntrap (s0=48)",
                    xy=(ti, complete[ti]["fp16_p50_ms"] / complete[ti]["int8_p50_ms"]),
                    xytext=(ti + 0.6, 1.15),
                    fontsize=7, color=TRAP_COLOR,
                    arrowprops=dict(arrowstyle="->", color=TRAP_COLOR, lw=0.8))

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("INT8 / FP16 backbone latency speedup", fontsize=11)
    ax.set_title("Q轴 INT8 Speedup by Width (4090 TRT backbone-only, n=200)\n"
                 "蓝=s0对齐(aligned)  红=对齐陷阱(trap, s0=48)  灰=待测(pending)",
                 fontsize=10, pad=12)
    ax.set_ylim(0.85, 1.8)
    ax.grid(axis="y", alpha=0.4, zorder=0)

    legend_patches = [
        mpatches.Patch(color=ALIGNED_COLOR, label="s0 aligned (multiple of 32/64)"),
        mpatches.Patch(color=TRAP_COLOR, label="s0 non-aligned (alignment trap)"),
        mpatches.Patch(color="#BDBDBD", label="pending hw-optimizer build"),
    ]
    ax.legend(handles=legend_patches, fontsize=8, loc="upper right")

    ax.text(0.01, 0.02,
            "WARNING: FP16 speedup values are 4090 TRT backbone-only.\n"
            "H800 TVM latency uses separate hardware — cross-hardware ratio is estimated.",
            transform=ax.transAxes, fontsize=6.5, color="#757575", va="bottom")

    fig.tight_layout()
    out = OUT_DIR / "fig_q_axis_speedup.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[fig1] {out}")


def fig2_ap_delta(complete):
    """INT8 AP70 delta per width (e2e pipeline)."""
    fig, ax = plt.subplots(figsize=(8, 4))

    has_ap = [w for w in complete if "ap70_delta_int8_vs_fp16_e2e" in w and w["ap70_delta_int8_vs_fp16_e2e"] is not None]
    if not has_ap:
        print("[fig2] No AP delta data yet — skipping")
        plt.close(fig)
        return

    labels = [f"{w['label']}\n{w['num_filters']}" for w in has_ap]
    deltas = [w["ap70_delta_int8_vs_fp16_e2e"] for w in has_ap]
    fp16_aps = [w.get("ap70_fp16_e2e", 0) for w in has_ap]

    x = np.arange(len(labels))
    ax.bar(x, deltas, color="#FF7043", edgecolor="black", linewidth=0.7, alpha=0.85)
    for i, (d, fp16) in enumerate(zip(deltas, fp16_aps)):
        ax.text(i, d - 0.001, f"{d:+.4f}", ha="center", va="top", fontsize=7.5, color="white", fontweight="bold")
        ax.text(i, 0.003, f"FP16={fp16:.3f}", ha="center", va="bottom", fontsize=6.5, color="#333")

    # Median line
    med = np.median(deltas)
    ax.axhline(med, color="#1565C0", linewidth=1.5, linestyle="--",
               label=f"median={med:+.4f} AP70")
    ax.axhline(0, color="black", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("INT8 AP70 - FP16 AP70  (Δ, e2e pipeline, minmax calib.)", fontsize=10)
    ax.set_title("Q轴 INT8 AP70 代价 (e2e DAIR val 1618/1789, minmax calibrator)\n"
                 "注: e2e abs值(~0.37) vs stage_a abs值(~0.63) 口径不同; 此图仅看Δ", fontsize=9)
    ax.legend(fontsize=9)
    ax.set_ylim(min(deltas) - 0.005, 0.015)
    ax.grid(axis="y", alpha=0.4)

    ax.text(0.01, 0.01,
            "Pipeline: e2e TRT DAIR val (1618/1789); NOT stage_a full eval.\n"
            "AP delta extrapolated to stage_a FP16 AP (~0.631) for search model.",
            transform=ax.transAxes, fontsize=6.5, color="#757575")

    fig.tight_layout()
    out = OUT_DIR / "fig_q_axis_ap_delta.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[fig2] {out}")


def fig3_pareto_coverage(complete, pending):
    """(H800_TVM_latency, AP70) Pareto coverage: FP16 vs INT8 (estimated)."""
    h800 = load_h800_lut()
    if not h800:
        print("[fig3] H800 LUT not found — skipping")
        return

    fig, ax = plt.subplots(figsize=(9, 6))

    fp16_pts, int8_pts = [], []
    for w in complete:
        nf = tuple(w["num_filters"])
        if nf not in h800:
            continue
        ap_fp16 = w.get("ap70_fp16_stagea", 0)
        if not ap_fp16:
            continue
        ap_int8 = ap_fp16 - 0.007
        # H800 TVM latency (tuned) in us
        lat_fp16_us = float(h800[nf].get("tuned_us", 0))
        if not lat_fp16_us:
            continue
        q_ratio = w["fp16_p50_ms"] / w["int8_p50_ms"]
        lat_int8_us = lat_fp16_us / q_ratio

        fp16_pts.append((lat_fp16_us, ap_fp16, w["label"], nf))
        int8_pts.append((lat_int8_us, ap_int8, w["label"], nf))

    for p in pending:
        nf = tuple(p["num_filters"])
        if nf not in h800:
            continue
        ap_fp16 = p.get("ap70_fp16_stagea", 0)
        if not ap_fp16:
            continue
        lat_fp16_us = float(h800[nf].get("tuned_us", 0))
        if not lat_fp16_us:
            continue
        fp16_pts.append((lat_fp16_us, ap_fp16, p["label"] + " [pending]", nf))

    # Plot
    if fp16_pts:
        xs, ys, labs, _ = zip(*fp16_pts)
        ax.scatter(xs, ys, c="#2196F3", s=60, marker="o", label="FP16 (H800 TVM tuned, measured)", zorder=4, alpha=0.85)
        for x, y, lab in zip(xs, ys, labs):
            ax.annotate(lab, (x, y), textcoords="offset points",
                        xytext=(4, 3), fontsize=7, color="#2196F3")

    if int8_pts:
        xs, ys, labs, _ = zip(*int8_pts)
        ax.scatter(xs, ys, c="#F44336", s=60, marker="s",
                   label="INT8 (H800 TVM × 4090 Q-ratio, estimated)", zorder=4, alpha=0.85)
        for x, y, lab in zip(xs, ys, labs):
            ax.annotate(lab, (x, y), textcoords="offset points",
                        xytext=(4, -10), fontsize=7, color="#F44336")

    # Draw connecting lines FP16→INT8
    fp16_by_w = {nf: (x, y) for x, y, _, nf in fp16_pts if "[pending]" not in _}
    int8_by_w = {nf: (x, y) for x, y, _, nf in int8_pts}
    for nf in int8_by_w:
        if nf in fp16_by_w:
            x1, y1 = fp16_by_w[nf]
            x2, y2 = int8_by_w[nf]
            ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle="->", color="#757575", lw=0.8, alpha=0.5))

    ax.set_xlabel("H800 TVM Backbone Latency (tuned schedule, μs)", fontsize=11)
    ax.set_ylabel("AP70 (stage_a DAIR val 1789, collaborative)", fontsize=11)
    ax.set_title("Q轴 Pareto覆盖: FP16 vs INT8 (P_g/W_g对视角)\n"
                 "INT8延迟 = H800_TVM / 4090_Q_ratio (estimated, cross-hardware)", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)

    ax.text(0.01, 0.01,
            "CROSS-HARDWARE WARNING: INT8 latency scaled by 4090 Q-ratio.\n"
            "H800 INT8 actual speedup may differ. Use ratio for rank-flip analysis only.",
            transform=ax.transAxes, fontsize=6.5, color="#757575")

    fig.tight_layout()
    out = OUT_DIR / "fig_q_axis_pareto.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[fig3] {out}")


def main():
    if not Q_LUT.exists():
        print(f"ERROR: {Q_LUT} not found. Run T1 build pipeline first.")
        return 1

    complete, pending, data = load_q_lut()
    print(f"Q-LUT: {len(complete)} complete FP16+INT8, {len(pending)} pending")

    fig1_speedup(complete, pending)
    fig2_ap_delta(complete)
    fig3_pareto_coverage(complete, pending)
    print("\n[done] Q-axis figures written to multi_agent/figure/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
