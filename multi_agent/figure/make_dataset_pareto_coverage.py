#!/usr/bin/env python3
"""Dataset stocktake + main-line Pareto/coupling/cross-hardware figure (dataset_v2, 61 rows).

Four panels mandated by HANDOFF_rebuild_figures.md (all numbers from real result files):
  A. 2D cost Pareto (latency, energy) — collab2/4090 front pool; color=precision,
     marker=prune; AP annotated as constraint; p25 INT8 coupling-trap highlighted.
  B. Coupling trap — INT8 speedup per width tier: aligned base/p50/p75/cliff2_c
     1.24-1.57x vs p25(48,96,192) 1.06x; P0-2 [32,64,136] 1.34x = non-aligned-but-
     NO-cliff counter-evidence -> scope limited to grouped-conv(3x3 g32).
  C. Cross-hardware — 4090 vs Orin FP16 complete points (latency + energy; energy
     domains differ: 4090 NVML board-rail vs Orin module-total VIN_SYS, labeled).
  D. Coverage/stocktake — 5-metric coverage by dataset_src + effective-anchor count.

Output: multi_agent/figure/dataset_pareto_coverage.png
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = "/home/jichengzhi/V2X"
os.chdir(REPO)

# ---------- load (all real-measured) ----------
main = pd.read_parquet("multi_agent/data/dataset_v2.parquet")
qcliff = pd.read_csv("results/Q_int8_cliff_4090.csv")  # clean per-width fp16/int8 pairs


def prec_label(row):
    ps = {row["stage0_prec"], row["stage1_prec"], row["stage2_prec"]}
    return "int8" if "int8" in {str(p).lower() for p in ps} else (
        "fp16" if "fp16" in {str(p).lower() for p in ps} else "fp32")


main["prec"] = main.apply(prec_label, axis=1)
cmap = {"fp32": "#888", "fp16": "#1f77b4", "int8": "#d62728"}
mark = {0.0: "o", 0.25: "^", 0.5: "s", 0.75: "D"}

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("UniV2X co-opt dataset_v2 — main-line Pareto / coupling trap / cross-hardware  (n=61, all real-measured)",
             fontsize=14, fontweight="bold")

# ============ Panel A: 2D cost Pareto (latency, energy) ============
ax = axes[0, 0]
# front pool, 4090 collab2 double-agent body, with both lat & energy
fa = main[(main["latency_kind"] == "body_subnet_collab2") & (main["hardware"] == "rtx4090")
          & (main["regime"] == "front") & main["lat_p50_ms"].notna()
          & main["energy_per_frame_mj"].notna()].copy()
for pr, mk in mark.items():
    for prec, c in cmap.items():
        s = fa[(fa["prune_rate"].fillna(-1) == pr) & (fa["prec"] == prec)]
        if len(s):
            ax.scatter(s["lat_p50_ms"], s["energy_per_frame_mj"], c=c, marker=mk, s=90,
                       edgecolor="k", linewidth=0.5, alpha=0.9, zorder=3)
# 2D Pareto front (min latency, min energy)
pts = list(zip(fa["lat_p50_ms"], fa["energy_per_frame_mj"]))
nd = [a for a in pts if not any((b[0] <= a[0] and b[1] <= a[1]) and (b[0] < a[0] or b[1] < a[1])
                                for b in pts if b is not a)]
nd = sorted(nd)
ax.plot([p[0] for p in nd], [p[1] for p in nd], "k--", lw=1.2, zorder=2,
        label="2D cost Pareto front (min lat, min energy)")
# highlight p25 INT8 coupling-trap point
trap = fa[(fa["prune_rate"] == 0.25) & (fa["prec"] == "int8")]
for _, t in trap.iterrows():
    ax.scatter(t["lat_p50_ms"], t["energy_per_frame_mj"], s=360, facecolors="none",
               edgecolors="#ff7f0e", linewidth=2.6, zorder=4)
    ax.annotate("p25 INT8 coupling-trap\n(only 1.06x; grouped-conv\nstage0=48 kernel cliff)",
                (t["lat_p50_ms"], t["energy_per_frame_mj"]), textcoords="offset points",
                xytext=(14, 6), fontsize=8, color="#cc5500",
                bbox=dict(boxstyle="round", fc="#fff6ec", ec="#ff7f0e", alpha=0.95))
# annotate AP70 (constraint axis) near each point
for _, r in fa.iterrows():
    if pd.notna(r["ap70"]):
        ax.annotate(f"{r['ap70']:.2f}", (r["lat_p50_ms"], r["energy_per_frame_mj"]),
                    textcoords="offset points", xytext=(3, -10), fontsize=6.5, color="#444")
ax.set_xlabel("latency p50 (ms)  [body_subnet_collab2, 4090]")
ax.set_ylabel("energy per frame (mJ)  [E5 collab2 NVML board]")
ax.set_title(f"A. Main-line 2D cost Pareto (latency, energy), front pool n={len(fa)}")
ax.grid(alpha=0.3)
leg1 = [Line2D([0], [0], marker="o", color="w", markerfacecolor=cmap[k], markeredgecolor="k",
               markersize=9, label=k) for k in ("fp16", "int8")]
leg2 = [Line2D([0], [0], marker=mark[k], color="w", markerfacecolor="#aaa", markeredgecolor="k",
               markersize=9, label=f"prune {int(k*100)}%") for k in mark]
ax.legend(handles=leg1 + leg2 + [Line2D([0], [0], color="k", ls="--", label="2D Pareto front")],
          fontsize=7.5, ncol=2, loc="upper right")
ax.text(0.02, 0.03, "AP70 (labels) spans 0.49–0.64 across whole front → AP near-flat\nconstraint (Pyramid/DAIR over-parameterized). Cost frontier = (lat, energy).",
        transform=ax.transAxes, fontsize=8, color="#a00",
        bbox=dict(boxstyle="round", fc="#fff3f3", ec="#a00", alpha=0.9))

# ============ Panel B: coupling trap — INT8 speedup per width tier ============
ax = axes[0, 1]
# clean fp16/int8 pairs from Q_int8_cliff (4dp AP, same pipeline)
qc = qcliff[qcliff["precision"].isin(["trt_fp16", "trt_int8"])].copy()
tiers = []  # (label, planes_str, speedup, aligned?, note)
TIER_ORDER = [("base", "64,128,256", True), ("pruned50", "32,64,128", True),
              ("pruned75", "16,32,64", True), ("cliff2_c", "16,32,64", True),
              ("pruned25", "48,96,192", False)]
for var, planes, aligned in TIER_ORDER:
    g = qc[qc["variant"] == var]
    f16 = g[g["precision"] == "trt_fp16"]["lat_p50_ms"]
    i8 = g[g["precision"] == "trt_int8"]["lat_p50_ms"]
    if len(f16) and len(i8):
        tiers.append((var, planes, float(f16.values[0]) / float(i8.values[0]), aligned))
# add P0-2 [32,64,136] from dataset (non-aligned stage2=136, but NO cliff)
p136 = main[main["dataset_src"] == "P0_2_136"]
if len(p136) == 2:
    sf = p136[p136["prec"] == "fp16"]["lat_p50_ms"].values[0]
    si = p136[p136["prec"] == "int8"]["lat_p50_ms"].values[0]
    tiers.append(("P0-2 [32,64,136]", "32,64,136", sf / si, "nonaligned_nocliff"))
labels = [f"{t[0]}\n({t[1]})" for t in tiers]
spd = [t[2] for t in tiers]
colors = []
for t in tiers:
    if t[3] is True:
        colors.append("#1f77b4")          # aligned -> healthy INT8 speedup
    elif t[3] == "nonaligned_nocliff":
        colors.append("#2ca02c")          # 136: non-aligned but NO cliff (1×1 conv)
    else:
        colors.append("#ff7f0e")          # p25: grouped-conv non-aligned -> CLIFF
bars = ax.bar(range(len(tiers)), spd, color=colors, edgecolor="k", alpha=0.9)
for i, v in enumerate(spd):
    ax.text(i, v + 0.02, f"{v:.2f}x", ha="center", fontsize=9, fontweight="bold")
ax.axhline(1.0, color="#999", ls=":", lw=1)
ax.set_xticks(range(len(tiers)))
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("INT8 speedup (fp16_lat / int8_lat), collab2")
ax.set_title("B. Coupling trap: aligned 1.24–1.57x vs p25 1.06x (grouped-conv cliff)")
ax.set_ylim(0, max(spd) * 1.25)
ax.grid(alpha=0.3, axis="y")
ax.legend(handles=[
    Line2D([0], [0], marker="s", color="w", markerfacecolor="#1f77b4", markersize=10,
           label="aligned width (32-mult) → healthy 1.24–1.57x"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor="#2ca02c", markersize=10,
           label="[32,64,136] non-aligned but 1×1-conv → 1.34x, NO cliff"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor="#ff7f0e", markersize=10,
           label="p25 (48,96,192) grouped-conv non-aligned → 1.06x CLIFF"),
], fontsize=7.3, loc="upper right")
ax.text(0.02, 0.95, "Trap scope (precise): kernel-selection cliff is limited to\ngrouped-conv (3x3, g32) at non-tile-aligned width (stage0=48).\n1×1 conv at non-aligned 136 does NOT trigger it (1.34x).",
        transform=ax.transAxes, fontsize=8, va="top", color="#070",
        bbox=dict(boxstyle="round", fc="#f3fff3", ec="#070", alpha=0.92))

# ============ Panel C: cross-hardware 4090 vs Orin FP16 ============
ax = axes[1, 0]
CFG = [("base", (64, 128, 256)), ("p50", (32, 64, 128)), ("p75", (16, 32, 64))]
orin = main[main["dataset_src"] == "E6_orin_p03"]
c2 = main[(main["latency_kind"] == "body_subnet_collab2") & (main["hardware"] == "rtx4090")]
lat_4090, lat_orin, en_4090, en_orin, xt = [], [], [], [], []
for name, pl in CFG:
    o = orin[(orin["config_label"] == f"orin_{name}_fp16")]
    g = c2[(c2["stage0_planes"] == pl[0]) & (c2["stage1_planes"] == pl[1])
           & (c2["stage2_planes"] == pl[2]) & (c2["prec"] == "fp16")]
    if len(o) and len(g):
        xt.append(f"{name}\n{pl}")
        lat_4090.append(float(g["lat_p50_ms"].iloc[0]))
        lat_orin.append(float(o["lat_p50_ms"].iloc[0]))
        en_4090.append(float(g["energy_per_frame_mj"].iloc[0]) if pd.notna(g["energy_per_frame_mj"].iloc[0]) else np.nan)
        en_orin.append(float(o["energy_per_frame_mj"].iloc[0]))
x = np.arange(len(xt)); w = 0.35
ax.bar(x - w / 2, lat_4090, w, color="#1f77b4", edgecolor="k", label="4090 FP16 latency")
ax.bar(x + w / 2, lat_orin, w, color="#9467bd", edgecolor="k", label="Orin FP16 latency (30W)")
for i in range(len(xt)):
    ax.text(x[i] - w / 2, lat_4090[i] + 0.4, f"{lat_4090[i]:.2f}", ha="center", fontsize=7.5)
    ax.text(x[i] + w / 2, lat_orin[i] + 0.4, f"{lat_orin[i]:.1f}", ha="center", fontsize=7.5)
ax.set_xticks(x); ax.set_xticklabels(xt, fontsize=8.5)
ax.set_ylabel("latency p50 (ms)")
ax.set_title("C. Cross-hardware FP16: 4090 vs Orin AGX (collab2 body)")
ax.grid(alpha=0.3, axis="y")
ax2 = ax.twinx()
ax2.plot(x - w / 2, en_4090, "o--", color="#1f77b4", alpha=0.6, label="4090 energy (board)")
ax2.plot(x + w / 2, en_orin, "s--", color="#9467bd", alpha=0.6, label="Orin energy (module-total)")
ax2.set_ylabel("energy per frame (mJ)")
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=7.3, loc="upper right")
_slow = [lat_orin[i] / lat_4090[i] for i in range(len(xt))] if xt else [0]
ax.text(0.02, 0.96, f"Orin ~{min(_slow):.0f}–{max(_slow):.0f}x slower than 4090 (30W edge). Energy domains DIFFER:\n4090=NVML board-rail (E5), Orin=module-total VIN_SYS_5V0.\nAbsolute energy NOT directly comparable; trend only.\nOrin INT8 omitted: output NOT match 4090 (ap_valid=False).",
        transform=ax.transAxes, fontsize=7.6, va="top", color="#553",
        bbox=dict(boxstyle="round", fc="#fffbe6", ec="#a80", alpha=0.92))

# ============ Panel D: coverage / stocktake ============
ax = axes[1, 1]
metrics = ["AP", "latency", "throughput", "energy", "model_size"]
cols5 = ["ap70", "lat_p50_ms", "throughput_fps", "energy_per_frame_mj", "engine_size_mb"]
SRC_ORDER = ["complete_points_v1", "perstage_AP_v2", "P0_1_p25_trap", "P0_2_136",
             "pathA_forced_int8", "pathB_head_int8", "E4_energy_v1", "E6_orin_p03",
             "E3_orin_dla_pipe_v1", "P12_collab2_throughput"]
rows_lbl, M = [f"TOTAL (n={len(main)})"], [[int(main[c].notna().sum()) for c in cols5]]
for src in SRC_ORDER:
    s = main[main["dataset_src"] == src]
    if len(s):
        rows_lbl.append(f"  {src} (n={len(s)})")
        M.append([int(s[c].notna().sum()) for c in cols5])
M = np.array(M, dtype=float)
ax.imshow(np.where(M > 0, 1.0, 0.0), cmap="Greens", vmin=0, vmax=1.5, aspect="auto")
ax.set_xticks(range(len(metrics))); ax.set_xticklabels(metrics, fontsize=9)
ax.set_yticks(range(len(rows_lbl))); ax.set_yticklabels(rows_lbl, fontsize=8)
for i in range(len(rows_lbl)):
    for j in range(len(metrics)):
        v = int(M[i, j])
        ax.text(j, i, str(v) if v else "—", ha="center", va="center",
                color="k" if v else "#999", fontsize=8.5, fontweight="bold" if v else "normal")
n5 = int((main[cols5].notna().sum(axis=1) == 5).sum())
ax.set_title(f"D. Coverage in dataset_v2 ({n5} rows 5-metric complete)")
# effective independent anchors = distinct INDEPENDENTLY-TRAINED backbone architectures
# (planes) in the front pool. precision/hardware are cheap variations of one anchor, so
# they are NOT counted as separate anchors (HANDOFF: declare ~6-8 effective anchors).
front = main[main["regime"] == "front"]
arch = front.apply(lambda r: (int(r.stage0_planes), int(r.stage1_planes), int(r.stage2_planes)),
                   axis=1)
anchors = arch.nunique()
cfgs = front.apply(lambda r: (int(r.stage0_planes), int(r.stage1_planes), int(r.stage2_planes),
                              r.prec, r.hardware), axis=1).nunique()
ax.text(0.0, -0.2, f"Effective independent anchors = {anchors} distinct trained backbone archs in front "
                   f"({sorted(set(arch))})\n→ {cfgs} planes×prec×hw front configs over them. Ablation rows "
                   "(forced-int8 / head-int8 / throughput-saturation) NOT in front.\n"
                   "Irreducible gaps: Orin rows (no model_size; INT8 AP invalid), E3 (no energy probe).",
        transform=ax.transAxes, fontsize=7.6, color="#070")

plt.tight_layout(rect=[0, 0.02, 1, 0.97])
out = "multi_agent/figure/dataset_pareto_coverage.png"
plt.savefig(out, dpi=130, bbox_inches="tight")
print("saved:", out)

# ---------- text stocktake ----------
print("\n=== STOCKTAKE (dataset_v2) ===")
print(f"rows: {len(main)} (real; pyramid_fusion; rtx4090 + orin_agx)")
print("5-axis coverage:", {m: int(main[c].notna().sum()) for m, c in zip(metrics, cols5)})
print(f"5-metric COMPLETE rows: {n5}")
print("throughput_kind:", main["throughput_kind"].value_counts().to_dict())
print("regime:", main["regime"].value_counts().to_dict())
print(f"front-pool effective independent anchors: {anchors}")
print("INT8 speedup per tier:", {t[0]: round(t[2], 3) for t in tiers})
