"""Compare AP distributions across:
  - 4k+ e2e_bench_v1 (buggy-subnet AP, full search-space sweep, no FT axis)
  - plan v3 deployment view (FT=8 unique B×Q, 15 anchors)
  - plan v3 FT=4 + FT=15 (partial coverage)

Output: stats_v3/15_ap_dist_4k_vs_v3.png
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path("/home/jichengzhi/UniV2X")
OUT = REPO / "paper_learning/2. AAAI最终故事/data/stats_v3"

# 4k bench
bench = pd.read_csv(REPO / "paper_learning/2. AAAI最终故事/data/e2e_bench_v1.csv")
bench_ap = bench["ap50"].dropna().values
print(f"4k bench: N={len(bench_ap)}, mean={bench_ap.mean():.3f}, "
      f"std={bench_ap.std():.3f}, range={bench_ap.max()-bench_ap.min():.3f}")

# plan v3 deployment 54 anchor (含 D dup, FT∈{8,15})
v3_dep = pd.read_csv(OUT / "all_anchors.csv")
v3_ap = v3_dep["ap50"].dropna().values
print(f"plan v3 deployment: N={len(v3_ap)}, mean={v3_ap.mean():.3f}, "
      f"std={v3_ap.std():.3f}, range={v3_ap.max()-v3_ap.min():.3f}")

# plan v3 by FT - from spread_3a.csv (plan v2 phase 3a results)
spread = pd.read_csv("/tmp/a10_phase3a/spread_3a.csv")
ft_data = {}
for _, r in spread.iterrows():
    for ft_col, ft in [("FT4", 4), ("FT8", 8), ("FT15", 15)]:
        ft_data.setdefault(ft, []).append(r[ft_col])

# Also add Phase 3b (FT=8, 4 Q variants × 3 triplet = 12 anchors)
combo = pd.read_csv("/tmp/a10_phase3b/combo_3b.csv")
for _, r in combo.iterrows():
    for q in ["fp16", "int8_mm", "int8_pc_wo", "int8_ent"]:
        if q in combo.columns and pd.notna(r[q]):
            ft_data[8].append(float(r[q]))

# Plan v3 §1.3 noise 5 seeds at FT=4
noise = json.loads(Path("/tmp/a11_noise/noise.json").read_text())
ft_data[4].extend(noise["aps"])

# Plan v3 §1.1 underfit also has FT=2 (relevant for context)
underfit = pd.read_csv("/tmp/a11_underfit/underfit.csv")
ft_data[2] = []
for _, r in underfit.iterrows():
    if r["ft"] == 2:
        for q in ["fp16", "int8_mm", "int8_pc_wo", "int8_ent"]:
            if q in underfit.columns:
                try:
                    v = float(r[q]) if r[q] != "NA" else None
                except (ValueError, TypeError):
                    v = None
                if v is not None:
                    ft_data[2].append(v)

for ft, vals in sorted(ft_data.items()):
    arr = np.array([v for v in vals if v is not None and not np.isnan(v)])
    if len(arr):
        print(f"  FT={ft}: N={len(arr)}, mean={arr.mean():.3f}, "
              f"std={arr.std():.3f}, range={arr.max()-arr.min():.3f}")

# === Plot 4-panel comparison ===
fig, axes = plt.subplots(2, 2, figsize=(15, 9))
axes_flat = axes.flatten()
colors = ["#2980B9", "#E67E22", "#27AE60", "#C0392B"]

# Panel 0: 4k bench
ax = axes_flat[0]
ax.hist(bench_ap, bins=40, color=colors[0], alpha=0.7, edgecolor="k")
ax.axvline(0.30, color="red", ls="--", alpha=0.6, label="collapse 0.30")
ax.axvline(0.50, color="orange", ls="--", alpha=0.6, label="degrade 0.50")
ax.set_xlabel("AP50"); ax.set_ylabel("count")
ax.legend(loc="upper left")
ax.set_title(f"P0: 4k bench (subnet AP, full B×Q×D sweep, NO FT axis)\n"
             f"N={len(bench_ap)}, std={bench_ap.std():.3f}, "
             f"range={bench_ap.max()-bench_ap.min():.3f}, mean={bench_ap.mean():.3f}")

# Panels 1-3: FT=4, 6, 8 from plan v3
ft_levels = [4, 8, 15]
for i, ft in enumerate(ft_levels, start=1):
    ax = axes_flat[i]
    arr = np.array([v for v in ft_data.get(ft, []) if v is not None and not np.isnan(v)])
    if len(arr) == 0:
        ax.text(0.5, 0.5, f"FT={ft}: NO DATA", ha="center", va="center")
        continue
    ax.hist(arr, bins=max(8, int(len(arr)/2)), color=colors[i],
            alpha=0.7, edgecolor="k")
    ax.axvline(0.30, color="red", ls="--", alpha=0.6, label="collapse 0.30")
    ax.axvline(0.50, color="orange", ls="--", alpha=0.6, label="degrade 0.50")
    ax.set_xlabel("AP50"); ax.set_ylabel("count")
    ax.set_xlim(-0.05, 0.85)  # uniform x-axis for comparison
    ax.legend(loc="upper left")
    ax.set_title(f"P{i}: plan v3 FT={ft} (Pyramid g8 + extreme prune)\n"
                 f"N={len(arr)}, std={arr.std():.3f}, "
                 f"range={arr.max()-arr.min():.3f}, mean={arr.mean():.3f}")

fig.suptitle("AP Distribution Comparison: 4k bench (full sweep) vs plan v3 (FT axis)",
             fontsize=14, fontweight="bold")
fig.tight_layout()
out = OUT / "15_ap_dist_4k_vs_v3.png"
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"\nSaved: {out}")
