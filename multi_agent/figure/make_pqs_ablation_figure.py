#!/usr/bin/env python3
"""P×Q×S three-arm ablation figure (preliminary: 4090 Q-ratios, pending H800 TVM int8 confirmation)."""
import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/home/jichengzhi/V2X/multi_agent/figure/pqs_ablation_preliminary.png"

# Data from smoke_pqs_v2_summary.json
with open("/home/jichengzhi/V2X/results/smoke_pqs_v2_summary.json") as f:
    summary = json.load(f)

hv = summary["hv_distribution"]
wil = summary["wilcoxon_joint_vs_serial"]
ps_pairs = summary["p_s_pairs"]
q_pairs = summary["q_rank_flip_pairs"]

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle(
    "P×Q×S Three-Arm Ablation — PyramidFusion @ H800 TVM (preliminary: estimated INT8 via 4090 TRT Q-ratios)\n"
    "9 widths, 12 seeds, budget=90, pop=10",
    fontsize=10, fontweight="bold"
)

# === Panel 1: HV Bars ===
ax = axes[0]
arms = ["A-joint-PQS", "A-serial-PQS", "A-noS-PQS"]
means = [hv[a]["mean"] for a in arms]
pcts  = [hv[a]["pct_of_joint"] for a in arms]
stds  = [hv[a]["std"] for a in arms]
colors = ["#2196F3", "#FF9800", "#9E9E9E"]
bars = ax.bar(["A-joint\n(P×Q×S)", "A-serial\n(Q-blind)", "A-noS\n(P×Q only)"],
              means, color=colors, alpha=0.85, edgecolor="black", linewidth=0.8)
ax.errorbar(range(3), means, yerr=stds, fmt="none", color="black", capsize=4, linewidth=1.5)
for i, (m, p) in enumerate(zip(means, pcts)):
    ax.text(i, m + 50, f"{p:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_ylabel("Hypervolume (HV)", fontsize=9)
ax.set_title("HV per Arm (↑ better)", fontsize=9)
ax.set_ylim(0, max(means)*1.12)
ax.axhline(means[1], color=colors[1], linestyle="--", alpha=0.5, linewidth=0.8)
# Annotation: gap
ax.annotate(f"Gap: {means[0]/means[1]*100-100:.1f}%\np={wil['p_value']:.1e}",
            xy=(0, means[0]), xytext=(0.5, means[0]*1.06),
            fontsize=8, ha="center",
            arrowprops=dict(arrowstyle="->", color="red"),
            color="red")

# === Panel 2: Q-rank-flip mechanism ===
ax2 = axes[1]
widths_lut = {
    "trap25\n[48,96,192]": {"fp16_d": 42355, "int8_d": 39578, "label": "W_g pair1"},
    "pad64\n[64,96,192]":  {"fp16_d": 47407, "int8_d": 43302, "label": "P_g pair1"},
    "mix_d\n[48,128,128]": {"fp16_d": 42447, "int8_d": 36826, "label": "W_g pair3 ★flip"},
    "s2_128\n[64,128,128]":{"fp16_d": 47435, "int8_d": 34060, "label": "P_g pair3 ★flip"},
    "base\n[64,128,256]":  {"fp16_d": 56321, "int8_d": 35703, "label": "base"},
}
names = list(widths_lut.keys())
fp16s = [widths_lut[n]["fp16_d"]/1000 for n in names]
int8s = [widths_lut[n]["int8_d"]/1000 for n in names]
x = np.arange(len(names))
w = 0.35
ax2.bar(x - w/2, fp16s, w, label="FP16 default", color="#42A5F5", alpha=0.8)
ax2.bar(x + w/2, int8s, w, label="INT8 default", color="#EF5350", alpha=0.8)
ax2.set_xticks(x); ax2.set_xticklabels(names, fontsize=7.5)
ax2.set_ylabel("Latency (ms)", fontsize=9)
ax2.set_title("Q-rank-flip: FP16 vs INT8 ordering @ default sched", fontsize=9)
ax2.legend(fontsize=8)
# Annotate pair3 flip
ax2.annotate("Q-rank-flip!\n(FP16: mix_d<s2_128\nINT8: s2_128<mix_d)",
             xy=(3 + w/2, int8s[3]), xytext=(3.3, 40),
             fontsize=7, ha="left", color="darkred",
             arrowprops=dict(arrowstyle="->", color="darkred"))

# === Panel 3: Rank-flip summary table ===
ax3 = axes[2]
ax3.axis("off")
col_labels = ["Pair", "s0-W_g", "s0-P_g", "P×S flip", "Q flip", "Combined"]
rows_data = [
    ["pair1", "trap25[48,96,192]", "pad64[64,96,192]", "3.51×", "—", "P×S only"],
    ["pair2", "mix_b[48,64,256]",  "s1_64[64,64,256]", "3.29×", "—", "P×S only"],
    ["pair3", "mix_d[48,128,128]","s2_128[64,128,128]","3.83×","1.08×","P×Q×S ★"],
]
table = ax3.table(
    cellText=rows_data,
    colLabels=col_labels,
    cellLoc="center", loc="center",
    colWidths=[0.08, 0.23, 0.23, 0.12, 0.10, 0.14]
)
table.auto_set_font_size(False); table.set_fontsize(7.5)
table.scale(1, 1.8)
for j in range(len(col_labels)):
    table[0, j].set_facecolor("#1565C0"); table[0, j].set_text_props(color="white", fontweight="bold")
# Highlight pair3
for j in range(len(col_labels)):
    table[3, j].set_facecolor("#FFE0B2")
ax3.set_title("Rank-flip pairs (P×S confirmed; Q estimated via 4090 TRT)", fontsize=9)

# Status note
fig.text(0.5, 0.01,
    "⚠️ Estimated: INT8 latency via 4090 TRT Q-ratios. H800 TVM INT8 pending (hw-optimizer STEP1+2). "
    "Q-rank-flip pair3 may differ on H800 (single-conv benchmark: w48 is FASTER on H800 TVM).",
    ha="center", fontsize=7, style="italic", color="gray")

plt.tight_layout(rect=[0, 0.05, 1, 0.94])
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"[fig] {OUT}")
