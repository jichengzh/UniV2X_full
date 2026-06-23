"""Task#1 证据图: AP70 vs mAOE 沿剪枝率(标CI) + INT8量化噪声子图
- 主图: mAOE(FP16/INT8)沿剪枝率 + AP70(FP16)双 y轴对比, 每点标95%CI errorbars
- 子图: ΔmAOE(INT8-FP16)量化噪声, 标SNR (终核数 ISS-029)
- 数据源: multi_agent/data/dataset_v2.csv (complete_points_v1 + P0_1_p25_trap)
- 输出: multi_agent/figure/fig_maoe_evidence_v1.png + 可复跑
- 可复跑: python multi_agent/figure/make_maoe_evidence_v1.py

数字权威来源(ISS-020/024/029终核PASS):
  剪枝轴 SNR=14.2×(全档均半宽口径); pairwise CI全不重叠除p25↔p50
  INT8: p25/p50/p75 SNR=0.58/0.05/0.82×(噪声级,CI重叠,非单调变号), 无量化轴信号
  base INT8 Δ+0.0030(2.4×半宽CI不重叠)但非单调+量级仅5%, 按ISS-020#4不构成轴信号
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

ROOT = Path("/home/jichengzhi/V2X")
OUT = ROOT / "multi_agent/figure/fig_maoe_evidence_v1.png"

# ── 读取数据 ────────────────────────────────────────────────
df = pd.read_csv(ROOT / "multi_agent/data/dataset_v2.csv")

# 筛选 measured 8-anchor (uniform, ap_valid, mAOE非空)
_anchor = df[
    df["mAOE_basis"].isin(["measured"]) &
    df["ap_valid"].astype(bool) &
    df["mAOE"].notna() &
    (df["stage0_prec"] == df["stage1_prec"]) & (df["stage1_prec"] == df["stage2_prec"])
].copy()

fp16 = _anchor[_anchor["stage0_prec"] == "FP16"].sort_values("prune_rate").reset_index(drop=True)
int8 = _anchor[_anchor["stage0_prec"] == "INT8"].sort_values("prune_rate").reset_index(drop=True)

# x 坐标与标签
X_TICKS  = [0.0, 0.25, 0.50, 0.75]
X_LABELS = ["base\n(0.0)", "p25\n(0.25)", "p50\n(0.50)", "p75\n(0.75)"]
CLRS = {"FP16": "#1976D2", "INT8": "#D32F2F"}
ALPHA_CI = 0.18

# CI errorbars: 每点半宽 = (hi - lo) / 2
def ci_err(rows):
    lo = rows["mAOE_ci_lo"].values
    hi = rows["mAOE_ci_hi"].values
    return np.array([rows["mAOE"].values - lo, hi - rows["mAOE"].values])

# ── 终核 SNR 数字 (supervisor ISS-029, 勿自行修改) ──────────
SNR_PRUNING = 14.2   # 全档均半宽口径 (备用: 13.1×avg-half / 17.1×pooled-SE)
# INT8 delta mAOE per anchor (p25/p50/p75 SNR, base另算)
INT8_SNR = {0.25: 0.58, 0.50: 0.05, 0.75: 0.82}  # noise-level
BASE_INT8_DELTA = 0.0030   # base档 Δ+0.0030, 2.4×半宽 CI不重叠, 但非单调+量级5%, ISS-020#4非轴信号

# ── 图布局: 1行2列 ─────────────────────────────────────────
fig = plt.figure(figsize=(15, 7))
gs = gridspec.GridSpec(1, 2, width_ratios=[1.6, 1.0], wspace=0.32)
ax_main = fig.add_subplot(gs[0])
ax_sub  = fig.add_subplot(gs[1])

# ════════════════════════════════════════════════════════════
# 主图: mAOE + AP70 双 y 轴
# ════════════════════════════════════════════════════════════
ax_ap = ax_main.twinx()

# ── mAOE FP16 (主 y轴) ──
err_fp16 = ci_err(fp16)
ax_main.errorbar(fp16["prune_rate"], fp16["mAOE"],
                 yerr=err_fp16, fmt="-o", color=CLRS["FP16"],
                 linewidth=2.2, markersize=9, capsize=5, capthick=1.5,
                 label="mAOE FP16 (剪枝信号 SNR=14.2×)", zorder=4)
ax_main.fill_between(fp16["prune_rate"],
                     fp16["mAOE_ci_lo"], fp16["mAOE_ci_hi"],
                     color=CLRS["FP16"], alpha=ALPHA_CI)
# 标数值
for _, r in fp16.iterrows():
    ax_main.annotate(f'{r["mAOE"]:.4f}', (r["prune_rate"], r["mAOE"]),
                     xytext=(0, 10), textcoords="offset points",
                     ha="center", fontsize=8.5, color=CLRS["FP16"], fontweight="bold")

# ── mAOE INT8 (主 y轴, 虚线) ──
err_int8 = ci_err(int8)
ax_main.errorbar(int8["prune_rate"], int8["mAOE"],
                 yerr=err_int8, fmt="--s", color=CLRS["INT8"],
                 linewidth=2, markersize=8, capsize=5, capthick=1.5,
                 label="mAOE INT8", zorder=4)
ax_main.fill_between(int8["prune_rate"],
                     int8["mAOE_ci_lo"], int8["mAOE_ci_hi"],
                     color=CLRS["INT8"], alpha=ALPHA_CI)
for _, r in int8.iterrows():
    ax_main.annotate(f'{r["mAOE"]:.4f}', (r["prune_rate"], r["mAOE"]),
                     xytext=(0, -16), textcoords="offset points",
                     ha="center", fontsize=8.5, color=CLRS["INT8"])

# ── AP70 FP16 (次 y轴, 灰色) ──
ax_ap.plot(fp16["prune_rate"], fp16["ap70"], ":", color="gray",
           linewidth=1.8, marker="^", markersize=7, label="AP70 FP16 (高原, 右轴)", zorder=3)
ax_ap.set_ylabel("AP70 (高原, 约束轴)", color="gray", fontsize=10)
ax_ap.tick_params(axis="y", labelcolor="gray")
ax_ap.set_ylim(0.48, 0.68)  # AP 高原范围

# ── p25↔p50 CI 重叠标注 ──
x_mid = (0.25 + 0.50) / 2
y_top = fp16[fp16["prune_rate"] == 0.25]["mAOE_ci_hi"].values[0]
ax_main.annotate("", xy=(0.50, y_top + 0.0008), xytext=(0.25, y_top + 0.0008),
                 arrowprops=dict(arrowstyle="<->", color="darkorange", lw=1.5))
ax_main.text(x_mid, y_top + 0.0015, "CI重叠\n(p25↔p50\n不可分辨)",
             ha="center", fontsize=8, color="darkorange",
             bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow", alpha=0.9))

# 主图装饰
ax_main.set_xlabel("剪枝率 (prune_rate)", fontsize=11)
ax_main.set_ylabel("mAOE (rad, ↑=退化, 越小越好)", fontsize=11)
ax_main.set_title("mAOE 沿剪枝率单调上升 (剪枝约束信号)\nSNR=14.2× 全档均半宽 | 95% bootstrap CI",
                  fontsize=11, fontweight="bold")
ax_main.set_xticks(X_TICKS)
ax_main.set_xticklabels(X_LABELS, fontsize=10)
ax_main.set_xlim(-0.08, 0.83)
ax_main.grid(True, alpha=0.3)
ax_main.set_ylim(0.052, 0.098)

# 合并图例
lines1, labels1 = ax_main.get_legend_handles_labels()
lines2, labels2 = ax_ap.get_legend_handles_labels()
ax_main.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")

# ════════════════════════════════════════════════════════════
# 子图: ΔmAOE (INT8 - FP16) 量化噪声
# ════════════════════════════════════════════════════════════
# 对齐 FP16/INT8 到相同 prune_rate
fp16_dict = dict(zip(fp16["prune_rate"], fp16["mAOE"]))
int8_dict = dict(zip(int8["prune_rate"], int8["mAOE"]))
fp16_ci_w = dict(zip(fp16["prune_rate"], (fp16["mAOE_ci_hi"] - fp16["mAOE_ci_lo"]) / 2))
int8_ci_w = dict(zip(int8["prune_rate"], (int8["mAOE_ci_hi"] - int8["mAOE_ci_lo"]) / 2))

prune_rates_sub = sorted(set(fp16_dict) & set(int8_dict))
deltas = [int8_dict[pr] - fp16_dict[pr] for pr in prune_rates_sub]
# combined CI width (RSS, conservative)
ci_combined = [np.sqrt(fp16_ci_w[pr]**2 + int8_ci_w[pr]**2) for pr in prune_rates_sub]

bar_colors = []
for pr, d in zip(prune_rates_sub, deltas):
    # base档: +0.0030, non-monotone, ISS-020#4 非轴信号
    # p25/p50/p75: SNR<1, noise
    bar_colors.append("#FF7043" if abs(d) > ci_combined[prune_rates_sub.index(pr)] else "#90CAF9")

bars = ax_sub.bar(prune_rates_sub, deltas, width=0.18, color=bar_colors,
                  edgecolor="gray", linewidth=0.8, zorder=3)
ax_sub.errorbar(prune_rates_sub, deltas, yerr=ci_combined,
                fmt="none", color="dimgray", capsize=5, capthick=1.5, zorder=4)
ax_sub.axhline(0, color="black", linewidth=1.2)

# 标 SNR 数字 (从终核报告取值, 不使用自算近似)
snr_map = {0.0: ("base\n2.4×半宽\n(非轴信号\nISS-020#4)", "#FFB300"),
           0.25: (f"SNR=0.58×\n(噪声级)", "#90CAF9"),
           0.50: (f"SNR=0.05×\n(噪声级)", "#90CAF9"),
           0.75: (f"SNR=0.82×\n(噪声级)", "#90CAF9")}
for pr, d in zip(prune_rates_sub, deltas):
    lbl, clr = snr_map.get(pr, ("", "gray"))
    y_off = 0.0006 if d >= 0 else -0.0010
    ax_sub.text(pr, d + y_off, lbl, ha="center", fontsize=7.5, color="navy",
                bbox=dict(boxstyle="round,pad=0.15", facecolor=clr, alpha=0.7))
    ax_sub.text(pr, d, f"{d:+.4f}", ha="center", va="center", fontsize=8,
                fontweight="bold", color="white" if abs(d) < 0.001 else "black")

# 子图装饰
ax_sub.set_xlabel("剪枝率 (prune_rate)", fontsize=11)
ax_sub.set_ylabel("ΔmAOE = INT8 − FP16 (rad)", fontsize=10)
ax_sub.set_title("量化噪声子图: INT8 vs FP16 ΔmAOE\n3/4档 SNR<1×, 非单调 → 无量化轴信号",
                 fontsize=10, fontweight="bold")
ax_sub.set_xticks(prune_rates_sub)
ax_sub.set_xticklabels(X_LABELS[:len(prune_rates_sub)], fontsize=10)
ax_sub.grid(True, alpha=0.3, axis="y")
ax_sub.set_ylim(-0.004, 0.010)

# 图说明框
from matplotlib.patches import Patch
legend_elems = [
    Patch(facecolor="#FF7043", label="|Δ| > CI (base档, 非单调→非轴信号)"),
    Patch(facecolor="#90CAF9", label="|Δ| ≤ CI (噪声级, SNR<1×)"),
]
ax_sub.legend(handles=legend_elems, fontsize=8, loc="upper left")

# ── 全图标题 + 数据出处 ──────────────────────────────────────
fig.suptitle(
    "Phase M 证据图 — mAOE 沿剪枝率变化 vs INT8 量化噪声 (ISS-020/024/029 终核 PASS)\n"
    "结论: 剪枝轴 mAOE SNR=14.2×(约束信号成立); INT8 3/4档噪声级+非单调 → 无量化轴信号",
    fontsize=11.5, fontweight="bold", y=1.01
)
fig.text(
    0.5, -0.02,
    "数据源: multi_agent/data/dataset_v2.csv (measured 8行, ISS-024修正版)\n"
    "epoch: base→bestval(epoch23) / pruned×3→net_epoch25.pth | CI: 同人口全集bootstrap 95%",
    ha="center", fontsize=8.5, color="dimgray", style="italic"
)

plt.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"[Task#1] 证据图已保存: {OUT}")

# ── 打印数字摘要 (供 supervisor 复核) ──────────────────────
print("\n=== 图中数字摘要 (ISS-029 终核 PASS 来源) ===")
print(f"FP16 mAOE: {list(fp16['prune_rate'].values)} → {list(fp16['mAOE'].round(6).values)}")
print(f"FP16 CI_lo: {list(fp16['mAOE_ci_lo'].round(6).values)}")
print(f"FP16 CI_hi: {list(fp16['mAOE_ci_hi'].round(6).values)}")
print(f"INT8 mAOE: {list(int8['prune_rate'].values)} → {list(int8['mAOE'].round(6).values)}")
print(f"ΔmAOE(INT8-FP16): {[round(int8_dict[pr]-fp16_dict[pr],6) for pr in sorted(set(fp16_dict)&set(int8_dict))]}")
print(f"SNR标注: 剪枝轴={SNR_PRUNING}× | INT8: {INT8_SNR} (终核数, 勿自算替换)")
print(f"p25↔p50 CI重叠: fp16 p25 CI_hi={fp16[fp16.prune_rate==0.25]['mAOE_ci_hi'].values[0]:.6f}, "
      f"p50 CI_lo={fp16[fp16.prune_rate==0.50]['mAOE_ci_lo'].values[0]:.6f} "
      f"→ {'重叠✓' if fp16[fp16.prune_rate==0.25]['mAOE_ci_hi'].values[0] > fp16[fp16.prune_rate==0.50]['mAOE_ci_lo'].values[0] else '未重叠❌'}")
print(f"anchor 点: {len(fp16)} FP16 + {len(int8)} INT8 = {len(_anchor)} total")
