"""Phase M 判据图: AP 平坦 vs TP 误差项信号对比
- 展示 mAOE 在 AP 高原处是否携带更强的剪枝/量化退化信号
- 输出: multi_agent/figure/fig_ap_vs_tp_error.png (4 子图)
- 可复跑: python multi_agent/figure/make_ap_vs_tp_error.py

数据来源: multi_agent/data/dataset_v2.csv (complete_points_v1, uniform FP16/INT8, ap_valid=True)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

ROOT = Path("/home/jichengzhi/V2X")
OUT = ROOT / "multi_agent/figure/fig_ap_vs_tp_error.png"

# 读取数据
df = pd.read_csv(ROOT / "multi_agent/data/dataset_v2.csv")

# 筛选: 4 个 front anchor (complete_points_v1), uniform prec, ap_valid=True, mAOE 有值
_anchor = df[
    (df["dataset_src"] == "complete_points_v1") &
    (df["ap_valid"] == True) &
    df["mAOE"].notna() &
    (df["stage0_prec"] == df["stage1_prec"]) & (df["stage1_prec"] == df["stage2_prec"])
].copy()

fp16 = _anchor[_anchor["stage0_prec"] == "FP16"].sort_values("prune_rate").reset_index(drop=True)
int8 = _anchor[_anchor["stage0_prec"] == "INT8"].sort_values("prune_rate").reset_index(drop=True)

# prune_rate 映射到 x 轴标签
def planes_label(row):
    return f"({int(row.stage0_planes)},{int(row.stage1_planes)},{int(row.stage2_planes)})"

fp16["x_label"] = fp16.apply(planes_label, axis=1)
int8["x_label"] = int8.apply(planes_label, axis=1)

# ──────────────────────────────────────────────────────────
# 正规化: 相对于 base FP16 的百分比变化
# metric Δ% = (val - base_val) / base_val * 100
# ──────────────────────────────────────────────────────────
base_fp16 = fp16[fp16["prune_rate"] == 0.0].iloc[0]
base_int8 = int8[int8["prune_rate"] == 0.0].iloc[0] if len(int8[int8["prune_rate"]==0]) > 0 else None

metrics = {
    "AP70":  "ap70",
    "mATE":  "mATE",
    "mASE":  "mASE",
    "mAOE":  "mAOE",
}
# 对 AP70: 正方向=好(降表示退化); 对 TP 误差: 正方向=坏(升表示退化)
# 统一绘制: Δ% from baseline, 正值=退化

def delta_pct(series, baseline):
    return (series - baseline) / abs(baseline) * 100

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle(
    "Phase M: AP 平坦 vs TP 几何误差项信号\n"
    "mAOE/mASE/mATE 在 AP 高原处是否携带更强的剪枝/量化退化信号",
    fontsize=13, fontweight="bold"
)

COLORS = {"FP16": "#2196F3", "INT8": "#F44336"}
MARKERS = {"FP16": "o", "INT8": "s"}

def plot_metric(ax, key, col, baseline_fp16, baseline_int8, sign=1):
    """sign=+1: 指标升=退化 (mATE/mASE/mAOE); sign=-1: 指标降=退化 (AP70)"""
    # FP16
    if len(fp16) > 0 and col in fp16.columns:
        y_fp16 = delta_pct(fp16[col], baseline_fp16[col]) * sign
        ax.plot(fp16["prune_rate"], y_fp16, color=COLORS["FP16"], marker=MARKERS["FP16"],
                linewidth=2, markersize=8, label="FP16", zorder=3)
        for _, r in fp16.iterrows():
            ax.annotate(
                f"{r[col]:.4f}",
                (r["prune_rate"], delta_pct(pd.Series([r[col]]), baseline_fp16[col]).iloc[0] * sign),
                textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7.5,
                color=COLORS["FP16"]
            )
    # INT8
    if len(int8) > 0 and col in int8.columns and int8[col].notna().any():
        y_int8 = delta_pct(int8[col], baseline_int8[col] if baseline_int8 is not None else baseline_fp16[col]) * sign
        ax.plot(int8["prune_rate"], y_int8, color=COLORS["INT8"], marker=MARKERS["INT8"],
                linewidth=2, markersize=8, label="INT8", zorder=3, linestyle="--")
        for _, r in int8.iterrows():
            b = baseline_int8[col] if baseline_int8 is not None else baseline_fp16[col]
            ax.annotate(
                f"{r[col]:.4f}",
                (r["prune_rate"], delta_pct(pd.Series([r[col]]), b).iloc[0] * sign),
                textcoords="offset points", xytext=(0, -14), ha="center", fontsize=7.5,
                color=COLORS["INT8"]
            )
    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_xlabel("剪枝率 (prune_rate)", fontsize=10)
    ax.set_ylabel(f"退化幅度 Δ% (from base FP16)", fontsize=9)
    ax.set_title(f"{key} {'(↑=退化, 越小越好)' if sign==1 else '(↓=退化, 越大越好)'}", fontsize=11)
    ax.set_xticks([0, 0.25, 0.5, 0.75])
    ax.set_xticklabels(["base\n(0.0)", "p25\n(0.25)", "p50\n(0.50)", "p75\n(0.75)"])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

# 子图1: AP70 (信号弱, 高原)
plot_metric(axes[0, 0], "AP70", "ap70", base_fp16, base_int8, sign=-1)

# 子图2: mAOE (预期信号最强)
plot_metric(axes[0, 1], "mAOE (rad, 越小越好)", "mAOE", base_fp16, base_int8, sign=1)

# 子图3: mASE
plot_metric(axes[1, 0], "mASE (0-1, 越小越好)", "mASE", base_fp16, base_int8, sign=1)

# 子图4: mATE (m)
plot_metric(axes[1, 1], "mATE (m, 越小越好)", "mATE", base_fp16, base_int8, sign=1)

# 注释
# 计算信号强度 summary
_span_ap70_norm = (fp16["ap70"].max() - fp16["ap70"].min()) / base_fp16["ap70"]
_span_maoe_norm = (fp16["mAOE"].max() - fp16["mAOE"].min()) / base_fp16["mAOE"]
_span_mate_norm = (fp16["mATE"].max() - fp16["mATE"].min()) / base_fp16["mATE"]
_span_mase_norm = (fp16["mASE"].max() - fp16["mASE"].min()) / base_fp16["mASE"]

# INT8 vs FP16 at base
if base_int8 is not None:
    _int8_ap70_delta = (base_int8["ap70"] - base_fp16["ap70"]) / base_fp16["ap70"] * 100
    _int8_maoe_delta = (base_int8["mAOE"] - base_fp16["mAOE"]) / base_fp16["mAOE"] * 100
    int8_note = (f"INT8 量化灵敏度 (base): AP70 Δ={_int8_ap70_delta:+.1f}% | "
                 f"mAOE Δ={_int8_maoe_delta:+.1f}%  ({abs(_int8_maoe_delta/max(abs(_int8_ap70_delta),0.001)):.1f}× 更敏感)")
else:
    int8_note = "INT8 base 无对应点"

summary_text = (
    f"信号强度 (FP16 4-anchor, normalized span):\n"
    f"  AP70:  {_span_ap70_norm:.3f} (基准)\n"
    f"  mAOE:  {_span_maoe_norm:.3f}  → {_span_maoe_norm/_span_ap70_norm:.1f}× AP70\n"
    f"  mASE:  {_span_mase_norm:.3f}  → {_span_mase_norm/_span_ap70_norm:.1f}× AP70\n"
    f"  mATE:  {_span_mate_norm:.3f}  → {_span_mate_norm/_span_ap70_norm:.1f}× AP70\n\n"
    f"{int8_note}\n\n"
    f"⚠️ pruned50 INT8 mATE/mAOE 幸存者偏差 (n_tp-828, 不可直接比)\n"
    f"⚠️ pruned75 AP70 tp=0.5078 vs stage_a=0.5300 (GPU3差异, mAOE趋势有效)\n\n"
    f"结论: mAOE 在 AP 高原处携带 {_span_maoe_norm/_span_ap70_norm:.1f}× 信号 → 换指标路线成立"
)

fig.text(0.01, 0.01, summary_text, fontsize=8.5, family="monospace",
         verticalalignment="bottom",
         bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

plt.tight_layout(rect=[0, 0.20, 1, 0.96])
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"图已保存: {OUT}")

# 打印数字摘要
print("\n=== 信号强度摘要 ===")
print(f"FP16 normalized span (relative to base FP16):")
print(f"  AP70: {_span_ap70_norm:.4f} ({_span_ap70_norm*100:.1f}%)")
print(f"  mAOE: {_span_maoe_norm:.4f} ({_span_maoe_norm*100:.1f}%) → {_span_maoe_norm/_span_ap70_norm:.2f}x AP70")
print(f"  mASE: {_span_mase_norm:.4f} ({_span_mase_norm*100:.1f}%) → {_span_mase_norm/_span_ap70_norm:.2f}x AP70")
print(f"  mATE: {_span_mate_norm:.4f} ({_span_mate_norm*100:.1f}%) → {_span_mate_norm/_span_ap70_norm:.2f}x AP70")
if base_int8 is not None:
    print(f"\nINT8 量化灵敏度 (base FP16→INT8):")
    print(f"  AP70 Δ: {_int8_ap70_delta:+.2f}%")
    print(f"  mAOE Δ: {_int8_maoe_delta:+.2f}% → {abs(_int8_maoe_delta/max(abs(_int8_ap70_delta),0.001)):.1f}x 更敏感")
print(f"\n数据: {len(fp16)} FP16 + {len(int8)} INT8 = {len(_anchor)} anchor 点")
print(f"source: multi_agent/data/dataset_v2.csv (complete_points_v1, uniform, ap_valid=True)")
