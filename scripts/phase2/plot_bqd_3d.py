"""绘制 e2e_bench_v1 在 B × Q × D 三维上的分布.

- B 轴: 剪枝率 (从 triplet 推导 — T1=0%, T2=25%, T3=37%, T4=50%, T5=62%, T6=75%, T7/T8=非标特殊形状)
- Q 轴: q_tag (7 分类: Q_fp32, Q_fp16, Q_int8_mm, Q_int8_ent, Q_mix_s0, Q_mix_s2, Q_int8_pc_wo)
- D 轴: D-config (4 分类: default/4GB, with_cudnn/8GB, cublas_lt/16GB, all_enabled/1GB)
- color: throughput_fps
- size: ap50 (越大点越大)

输出:
  paper_learning/2. AAAI最终故事/data/figs/bqd_3d_scatter.png  (主 3D 散点)
  paper_learning/2. AAAI最终故事/data/figs/bqd_heatmap_4panel.png  (4 D-config 各一张 B×Q heatmap)
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd

REPO = Path("/home/jichengzhi/UniV2X")
CSV = REPO / "paper_learning/2. AAAI最终故事/data/e2e_bench_v1.csv"
FIG_DIR = REPO / "paper_learning/2. AAAI最终故事/data/figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)

TRIPLET_ORDER = ["T1_base", "T2_p25", "T3_p37", "T4_p50",
                 "T5_p62", "T6_p75", "T7_wide_shallow", "T8_narrow_deep"]
TRIPLET_PRUNE = {  # 大致剪枝率/标签轴位置
    "T1_base": 0, "T2_p25": 25, "T3_p37": 37, "T4_p50": 50,
    "T5_p62": 62, "T6_p75": 75, "T7_wide_shallow": 40, "T8_narrow_deep": 55,
}
Q_ORDER = ["Q_fp32", "Q_fp16", "Q_int8_mm", "Q_int8_ent",
           "Q_mix_s0", "Q_mix_s2", "Q_int8_pc_wo"]
D_ORDER = [
    ("default", 4, "D1_default_4GB"),
    ("with_cudnn", 8, "D2_with_cudnn_8GB"),
    ("cublas_lt", 16, "D3_cublas_lt_16GB"),
    ("all_enabled", 1, "D4_all_enabled_1GB"),
]
D_LABEL = {(t, w): lbl for t, w, lbl in D_ORDER}


def load_df():
    df = pd.read_csv(CSV)
    df = df[df["build_success"] == True].copy()
    df["prune_pct"] = df["triplet"].map(TRIPLET_PRUNE)
    df["q_idx"] = df["q_tag"].map({q: i for i, q in enumerate(Q_ORDER)})
    df["d_idx"] = df.apply(lambda r: next(
        (i for i, (t, w, _) in enumerate(D_ORDER)
         if r["d_tactic"] == t and r["d_workspace_gb"] == w), -1), axis=1)
    df = df[df["d_idx"] >= 0].copy()
    return df


def plot_3d_scatter(df):
    fig = plt.figure(figsize=(13, 9))
    ax = fig.add_subplot(111, projection="3d")

    # 拆 healthy vs crashed (AP50 < 0.4 → crashed, 16/224 = 7%, 全部 Q_int8_ent)
    healthy = df[df["ap50"] >= 0.4].copy()
    crashed = df[df["ap50"] < 0.4].copy()

    # healthy: 圆点, 颜色 = fps, 大小 = fps (Pareto 真信号)
    fps_h = healthy["throughput_fps"].values
    fps_min, fps_max = df["throughput_fps"].min(), df["throughput_fps"].max()
    sizes_h = 30 + 220 * (fps_h - fps_min) / (fps_max - fps_min)
    sc = ax.scatter(healthy["prune_pct"], healthy["q_idx"], healthy["d_idx"],
                    c=fps_h, s=sizes_h, cmap="viridis",
                    vmin=fps_min, vmax=fps_max,
                    edgecolors="black", linewidths=0.4, alpha=0.85,
                    label=f"healthy (AP50>=0.4, n={len(healthy)})")
    # crashed: 红 X, 大小固定
    if len(crashed) > 0:
        ax.scatter(crashed["prune_pct"], crashed["q_idx"], crashed["d_idx"],
                   c="red", marker="x", s=140, linewidths=2.2,
                   label=f"crashed (AP50<0.4, n={len(crashed)}, all Q_int8_ent)")

    cb = fig.colorbar(sc, ax=ax, shrink=0.65, pad=0.10)
    cb.set_label("throughput_fps (healthy points)", fontsize=11)

    ax.set_xlabel("B: prune ratio (%)", fontsize=11, labelpad=8)
    ax.set_ylabel("Q: q_tag", fontsize=11, labelpad=10)
    ax.set_zlabel("D: tactic / workspace", fontsize=11, labelpad=10)
    ax.set_yticks(range(len(Q_ORDER)))
    ax.set_yticklabels(Q_ORDER, fontsize=8)
    ax.set_zticks(range(len(D_ORDER)))
    ax.set_zticklabels([lbl for _, _, lbl in D_ORDER], fontsize=8)
    ax.set_title(f"e2e_bench_v1: B x Q x D 3D distribution (N={len(df)} anchor)\n"
                 f"circle: healthy AP50 in [0.528, 0.568] -- color/size = throughput_fps\n"
                 f"red X: AP collapse (AP50 < 0.4) -- only Q_int8_ent path",
                 fontsize=11, pad=14)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.92), fontsize=9)
    ax.view_init(elev=22, azim=-58)
    plt.tight_layout()
    out = FIG_DIR / "bqd_3d_scatter.png"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out}")
    return out


def plot_4panel_heatmap(df):
    fig, axes = plt.subplots(1, 4, figsize=(20, 6), constrained_layout=True)
    fps_min, fps_max = df["throughput_fps"].min(), df["throughput_fps"].max()
    for i, (tac, ws, lbl) in enumerate(D_ORDER):
        sub = df[(df["d_tactic"] == tac) & (df["d_workspace_gb"] == ws)]
        pivot = sub.pivot_table(index="triplet", columns="q_tag",
                                values="throughput_fps", aggfunc="first")
        pivot = pivot.reindex(index=TRIPLET_ORDER, columns=Q_ORDER)
        ax = axes[i]
        im = ax.imshow(pivot.values, cmap="viridis",
                       vmin=fps_min, vmax=fps_max, aspect="auto")
        ax.set_title(lbl, fontsize=11)
        ax.set_xticks(range(len(Q_ORDER)))
        ax.set_xticklabels(Q_ORDER, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(TRIPLET_ORDER)))
        ax.set_yticklabels(TRIPLET_ORDER, fontsize=8)
        # annotate fps values
        for r in range(pivot.shape[0]):
            for c in range(pivot.shape[1]):
                v = pivot.values[r, c]
                if not np.isnan(v):
                    ax.text(c, r, f"{v:.0f}", ha="center", va="center",
                            fontsize=7, color="white" if v < (fps_min+fps_max)/2 else "black")
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cb.set_label("throughput_fps", fontsize=11)
    fig.suptitle("B × Q heatmap per D-config (throughput_fps)", fontsize=14)
    out = FIG_DIR / "bqd_heatmap_4panel.png"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out}")
    return out


def main():
    df = load_df()
    print(f"loaded {len(df)} rows from {CSV.name}")
    print(f"  B (triplet): {df['triplet'].nunique()} unique")
    print(f"  Q (q_tag):   {df['q_tag'].nunique()} unique")
    print(f"  D (cfg):     {df['d_idx'].nunique()} unique")
    plot_3d_scatter(df)
    plot_4panel_heatmap(df)


if __name__ == "__main__":
    main()
