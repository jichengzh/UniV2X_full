"""Plan 4 LAT 图集 — 对照 stats_v3_plan4/ AP 模板, 输出 lat 维度图.

Source: bench v1.csv (4584 anchor = 21 g32 triplet × 7 Q × 32 D)
Target: lat_p50_ms = 1000 / throughput_fps
Output: paper_learning/2. AAAI最终故事/data/stats_v3_plan4_lat/

8 张图:
  01_lat_marginal.png       — 4 axis marginal (triplet/Q/D/prune)
  02_lat_grid_BxQ.png        — 21×7 heatmap (mean lat per cell)
  03_lat_grid_BxD.png        — 21×32 heatmap
  04_lat_cell_std_BxQ.png    — 21×7 heatmap (cell-内 D std)
  05_lat_histogram.png       — 4584 anchor lat hist
  06_lat_by_axis_box.png     — by triplet / Q / D boxplots
  07_lat_vs_planes.png       — scatter + Spearman rho per Q
  08_lat_vs_ap_compare.png   — same cell AP std vs LAT std
"""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as scistats

REPO = Path("/home/jichengzhi/UniV2X")
DEFAULT_OUT = REPO / "paper_learning/2. AAAI最终故事/data/stats_v3_plan4_lat"
sns.set_theme(style="whitegrid", palette="colorblind")

TRIPLET_PRUNE = {
    'T1_base': 0.0, 'T2_p25': 0.25, 'T3_p37': 0.37, 'T4_p50': 0.50,
    'T5_p62': 0.62, 'T6_p75': 0.75, 'T7_wide_shallow': 0.40, 'T8_narrow_deep': 0.30,
    'T10_p11': 0.11, 'T11_p14': 0.14, 'T12_p21': 0.21, 'T13_p29': 0.29,
    'T14_p36': 0.36, 'T15_p39': 0.39, 'T16_p54': 0.54, 'T17_p57': 0.57,
    'T18_p64': 0.64, 'T19_p71': 0.71, 'T20_p79': 0.79, 'T21_p82': 0.82,
    'T22_p89': 0.89,
}


def load_data():
    df = pd.read_csv(REPO / 'paper_learning/2. AAAI最终故事/data/e2e_bench_v1.csv')
    for c in ['throughput_fps', 'stage0_planes', 'stage1_planes', 'stage2_planes',
              'ap50', 'build_secs', 'engine_size_mb']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df[df['build_success'] == True].dropna(subset=['throughput_fps']).reset_index(drop=True)
    df['lat_p50_ms'] = 1000.0 / df['throughput_fps']
    df['planes_total'] = df['stage0_planes'] + df['stage1_planes'] + df['stage2_planes']
    df['prune_pct'] = df['triplet'].map(TRIPLET_PRUNE)
    # Triplet 按 prune% 排序的顺序
    df['triplet_order'] = df['prune_pct'].rank(method='first')
    return df


def fig_01_marginal(df, out):
    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    # triplet (21 levels)
    ax = axes[0, 0]
    order = sorted(df['triplet'].unique(), key=lambda t: TRIPLET_PRUNE.get(t, 0))
    counts = df['triplet'].value_counts().reindex(order)
    ax.bar(range(len(counts)), counts.values, color='#2980B9')
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(order, rotation=60, ha='right', fontsize=8)
    ax.set_title(f'Triplet (21, sorted by prune%)\nN total = {len(df)}')
    ax.set_ylabel('anchor count')

    # Q (7 levels)
    ax = axes[0, 1]
    counts = df['q_tag'].value_counts()
    ax.bar(range(len(counts)), counts.values, color='#27AE60')
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index, rotation=45, ha='right', fontsize=9)
    ax.set_title(f'Q (7 variants)')
    ax.set_ylabel('anchor count')

    # D (32 levels)
    ax = axes[1, 0]
    counts = df['d_tag'].value_counts()
    ax.bar(range(len(counts)), counts.values, color='#E67E22')
    ax.set_xticks([])
    ax.set_title(f'D (32 cells)')
    ax.set_ylabel('anchor count')

    # prune (continuous via triplet)
    ax = axes[1, 1]
    pp = df.groupby('triplet')['prune_pct'].first().sort_values().values * 100
    ax.scatter(range(len(pp)), pp, s=40, color='#C0392B')
    ax.set_xticks(range(len(pp)))
    ax.set_xticklabels([f"{p:.0f}%" for p in pp], rotation=45, fontsize=8)
    ax.set_xlabel('triplet (sorted by prune%)')
    ax.set_ylabel('prune ratio (%)')
    ax.set_title('Prune ratio coverage (g32 only)')

    fig.suptitle('Lat figs / A1: Marginal Coverage [bench v1, 4584 anchor]', fontsize=13)
    fig.tight_layout()
    fig.savefig(out / '01_lat_marginal.png', dpi=130)
    plt.close(fig)
    print(f"  → 01_lat_marginal.png")


def fig_02_grid_bxq(df, out):
    """B × Q heatmap, value = mean lat per cell (averaged over 32 D)."""
    pivot = df.pivot_table(values='lat_p50_ms', index='triplet', columns='q_tag',
                          aggfunc='mean')
    order = sorted(pivot.index, key=lambda t: TRIPLET_PRUNE.get(t, 0))
    pivot = pivot.reindex(order)

    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='YlOrRd',
                cbar_kws={'label': 'mean lat (ms, over 32 D)'}, ax=ax,
                linewidths=0.3)
    ax.set_title('Lat A2: B × Q mean lat (over 32 D)\n21 triplet × 7 Q')
    fig.tight_layout()
    fig.savefig(out / '02_lat_grid_BxQ.png', dpi=130)
    plt.close(fig)
    print(f"  → 02_lat_grid_BxQ.png")


def fig_03_grid_bxd(df, out):
    """B × D heatmap, value = mean lat per cell (averaged over 7 Q)."""
    pivot = df.pivot_table(values='lat_p50_ms', index='triplet', columns='d_tag',
                          aggfunc='mean')
    order = sorted(pivot.index, key=lambda t: TRIPLET_PRUNE.get(t, 0))
    pivot = pivot.reindex(order)

    fig, ax = plt.subplots(figsize=(18, 8))
    sns.heatmap(pivot, annot=False, cmap='YlOrRd',
                cbar_kws={'label': 'mean lat (ms, over 7 Q)'}, ax=ax,
                linewidths=0.1)
    ax.set_title('Lat A3: B × D mean lat (over 7 Q)\n21 triplet × 32 D')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=60, ha='right', fontsize=7)
    fig.tight_layout()
    fig.savefig(out / '03_lat_grid_BxD.png', dpi=130)
    plt.close(fig)
    print(f"  → 03_lat_grid_BxD.png")


def fig_04_cell_std_bxq(df, out):
    """B × Q heatmap, value = cell-内 D std (跨 32 D)."""
    pivot = df.pivot_table(values='lat_p50_ms', index='triplet', columns='q_tag',
                          aggfunc='std')
    order = sorted(pivot.index, key=lambda t: TRIPLET_PRUNE.get(t, 0))
    pivot = pivot.reindex(order)

    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='Blues',
                cbar_kws={'label': 'cell-内 lat std (ms, 跨 32 D)'}, ax=ax,
                linewidths=0.3)
    ax.set_title('Lat A4: B × Q cell-内 D std (跨 32 D 复制)\n反映 D 在该 cell 上的 lat 影响幅度')
    fig.tight_layout()
    fig.savefig(out / '04_lat_cell_std_BxQ.png', dpi=130)
    plt.close(fig)
    print(f"  → 04_lat_cell_std_BxQ.png")


def fig_05_histogram(df, out):
    lats = df['lat_p50_ms'].values
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.histplot(lats, bins=60, kde=True, ax=ax, color='#2980B9')
    ax.set_xlabel('lat_p50 (ms)')
    ax.set_ylabel('count')
    ax.axvline(np.median(lats), color='k', ls='--', label=f'median={np.median(lats):.2f}')
    ax.axvline(np.percentile(lats, 95), color='orange', ls='--',
               label=f'p95={np.percentile(lats, 95):.2f}')
    ax.legend()
    ax.set_title(f'Lat B1: lat_p50 histogram (N={len(lats)})\n'
                 f'mean={lats.mean():.2f} ms, std={lats.std():.2f}, '
                 f'range=[{lats.min():.2f}, {lats.max():.2f}], CV={100*lats.std()/lats.mean():.1f}%')
    fig.tight_layout()
    fig.savefig(out / '05_lat_histogram.png', dpi=130)
    plt.close(fig)
    print(f"  → 05_lat_histogram.png")


def fig_06_by_axis_box(df, out):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    # by triplet (sorted by prune%)
    ax = axes[0]
    order = sorted(df['triplet'].unique(), key=lambda t: TRIPLET_PRUNE.get(t, 0))
    sns.boxplot(data=df, x='triplet', y='lat_p50_ms', order=order, ax=ax,
                showfliers=False, color='#3498DB')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=60, ha='right', fontsize=8)
    ax.set_title('by triplet (sorted by prune% 0→89%)')
    ax.set_xlabel('triplet')
    ax.set_ylabel('lat_p50 (ms)')

    # by Q
    ax = axes[1]
    q_order = ['Q_fp32', 'Q_fp16', 'Q_int8_mm', 'Q_int8_pc_wo',
               'Q_int8_ent', 'Q_mix_s0', 'Q_mix_s2']
    sns.boxplot(data=df, x='q_tag', y='lat_p50_ms', order=q_order, ax=ax,
                showfliers=False, color='#27AE60')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    ax.set_title('by Q variant')
    ax.set_xlabel('Q')
    ax.set_ylabel('lat_p50 (ms)')

    # by D (32 cells)
    ax = axes[2]
    d_order = sorted(df['d_tag'].unique(),
                     key=lambda d: int(d.split('_')[0][1:]))
    sns.boxplot(data=df, x='d_tag', y='lat_p50_ms', order=d_order, ax=ax,
                showfliers=False, color='#E67E22')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='center', fontsize=6)
    ax.set_title('by D config (32 cells)')
    ax.set_xlabel('D')
    ax.set_ylabel('lat_p50 (ms)')

    fig.suptitle('Lat B3: by axis boxplots (median across other axes)', fontsize=13)
    fig.tight_layout()
    fig.savefig(out / '06_lat_by_axis_box.png', dpi=130)
    plt.close(fig)
    print(f"  → 06_lat_by_axis_box.png")


def fig_07_lat_vs_planes(df, out):
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    q_list = ['Q_fp32', 'Q_fp16', 'Q_int8_mm', 'Q_int8_pc_wo',
              'Q_int8_ent', 'Q_mix_s0', 'Q_mix_s2']
    # 用 D=D1 固定看每 Q 的 lat~planes
    for i, q in enumerate(q_list):
        ax = axes[i // 4, i % 4]
        sub = df[(df['q_tag'] == q) & (df['d_tag'] == 'D1_default_4gb')]
        if len(sub) < 5:
            ax.text(0.5, 0.5, 'no data', ha='center')
            continue
        ax.scatter(sub['planes_total'], sub['lat_p50_ms'], s=50, alpha=0.7)
        rho, p = scistats.spearmanr(sub['planes_total'], sub['lat_p50_ms'])
        ax.set_title(f'{q}\nρ={rho:+.3f}  p={p:.3g}', fontsize=10)
        ax.set_xlabel('planes_total')
        ax.set_ylabel('lat (ms)')

    # 第 8 个: 跨 32 D 平均 (D 噪声归 0)
    ax = axes[1, 3]
    g = df.groupby('triplet').agg(lat=('lat_p50_ms', 'mean'),
                                   planes=('planes_total', 'first'),
                                   prune=('prune_pct', 'first')).reset_index()
    ax.scatter(g['planes'], g['lat'], s=80, c=g['prune'], cmap='coolwarm',
               edgecolor='k', linewidth=0.5)
    rho, p = scistats.spearmanr(g['planes'], g['lat'])
    ax.set_title(f'avg over 7 Q × 32 D\nρ={rho:+.3f}  p={p:.3g} (D 噪声归 0)',
                 fontsize=10, fontweight='bold')
    ax.set_xlabel('planes_total')
    ax.set_ylabel('mean lat (ms)')

    fig.suptitle('Lat C1: lat vs planes_total per Q (D=D1) + 平均 (右下)', fontsize=13)
    fig.tight_layout()
    fig.savefig(out / '07_lat_vs_planes.png', dpi=130)
    plt.close(fig)
    print(f"  → 07_lat_vs_planes.png")


def fig_08_lat_vs_ap_compare(df, out):
    """同 cell 内: AP std vs LAT std (跨 32 D 复制)."""
    df2 = df.dropna(subset=['ap50']).copy()
    g = df2.groupby(['triplet', 'q_tag']).agg(
        ap_std=('ap50', 'std'), lat_std=('lat_p50_ms', 'std'),
        ap_mean=('ap50', 'mean'), lat_mean=('lat_p50_ms', 'mean'),
        n=('ap50', 'count')
    ).reset_index()
    g = g[g['n'] >= 2].copy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # (a) Hist of cell-内 std side by side
    ax = axes[0]
    ax.hist(g['ap_std'], bins=30, alpha=0.6, color='#E74C3C', label='AP std (跨 32 D)')
    ax2 = ax.twiny()
    ax2.hist(g['lat_std'], bins=30, alpha=0.6, color='#3498DB', label='Lat std (ms)')
    ax.set_xlabel('AP cell-内 std')
    ax2.set_xlabel('Lat cell-内 std (ms)')
    ax.set_ylabel('cell count (147 cells)')
    ax.set_title(f'cell-内 std 分布对比\n'
                 f'AP median={g["ap_std"].median():.4f}, max={g["ap_std"].max():.3f}\n'
                 f'Lat median={g["lat_std"].median():.2f} ms, max={g["lat_std"].max():.2f}')
    ax.legend(loc='upper left'); ax2.legend(loc='upper right')

    # (b) AP std vs Lat std 散点 (cell-by-cell)
    ax = axes[1]
    ax.scatter(g['ap_std'], g['lat_std'], s=40, alpha=0.6, c='purple')
    ax.set_xlabel('AP cell-内 std')
    ax.set_ylabel('Lat cell-内 std (ms)')
    ax.set_title('cell-by-cell: AP vs Lat 内部 D 散度\n(每点 = 1 个 (B, Q) cell)')
    rho, p = scistats.spearmanr(g['ap_std'], g['lat_std'])
    ax.text(0.05, 0.95, f'Spearman ρ={rho:+.3f}\np={p:.3g}',
            transform=ax.transAxes, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat'))

    # (c) 跨 cell 总览 box: AP/Lat std normalized
    ax = axes[2]
    data = [g['ap_std'] / g['ap_mean'] * 100,
            g['lat_std'] / g['lat_mean'] * 100]
    ax.boxplot(data, labels=['AP CV%', 'Lat CV%'], showfliers=False)
    ax.set_ylabel('Coefficient of Variation (%)')
    ax.set_title('AP vs Lat cell-内 变化 (CV %)\nAP 几乎为 0, Lat ~10-20%')

    fig.suptitle('Lat vs AP: D 维度对两者的影响完全相反', fontsize=13)
    fig.tight_layout()
    fig.savefig(out / '08_lat_vs_ap_compare.png', dpi=130)
    plt.close(fig)
    print(f"  → 08_lat_vs_ap_compare.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out-dir', default=str(DEFAULT_OUT))
    args = p.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load_data()
    print(f"[lat figs] N = {len(df)} bench v1 anchors")
    print(f"  triplet: {df['triplet'].nunique()}, Q: {df['q_tag'].nunique()}, "
          f"D: {df['d_tag'].nunique()}")
    print(f"  lat: mean={df['lat_p50_ms'].mean():.2f} ms, "
          f"std={df['lat_p50_ms'].std():.2f}, "
          f"range=[{df['lat_p50_ms'].min():.2f}, {df['lat_p50_ms'].max():.2f}]\n")

    fig_01_marginal(df, out)
    fig_02_grid_bxq(df, out)
    fig_03_grid_bxd(df, out)
    fig_04_cell_std_bxq(df, out)
    fig_05_histogram(df, out)
    fig_06_by_axis_box(df, out)
    fig_07_lat_vs_planes(df, out)
    fig_08_lat_vs_ap_compare(df, out)

    df.to_csv(out / 'all_anchors_lat.csv', index=False)
    print(f"\n→ {out}/")


if __name__ == '__main__':
    main()
