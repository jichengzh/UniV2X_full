"""Stage 2.5 覆盖度分析 — 对比"22 行 baseline" vs "22+30 行加上主动采样"的特征方差变化.

不实际跑 PTQ, 只是把 active_samples_plan.csv 当作"假设的 30 行新数据",
合并到 baseline_unified, 看哪些原本方差≈0 的列被解锁.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from framework.feature_encoder import CATEGORICAL_COLS, encode_baseline_df


def col_signal(df: pd.DataFrame, col: str) -> dict:
    """单列的"信号强度"指标."""
    s = df[col]
    n_unique = int(s.nunique(dropna=False))
    if s.dtype.kind in "fi":
        var = float(s.var())
        signal = "DEAD" if var < 1e-6 else ("LOW" if var < 1e-3 else "OK")
        return dict(unique=n_unique, var=var, signal=signal)
    else:
        # 类别: 用最常见类的占比衡量倾斜度
        vc = s.value_counts(dropna=False)
        max_share = float(vc.iloc[0] / len(s))
        if n_unique == 1:
            signal = "DEAD"
        elif max_share > 0.85:
            signal = "LOW"
        else:
            signal = "OK"
        return dict(unique=n_unique, var=max_share, signal=signal)


def main() -> None:
    print("=" * 70)
    print("Stage 2.5 — 覆盖度分析(对比假设主动采样后的特征方差)")
    print("=" * 70)

    # 22 行 baseline
    base_df = pd.read_parquet(ROOT / "data" / "phase1" / "baseline_unified.parquet")
    X_base = encode_baseline_df(base_df)
    print(f"\n[baseline] {len(X_base)} 行 / {len(X_base.columns)} 列")

    # 30 行 active samples (来自 plan CSV)
    plan = pd.read_csv(ROOT / "data" / "phase2" / "active_samples_plan.csv")
    # plan 字段映射到 encode_baseline_df 期望的字段
    active_rows = []
    for _, r in plan.iterrows():
        active_rows.append({
            "config_id": f"active_{r['id']}",
            "source": "active_sample",
            "prune_object": r["prune_object"],
            "prune_rate__backbone": r["prune_rate__backbone"],
            "prune_rate__encoder": r["prune_rate__encoder"],
            "prune_rate__decoder": r["prune_rate__decoder"],
            "prune_rate__heads": r["prune_rate__heads"],
            "prune_rate__v2x_comm": r["prune_rate__v2x_comm"],
            "q_bits__backbone": r["q_bits__backbone"],
            "q_bits__encoder": r["q_bits__encoder"],
            "q_bits__decoder": r["q_bits__decoder"],
            "q_bits__heads": r["q_bits__heads"],
            "q_bits__v2x_comm": r["q_bits__v2x_comm"],
            "q_granularity__encoder": r["q_granularity__encoder"],
            "q_object__encoder": r["q_object__encoder"],
            "prune_criterion__encoder": "L1",
            "d_routing__backbone": r["d_routing__backbone"],
        })
    active_df = pd.DataFrame(active_rows)
    X_active = encode_baseline_df(active_df)

    # 合并(模拟 Stage 2.5 完成后)
    X_after = pd.concat(
        [X_base, X_active], ignore_index=True
    )
    # 类别列的 union 必须重新设置为 category
    for c in CATEGORICAL_COLS:
        if c in X_after.columns:
            X_after[c] = X_after[c].astype(object).astype("category")
    print(f"[after Stage 2.5] {len(X_after)} 行")

    # ---------- 关键:逐列对比 signal 状态 ----------
    print("\n" + "-" * 70)
    print(f"{'feature':40s}  {'before':12s}  {'after':12s}  unlock?")
    print("-" * 70)

    unlocked: list = []
    upgraded: list = []
    still_dead: list = []
    no_change: list = []

    for col in X_base.columns:
        b = col_signal(X_base, col)
        a = col_signal(X_after, col)
        before = b["signal"]
        after = a["signal"]
        change = ""
        if before == "DEAD" and after != "DEAD":
            change = "✅ UNLOCK"
            unlocked.append(col)
        elif before == "LOW" and after == "OK":
            change = "🟢 STRONGER"
            upgraded.append(col)
        elif before == "DEAD" and after == "DEAD":
            change = "❌ STILL DEAD"
            still_dead.append(col)
        else:
            change = "—"
            no_change.append(col)

        # 只打印有变化的或仍 dead 的
        if change != "—":
            print(f"  {col:38s}  {before:12s}  {after:12s}  {change}")

    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print(f"✅ 新解锁的列 (DEAD → OK/LOW): {len(unlocked)}")
    for c in unlocked:
        print(f"     - {c}")
    print(f"\n🟢 信号增强的列 (LOW → OK):    {len(upgraded)}")
    for c in upgraded:
        print(f"     - {c}")
    print(f"\n❌ 仍然 DEAD 的列:             {len(still_dead)}")
    for c in still_dead:
        print(f"     - {c}")
    print(f"\n— 无变化:                      {len(no_change)}")

    # 关键派生特征对比
    print("\n" + "-" * 70)
    print("关键派生特征 — 解锁前后对比")
    print("-" * 70)
    for c in ["prune_rate__std", "prune_rate__min", "compound_pressure",
              "q_bits_num__std", "criterion_diversity", "is_dla_routed"]:
        if c not in X_base.columns:
            continue
        b_var = X_base[c].var() if X_base[c].dtype.kind in "fi" else 0
        a_var = X_after[c].var() if X_after[c].dtype.kind in "fi" else 0
        ratio = (a_var / b_var) if b_var > 1e-9 else float("inf")
        print(f"  {c:30s}  before var={b_var:.4f}  after var={a_var:.4f}  ratio={ratio:.1f}×")


if __name__ == "__main__":
    main()
