"""Phase 1A.8 — 4090 端端到端 sanity 测试.

流程:
  1. 加载 RTX 4090 capability
  2. 跑随机搜索器 v0 → 50 个合法配置
  3. 用粗略 latency 估计器排序
  4. 落盘 results/phase1a_sanity.csv

验收标准 (Phase1_2 §2.4):
  - 5 分钟内出 50 个候选
  - 100% 合法
  - latency 估计有合理排序
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from framework.capability_schema import HardwareCapability
from framework.config_schema import Config, UNIV2X_MODULES
from framework.constraints import is_legal
from framework.latency_estimator import estimate_latency
from framework.searcher_v0 import random_search


def main(hw_name: str = "rtx4090", n: int = 50, seed: int = 42) -> None:
    print(f"=== Phase 1A.8 sanity test ===")
    print(f"硬件: {hw_name}")
    print(f"目标候选数: {n}")
    print(f"随机种子: {seed}")
    print()

    t0 = time.time()
    hw = HardwareCapability.from_yaml(ROOT / "configs" / "hardware" / f"{hw_name}.yaml")

    # 1. 随机搜索
    candidates, stats = random_search(
        hw, n_candidates=n, seed=seed, lock_d_to_gpu_only=True, verbose=True
    )
    t_search = time.time() - t0
    print(f"\n[search] {stats['n_candidates']} 个合法候选 / {stats['n_attempts']} 尝试 "
          f"/ pass_rate={stats['pass_rate']:.2%} / 用时 {t_search:.1f}s")

    # 2. 100% 合法性二次校验 (防止搜索器 bug)
    n_illegal = 0
    for c in candidates:
        ok, reason = is_legal(c, hw)
        if not ok:
            n_illegal += 1
            print(f"  [WARN] {c.config_id}: {reason}")
    assert n_illegal == 0, f"{n_illegal} 个非法候选漏过"
    print(f"[verify] 100% 合法 ({len(candidates)}/{len(candidates)})")

    # 3. latency 估计 + Pareto 风格排序 (按 latency 升序)
    rows = []
    for c in candidates:
        lat = estimate_latency(c, hardware=hw_name)
        rows.append({
            "config_id": c.config_id,
            "prune_object": c.prune_object,
            "prune_rate__avg": c.avg_prune_rate(),
            "prune_rate__backbone": c.prune_rate.get("backbone", 0.0),
            "prune_rate__encoder": c.prune_rate.get("encoder", 0.0),
            "prune_rate__decoder": c.prune_rate.get("decoder", 0.0),
            "q_bits__backbone": c.q_bits.get("backbone"),
            "q_bits__encoder": c.q_bits.get("encoder"),
            "q_bits__decoder": c.q_bits.get("decoder"),
            "q_bits__heads": c.q_bits.get("heads"),
            "q_granularity__encoder": c.q_granularity.get("encoder"),
            "q_object__encoder": c.q_object.get("encoder"),
            "d_routing__backbone": c.d_routing.get("backbone"),
            "has_dla": c.has_dla(),
            "estimated_latency_ms": lat,
        })

    df = pd.DataFrame(rows).sort_values("estimated_latency_ms").reset_index(drop=True)

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"phase1a_sanity_{hw_name}.csv"
    df.to_csv(out_csv, index=False)

    # 4. 摘要
    print(f"\n[latency] 估计区间: [{df['estimated_latency_ms'].min():.1f}, "
          f"{df['estimated_latency_ms'].max():.1f}] ms")
    print(f"[latency] 中位数: {df['estimated_latency_ms'].median():.1f} ms")
    print(f"\nTop-5 最快:")
    for _, r in df.head(5).iterrows():
        print(f"  {r['config_id']} | bits=({r['q_bits__backbone']}/{r['q_bits__encoder']}/"
              f"{r['q_bits__heads']}) prune={r['prune_object']}@{r['prune_rate__avg']:.2f} "
              f"→ {r['estimated_latency_ms']:.1f} ms")
    print(f"\nBottom-5 最慢:")
    for _, r in df.tail(5).iterrows():
        print(f"  {r['config_id']} | bits=({r['q_bits__backbone']}/{r['q_bits__encoder']}/"
              f"{r['q_bits__heads']}) prune={r['prune_object']}@{r['prune_rate__avg']:.2f} "
              f"→ {r['estimated_latency_ms']:.1f} ms")

    print(f"\n[output] {out_csv}")
    t_total = time.time() - t0
    print(f"\n[time] 总用时 {t_total:.1f}s (验收要求 < 300s) — "
          + ("PASS" if t_total < 300 else "FAIL"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hw", default="rtx4090", choices=["rtx4090", "orin_agx", "orin_nano"])
    parser.add_argument("-n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.hw, args.n, args.seed)
