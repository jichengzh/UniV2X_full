"""A7: 50 候选随机搜索 + baseline 接通 → results/phase1a_pareto.csv

对应 v1.5 §6.1 Phase 1A.7 — 闭环 Phase 1A.

流程:
1. 加载 RTX 4090 capability (configs/hardware/rtx4090.yaml)
2. random_search 50 候选 (lock_d_to_gpu_only=True; 4090 无 DLA)
3. 用 K-NN 在 baseline_4090.parquet 的真实样本上估算每个候选的 amota/latency
4. 输出 Pareto 前沿 → results/phase1a_pareto.csv

K-NN 距离设计 (粗糙但够 Phase 1A 雏形用; 真精度预测器在 Phase 2.4):
- 各 module prune_rate 差的 L2
- q_bits 数值化 (FP32=2/FP16=1/INT8=0) 之差的 L2
- prune_object 不同 +1 惩罚

输出 schema:
    config_id, source ('a7_random'),
    is_pareto (bool),
    est_amota, est_lat_e2e_ms, est_uncertainty (NN 方差),
    nn_ids (str, 用于回溯), nn_distances (str),
    + 全部 Config 字段 (与 baseline_4090.parquet schema 对齐)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from framework.capability_schema import HardwareCapability
from framework.config_schema import Config, UNIV2X_MODULES
from framework.searcher_v0 import random_search

# ------- K-NN 距离 -------

BITS_TO_NUM = {"FP32": 2.0, "FP16": 1.0, "INT8": 0.0}


def config_to_vector(cfg: Config | dict) -> np.ndarray:
    """把 Config 投影到 (5 prune + 5 bits + 1 prune_object_marker) = 11 维向量."""
    if isinstance(cfg, Config):
        d = cfg.to_dict()
    else:
        d = cfg
    prune = [float(d.get("prune_rate", {}).get(m, 0.0)) if "prune_rate" in d
             else float(d.get(f"prune_rate__{m}", 0.0)) for m in UNIV2X_MODULES]
    bits_field = d.get("q_bits") if "q_bits" in d else None
    if isinstance(bits_field, dict):
        bits = [BITS_TO_NUM.get(bits_field.get(m, "FP32"), 2.0) for m in UNIV2X_MODULES]
    else:
        bits = [BITS_TO_NUM.get(d.get(f"q_bits__{m}", "FP32"), 2.0) for m in UNIV2X_MODULES]
    po = d.get("prune_object", "none")
    po_marker = {"none": 0.0, "channel": 1.0, "2:4": 2.0, "head": 3.0}.get(po, 0.0)
    return np.array(prune + bits + [po_marker], dtype=np.float64)


def baseline_vectors(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """从 baseline df 抽出 (vectors, amota, latency, ids), 只保留有 amota 的行."""
    df_real = df[df["amota"].notna() & df["lat_e2e_ms"].notna()].copy()
    vectors = np.stack([config_to_vector(row.to_dict()) for _, row in df_real.iterrows()])
    return (
        vectors,
        df_real["amota"].to_numpy(),
        df_real["lat_e2e_ms"].to_numpy(),
        df_real["config_id"].tolist(),
    )


def knn_estimate(
    candidate: Config,
    baseline_vecs: np.ndarray,
    baseline_amota: np.ndarray,
    baseline_lat: np.ndarray,
    baseline_ids: list[str],
    k: int = 3,
) -> dict:
    """K-NN 估算 candidate 的 amota / lat. 返回均值 + 不确定度."""
    v = config_to_vector(candidate)
    dists = np.linalg.norm(baseline_vecs - v, axis=1)
    nn_idx = np.argsort(dists)[:k]
    nn_dists = dists[nn_idx]
    weights = 1.0 / (nn_dists + 1e-6)
    weights /= weights.sum()
    est_amota = float((baseline_amota[nn_idx] * weights).sum())
    est_lat = float((baseline_lat[nn_idx] * weights).sum())
    return {
        "est_amota": est_amota,
        "est_lat_e2e_ms": est_lat,
        "est_uncertainty": float(baseline_amota[nn_idx].std()),
        "nn_ids": ",".join(baseline_ids[i] for i in nn_idx),
        "nn_distances": ",".join(f"{d:.3f}" for d in nn_dists),
    }


# ------- Pareto 前沿 -------

def pareto_mask(amota: np.ndarray, latency: np.ndarray) -> np.ndarray:
    """maximize amota, minimize latency. 返回布尔数组."""
    n = len(amota)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if amota[j] >= amota[i] and latency[j] <= latency[i] and (
                amota[j] > amota[i] or latency[j] < latency[i]
            ):
                is_pareto[i] = False
                break
    return is_pareto


# ------- Config 转 row -------

def cfg_to_row(cfg: Config) -> dict:
    row = {"config_id": cfg.config_id, "source": "a7_random",
           "prune_object": cfg.prune_object}
    for m in UNIV2X_MODULES:
        row[f"prune_rate__{m}"] = float(cfg.prune_rate.get(m, 0.0))
        row[f"prune_criterion__{m}"] = cfg.prune_criterion.get(m, "none")
        row[f"q_bits__{m}"] = cfg.q_bits.get(m, "FP32")
        row[f"q_granularity__{m}"] = cfg.q_granularity.get(m, "none")
        row[f"q_object__{m}"] = cfg.q_object.get(m, "none")
        row[f"d_routing__{m}"] = cfg.d_routing.get(m, "GPU")
    return row


# ------- 主流程 -------

def main(n_candidates: int = 50, seed: int = 42, k: int = 3) -> None:
    print("=" * 60)
    print(f"A7 — Phase 1A.7: random_search × NN-baseline → Pareto")
    print("=" * 60)

    # 1) load capability
    rtx4090 = HardwareCapability.from_yaml(REPO_ROOT / "configs/hardware/rtx4090.yaml")
    print(f"✅ Loaded capability: {rtx4090.name}")

    # 2) random_search
    candidates, stats = random_search(
        rtx4090, n_candidates=n_candidates, seed=seed,
        lock_d_to_gpu_only=True, verbose=False,
    )
    print(f"✅ Sampled {len(candidates)} candidates after {stats['n_attempts']} attempts")
    print(f"   pass_rate={stats['pass_rate']:.1%}, propagation_modified={stats['propagation_modified_rate']:.1%}")
    if stats["illegal_breakdown"]:
        print(f"   illegal_breakdown: {dict(sorted(stats['illegal_breakdown'].items(), key=lambda x: -x[1]))}")

    # 3) baseline KNN
    baseline = pd.read_parquet(REPO_ROOT / "data/baseline_4090.parquet")
    bv, ba, bl, bid = baseline_vectors(baseline)
    print(f"✅ Baseline: {len(baseline)} rows total, {len(bv)} with amota+lat for KNN")

    rows = []
    for cfg in candidates:
        row = cfg_to_row(cfg)
        est = knn_estimate(cfg, bv, ba, bl, bid, k=k)
        row.update(est)
        rows.append(row)
    df = pd.DataFrame(rows)

    # 4) Pareto 标记
    is_pareto = pareto_mask(df["est_amota"].to_numpy(), df["est_lat_e2e_ms"].to_numpy())
    df["is_pareto"] = is_pareto
    n_pareto = int(is_pareto.sum())

    # 列顺序
    fixed = ["config_id", "source", "is_pareto",
             "est_amota", "est_lat_e2e_ms", "est_uncertainty",
             "nn_ids", "nn_distances", "prune_object"]
    cfg_cols = []
    for m in UNIV2X_MODULES:
        for p in ("prune_rate", "prune_criterion", "q_bits",
                  "q_granularity", "q_object", "d_routing"):
            cfg_cols.append(f"{p}__{m}")
    df = df[[c for c in fixed + cfg_cols if c in df.columns]]

    # 5) 输出
    out_dir = REPO_ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out_csv = out_dir / "phase1a_pareto.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n✅ Wrote {out_csv}")
    print(f"   rows = {len(df)}, Pareto frontier = {n_pareto}")

    # Pareto 前沿摘要
    pf = df[df["is_pareto"]].sort_values("est_lat_e2e_ms")
    print(f"\n=== Pareto frontier (sorted by latency ↑) ===")
    print(pf[["config_id", "est_amota", "est_lat_e2e_ms", "prune_object",
              "q_bits__encoder", "q_bits__heads", "prune_rate__encoder",
              "prune_rate__decoder"]].to_string(index=False))

    # 全样本统计
    print(f"\n=== Stats over all {len(df)} candidates ===")
    print(f"  est_amota:         min={df['est_amota'].min():.4f}  max={df['est_amota'].max():.4f}  mean={df['est_amota'].mean():.4f}")
    print(f"  est_lat_e2e_ms:    min={df['est_lat_e2e_ms'].min():.1f}  max={df['est_lat_e2e_ms'].max():.1f}  mean={df['est_lat_e2e_ms'].mean():.1f}")
    print(f"  prune_object 分布: {df['prune_object'].value_counts().to_dict()}")
    bits_set = set()
    for m in UNIV2X_MODULES:
        bits_set.update(df[f"q_bits__{m}"].unique())
    print(f"  q_bits 取值集合:   {sorted(bits_set)}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--n_candidates", type=int, default=50)
    ap.add_argument("-s", "--seed", type=int, default=42)
    ap.add_argument("-k", "--knn_k", type=int, default=3)
    args = ap.parse_args()
    main(n_candidates=args.n_candidates, seed=args.seed, k=args.knn_k)
