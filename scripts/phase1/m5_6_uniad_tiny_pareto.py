"""M5.6: univ2x-tiny variant 50 候选 Pareto (latency × params 双轴).

复用 M4.7 流程, 切换 adapter:
  1. random_search 在 Orin AGX (soft) 上跑 50 候选 + uniad_tiny 物理约束过滤
  2. KNN (k=3) 估 latency 从 baseline_4090.parquet 中 model_class='uniad_tiny_variant' 35 行
  3. params 估算 (Pareto 第二轴, amota 待 M5.7 抽样实测/预测器代理)
  4. Pareto 标记 (minimize latency, minimize params)
  5. 输出 results/phase2_pareto_uniad_tiny.csv

注意:
  - uniad_tiny decoder 是 Transformer (Taylor/Wanda 默认准则即可, 不需 CNN remap)
  - v2x_comm 在 tiny variant 不存在 → adapter 过滤掉相关候选
  - 这是 framework 三模型 spectrum 的第二个 50-候选 Pareto, 与 M4.7 (Pyramid) 互补
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from framework.adapters import uniad_tiny as ut_adapter
from framework.capability_schema import HardwareCapability
from framework.config_schema import Config, UNIV2X_MODULES
from framework.searcher_v0 import random_search

BASELINE = REPO_ROOT / "data/baseline_4090.parquet"
RESULTS_DIR = REPO_ROOT / "results"

BITS_TO_NUM = {"FP32": 2.0, "FP16": 1.0, "INT8": 0.0}


def cfg_to_vector(cfg) -> np.ndarray:
    """11 维向量: 5 prune_rate + 5 q_bits 数值化 + 1 prune_object marker."""
    if isinstance(cfg, Config):
        cd = cfg.to_dict()
        prune = [float(cd["prune_rate"].get(m, 0.0)) for m in UNIV2X_MODULES]
        bits = [BITS_TO_NUM.get(cd["q_bits"].get(m, "FP32"), 2.0) for m in UNIV2X_MODULES]
        po = cd["prune_object"]
    else:
        cd = cfg
        prune = [float(cd.get(f"prune_rate__{m}", 0.0) or 0.0) for m in UNIV2X_MODULES]
        bits = [BITS_TO_NUM.get(cd.get(f"q_bits__{m}", "FP32"), 2.0) for m in UNIV2X_MODULES]
        po = cd.get("prune_object", "none") or "none"
    po_marker = {"none": 0.0, "channel": 1.0, "2:4": 2.0, "head": 3.0}.get(po, 0.0)
    return np.array(prune + bits + [po_marker], dtype=np.float64)


def baseline_uniad_tiny_vectors() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """从 baseline_4090.parquet 抽 model_class='uniad_tiny_variant' + lat/params 都齐全的行."""
    df = pd.read_parquet(BASELINE)
    df = df[
        (df["model_class"] == "uniad_tiny_variant")
        & df["lat_e2e_ms"].notna()
        & df["params_after_M"].notna()
    ].copy()
    print(f"  uniad_tiny_variant baseline (lat+params 齐): {len(df)} rows")
    print(f"    lat range: {df['lat_e2e_ms'].min():.2f} - {df['lat_e2e_ms'].max():.2f} ms")
    print(f"    params range: {df['params_after_M'].min():.2f} - {df['params_after_M'].max():.2f} M")

    vectors = np.stack([cfg_to_vector(row.to_dict()) for _, row in df.iterrows()])
    return (
        vectors,
        df["lat_e2e_ms"].to_numpy(),
        df["params_after_M"].to_numpy(),
        df["config_id"].tolist(),
    )


def knn_estimate(
    candidate: Config, bv: np.ndarray, blat: np.ndarray, bp: np.ndarray, bid: list[str], k: int = 3,
) -> dict:
    v = cfg_to_vector(candidate)
    dists = np.linalg.norm(bv - v, axis=1)
    nn_idx = np.argsort(dists)[:k]
    nn_dists = dists[nn_idx]
    weights = 1.0 / (nn_dists + 1e-6)
    weights /= weights.sum()
    return {
        "est_lat_4090_pytorch_ms": float((blat[nn_idx] * weights).sum()),
        "est_params_M": float((bp[nn_idx] * weights).sum()),
        "nn_ids": ",".join(bid[i] for i in nn_idx),
        "nn_distances": ",".join(f"{d:.3f}" for d in nn_dists),
    }


def pareto_mask_min_min(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pareto 前沿: minimize x, minimize y. 返回布尔数组."""
    n = len(x)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if x[j] <= x[i] and y[j] <= y[i] and (x[j] < x[i] or y[j] < y[i]):
                is_pareto[i] = False
                break
    return is_pareto


def cfg_to_row(cfg: Config) -> dict:
    row = {
        "config_id": cfg.config_id,
        "source": "m5_6_uniad_tiny_pareto",
        "model_class": "uniad_tiny_variant",
        "prune_object": cfg.prune_object,
    }
    for m in UNIV2X_MODULES:
        row[f"prune_rate__{m}"] = float(cfg.prune_rate.get(m, 0.0))
        row[f"prune_criterion__{m}"] = cfg.prune_criterion.get(m, "none")
        row[f"q_bits__{m}"] = cfg.q_bits.get(m, "FP32")
        row[f"q_granularity__{m}"] = cfg.q_granularity.get(m, "none")
        row[f"q_object__{m}"] = cfg.q_object.get(m, "none")
        row[f"d_routing__{m}"] = cfg.d_routing.get(m, "GPU")
    return row


def main(n_candidates: int = 50, seed: int = 42, k: int = 3) -> None:
    print("=" * 60)
    print("M5.6 — univ2x-tiny variant 50 候选 Pareto (latency × params)")
    print("=" * 60)

    # 1. capability + random_search
    # 用 Orin AGX (alignment="soft") 与 M4.7 一致, 让 channel 剪枝有机会通过.
    orin_agx = HardwareCapability.from_yaml(REPO_ROOT / "configs/hardware/orin_agx.yaml")
    print(f"✅ capability: {orin_agx.name} (alignment=soft)")

    candidates_raw, stats = random_search(
        orin_agx, n_candidates=n_candidates * 2, seed=seed,
        lock_d_to_gpu_only=True, verbose=False,
    )
    print(f"✅ random_search: {len(candidates_raw)} 候选 (raw)")
    print(f"   pass_rate={stats['pass_rate']:.1%}")

    # 2. uniad_tiny adapter: 物理约束过滤 (v2x_comm 不存在等)
    # 注意: uniad_tiny decoder 是 Transformer, 默认 Taylor/Wanda 准则就对, 不需 remap
    candidates = []
    filter_reasons = {}
    for cfg in candidates_raw:
        ok, reason = ut_adapter.is_valid_for_uniad_tiny(cfg)
        if not ok:
            tag = reason.split("]")[0].lstrip("[") if "]" in reason else reason[:30]
            filter_reasons[tag] = filter_reasons.get(tag, 0) + 1
            continue
        candidates.append(cfg.with_field(config_id=f"m5_6_uniad_tiny_{len(candidates):04d}"))
        if len(candidates) >= n_candidates:
            break
    print(f"✅ uniad_tiny adapter: {len(candidates)}/{len(candidates_raw)} 通过")
    if filter_reasons:
        print(f"   filter breakdown: {filter_reasons}")

    if len(candidates) == 0:
        print("❌ 0 候选通过 — 检查 random_search 输出 + adapter 约束")
        return

    # 3. KNN 估算 latency + params
    bv, blat, bp, bid = baseline_uniad_tiny_vectors()

    rows = []
    for cfg in candidates:
        row = cfg_to_row(cfg)
        est = knn_estimate(cfg, bv, blat, bp, bid, k=k)
        orin_fp16, unc_fp16 = ut_adapter.estimate_orin_latency(est["est_lat_4090_pytorch_ms"], "fp16")
        orin_int8, unc_int8 = ut_adapter.estimate_orin_latency(est["est_lat_4090_pytorch_ms"], "int8")
        row.update(est)
        row["est_orin_fp16_ms"] = orin_fp16
        row["est_orin_fp16_uncertainty_ms"] = unc_fp16
        row["est_orin_int8_ms"] = orin_int8
        row["est_orin_int8_uncertainty_ms"] = unc_int8
        row["amota_pending"] = "TBD: M5.7 抽样实测 + 精度预测器 v5.1 代理"
        rows.append(row)
    df = pd.DataFrame(rows)

    # 4. Pareto (minimize lat_4090, minimize params)
    is_pareto = pareto_mask_min_min(
        df["est_lat_4090_pytorch_ms"].to_numpy(),
        df["est_params_M"].to_numpy(),
    )
    df["is_pareto"] = is_pareto
    n_pareto = int(is_pareto.sum())

    # 5. 列顺序
    fixed = [
        "config_id", "model_class", "source", "is_pareto",
        "est_lat_4090_pytorch_ms", "est_params_M",
        "est_orin_fp16_ms", "est_orin_int8_ms",
        "est_orin_fp16_uncertainty_ms", "est_orin_int8_uncertainty_ms",
        "amota_pending",
        "nn_ids", "nn_distances", "prune_object",
    ]
    cfg_cols = []
    for m in UNIV2X_MODULES:
        for p in ("prune_rate", "prune_criterion", "q_bits", "q_granularity", "q_object", "d_routing"):
            cfg_cols.append(f"{p}__{m}")
    df = df[[c for c in fixed + cfg_cols if c in df.columns]]

    # 6. 输出
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "phase2_pareto_uniad_tiny.csv"
    df.to_csv(out, index=False)
    print(f"\n✅ Wrote {out}")
    print(f"   {len(df)} 候选, Pareto 前沿 {n_pareto} 点")

    # 7. Pareto 摘要
    pf = df[df["is_pareto"]].sort_values("est_lat_4090_pytorch_ms")
    print(f"\n=== Pareto frontier (sorted by est_lat_4090) ===")
    cols_show = ["config_id", "est_lat_4090_pytorch_ms", "est_params_M",
                 "est_orin_fp16_ms", "prune_object",
                 "q_bits__encoder", "prune_rate__encoder", "prune_rate__decoder"]
    print(pf[[c for c in cols_show if c in pf.columns]].to_string(index=False))

    # 8. 全样本统计
    print(f"\n=== Stats over all {len(df)} candidates ===")
    print(f"  est_lat_4090:    min={df['est_lat_4090_pytorch_ms'].min():.2f}  max={df['est_lat_4090_pytorch_ms'].max():.2f}  mean={df['est_lat_4090_pytorch_ms'].mean():.2f}")
    print(f"  est_params_M:    min={df['est_params_M'].min():.3f}  max={df['est_params_M'].max():.3f}  mean={df['est_params_M'].mean():.3f}")
    print(f"  est_orin_fp16:   min={df['est_orin_fp16_ms'].min():.2f}  max={df['est_orin_fp16_ms'].max():.2f}")
    print(f"  prune_object 分布: {df['prune_object'].value_counts().to_dict()}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--n_candidates", type=int, default=50)
    ap.add_argument("-s", "--seed", type=int, default=42)
    ap.add_argument("-k", "--knn_k", type=int, default=3)
    args = ap.parse_args()
    main(n_candidates=args.n_candidates, seed=args.seed, k=args.knn_k)
