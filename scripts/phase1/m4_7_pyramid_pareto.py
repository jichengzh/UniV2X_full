"""M4.7: Pyramid Fusion 50 候选 Pareto (latency × params 双轴).

复用 A7 流程 (random_search + KNN 估算) + pyramid_fusion adapter:
  1. random_search 在 4090 capability 上跑 50 候选 (Pyramid 物理约束过滤)
  2. KNN (k=3) 估 latency 从 baseline_4090.parquet 中 model_class='pyramid_fusion' 8 行
  3. params 估算 (Pareto 第二轴, 替代 amota 因为 AP 待 M4.5)
  4. Pareto 标记 (minimize latency, minimize params)
  5. 输出 results/phase2_pareto_pyramid.csv

注意:
  - amota 列暂留 NaN (Pyramid AP 待 OPV2V-H 下完后跑 HEAL inference 回填, M4.5 完整版)
  - Pareto 沿 latency × params 双轴 (vs A7 沿 latency × amota)
  - 这是 framework pipeline 通路 + Phase 2.5 minimal smoke test
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from framework.adapters import pyramid_fusion as pf_adapter
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


def baseline_pyramid_vectors() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """从 baseline_4090.parquet 抽 model_class='pyramid_fusion' 行 → (vec, lat, params, ids)."""
    df = pd.read_parquet(BASELINE)
    df = df[(df["model_class"] == "pyramid_fusion") & df["lat_e2e_ms"].notna()].copy()
    print(f"  Pyramid baseline: {len(df)} rows")
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
        "source": "m4_7_pyramid_pareto",
        "model_class": "pyramid_fusion",
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
    print("M4.7 — Pyramid Fusion 50 候选 Pareto (latency × params)")
    print("=" * 60)

    # 1. capability + random_search
    # 用 Orin AGX (alignment="soft"): 4090 hard alignment 把 channel 剪枝几乎全过滤,
    # Orin AGX 的 N2v2 实证 (TRT 8.5 implicit padding) 让 channels mod 32 经验规则失效,
    # 在 Orin 上软约束允许采样到剪枝变体. 这本身就是 v1.5 §0.3 经验硬约束的活案例.
    orin_agx = HardwareCapability.from_yaml(REPO_ROOT / "configs/hardware/orin_agx.yaml")
    print(f"✅ capability: {orin_agx.name} (alignment=soft, 利用 N2v2 实证)")

    candidates_raw, stats = random_search(
        orin_agx, n_candidates=n_candidates * 2, seed=seed,
        lock_d_to_gpu_only=True, verbose=False,  # Orin DLA 不参与 Pyramid 采样
    )
    print(f"✅ random_search: {len(candidates_raw)} 候选 (raw, 已过 capability 约束)")
    print(f"   pass_rate={stats['pass_rate']:.1%}")

    # 2. pyramid_fusion adapter: 准则映射 + 过滤
    # 关键: framework/searcher_v0 给 decoder 默认采 Taylor/Wanda (UniV2X Transformer 准则),
    # Pyramid decoder 是 CNN, 应自动 remap 到 L1/FPGM (论文 §4 C4 contribution 活案例)
    candidates = []
    pyramid_filter_reasons = {}
    n_remapped = 0
    for cfg in candidates_raw:
        # 自动 remap criterion: 把 UniV2X 默认的 Transformer 准则改成 Pyramid 用的 CNN 准则
        new_crit = dict(cfg.prune_criterion)
        for m in UNIV2X_MODULES:
            if cfg.prune_rate.get(m, 0.0) > 0:
                pool = pf_adapter.get_recommended_criterion_pool(m)
                if new_crit.get(m) not in pool and new_crit.get(m) != "none":
                    new_crit[m] = pool[0]  # 取第一个推荐 (默认 L1)
                    n_remapped += 1
        cfg_remapped = cfg.with_field(prune_criterion=new_crit)

        ok, reason = pf_adapter.is_valid_for_pyramid(cfg_remapped)
        if not ok:
            tag = reason.split("]")[0].lstrip("[") if "]" in reason else reason[:30]
            pyramid_filter_reasons[tag] = pyramid_filter_reasons.get(tag, 0) + 1
            continue
        candidates.append(cfg_remapped.with_field(config_id=f"m4_7_pyramid_{len(candidates):04d}"))
        if len(candidates) >= n_candidates:
            break
    print(f"✅ pyramid_fusion adapter: {len(candidates)}/{len(candidates_raw)} 通过 "
          f"({n_remapped} 处 criterion 自动 remap CNN→L1/FPGM)")
    if pyramid_filter_reasons:
        print(f"   filter breakdown: {pyramid_filter_reasons}")

    # 3. KNN 估算 latency + params
    bv, blat, bp, bid = baseline_pyramid_vectors()

    rows = []
    for cfg in candidates:
        row = cfg_to_row(cfg)
        est = knn_estimate(cfg, bv, blat, bp, bid, k=k)
        # 用 M2 f 把 4090 PyTorch latency 估到 Orin
        orin_fp16, unc_fp16 = pf_adapter.estimate_orin_latency(est["est_lat_4090_pytorch_ms"], "fp16")
        orin_int8, unc_int8 = pf_adapter.estimate_orin_latency(est["est_lat_4090_pytorch_ms"], "int8")
        row.update(est)
        row["est_orin_fp16_ms"] = orin_fp16
        row["est_orin_fp16_uncertainty_ms"] = unc_fp16
        row["est_orin_int8_ms"] = orin_int8
        row["est_orin_int8_uncertainty_ms"] = unc_int8
        row["amota_pending"] = "TBD: M4.5 OPV2V inference"  # 占位
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
    out = RESULTS_DIR / "phase2_pareto_pyramid.csv"
    df.to_csv(out, index=False)
    print(f"\n✅ Wrote {out}")
    print(f"   {len(df)} 候选, Pareto 前沿 {n_pareto} 点")

    # 7. Pareto 摘要
    pf = df[df["is_pareto"]].sort_values("est_lat_4090_pytorch_ms")
    print(f"\n=== Pareto frontier (latency × params 双轴, sorted by lat) ===")
    cols_show = ["config_id", "est_lat_4090_pytorch_ms", "est_params_M",
                 "est_orin_fp16_ms", "prune_object",
                 "q_bits__encoder", "prune_rate__encoder", "prune_rate__decoder"]
    print(pf[cols_show].to_string(index=False))

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
