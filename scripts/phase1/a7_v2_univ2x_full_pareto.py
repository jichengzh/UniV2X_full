"""A7 v2: univ2x_full 50 候选 Pareto 用 LGB v6 + rule-based lat.

vs A7 v1 (KNN over 3 sparse anchor 90/?/5640ms, lat × est_amota from LGB v3/v4):
  - amota 用 LGB v6 (跨模型, model_class='univ2x_full')
  - latency 用 rule-based: 4090 PyTorch FP32 e2e baseline 5640ms (含 R101 + DCN + Transformer decoder)
    剪枝 (DCN 剪不动, 其他可剪 30%) + 量化 (FP16/INT8)

输出:
  results/phase1a_pareto_v2.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from framework.capability_schema import HardwareCapability
from framework.config_schema import Config, UNIV2X_MODULES
from framework.searcher_v0 import random_search
from scripts.phase2.train_lgb_v6 import featurize_v6 as featurize

RESULTS_DIR = REPO_ROOT / "results"
LGB_AMOTA = REPO_ROOT / "models/lgb_v6_amota.txt"

# univ2x_full rule-based latency (PyTorch FP32 e2e ~5640ms, FP16 TRT ~90ms)
# 但 4090 端 PyTorch FP32 不实际使用; 用 baseline FP16 估计 ≈ 540ms (uniad_tiny 量级 + DCN overhead)
BASELINE_LAT_FP32 = 540.0   # PyTorch FP16 e2e (DCN 限制只到 FP16)
DCN_FRACTION = 0.4   # DCN 占总 lat ~40%, 不可剪不可量化
NON_DCN_BUDGET = BASELINE_LAT_FP32 * (1 - DCN_FRACTION)  # 324ms 可优化部分
BITS_FACTOR = {"FP32": 1.0, "FP16": 0.85, "INT8": 0.65}  # 非 DCN 部分


def cfg_to_predictor_row(cfg: Config) -> pd.Series:
    cd = cfg.to_dict()
    row = {
        "prune_object": cd["prune_object"],
        "d_pipelined": int(cd.get("d_pipelined", 0)),
        "d_temporal_cache_int8": int(cd.get("d_temporal_cache_int8", 0)),
        "d_runtime": cd.get("d_runtime", "pytorch_fp32"),
        "source": "synth_dspace",
        "model_class": "univ2x_full",  # ★
    }
    for m in UNIV2X_MODULES:
        row[f"prune_rate__{m}"] = float(cd["prune_rate"].get(m, 0.0))
        row[f"prune_criterion__{m}"] = cd["prune_criterion"].get(m, "none")
        row[f"q_bits__{m}"] = cd["q_bits"].get(m, "FP32")
        row[f"q_granularity__{m}"] = cd["q_granularity"].get(m, "none")
        row[f"q_object__{m}"] = cd["q_object"].get(m, "none")
        row[f"d_routing__{m}"] = cd["d_routing"].get(m, "GPU")
    return pd.Series(row)


def rule_based_lat(cfg: Config) -> float:
    """univ2x_full 特殊: backbone (R101+DCN) 占 DCN_FRACTION 不动, 其他可剪/可量化."""
    # 只算非 DCN 模块的 prune+quant
    rates = [cfg.prune_rate.get(m, 0.0) for m in ("encoder", "decoder", "heads", "v2x_comm")]
    avg_prune = sum(rates) / max(len(rates), 1)
    bits = [BITS_FACTOR.get(cfg.q_bits.get(m, "FP32"), 1.0) for m in ("encoder", "decoder", "heads", "v2x_comm")]
    avg_bits = sum(bits) / max(len(bits), 1)
    non_dcn_after = NON_DCN_BUDGET * (1 - avg_prune * 0.4) * avg_bits
    return BASELINE_LAT_FP32 * DCN_FRACTION + non_dcn_after  # DCN 部分不变


def pareto_mask(lat: np.ndarray, amota: np.ndarray) -> np.ndarray:
    n = len(lat)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]: continue
        for j in range(n):
            if i == j: continue
            if lat[j] <= lat[i] and amota[j] >= amota[i] and (lat[j] < lat[i] or amota[j] > amota[i]):
                is_pareto[i] = False
                break
    return is_pareto


def cfg_to_row(cfg: Config) -> dict:
    row = {"config_id": cfg.config_id, "source": "a7_v2_univ2x_full_pareto",
           "model_class": "univ2x_full", "prune_object": cfg.prune_object}
    for m in UNIV2X_MODULES:
        row[f"prune_rate__{m}"] = float(cfg.prune_rate.get(m, 0.0))
        row[f"prune_criterion__{m}"] = cfg.prune_criterion.get(m, "none")
        row[f"q_bits__{m}"] = cfg.q_bits.get(m, "FP32")
        row[f"q_granularity__{m}"] = cfg.q_granularity.get(m, "none")
        row[f"q_object__{m}"] = cfg.q_object.get(m, "none")
        row[f"d_routing__{m}"] = cfg.d_routing.get(m, "GPU")
    return row


def main(n: int = 50, seed: int = 42):
    print("=" * 60)
    print("A7 v2 — univ2x_full 50 候选 Pareto (LGB v6 + rule-based lat)")
    print("=" * 60)
    booster = lgb.Booster(model_file=str(LGB_AMOTA))

    # univ2x_full 用 4090 (hard alignment) — 默认 framework 设计
    rtx = HardwareCapability.from_yaml(REPO_ROOT / "configs/hardware/rtx4090.yaml")
    raw, stats = random_search(rtx, n_candidates=n*4, seed=seed, lock_d_to_gpu_only=True, verbose=False)
    print(f"✅ random_search: {len(raw)} raw, pass_rate={stats['pass_rate']:.1%}")
    cands = [cfg.with_field(config_id=f"a7_v2_univ2x_full_{i:04d}") for i, cfg in enumerate(raw[:n])]
    print(f"✅ Selected {len(cands)} candidates")

    df = pd.DataFrame([cfg_to_predictor_row(c) for c in cands])
    X, _ = featurize(df)
    pred_a = booster.predict(X)
    pred_l = np.array([rule_based_lat(c) for c in cands])
    is_pareto = pareto_mask(pred_l, pred_a)
    print(f"✅ Pareto {is_pareto.sum()} pts | amota [{pred_a.min():.3f}, {pred_a.max():.3f}] "
          f"| lat [{pred_l.min():.1f}, {pred_l.max():.1f}] ms")

    rows = []
    for i, cfg in enumerate(cands):
        r = cfg_to_row(cfg)
        r["predicted_amota"] = float(pred_a[i])
        r["predicted_lat_ms"] = float(pred_l[i])
        r["is_pareto"] = bool(is_pareto[i])
        rows.append(r)
    df = pd.DataFrame(rows)
    front = ["config_id", "model_class", "source", "is_pareto",
             "predicted_lat_ms", "predicted_amota", "prune_object"]
    cfg_cols = []
    for m in UNIV2X_MODULES:
        for p in ("prune_rate", "prune_criterion", "q_bits", "q_granularity", "q_object", "d_routing"):
            cfg_cols.append(f"{p}__{m}")
    df = df[[c for c in front + cfg_cols if c in df.columns]]
    out = RESULTS_DIR / "phase1a_pareto_v2.csv"
    df.to_csv(out, index=False)
    print(f"\n✅ Wrote {out}")
    pareto = df[df["is_pareto"]].sort_values("predicted_lat_ms")
    print(f"\n=== Pareto frontier ===")
    print(pareto[["config_id", "predicted_lat_ms", "predicted_amota", "prune_object",
                  "q_bits__encoder", "prune_rate__encoder", "prune_rate__decoder"]].to_string(index=False))


if __name__ == "__main__":
    main()
