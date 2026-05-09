"""M5.6 v2: uniad_tiny_variant 50 候选 Pareto 用 LGB v6 + rule-based lat.

vs M5.6 v1 (KNN over uniad_tiny baseline, lat × params):
  - amota 用 LGB v6 跨模型预测 (model_class='uniad_tiny_variant')
  - latency 用 rule-based: uniad_tiny baseline FP32 ~535ms (mean of 23 baseline rows)

输出:
  results/phase2_pareto_uniad_tiny_v2.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from framework.adapters import uniad_tiny as ut_adapter
from framework.capability_schema import HardwareCapability
from framework.config_schema import Config, UNIV2X_MODULES
from framework.searcher_v0 import random_search
from scripts.phase2.train_lgb_v6 import featurize_v6 as featurize

BASELINE = REPO_ROOT / "data/baseline_4090.parquet"
RESULTS_DIR = REPO_ROOT / "results"
LGB_AMOTA = REPO_ROOT / "models/lgb_v6_amota.txt"

# uniad_tiny rule-based latency (基于 baseline 23 行 lat range 502-562ms)
BASELINE_LAT_FP32 = 535.0  # 中位数
BITS_FACTOR = {"FP32": 1.0, "FP16": 0.95, "INT8": 0.85}  # uniad_tiny 没真实测 INT8, 估


def cfg_to_predictor_row(cfg: Config) -> pd.Series:
    cd = cfg.to_dict()
    row = {
        "prune_object": cd["prune_object"],
        "d_pipelined": int(cd.get("d_pipelined", 0)),
        "d_temporal_cache_int8": int(cd.get("d_temporal_cache_int8", 0)),
        "d_runtime": cd.get("d_runtime", "pytorch_fp32"),
        "source": "synth_dspace",
        "model_class": "uniad_tiny_variant",  # ★ v2 关键
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
    rates = [cfg.prune_rate.get(m, 0.0) for m in UNIV2X_MODULES]
    avg_prune = sum(rates) / len(rates)
    bits_factors = [BITS_FACTOR.get(cfg.q_bits.get(m, "FP32"), 1.0) for m in UNIV2X_MODULES]
    avg_bits = sum(bits_factors) / len(bits_factors)
    return BASELINE_LAT_FP32 * (1 - avg_prune * 0.4) * avg_bits


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
    row = {"config_id": cfg.config_id, "source": "m5_6_v2_uniad_tiny_pareto",
           "model_class": "uniad_tiny_variant", "prune_object": cfg.prune_object}
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
    print("M5.6 v2 — uniad_tiny 50 候选 Pareto (LGB v6 + rule-based lat)")
    print("=" * 60)
    booster = lgb.Booster(model_file=str(LGB_AMOTA))

    orin = HardwareCapability.from_yaml(REPO_ROOT / "configs/hardware/orin_agx.yaml")
    raw, stats = random_search(orin, n_candidates=n*2, seed=seed, lock_d_to_gpu_only=True, verbose=False)
    print(f"✅ random_search: {len(raw)} raw")

    cands = []
    filt = {}
    for cfg in raw:
        ok, reason = ut_adapter.is_valid_for_uniad_tiny(cfg)
        if not ok:
            tag = reason.split("]")[0].lstrip("[") if "]" in reason else reason[:30]
            filt[tag] = filt.get(tag, 0) + 1
            continue
        cands.append(cfg.with_field(config_id=f"m5_6_v2_uniad_tiny_{len(cands):04d}"))
        if len(cands) >= n: break
    print(f"✅ uniad_tiny adapter: {len(cands)}/{len(raw)} 通过. filter: {filt}")

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
    out = RESULTS_DIR / "phase2_pareto_uniad_tiny_v2.csv"
    df.to_csv(out, index=False)
    print(f"\n✅ Wrote {out}")
    pareto = df[df["is_pareto"]].sort_values("predicted_lat_ms")
    print(f"\n=== Pareto frontier ===")
    print(pareto[["config_id", "predicted_lat_ms", "predicted_amota", "prune_object",
                  "q_bits__encoder", "prune_rate__encoder", "prune_rate__decoder"]].to_string(index=False))


if __name__ == "__main__":
    main()
