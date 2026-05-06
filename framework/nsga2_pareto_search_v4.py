"""NSGA-II v4 — multi-objective Pareto search with D-space dimensions.

Search space (22 dims):
  19 prune+quant (locked: backbone=0)
  3 D-space:
    d_runtime ∈ {0, 1, 2, 3} = {pytorch_fp32, pytorch_fakequant, trt_fp16, trt_int8}
    d_pipelined_get_bevs ∈ {0, 1}
    d_temporal_cache_int8 ∈ {0, 1}

Two predictors:
  models/lgb_predictor_v4_amota.txt   (target: amota_v4)
  models/lgb_predictor_v4_latency.txt (target: est_trt_latency_v4_ms)
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM

ROOT = Path(__file__).resolve().parent.parent
LGB_AMOTA = lgb.Booster(model_file=str(ROOT / "models/lgb_predictor_v4_amota.txt"))
LGB_LAT = lgb.Booster(model_file=str(ROOT / "models/lgb_predictor_v4_latency.txt"))

# Original 19 prune+quant features
BASE_FEATURES = [
    "prune_rate__backbone", "prune_rate__encoder_ffn", "prune_rate__encoder_attn",
    "prune_rate__encoder_heads", "prune_rate__decoder_ffn", "prune_rate__decoder_attn",
    "prune_rate__decoder_heads", "prune_rate__heads_mid", "decoder_num_layers",
    "q_bits__global_w", "q_bits__global_a", "q_bits__backbone", "q_bits__encoder",
    "q_bits__decoder", "q_bits__heads", "q_bits__v2x_comm",
    "q_target", "q_granularity_w", "q_granularity_a",
]
# AMOTA predictor: 20 features = 19 + d_temporal_cache_int8
# Latency predictor: 21 features = 19 + d_runtime + d_pipelined_get_bevs

# Search bounds
LOW = np.array([
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # prune (backbone locked 0)
    3,                                           # decoder_num_layers
    4, 4, 4, 4, 4, 4, 4,                         # q_bits
    0, 0, 0,                                     # q_target / granularity
    0,                                            # d_runtime int 0..3
    0, 0,                                         # d_pipelined / d_temporal_cache
])
UP = np.array([
    0.0, 0.7, 0.5, 0.5, 0.7, 0.5, 0.5, 0.5,
    6,
    32, 32, 32, 32, 32, 32, 32,
    2, 1, 1,
    3,    # 4 runtime options
    1, 1,
])


def quantize(x):
    """Convert continuous NSGA-II vars to LightGBM categorical/discrete features."""
    f = x.copy()
    f[8] = int(round(f[8]))  # decoder_num_layers
    # q_bits: 4 / 8 / 16 / 32
    for i in range(9, 16):
        v = f[i]
        f[i] = 4 if v < 6 else 8 if v < 12 else 16 if v < 22 else 32
    f[16] = int(round(f[16]))  # q_target
    f[17] = int(round(f[17]))  # q_granularity_w
    f[18] = int(round(f[18]))  # q_granularity_a
    f[19] = int(round(f[19]))  # d_runtime
    f[20] = int(round(f[20]))  # d_pipelined
    f[21] = int(round(f[21]))  # d_temporal_cache
    return f


class CoopAccelV4Problem(Problem):
    def __init__(self):
        super().__init__(n_var=22, n_obj=2, n_ieq_constr=0, xl=LOW, xu=UP)

    def _evaluate(self, X, out, *args, **kwargs):
        results = np.zeros((len(X), 2), dtype=np.float64)
        for i, x in enumerate(X):
            x = quantize(x)
            # AMOTA features: 19 base + d_temporal_cache_int8
            amota_feat = np.append(x[:19], x[21]).reshape(1, -1)
            # Latency features: 19 base + d_runtime + d_pipelined
            lat_feat = np.append(x[:19], [x[19], x[20]]).reshape(1, -1)

            amota = float(LGB_AMOTA.predict(amota_feat)[0])
            latency = float(LGB_LAT.predict(lat_feat)[0])
            results[i, 0] = -amota
            results[i, 1] = latency

        out["F"] = results


def main():
    print("=== NSGA-II v4 Pareto Search ===")
    print("Search space: 22 dims (19 prune+quant + 3 D-space)")
    print("Predictors: lgb_predictor_v4_amota.txt + lgb_predictor_v4_latency.txt")

    problem = CoopAccelV4Problem()
    algorithm = NSGA2(
        pop_size=100, n_offsprings=50,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
    )
    print("\nRunning 200 generations × 100 individuals...")
    res = minimize(problem, algorithm, ("n_gen", 200), seed=42, verbose=False)

    pareto_X = res.X
    pareto_F = res.F
    print(f"\nPareto frontier: {len(pareto_X)} configs found")

    # Decode to readable
    rt_names = ["pytorch_fp32", "pytorch_fakequant", "trt_fp16", "trt_int8"]
    rows = []
    for i, x in enumerate(pareto_X):
        x = quantize(x)
        rows.append({
            "config_id": f"NSGA_v4_{i:03d}",
            "pred_amota": -pareto_F[i, 0],
            "pred_latency_ms": pareto_F[i, 1],
            "prune_rate__backbone": x[0],
            "prune_rate__encoder_ffn": x[1],
            "prune_rate__decoder_ffn": x[4],
            "prune_rate__heads_mid": x[7],
            "decoder_num_layers": int(x[8]),
            "q_bits__encoder": int(x[12]),
            "q_target": ["none", "W", "W+A"][int(x[16])],
            "d_runtime": rt_names[int(x[19])],
            "d_pipelined": bool(x[20]),
            "d_temporal_cache_int8": bool(x[21]),
        })

    df = pd.DataFrame(rows).sort_values("pred_latency_ms")
    out_csv = ROOT / "data/phase4/pareto_frontier_v4.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    # Print Pareto frontier extremes
    print("\n=== Pareto Frontier Extremes ===")
    print("\nFastest 5 (lowest latency):")
    print(df.head(5)[["config_id", "pred_amota", "pred_latency_ms",
                       "d_runtime", "d_pipelined", "d_temporal_cache_int8",
                       "prune_rate__encoder_ffn", "q_bits__encoder"]].to_string(index=False))
    print("\nMost accurate 5 (highest AMOTA):")
    print(df.nlargest(5, "pred_amota")[["config_id", "pred_amota", "pred_latency_ms",
                                          "d_runtime", "d_pipelined", "d_temporal_cache_int8",
                                          "prune_rate__encoder_ffn"]].to_string(index=False))

    # Compute speedup
    pytorch_baseline_lat = 515.5
    extreme = df.pred_latency_ms.min()
    print(f"\n=== Comparison vs PyTorch FP32 baseline (515.5 ms) ===")
    print(f"  v3 (no D-space) extreme latency: 400.8 ms (-22.3%)")
    print(f"  v4 (with D-space) extreme:       {extreme:.1f} ms (-{(1-extreme/pytorch_baseline_lat)*100:.1f}%) = {pytorch_baseline_lat/extreme:.2f}× speedup")

    # Save metrics
    out_json = ROOT / "results/phase4_nsga2_v4_metrics.json"
    with open(out_json, "w") as f:
        json.dump({
            "n_pareto": int(len(df)),
            "amota_min": float(df.pred_amota.min()),
            "amota_max": float(df.pred_amota.max()),
            "lat_min": float(df.pred_latency_ms.min()),
            "lat_max": float(df.pred_latency_ms.max()),
            "speedup_vs_pytorch": float(pytorch_baseline_lat / extreme),
        }, f, indent=2)


if __name__ == "__main__":
    main()
