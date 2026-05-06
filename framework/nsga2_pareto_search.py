"""NSGA-II multi-objective Pareto search using LightGBM v3 predictors.

Search space (19 dims, matching v3 CSV features):
  Pruning (continuous, 0..max):
    prune_rate__backbone:    0.0  (locked, DCN constraint)
    prune_rate__encoder_ffn: 0.0..0.7
    prune_rate__encoder_attn: 0.0..0.5
    prune_rate__encoder_heads: 0.0..0.5
    prune_rate__decoder_ffn: 0.0..0.7
    prune_rate__decoder_attn: 0.0..0.5
    prune_rate__decoder_heads: 0.0..0.5
    prune_rate__heads_mid:   0.0..0.5
    decoder_num_layers:      3..6 (int)

  Quantization (discrete):
    q_bits__global_w/a, backbone, encoder, decoder, heads, v2x_comm: {32, 16, 8, 4}
    q_target: {none, W, W+A}
    q_granularity_w/a: {per_tensor, per_channel}

Objectives (both minimize):
  -AMOTA   (so larger AMOTA = lower negated value)
  +est_trt_latency_ms

Output:
  data/phase4/pareto_frontier.csv  (Pareto configs from NSGA-II)
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
LGB_AMOTA = lgb.Booster(model_file=str(ROOT / "models/lgb_predictor_v3.txt"))
LGB_LAT = lgb.Booster(model_file=str(ROOT / "models/lgb_predictor_v3_latency.txt"))
V3_CSV = ROOT / "data/phase4/stage5_baseline_v3.csv"

# Feature list (must match training schema exactly)
FEATURES = [
    "prune_rate__backbone",
    "prune_rate__encoder_ffn",
    "prune_rate__encoder_attn",
    "prune_rate__encoder_heads",
    "prune_rate__decoder_ffn",
    "prune_rate__decoder_attn",
    "prune_rate__decoder_heads",
    "prune_rate__heads_mid",
    "decoder_num_layers",
    "q_bits__global_w",
    "q_bits__global_a",
    "q_bits__backbone",
    "q_bits__encoder",
    "q_bits__decoder",
    "q_bits__heads",
    "q_bits__v2x_comm",
    "q_target",          # 0=none, 1=W, 2=W+A
    "q_granularity_w",   # 0=per_tensor, 1=per_channel
    "q_granularity_a",
]

# Bounds
LOW = np.array([
    0.0,  # backbone(locked)
    0.0, 0.0, 0.0,  # encoder ffn/attn/heads
    0.0, 0.0, 0.0,  # decoder ffn/attn/heads
    0.0,  # heads_mid
    3,    # decoder_num_layers
    4, 4, 4, 4, 4, 4, 4,  # q_bits (lower = INT4)
    0,  # q_target none
    0, 0,  # granularity
])
UP = np.array([
    0.0,  # backbone locked at 0(DCN constraint)
    0.7, 0.5, 0.5,
    0.7, 0.5, 0.5,
    0.5,
    6,
    32, 32, 32, 32, 32, 32, 32,
    2,  # q_target W+A
    1, 1,
])


def encode_categorical_index(features_dict: dict) -> dict:
    """Convert real-valued NSGA-II output to LightGBM-compatible features.

    decoder_num_layers / q_bits / q_target / q_granularity are categorical-int.
    NSGA-II provides them as floats; we need to round/quantize.
    """
    f = features_dict.copy()
    f["decoder_num_layers"] = int(round(f["decoder_num_layers"]))

    # Quantize q_bits to {4, 8, 16, 32}
    for k in ["q_bits__global_w", "q_bits__global_a", "q_bits__backbone",
              "q_bits__encoder", "q_bits__decoder", "q_bits__heads", "q_bits__v2x_comm"]:
        v = f[k]
        if v < 6:
            f[k] = 4
        elif v < 12:
            f[k] = 8
        elif v < 22:
            f[k] = 16
        else:
            f[k] = 32

    # q_target: 0..2 -> categorical int 0/1/2
    f["q_target"] = int(round(f["q_target"]))
    f["q_granularity_w"] = int(round(f["q_granularity_w"]))
    f["q_granularity_a"] = int(round(f["q_granularity_a"]))

    return f


def make_feature_vector(features_dict: dict) -> np.ndarray:
    """Build 19-dim feature vector in correct order."""
    return np.array([features_dict[k] for k in FEATURES], dtype=np.float64)


class CoopAccelProblem(Problem):
    """Two objectives:
    f1 = -AMOTA  (minimize -> maximize AMOTA)
    f2 = est_trt_latency_ms  (minimize)
    """

    def __init__(self):
        super().__init__(n_var=19, n_obj=2, n_ieq_constr=0, xl=LOW, xu=UP)

    def _evaluate(self, X, out, *args, **kwargs):
        # X: (pop_size, 19) array
        results = np.zeros((len(X), 2), dtype=np.float64)
        for i, x in enumerate(X):
            features = {k: x[j] for j, k in enumerate(FEATURES)}
            features = encode_categorical_index(features)
            vec = make_feature_vector(features).reshape(1, -1)

            amota = float(LGB_AMOTA.predict(vec)[0])
            latency = float(LGB_LAT.predict(vec)[0])
            results[i, 0] = -amota  # minimize -amota
            results[i, 1] = latency

        out["F"] = results


def main():
    print("=== NSGA-II Pareto Search ===")
    print(f"Search space: 19 dims")
    print(f"Predictors: lgb_predictor_v3.txt (AMOTA) + lgb_predictor_v3_latency.txt (latency)")

    problem = CoopAccelProblem()
    algorithm = NSGA2(
        pop_size=100,
        n_offsprings=50,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
    )

    print("\nRunning 200 generations × 100 individuals...")
    res = minimize(
        problem,
        algorithm,
        ("n_gen", 200),
        seed=42,
        verbose=False,
    )

    pareto_X = res.X    # Pareto-optimal configs (decision space)
    pareto_F = res.F    # Pareto-optimal objectives ([(-amota, latency), ...])

    print(f"\nPareto frontier: {len(pareto_X)} configs found")

    # Save Pareto configs to CSV
    rows = []
    for i, x in enumerate(pareto_X):
        features = {k: x[j] for j, k in enumerate(FEATURES)}
        features = encode_categorical_index(features)
        rows.append({
            **features,
            "pred_amota": -pareto_F[i, 0],
            "pred_latency_ms": pareto_F[i, 1],
            "config_id": f"NSGA_{i:03d}",
        })

    df = pd.DataFrame(rows).sort_values("pred_latency_ms")
    out_csv = ROOT / "data/phase4/pareto_frontier_nsga2.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved Pareto frontier: {out_csv}")

    # Print Pareto frontier summary
    print("\n=== Pareto Frontier (sorted by latency ASC) ===")
    print(df[["config_id", "pred_amota", "pred_latency_ms",
              "prune_rate__encoder_ffn", "prune_rate__decoder_ffn",
              "q_bits__global_w", "q_target"]].head(20).to_string(index=False))

    # Compare to current 23 configs
    print("\n=== Comparison: NSGA-II Pareto vs current 23 manual configs ===")
    current = pd.read_csv(V3_CSV)
    print(f"Current 23 configs amota range: [{current.amota.min():.4f}, {current.amota.max():.4f}]")
    print(f"Current 23 configs lat range:   [{current.est_trt_latency_ms.min():.1f}, {current.est_trt_latency_ms.max():.1f}] ms")
    print(f"NSGA-II Pareto amota range:     [{df.pred_amota.min():.4f}, {df.pred_amota.max():.4f}]")
    print(f"NSGA-II Pareto lat range:       [{df.pred_latency_ms.min():.1f}, {df.pred_latency_ms.max():.1f}] ms")

    # Summary metrics
    metrics = {
        "n_pareto_configs": int(len(df)),
        "amota_min": float(df.pred_amota.min()),
        "amota_max": float(df.pred_amota.max()),
        "latency_min_ms": float(df.pred_latency_ms.min()),
        "latency_max_ms": float(df.pred_latency_ms.max()),
        "extreme_speedup_pct": float((current.est_trt_latency_ms.max() - df.pred_latency_ms.min()) / current.est_trt_latency_ms.max() * 100),
        "knee_amota_drop_pct": None,
        "n_generations": 200,
        "pop_size": 100,
    }
    out_json = ROOT / "results/phase4_nsga2_pareto_metrics.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics: {out_json}")
    print(f"\nExtreme speedup vs baseline: -{metrics['extreme_speedup_pct']:.1f}% e2e latency")


if __name__ == "__main__":
    main()
