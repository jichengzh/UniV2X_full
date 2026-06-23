"""M4.9 v2 — Framework closed-loop search demo on Pyramid_DAIR_m1.

Pipeline (区别于 v1 hand-picked anchors):

    1. random_search() 产生 N raw candidates (Pyramid-scoped:
       backbone-only prune × global precision)
    2. propagate hard constraints + 经验约束 (incl. NEW resnext_width_pow2)
    3. dedup by signature → unique legal candidates
    4. predict (lat, AP) using:
         - Pyramid-tuned rule-based latency model (基线 5.71 ms FP32 e2e)
         - Pyramid empirical AP model (基于 M4.8 实测)
    5. 计算 predicted Pareto frontier
    6. real-measure top-K Pareto candidates that have available ckpts
       (prune_rate ∈ {0, 0.5}, backbone-only)
    7. 比对 predicted vs measured + 保存 demo CSV

Search space (stress-tests resnext_width_pow2 constraint):
    backbone prune_rate ∈ {0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75}
    global precision    ∈ {FP32, FP16, INT8}
    others (encoder/decoder/heads/v2x_comm) prune = 0 to satisfy gradient
    所有 module q_bits 一致 (因为 measure_pyramid 不支持 per-module mixed)
    → 7 × 3 = 21 raw configurations
"""
from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from framework.capability_schema import HardwareCapability, IPCapability, Features, Alignment
from framework.config_schema import Config, UNIV2X_MODULES
from framework.constraints import is_legal_for_hardware, list_violations


# Pyramid-specific search space
PRUNE_RATES = (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75)
PRECISIONS = ("FP32", "FP16", "INT8")


def make_pyramid_config(prune_rate: float, precision: str) -> Config:
    """Pyramid-scoped: only backbone gets pruned, all modules same precision."""
    M = UNIV2X_MODULES
    return Config(
        prune_object="channel" if prune_rate > 0 else "none",
        prune_rate={m: (prune_rate if m == "backbone" else 0.0) for m in M},
        prune_criterion={m: ("L1" if m == "backbone" and prune_rate > 0 else "none") for m in M},
        q_bits={m: precision for m in M},
        q_granularity={m: ("per-channel" if precision != "FP32" else "none") for m in M},
        q_object={m: ("W+A" if precision != "FP32" else "none") for m in M},
        d_routing={m: "GPU" for m in M},
        config_id=f"prune{int(prune_rate*1000):03d}_{precision}",
    )


# ============================================================
# Pyramid-tuned predictors (rule-based, anchored on M4.8 measurements)
# ============================================================

PYRAMID_BASELINE_LAT_MS = 5.713  # PT FP32 e2e collab N=2

TRT_SPEEDUP = {
    "FP32": 1.0,    # PyTorch fallback only
    "FP16": 4.50,   # A1: 5.713 / 1.270
    "INT8": 7.06,   # A2: 5.713 / 0.809
}

PRUNE_EXTRA_SPEEDUP = {
    0.0: 1.00,
    0.5: 1.04,    # A4 / A2 = 0.809 / 0.776
    0.75: 1.10,   # extrapolated
}

PRUNE_AP_DELTA_PP = {
    0.0: 0.0,
    0.5: -3.77,   # measured A4 vs A0
    0.75: -8.0,   # extrapolated
}

# ----------------------------------------------------------------
# P0.6 — LGB lat predictor (replaces rule-based for non-FP32 configs).
# Hold-out spearman 0.86 / MAE 0.045 ms (5.8% relative).
# ----------------------------------------------------------------
_LGB_PYRAMID = None
_LGB_PATH = REPO_ROOT / "models/lgb_pyramid_lat.txt"


def _get_lgb():
    global _LGB_PYRAMID
    if _LGB_PYRAMID is None and _LGB_PATH.exists():
        import lightgbm as lgb
        _LGB_PYRAMID = lgb.Booster(model_file=str(_LGB_PATH))
        print(f"[predictor] loaded LGB from {_LGB_PATH.name}")
    return _LGB_PYRAMID


def _cfg_to_stage_planes(cfg: Config) -> tuple[int, int, int]:
    """Map Config (uniform prune rate) → 3-stage planes for LGB feature.
    Pyramid baseline planes = (64, 128, 256). Uniform rate r → (64*(1-r), 128*(1-r), 256*(1-r)).
    Returns nearest valid power-of-2-snapped triplet.
    """
    rate = float(cfg.prune_rate.get("model", cfg.prune_rate.get("backbone", 0.0)))
    s0 = max(8, int(round(64 * (1 - rate) / 8)) * 8)
    s1 = max(8, int(round(128 * (1 - rate) / 8)) * 8)
    s2 = max(8, int(round(256 * (1 - rate) / 8)) * 8)
    return s0, s1, s2


def predict_lat_ms_lgb(cfg: Config) -> float:
    """LGB-based lat predictor (P0.6). Uses 5 features: stage{0,1,2}_planes + precision."""
    import numpy as np
    bits = set(cfg.q_bits.values())
    precision = "INT8" if "INT8" in bits else ("FP16" if "FP16" in bits else "FP32")
    if precision == "FP32":
        # LGB trained only on FP16/INT8 TRT engines. FP32 = PyTorch fallback (no TRT)
        return PYRAMID_BASELINE_LAT_MS
    booster = _get_lgb()
    if booster is None:
        return predict_lat_ms_rule(cfg)
    s0, s1, s2 = _cfg_to_stage_planes(cfg)
    x = np.array([[s0, s1, s2,
                   1 if precision == "FP16" else 0,
                   1 if precision == "INT8" else 0]], dtype=np.float32)
    return float(booster.predict(x)[0])


def predict_lat_ms_rule(cfg: Config) -> float:
    """Original rule-based lat predictor (kept for fallback + comparison)."""
    bits = set(cfg.q_bits.values())
    if "INT8" in bits: precision = "INT8"
    elif "FP16" in bits: precision = "FP16"
    else: precision = "FP32"
    backbone_rate = float(cfg.prune_rate.get("model",
                          cfg.prune_rate.get("backbone", 0.0)))
    grid_rate = min(PRUNE_EXTRA_SPEEDUP.keys(), key=lambda x: abs(x - backbone_rate))
    return PYRAMID_BASELINE_LAT_MS / (TRT_SPEEDUP[precision] * PRUNE_EXTRA_SPEEDUP[grid_rate])


def predict_lat_ms(cfg: Config) -> float:
    """Public API — defaults to LGB if available, else rule-based."""
    if _LGB_PATH.exists():
        return predict_lat_ms_lgb(cfg)
    return predict_lat_ms_rule(cfg)


def predict_ap50(cfg: Config) -> float:
    BASE = 0.7907
    backbone_rate = float(cfg.prune_rate.get("backbone", 0.0))
    grid_rate = min(PRUNE_AP_DELTA_PP.keys(), key=lambda x: abs(x - backbone_rate))
    return BASE + PRUNE_AP_DELTA_PP[grid_rate] / 100


# ============================================================
# Pareto frontier
# ============================================================

def pareto_filter(df: pd.DataFrame, lat_col: str, ap_col: str) -> pd.DataFrame:
    out_idx = []
    for i, r in df.iterrows():
        is_pareto = True
        for j, q in df.iterrows():
            if i == j: continue
            if (q[lat_col] <= r[lat_col] and q[ap_col] >= r[ap_col] and
                (q[lat_col] < r[lat_col] or q[ap_col] > r[ap_col])):
                is_pareto = False
                break
        if is_pareto:
            out_idx.append(i)
    return df.loc[out_idx].sort_values(lat_col).reset_index(drop=True)


# ============================================================
# Main
# ============================================================

def make_rtx4090() -> HardwareCapability:
    return HardwareCapability(
        name="NVIDIA RTX 4090",
        arch="Ada sm89",
        ips={"gpu": IPCapability(precisions=["FP32", "FP16", "INT8"], tensor_core_gen=4, sparse_tc=True)},
        features=Features(tensor_core=True, sparse_tc=True),
        alignment=Alignment(int8_channel=32, fp16_channel=16, alignment_enforcement="hard"),
    )


def main():
    OUT_DIR = REPO_ROOT / "results"
    print("=" * 76)
    print("M4.9 v2 — Framework Closed-Loop Search Demo (Pyramid_DAIR_m1, RTX 4090)")
    print("=" * 76)

    hw = make_rtx4090()
    print(f"\n[Stage 1] Enumerate Pyramid search space ({len(PRUNE_RATES)} prune × {len(PRECISIONS)} prec = "
          f"{len(PRUNE_RATES)*len(PRECISIONS)} raw configs)")

    raw_rows = []
    for rate, prec in product(PRUNE_RATES, PRECISIONS):
        c = make_pyramid_config(rate, prec)
        ok, why = is_legal_for_hardware(c, hw)
        violations = [(k, n, r) for k, n, r in list_violations(c, hw)
                      if k in ("physical", "empirical")]
        rejection_tags = ";".join(f"{k}/{n}" for k, n, _ in violations) if violations else ""
        raw_rows.append({
            "config_id": c.config_id,
            "backbone_prune": rate,
            "precision_global": prec,
            "is_legal": ok,
            "rejection": rejection_tags,
        })

    raw_df = pd.DataFrame(raw_rows)
    raw_path = OUT_DIR / "m4_9_closed_loop_v2_raw.csv"
    raw_df.to_csv(raw_path, index=False)
    print(f"  saved -> {raw_path}")
    print(f"\n  legal / total: {raw_df['is_legal'].sum()} / {len(raw_df)}")
    print(f"  rejection breakdown:")
    for _, r in raw_df[~raw_df['is_legal']].iterrows():
        print(f"    [{r['config_id']:25}] {r['rejection']}")

    print(f"\n[Stage 2] Predict (lat, AP) for {raw_df['is_legal'].sum()} legal candidates")
    legal_df = raw_df[raw_df["is_legal"]].copy()
    legal_df["predicted_lat_ms"] = legal_df.apply(
        lambda r: round(predict_lat_ms(make_pyramid_config(r["backbone_prune"], r["precision_global"])), 4),
        axis=1)
    legal_df["predicted_ap50"] = legal_df.apply(
        lambda r: round(predict_ap50(make_pyramid_config(r["backbone_prune"], r["precision_global"])), 4),
        axis=1)
    cand_path = OUT_DIR / "m4_9_closed_loop_v2_candidates.csv"
    legal_df.to_csv(cand_path, index=False)
    print(legal_df[["config_id", "backbone_prune", "precision_global",
                    "predicted_lat_ms", "predicted_ap50"]].to_string(index=False))

    print(f"\n[Stage 3] Pareto-filter (min lat, max AP)")
    pareto = pareto_filter(legal_df, "predicted_lat_ms", "predicted_ap50")
    pareto_path = OUT_DIR / "m4_9_closed_loop_v2_pareto.csv"
    pareto.to_csv(pareto_path, index=False)
    print(f"  {len(pareto)} Pareto-optimal predicted candidates:")
    print(pareto[["config_id", "backbone_prune", "precision_global",
                  "predicted_lat_ms", "predicted_ap50"]].to_string(index=False))

    print(f"\n[Stage 4] Real-measure validation (cached M4.8 anchors)")
    measured_map = {
        ("FP16", 0.0): ("A1_trt_fp16_collab",       1.270, 0.7909),
        ("INT8", 0.0): ("A2_trt_int8_collab",       0.809, 0.7910),
        ("FP16", 0.5): ("A3_pruned50_ft_fp16",      1.009, 0.7644),
        ("INT8", 0.5): ("A4_pruned50_ft_int8",      0.776, 0.7530),
        ("FP32", 0.0): ("A0_pytorch_fp32_baseline", 5.713, 0.7907),
    }

    measured_rows = []
    for _, r in pareto.iterrows():
        key = (r["precision_global"], r["backbone_prune"])
        if key in measured_map:
            tag, lat_meas, ap_meas = measured_map[key]
            measured_rows.append({
                "config_id": r["config_id"],
                "anchor_real": tag,
                "precision": key[0],
                "prune": key[1],
                "predicted_lat_ms": r["predicted_lat_ms"],
                "measured_lat_ms": lat_meas,
                "lat_err_ms": round(r["predicted_lat_ms"] - lat_meas, 3),
                "lat_rel_err_pct": round(100 * (r["predicted_lat_ms"] - lat_meas) / lat_meas, 1),
                "predicted_ap50": r["predicted_ap50"],
                "measured_ap50": ap_meas,
                "ap50_err_pp": round(100 * (r["predicted_ap50"] - ap_meas), 2),
            })

    if measured_rows:
        mdf = pd.DataFrame(measured_rows)
        mdf_path = OUT_DIR / "m4_9_closed_loop_v2_measured.csv"
        mdf.to_csv(mdf_path, index=False)
        print(f"\n  saved -> {mdf_path}")
        print(mdf.to_string(index=False))
        print(f"\n  lat MAE = {mdf['lat_err_ms'].abs().mean():.3f} ms ({mdf['lat_rel_err_pct'].abs().mean():.1f}% rel)")
        print(f"  AP50 MAE = {mdf['ap50_err_pp'].abs().mean():.3f} pp")

    print(f"\n[Stage 5] Constraint-driven space reduction")
    illegal_pow2 = raw_df[raw_df["rejection"].str.contains("resnext_width_pow2", na=False)]
    print(f"  {len(illegal_pow2)} / {len(raw_df)} configs rejected by resnext_width_pow2")
    print(f"  这些配置预测会比 prune50 INT8 慢 3.35× — framework 在 search 阶段提前过滤,")
    print(f"  避免浪费 build engine + AP eval ({len(illegal_pow2) * 5} 分钟节省)")
    print()
    print(f"  rejected prune rates: {sorted(illegal_pow2['backbone_prune'].unique().tolist())}")
    print(f"  surviving prune rates: {sorted(legal_df['backbone_prune'].unique().tolist())}")


if __name__ == "__main__":
    main()
