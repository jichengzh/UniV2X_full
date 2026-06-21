"""Closed-loop driving-score objective query helper for B4 co-design search.

Usage
-----
    from scripts.phase2.closedloop_objective_query import (
        driving_score,
        backbone_to_e2e_latency,
        config_driving_score,
    )

    # Direct latency → DS (latency-only model, beta=0)
    ds = driving_score(ap70=0.590, e2e_latency_ms=216.0)

    # H800 TVM backbone us → Orin e2e ms → DS
    e2e = backbone_to_e2e_latency(h800_backbone_us=6152, scale_tier="fp32")
    ds  = driving_score(ap70=0.590, e2e_latency_ms=e2e)

    # Config-label shortcut (uses gap1_grid_corrected.json anchors)
    ds  = config_driving_score("pad64", ap70=0.590, schedule="tuned")

Data sources
------------
- DS(tau_ms) model: fitted to multi_agent/figure/data/percinj_curve_full.csv
  (CoDriving, H800 V2Xverse, 18 points, N=105 per point, full-traffic).
- E2E latency anchors: results/E7_orin_e2e_baseline_vs_best.csv +
  results/E8_orin_e2e_fullchain.csv (Orin AGX, MODE_30W, 2-agent collab).
- Backbone latency grid: results/gap1_grid_corrected.csv (H800 TVM tuned).
- Model JSON: results/closedloop_objective_model.json

IMPORTANT CAVEATS
-----------------
1. All driving-score data is from **CoDriving**, not Pyramid.  Applied as
   same sensitivity-curve shape; absolute DS values may differ for Pyramid.
2. The 108-219ms e2e range (all tuned B4 configs except trap25/iso_s0) has
   DS spread of only ~3-7 points — within simulation noise (CI ±3-5 DS).
   The DS axis is most discriminating for trap25/iso_s0 (~535ms → DS≈87)
   vs pad64/base (~215ms → DS≈96).
3. AP70 → DS relationship has NO empirical data.  Default beta=0 (latency-
   only).  beta=0.5 adds a mild assumed correction; treat as sensitivity.
4. E2E latency for non-base configs is estimated via linear scaling from H800
   backbone (±30% rough estimate).  Real Orin TRT body measurements needed
   for authoritative values.
5. Do NOT cite raw DS numbers as "real closed-loop results".  Label as
   "model-estimated DS" in any report.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

# ---------------------------------------------------------------------------
# Model parameters (fitted to percinj_curve_full.csv)
# ---------------------------------------------------------------------------
_DS_MAX = 100.0
_TAU_MID_MS = 976.0       # logistic midpoint
_TAU_SCALE_MS = 247.0     # logistic scale (higher = slower decay)
_AP70_BASE = 0.631        # base Pyramid model AP70 (DAIR val)
_FIXED_OVERHEAD_MS = 87.5  # Orin 2-agent constant: pre-body + NMS (ms)

# Backbone scale factors: Orin TRT body ms per H800-TVM backbone us
# Calibrated at base-tuned (6320us → 131ms FP32, 47.99ms FP16, 20.02ms INT8)
_SCALE_MS_PER_US: dict[str, float] = {
    "fp32": 0.02073,
    "fp16": 0.007592,
    "int8": 0.003168,
}

# ---------------------------------------------------------------------------
# Gap1 config grid (H800 TVM tuned/default backbone latencies)
# keys match gap1_grid_corrected.json labels
# ---------------------------------------------------------------------------
_CONFIG_GRID: dict[str, dict[str, float]] = {
    "base":   {"default_us": 56321.0, "tuned_us": 6320.0,  "ap70": 0.631},
    "p50":    {"default_us": 26242.0, "tuned_us": 3011.0,  "ap70": 0.564},
    "p75":    {"default_us": 12689.0, "tuned_us": 484.0,   "ap70": 0.530},
    "trap25": {"default_us": 42355.0, "tuned_us": 21615.0, "ap70": 0.590},
    "pad64":  {"default_us": 47407.0, "tuned_us": 6152.0,  "ap70": 0.590},
    "iso_s0": {"default_us": 51249.0, "tuned_us": 21752.0, "ap70": 0.630},
    "iso_s1": {"default_us": 51622.0, "tuned_us": 6911.0,  "ap70": 0.634},
    "iso_s2": {"default_us": 51992.0, "tuned_us": 7244.0,  "ap70": 0.634},
}

# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    # exp(x) won't overflow for x < 0
    e = math.exp(x)
    return e / (1.0 + e)


def ds_from_latency(e2e_latency_ms: float) -> float:
    """DS(tau_ms) = 100 * sigmoid((tau_mid - tau_ms) / tau_scale).

    Fitted to percinj_curve_full.csv (CoDriving, H800).
    Valid range: 0 - 1500ms.  Saturates at ~98.1 for 0ms, ~47.7 at 1000ms.

    Args:
        e2e_latency_ms: End-to-end perception latency (ms) felt by ego.

    Returns:
        Predicted driving score [0, 100].
    """
    z = (_TAU_MID_MS - e2e_latency_ms) / _TAU_SCALE_MS
    return _DS_MAX * _sigmoid(z)


def driving_score(
    ap70: float,
    e2e_latency_ms: float,
    beta: float = 0.0,
) -> float:
    """Combined driving-score model: DS(ap70, e2e_latency_ms).

    DS = DS_lat(e2e_latency_ms) * (ap70 / AP70_BASE)^beta

    Args:
        ap70:           Perception AP70 of the config (DAIR val).
        e2e_latency_ms: End-to-end RSU perception latency (ms).
        beta:           AP70 sensitivity exponent. Default 0.0 = latency-only.
                        Use 0.5 for mild assumed AP70 correction (no real data).

    Returns:
        Predicted driving score [0, 100].
    """
    ds_lat = ds_from_latency(e2e_latency_ms)
    if beta == 0.0 or ap70 == _AP70_BASE:
        return ds_lat
    ap_factor = (ap70 / _AP70_BASE) ** beta
    return ds_lat * ap_factor


def backbone_to_e2e_latency(
    h800_backbone_us: float,
    scale_tier: str = "fp32",
) -> float:
    """Map H800 TVM backbone latency (microseconds) → Orin e2e latency (ms).

    Formula: e2e_ms = FIXED_OVERHEAD_MS + h800_backbone_us * scale_ms_per_us

    Calibrated on base-tuned anchor (6320us → 131ms FP32 Orin TRT body).
    APPROXIMATE: ±30% for non-base configs. Real Orin body measurements
    needed for authoritative values.

    Args:
        h800_backbone_us: H800 TVM backbone latency in microseconds (B1 LUT).
        scale_tier:       Precision tier: 'fp32', 'fp16', or 'int8'.
                          Use 'fp32' for B4 (TVM runs in FP32 on H800).

    Returns:
        Estimated Orin e2e perception latency (ms).
    """
    if scale_tier not in _SCALE_MS_PER_US:
        raise ValueError(f"scale_tier must be one of {list(_SCALE_MS_PER_US)}; got {scale_tier!r}")
    scale = _SCALE_MS_PER_US[scale_tier]
    return _FIXED_OVERHEAD_MS + h800_backbone_us * scale


def config_driving_score(
    config_label: str,
    ap70: float | None = None,
    schedule: str = "tuned",
    scale_tier: str = "fp32",
    beta: float = 0.0,
) -> dict[str, float]:
    """Convenience wrapper: config label → DS estimate.

    Args:
        config_label: One of: base, p50, p75, trap25, pad64, iso_s0, iso_s1, iso_s2.
        ap70:         Override AP70 (defaults to grid value).
        schedule:     'tuned' or 'default'.
        scale_tier:   'fp32', 'fp16', or 'int8'.
        beta:         AP70 sensitivity exponent (default 0.0 = latency-only).

    Returns:
        Dict with keys: e2e_ms, driving_score, ap70, h800_backbone_us.
    """
    if config_label not in _CONFIG_GRID:
        raise ValueError(
            f"Unknown config {config_label!r}. "
            f"Available: {sorted(_CONFIG_GRID)}"
        )
    row = _CONFIG_GRID[config_label]
    us_key = "tuned_us" if schedule == "tuned" else "default_us"
    h800_us = row[us_key]
    used_ap70 = ap70 if ap70 is not None else row["ap70"]
    e2e_ms = backbone_to_e2e_latency(h800_us, scale_tier=scale_tier)
    ds = driving_score(used_ap70, e2e_ms, beta=beta)
    return {
        "config_label": config_label,
        "schedule": schedule,
        "h800_backbone_us": h800_us,
        "e2e_ms": round(e2e_ms, 1),
        "ap70": used_ap70,
        "driving_score": round(ds, 2),
        "beta": beta,
        "scale_tier": scale_tier,
        "caveat": (
            "MODEL-ESTIMATED (CoDriving curve, H800 backbone scaling). "
            "Not a real Pyramid closed-loop result. ±30% e2e_ms for non-base configs."
        ),
    }


def b4_config_table(
    schedule: str = "tuned",
    scale_tier: str = "fp32",
    beta: float = 0.0,
) -> list[dict]:
    """Print a summary table of DS estimates for all B4 configs."""
    configs = ["base", "p50", "p75", "trap25", "pad64", "iso_s0", "iso_s1", "iso_s2"]
    return [config_driving_score(c, schedule=schedule, scale_tier=scale_tier, beta=beta)
            for c in configs]


# ---------------------------------------------------------------------------
# B4 plug-in helper: minimize objective including -DS
# ---------------------------------------------------------------------------

def ds_objective_for_b4(
    h800_backbone_us: float,
    ap70: float,
    scale_tier: str = "fp32",
    beta: float = 0.0,
) -> float:
    """Return -driving_score for minimization objective vector in B4.

    Designed for injection into CostModel.evaluate() as a 3rd objective:

        lat = self.lut.latency(width, sched)
        ap  = self.apm.ap70(width)
        ds  = -ds_objective_for_b4(lat, ap, scale_tier="fp32")  # maximize DS
        obj = (lat, -ap, -ds)   # all minimized

    Args:
        h800_backbone_us: From lut.latency(width, sched) in microseconds.
        ap70:             From apm.ap70(width).
        scale_tier:       'fp32' for B4 (TVM backbone on H800).
        beta:             AP70 exponent (0.0 = latency-only).

    Returns:
        -driving_score (negated for minimization).
    """
    e2e_ms = backbone_to_e2e_latency(h800_backbone_us, scale_tier=scale_tier)
    return -driving_score(ap70, e2e_ms, beta=beta)


# ---------------------------------------------------------------------------
# CLI / self-check
# ---------------------------------------------------------------------------

def _self_check() -> None:
    """Validate model against known curve points from percinj_curve_full.csv."""
    checks = [
        (0,    98.7, 98.1),
        (200,  95.0, 95.9),
        (500,  87.3, 87.3),
        (800,  66.4, 67.0),
        (1000, 47.7, 47.6),
    ]
    print("=== DS latency model self-check ===")
    print(f"{'tau_ms':>8} {'actual_DS':>10} {'predicted':>10} {'delta':>8}")
    for tau, actual, expected_approx in checks:
        pred = ds_from_latency(tau)
        print(f"{tau:>8} {actual:>10.1f} {pred:>10.2f} {pred-actual:>+8.2f}")
    print()

    print("=== B4 config table (schedule=tuned, FP32, beta=0) ===")
    print(f"{'config':>8} {'h800us':>8} {'e2e_ms':>8} {'ap70':>6} {'DS':>6}")
    for row in b4_config_table():
        print(f"{row['config_label']:>8} {row['h800_backbone_us']:>8.0f} "
              f"{row['e2e_ms']:>8.1f} {row['ap70']:>6.3f} {row['driving_score']:>6.2f}")
    print()

    print("=== E2E latency anchors cross-check ===")
    for label, real_e2e in [("base", 218.83), ("p75", 107.52)]:
        row = _CONFIG_GRID[label]
        est = backbone_to_e2e_latency(row["tuned_us"], "fp32")
        if label == "p75":
            est = backbone_to_e2e_latency(row["tuned_us"], "int8")
        real_str = f"{real_e2e:.1f}" if real_e2e else "N/A"
        print(f"  {label}: estimated={est:.1f}ms, real={real_str}ms "
              f"({'fp32' if label=='base' else 'int8'})")


if __name__ == "__main__":
    _self_check()
