"""
latency_lut_query.py  —  B1 LUT query helper

Reads results/latency_lut_pyramid.json and provides:
  lookup(s0, s1, s2, sched="tuned") -> latency_us  (float)

Convention:
  - sched in {"default", "tuned"}  (default = dlight, tuned = MetaSchedule 1000-trial)
  - s0 in {16, 32, 48, 64}  (stage-0 num_filters)
  - s1 in {32, 64, 96, 128} (stage-1 num_filters)
  - s2 in {64, 128, 192, 256} (stage-2 num_filters)
  - Widths outside the calibrated grid are linearly interpolated between nearest neighbours.
  - Returns float("nan") if a stage has no data at all.

Example:
  from scripts.phase2.latency_lut_query import lookup
  lat_us = lookup(48, 96, 192, "tuned")   # trap25
  lat_us = lookup(64, 96, 192, "tuned")   # pad64 (prediction)
"""
from __future__ import annotations
import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_LUT_PATH = _REPO_ROOT / "results/latency_lut_pyramid.json"

_cached_lut: dict | None = None
_cached_path: str | None = None


def _load(lut_path: str | Path | None = None) -> dict:
    global _cached_lut, _cached_path
    path = str(lut_path or _DEFAULT_LUT_PATH)
    if _cached_lut is None or _cached_path != path:
        with open(path) as f:
            _cached_lut = json.load(f)
        _cached_path = path
    return _cached_lut


def _interpolate(table: dict, width: int, key: str) -> float:
    """Linear interpolate on integer-keyed table (keys are stage widths as strings in JSON).

    Only includes table entries that actually contain `key` — prevents NaN propagation
    when a width has partial records (e.g. s0=16 has default but no tuned data).
    """
    # JSON keys are strings; only include entries that HAVE the requested key
    avail = {int(w): v for w, v in table.items()
             if v is not None and isinstance(v, dict) and v.get(key) is not None}
    if not avail:
        return float("nan")
    if width in avail:
        return avail[width].get(key, float("nan"))
    ws = sorted(avail.keys())
    lo = max((w for w in ws if w <= width), default=None)
    hi = min((w for w in ws if w >= width), default=None)
    if lo is None:
        return avail[hi].get(key, float("nan"))
    if hi is None:
        return avail[lo].get(key, float("nan"))
    alpha = (width - lo) / (hi - lo)
    v_lo = avail[lo].get(key, float("nan"))
    v_hi = avail[hi].get(key, float("nan"))
    if isinstance(v_lo, float) and isinstance(v_hi, float):
        return v_lo + alpha * (v_hi - v_lo)
    return float("nan")


def lookup(s0: int, s1: int, s2: int,
           sched: str = "tuned",
           lut_path: str | Path | None = None) -> float:
    """
    Predict latency in microseconds for Pyramid backbone with widths [s0, s1, s2].

    Args:
      s0: stage-0 num_filters (e.g. 64 for base)
      s1: stage-1 num_filters (e.g. 128 for base)
      s2: stage-2 num_filters (e.g. 256 for base)
      sched: "tuned" (MetaSchedule) or "default" (dlight)
      lut_path: override default LUT JSON path

    Returns:
      Predicted latency in microseconds (float).
      Returns float("nan") if any stage has no data.
    """
    lut = _load(lut_path)
    pt = lut["per_stage_tables"]
    base = pt["base_tuned_us"] if sched == "tuned" else pt["base_default_us"]
    delta_key = "delta_tuned_us" if sched == "tuned" else "delta_default_us"

    d0 = _interpolate(pt["s0"], s0, delta_key)
    d1 = _interpolate(pt["s1"], s1, delta_key)
    d2 = _interpolate(pt["s2"], s2, delta_key)

    if any(v != v for v in [d0, d1, d2]):  # NaN check
        return float("nan")
    return base + d0 + d1 + d2


def lookup_info(s0: int, s1: int, s2: int,
                sched: str = "tuned",
                lut_path: str | Path | None = None) -> dict:
    """Like lookup() but returns a dict with breakdown and interpolation flags."""
    lut = _load(lut_path)
    pt = lut["per_stage_tables"]
    base = pt["base_tuned_us"] if sched == "tuned" else pt["base_default_us"]
    delta_key = "delta_tuned_us" if sched == "tuned" else "delta_default_us"

    d0 = _interpolate(pt["s0"], s0, delta_key)
    d1 = _interpolate(pt["s1"], s1, delta_key)
    d2 = _interpolate(pt["s2"], s2, delta_key)
    total = base + d0 + d1 + d2 if all(v == v for v in [d0, d1, d2]) else float("nan")

    return {
        "s0": s0, "s1": s1, "s2": s2, "sched": sched,
        "pred_us": round(total, 1),
        "base_us": base,
        "delta_s0_us": round(d0, 1),
        "delta_s1_us": round(d1, 1),
        "delta_s2_us": round(d2, 1),
    }


def additivity_verdict(lut_path: str | Path | None = None) -> str:
    """Return the stored additivity verdict string."""
    lut = _load(lut_path)
    return lut.get("additivity_validation", {}).get("summary", {}).get("verdict", "UNKNOWN")


if __name__ == "__main__":
    # Quick smoke test
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else None
    print(f"Additivity verdict: {additivity_verdict(path)}")
    test_cases = [
        ([64, 128, 256], "base — should ≈ base_tuned_us"),
        ([48, 128, 256], "iso_s0 — should ≈ 21752µs"),
        ([64,  96, 256], "iso_s1 — should ≈ 6911µs"),
        ([64, 128, 192], "iso_s2 — should ≈ 7244µs"),
        ([32,  64, 128], "p50   — multi-stage combo"),
        ([48,  96, 192], "trap25 — multi-stage combo"),
        ([64,  96, 192], "pad64  — multi-stage combo"),
    ]
    for nf, label in test_cases:
        lat = lookup(*nf, "tuned", path)
        info = lookup_info(*nf, "tuned", path)
        print(f"  {label:40s}: {lat:8.0f}µs  (Δs0={info['delta_s0_us']:+.0f} Δs1={info['delta_s1_us']:+.0f} Δs2={info['delta_s2_us']:+.0f})")
