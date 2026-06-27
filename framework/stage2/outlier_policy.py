"""Latency outlier policy for Stage2 LUT rows."""

from __future__ import annotations

from statistics import median
from typing import Any


CALIBRATION_REPEAT_MAX_MIN = 2.0
PAPER_REPEAT_MAX_MIN = 1.5
CALIBRATION_MULTI_RUN_SPREAD = 0.15
PAPER_MULTI_RUN_SPREAD = 0.10
PAPER_MULTI_RUN_ABS_MS = 0.5


def _grade_thresholds(grade: str) -> dict[str, float]:
    if grade == "paper":
        return {
            "repeat_max_min": PAPER_REPEAT_MAX_MIN,
            "multi_run_spread": PAPER_MULTI_RUN_SPREAD,
            "multi_run_abs_ms": PAPER_MULTI_RUN_ABS_MS,
        }
    return {
        "repeat_max_min": CALIBRATION_REPEAT_MAX_MIN,
        "multi_run_spread": CALIBRATION_MULTI_RUN_SPREAD,
        "multi_run_abs_ms": float("inf"),
    }


def _row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("model") or ""),
        str(row.get("config_id") or ""),
        str(row.get("schedule_policy") or ""),
    )


def _latency_ms(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value is None:
        return None
    return float(value) / 1000.0


def _repeat_ratio(row: dict[str, Any]) -> float | None:
    min_us = row.get("latency_min_us")
    max_us = row.get("latency_max_us")
    if min_us is None or max_us is None:
        return None
    min_value = float(min_us)
    if min_value <= 0:
        return None
    return float(max_us) / min_value


def classify_latency_row(
    row: dict[str, Any],
    *,
    group_p50_ms: list[float] | None = None,
    grade: str = "calibration",
    historical_anchor_ms: float | None = None,
) -> dict[str, Any]:
    thresholds = _grade_thresholds(grade)
    reasons: list[str] = []
    quality_flag = "stable"
    claim_status = "claimable"

    if row.get("schema") != "latency_lut_row_v1":
        quality_flag = "not_latency"
        claim_status = "no_claim"
        reasons.append("not_latency_lut_row")
    elif row.get("measurement_status") != "measured":
        quality_flag = "not_measured"
        claim_status = "no_claim"
        reasons.append(f"measurement_status={row.get('measurement_status')}")
    else:
        ratio = _repeat_ratio(row)
        if ratio is None:
            quality_flag = "missing_raw_repeats"
            claim_status = "no_claim"
            reasons.append("missing_latency_min_or_max")
        elif ratio > thresholds["repeat_max_min"]:
            quality_flag = "unstable_repeat"
            claim_status = "no_claim"
            reasons.append(
                f"repeat_max_min_ratio={ratio:.3f}>{thresholds['repeat_max_min']:.3f}"
            )

        if group_p50_ms and len(group_p50_ms) >= 2:
            spread = max(group_p50_ms) - min(group_p50_ms)
            med = median(group_p50_ms)
            rel = spread / med if med > 0 else float("inf")
            abs_ok = spread <= thresholds["multi_run_abs_ms"]
            rel_ok = rel <= thresholds["multi_run_spread"]
            if not (rel_ok or abs_ok):
                if quality_flag == "stable":
                    quality_flag = "unstable_multi_run"
                claim_status = "no_claim"
                reasons.append(
                    "multi_run_p50_spread="
                    f"{rel:.3f}>{thresholds['multi_run_spread']:.3f}"
                )

        if historical_anchor_ms is not None:
            p50_ms = _latency_ms(row, "latency_p50_us")
            if p50_ms is not None:
                drift = abs(p50_ms - historical_anchor_ms)
                rel_drift = drift / historical_anchor_ms if historical_anchor_ms else 0.0
                if grade == "paper" and rel_drift > 0.10 and drift > 0.5:
                    if quality_flag == "stable":
                        quality_flag = "anchor_drift"
                    claim_status = "no_claim"
                    reasons.append(
                        f"historical_anchor_drift={rel_drift:.3f}>0.100"
                    )
                elif rel_drift > 0.15 and drift > 0.5:
                    reasons.append(
                        f"historical_anchor_warning={rel_drift:.3f}>0.150"
                    )

    return {
        "row_id": row.get("row_id"),
        "run_id": row.get("run_id"),
        "config_id": row.get("config_id"),
        "model": row.get("model"),
        "candidate_id": row.get("candidate_id"),
        "schedule_policy": row.get("schedule_policy"),
        "latency_p50_ms": _latency_ms(row, "latency_p50_us"),
        "latency_min_ms": _latency_ms(row, "latency_min_us"),
        "latency_max_ms": _latency_ms(row, "latency_max_us"),
        "quality_flag": quality_flag,
        "claim_status": claim_status,
        "reasons": reasons,
    }


def detect_latency_outliers(
    rows: list[dict[str, Any]],
    *,
    grade: str = "calibration",
    historical_anchors_ms: dict[str, float] | None = None,
) -> dict[str, Any]:
    historical_anchors_ms = historical_anchors_ms or {}
    p50_by_key: dict[tuple[str, str, str], list[float]] = {}
    for row in rows:
        p50 = _latency_ms(row, "latency_p50_us")
        if p50 is not None:
            p50_by_key.setdefault(_row_key(row), []).append(p50)

    report_rows: list[dict[str, Any]] = []
    for row in rows:
        key = _row_key(row)
        anchor_key = "|".join(key)
        report_rows.append(
            classify_latency_row(
                row,
                group_p50_ms=p50_by_key.get(key),
                grade=grade,
                historical_anchor_ms=historical_anchors_ms.get(anchor_key),
            )
        )

    unstable = [
        row
        for row in report_rows
        if row["claim_status"] != "claimable" or row["quality_flag"] != "stable"
    ]
    return {
        "schema": "stage2_latency_outlier_report_v1",
        "grade": grade,
        "total_rows": len(report_rows),
        "unstable_rows": len(unstable),
        "claimable_rows": len(report_rows) - len(unstable),
        "rows": report_rows,
    }
