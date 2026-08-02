"""Normalize and aggregate backend-independent S1 structural probe metrics."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


SCHEMA_VERSION = "stage2_s1_probe_metrics_v3"
RATIO_FIELDS = {
    "int8_precision_propagation_ratio": ("int8_propagated_ops", "precision_eligible_ops"),
    "qdq_fold_ratio": ("qdq_folded_pairs", "qdq_pairs"),
    "reformat_rate": ("reformat_ops", "total_ops"),
    "fusion_coverage": ("fused_ops", "fusible_ops"),
}
FORBIDDEN_NAME_PARTS = ("latency", "energy", "ap70", "map", "accuracy")


def _reject_forbidden_names(values: Mapping[str, Any], *, context: str) -> None:
    forbidden = sorted(
        str(name) for name in values if any(part in str(name).lower() for part in FORBIDDEN_NAME_PARTS)
    )
    if forbidden:
        raise ValueError(f"{context} must not contain latency/energy fields: {forbidden}")


def _finite_nonnegative(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite non-negative number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    if number < 0.0:
        raise ValueError(f"{field} must be non-negative")
    return number


def normalize_probe_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one runner record and return a detached normalized copy."""

    if not isinstance(record, Mapping):
        raise ValueError("probe record must be a mapping")
    _reject_forbidden_names(record, context="probe record")
    probe_id = str(record.get("probe_id") or "")
    q_mode = str(record.get("q_mode") or "")
    if not probe_id or not q_mode:
        raise ValueError("probe_id and q_mode must be non-empty")
    build_success = record.get("build_success")
    if not isinstance(build_success, bool):
        raise ValueError("build_success must be boolean")

    normalized: dict[str, Any] = {
        "probe_id": probe_id,
        "q_mode": q_mode,
        "build_success": build_success,
    }
    for field in ("probe_seconds", "build_seconds"):
        normalized[field] = _finite_nonnegative(record.get(field), field=field)
    for numerator, denominator in RATIO_FIELDS.values():
        numerator_value = record.get(numerator)
        denominator_value = record.get(denominator)
        if numerator_value is None and denominator_value is None:
            normalized[numerator] = None
            normalized[denominator] = None
            continue
        if numerator_value is None or denominator_value is None:
            raise ValueError(f"{numerator} and {denominator} must both be present or null")
        normalized[numerator] = _finite_nonnegative(numerator_value, field=numerator)
        normalized[denominator] = _finite_nonnegative(denominator_value, field=denominator)
        if normalized[denominator] <= 0.0:
            raise ValueError(f"{denominator} denominator must be positive")
        if normalized[numerator] > normalized[denominator]:
            raise ValueError(f"{numerator} cannot exceed {denominator}")
    return normalized


def _ratio(record: Mapping[str, Any], numerator: str, denominator: str) -> float | None:
    if record[numerator] is None:
        return None
    return float(record[numerator]) / float(record[denominator])


def summarize_probe_records(
    records: Sequence[Mapping[str, Any]], *, stability_tolerance: float = 0.05
) -> dict[str, Any]:
    """Pool structural counts by probe/q-mode and report evidence quality."""

    tolerance = _finite_nonnegative(stability_tolerance, field="stability_tolerance")
    if not records:
        raise ValueError("probe records must not be empty")
    normalized = [normalize_probe_record(record) for record in records]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in normalized:
        grouped[(record["probe_id"], record["q_mode"])].append(record)

    groups: list[dict[str, Any]] = []
    missing_metrics: list[str] = []
    spreads: dict[str, float] = {}
    for (probe_id, q_mode), rows in sorted(grouped.items()):
        result: dict[str, Any] = {
            "probe_id": probe_id,
            "q_mode": q_mode,
            "sample_count": len(rows),
            "build_success_rate": sum(row["build_success"] for row in rows) / len(rows),
            "probe_cost_seconds": sum(row["probe_seconds"] for row in rows),
            "build_seconds": sum(row["build_seconds"] for row in rows),
            "mean_probe_cost_seconds": sum(row["probe_seconds"] for row in rows) / len(rows),
            "mean_build_seconds": sum(row["build_seconds"] for row in rows) / len(rows),
        }
        build_values = [float(row["build_success"]) for row in rows]
        spreads[f"{probe_id}:{q_mode}:build_success_rate"] = max(build_values) - min(build_values)
        for metric, (numerator, denominator) in RATIO_FIELDS.items():
            observed = [row for row in rows if row[numerator] is not None]
            metric_key = f"{probe_id}:{q_mode}:{metric}"
            if len(observed) != len(rows):
                result[metric] = None
                missing_metrics.append(metric_key)
                continue
            result[metric] = sum(row[numerator] for row in observed) / sum(
                row[denominator] for row in observed
            )
            values = [_ratio(row, numerator, denominator) for row in observed]
            spreads[metric_key] = max(values) - min(values)  # type: ignore[arg-type]
        groups.append(result)

    max_spread = max(spreads.values(), default=0.0)
    return {
        "schema_version": SCHEMA_VERSION,
        "groups": groups,
        "record_count": len(normalized),
        "completeness": {
            "complete": not missing_metrics,
            "missing_metrics": sorted(missing_metrics),
        },
        "stability": {
            "stable": max_spread <= tolerance,
            "tolerance": tolerance,
            "max_abs_spread": max_spread,
            "metric_abs_spread": dict(sorted(spreads.items())),
        },
    }


def _validate_capability_features(features: Mapping[str, Any], *, level: str) -> dict[str, float | int | None]:
    if not isinstance(features, Mapping):
        raise ValueError(f"{level} capability features must be a mapping")
    _reject_forbidden_names(features, context=f"{level} capability features")
    result: dict[str, float | int | None] = {}
    for raw_name, value in features.items():
        name = str(raw_name)
        if not name:
            raise ValueError(f"{level} capability feature names must be non-empty")
        if value is None:
            result[name] = None
        elif isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"{level} capability feature {name} must be finite numeric or null")
        else:
            result[name] = value
    return result


def merge_c0_c1_capability_features(
    c0_features: Mapping[str, Any], c1_features: Mapping[str, Any]
) -> dict[str, float | int | None]:
    """Merge detached C0 static and C1 probe features without silent overwrite."""

    c0 = _validate_capability_features(c0_features, level="C0")
    c1 = _validate_capability_features(c1_features, level="C1")
    conflicts = sorted(name for name in c0.keys() & c1.keys() if c0[name] != c1[name])
    if conflicts:
        raise ValueError(f"conflicting C0/C1 capability features: {conflicts}")
    return {**c0, **c1}


def check_capability_admission(
    summary: Mapping[str, Any], *, max_abs_spread: float = 0.05, min_build_success_rate: float = 1.0
) -> dict[str, Any]:
    """Return deterministic reasons for rejecting incomplete or unstable C1 evidence."""

    spread_limit = _finite_nonnegative(max_abs_spread, field="max_abs_spread")
    build_limit = _finite_nonnegative(min_build_success_rate, field="min_build_success_rate")
    if build_limit > 1.0:
        raise ValueError("min_build_success_rate must be in [0, 1]")
    if summary.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected S1 probe metric summary schema")
    reasons: list[str] = []
    if not summary.get("completeness", {}).get("complete", False):
        reasons.append("incomplete_metrics")
    if float(summary.get("stability", {}).get("max_abs_spread", math.inf)) > spread_limit:
        reasons.append("unstable_metrics")
    groups = summary.get("groups")
    if not isinstance(groups, Sequence) or not groups:
        reasons.append("missing_probe_groups")
    elif any(float(group["build_success_rate"]) < build_limit for group in groups):
        reasons.append("build_success_below_threshold")
    return {"admitted": not reasons, "reasons": reasons}


aggregate_s1_probe_metrics = summarize_probe_records
merge_capability_features = merge_c0_c1_capability_features
check_admission = check_capability_admission


__all__ = [
    "SCHEMA_VERSION",
    "aggregate_s1_probe_metrics",
    "check_admission",
    "check_capability_admission",
    "merge_c0_c1_capability_features",
    "merge_capability_features",
    "normalize_probe_record",
    "summarize_probe_records",
]
