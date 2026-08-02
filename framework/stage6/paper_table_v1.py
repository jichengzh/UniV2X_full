"""Evidence-conservative Stage6 paper-table selection."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


DEFAULT_ARMS = (
    "compression_only",
    "schedule_only",
    "compress_then_tune",
    "tune_then_compress",
    "joint_shcosearch",
)
LATENCY_CLOSE_FRACTION = 0.01


def _finite(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite metric: {value!r}")
    return number


def _trusted(point: Mapping[str, Any]) -> bool:
    return (
        point.get("terminal_status") == "measured_success_gold"
        and point.get("independent_validation_passed") is True
        and point.get("evidence_sha_verified") is True
    )


def _select(points: Sequence[Mapping[str, Any]], floor: float) -> Mapping[str, Any] | None:
    eligible = [point for point in points if _trusted(point) and _finite(point["AP70"]) >= floor]
    if not eligible:
        return None
    minimum = min(_finite(point["latency_ms"]) for point in eligible)
    close = [
        point
        for point in eligible
        if _finite(point["latency_ms"]) <= minimum * (1.0 + LATENCY_CLOSE_FRACTION)
    ]
    return min(close, key=lambda point: (_finite(point["energy_j"]), _finite(point["latency_ms"])))


def _fastest_trusted(points: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    eligible = [point for point in points if _trusted(point)]
    if not eligible:
        return None
    minimum = min(_finite(point["latency_ms"]) for point in eligible)
    close = [
        point
        for point in eligible
        if _finite(point["latency_ms"]) <= minimum * (1.0 + LATENCY_CLOSE_FRACTION)
    ]
    return min(close, key=lambda point: (_finite(point["energy_j"]), _finite(point["latency_ms"])))


def build_backend_tables(
    *,
    backend: str,
    baseline: Mapping[str, Any],
    arms: Mapping[str, Mapping[str, Any]],
    deltas: Sequence[float],
    required_arms: Sequence[str] = DEFAULT_ARMS,
) -> dict[str, Any]:
    baseline_ap = _finite(baseline["AP70"])
    baseline_latency = _finite(baseline["latency_ms"])
    missing = [arm for arm in required_arms if arm not in arms]
    nonterminal = []
    for arm in required_arms:
        if arm not in arms:
            continue
        summary = arms[arm]
        status = summary.get("status")
        if status == "complete":
            ready = summary.get("independent_validation_complete") is True
        elif status == "complete_failure":
            ready = summary.get("failure_evidence_sha_verified") is True
        else:
            ready = False
        if not ready:
            nonterminal.append(arm)
    tables: dict[str, list[dict[str, Any]]] = {}
    for delta in deltas:
        if not 0 < float(delta) < 1:
            raise ValueError("AP delta must be between zero and one")
        floor = baseline_ap - float(delta)
        rows = [
            {
                "method": "original_default",
                "backend": baseline.get("backend") or "pytorch_eager",
                "delta_ap_max": float(delta),
                "ap70_floor": floor,
                "selection_status": "selected",
                "outcome": "selected",
                "failure_reason": None,
                "ap_constraint_violated": False,
                "config": baseline.get("config") or [64, 128, 256, "fp32"],
                "AP70": baseline_ap,
                "latency_ms": baseline_latency,
                "energy_j": _finite(baseline["energy_j"]),
                "speedup": 1.0,
                "HV": None,
                "pareto_count": 1,
                "outer_genomes": 1,
                "tuning_trials": 0,
                "schedule_policy": baseline.get("schedule_policy"),
                "gpu_hours": baseline.get("gpu_hours"),
                "wallclock_s": baseline.get("wallclock_s"),
                "failure_count": 0,
                "failure_rate": 0.0,
                "evidence_origin": baseline.get("evidence_origin")
                or "stage6_new_original_default_measurement",
            }
        ]
        for arm in required_arms:
            summary = arms.get(arm) or {}
            selected = _select(summary.get("points") or [], floor)
            representative = selected or _fastest_trusted(summary.get("points") or [])
            outer = int(summary.get("outer_genomes") or 0)
            failures = int(summary.get("failure_count") or 0)
            failure_reason = summary.get("failure_reason")
            if selected:
                selection_status = "selected"
                outcome = "selected"
            elif summary.get("status") == "complete_failure":
                selection_status = "feasibility_failure"
                outcome = f"feasibility_failure:{failure_reason or 'unspecified'}"
            else:
                selection_status = "no_feasible_point"
                outcome = "no_point_satisfies_ap_floor"
            row = {
                "method": arm,
                "backend": backend,
                "delta_ap_max": float(delta),
                "ap70_floor": floor,
                "selection_status": selection_status,
                "outcome": outcome,
                "failure_reason": failure_reason,
                "ap_constraint_violated": (
                    selected is None and representative is not None
                    if summary.get("status") != "complete_failure"
                    else None
                ),
                "config": representative.get("config") if representative else None,
                "AP70": _finite(representative["AP70"]) if representative else None,
                "latency_ms": _finite(representative["latency_ms"]) if representative else None,
                "energy_j": _finite(representative["energy_j"]) if representative else None,
                "speedup": (
                    baseline_latency / _finite(representative["latency_ms"])
                    if representative
                    else None
                ),
                "HV": summary.get("HV"),
                "pareto_count": summary.get("pareto_count"),
                "outer_genomes": outer,
                "tuning_trials": summary.get("tuning_trials"),
                "gpu_hours": summary.get("gpu_hours"),
                "wallclock_s": summary.get("wallclock_s"),
                "failure_count": failures,
                "failure_rate": failures / outer if outer else None,
                "evidence_origin": (
                    representative.get("evidence_origin")
                    if representative
                    else summary.get("evidence_origin")
                ),
            }
            rows.append(row)
        ranked = sorted(
            (row for row in rows if row["selection_status"] == "selected"),
            key=lambda row: (row["latency_ms"], row["energy_j"]),
        )
        for rank, row in enumerate(ranked, start=1):
            row["latency_rank"] = rank
        tables[f"delta_{float(delta):.2f}"] = rows
    return {
        "schema_version": "stage6_paper_main_tables_v1",
        "backend": backend,
        "baseline_ap70": baseline_ap,
        "baseline_latency_ms": baseline_latency,
        "latency_close_fraction": LATENCY_CLOSE_FRACTION,
        "deltas": [float(value) for value in deltas],
        "tables": tables,
        "missing_arms": missing,
        "nonterminal_arms": nonterminal,
        "paper_ready": not missing and not nonterminal,
    }
