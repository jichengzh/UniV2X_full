"""Small deterministic protocol probes for Stage6 serial and blind-search arms."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence


BLIND_FIELDS = ("parameter_count", "flops", "ap_surrogate")
BACKEND_LABEL_FIELDS = {
    "latency_ms",
    "energy_j",
    "latency_prediction",
    "energy_prediction",
    "capability_profile",
    "dispatch_key",
    "backend",
}


def _normalize(values: Sequence[float], value: float) -> float:
    low, high = min(values), max(values)
    return 0.0 if high == low else (value - low) / (high - low)


def select_hardware_blind_batch(
    candidates: Iterable[Mapping[str, Any]], *, batch_size: int
) -> list[dict[str, Any]]:
    rows = [dict(row) for row in candidates]
    if len(rows) < batch_size or batch_size <= 0:
        raise ValueError("candidate count must cover a positive batch")
    for row in rows:
        missing = [field for field in BLIND_FIELDS if field not in row]
        if missing:
            raise ValueError(f"hardware-blind candidate missing fields: {missing}")
    columns = {
        field: [float(row[field]) for row in rows]
        for field in BLIND_FIELDS
    }
    scored = []
    for row in rows:
        score = (
            _normalize(columns["ap_surrogate"], float(row["ap_surrogate"]))
            - 0.5 * _normalize(columns["parameter_count"], float(row["parameter_count"]))
            - 0.5 * _normalize(columns["flops"], float(row["flops"]))
        )
        scored.append({**row, "hardware_blind_acquisition_score": score})
    scored.sort(key=lambda row: (-row["hardware_blind_acquisition_score"], str(row["candidate_id"])))
    return [
        {key: value for key, value in row.items() if key not in BACKEND_LABEL_FIELDS}
        for row in scored[:batch_size]
    ]


def lock_compress_then_tune(screened: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [dict(row) for row in screened]
    if len(rows) != 12:
        raise ValueError("compress-then-tune requires exactly 12 screened candidates")
    ranked = sorted(rows, key=lambda row: (int(row["screen_rank"]), str(row["candidate_id"])))
    locked = [dict(row) for row in ranked[:4]]
    return {
        "schema_version": "stage6_compress_then_tune_lock_smoke_v1",
        "screen_count": len(rows),
        "locked_count": len(locked),
        "locked_candidate_ids": [str(row["candidate_id"]) for row in locked],
        "lock_precedes_tuning": True,
        "tuning_started": False,
        "runtime_budget_mutation": False,
    }


def record_reverse_transfer_attempts(
    attempts: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = []
    for source in attempts:
        row = dict(source)
        applicable = row.get("applicable") is True
        fallback_used = row.get("fallback_used") is True
        retuned = row.get("compressed_shape_retuned") is True
        rows.append(
            {
                **row,
                "terminal_status": "transferred_success" if applicable else "feasibility_failure",
                "fallback_used": fallback_used,
                "compressed_shape_retuned": retuned,
            }
        )
    failures = sum(row["terminal_status"] == "feasibility_failure" for row in rows)
    fallback_count = sum(row["fallback_used"] for row in rows)
    retune_count = sum(row["compressed_shape_retuned"] for row in rows)
    return {
        "schema_version": "stage6_tune_then_compress_accounting_smoke_v1",
        "attempt_count": len(rows),
        "success_count": len(rows) - failures,
        "failure_count": failures,
        "fallback_count": fallback_count,
        "retune_count": retune_count,
        "attempts": rows,
    }
