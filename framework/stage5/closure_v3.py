"""Pure closure validation and measured Pareto/HV reporting for Stage5 v3."""

from __future__ import annotations

import math
import hashlib
import json
from collections import Counter
from typing import Any, Mapping, Sequence


SUCCESS_STATUS = "measured_success_gold"
TRUE_FEASIBILITY_STATUSES = {
    "feasibility_failure",
    "numerical_feasibility_failure",
}
TERMINAL_STATUSES = {SUCCESS_STATUS, *TRUE_FEASIBILITY_STATUSES}
CONTEXT_FIELDS = (
    "task_id",
    "task_sha256",
    "model",
    "hardware_id",
    "capability_profile_id",
    "dispatch_key",
)
IDENTITY_FIELDS = (
    "row_id",
    "manifest_job_id",
    "group_id",
    "model",
    "hardware_id",
    "capability_profile_id",
    "dispatch_key",
    "width",
    "q_mode",
    "genome",
)
OBJECTIVES = ("latency_ms", "energy_j", "ap70")
HV_MARGIN = 0.05


def _row_id(row: Mapping[str, Any]) -> str:
    return str(row.get("manifest_job_id") or row.get("row_id") or "")


def _payload_sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _validate_request_hashes(
    request: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> None:
    row_ids = [_row_id(row) for row in rows]
    row_sha = request.get("row_sha256")
    if not isinstance(row_sha, Mapping) or set(row_sha) != set(row_ids):
        raise ValueError("measurement request row SHA map mismatch")
    for row_id, row in zip(row_ids, rows):
        if row_sha.get(row_id) != _payload_sha(row):
            raise ValueError(f"measurement request row SHA mismatch: {row_id}")
    payload = {key: value for key, value in request.items() if key != "measurement_request_sha256"}
    if request.get("measurement_request_sha256") != _payload_sha(payload):
        raise ValueError("measurement request SHA mismatch")


def _rows(batch: Any, *, label: str) -> list[Mapping[str, Any]]:
    source = batch.get("rows") if isinstance(batch, Mapping) else batch
    if not isinstance(source, Sequence) or isinstance(source, (str, bytes)):
        raise ValueError(f"{label} rows must be a sequence")
    if not all(isinstance(row, Mapping) for row in source):
        raise ValueError(f"{label} rows must contain mappings")
    return list(source)


def _finite(value: Any) -> bool:
    try:
        return not isinstance(value, bool) and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _genome(row: Mapping[str, Any]) -> tuple[Any, ...]:
    width = row.get("width")
    q_mode = row.get("q_mode")
    derived = None
    if (
        isinstance(width, Sequence)
        and not isinstance(width, (str, bytes))
        and len(width) == 3
        and q_mode in {"fp16", "int8"}
    ):
        derived = (*width, q_mode)
    genome = row.get("genome")
    if not isinstance(genome, Sequence) or isinstance(genome, (str, bytes)):
        if derived is not None:
            return tuple(derived)
        raise ValueError(f"genome identity missing: {_row_id(row)}")
    explicit = tuple(genome)
    if derived is not None and explicit != tuple(derived):
        raise ValueError(f"genome/width/q_mode identity drift: {_row_id(row)}")
    return explicit


def _context(row: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(row.get(field) or "") for field in CONTEXT_FIELDS)


def _validate_context(row: Mapping[str, Any], expected: tuple[str, ...]) -> None:
    if _context(row) != expected:
        raise ValueError(f"fixed task context drift: {_row_id(row)}")


def _validate_terminal(row: Mapping[str, Any]) -> None:
    status = str(row.get("terminal_status") or "")
    if status not in TERMINAL_STATUSES:
        raise ValueError(
            "terminal status must be measured_success_gold or a true feasibility terminal: "
            f"{_row_id(row)}:{status or '<empty>'}"
        )
    if status == SUCCESS_STATUS and not all(_finite(row.get(metric)) for metric in OBJECTIVES):
        raise ValueError(f"successful measured row lacks finite objectives: {_row_id(row)}")


def _identity_matches(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return all(left.get(field) == right.get(field) for field in IDENTITY_FIELDS)


def _dominates(left: Sequence[float], right: Sequence[float]) -> bool:
    return all(a <= b for a, b in zip(left, right)) and any(
        a < b for a, b in zip(left, right)
    )


def _objective(row: Mapping[str, Any]) -> tuple[float, float, float]:
    return (
        float(row["latency_ms"]),
        float(row["energy_j"]),
        -float(row["ap70"]),
    )


def _frontier(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    vectors = [_objective(row) for row in rows]
    return [
        row
        for index, row in enumerate(rows)
        if not any(
            other_index != index and _dominates(other, vectors[index])
            for other_index, other in enumerate(vectors)
        )
    ]


def _pareto_point(row: Mapping[str, Any], *, round_index: int | None) -> dict[str, Any]:
    point = {
        "manifest_job_id": _row_id(row),
        "group_id": str(row.get("group_id") or ""),
        "genome": list(_genome(row)),
        "q_mode": str(row.get("q_mode") or ""),
        "terminal_status": str(row.get("terminal_status") or ""),
        "round_index": round_index,
        "objectives": {
            "latency_ms": float(row["latency_ms"]),
            "energy_j": float(row["energy_j"]),
            "ap70": float(row["ap70"]),
        },
    }
    for field in (
        "performance_result_json",
        "performance_result_sha256",
        "ap_report_path",
        "ap_report_sha256",
        "materialized_source_evidence_path",
        "materialized_source_evidence_sha256",
    ):
        if row.get(field) not in {None, ""}:
            point[field] = row[field]
    return point


def _hypervolume_2d(
    points: Sequence[tuple[float, float]], reference: tuple[float, float]
) -> float:
    eligible = sorted(
        {
            point
            for point in points
            if point[0] < reference[0] and point[1] < reference[1]
        }
    )
    if not eligible:
        return 0.0
    xs = sorted({point[0] for point in eligible} | {reference[0]})
    area = 0.0
    for left, right in zip(xs, xs[1:]):
        active_y = [y for x, y in eligible if x <= left]
        area += (right - left) * max(0.0, reference[1] - min(active_y))
    return area


def _hypervolume_3d(
    points: Sequence[tuple[float, float, float]],
    reference: tuple[float, float, float],
) -> float:
    eligible = [
        point
        for point in set(points)
        if all(value < reference[i] for i, value in enumerate(point))
    ]
    eligible = [
        point
        for index, point in enumerate(eligible)
        if not any(
            other_index != index and _dominates(other, point)
            for other_index, other in enumerate(eligible)
        )
    ]
    if not eligible:
        return 0.0
    xs = sorted({point[0] for point in eligible} | {reference[0]})
    volume = 0.0
    for left, right in zip(xs, xs[1:]):
        active = [(y, z) for x, y, z in eligible if x <= left]
        volume += (right - left) * _hypervolume_2d(
            active, (reference[1], reference[2])
        )
    return float(volume)


def _normalizer(
    final_rows: Sequence[Mapping[str, Any]],
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, float, float]]:
    if not final_rows:
        raise ValueError("final measured set contains no successful rows")
    vectors = [_objective(row) for row in final_rows]
    lows = tuple(min(vector[axis] for vector in vectors) for axis in range(3))
    highs = tuple(max(vector[axis] for vector in vectors) for axis in range(3))
    spans = tuple(
        high - low if high > low else max(abs(low), 1.0)
        for low, high in zip(lows, highs)
    )
    reference = tuple(
        (high - low) / span + HV_MARGIN
        for low, high, span in zip(lows, highs, spans)
    )
    return lows, spans, reference  # type: ignore[return-value]


def _normalized_points(
    rows: Sequence[Mapping[str, Any]], lows: Sequence[float], spans: Sequence[float]
) -> list[tuple[float, float, float]]:
    return [
        tuple((value - lows[axis]) / spans[axis] for axis, value in enumerate(_objective(row)))
        for row in rows
    ]  # type: ignore[return-value]


def build_stage5_closure_audit(
    gold176_rows: Sequence[Mapping[str, Any]],
    candidate_manifest: Mapping[str, Any],
    measurement_requests: Sequence[Mapping[str, Any]],
    feedback_batches: Sequence[Any],
    atomic_audits: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate one completed Stage5 task and return its measured closure report."""
    if len(gold176_rows) != 176:
        raise ValueError("Gold176 input must contain exactly 176 rows")
    if not (
        len(measurement_requests) == len(feedback_batches) == len(atomic_audits) == 4
    ):
        raise ValueError(
            "Stage5 closure requires exactly four measurement requests, feedback batches, and atomic audits"
        )
    if candidate_manifest.get("schema_version") != "stage5_task_candidate_manifest_v2":
        raise ValueError("unexpected candidate manifest schema")

    gold_by_id = {_row_id(row): row for row in gold176_rows}
    if "" in gold_by_id or len(gold_by_id) != len(gold176_rows):
        raise ValueError("Gold176 identities must be non-empty and unique")
    initial_ids = {
        str(item.get("row_id") or item.get("manifest_job_id") or "")
        for item in candidate_manifest.get("excluded") or []
        if isinstance(item, Mapping) and item.get("reason") == "already_measured"
    }
    if "" in initial_ids or not initial_ids:
        raise ValueError("candidate manifest has no valid already_measured exclusions")
    missing_initial = initial_ids - set(gold_by_id)
    if missing_initial:
        raise ValueError(f"already_measured exclusions missing from Gold176: {sorted(missing_initial)}")
    initial_rows = [gold_by_id[row_id] for row_id in sorted(initial_ids)]
    for row in initial_rows:
        _validate_terminal(row)

    candidate_rows = _rows(candidate_manifest.get("rows") or [], label="candidate manifest")
    candidate_by_id = {_row_id(row): row for row in candidate_rows}
    if "" in candidate_by_id or len(candidate_by_id) != len(candidate_rows):
        raise ValueError("candidate manifest identities must be non-empty and unique")
    if int(candidate_manifest.get("eligible_row_count", -1)) != len(candidate_rows):
        raise ValueError("candidate manifest eligible count drift")
    if not candidate_rows:
        raise ValueError("candidate manifest has no eligible rows")

    expected_context = _context(candidate_rows[0])
    if any(not value for value in expected_context):
        raise ValueError("candidate manifest fixed task context is incomplete")
    for row in candidate_rows:
        _validate_context(row, expected_context)
    task_id, task_sha, model, hardware, profile, dispatch = expected_context
    expected_by_field = dict(zip(CONTEXT_FIELDS, expected_context))
    for row in initial_rows:
        for field in CONTEXT_FIELDS:
            if (
                row.get(field) not in {None, ""}
                and str(row[field]) != expected_by_field[field]
            ):
                raise ValueError(f"initial observed row fixed task context drift: {_row_id(row)}")
    if (
        str(candidate_manifest.get("task_id") or "") != task_id
        or str(candidate_manifest.get("task_sha256") or "") != task_sha
        or str(candidate_manifest.get("target_model") or "") != model
        or str(candidate_manifest.get("capability_profile_id") or "") != profile
    ):
        raise ValueError("candidate manifest fixed task context drift")

    request_rows_by_round: list[list[Mapping[str, Any]]] = []
    online_ids: list[str] = []
    online_genomes: list[tuple[Any, ...]] = []
    for expected_round, request in enumerate(measurement_requests):
        rows = _rows(request, label=f"measurement request round {expected_round}")
        if (
            request.get("schema_version") != "stage5_measurement_request_v2"
            or int(request.get("round_index", -1)) != expected_round
            or int(request.get("batch_size", -1)) != 4
            or int(request.get("sample_budget", -1)) != 16
            or request.get("atomic_feedback") is not True
            or len(rows) != 4
        ):
            raise ValueError("B=4/T=16 requires four ordered atomic rounds of four rows")
        if str(request.get("task_id") or "") != task_id or str(
            request.get("task_sha256") or ""
        ) != task_sha:
            raise ValueError("fixed task context drift in measurement request")
        for row in rows:
            _validate_context(row, expected_context)
            online_ids.append(_row_id(row))
            online_genomes.append(_genome(row))
        request_rows_by_round.append(rows)

    if any(not row_id for row_id in online_ids) or len(set(online_ids)) != 16:
        raise ValueError("online row identities must be non-empty and unique")
    if len(set(online_genomes)) != 16:
        raise ValueError("duplicate online genome identity")
    overlap = initial_ids & set(online_ids)
    if overlap:
        raise ValueError(f"online request overlaps initial observed IDs: {sorted(overlap)}")
    for rows in request_rows_by_round:
        for row in rows:
            row_id = _row_id(row)
            candidate = candidate_by_id.get(row_id)
            if candidate is None or not _identity_matches(row, candidate):
                raise ValueError(f"measurement request identity drift from candidate manifest: {row_id}")
    for request, rows in zip(measurement_requests, request_rows_by_round):
        _validate_request_hashes(request, rows)

    online_rows: list[Mapping[str, Any]] = []
    online_rounds: list[list[Mapping[str, Any]]] = []
    for round_index, (request_rows, feedback_batch, audit) in enumerate(
        zip(request_rows_by_round, feedback_batches, atomic_audits)
    ):
        feedback_rows = _rows(feedback_batch, label=f"feedback batch round {round_index}")
        request_by_id = {_row_id(row): row for row in request_rows}
        feedback_by_id = {_row_id(row): row for row in feedback_rows}
        if len(feedback_rows) != 4 or set(feedback_by_id) != set(request_by_id):
            raise ValueError(f"feedback batch round {round_index} does not match its request")
        for row_id, row in feedback_by_id.items():
            _validate_context(row, expected_context)
            if not _identity_matches(row, request_by_id[row_id]):
                raise ValueError(f"feedback identity drift from request: {row_id}")
            _validate_terminal(row)
        released_rows = _rows(
            audit.get("released_feedback_rows") or [],
            label=f"atomic audit round {round_index}",
        )
        released_by_id = {_row_id(row): row for row in released_rows}
        if (
            audit.get("schema_version") != "stage5_atomic_batch_audit_v2"
            or audit.get("feedback_released") is not True
            or audit.get("batch_quarantined") is not False
            or int(audit.get("budget_consumed", -1)) != 4
            or set(released_by_id) != set(feedback_by_id)
            or len(released_rows) != 4
            or any(
                released_by_id[row_id] != row
                for row_id, row in feedback_by_id.items()
            )
        ):
            raise ValueError(f"released atomic batch audit required for round {round_index}")
        ordered_feedback = [feedback_by_id[_row_id(row)] for row in request_rows]
        online_rounds.append(ordered_feedback)
        online_rows.extend(ordered_feedback)

    combined_rows = [*initial_rows, *online_rows]
    successful_final = [
        row for row in combined_rows if row.get("terminal_status") == SUCCESS_STATUS
    ]
    lows, spans, reference = _normalizer(successful_final)
    initial_successes = [
        row for row in initial_rows if row.get("terminal_status") == SUCCESS_STATUS
    ]
    cumulative = list(initial_successes)
    round_by_id: dict[str, int | None] = {
        _row_id(row): None for row in initial_successes
    }
    hv_initial = _hypervolume_3d(_normalized_points(cumulative, lows, spans), reference)
    hv_after_rounds = []
    frontier_after_rounds = []
    for round_index, rows in enumerate(online_rounds):
        round_by_id.update({_row_id(row): round_index for row in rows})
        cumulative.extend(row for row in rows if row.get("terminal_status") == SUCCESS_STATUS)
        hv_after_rounds.append(
            _hypervolume_3d(_normalized_points(cumulative, lows, spans), reference)
        )
        frontier_after_rounds.append(
            {
                "round_index": round_index,
                "frontier_ids": sorted(_row_id(row) for row in _frontier(cumulative)),
            }
        )

    terminal_counts = Counter(str(row["terminal_status"]) for row in combined_rows)
    frontier_rows = sorted(_frontier(successful_final), key=_row_id)
    frontier_ids = [_row_id(row) for row in frontier_rows]
    frontier_points = [
        _pareto_point(row, round_index=round_by_id.get(_row_id(row)))
        for row in frontier_rows
    ]
    return {
        "schema_version": "stage5_task_closure_audit_v3",
        "task_id": task_id,
        "task_sha256": task_sha,
        "model": model,
        "hardware_id": hardware,
        "capability_profile_id": profile,
        "dispatch_key": dispatch,
        "batch_size": 4,
        "sample_budget": 16,
        "round_count": 4,
        "initial_count": len(initial_rows),
        "online_count": len(online_rows),
        "terminal_counts": {
            status: int(terminal_counts.get(status, 0)) for status in sorted(TERMINAL_STATUSES)
        },
        "frontier_ids": frontier_ids,
        "frontier_points": frontier_points,
        "frontier_after_rounds": frontier_after_rounds,
        "independent_validation_ids": frontier_ids[:4],
        "hv_initial": float(hv_initial),
        "hv_after_rounds": [float(value) for value in hv_after_rounds],
        "hv_curve": [float(hv_initial), *[float(value) for value in hv_after_rounds]],
        "hv_normalization": {
            "objective_order": ["latency_ms", "energy_j", "negative_ap70"],
            "minimum": [float(value) for value in lows],
            "span": [float(value) for value in spans],
            "reference": [float(value) for value in reference],
            "derived_from_successful_count": len(successful_final),
        },
        "closure": True,
    }


__all__ = ["build_stage5_closure_audit"]
