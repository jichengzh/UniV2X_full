"""Exact post-selection cache and atomic mixed-feedback contracts for Stage7."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from framework.stage7.online_component_ablation_v1 import (
    build_measurement_cache_key,
    classify_failure,
    validate_result_row,
)


REQUEST_SCHEMA = "stage5_measurement_request_v2"
BINDING_SCHEMA = "stage7_request_binding_v1"
CACHE_ENTRY_SCHEMA = "stage7_measurement_cache_entry_v1"
CACHE_REVEAL_SCHEMA = "stage7_selected_batch_cache_reveal_v1"
MISS_PLAN_SCHEMA = "stage7_miss_only_performance_plan_v1"
FINALIZATION_SCHEMA = "stage7_atomic_batch_feedback_v1"
SUCCESS_STATUSES = {"success", "measured_success", "measured_success_gold"}

_EVIDENCE_FIELDS = (
    "terminal_evidence",
    "latency_artifact",
    "energy_artifact",
    "ap_artifact",
    "source_checkpoint",
    "onnx",
    "materialized_source_evidence",
)
_COPIED_RESULT_FIELDS = (
    "latency_ms",
    "energy_j",
    "ap30",
    "ap50",
    "ap70",
    *tuple(
        field
        for label in _EVIDENCE_FIELDS
        for field in (f"{label}_path", f"{label}_sha256")
    ),
)


def _sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _row_id(row: Mapping[str, Any]) -> str:
    return str(row.get("manifest_job_id") or row.get("row_id") or "")


def _without_sha(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in payload.items()
        if key != field
    }


def _verified_request(request: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    copied = copy.deepcopy(dict(request))
    if copied.get("schema_version") != REQUEST_SCHEMA:
        raise ValueError("unexpected measurement request schema")
    expected_request_sha = _sha(_without_sha(copied, "measurement_request_sha256"))
    if copied.get("measurement_request_sha256") != expected_request_sha:
        raise ValueError("measurement request SHA mismatch")
    rows_value = copied.get("rows")
    if not isinstance(rows_value, list) or len(rows_value) != 4:
        raise ValueError("measurement request must contain exactly four rows")
    if not all(isinstance(row, Mapping) for row in rows_value):
        raise ValueError("measurement request rows must be mappings")
    rows = [copy.deepcopy(dict(row)) for row in rows_value]
    row_ids = [_row_id(row) for row in rows]
    if any(not row_id for row_id in row_ids) or len(set(row_ids)) != 4:
        raise ValueError("measurement request row identities must be unique")
    row_sha = copied.get("row_sha256")
    if not isinstance(row_sha, Mapping) or set(row_sha) != set(row_ids):
        raise ValueError("measurement request row SHA map mismatch")
    for row, row_id in zip(rows, row_ids):
        if row_sha.get(row_id) != _sha(row):
            raise ValueError(f"measurement request row SHA drift: {row_id}")
        if (
            row.get("task_id") != copied.get("task_id")
            or row.get("task_sha256") != copied.get("task_sha256")
        ):
            raise ValueError(f"measurement request row task identity drift: {row_id}")
    if copied.get("batch_size") != 4:
        raise ValueError("Stage7 request batch size must remain four")
    return copied, rows


def _verified_binding(
    request: Mapping[str, Any],
    binding: Mapping[str, Any],
    *,
    trajectory_contract_sha256: str,
) -> tuple[dict[str, Any], str]:
    request_copy, rows = _verified_request(request)
    copied = copy.deepcopy(dict(binding))
    if copied.get("schema_version") != BINDING_SCHEMA:
        raise ValueError("unexpected Stage7 request binding schema")
    if not _is_sha256(trajectory_contract_sha256):
        raise ValueError("trajectory contract SHA is invalid")
    row_ids = [_row_id(row) for row in rows]
    checks = (
        copied.get("measurement_request_sha256")
        == request_copy["measurement_request_sha256"],
        copied.get("trajectory_contract_sha256")
        == trajectory_contract_sha256,
        copied.get("round_index") == request_copy.get("round_index"),
        copied.get("selected_row_ids") == row_ids,
        copied.get("selected_row_ids_sha256") == _sha(row_ids),
    )
    if not all(checks):
        raise ValueError("request binding identity or SHA mismatch")
    binding_payload = _without_sha(copied, "request_binding_sha256")
    binding_sha = _sha(binding_payload)
    recorded_sha = copied.get("request_binding_sha256")
    if not _is_sha256(recorded_sha) or recorded_sha != binding_sha:
        raise ValueError("request binding self SHA mismatch")
    return copied, binding_sha


def _verify_artifact(path_value: Any, sha_value: Any, label: str) -> Path:
    if not _is_sha256(sha_value):
        raise ValueError(f"{label} SHA256 is missing or invalid")
    path = Path(str(path_value or ""))
    if not path.is_file():
        raise ValueError(f"{label} artifact missing")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != sha_value:
        raise ValueError(f"{label} artifact SHA mismatch")
    return path


def _read_json_evidence(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} evidence is not valid JSON") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} evidence must be a mapping")
    return copy.deepcopy(dict(value))


def _cache_entry_result(entry: Mapping[str, Any]) -> dict[str, Any]:
    result = entry.get("result")
    if not isinstance(result, Mapping):
        raise ValueError("cache entry result is missing")
    return copy.deepcopy(dict(result))


def bind_cached_result_to_request(
    request: Mapping[str, Any],
    request_row: Mapping[str, Any],
    key_dimensions: Mapping[str, Any],
    cache_entry: Mapping[str, Any],
    *,
    trajectory_contract_sha256: str,
) -> dict[str, Any]:
    """Validate one exact successful cache entry and rebind it to the S7 request."""
    request_copy, request_rows = _verified_request(request)
    current_row_id = _row_id(request_row)
    matching = [row for row in request_rows if _row_id(row) == current_row_id]
    if len(matching) != 1 or matching[0] != dict(request_row):
        raise ValueError("cached result request-row identity drift")
    if not _is_sha256(trajectory_contract_sha256):
        raise ValueError("trajectory contract SHA is invalid")

    key_sha = build_measurement_cache_key(key_dimensions)
    entry = copy.deepcopy(dict(cache_entry))
    if (
        entry.get("schema_version") != CACHE_ENTRY_SCHEMA
        or entry.get("cache_key_sha256") != key_sha
    ):
        raise ValueError("cache entry exact-key identity mismatch")
    entry_sha = entry.get("cache_entry_sha256")
    if not _is_sha256(entry_sha) or entry_sha != _sha(
        _without_sha(entry, "cache_entry_sha256")
    ):
        raise ValueError("cache entry SHA mismatch")
    origin = entry.get("origin")
    if not isinstance(origin, Mapping):
        raise ValueError("cache entry origin provenance is missing")

    result = _cache_entry_result(entry)
    if result.get("terminal_status") not in SUCCESS_STATUSES:
        raise ValueError("cache hit is not successful terminal evidence")
    evidence_paths = {}
    for label in _EVIDENCE_FIELDS:
        evidence_paths[label] = _verify_artifact(
            result.get(f"{label}_path"),
            result.get(f"{label}_sha256"),
            label,
        )
    if (
        result.get("source_checkpoint_sha256")
        != key_dimensions.get("source_checkpoint_sha256")
        or result.get("onnx_sha256") != key_dimensions.get("onnx_sha256")
    ):
        raise ValueError("cache entry current source checkpoint or ONNX SHA mismatch")
    validated_origin = validate_result_row(result)["row"]
    terminal = _read_json_evidence(
        evidence_paths["terminal_evidence"], "terminal"
    )
    expected_terminal = {
        "schema_version": "stage5_terminal_measurement_evidence_v1",
        "row_id": _row_id(validated_origin),
        "terminal_status": validated_origin["terminal_status"],
        **{
            metric: validated_origin[metric]
            for metric in ("latency_ms", "energy_j", "ap30", "ap50", "ap70")
        },
    }
    if any(terminal.get(key) != value for key, value in expected_terminal.items()):
        raise ValueError("terminal evidence semantic drift")
    materialized = _read_json_evidence(
        evidence_paths["materialized_source_evidence"], "materialized source"
    )
    expected_materialized = {
        "schema_version": "stage5_source_materialization_evidence_v1",
        "status": "ready",
        "group_id": request_row.get("group_id"),
        "model": request_row.get("model"),
        "width": "x".join(str(value) for value in request_row.get("width") or []),
        "source_plan_sha256": request_row.get("source_evidence_sha256"),
        "checkpoint_sha256": key_dimensions.get("source_checkpoint_sha256"),
        "onnx_path": validated_origin.get("onnx_path"),
        "onnx_sha256": key_dimensions.get("onnx_sha256"),
    }
    if any(
        materialized.get(key) != value
        for key, value in expected_materialized.items()
    ):
        raise ValueError("materialized source evidence semantic drift")
    _verify_artifact(
        materialized.get("onnx_path"),
        materialized.get("onnx_sha256"),
        "materialized source ONNX",
    )

    current = matching[0]
    rebound = {
        **copy.deepcopy(current),
        "round_index": request_copy["round_index"],
        "measurement_request_row_sha256": request_copy["row_sha256"][
            current_row_id
        ],
        "terminal_status": "measured_success_gold",
        **{
            field: copy.deepcopy(validated_origin[field])
            for field in _COPIED_RESULT_FIELDS
            if field in validated_origin
        },
        "cache_provenance": {
            "origin_row_id": _row_id(validated_origin),
            "source_cache_entry_sha256": entry_sha,
            "origin": copy.deepcopy(dict(origin)),
            "origin_terminal_status": validated_origin["terminal_status"],
            "trajectory_contract_sha256": trajectory_contract_sha256,
        },
    }
    return validate_result_row(rebound)["row"]


def reveal_selected_batch_cache(
    request: Mapping[str, Any],
    request_binding: Mapping[str, Any],
    *,
    trajectory_contract_sha256: str,
    key_dimensions_by_row: Mapping[str, Mapping[str, Any]],
    cache: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Reveal exact usable hits only after the frozen request binding is verified."""
    request_copy, rows = _verified_request(request)
    _, binding_sha = _verified_binding(
        request_copy,
        request_binding,
        trajectory_contract_sha256=trajectory_contract_sha256,
    )
    row_ids = [_row_id(row) for row in rows]
    unexpected = set(key_dimensions_by_row) - set(row_ids)
    if unexpected:
        raise ValueError(
            "cache key dimensions contain mixed request identities: "
            + ",".join(sorted(unexpected))
        )

    reveal_rows: list[dict[str, Any]] = []
    for request_row, row_id in zip(rows, row_ids):
        dimensions = key_dimensions_by_row.get(row_id)
        if not isinstance(dimensions, Mapping):
            reveal_rows.append(
                {
                    "row_id": row_id,
                    "measurement_request_row_sha256": request_copy["row_sha256"][
                        row_id
                    ],
                    "disposition": "miss",
                    "reason": "invalid_exact_key_dimensions",
                }
            )
            continue
        try:
            key_sha = build_measurement_cache_key(dimensions)
        except ValueError:
            reveal_rows.append(
                {
                    "row_id": row_id,
                    "measurement_request_row_sha256": request_copy["row_sha256"][
                        row_id
                    ],
                    "disposition": "miss",
                    "reason": "invalid_exact_key_dimensions",
                }
            )
            continue
        entry = cache.get(key_sha)
        if entry is None:
            reveal_rows.append(
                {
                    "row_id": row_id,
                    "cache_key_sha256": key_sha,
                    "measurement_request_row_sha256": request_copy["row_sha256"][
                        row_id
                    ],
                    "disposition": "miss",
                    "reason": "no_exact_cache_entry",
                }
            )
            continue
        try:
            bound = bind_cached_result_to_request(
                request_copy,
                request_row,
                dimensions,
                entry,
                trajectory_contract_sha256=trajectory_contract_sha256,
            )
        except (OSError, ValueError) as error:
            message = str(error)
            reason = (
                "cache_entry_unusable_missing_evidence"
                if "artifact" in message or "SHA" in message
                else "cache_entry_unusable_invalid_evidence"
            )
            reveal_rows.append(
                {
                    "row_id": row_id,
                    "cache_key_sha256": key_sha,
                    "measurement_request_row_sha256": request_copy["row_sha256"][
                        row_id
                    ],
                    "disposition": "miss",
                    "reason": reason,
                }
            )
            continue
        reveal_rows.append(
            {
                "row_id": row_id,
                "cache_key_sha256": key_sha,
                "measurement_request_row_sha256": request_copy["row_sha256"][
                    row_id
                ],
                "disposition": "exact_hit",
                "source_cache_entry_sha256": entry["cache_entry_sha256"],
                "source_evidence_sha256": bound[
                    "materialized_source_evidence_sha256"
                ],
                "revealed_after_request_sha256": request_copy[
                    "measurement_request_sha256"
                ],
                "trajectory_contract_sha256": trajectory_contract_sha256,
                "bound_result": bound,
            }
        )

    hit_count = sum(row["disposition"] == "exact_hit" for row in reveal_rows)
    payload = {
        "schema_version": CACHE_REVEAL_SCHEMA,
        "measurement_request_sha256": request_copy["measurement_request_sha256"],
        "request_binding_sha256": binding_sha,
        "trajectory_contract_sha256": trajectory_contract_sha256,
        "row_count": 4,
        "exact_hit_count": hit_count,
        "miss_count": 4 - hit_count,
        "rows": reveal_rows,
    }
    return {**payload, "cache_reveal_sha256": _sha(payload)}


def _verified_reveal(
    reveal: Mapping[str, Any],
    request: Mapping[str, Any],
    *,
    binding_sha: str,
    trajectory_contract_sha256: str,
) -> dict[str, Any]:
    copied = copy.deepcopy(dict(reveal))
    if copied.get("schema_version") != CACHE_REVEAL_SCHEMA:
        raise ValueError("unexpected cache reveal schema")
    if copied.get("cache_reveal_sha256") != _sha(
        _without_sha(copied, "cache_reveal_sha256")
    ):
        raise ValueError("cache reveal SHA mismatch")
    if (
        copied.get("measurement_request_sha256")
        != request.get("measurement_request_sha256")
        or copied.get("request_binding_sha256") != binding_sha
        or copied.get("trajectory_contract_sha256")
        != trajectory_contract_sha256
    ):
        raise ValueError("cache reveal request identity drift")
    reveal_rows = copied.get("rows")
    request_rows = request.get("rows")
    if not isinstance(reveal_rows, list) or len(reveal_rows) != 4:
        raise ValueError("cache reveal must contain exactly four rows")
    expected_ids = [_row_id(row) for row in request_rows]
    actual_ids = [_row_id(row) for row in reveal_rows if isinstance(row, Mapping)]
    if actual_ids != expected_ids:
        raise ValueError("cache reveal row order or identity drift")
    for row_id, row in zip(expected_ids, reveal_rows):
        if row.get("measurement_request_row_sha256") != request["row_sha256"][
            row_id
        ]:
            raise ValueError(f"cache reveal row SHA drift: {row_id}")
        disposition = row.get("disposition")
        if disposition not in {"exact_hit", "miss"}:
            raise ValueError(f"invalid cache disposition: {row_id}")
        if disposition == "exact_hit" and not isinstance(
            row.get("bound_result"), Mapping
        ):
            raise ValueError(f"exact cache hit lacks bound result: {row_id}")
        if disposition == "miss" and "bound_result" in row:
            raise ValueError(f"cache miss contains a bound result: {row_id}")
    hit_count = sum(row["disposition"] == "exact_hit" for row in reveal_rows)
    if (
        copied.get("row_count") != 4
        or copied.get("exact_hit_count") != hit_count
        or copied.get("miss_count") != 4 - hit_count
    ):
        raise ValueError("cache reveal counts drift")
    return copied


def _unique_rows(
    rows: Sequence[Mapping[str, Any]], *, label: str
) -> tuple[list[dict[str, Any]], list[str]]:
    copied = [copy.deepcopy(dict(row)) for row in rows]
    row_ids = [_row_id(row) for row in copied]
    if any(not row_id for row_id in row_ids) or len(set(row_ids)) != len(row_ids):
        raise ValueError(f"{label} has duplicate or empty row identities")
    return copied, row_ids


def derive_miss_only_plan(
    full_plan: Mapping[str, Any],
    request: Mapping[str, Any],
    request_binding: Mapping[str, Any],
    cache_reveal: Mapping[str, Any],
) -> dict[str, Any]:
    """Filter a four-row Stage5 plan to misses without mutating its lineage."""
    request_copy, request_rows = _verified_request(request)
    trajectory_sha = str(cache_reveal.get("trajectory_contract_sha256") or "")
    _, binding_sha = _verified_binding(
        request_copy,
        request_binding,
        trajectory_contract_sha256=trajectory_sha,
    )
    reveal = _verified_reveal(
        cache_reveal,
        request_copy,
        binding_sha=binding_sha,
        trajectory_contract_sha256=trajectory_sha,
    )
    plan = copy.deepcopy(dict(full_plan))
    manifest_value = plan.get("manifest")
    performance_value = plan.get("performance_jobs")
    if not isinstance(manifest_value, Mapping) or not isinstance(
        performance_value, list
    ):
        raise ValueError("full performance plan shape is invalid")
    manifest = copy.deepcopy(dict(manifest_value))
    manifest_jobs_value = manifest.get("jobs")
    if not isinstance(manifest_jobs_value, list):
        raise ValueError("full performance manifest jobs are missing")
    manifest_jobs, manifest_ids = _unique_rows(
        manifest_jobs_value, label="full manifest jobs"
    )
    performance_jobs, performance_ids = _unique_rows(
        performance_value, label="full performance jobs"
    )
    request_ids = [_row_id(row) for row in request_rows]
    if (
        manifest_ids != request_ids
        or performance_ids != request_ids
        or manifest.get("source_request_sha256")
        != request_copy["measurement_request_sha256"]
        or manifest.get("row_count") != 4
        or manifest.get("genome_count") != 4
    ):
        raise ValueError("full performance plan request identity drift")
    miss_ids = [
        row["row_id"]
        for row in reveal["rows"]
        if row["disposition"] == "miss"
    ]
    miss_set = set(miss_ids)
    filtered_manifest_jobs = [
        row for row in manifest_jobs if _row_id(row) in miss_set
    ]
    filtered_performance_jobs = [
        row for row in performance_jobs if _row_id(row) in miss_set
    ]
    group_ids = list(
        dict.fromkeys(str(row.get("group_id") or "") for row in filtered_manifest_jobs)
    )
    if any(not group_id for group_id in group_ids):
        raise ValueError("miss-only plan contains an empty source group")
    filtered_manifest = {
        **manifest,
        "source_request_sha256": request_copy["measurement_request_sha256"],
        "genome_count": len(miss_ids),
        "row_count": len(miss_ids),
        "group_count": len(group_ids),
        "group_ids": group_ids,
        "jobs": filtered_manifest_jobs,
        "stage7_miss_only": True,
        "stage7_request_binding_sha256": binding_sha,
        "stage7_cache_reveal_sha256": reveal["cache_reveal_sha256"],
        "stage7_original_request_row_sha256": {
            row_id: request_copy["row_sha256"][row_id] for row_id in miss_ids
        },
    }
    payload = {
        "schema_version": MISS_PLAN_SCHEMA,
        "original_measurement_request_sha256": request_copy[
            "measurement_request_sha256"
        ],
        "request_binding_sha256": binding_sha,
        "cache_reveal_sha256": reveal["cache_reveal_sha256"],
        "trajectory_contract_sha256": trajectory_sha,
        "original_request_row_sha256": {
            row_id: request_copy["row_sha256"][row_id] for row_id in miss_ids
        },
        "miss_row_ids": miss_ids,
        "manifest": filtered_manifest,
        "performance_jobs": filtered_performance_jobs,
    }
    return {**payload, "miss_plan_sha256": _sha(payload)}


def _validate_feedback_identity(
    request: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    allow_graph_promotion: bool = False,
) -> dict[str, Any]:
    row_copy = copy.deepcopy(dict(row))
    row_id = _row_id(row_copy)
    requested = {
        _row_id(request_row): request_row for request_row in request["rows"]
    }
    if row_id not in requested:
        raise ValueError(f"feedback has mixed request identity: {row_id}")
    expected = requested[row_id]
    if row_copy.get("measurement_request_row_sha256") != request["row_sha256"][
        row_id
    ]:
        raise ValueError(f"feedback identity row SHA drift: {row_id}")
    for field, expected_value in expected.items():
        if allow_graph_promotion and field == "graph_features":
            continue
        if row_copy.get(field) != expected_value:
            raise ValueError(f"feedback identity field drift: {row_id}:{field}")
    validate_result_row(row_copy)
    return row_copy


def _callback_rows(value: Any, *, label: str) -> list[dict[str, Any]]:
    rows: Any = value
    if isinstance(value, Mapping):
        for field in ("rows", "promoted_rows", "released_feedback_rows"):
            if isinstance(value.get(field), list):
                rows = value[field]
                break
    if (
        not isinstance(rows, Sequence)
        or isinstance(rows, (str, bytes))
        or not all(isinstance(row, Mapping) for row in rows)
    ):
        raise ValueError(f"{label} did not return feedback rows")
    copied, row_ids = _unique_rows(rows, label=label)
    if len(copied) != 4 or len(row_ids) != 4:
        raise ValueError(f"{label} must return exactly four rows")
    return copied


def _validated_promotion(
    value: Any,
    request: Mapping[str, Any],
    historical_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise ValueError("promotion callback must return rows and audit mappings")
    audit_value = value.get("audit")
    if not isinstance(audit_value, Mapping):
        raise ValueError("promotion callback audit is required")
    promoted_rows = _callback_rows(value.get("rows"), label="promotion callback")
    audit = copy.deepcopy(dict(audit_value))
    expected_group_count = len(
        {str(row.get("group_id") or "") for row in request["rows"]}
    )
    if (
        audit.get("schema_version")
        != "stage5_actual_feedback_batch_audit_v3"
        or audit.get("task_id") != request.get("task_id")
        or audit.get("round_index") != request.get("round_index")
        or audit.get("promoted_row_count") != 4
        or audit.get("actual_group_count") != expected_group_count
        or audit.get("silent_surrogate_fallback_count") != 0
    ):
        raise ValueError("promotion callback audit contract mismatch")
    audit_rows_value = audit.get("rows")
    if not isinstance(audit_rows_value, list) or len(audit_rows_value) != 4:
        raise ValueError("promotion callback audit must bind exactly four rows")
    if not all(isinstance(row, Mapping) for row in audit_rows_value):
        raise ValueError("promotion callback audit rows must be mappings")

    validated_rows: list[dict[str, Any]] = []
    for request_row, historical, promoted, audit_row in zip(
        request["rows"], historical_rows, promoted_rows, audit_rows_value
    ):
        row_id = _row_id(request_row)
        validated = _validate_feedback_identity(
            request, promoted, allow_graph_promotion=True
        )
        candidate = request_row.get("graph_features")
        actual = validated.get("graph_features")
        if not isinstance(candidate, Mapping) or not isinstance(actual, Mapping):
            raise ValueError(f"promotion graph feature mapping missing: {row_id}")
        expected_actual_identity = {
            "schema": "stage5_actual_graph_features_v1",
            "group_id": request_row.get("group_id"),
            "model": request_row.get("model"),
            "width": request_row.get("width"),
            "graph_feature_provenance": "materialized_onnx_extracted_v1",
        }
        if any(
            actual.get(field) != expected
            for field, expected in expected_actual_identity.items()
        ):
            raise ValueError(
                f"promotion actual graph identity/provenance drift: {row_id}"
            )
        if actual == candidate:
            raise ValueError(f"promotion silently retained surrogate graph: {row_id}")
        expected_candidate_sha = _sha(candidate)
        expected_actual_sha = _sha(actual)
        expected_historical_sha = _sha(historical)
        expected_promoted_sha = _sha(
            _without_sha(validated, "actual_feedback_row_sha256")
        )
        expected_fields = {
            "candidate_graph_features": candidate,
            "candidate_graph_features_sha256": expected_candidate_sha,
            "materialized_graph_features_sha256": expected_actual_sha,
            "historical_feedback_row_sha256": expected_historical_sha,
            "feedback_feature_contract": "actual_feedback_v3",
            "graph_feature_promotion_schema": "stage5_actual_feedback_promotion_v3",
            "actual_feedback_row_sha256": expected_promoted_sha,
        }
        if any(
            validated.get(field) != expected
            for field, expected in expected_fields.items()
        ):
            raise ValueError(f"promotion row contract or SHA drift: {row_id}")
        expected_audit_row = {
            "manifest_job_id": row_id,
            "group_id": request_row.get("group_id"),
            "candidate_graph_features_sha256": expected_candidate_sha,
            "materialized_graph_features_sha256": expected_actual_sha,
            "actual_feedback_row_sha256": expected_promoted_sha,
        }
        if dict(audit_row) != expected_audit_row:
            raise ValueError(f"promotion audit row drift: {row_id}")
        validated_rows.append(validated)
    if [_row_id(row) for row in validated_rows] != [
        _row_id(row) for row in request["rows"]
    ]:
        raise ValueError("promotion callback request-order drift")
    return validated_rows, audit


def finalize_stage7_atomic_batch(
    request: Mapping[str, Any],
    request_binding: Mapping[str, Any],
    cache_reveal: Mapping[str, Any],
    miss_results: Sequence[Mapping[str, Any]],
    *,
    promote_feedback_batch: Callable[
        [Mapping[str, Any], Sequence[Mapping[str, Any]]], Any
    ],
    finalize_atomic_batch: Callable[
        [Mapping[str, Any], Sequence[Mapping[str, Any]]], Mapping[str, Any]
    ],
) -> dict[str, Any]:
    """Merge four identities, then call promotion and Stage5 atomic finalization."""
    request_copy, request_rows = _verified_request(request)
    trajectory_sha = str(cache_reveal.get("trajectory_contract_sha256") or "")
    _, binding_sha = _verified_binding(
        request_copy,
        request_binding,
        trajectory_contract_sha256=trajectory_sha,
    )
    reveal = _verified_reveal(
        cache_reveal,
        request_copy,
        binding_sha=binding_sha,
        trajectory_contract_sha256=trajectory_sha,
    )
    expected_miss_ids = [
        row["row_id"]
        for row in reveal["rows"]
        if row["disposition"] == "miss"
    ]
    misses, miss_ids = _unique_rows(miss_results, label="miss results")
    if miss_ids != expected_miss_ids:
        raise ValueError("miss result duplicate, omission, or request-order drift")
    validated_misses = [
        _validate_feedback_identity(request_copy, row) for row in misses
    ]
    miss_by_id = {_row_id(row): row for row in validated_misses}
    hit_by_id = {
        row["row_id"]: _validate_feedback_identity(
            request_copy, row["bound_result"]
        )
        for row in reveal["rows"]
        if row["disposition"] == "exact_hit"
    }
    merged = [
        copy.deepcopy(hit_by_id.get(_row_id(row)) or miss_by_id.get(_row_id(row)))
        for row in request_rows
    ]
    if any(row is None for row in merged):
        raise ValueError("atomic feedback merge contains an omission")
    merged_rows = [dict(row) for row in merged]
    merged_ids = [_row_id(row) for row in merged_rows]
    if merged_ids != [_row_id(row) for row in request_rows]:
        raise ValueError("atomic feedback merge request order drift")

    infrastructure_failures = []
    for row in merged_rows:
        if row.get("terminal_status") in SUCCESS_STATUSES:
            continue
        classification = classify_failure(str(row.get("failure_kind") or ""))
        if classification["same_request_retry"]:
            infrastructure_failures.append(
                {
                    "row_id": _row_id(row),
                    **classification,
                    "failure_reason": row.get("failure_reason"),
                }
            )
    if infrastructure_failures:
        retry_payload = {
            "schema_version": FINALIZATION_SCHEMA,
            "feedback_released": False,
            "batch_quarantined": True,
            "same_request_retry": True,
            "budget_consumed": 0,
            "measurement_request_sha256": request_copy[
                "measurement_request_sha256"
            ],
            "request_binding_sha256": binding_sha,
            "cache_reveal_sha256": reveal["cache_reveal_sha256"],
            "request": copy.deepcopy(request_copy),
            "infrastructure_failures": infrastructure_failures,
        }
        return {
            **retry_payload,
            "stage7_finalization_sha256": _sha(retry_payload),
        }

    promoted_value = promote_feedback_batch(
        copy.deepcopy(request_copy), copy.deepcopy(merged_rows)
    )
    promoted_rows, promotion_audit = _validated_promotion(
        promoted_value, request_copy, merged_rows
    )

    atomic_value = finalize_atomic_batch(
        copy.deepcopy(request_copy), copy.deepcopy(promoted_rows)
    )
    if not isinstance(atomic_value, Mapping):
        raise ValueError("atomic finalizer did not return an audit mapping")
    atomic_audit = copy.deepcopy(dict(atomic_value))
    if (
        atomic_audit.get("feedback_released") is not True
        or atomic_audit.get("batch_quarantined") is True
        or atomic_audit.get("budget_consumed") != 4
    ):
        raise ValueError("atomic finalizer did not release the valid four-row batch")
    released_value = atomic_audit.get("released_feedback_rows", promoted_rows)
    released_rows = _callback_rows(
        released_value, label="atomic finalizer"
    )
    released_rows = [
        _validate_feedback_identity(
            request_copy, row, allow_graph_promotion=True
        )
        for row in released_rows
    ]
    if [_row_id(row) for row in released_rows] != [
        _row_id(row) for row in request_rows
    ]:
        raise ValueError("atomic finalizer request-order drift")
    payload = {
        "schema_version": FINALIZATION_SCHEMA,
        "feedback_released": True,
        "batch_quarantined": False,
        "same_request_retry": False,
        "budget_consumed": 4,
        "measurement_request_sha256": request_copy[
            "measurement_request_sha256"
        ],
        "request_binding_sha256": binding_sha,
        "cache_reveal_sha256": reveal["cache_reveal_sha256"],
        "exact_hit_count": reveal["exact_hit_count"],
        "miss_count": reveal["miss_count"],
        "callback_order": ["promote_feedback_batch", "finalize_atomic_batch"],
        "promotion_audit": promotion_audit,
        "released_feedback_rows": released_rows,
        "atomic_batch_audit": atomic_audit,
    }
    return {**payload, "stage7_finalization_sha256": _sha(payload)}
