"""Selected-only, append-only exact cache contracts for Stage7 v2.

This module intentionally does not adapt historical cache wrappers.  A v2
cache is an immutable lineage of terminal evidence created by this run; cache
membership is observable only after an ordered selection binding is frozen.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Any, Mapping

from framework.stage7.online_component_ablation_v1 import (
    CACHE_KEY_DIMENSIONS,
    build_measurement_cache_key,
)


CACHE_SCHEMA = "stage7_core_cache_v2"
SELECTION_BINDING_SCHEMA = "stage7_actual_v3_selection_binding_v2"
TERMINAL_EVIDENCE_SCHEMA = "stage7_v2_terminal_evidence_v1"
ACTUAL_V3_TERMINAL_EVIDENCE_SCHEMA = "stage7_actual_v3_terminal_wrapper_v2"
LINEAGE_SCHEMA = "stage7_v2_cache_lineage_record_v1"
V2_PRODUCER = "stage7_core_cache_v2"
ACTUAL_V3_PRODUCER = "stage7_actual_v3_adapter_v2"

_METRIC_FIELDS = ("latency_ms", "energy_j", "ap30", "ap50", "ap70")
METRIC_FIELDS = _METRIC_FIELDS
_CACHEABLE_TERMINAL_STATUSES = {"measured_success", "measured_success_gold"}
_CANDIDATE_FAILURE_REASONS = {
    "backend_capability_failure",
    "backend_failure",
    "build_failure",
    "unsupported_precision",
    "quantization_failure",
    "numerical_failure",
    "candidate_runtime_capability_failure",
}
_INFRASTRUCTURE_FAILURE_REASONS = {
    "gpu_occupancy_drift",
    "gpu_unavailable",
    "unrelated_process_oom",
    "contention",
    "ssh_failure",
    "network_failure",
    "missing_source_artifact",
    "permission_failure",
    "runner_bug",
}
_EVIDENCE_FAILURE_REASONS = {
    "terminal_artifact_sha_mismatch",
    "terminal_evidence_sha_mismatch",
    "evidence_schema_mismatch",
    "evidence_missing",
}
_LOGICAL_REQUEST_FIELDS = frozenset(
    (
        "schema_version",
        "task_id",
        "task_sha256",
        "round_index",
        "batch_size",
        "sample_budget",
        "required_metrics",
        "atomic_feedback",
        "real_h800_measurement_required",
        "row_sha256",
        "rows",
    )
)
EXACT_CACHE_KEY_DIMENSIONS = (
    "candidate_id",
    *CACHE_KEY_DIMENSIONS,
    "dispatch_key",
    "runtime_contract_sha256",
)
_EXACT_DIMENSION_FIELDS = frozenset(EXACT_CACHE_KEY_DIMENSIONS)
_TERMINAL_EVIDENCE_FIELDS = frozenset(
    (
        "schema_version",
        "created_by",
        "terminal_status",
        "candidate_id",
        "exact_key_dimensions",
        "exact_cache_key_sha256",
        "request_sha256",
        *_METRIC_FIELDS,
        "terminal_evidence_sha256",
    )
)
_ACTUAL_V3_TERMINAL_EVIDENCE_FIELDS = frozenset(
    (
        "schema_version",
        "created_by",
        "terminal_status",
        "candidate_id",
        "exact_key_dimensions",
        "exact_cache_key_sha256",
        "stage5_terminal_sha256",
        "stage5_terminal_artifact",
        "stage3_performance_artifact",
        "stage3_ap_artifact",
        "actual_graph_features_sha256",
        *_METRIC_FIELDS,
        "terminal_evidence_sha256",
    )
)
_RAW_STAGE3_ARTIFACT_REFERENCE_FIELDS = frozenset(
    (
        "artifact_kind",
        "raw_artifact_path",
        "raw_artifact_sha256",
        "executor_state_path",
        "executor_state_sha256",
        "job_manifest_path",
        "job_manifest_sha256",
    )
)
_RAW_STAGE3_PERFORMANCE_REFERENCE_FIELDS = frozenset(
    (
        *_RAW_STAGE3_ARTIFACT_REFERENCE_FIELDS,
        "performance_jobs_path",
        "performance_jobs_sha256",
    )
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


def _without_sha(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in payload.items()
        if key != field
    }


def validate_logical_request(request: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one immutable Stage5 v2 logical request and its ordered rows."""
    if not isinstance(request, Mapping):
        raise ValueError("logical request must be a mapping")
    copied = copy.deepcopy(dict(request))
    recorded = copied.pop("measurement_request_sha256", None)
    if copied.get("schema_version") != "stage5_measurement_request_v2":
        raise ValueError("unexpected logical request schema")
    if set(copied) != _LOGICAL_REQUEST_FIELDS:
        raise ValueError("logical request shape is invalid")
    if not _is_sha256(recorded) or recorded != _sha(copied):
        raise ValueError("logical request SHA mismatch")
    rows = copied.get("rows")
    row_sha = copied.get("row_sha256")
    if (
        not isinstance(rows, list)
        or not rows
        or not all(isinstance(row, Mapping) for row in rows)
        or not isinstance(row_sha, Mapping)
    ):
        raise ValueError("logical request rows are invalid")
    row_ids = [
        str(row.get("row_id") or row.get("manifest_job_id") or "")
        for row in rows
    ]
    if any(not row_id for row_id in row_ids) or len(set(row_ids)) != len(row_ids):
        raise ValueError("logical request row identities are invalid")
    if set(row_sha) != set(row_ids):
        raise ValueError("logical request row SHA identities drift")
    for row_id, row in zip(row_ids, rows):
        if row_sha[row_id] != _sha(dict(row)):
            raise ValueError("logical request row SHA mismatch")
    if copied.get("batch_size") != len(rows):
        raise ValueError("logical request batch size mismatch")
    return {
        "request": {**copied, "measurement_request_sha256": recorded},
        "logical_request_sha256": str(recorded),
        "row_ids": row_ids,
        "row_sha256": copy.deepcopy(dict(row_sha)),
        "rows": copy.deepcopy(rows),
    }


def _verified_request_sha256(request: Mapping[str, Any]) -> str:
    return str(validate_logical_request(request)["logical_request_sha256"])


def build_v2_exact_cache_key(dimensions: Mapping[str, Any]) -> str:
    """Hash all exact identity dimensions, including candidate and runtime."""
    validated = validate_formal_exact_dimensions(dimensions)
    return _sha(
        {
            field: copy.deepcopy(validated[field])
            for field in EXACT_CACHE_KEY_DIMENSIONS
        }
    )


def _validate_exact_dimensions(
    value: Any, *, require_runtime: bool = False
) -> dict[str, Any]:
    """Legacy v2 audit validator.

    ``require_runtime=False`` exists only for the blocked pre-adapter finalizer
    fixtures.  Formal cache/reveal/adapter paths call
    :func:`validate_formal_exact_dimensions`.
    """
    if not isinstance(value, Mapping):
        raise ValueError("exact cache key dimensions are missing")
    dimensions = copy.deepcopy(dict(value))
    legacy_fields = frozenset(("candidate_id", *CACHE_KEY_DIMENSIONS))
    required_fields = (
        _EXACT_DIMENSION_FIELDS if require_runtime else legacy_fields
    )
    missing = sorted(required_fields - set(dimensions))
    if missing:
        raise ValueError("exact contract protocol SHA is missing: " + ",".join(missing))
    if set(dimensions) not in {legacy_fields, _EXACT_DIMENSION_FIELDS}:
        raise ValueError("exact cache key dimensions shape is invalid")
    if not isinstance(dimensions["candidate_id"], str) or not dimensions["candidate_id"]:
        raise ValueError("exact cache key candidate identity is invalid")
    invalid_shas = [
        field
        for field in EXACT_CACHE_KEY_DIMENSIONS
        if field.endswith("_sha256")
        and field in dimensions
        and not _is_sha256(dimensions.get(field))
    ]
    if invalid_shas:
        raise ValueError(
            "exact contract protocol SHA is missing or invalid: "
            + ",".join(invalid_shas)
        )
    try:
        build_measurement_cache_key(dimensions)
    except ValueError as error:
        raise ValueError("invalid exact cache key dimensions") from error
    return dimensions


def validate_formal_exact_dimensions(value: Any) -> dict[str, Any]:
    """Require the complete Stage7 actual-v3 exact identity contract."""
    return _validate_exact_dimensions(value, require_runtime=True)


def validate_selection_binding(
    request: Mapping[str, Any], binding: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate the selection completely before any cache object is touched."""
    logical = validate_logical_request(request)
    if not isinstance(binding, Mapping) or binding.get("selection_frozen") is not True:
        raise ValueError("selection binding must be frozen")
    copied = copy.deepcopy(dict(binding))
    if copied.get("schema_version") != SELECTION_BINDING_SCHEMA:
        raise ValueError("unexpected selection binding schema")
    recorded_sha = copied.get("selection_binding_sha256")
    if not _is_sha256(recorded_sha) or recorded_sha != _sha(
        _without_sha(copied, "selection_binding_sha256")
    ):
        raise ValueError("selection binding SHA mismatch")
    if copied.get("logical_request_sha256") != logical["logical_request_sha256"]:
        raise ValueError("selection binding logical request SHA mismatch")
    selected = copied.get("selected_candidates")
    if not isinstance(selected, list) or not selected:
        raise ValueError("frozen selection must contain ordered candidates")
    if not all(isinstance(entry, Mapping) for entry in selected):
        raise ValueError("frozen selection entries must be mappings")

    candidate_ids = [str(entry.get("candidate_id") or "") for entry in selected]
    if any(not candidate_id for candidate_id in candidate_ids) or len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("frozen selection candidate identities must be unique")
    if copied.get("selected_candidate_ids_sha256") != _sha(candidate_ids):
        raise ValueError("frozen selection order SHA mismatch")
    if candidate_ids != logical["row_ids"]:
        raise ValueError("selection binding order differs from logical request")

    validated: list[dict[str, Any]] = []
    for candidate_id, entry in zip(candidate_ids, selected):
        dimensions = validate_formal_exact_dimensions(
            entry.get("exact_key_dimensions")
        )
        if dimensions["candidate_id"] != candidate_id:
            raise ValueError("selection candidate identity mismatch")
        row_sha = entry.get("logical_row_sha256")
        if row_sha != logical["row_sha256"][candidate_id]:
            raise ValueError("selection logical row SHA mismatch")
        key = build_v2_exact_cache_key(dimensions)
        if entry.get("exact_cache_key_sha256") != key:
            raise ValueError("selection exact cache key mismatch")
        cross_payload = {
            "candidate_id": candidate_id,
            "logical_row_sha256": row_sha,
            "exact_key_dimensions": dimensions,
            "exact_cache_key_sha256": key,
        }
        if entry.get("logical_exact_binding_sha256") != _sha(
            cross_payload
        ):
            raise ValueError("logical/exact row binding SHA mismatch")
        validated.append(
            {
                **cross_payload,
                "logical_exact_binding_sha256": entry[
                    "logical_exact_binding_sha256"
                ],
            }
        )
    return {
        "logical_request": logical,
        "binding": copied,
        "selected_candidates": validated,
        "selection_binding_sha256": str(recorded_sha),
    }


def _cache_entries(cache: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(cache, Mapping) or cache.get("schema_version") != CACHE_SCHEMA:
        raise ValueError("unexpected v2 cache schema")
    entries = cache.get("entries")
    lineage = cache.get("lineage")
    if not isinstance(entries, Mapping) or not isinstance(lineage, list):
        raise ValueError("v2 cache entries or lineage are invalid")
    return entries


def validate_terminal_evidence(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Validate only the blocked legacy-v2 cache wrapper."""
    if not isinstance(evidence, Mapping):
        raise ValueError("terminal evidence must be a mapping")
    copied = copy.deepcopy(dict(evidence))
    if copied.get("schema_version") == ACTUAL_V3_TERMINAL_EVIDENCE_SCHEMA:
        raise ValueError(
            "legacy terminal validator rejects formal actual-v3 schema"
        )
    if set(copied) != _TERMINAL_EVIDENCE_FIELDS:
        raise ValueError("terminal evidence shape is invalid")
    dimensions = _validate_exact_dimensions(
        copied.get("exact_key_dimensions")
    )
    if copied.get("schema_version") != TERMINAL_EVIDENCE_SCHEMA:
        raise ValueError("historical terminal evidence wrappers are not accepted")
    if copied.get("created_by") != V2_PRODUCER:
        raise ValueError("terminal evidence was not created by this v2 run")
    evidence_sha = copied.get("terminal_evidence_sha256")
    if not _is_sha256(evidence_sha) or evidence_sha != _sha(
        _without_sha(copied, "terminal_evidence_sha256")
    ):
        raise ValueError("terminal evidence SHA mismatch")
    if copied.get("terminal_status") not in _CACHEABLE_TERMINAL_STATUSES:
        raise ValueError("only successful terminal evidence may be cached")
    if not str(copied.get("candidate_id") or ""):
        raise ValueError("terminal evidence candidate identity is missing")
    if copied["candidate_id"] != dimensions["candidate_id"]:
        raise ValueError("terminal evidence candidate identity mismatch")
    if not _is_sha256(copied.get("request_sha256")):
        raise ValueError("terminal evidence request SHA is missing or invalid")
    expected_key = (
        build_v2_exact_cache_key(dimensions)
        if "runtime_contract_sha256" in dimensions
        else build_measurement_cache_key(dimensions)
    )
    if copied.get("exact_cache_key_sha256") != expected_key:
        raise ValueError("terminal evidence exact cache key mismatch")
    for field in _METRIC_FIELDS:
        value = copied.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"successful terminal evidence metric is invalid: {field}")
    return copied


def validate_actual_v3_terminal_evidence(
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the formal actual-v3 terminal wrapper without legacy fallback."""
    if not isinstance(evidence, Mapping):
        raise ValueError("formal actual-v3 terminal evidence must be a mapping")
    copied = copy.deepcopy(dict(evidence))
    if copied.get("schema_version") != ACTUAL_V3_TERMINAL_EVIDENCE_SCHEMA:
        raise ValueError("formal actual-v3 terminal schema is required")
    if set(copied) != _ACTUAL_V3_TERMINAL_EVIDENCE_FIELDS:
        raise ValueError("formal actual-v3 terminal evidence shape is invalid")
    dimensions = validate_formal_exact_dimensions(
        copied.get("exact_key_dimensions")
    )
    if copied.get("created_by") != ACTUAL_V3_PRODUCER:
        raise ValueError("formal actual-v3 producer drift")
    evidence_sha = copied.get("terminal_evidence_sha256")
    if not _is_sha256(evidence_sha) or evidence_sha != _sha(
        _without_sha(copied, "terminal_evidence_sha256")
    ):
        raise ValueError("formal actual-v3 terminal evidence SHA mismatch")
    if copied.get("terminal_status") not in _CACHEABLE_TERMINAL_STATUSES:
        raise ValueError("formal actual-v3 terminal is not successful")
    if copied.get("candidate_id") != dimensions["candidate_id"]:
        raise ValueError("formal actual-v3 candidate identity mismatch")
    if copied.get("exact_cache_key_sha256") != build_v2_exact_cache_key(
        dimensions
    ):
        raise ValueError("formal actual-v3 exact cache key mismatch")
    sha_fields = ("stage5_terminal_sha256", "actual_graph_features_sha256")
    if any(not _is_sha256(copied.get(field)) for field in sha_fields):
        raise ValueError("formal actual-v3 lineage SHA is invalid")
    stage5 = copied.get("stage5_terminal_artifact")
    if (
        not isinstance(stage5, Mapping)
        or set(stage5) != {"artifact_kind", "path", "artifact_sha256"}
        or stage5.get("artifact_kind") != "stage5_terminal"
        or not str(stage5.get("path") or "")
        or not _is_sha256(stage5.get("artifact_sha256"))
    ):
        raise ValueError(
            "formal actual-v3 artifact reference is invalid: "
            "stage5_terminal_artifact"
        )
    for field, kind in (
        ("stage3_performance_artifact", "stage3_performance_raw_v3"),
        ("stage3_ap_artifact", "stage3_ap_raw_v3"),
    ):
        artifact = copied.get(field)
        reference_fields = (
            _RAW_STAGE3_PERFORMANCE_REFERENCE_FIELDS
            if kind == "stage3_performance_raw_v3"
            else _RAW_STAGE3_ARTIFACT_REFERENCE_FIELDS
        )
        path_fields = [
            "raw_artifact_path",
            "executor_state_path",
            "job_manifest_path",
        ]
        sha_fields = [
            "raw_artifact_sha256",
            "executor_state_sha256",
            "job_manifest_sha256",
        ]
        if kind == "stage3_performance_raw_v3":
            path_fields.append("performance_jobs_path")
            sha_fields.append("performance_jobs_sha256")
        if (
            not isinstance(artifact, Mapping)
            or set(artifact) != reference_fields
            or artifact.get("artifact_kind") != kind
            or any(
                not str(artifact.get(path_field) or "")
                for path_field in path_fields
            )
            or any(
                not _is_sha256(artifact.get(sha_field))
                for sha_field in sha_fields
            )
        ):
            raise ValueError(
                f"formal actual-v3 artifact reference is invalid: {field}"
            )
    for field in _METRIC_FIELDS:
        value = copied.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"formal actual-v3 metric is invalid: {field}")
    return copied


def _validated_present_legacy_cache_entry(
    entries: Mapping[str, Any], key: str
) -> dict[str, Any] | None:
    """Blocked legacy cache lookup used only by legacy append tests."""
    if key not in entries:
        return None
    evidence = entries[key]
    if evidence is None:
        raise ValueError("present exact-cache evidence is invalid")
    validated = validate_terminal_evidence(evidence)
    if validated["exact_cache_key_sha256"] != key:
        raise ValueError("present exact-cache evidence key mismatch")
    return validated


def validate_actual_v3_cache(cache: Mapping[str, Any]) -> dict[str, Any]:
    """Replay and authenticate a formal actual-v3 cache lineage."""
    entries = _cache_entries(cache)
    lineage = cache["lineage"]
    if any(value is None for value in entries.values()):
        raise ValueError("present exact-cache evidence is invalid")
    validated_entries = {
        str(key): validate_actual_v3_terminal_evidence(value)
        for key, value in entries.items()
    }
    replay_entries: dict[str, Any] = {}
    replay_lineage: list[dict[str, Any]] = []
    for raw_record in lineage:
        if not isinstance(raw_record, Mapping):
            raise ValueError("formal cache lineage record is invalid")
        record = copy.deepcopy(dict(raw_record))
        recorded_sha = record.pop("lineage_record_sha256", None)
        if (
            set(record)
            != {
                "schema_version",
                "action",
                "parent_cache_lineage_sha256",
                "exact_cache_key_sha256",
                "evidence_sha256",
            }
            or record.get("schema_version") != LINEAGE_SCHEMA
            or record.get("action")
            != "append_actual_v3_terminal_evidence"
            or not _is_sha256(recorded_sha)
            or recorded_sha != _sha(record)
            or record.get("parent_cache_lineage_sha256")
            != _sha(
                {
                    "entries": replay_entries,
                    "lineage": replay_lineage,
                }
            )
        ):
            raise ValueError("formal cache lineage authentication failed")
        key = str(record.get("exact_cache_key_sha256") or "")
        evidence = validated_entries.get(key)
        if (
            evidence is None
            or key in replay_entries
            or record.get("evidence_sha256")
            != evidence["terminal_evidence_sha256"]
        ):
            raise ValueError("formal cache lineage evidence binding failed")
        replay_entries[key] = copy.deepcopy(evidence)
        replay_lineage.append(
            {**record, "lineage_record_sha256": recorded_sha}
        )
    if replay_entries != validated_entries:
        raise ValueError("formal cache lineage does not cover cache entries")
    return {
        "schema_version": CACHE_SCHEMA,
        "entries": replay_entries,
        "lineage": replay_lineage,
    }


def _formal_cache_snapshot_identity(
    validated_cache: Mapping[str, Any],
) -> tuple[str, str]:
    lineage = validated_cache["lineage"]
    lineage_head = (
        str(lineage[-1]["lineage_record_sha256"])
        if lineage
        else _sha(
            {
                "schema_version": LINEAGE_SCHEMA,
                "empty_actual_v3_lineage": True,
            }
        )
    )
    return _sha(dict(validated_cache)), lineage_head


def reveal_v2_cache_after_selection(
    request: Mapping[str, Any],
    binding: Mapping[str, Any],
    cache: Mapping[str, Any],
) -> dict[str, Any]:
    """Reveal only selected exact entries, after binding validation is complete.

    Cache hits reuse verified terminal truth but never erase the selected-event
    cost of the selection that caused the lookup.
    """
    verified = validate_selection_binding(request, binding)
    selected = verified["selected_candidates"]
    formal_cache = validate_actual_v3_cache(cache)
    entries = formal_cache["entries"]
    cache_snapshot_sha, lineage_head_sha = _formal_cache_snapshot_identity(
        formal_cache
    )
    reveal_entries: list[dict[str, Any]] = []
    for selection in selected:
        key = selection["exact_cache_key_sha256"]
        validated = entries.get(key)
        disposition = (
            "hit"
            if (
                validated is not None
                and validated["exact_cache_key_sha256"] == key
                and validated["candidate_id"] == selection["candidate_id"]
            )
            else "miss"
        )
        terminal_sha = (
            validated["terminal_evidence_sha256"]
            if disposition == "hit"
            else None
        )
        reveal_use = {
            "candidate_id": selection["candidate_id"],
            "logical_request_sha256": verified["logical_request"][
                "logical_request_sha256"
            ],
            "selection_binding_sha256": verified[
                "selection_binding_sha256"
            ],
            "logical_row_sha256": selection["logical_row_sha256"],
            "logical_exact_binding_sha256": selection[
                "logical_exact_binding_sha256"
            ],
            "exact_cache_key_sha256": key,
            "terminal_evidence_sha256": terminal_sha,
            "disposition": disposition,
            "cache_snapshot_sha256": cache_snapshot_sha,
            "lineage_head_sha256": lineage_head_sha,
        }
        if (
            disposition == "hit"
        ):
            reveal_entries.append(
                {
                    "candidate_id": selection["candidate_id"],
                    "logical_row_sha256": selection["logical_row_sha256"],
                    "logical_exact_binding_sha256": selection[
                        "logical_exact_binding_sha256"
                    ],
                    "exact_cache_key_sha256": key,
                    "disposition": "hit",
                    "selected_event_budget_delta": 1,
                    "hardware_measurement_required": False,
                    "terminal_evidence_sha256": terminal_sha,
                    "terminal_evidence": validated,
                    "reveal_use_binding_sha256": _sha(reveal_use),
                }
            )
        else:
            reveal_entries.append(
                {
                    "candidate_id": selection["candidate_id"],
                    "logical_row_sha256": selection["logical_row_sha256"],
                    "logical_exact_binding_sha256": selection[
                        "logical_exact_binding_sha256"
                    ],
                    "exact_cache_key_sha256": key,
                    "disposition": "miss",
                    "selected_event_budget_delta": 1,
                    "hardware_measurement_required": True,
                    "terminal_evidence_sha256": None,
                    "reveal_use_binding_sha256": _sha(reveal_use),
                }
            )
    payload = {
        "schema_version": "stage7_actual_v3_selected_cache_reveal_v2",
        "formal_actual_v3_cache_reveal": True,
        "logical_request_sha256": verified["logical_request"][
            "logical_request_sha256"
        ],
        "selection_binding_sha256": verified["selection_binding_sha256"],
        "cache_snapshot_sha256": cache_snapshot_sha,
        "lineage_head_sha256": lineage_head_sha,
        "entries": reveal_entries,
        "selected_event_budget_delta": len(reveal_entries),
    }
    return {**payload, "cache_reveal_sha256": _sha(payload)}


def append_v2_terminal_evidence(
    cache: Mapping[str, Any], evidence: Mapping[str, Any]
) -> dict[str, Any]:
    """Blocked legacy append helper; formal actual-v3 evidence is rejected."""
    if (
        isinstance(evidence, Mapping)
        and evidence.get("schema_version")
        == ACTUAL_V3_TERMINAL_EVIDENCE_SCHEMA
    ):
        raise ValueError("legacy cache append rejects formal actual-v3 evidence")
    entries = _cache_entries(cache)
    validated = validate_terminal_evidence(evidence)
    key = validated["exact_cache_key_sha256"]
    existing = _validated_present_legacy_cache_entry(entries, key)
    if existing is not None:
        if existing != validated:
            raise ValueError("conflicting exact-cache evidence")
        return copy.deepcopy(dict(cache))

    copied_cache = copy.deepcopy(dict(cache))
    copied_entries = copy.deepcopy(dict(entries))
    copied_lineage = copy.deepcopy(list(copied_cache["lineage"]))
    parent_lineage_sha = _sha(
        {"entries": copied_entries, "lineage": copied_lineage}
    )
    lineage_payload = {
        "schema_version": LINEAGE_SCHEMA,
        "action": "append_terminal_evidence",
        "parent_cache_lineage_sha256": parent_lineage_sha,
        "exact_cache_key_sha256": key,
        "evidence_sha256": validated["terminal_evidence_sha256"],
    }
    copied_entries[key] = validated
    return {
        **copied_cache,
        "entries": copied_entries,
        "lineage": [
            *copied_lineage,
            {
                **lineage_payload,
                "lineage_record_sha256": _sha(lineage_payload),
            },
        ],
    }


def append_actual_v3_terminal_evidence(
    cache: Mapping[str, Any], evidence: Mapping[str, Any]
) -> dict[str, Any]:
    """Append one formal actual-v3 terminal after full lineage replay."""
    validated_cache = validate_actual_v3_cache(cache)
    validated = validate_actual_v3_terminal_evidence(evidence)
    entries = validated_cache["entries"]
    key = validated["exact_cache_key_sha256"]
    existing = entries.get(key)
    if existing is not None:
        if existing != validated:
            raise ValueError("conflicting formal actual-v3 cache evidence")
        return copy.deepcopy(validated_cache)
    copied_entries = copy.deepcopy(entries)
    copied_lineage = copy.deepcopy(validated_cache["lineage"])
    lineage_payload = {
        "schema_version": LINEAGE_SCHEMA,
        "action": "append_actual_v3_terminal_evidence",
        "parent_cache_lineage_sha256": _sha(
            {"entries": copied_entries, "lineage": copied_lineage}
        ),
        "exact_cache_key_sha256": key,
        "evidence_sha256": validated["terminal_evidence_sha256"],
    }
    copied_entries[key] = validated
    return {
        "schema_version": CACHE_SCHEMA,
        "entries": copied_entries,
        "lineage": [
            *copied_lineage,
            {
                **lineage_payload,
                "lineage_record_sha256": _sha(lineage_payload),
            },
        ],
    }


def _failure_terminal(
    request: Mapping[str, Any], *, reason: str, consumes_budget: bool, kind: str
) -> dict[str, Any]:
    if not isinstance(reason, str) or not reason:
        raise ValueError("failure reason is required")
    request_sha = _verified_request_sha256(request)
    terminal = {
        "schema_version": "stage7_v2_terminal_failure_v1",
        "terminal_status": kind,
        "failure_reason": reason,
        "request_sha256": request_sha,
        "consumes_selected_event_budget": consumes_budget,
        "selected_event_budget_delta": int(consumes_budget),
        **{field: None for field in _METRIC_FIELDS},
    }
    if not consumes_budget:
        terminal["retry_request_sha256"] = request_sha
    return terminal


def finalize_candidate_failure(
    request: Mapping[str, Any], *, reason: str
) -> dict[str, Any]:
    """Record a candidate/backend/build/quant/numerical terminal failure."""
    if reason not in _CANDIDATE_FAILURE_REASONS:
        raise ValueError("candidate failure reason is not budget-consuming")
    return _failure_terminal(
        request, reason=reason, consumes_budget=True, kind="candidate_failure"
    )


def finalize_infrastructure_failure(
    request: Mapping[str, Any], *, reason: str
) -> dict[str, Any]:
    """Preserve the request identity for an infrastructure retry with zero cost."""
    if reason not in _INFRASTRUCTURE_FAILURE_REASONS:
        raise ValueError("infrastructure failure reason is invalid")
    return _failure_terminal(
        request, reason=reason, consumes_budget=False, kind="infrastructure_failure"
    )


def finalize_evidence_failure(
    request: Mapping[str, Any], *, reason: str
) -> dict[str, Any]:
    """Preserve the request identity for missing or invalid evidence retries."""
    if reason not in _EVIDENCE_FAILURE_REASONS:
        raise ValueError("evidence failure reason is invalid")
    return _failure_terminal(
        request, reason=reason, consumes_budget=False, kind="evidence_failure"
    )
