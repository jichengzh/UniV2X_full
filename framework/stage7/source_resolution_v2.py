"""Fail-closed source resolution contracts for Stage7 actual-feedback v3.

This pure module authenticates plans and source evidence but never launches the
Stage5 materializer, reads a measurement cache, or accepts objective labels.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from framework.stage5.measurement_plan_v1 import _source_plan_sha
from framework.stage5.measurement_plan_v2 import _validate_request
from framework.stage7.core_cache_v2 import validate_logical_request


PLAN_SCHEMA = "stage7_source_resolution_plan_v2"
FORMAL_RESULT_SCHEMA = "stage7_source_resolution_result_v2"
RETRY_SCHEMA = "stage7_source_resolution_retry_v2"
SYNTHETIC_SCHEMA = "stage7_source_resolution_synthetic_dryrun_v2"
MATERIALIZER_PATH = Path("/home/jichengzhi/V2X/scripts/stage5_materialize_round_sources_v1.sh")  # fmt: skip
MATERIALIZER_SHA256 = "d978e6287afc239c63471cadf729b37b4617bf46d1b8b8ea13cb4e2433bf4abe"

# fmt: off
_PLAN_FIELDS = set("schema_version status logical_request_sha256 ordered_row_ids ordered_row_sha256 row_count materializer rows source_resolution_plan_sha256".split())
_PLAN_ROW_FIELDS = set("candidate_id logical_row_index logical_row_sha256 model group_id width q_mode materialization_kind source_plan_sha256 source_contract_sha256 source_evidence_path checkpoint_path checkpoint_sha256 checkpoint_allowed_root onnx_path onnx_sha256 calibration_path calibration_summary_path source_key_sha256".split())
_EVIDENCE_FIELDS = set("schema_version group_id model width source_plan_sha256 checkpoint_path checkpoint_sha256 onnx_path onnx_sha256 calibration_path calibration_sha256 calibration_summary_path calibration_summary_sha256 status".split())
_FORMAL_ROW_FIELDS = set("candidate_id logical_row_index logical_row_sha256 source_key_sha256 source_evidence_path source_evidence_file_sha256 evidence_source_plan_sha256 checkpoint_path checkpoint_sha256 onnx_path onnx_sha256 calibration_path calibration_sha256 calibration_summary_path calibration_summary_sha256 resolved_source_sha256".split())
_FORMAL_FIELDS = set("schema_version status logical_request_sha256 source_resolution_plan_sha256 ordered_row_ids row_count rows synthetic_nonfinal actual_v3_hardware_evidence eligible_for_exact_cache_reveal eligible_for_cache_append eligible_for_finalization source_resolution_result_sha256".split())
_RETRY_FIELDS = set("schema_version status logical_request_sha256 source_resolution_plan_sha256 ordered_row_ids row_count retry_candidate_ids reason_code selected_event_budget_delta partial_reveal_allowed eligible_for_exact_cache_reveal eligible_for_cache_append eligible_for_finalization source_resolution_retry_sha256".split())
_SYNTHETIC_ROW_FIELDS = set("candidate_id logical_row_index logical_row_sha256 source_key_sha256 synthetic_checkpoint_sha256 synthetic_onnx_sha256 synthetic_resolved_source_sha256".split())
_SYNTHETIC_FIELDS = set("schema_version status logical_request_sha256 source_resolution_plan_sha256 ordered_row_ids row_count rows synthetic_nonfinal actual_v3_hardware_evidence eligible_for_protocol_exact_reveal eligible_for_exact_cache_reveal eligible_for_cache_append eligible_for_finalization synthetic_dryrun_result_sha256".split())
# fmt: on
_RETRY_REASONS = {
    "source_interrupted",
    "evidence_unavailable",
    "evidence_invalid",
    "infrastructure_unavailable",
}
_DEFAULT_MATERIALIZATION_KIND = {
    "pyramid": "pyramid_checkpoint_export",
    "codriving": "codriving_prepare_train_export",
    "fcooper": "fcooper_scanner_materialize_export",
}


def _sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any, *, allow_placeholder: bool = False) -> bool:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        return False
    return allow_placeholder or len(set(value)) > 1


def _verify_materializer() -> dict[str, str]:
    if (
        not MATERIALIZER_PATH.is_file()
        or _file_sha(MATERIALIZER_PATH) != MATERIALIZER_SHA256
    ):
        raise ValueError("frozen Stage5 source materializer SHA drift")
    return {"path": str(MATERIALIZER_PATH), "sha256": MATERIALIZER_SHA256}


def _canonical_field(name: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def _forbidden_field(name: Any) -> bool:
    field = _canonical_field(name)
    segments = tuple(segment for segment in field.split("_") if segment)
    metric = (
        "latency" in segments
        or "energy" in segments
        or "map" in segments
        or any(
            segment == "ap" or re.fullmatch(r"ap\d+", segment) for segment in segments
        )
    )
    return (
        metric
        or field == "terminal_status"
        or field.startswith("terminal_")
        or field == "failure_reason"
        or field.startswith("failure_")
        or field in {"cache", "objective", "objectives"}
        or field.startswith(("objective_", "cache_", "cached_"))
    )


def _forbidden_paths(value: Any, prefix: str = "") -> list[str]:
    if isinstance(value, Mapping):
        paths: list[str] = []
        for key, nested in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if _forbidden_field(key):
                paths.append(path)
            paths.extend(_forbidden_paths(nested, path))
        return paths
    if isinstance(value, (list, tuple)):
        return [
            path
            for index, nested in enumerate(value)
            for path in _forbidden_paths(nested, f"{prefix}[{index}]")
        ]
    return []


def _canonical_planned_path(value: Any, *, required: bool) -> str | None:
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError("source-resolution planned path is missing")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("source-resolution planned path escape")
    return str(path.resolve(strict=False))


def _optional_sha(value: Any, *, path: str | None) -> str | None:
    if value is None:
        return None
    if not _is_sha256(value):
        label = (
            "placeholder" if _is_sha256(value, allow_placeholder=True) else "invalid"
        )
        raise ValueError(f"source-resolution {label} source SHA")
    if path is None:
        raise ValueError("source-resolution SHA has no planned path")
    return str(value)


def _materialization_kind(row: Mapping[str, Any]) -> str:
    model = str(row.get("model") or "")
    return str(
        row.get("materialization_kind")
        or _DEFAULT_MATERIALIZATION_KIND.get(model)
        or ""
    )


def _evidence_path(source: Mapping[str, Any]) -> str:
    explicit = source.get("materialization_evidence_path")
    if explicit is not None:
        return str(_canonical_planned_path(explicit, required=True))
    marker = _canonical_planned_path(source.get("source_done_marker"), required=True)
    assert marker is not None
    if not marker.endswith(".done"):
        raise ValueError("source done marker must end in .done")
    return marker[:-5] + "_evidence.json"


def _checkpoint_allowed_root(
    source: Mapping[str, Any], checkpoint_path: str | None
) -> str | None:
    if checkpoint_path is not None:
        return str(Path(checkpoint_path).parent)
    model_dir = source.get("model_dir")
    return _canonical_planned_path(model_dir, required=False)


def _plan_row(
    row: Mapping[str, Any],
    *,
    row_index: int,
    logical_row_sha256: str,
) -> dict[str, Any]:
    leaks = _forbidden_paths(row)
    if leaks:
        raise ValueError("forbidden source-resolution label paths: " + ",".join(leaks))
    candidate_id = str(row.get("row_id") or row.get("manifest_job_id") or "")
    source = row.get("source_contract")
    if not candidate_id or not isinstance(source, Mapping):
        raise ValueError("source-resolution row identity/source contract missing")
    source_plan_sha = row.get("source_evidence_sha256")
    if not _is_sha256(source_plan_sha):
        label = (
            "placeholder"
            if _is_sha256(source_plan_sha, allow_placeholder=True)
            else "invalid"
        )
        raise ValueError(f"{label} source plan SHA")
    if _source_plan_sha(row) != source_plan_sha:
        raise ValueError("source plan SHA drift")
    width = [int(value) for value in row.get("width") or []]
    model = str(row.get("model") or "")
    group_id = str(row.get("group_id") or "")
    if len(width) != 3 or group_id != f"{model}|{'x'.join(map(str, width))}":
        raise ValueError("source-resolution group/width identity drift")
    checkpoint_path = _canonical_planned_path(
        source.get("checkpoint_path"), required=False
    )
    onnx_path = _canonical_planned_path(source.get("onnx_path"), required=False)
    calibration_path = _canonical_planned_path(
        source.get("calibration_npz"), required=True
    )
    summary_path = _canonical_planned_path(
        source.get("calibration_summary"), required=True
    )
    payload = {
        "candidate_id": candidate_id,
        "logical_row_index": row_index,
        "logical_row_sha256": logical_row_sha256,
        "model": model,
        "group_id": group_id,
        "width": width,
        "q_mode": str(row.get("q_mode") or ""),
        "materialization_kind": _materialization_kind(row),
        "source_plan_sha256": str(source_plan_sha),
        "source_contract_sha256": _sha(dict(source)),
        "source_evidence_path": _evidence_path(source),
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": _optional_sha(
            source.get("checkpoint_sha256"), path=checkpoint_path
        ),
        "checkpoint_allowed_root": _checkpoint_allowed_root(source, checkpoint_path),
        "onnx_path": onnx_path,
        "onnx_sha256": _optional_sha(source.get("onnx_sha256"), path=onnx_path),
        "calibration_path": calibration_path,
        "calibration_summary_path": summary_path,
    }
    source_key_payload = {
        key: copy.deepcopy(payload[key])
        for key in (
            "candidate_id",
            "logical_row_sha256",
            "model",
            "group_id",
            "width",
            "q_mode",
            "materialization_kind",
            "source_plan_sha256",
            "source_contract_sha256",
        )
    }
    return {**payload, "source_key_sha256": _sha(source_key_payload)}


def _unsigned_plan(logical_request: Mapping[str, Any]) -> dict[str, Any]:
    logical = validate_logical_request(logical_request)
    for row in logical["rows"]:
        source_plan = row.get("source_evidence_sha256")
        if not _is_sha256(source_plan):
            label = (
                "placeholder"
                if _is_sha256(source_plan, allow_placeholder=True)
                else "invalid"
            )
            raise ValueError(f"{label} source plan SHA")
    _validate_request(logical["request"])
    rows = [
        _plan_row(
            row,
            row_index=index,
            logical_row_sha256=logical["row_sha256"][candidate_id],
        )
        for index, (candidate_id, row) in enumerate(
            zip(logical["row_ids"], logical["rows"])
        )
    ]
    return {
        "schema_version": PLAN_SCHEMA,
        "status": "frozen",
        "logical_request_sha256": logical["logical_request_sha256"],
        "ordered_row_ids": logical["row_ids"],
        "ordered_row_sha256": [
            logical["row_sha256"][candidate_id] for candidate_id in logical["row_ids"]
        ],
        "row_count": 4,
        "materializer": _verify_materializer(),
        "rows": rows,
    }


def build_source_resolution_plan(
    logical_request: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze source-only identity before checkpoint/ONNX exact binding."""
    payload = _unsigned_plan(logical_request)
    if len(payload["rows"]) != 4:
        raise ValueError("source-resolution plan requires atomic four rows")
    return {
        **payload,
        "source_resolution_plan_sha256": _sha(payload),
    }


def _validate_plan_row(row: Any, *, expected_index: int) -> dict[str, Any]:
    if not isinstance(row, Mapping) or set(row) != _PLAN_ROW_FIELDS:
        raise ValueError("source-resolution plan row shape drift")
    copied = copy.deepcopy(dict(row))
    if copied.get("logical_row_index") != expected_index:
        raise ValueError("source-resolution ordered row index drift")
    sha_fields = (
        "logical_row_sha256",
        "source_plan_sha256",
        "source_contract_sha256",
        "source_key_sha256",
    )
    if any(not _is_sha256(copied.get(field)) for field in sha_fields):
        raise ValueError("source-resolution plan row SHA invalid/placeholder")
    for path_field, required in (
        ("source_evidence_path", True),
        ("checkpoint_path", False),
        ("checkpoint_allowed_root", False),
        ("onnx_path", False),
        ("calibration_path", True),
        ("calibration_summary_path", True),
    ):
        canonical = _canonical_planned_path(copied[path_field], required=required)
        if canonical != copied[path_field]:
            raise ValueError("source-resolution plan path drift")
    for sha_field, path_field in (
        ("checkpoint_sha256", "checkpoint_path"),
        ("onnx_sha256", "onnx_path"),
    ):
        if (
            _optional_sha(copied[sha_field], path=copied[path_field])
            != copied[sha_field]
        ):
            raise ValueError("source-resolution plan optional SHA drift")
    source_key_payload = {
        key: copy.deepcopy(copied[key])
        for key in (
            "candidate_id",
            "logical_row_sha256",
            "model",
            "group_id",
            "width",
            "q_mode",
            "materialization_kind",
            "source_plan_sha256",
            "source_contract_sha256",
        )
    }
    if copied["source_key_sha256"] != _sha(source_key_payload):
        raise ValueError("source-resolution source key SHA drift")
    return copied


def validate_source_resolution_plan(
    plan: Mapping[str, Any],
    *,
    logical_request: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Authenticate plan shape, order, materializer, and optional request."""
    if not isinstance(plan, Mapping) or set(plan) != _PLAN_FIELDS:
        raise ValueError("source-resolution plan shape drift")
    copied = copy.deepcopy(dict(plan))
    recorded = copied.pop("source_resolution_plan_sha256")
    if (
        copied.get("schema_version") != PLAN_SCHEMA
        or copied.get("status") != "frozen"
        or copied.get("row_count") != 4
        or copied.get("materializer") != _verify_materializer()
        or not _is_sha256(recorded)
        or recorded != _sha(copied)
    ):
        raise ValueError("source-resolution plan authentication failed")
    rows = copied.get("rows")
    ordered = copied.get("ordered_row_ids")
    ordered_sha = copied.get("ordered_row_sha256")
    if (
        not isinstance(rows, list)
        or len(rows) != 4
        or not isinstance(ordered, list)
        or len(ordered) != 4
        or len(set(map(str, ordered))) != 4
        or not isinstance(ordered_sha, list)
        or len(ordered_sha) != 4
    ):
        raise ValueError("source-resolution atomic ordered rows invalid")
    validated_rows = [
        _validate_plan_row(row, expected_index=index) for index, row in enumerate(rows)
    ]
    if [row["candidate_id"] for row in validated_rows] != ordered or [
        row["logical_row_sha256"] for row in validated_rows
    ] != ordered_sha:
        raise ValueError("source-resolution ordered row identity drift")
    if logical_request is not None:
        expected = _unsigned_plan(logical_request)
        if copied != expected:
            raise ValueError("source-resolution plan/logical request binding drift")
    return {**copied, "source_resolution_plan_sha256": recorded}


def _read_json_mapping(path: Path, *, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is unavailable or invalid") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"{name} payload must be a mapping")
    return copy.deepcopy(dict(payload))


def _validated_file(
    path_value: Any,
    sha_value: Any,
    *,
    expected_path: str | None,
    allowed_root: str | None = None,
) -> tuple[str, str]:
    if (
        not isinstance(path_value, str)
        or not Path(path_value).is_absolute()
        or ".." in Path(path_value).parts
    ):
        raise ValueError("source evidence artifact path escape")
    try:
        resolved = Path(path_value).resolve(strict=True)
    except OSError as error:
        raise ValueError("source evidence artifact path unavailable") from error
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise ValueError("source evidence artifact is not a regular file")
    if expected_path is not None:
        if resolved != Path(expected_path).resolve(strict=False):
            raise ValueError("source evidence artifact path drift")
    elif allowed_root is not None:
        try:
            resolved.relative_to(Path(allowed_root).resolve(strict=False))
        except ValueError as error:
            raise ValueError("source evidence artifact path escape") from error
    else:
        raise ValueError("source evidence artifact path is unbound")
    if not _is_sha256(sha_value):
        raise ValueError("source evidence artifact SHA invalid/placeholder")
    actual_sha = _file_sha(resolved)
    if actual_sha != sha_value:
        raise ValueError("source evidence artifact SHA mismatch")
    return str(resolved), actual_sha


def _formal_row(
    plan_row: Mapping[str, Any],
    *,
    evidence_path_value: Any,
) -> dict[str, Any]:
    evidence_path, evidence_file_sha = _validated_file(
        evidence_path_value,
        _file_sha(Path(str(evidence_path_value))),
        expected_path=str(plan_row["source_evidence_path"]),
    )
    evidence = _read_json_mapping(
        Path(evidence_path), name="source materialization evidence"
    )
    if set(evidence) != _EVIDENCE_FIELDS:
        raise ValueError("source materialization evidence shape drift")
    expected_width = "x".join(map(str, plan_row["width"]))
    if (
        evidence.get("schema_version") != "stage5_source_materialization_evidence_v1"
        or evidence.get("status") != "ready"
        or evidence.get("model") != plan_row["model"]
        or evidence.get("group_id") != plan_row["group_id"]
        or evidence.get("width") != expected_width
        or evidence.get("source_plan_sha256") != plan_row["source_plan_sha256"]
    ):
        raise ValueError("source materialization evidence contract drift")
    checkpoint_path, checkpoint_sha = _validated_file(
        evidence["checkpoint_path"],
        evidence["checkpoint_sha256"],
        expected_path=plan_row["checkpoint_path"],
        allowed_root=plan_row["checkpoint_allowed_root"],
    )
    onnx_path, onnx_sha = _validated_file(
        evidence["onnx_path"],
        evidence["onnx_sha256"],
        expected_path=plan_row["onnx_path"],
    )
    if (
        plan_row["checkpoint_sha256"] is not None
        and checkpoint_sha != plan_row["checkpoint_sha256"]
    ):
        raise ValueError("source evidence planned source SHA drift")
    if plan_row["onnx_sha256"] is not None and onnx_sha != plan_row["onnx_sha256"]:
        raise ValueError("source evidence planned source SHA drift")
    calibration_path, calibration_sha = _validated_file(
        evidence["calibration_path"],
        evidence["calibration_sha256"],
        expected_path=plan_row["calibration_path"],
    )
    summary_path, summary_sha = _validated_file(
        evidence["calibration_summary_path"],
        evidence["calibration_summary_sha256"],
        expected_path=plan_row["calibration_summary_path"],
    )
    payload = {
        "candidate_id": plan_row["candidate_id"],
        "logical_row_index": plan_row["logical_row_index"],
        "logical_row_sha256": plan_row["logical_row_sha256"],
        "source_key_sha256": plan_row["source_key_sha256"],
        "source_evidence_path": evidence_path,
        "source_evidence_file_sha256": evidence_file_sha,
        "evidence_source_plan_sha256": evidence["source_plan_sha256"],
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": checkpoint_sha,
        "onnx_path": onnx_path,
        "onnx_sha256": onnx_sha,
        "calibration_path": calibration_path,
        "calibration_sha256": calibration_sha,
        "calibration_summary_path": summary_path,
        "calibration_summary_sha256": summary_sha,
    }
    return {
        **payload,
        "resolved_source_sha256": _sha(payload),
    }


def _unsigned_formal_result(
    plan: Mapping[str, Any],
    evidence_paths_by_candidate: Mapping[str, Any],
) -> dict[str, Any]:
    validated_plan = validate_source_resolution_plan(plan)
    ordered = validated_plan["ordered_row_ids"]
    if not isinstance(evidence_paths_by_candidate, Mapping) or set(
        evidence_paths_by_candidate
    ) != set(ordered):
        raise ValueError("formal source evidence candidate coverage drift")
    rows = [
        _formal_row(
            plan_row,
            evidence_path_value=evidence_paths_by_candidate[candidate_id],
        )
        for candidate_id, plan_row in zip(ordered, validated_plan["rows"])
    ]
    return {
        "schema_version": FORMAL_RESULT_SCHEMA,
        "status": "all_ready",
        "logical_request_sha256": validated_plan["logical_request_sha256"],
        "source_resolution_plan_sha256": validated_plan[
            "source_resolution_plan_sha256"
        ],
        "ordered_row_ids": copy.deepcopy(ordered),
        "row_count": 4,
        "rows": rows,
        "synthetic_nonfinal": False,
        "actual_v3_hardware_evidence": False,
        "eligible_for_exact_cache_reveal": True,
        "eligible_for_cache_append": False,
        "eligible_for_finalization": False,
    }


def build_formal_source_resolution_result(
    plan: Mapping[str, Any],
    *,
    evidence_paths_by_candidate: Mapping[str, Any],
) -> dict[str, Any]:
    """Build an all-or-nothing formal result from actual Stage5 evidence."""
    payload = _unsigned_formal_result(plan, evidence_paths_by_candidate)
    return {
        **payload,
        "source_resolution_result_sha256": _sha(payload),
    }


def validate_formal_source_resolution_result(
    result: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Re-read every artifact and reject synthetic or partial results."""
    if (
        not isinstance(result, Mapping)
        or result.get("schema_version") != FORMAL_RESULT_SCHEMA
    ):
        raise ValueError("formal source-resolution result schema required")
    if set(result) != _FORMAL_FIELDS:
        raise ValueError("formal source-resolution result shape drift")
    copied = copy.deepcopy(dict(result))
    recorded = copied.pop("source_resolution_result_sha256")
    if not _is_sha256(recorded) or recorded != _sha(copied):
        raise ValueError("formal source-resolution result SHA drift")
    rows = copied.get("rows")
    ordered = copied.get("ordered_row_ids")
    if (
        copied.get("status") != "all_ready"
        or copied.get("row_count") != 4
        or copied.get("synthetic_nonfinal") is not False
        or copied.get("actual_v3_hardware_evidence") is not False
        or copied.get("eligible_for_exact_cache_reveal") is not True
        or copied.get("eligible_for_cache_append") is not False
        or copied.get("eligible_for_finalization") is not False
        or not isinstance(rows, list)
        or len(rows) != 4
        or not all(
            isinstance(row, Mapping) and set(row) == _FORMAL_ROW_FIELDS for row in rows
        )
        or not isinstance(ordered, list)
        or [row["candidate_id"] for row in rows] != ordered
        or len(set(ordered)) != 4
    ):
        raise ValueError("formal source-resolution atomic/ordered/duplicate row drift")
    validated_plan = validate_source_resolution_plan(plan)
    evidence_paths = {row["candidate_id"]: row["source_evidence_path"] for row in rows}
    expected = _unsigned_formal_result(validated_plan, evidence_paths)
    if copied != expected:
        raise ValueError("formal source-resolution result semantic drift")
    return {**copied, "source_resolution_result_sha256": recorded}


def build_source_resolution_retry_result(
    plan: Mapping[str, Any],
    *,
    retry_candidate_ids: Sequence[str],
    reason_code: str,
) -> dict[str, Any]:
    """Freeze a zero-budget retry without exposing partially ready sources."""
    validated_plan = validate_source_resolution_plan(plan)
    requested = [str(value) for value in retry_candidate_ids]
    ordered = validated_plan["ordered_row_ids"]
    if (
        not requested
        or len(set(requested)) != len(requested)
        or not set(requested).issubset(set(ordered))
        or reason_code not in _RETRY_REASONS
    ):
        raise ValueError("source-resolution retry identity/reason invalid")
    payload = {
        "schema_version": RETRY_SCHEMA,
        "status": "retry_required",
        "logical_request_sha256": validated_plan["logical_request_sha256"],
        "source_resolution_plan_sha256": validated_plan[
            "source_resolution_plan_sha256"
        ],
        "ordered_row_ids": copy.deepcopy(ordered),
        "row_count": 4,
        "retry_candidate_ids": [
            candidate_id for candidate_id in ordered if candidate_id in set(requested)
        ],
        "reason_code": reason_code,
        "selected_event_budget_delta": 0,
        "partial_reveal_allowed": False,
        "eligible_for_exact_cache_reveal": False,
        "eligible_for_cache_append": False,
        "eligible_for_finalization": False,
    }
    return {**payload, "source_resolution_retry_sha256": _sha(payload)}


def validate_source_resolution_retry_result(
    result: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate immutable retry identity and zero-budget/no-reveal flags."""
    if not isinstance(result, Mapping) or set(result) != _RETRY_FIELDS:
        raise ValueError("source-resolution retry result shape drift")
    copied = copy.deepcopy(dict(result))
    recorded = copied.pop("source_resolution_retry_sha256")
    if not _is_sha256(recorded) or recorded != _sha(copied):
        raise ValueError("source-resolution retry result SHA drift")
    validated_plan = validate_source_resolution_plan(plan)
    retry_ids = copied.get("retry_candidate_ids")
    if (
        copied.get("schema_version") != RETRY_SCHEMA
        or copied.get("status") != "retry_required"
        or copied.get("logical_request_sha256")
        != validated_plan["logical_request_sha256"]
        or copied.get("source_resolution_plan_sha256")
        != validated_plan["source_resolution_plan_sha256"]
        or copied.get("ordered_row_ids") != validated_plan["ordered_row_ids"]
        or copied.get("row_count") != 4
        or not isinstance(retry_ids, list)
        or not retry_ids
        or retry_ids
        != [
            candidate_id
            for candidate_id in validated_plan["ordered_row_ids"]
            if candidate_id in set(retry_ids)
        ]
        or copied.get("reason_code") not in _RETRY_REASONS
        or copied.get("selected_event_budget_delta") != 0
        or copied.get("partial_reveal_allowed") is not False
        or copied.get("eligible_for_exact_cache_reveal") is not False
        or copied.get("eligible_for_cache_append") is not False
        or copied.get("eligible_for_finalization") is not False
    ):
        raise ValueError("source-resolution retry contract drift")
    return {**copied, "source_resolution_retry_sha256": recorded}


def _unsigned_synthetic(plan: Mapping[str, Any]) -> dict[str, Any]:
    validated_plan = validate_source_resolution_plan(plan)
    rows = []
    for plan_row in validated_plan["rows"]:
        base = {
            "candidate_id": plan_row["candidate_id"],
            "logical_row_index": plan_row["logical_row_index"],
            "logical_row_sha256": plan_row["logical_row_sha256"],
            "source_key_sha256": plan_row["source_key_sha256"],
        }
        rows.append(
            {
                **base,
                "synthetic_checkpoint_sha256": _sha(
                    {**base, "kind": "synthetic_checkpoint"}
                ),
                "synthetic_onnx_sha256": _sha({**base, "kind": "synthetic_onnx"}),
                "synthetic_resolved_source_sha256": _sha(
                    {**base, "kind": "synthetic_resolved_source"}
                ),
            }
        )
    return {
        "schema_version": SYNTHETIC_SCHEMA,
        "status": "synthetic_protocol_ready",
        "logical_request_sha256": validated_plan["logical_request_sha256"],
        "source_resolution_plan_sha256": validated_plan[
            "source_resolution_plan_sha256"
        ],
        "ordered_row_ids": copy.deepcopy(validated_plan["ordered_row_ids"]),
        "row_count": 4,
        "rows": rows,
        "synthetic_nonfinal": True,
        "actual_v3_hardware_evidence": False,
        "eligible_for_protocol_exact_reveal": True,
        "eligible_for_exact_cache_reveal": False,
        "eligible_for_cache_append": False,
        "eligible_for_finalization": False,
    }


def build_synthetic_dryrun_source_resolution_result(
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Build an isolated protocol fixture that cannot become formal truth."""
    payload = _unsigned_synthetic(plan)
    return {**payload, "synthetic_dryrun_result_sha256": _sha(payload)}


def validate_synthetic_dryrun_source_resolution_result(
    result: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the separate synthetic protocol-only schema."""
    if not isinstance(result, Mapping) or set(result) != _SYNTHETIC_FIELDS:
        raise ValueError("synthetic source-resolution result shape drift")
    copied = copy.deepcopy(dict(result))
    recorded = copied.pop("synthetic_dryrun_result_sha256")
    if not _is_sha256(recorded) or recorded != _sha(copied):
        raise ValueError("synthetic source-resolution result SHA drift")
    rows = copied.get("rows")
    if (
        copied.get("schema_version") != SYNTHETIC_SCHEMA
        or copied.get("status") != "synthetic_protocol_ready"
        or copied.get("row_count") != 4
        or copied.get("synthetic_nonfinal") is not True
        or copied.get("actual_v3_hardware_evidence") is not False
        or copied.get("eligible_for_protocol_exact_reveal") is not True
        or copied.get("eligible_for_exact_cache_reveal") is not False
        or copied.get("eligible_for_cache_append") is not False
        or copied.get("eligible_for_finalization") is not False
        or not isinstance(rows, list)
        or len(rows) != 4
        or not all(
            isinstance(row, Mapping) and set(row) == _SYNTHETIC_ROW_FIELDS
            for row in rows
        )
    ):
        raise ValueError("synthetic source-resolution isolation drift")
    expected = _unsigned_synthetic(plan)
    if copied != expected:
        raise ValueError("synthetic source-resolution semantic drift")
    return {**copied, "synthetic_dryrun_result_sha256": recorded}


__all__ = [
    "build_source_resolution_plan",
    "validate_source_resolution_plan",
    "build_formal_source_resolution_result",
    "validate_formal_source_resolution_result",
    "build_source_resolution_retry_result",
    "validate_source_resolution_retry_result",
    "build_synthetic_dryrun_source_resolution_result",
    "validate_synthetic_dryrun_source_resolution_result",
]
