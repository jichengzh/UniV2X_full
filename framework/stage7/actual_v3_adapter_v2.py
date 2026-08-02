"""Pure Stage7 bindings around the reviewed actual-feedback-v3 execution path.

The functions in this module only authenticate and transform manifests.  They
never inspect a cache before selection is frozen, execute hardware, or
reimplement Stage3 measurement/AP behavior.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from framework.stage7 import core_cache_v2 as cache_v2
from scripts import stage3_execute_ap_plan_v3 as stage3_ap_executor
from scripts import stage3_execute_performance_plan_v3 as stage3_performance_executor


PHYSICAL_REQUEST_SCHEMA = "stage7_actual_v3_miss_only_physical_request_v2"
TERMINAL_WRAPPER_SCHEMA = cache_v2.ACTUAL_V3_TERMINAL_EVIDENCE_SCHEMA
FAILURE_SCHEMA = "stage7_actual_v3_terminal_failure_v2"
FROZEN_STAGE3_AP_EXECUTOR_SHA256 = (
    "da71d1ba94480b9034c9dc3b16011ba0bba83166e478f865934b80b50dbecaff"
)
_FAILURE_REASON_BY_CLASS = {
    "candidate": {
        "backend_capability_failure",
        "backend_failure",
        "build_failure",
        "unsupported_precision",
        "quantization_failure",
        "numerical_failure",
        "candidate_runtime_capability_failure",
    },
    "infrastructure": {
        "gpu_occupancy_drift",
        "gpu_unavailable",
        "unrelated_process_oom",
        "contention",
        "ssh_failure",
        "network_failure",
        "missing_source_artifact",
        "permission_failure",
        "runner_bug",
    },
    "evidence": {
        "terminal_artifact_sha_mismatch",
        "terminal_evidence_sha_mismatch",
        "evidence_schema_mismatch",
        "evidence_missing",
    },
}
_FAILURE_STATUS = {
    "candidate": "candidate_failure",
    "infrastructure": "infrastructure_failure",
    "evidence": "evidence_failure",
}


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
    return str(row.get("row_id") or row.get("manifest_job_id") or "")


def bind_selector_output(
    selection: Mapping[str, Any],
    *,
    exact_dimensions_by_candidate: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Freeze the ordered selector request and its complete exact-key binding."""
    if not isinstance(selection, Mapping):
        raise ValueError("selector output must be a mapping")
    request = selection.get("measurement_request")
    acquisition = selection.get("acquisition")
    if not isinstance(request, Mapping) or not isinstance(acquisition, Mapping):
        raise ValueError("selector output lacks acquisition or logical request")
    logical = cache_v2.validate_logical_request(request)
    row_ids = logical["row_ids"]
    selected_ids = [
        str(value) for value in acquisition.get("selected_row_ids") or ()
    ]
    if len(row_ids) != 4 or selected_ids != row_ids:
        raise ValueError("selector and four-row logical request order drift")
    if set(exact_dimensions_by_candidate) != set(row_ids):
        raise ValueError("exact dimension candidate coverage drift")

    selected_candidates = []
    rows_by_id = {
        _row_id(row): copy.deepcopy(dict(row))
        for row in logical["rows"]
    }
    semantic_fields = (
        "model",
        "genome",
        "q_mode",
        "capability_profile_id",
        "hardware_id",
        "dispatch_key",
    )
    for candidate_id in row_ids:
        dimensions = copy.deepcopy(
            dict(exact_dimensions_by_candidate[candidate_id])
        )
        row = rows_by_id[candidate_id]
        for field in semantic_fields:
            if copy.deepcopy(row.get(field)) != copy.deepcopy(
                dimensions.get(field)
            ):
                raise ValueError(
                    "logical row exact identity mismatch: "
                    f"{candidate_id}:{field}"
                )
        if dimensions.get("candidate_id") != candidate_id:
            raise ValueError(
                "logical row exact identity mismatch: "
                f"{candidate_id}:candidate_id"
            )
        key = cache_v2.build_v2_exact_cache_key(dimensions)
        cross_payload = {
            "candidate_id": candidate_id,
            "logical_row_sha256": logical["row_sha256"][candidate_id],
            "exact_key_dimensions": dimensions,
            "exact_cache_key_sha256": key,
        }
        selected_candidates.append(
            {
                **cross_payload,
                "logical_exact_binding_sha256": _sha(cross_payload),
            }
        )
    payload = {
        "schema_version": cache_v2.SELECTION_BINDING_SCHEMA,
        "selection_frozen": True,
        "logical_request_sha256": logical["logical_request_sha256"],
        "selected_candidates": selected_candidates,
        "selected_candidate_ids_sha256": _sha(row_ids),
    }
    binding = {
        **payload,
        "selection_binding_sha256": _sha(payload),
    }
    cache_v2.validate_selection_binding(request, binding)
    return {
        "logical_request": copy.deepcopy(dict(request)),
        "selection_binding": binding,
    }


def _validate_reveal(
    logical: Mapping[str, Any],
    binding: Mapping[str, Any],
    reveal: Mapping[str, Any],
    cache_snapshot: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    verified = cache_v2.validate_selection_binding(logical, binding)
    if not isinstance(reveal, Mapping):
        raise ValueError("cache reveal must be a mapping")
    copied = copy.deepcopy(dict(reveal))
    expected = cache_v2.reveal_v2_cache_after_selection(
        logical, binding, cache_snapshot
    )
    if copied != expected:
        raise ValueError(
            "cache snapshot/reveal authentication failed"
        )
    entries = copied.get("entries")
    if not isinstance(entries, list) or len(entries) != len(
        verified["selected_candidates"]
    ):
        raise ValueError("cache reveal row count drift")
    for selected, entry in zip(verified["selected_candidates"], entries):
        if (
            not isinstance(entry, Mapping)
            or entry.get("candidate_id") != selected["candidate_id"]
            or entry.get("logical_row_sha256")
            != selected["logical_row_sha256"]
            or entry.get("logical_exact_binding_sha256")
            != selected["logical_exact_binding_sha256"]
            or entry.get("exact_cache_key_sha256")
            != selected["exact_cache_key_sha256"]
            or entry.get("disposition") not in {"hit", "miss"}
            or not _is_sha256(
                entry.get("reveal_use_binding_sha256")
            )
        ):
            raise ValueError("cache reveal row binding drift")
        if entry["disposition"] == "hit":
            terminal = entry.get("terminal_evidence")
            try:
                validated_terminal = validate_actual_v3_terminal_wrapper(
                    terminal
                )
            except (OSError, ValueError) as error:
                raise ValueError(
                    "actual-v3 hit terminal evidence is invalid"
                ) from error
            if (
                validated_terminal["candidate_id"]
                != selected["candidate_id"]
                or validated_terminal["exact_cache_key_sha256"]
                != selected["exact_cache_key_sha256"]
                or validated_terminal["terminal_evidence_sha256"]
                != entry["terminal_evidence_sha256"]
            ):
                raise ValueError(
                    "actual-v3 hit terminal evidence binding drift"
                )
    return verified, [copy.deepcopy(dict(entry)) for entry in entries]


def build_miss_only_physical_plan(
    logical_request: Mapping[str, Any],
    selection_binding: Mapping[str, Any],
    reveal: Mapping[str, Any],
    *,
    cache_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive a zero-to-four-row physical plan without changing logical budget."""
    verified, reveal_entries = _validate_reveal(
        logical_request, selection_binding, reveal, cache_snapshot
    )
    rows_by_id = {
        _row_id(row): copy.deepcopy(dict(row))
        for row in verified["logical_request"]["rows"]
    }
    physical_rows = [
        rows_by_id[entry["candidate_id"]]
        for entry in reveal_entries
        if entry["disposition"] == "miss"
    ]
    physical_row_sha = {
        _row_id(row): _sha(row)
        for row in physical_rows
    }
    payload = {
        "schema_version": PHYSICAL_REQUEST_SCHEMA,
        "logical_request_sha256": verified["logical_request"][
            "logical_request_sha256"
        ],
        "selection_binding_sha256": verified["selection_binding_sha256"],
        "cache_snapshot_sha256": reveal["cache_snapshot_sha256"],
        "lineage_head_sha256": reveal["lineage_head_sha256"],
        "cache_reveal_sha256": reveal["cache_reveal_sha256"],
        "logical_row_count": len(reveal_entries),
        "physical_row_count": len(physical_rows),
        "row_sha256": physical_row_sha,
        "rows": physical_rows,
    }
    physical_sha = _sha(payload)
    logical_bindings = [
        {
            "logical_row_index": index,
            "candidate_id": selected["candidate_id"],
            "logical_row_sha256": selected["logical_row_sha256"],
            "logical_exact_binding_sha256": selected[
                "logical_exact_binding_sha256"
            ],
            "exact_cache_key_sha256": selected["exact_cache_key_sha256"],
            "disposition": entry["disposition"],
            "physical_row_sha256": physical_row_sha.get(
                selected["candidate_id"]
            ),
        }
        for index, (selected, entry) in enumerate(zip(
            verified["selected_candidates"], reveal_entries
        ))
    ]
    return {
        **payload,
        "logical_row_bindings": logical_bindings,
        "physical_request_sha256": physical_sha,
    }


def _validate_artifact(
    artifact: Mapping[str, Any], *, name: str, artifact_kind: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(artifact, Mapping):
        raise ValueError(f"{name} artifact must be a mapping")
    copied = copy.deepcopy(dict(artifact))
    if (
        set(copied) != {"artifact_kind", "path", "artifact_sha256"}
        or copied.get("artifact_kind") != artifact_kind
        or not _is_sha256(copied.get("artifact_sha256"))
    ):
        raise ValueError(f"{name} artifact reference is invalid")
    path = Path(str(copied.get("path") or ""))
    try:
        encoded = path.read_bytes()
    except OSError as error:
        raise ValueError(f"{name} artifact file is unavailable") from error
    if hashlib.sha256(encoded).hexdigest() != copied["artifact_sha256"]:
        raise ValueError(f"{name} artifact file SHA mismatch")
    try:
        payload = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} artifact JSON is invalid") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"{name} artifact payload must be a mapping")
    return copied, copy.deepcopy(dict(payload))


_RAW_REFERENCE_FIELDS = {
    "artifact_kind",
    "raw_artifact_path",
    "raw_artifact_sha256",
    "executor_state_path",
    "executor_state_sha256",
    "job_manifest_path",
    "job_manifest_sha256",
}
_PERFORMANCE_RAW_REFERENCE_FIELDS = {
    *_RAW_REFERENCE_FIELDS,
    "performance_jobs_path",
    "performance_jobs_sha256",
}


def _read_authenticated_file(
    path_value: Any, sha_value: Any, *, name: str
) -> bytes:
    path = Path(str(path_value or ""))
    expected = str(sha_value or "")
    if not _is_sha256(expected):
        raise ValueError(f"{name} SHA is invalid")
    try:
        encoded = path.read_bytes()
    except OSError as error:
        raise ValueError(f"{name} file is unavailable") from error
    if hashlib.sha256(encoded).hexdigest() != expected:
        raise ValueError(f"{name} file SHA mismatch")
    return encoded


def _json_mapping(encoded: bytes, *, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} JSON is invalid") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"{name} payload must be a mapping")
    return copy.deepcopy(dict(payload))


def _jsonl_mappings(
    encoded: bytes, *, name: str
) -> list[dict[str, Any]]:
    try:
        lines = encoded.decode("utf-8").splitlines()
        rows = [json.loads(line) for line in lines if line.strip()]
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} JSONL is invalid") from error
    if not rows or not all(isinstance(row, Mapping) for row in rows):
        raise ValueError(f"{name} rows are invalid")
    return [copy.deepcopy(dict(row)) for row in rows]


def _raw_reference(
    artifact: Mapping[str, Any],
    *,
    name: str,
    artifact_kind: str,
    reference_fields: set[str],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], bytes]:
    if not isinstance(artifact, Mapping):
        raise ValueError(f"{name} raw Stage3 reference must be a mapping")
    copied = copy.deepcopy(dict(artifact))
    if (
        set(copied) != reference_fields
        or copied.get("artifact_kind") != artifact_kind
    ):
        raise ValueError(f"{name} raw Stage3 reference is invalid")
    raw = _json_mapping(
        _read_authenticated_file(
            copied["raw_artifact_path"],
            copied["raw_artifact_sha256"],
            name=f"{name} raw Stage3 artifact",
        ),
        name=f"{name} raw Stage3 artifact",
    )
    state = _jsonl_mappings(
        _read_authenticated_file(
            copied["executor_state_path"],
            copied["executor_state_sha256"],
            name=f"{name} executor state",
        ),
        name=f"{name} executor state",
    )
    manifest_bytes = _read_authenticated_file(
        copied["job_manifest_path"],
        copied["job_manifest_sha256"],
        name=f"{name} job manifest",
    )
    return copied, raw, state, manifest_bytes


def _validate_frozen_stage3_ap_executor() -> None:
    source_path = Path(str(stage3_ap_executor.__file__ or ""))
    try:
        source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    except OSError as error:
        raise ValueError(
            "frozen Stage3 AP executor source is unavailable"
        ) from error
    if source_sha256 != FROZEN_STAGE3_AP_EXECUTOR_SHA256:
        raise ValueError("frozen Stage3 AP executor source SHA drift")


def _semantic_drift(
    actual: Any,
    expected: Any,
) -> bool:
    return copy.deepcopy(actual) != copy.deepcopy(expected)


def _validate_present_exact_dimensions(
    source: Mapping[str, Any],
    dimensions: Mapping[str, Any],
    *,
    candidate_id: str | None,
    identity_fields: tuple[str, ...] = (),
    name: str = "Stage3 performance manifest",
) -> None:
    if candidate_id is not None:
        for field in identity_fields:
            if field in source and str(source[field]) != candidate_id:
                raise ValueError(f"{name} semantic drift")

    for field in cache_v2.EXACT_CACHE_KEY_DIMENSIONS:
        if field in source and _semantic_drift(
            source[field], dimensions[field]
        ):
            raise ValueError(f"{name} semantic drift")

    genome = list(dimensions["genome"])
    aliases = {
        "width": genome[:3],
        "widths": genome[:3],
        "precision": dimensions["q_mode"],
        "q": dimensions["q_mode"],
        "profile": dimensions["capability_profile_id"],
        "batch": dimensions["batch_size"],
        "checkpoint_sha256": dimensions["source_checkpoint_sha256"],
    }
    for field, expected in aliases.items():
        if field in source and _semantic_drift(source[field], expected):
            raise ValueError(f"{name} semantic drift")

    source_contract = source.get("source_contract")
    if source_contract is not None:
        if not isinstance(source_contract, Mapping):
            raise ValueError(f"{name} semantic drift")
        _validate_present_exact_dimensions(
            source_contract,
            dimensions,
            candidate_id=None,
            name=name,
        )


def _validate_stage3_performance_lineage(
    artifact: Mapping[str, Any],
    *,
    candidate_id: str,
    dimensions: Mapping[str, Any],
    stage5_terminal: Mapping[str, Any],
    logical_request_sha256: str | None,
) -> dict[str, Any]:
    copied, raw, states, manifest_bytes = _raw_reference(
        artifact,
        name="Stage3 performance",
        artifact_kind="stage3_performance_raw_v3",
        reference_fields=_PERFORMANCE_RAW_REFERENCE_FIELDS,
    )
    performance_jobs = _jsonl_mappings(
        _read_authenticated_file(
            copied["performance_jobs_path"],
            copied["performance_jobs_sha256"],
            name="Stage3 performance jobs",
        ),
        name="Stage3 performance jobs",
    )
    manifest = _json_mapping(
        manifest_bytes, name="Stage3 performance job manifest"
    )
    jobs = manifest.get("jobs")
    if (
        manifest.get("schema_version")
        != "stage5_performance_manifest_v2"
        or not isinstance(jobs, list)
        or (
            logical_request_sha256 is not None
            and manifest.get("source_request_sha256")
            != logical_request_sha256
        )
    ):
        raise ValueError("Stage3 performance manifest contract drift")
    matching_jobs = [
        dict(job)
        for job in jobs
        if isinstance(job, Mapping)
        and str(job.get("manifest_job_id") or "") == candidate_id
    ]
    if len(matching_jobs) != 1:
        raise ValueError("Stage3 performance manifest identity drift")
    job = matching_jobs[0]
    _validate_present_exact_dimensions(
        manifest,
        dimensions,
        candidate_id=None,
    )
    _validate_present_exact_dimensions(
        job,
        dimensions,
        candidate_id=candidate_id,
        identity_fields=(
            "candidate_id",
            "manifest_job_id",
            "row_id",
            "job_id",
        ),
    )
    expected_performance_runner = {
        "fp16": "tvm_fp16",
        "int8": "tvm_int8",
    }.get(str(dimensions.get("q_mode") or ""))
    matching_execution_jobs = [
        dict(execution_job)
        for execution_job in performance_jobs
        if execution_job.get("schema_version")
        == "stage5_performance_job_v1"
        and str(execution_job.get("manifest_job_id") or "")
        == candidate_id
        and execution_job.get("runner_key")
        == expected_performance_runner
    ]
    if len(matching_execution_jobs) != 1:
        raise ValueError("Stage3 performance execution job identity drift")
    execution_job = matching_execution_jobs[0]
    _validate_present_exact_dimensions(
        execution_job,
        dimensions,
        candidate_id=candidate_id,
        identity_fields=("candidate_id", "manifest_job_id", "row_id"),
        name="Stage3 performance execution job",
    )
    if (
        not str(execution_job.get("job_id") or "")
        or execution_job.get("source_contract")
        != job.get("source_contract")
    ):
        raise ValueError("Stage3 performance execution job semantic drift")
    job_id = str(execution_job["job_id"])
    terminal_states = [
        row
        for row in states
        if row.get("schema_version")
        == "stage3_execute_performance_plan_v3_state"
        and str(row.get("job_id") or "") == job_id
        and row.get("status") == "success"
    ]
    if len(terminal_states) != 1:
        raise ValueError("Stage3 performance executor terminal drift")
    terminal = terminal_states[0]
    if (
        terminal.get("returncode") != 0
        or terminal.get("result_json") != copied["raw_artifact_path"]
        or terminal.get("result_sha256")
        != copied["raw_artifact_sha256"]
        or stage5_terminal.get("performance_result_json")
        != copied["raw_artifact_path"]
        or stage5_terminal.get("performance_result_sha256")
        != copied["raw_artifact_sha256"]
    ):
        raise ValueError("Stage3 performance raw lineage drift")
    expected_schema = {
        "fp16": "route_b_fp16_auto_result_v1",
        "int8": "route_b_int8_auto_decomp_result_v1",
    }.get(str(dimensions.get("q_mode") or ""))
    success, correctness = (
        stage3_performance_executor._extract_success_and_correctness(raw)
    )
    latency = stage3_performance_executor._extract_latency(raw)
    energy = stage3_performance_executor._extract_energy(raw)
    if (
        raw.get("schema") != expected_schema
        or not success
        or not correctness
        or latency != stage5_terminal.get("latency_ms")
        or energy != stage5_terminal.get("energy_j")
    ):
        raise ValueError("Stage3 performance raw contract drift")
    return {
        "reference": copied,
        "manifest": manifest,
        "manifest_job": job,
        "execution_job": execution_job,
        "terminal_state": copy.deepcopy(dict(terminal)),
        "raw": raw,
    }


def _validate_stage3_ap_lineage(
    artifact: Mapping[str, Any],
    *,
    candidate_id: str,
    dimensions: Mapping[str, Any],
    stage5_terminal: Mapping[str, Any],
    performance_lineage: Mapping[str, Any],
) -> dict[str, Any]:
    copied, raw, states, manifest_bytes = _raw_reference(
        artifact,
        name="Stage3 AP",
        artifact_kind="stage3_ap_raw_v3",
        reference_fields=_RAW_REFERENCE_FIELDS,
    )
    plans = _jsonl_mappings(
        manifest_bytes, name="Stage3 AP job manifest"
    )
    matching_plans = [
        row
        for row in plans
        if row.get("schema_version") == "stage5_ap_plan_v2"
        and str(row.get("manifest_job_id") or "") == candidate_id
    ]
    if len(matching_plans) != 1:
        raise ValueError("Stage3 AP manifest identity drift")
    plan = matching_plans[0]
    expected_runner = {
        "fp16": "pyramid_tvm_fp16_bridge",
        "int8": "pyramid_tvm_int8_numeric_gate",
    }.get(str(dimensions.get("q_mode") or ""))
    _validate_present_exact_dimensions(
        plan,
        dimensions,
        candidate_id=candidate_id,
        identity_fields=("candidate_id", "manifest_job_id", "row_id"),
        name="Stage3 AP manifest",
    )
    performance_reference = performance_lineage["reference"]
    performance_manifest_job = performance_lineage["manifest_job"]
    performance_execution_job = performance_lineage["execution_job"]
    performance_terminal = performance_lineage["terminal_state"]
    if (
        plan.get("model") != dimensions.get("model")
        or (plan.get("q") or plan.get("q_mode"))
        != dimensions.get("q_mode")
        or plan.get("runner_key") != expected_runner
        or plan.get("ap_terminal") != "ready"
        or plan.get("performance_terminal") != "success"
        or plan.get("performance_job_id")
        != performance_execution_job.get("job_id")
        or plan.get("performance_result_json")
        != performance_reference["raw_artifact_path"]
        or (
            "performance_result_sha256" in plan
            and plan.get("performance_result_sha256")
            != performance_reference["raw_artifact_sha256"]
        )
        or performance_terminal.get("result_json")
        != plan.get("performance_result_json")
        or performance_terminal.get("result_sha256")
        != performance_reference["raw_artifact_sha256"]
        or plan.get("source_contract")
        != performance_manifest_job.get("source_contract")
        or (
            "required_metrics" in plan
            and "required_metrics" in performance_manifest_job
            and plan.get("required_metrics")
            != performance_manifest_job.get("required_metrics")
        )
    ):
        raise ValueError("Stage3 AP manifest semantic drift")
    terminal_states = [
        row
        for row in states
        if row.get("record_type") == "job_terminal"
        and str(row.get("job_id") or "") == candidate_id
        and row.get("model") == dimensions.get("model")
        and row.get("stage") == "full"
        and row.get("status") == "success"
    ]
    if len(terminal_states) != 1:
        raise ValueError("Stage3 AP executor terminal drift")
    terminal = terminal_states[0]
    _validate_frozen_stage3_ap_executor()
    expected_plan_fingerprint = stage3_ap_executor.plan_fingerprint(
        plan, "full"
    )
    if (
        terminal.get("plan_fingerprint") != expected_plan_fingerprint
        or
        terminal.get("report_path") != copied["raw_artifact_path"]
        or terminal.get("report_sha256")
        != copied["raw_artifact_sha256"]
        or stage5_terminal.get("ap_report_path")
        != copied["raw_artifact_path"]
        or stage5_terminal.get("ap_report_sha256")
        != copied["raw_artifact_sha256"]
    ):
        if terminal.get("plan_fingerprint") != expected_plan_fingerprint:
            raise ValueError("Stage3 AP plan fingerprint drift")
        raise ValueError("Stage3 AP raw lineage drift")
    valid, _, ap = stage3_ap_executor.validate_report(
        str(dimensions.get("model") or ""),
        "full",
        raw,
        runner_key=str(expected_runner or ""),
    )
    if (
        not valid
        or any(
            ap.get(field) != stage5_terminal.get(field)
            for field in ("ap30", "ap50", "ap70")
        )
        or terminal.get("ap")
        != {
            field: stage5_terminal[field]
            for field in ("ap30", "ap50", "ap70")
        }
    ):
        raise ValueError("Stage3 AP raw contract drift")
    return copied


def _validate_physical_plan(
    logical_request: Mapping[str, Any],
    selection_binding: Mapping[str, Any],
    physical_plan: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    verified = cache_v2.validate_selection_binding(
        logical_request, selection_binding
    )
    copied = copy.deepcopy(dict(physical_plan))
    recorded = copied.pop("physical_request_sha256", None)
    logical_bindings = copied.pop("logical_row_bindings", None)
    if (
        copied.get("schema_version") != PHYSICAL_REQUEST_SCHEMA
        or not _is_sha256(recorded)
        or recorded != _sha(copied)
        or not isinstance(logical_bindings, list)
        or len(logical_bindings) != len(verified["selected_candidates"])
    ):
        raise ValueError("physical request authentication failed")
    physical_rows = copied.get("rows")
    physical_row_sha = copied.get("row_sha256")
    if not isinstance(physical_rows, list) or not isinstance(
        physical_row_sha, Mapping
    ):
        raise ValueError("physical request authentication failed")
    physical_ids = [_row_id(row) for row in physical_rows]
    physical_id_set = set(physical_ids)
    if (
        copied.get("physical_row_count") != len(physical_rows)
        or copied.get("logical_row_count")
        != len(verified["selected_candidates"])
        or set(physical_row_sha) != set(physical_ids)
        or len(set(physical_ids)) != len(physical_ids)
        or any(
            not _is_sha256(copied.get(field))
            for field in (
                "cache_snapshot_sha256",
                "lineage_head_sha256",
                "cache_reveal_sha256",
            )
        )
    ):
        raise ValueError("physical request authentication failed")
    expected_physical_ids = [
        selected["candidate_id"]
        for selected in verified["selected_candidates"]
        if selected["candidate_id"] in physical_id_set
    ]
    if physical_ids != expected_physical_ids:
        raise ValueError(
            "physical rows are not in strict logical order"
        )
    logical_rows = {
        _row_id(row): copy.deepcopy(dict(row))
        for row in verified["logical_request"]["rows"]
    }
    for row in physical_rows:
        candidate_id = _row_id(row)
        if (
            physical_row_sha.get(candidate_id) != _sha(row)
            or row != logical_rows.get(candidate_id)
        ):
            raise ValueError(
                "physical row differs from logical request row"
            )
    expected_bindings = []
    for logical_row_index, selected in enumerate(
        verified["selected_candidates"]
    ):
        candidate_id = selected["candidate_id"]
        is_miss = candidate_id in physical_id_set
        expected_bindings.append(
            {
                "logical_row_index": logical_row_index,
                "candidate_id": candidate_id,
                "logical_row_sha256": selected["logical_row_sha256"],
                "logical_exact_binding_sha256": selected[
                    "logical_exact_binding_sha256"
                ],
                "exact_cache_key_sha256": selected[
                    "exact_cache_key_sha256"
                ],
                "disposition": "miss" if is_miss else "hit",
                "physical_row_sha256": (
                    physical_row_sha.get(candidate_id) if is_miss else None
                ),
            }
        )
    if logical_bindings != expected_bindings:
        raise ValueError("physical request authentication failed")
    return verified, {
        **copied,
        "logical_row_bindings": copy.deepcopy(logical_bindings),
        "physical_request_sha256": recorded,
    }


def wrap_actual_v3_terminal(
    logical_request: Mapping[str, Any],
    selection_binding: Mapping[str, Any],
    physical_plan: Mapping[str, Any],
    *,
    candidate_id: str,
    stage5_terminal: Mapping[str, Any],
    stage5_terminal_artifact: Mapping[str, Any],
    stage3_performance_artifact: Mapping[str, Any],
    stage3_ap_artifact: Mapping[str, Any],
    actual_graph_features: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind one successful miss to reviewed Stage5/Stage3 terminal evidence."""
    verified, plan = _validate_physical_plan(
        logical_request, selection_binding, physical_plan
    )
    selected_by_id = {
        entry["candidate_id"]: entry
        for entry in verified["selected_candidates"]
    }
    plan_by_id = {
        entry["candidate_id"]: entry
        for entry in plan["logical_row_bindings"]
    }
    if candidate_id not in selected_by_id or plan_by_id.get(
        candidate_id, {}
    ).get("disposition") != "miss":
        raise ValueError("terminal candidate is not a physical miss")

    terminal = copy.deepcopy(dict(stage5_terminal))
    recorded_terminal_sha = terminal.pop("actual_feedback_row_sha256", None)
    if (
        _row_id(terminal) != candidate_id
        or terminal.get("terminal_status")
        not in {"measured_success", "measured_success_gold"}
        or not _is_sha256(recorded_terminal_sha)
        or recorded_terminal_sha != _sha(terminal)
    ):
        raise ValueError("Stage5 terminal authentication failed")
    graph = copy.deepcopy(dict(actual_graph_features))
    graph_sha = _sha(graph)
    if terminal.get("materialized_graph_features_sha256") != graph_sha:
        raise ValueError("actual graph feature SHA mismatch")
    metrics = {}
    for field in cache_v2.METRIC_FIELDS:
        value = terminal.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"Stage5 terminal metric is invalid: {field}")
        metrics[field] = value

    selected = selected_by_id[candidate_id]
    stage5_artifact, stage5_payload = _validate_artifact(
        stage5_terminal_artifact,
        name="Stage5 terminal",
        artifact_kind="stage5_terminal",
    )
    authenticated_terminal = {
        **terminal,
        "actual_feedback_row_sha256": recorded_terminal_sha,
    }
    if stage5_payload != authenticated_terminal:
        raise ValueError("Stage5 terminal artifact payload mismatch")
    if terminal.get("measurement_request_row_sha256") != selected[
        "logical_row_sha256"
    ]:
        raise ValueError("Stage5 terminal request row binding mismatch")
    performance_lineage = _validate_stage3_performance_lineage(
        stage3_performance_artifact,
        candidate_id=candidate_id,
        dimensions=selected["exact_key_dimensions"],
        stage5_terminal=authenticated_terminal,
        logical_request_sha256=verified["logical_request"][
            "logical_request_sha256"
        ],
    )
    ap_artifact = _validate_stage3_ap_lineage(
        stage3_ap_artifact,
        candidate_id=candidate_id,
        dimensions=selected["exact_key_dimensions"],
        stage5_terminal=authenticated_terminal,
        performance_lineage=performance_lineage,
    )
    payload = {
        "schema_version": TERMINAL_WRAPPER_SCHEMA,
        "created_by": cache_v2.ACTUAL_V3_PRODUCER,
        "terminal_status": terminal["terminal_status"],
        "candidate_id": candidate_id,
        "exact_key_dimensions": copy.deepcopy(
            selected["exact_key_dimensions"]
        ),
        "exact_cache_key_sha256": selected["exact_cache_key_sha256"],
        "stage5_terminal_sha256": str(recorded_terminal_sha),
        "stage5_terminal_artifact": stage5_artifact,
        "stage3_performance_artifact": performance_lineage["reference"],
        "stage3_ap_artifact": ap_artifact,
        "actual_graph_features_sha256": graph_sha,
        **metrics,
    }
    wrapped = {
        **payload,
        "terminal_evidence_sha256": _sha(payload),
    }
    return validate_actual_v3_terminal_wrapper(wrapped)


def validate_actual_v3_terminal_wrapper(
    terminal_wrapper: Mapping[str, Any],
) -> dict[str, Any]:
    """Revalidate formal wrapper, artifact files, and semantic row bindings."""
    validated = cache_v2.validate_actual_v3_terminal_evidence(
        terminal_wrapper
    )
    stage5_ref, stage5 = _validate_artifact(
        validated["stage5_terminal_artifact"],
        name="Stage5 terminal",
        artifact_kind="stage5_terminal",
    )
    stage5_sha = stage5.pop("actual_feedback_row_sha256", None)
    if (
        stage5_sha != validated["stage5_terminal_sha256"]
        or stage5_sha != _sha(stage5)
        or _row_id(stage5) != validated["candidate_id"]
        or stage5.get("terminal_status") != validated["terminal_status"]
        or stage5.get("materialized_graph_features_sha256")
        != validated["actual_graph_features_sha256"]
        or any(
            stage5.get(field) != validated[field]
            for field in cache_v2.METRIC_FIELDS
        )
    ):
        raise ValueError("Stage5 terminal semantic binding mismatch")
    authenticated_stage5 = {
        **stage5,
        "actual_feedback_row_sha256": stage5_sha,
    }
    performance_lineage = _validate_stage3_performance_lineage(
        validated["stage3_performance_artifact"],
        candidate_id=validated["candidate_id"],
        dimensions=validated["exact_key_dimensions"],
        stage5_terminal=authenticated_stage5,
        logical_request_sha256=None,
    )
    ap_ref = _validate_stage3_ap_lineage(
        validated["stage3_ap_artifact"],
        candidate_id=validated["candidate_id"],
        dimensions=validated["exact_key_dimensions"],
        stage5_terminal=authenticated_stage5,
        performance_lineage=performance_lineage,
    )
    if (
        stage5_ref != validated["stage5_terminal_artifact"]
        or performance_lineage["reference"]
        != validated["stage3_performance_artifact"]
        or ap_ref != validated["stage3_ap_artifact"]
    ):
        raise ValueError("formal actual-v3 artifact reference drift")
    return validated


def append_terminal_wrapper(
    cache: Mapping[str, Any], terminal_wrapper: Mapping[str, Any]
) -> dict[str, Any]:
    """Immutably append one formal selected-only actual-v3 terminal."""
    if (
        not isinstance(terminal_wrapper, Mapping)
        or terminal_wrapper.get("schema_version")
        != cache_v2.ACTUAL_V3_TERMINAL_EVIDENCE_SCHEMA
    ):
        raise ValueError("formal actual-v3 terminal schema is required")
    validated = validate_actual_v3_terminal_wrapper(terminal_wrapper)
    return cache_v2.append_actual_v3_terminal_evidence(cache, validated)


def finalize_actual_v3_failure(
    logical_request: Mapping[str, Any],
    selection_binding: Mapping[str, Any],
    physical_plan: Mapping[str, Any],
    *,
    candidate_id: str,
    failure_class: str,
    reason: str,
) -> dict[str, Any]:
    """Bind failure accounting to one selected row and the same logical SHA."""
    verified, plan = _validate_physical_plan(
        logical_request, selection_binding, physical_plan
    )
    selected = {
        row["candidate_id"]: row for row in verified["selected_candidates"]
    }
    plan_by_id = {
        row["candidate_id"]: row for row in plan["logical_row_bindings"]
    }
    if (
        candidate_id not in selected
        or plan_by_id.get(candidate_id, {}).get("disposition") != "miss"
    ):
        raise ValueError("failure candidate is outside frozen selection")
    if failure_class == "candidate":
        base = cache_v2.finalize_candidate_failure(
            logical_request, reason=reason
        )
    elif failure_class == "infrastructure":
        base = cache_v2.finalize_infrastructure_failure(
            logical_request, reason=reason
        )
    elif failure_class == "evidence":
        base = cache_v2.finalize_evidence_failure(
            logical_request, reason=reason
        )
    else:
        raise ValueError("unknown failure class")
    logical_sha = verified["logical_request"]["logical_request_sha256"]
    consumes = bool(base["consumes_selected_event_budget"])
    payload = {
        "schema_version": FAILURE_SCHEMA,
        "failure_class": failure_class,
        "terminal_status": base["terminal_status"],
        "failure_reason": reason,
        "candidate_id": candidate_id,
        "logical_request_sha256": logical_sha,
        "logical_row_sha256": selected[candidate_id][
            "logical_row_sha256"
        ],
        "selection_binding_sha256": verified[
            "selection_binding_sha256"
        ],
        "logical_exact_binding_sha256": selected[candidate_id][
            "logical_exact_binding_sha256"
        ],
        "exact_cache_key_sha256": selected[candidate_id][
            "exact_cache_key_sha256"
        ],
        "physical_request_sha256": plan["physical_request_sha256"],
        "physical_row_sha256": plan_by_id[candidate_id][
            "physical_row_sha256"
        ],
        "consumes_selected_event_budget": consumes,
        "selected_event_budget_delta": int(consumes),
        "retry_logical_request_sha256": (
            None if consumes else logical_sha
        ),
        **{field: None for field in cache_v2.METRIC_FIELDS},
    }
    wrapped = {
        **payload,
        "failure_wrapper_sha256": _sha(payload),
    }
    return validate_actual_v3_failure(
        wrapped, logical_request, selection_binding, physical_plan
    )


def validate_actual_v3_failure(
    failure: Mapping[str, Any],
    logical_request: Mapping[str, Any],
    selection_binding: Mapping[str, Any],
    physical_plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate formal failure classification, accounting, and lineage."""
    verified, plan = _validate_physical_plan(
        logical_request, selection_binding, physical_plan
    )
    if not isinstance(failure, Mapping):
        raise ValueError("formal actual-v3 failure must be a mapping")
    copied = copy.deepcopy(dict(failure))
    recorded = copied.pop("failure_wrapper_sha256", None)
    expected_fields = {
        "schema_version",
        "failure_class",
        "terminal_status",
        "failure_reason",
        "candidate_id",
        "logical_request_sha256",
        "logical_row_sha256",
        "selection_binding_sha256",
        "logical_exact_binding_sha256",
        "exact_cache_key_sha256",
        "physical_request_sha256",
        "physical_row_sha256",
        "consumes_selected_event_budget",
        "selected_event_budget_delta",
        "retry_logical_request_sha256",
        *cache_v2.METRIC_FIELDS,
    }
    if (
        set(copied) != expected_fields
        or copied.get("schema_version") != FAILURE_SCHEMA
        or not _is_sha256(recorded)
        or recorded != _sha(copied)
    ):
        raise ValueError("formal actual-v3 failure wrapper authentication failed")
    failure_class = str(copied.get("failure_class") or "")
    reason = str(copied.get("failure_reason") or "")
    consumes = failure_class == "candidate"
    if (
        failure_class not in _FAILURE_REASON_BY_CLASS
        or reason not in _FAILURE_REASON_BY_CLASS[failure_class]
        or copied.get("terminal_status")
        != _FAILURE_STATUS.get(failure_class)
        or copied.get("consumes_selected_event_budget") is not consumes
        or copied.get("selected_event_budget_delta") != int(consumes)
        or copied.get("retry_logical_request_sha256")
        != (
            None
            if consumes
            else copied.get("logical_request_sha256")
        )
        or any(copied.get(field) is not None for field in cache_v2.METRIC_FIELDS)
    ):
        raise ValueError("failure classification budget contract drift")
    sha_fields = (
        "logical_request_sha256",
        "logical_row_sha256",
        "selection_binding_sha256",
        "logical_exact_binding_sha256",
        "exact_cache_key_sha256",
        "physical_request_sha256",
        "physical_row_sha256",
    )
    if (
        not str(copied.get("candidate_id") or "")
        or any(not _is_sha256(copied.get(field)) for field in sha_fields)
    ):
        raise ValueError("formal actual-v3 failure lineage is invalid")
    candidate_id = str(copied["candidate_id"])
    selected = {
        row["candidate_id"]: row for row in verified["selected_candidates"]
    }
    physical = {
        row["candidate_id"]: row for row in plan["logical_row_bindings"]
    }
    selected_row = selected.get(candidate_id)
    physical_row = physical.get(candidate_id)
    if (
        selected_row is None
        or physical_row is None
        or physical_row.get("disposition") != "miss"
        or copied["logical_request_sha256"]
        != verified["logical_request"]["logical_request_sha256"]
        or copied["logical_row_sha256"]
        != selected_row["logical_row_sha256"]
        or copied["selection_binding_sha256"]
        != verified["selection_binding_sha256"]
        or copied["logical_exact_binding_sha256"]
        != selected_row["logical_exact_binding_sha256"]
        or copied["exact_cache_key_sha256"]
        != selected_row["exact_cache_key_sha256"]
        or copied["physical_request_sha256"]
        != plan["physical_request_sha256"]
        or copied["physical_row_sha256"]
        != physical_row["physical_row_sha256"]
    ):
        raise ValueError("formal actual-v3 failure context binding drift")
    return {**copied, "failure_wrapper_sha256": recorded}


__all__ = [
    "FAILURE_SCHEMA",
    "PHYSICAL_REQUEST_SCHEMA",
    "TERMINAL_WRAPPER_SCHEMA",
    "append_terminal_wrapper",
    "bind_selector_output",
    "build_miss_only_physical_plan",
    "finalize_actual_v3_failure",
    "validate_actual_v3_failure",
    "validate_actual_v3_terminal_wrapper",
    "wrap_actual_v3_terminal",
]
