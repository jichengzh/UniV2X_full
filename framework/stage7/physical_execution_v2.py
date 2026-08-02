"""Authenticated Stage7 miss projection into existing Stage5 performance planning.

This module deliberately owns no GPU, TVM, quantization, AP, or subprocess
semantics.  It authenticates the already-frozen Stage7 round artifacts, emits
the independent 0--4-row Stage5 request, and calls the existing Stage5
framework planner for non-empty miss sets.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from framework.stage5 import measurement_plan_v1, measurement_plan_v2
from framework.stage7 import actual_v3_adapter_v2 as actual_adapter
from framework.stage7 import core_cache_v2 as cache_v2
from framework.stage7 import deployment_bundle_v2
from framework.stage7 import source_resolution_v2 as source_resolution
from framework.stage7.core_ablation_v2 import canonical_sha256
from scripts import stage35_gold32_performance_plan_v1
from scripts.stage7_scheduler_requests_v2 import resolved_source_lock_sha256


INDEPENDENT_REQUEST_SCHEMA = "stage5_independent_validation_request_v1"
EMPTY_TERMINAL_SCHEMA = "stage7_actual_v3_empty_physical_terminal_v2"
PERFORMANCE_ARTIFACT_SCHEMA = "stage7_authenticated_performance_artifacts_v2"
PHYSICAL_REQUEST_SCHEMA = "stage7_actual_v3_miss_only_physical_request_v2"
CACHE_REVEAL_SCHEMA = "stage7_actual_v3_selected_cache_reveal_v2"
ADMISSION_SCHEMA = "stage7_actual_v3_executor_admission_v2"


def _sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _row_id(row: Mapping[str, Any]) -> str:
    value = str(row.get("manifest_job_id") or row.get("row_id") or "")
    if not value:
        raise ValueError("physical projection row identity is missing")
    return value


def _validated_deployment_bundle(
    value: Mapping[str, Any],
    *,
    deployment_root: Path,
    frozen_repo_root: Path,
    expected_release_sha256: str | None,
    expected_manifest_file_sha256: str | None,
) -> str:
    """Consume the authenticated record returned by validate_deployment_bundle."""
    if not _is_sha256(expected_release_sha256):
        raise ValueError("expected release SHA256 external deployment pin is invalid")
    if not _is_sha256(expected_manifest_file_sha256):
        raise ValueError(
            "expected manifest-file SHA256 external deployment pin is invalid"
        )
    if not isinstance(value, Mapping):
        raise ValueError("authenticated deployment bundle is missing")
    required = {
        "deployment_release_sha256",
        "deployment_manifest_sha256",
        "deployment_manifest_file_sha256",
        "deployment_bundle_sha256",
        "bundle_code_root",
        "frozen_repo_root",
        "owner",
        "owner_uid",
        "files",
    }
    if set(value) != required:
        raise ValueError("authenticated deployment bundle shape drift")
    for field in (
        "deployment_release_sha256",
        "deployment_manifest_sha256",
        "deployment_manifest_file_sha256",
        "deployment_bundle_sha256",
    ):
        if not _is_sha256(value.get(field)):
            raise ValueError("authenticated deployment bundle SHA drift")
    if (
        not isinstance(value.get("bundle_code_root"), str)
        or not Path(value["bundle_code_root"]).is_absolute()
        or not isinstance(value.get("frozen_repo_root"), str)
        or not Path(value["frozen_repo_root"]).is_absolute()
        or not isinstance(value.get("owner"), str)
        or not value["owner"]
        or isinstance(value.get("owner_uid"), bool)
        or not isinstance(value.get("owner_uid"), int)
        or value["owner_uid"] < 0
        or not isinstance(value.get("files"), list)
        or not value["files"]
        or not all(
            isinstance(path, str) and Path(path).is_absolute()
            for path in value["files"]
        )
    ):
        raise ValueError("authenticated deployment bundle identity drift")
    try:
        recomputed = deployment_bundle_v2.validate_deployment_bundle(
            Path(deployment_root),
            frozen_repo_root=Path(frozen_repo_root),
            expected_release_sha256=expected_release_sha256,
            expected_manifest_file_sha256=expected_manifest_file_sha256,
        )
    except (OSError, ValueError) as error:
        raise ValueError("canonical deployment bundle validation failed") from error
    if copy.deepcopy(dict(value)) != recomputed:
        raise ValueError("supplied deployment record differs from canonical bundle")
    return str(recomputed["deployment_bundle_sha256"])


def _validated_reveal(
    *,
    logical_request: Mapping[str, Any],
    verified_selection: Mapping[str, Any],
    reveal: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not isinstance(reveal, Mapping):
        raise ValueError("cache reveal is missing")
    copied = copy.deepcopy(dict(reveal))
    recorded = copied.pop("cache_reveal_sha256", None)
    selected = verified_selection["selected_candidates"]
    entries = copied.get("entries")
    if (
        copied.get("schema_version") != CACHE_REVEAL_SCHEMA
        or copied.get("formal_actual_v3_cache_reveal") is not True
        or recorded != _sha(copied)
        or copied.get("logical_request_sha256")
        != logical_request["measurement_request_sha256"]
        or copied.get("selection_binding_sha256")
        != verified_selection["selection_binding_sha256"]
        or not _is_sha256(copied.get("cache_snapshot_sha256"))
        or not _is_sha256(copied.get("lineage_head_sha256"))
        or copied.get("selected_event_budget_delta") != len(selected)
        or not isinstance(entries, list)
        or len(entries) != len(selected)
    ):
        raise ValueError("cache reveal authentication failed")
    result = []
    for selected_row, raw_entry in zip(selected, entries):
        if not isinstance(raw_entry, Mapping):
            raise ValueError("cache reveal row binding drift")
        entry = copy.deepcopy(dict(raw_entry))
        disposition = entry.get("disposition")
        terminal_sha = entry.get("terminal_evidence_sha256")
        expected_keys = {
            "candidate_id",
            "logical_row_sha256",
            "logical_exact_binding_sha256",
            "exact_cache_key_sha256",
            "disposition",
            "selected_event_budget_delta",
            "hardware_measurement_required",
            "terminal_evidence_sha256",
            "reveal_use_binding_sha256",
        }
        if disposition == "hit":
            expected_keys.add("terminal_evidence")
        if (
            set(entry) != expected_keys
            or disposition not in {"hit", "miss"}
            or entry.get("candidate_id") != selected_row["candidate_id"]
            or entry.get("logical_row_sha256") != selected_row["logical_row_sha256"]
            or entry.get("logical_exact_binding_sha256")
            != selected_row["logical_exact_binding_sha256"]
            or entry.get("exact_cache_key_sha256")
            != selected_row["exact_cache_key_sha256"]
            or entry.get("selected_event_budget_delta") != 1
            or entry.get("hardware_measurement_required") is not (disposition == "miss")
        ):
            raise ValueError("cache reveal row binding drift")
        if disposition == "hit":
            if not _is_sha256(terminal_sha) or not isinstance(
                entry.get("terminal_evidence"), Mapping
            ):
                raise ValueError("cache hit terminal identity is missing")
            try:
                terminal = actual_adapter.validate_actual_v3_terminal_wrapper(
                    entry["terminal_evidence"]
                )
            except (OSError, ValueError) as error:
                raise ValueError(
                    "cache hit actual-v3 terminal validation failed"
                ) from error
            if (
                terminal.get("terminal_evidence_sha256") != terminal_sha
                or terminal.get("candidate_id") != selected_row["candidate_id"]
                or terminal.get("exact_cache_key_sha256")
                != selected_row["exact_cache_key_sha256"]
                or terminal.get("exact_key_dimensions")
                != selected_row["exact_key_dimensions"]
            ):
                raise ValueError("cache hit terminal selection binding drift")
        elif terminal_sha is not None or "terminal_evidence" in entry:
            raise ValueError("cache miss revealed terminal truth")
        use = {
            "candidate_id": selected_row["candidate_id"],
            "logical_request_sha256": logical_request["measurement_request_sha256"],
            "selection_binding_sha256": verified_selection["selection_binding_sha256"],
            "logical_row_sha256": selected_row["logical_row_sha256"],
            "logical_exact_binding_sha256": selected_row[
                "logical_exact_binding_sha256"
            ],
            "exact_cache_key_sha256": selected_row["exact_cache_key_sha256"],
            "terminal_evidence_sha256": terminal_sha,
            "disposition": disposition,
            "cache_snapshot_sha256": copied["cache_snapshot_sha256"],
            "lineage_head_sha256": copied["lineage_head_sha256"],
        }
        if entry.get("reveal_use_binding_sha256") != _sha(use):
            raise ValueError("cache reveal use-binding SHA drift")
        result.append(entry)
    return result


def _validated_physical_plan(
    *,
    logical_request: Mapping[str, Any],
    verified_selection: Mapping[str, Any],
    reveal: Mapping[str, Any],
    reveal_entries: Sequence[Mapping[str, Any]],
    physical_plan: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(physical_plan, Mapping):
        raise ValueError("physical request is missing")
    copied = copy.deepcopy(dict(physical_plan))
    recorded = copied.pop("physical_request_sha256", None)
    bindings = copied.pop("logical_row_bindings", None)
    rows = copied.get("rows")
    row_sha = copied.get("row_sha256")
    selected = verified_selection["selected_candidates"]
    if (
        copied.get("schema_version") != PHYSICAL_REQUEST_SCHEMA
        or recorded != _sha(copied)
        or copied.get("logical_request_sha256")
        != logical_request["measurement_request_sha256"]
        or copied.get("selection_binding_sha256")
        != verified_selection["selection_binding_sha256"]
        or copied.get("cache_snapshot_sha256") != reveal["cache_snapshot_sha256"]
        or copied.get("lineage_head_sha256") != reveal["lineage_head_sha256"]
        or copied.get("cache_reveal_sha256") != reveal["cache_reveal_sha256"]
        or copied.get("logical_row_count") != len(selected)
        or not isinstance(rows, list)
        or copied.get("physical_row_count") != len(rows)
        or len(rows) not in range(5)
        or not isinstance(row_sha, Mapping)
        or not isinstance(bindings, list)
        or len(bindings) != len(selected)
    ):
        raise ValueError("physical request authentication failed")
    logical_rows = {
        _row_id(row): copy.deepcopy(dict(row)) for row in logical_request["rows"]
    }
    expected_ids = [
        entry["candidate_id"]
        for entry in reveal_entries
        if entry["disposition"] == "miss"
    ]
    physical_ids = [_row_id(row) for row in rows]
    if (
        physical_ids != expected_ids
        or set(row_sha) != set(expected_ids)
        or any(
            row != logical_rows.get(candidate_id)
            or row_sha.get(candidate_id) != _sha(row)
            for candidate_id, row in zip(physical_ids, rows)
        )
    ):
        raise ValueError("physical rows differ from strict logical miss order")
    expected_bindings = [
        {
            "logical_row_index": index,
            "candidate_id": selected_row["candidate_id"],
            "logical_row_sha256": selected_row["logical_row_sha256"],
            "logical_exact_binding_sha256": selected_row[
                "logical_exact_binding_sha256"
            ],
            "exact_cache_key_sha256": selected_row["exact_cache_key_sha256"],
            "disposition": reveal_entry["disposition"],
            "physical_row_sha256": row_sha.get(selected_row["candidate_id"]),
        }
        for index, (selected_row, reveal_entry) in enumerate(
            zip(selected, reveal_entries)
        )
    ]
    if bindings != expected_bindings:
        raise ValueError("physical logical-row binding drift")
    return {
        **copied,
        "logical_row_bindings": copy.deepcopy(bindings),
        "physical_request_sha256": recorded,
    }


def _validated_admission(
    value: Mapping[str, Any],
    *,
    logical_request_sha256: str,
    physical_request_sha256: str,
    physical_row_count: int,
) -> str:
    if not isinstance(value, Mapping):
        raise ValueError("executor admission is missing")
    copied = copy.deepcopy(dict(value))
    recorded = copied.pop("admission_sha256", None)
    expected_keys = {
        "schema_version",
        "admission_passed",
        "contract_sha256",
        "logical_request_sha256",
        "physical_request_sha256",
        "logical_row_count",
        "physical_row_count",
        "execution_primitive",
        "gpu_jobs_launched",
    }
    if (
        set(copied) != expected_keys
        or copied.get("schema_version") != ADMISSION_SCHEMA
        or copied.get("admission_passed") is not True
        or not _is_sha256(copied.get("contract_sha256"))
        or copied.get("logical_request_sha256") != logical_request_sha256
        or copied.get("physical_request_sha256") != physical_request_sha256
        or copied.get("logical_row_count") != 4
        or copied.get("physical_row_count") != physical_row_count
        or copied.get("execution_primitive")
        != "existing_stage5_stage3_actual_feedback_v3"
        or copied.get("gpu_jobs_launched") != 0
        or recorded != canonical_sha256(copied)
    ):
        raise ValueError("executor admission authentication failed")
    return str(recorded)


def _source_lineage(
    physical: Mapping[str, Any],
    result: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> list[dict[str, Any]]:
    result_by_id = {row["candidate_id"]: row for row in result["rows"]}
    selected_by_id = {
        row["candidate_id"]: row for row in selection["selected_candidates"]
    }
    binding_by_id = {
        row["candidate_id"]: row
        for row in physical["logical_row_bindings"]
        if row["disposition"] == "miss"
    }
    rows = []
    for physical_row in physical["rows"]:
        candidate_id = _row_id(physical_row)
        source = result_by_id.get(candidate_id)
        exact = selected_by_id.get(candidate_id)
        binding = binding_by_id.get(candidate_id)
        if source is None or exact is None or binding is None:
            raise ValueError("physical miss source/exact lineage is incomplete")
        rows.append(
            {
                "candidate_id": candidate_id,
                "logical_row_index": binding["logical_row_index"],
                "logical_row_sha256": binding["logical_row_sha256"],
                "logical_exact_binding_sha256": exact["logical_exact_binding_sha256"],
                "exact_cache_key_sha256": exact["exact_cache_key_sha256"],
                "physical_row_sha256": binding["physical_row_sha256"],
                "resolved_source_sha256": source["resolved_source_sha256"],
                "checkpoint_sha256": source["checkpoint_sha256"],
                "onnx_sha256": source["onnx_sha256"],
                "source_evidence_file_sha256": source["source_evidence_file_sha256"],
            }
        )
    return rows


def build_independent_projection(
    *,
    logical_request: Mapping[str, Any],
    selection_binding: Mapping[str, Any],
    cache_reveal: Mapping[str, Any],
    physical_plan: Mapping[str, Any],
    source_plan: Mapping[str, Any],
    source_result: Mapping[str, Any],
    executor_admission: Mapping[str, Any],
    deployment_manifest: Mapping[str, Any],
    deployment_root: Path,
    frozen_repo_root: Path,
    expected_release_sha256: str | None = None,
    expected_manifest_file_sha256: str | None = None,
) -> dict[str, Any]:
    """Authenticate a Stage7 round and emit its ordered miss-only Stage5 request."""
    verified_selection = cache_v2.validate_selection_binding(
        logical_request, selection_binding
    )
    verified_plan = source_resolution.validate_source_resolution_plan(
        source_plan, logical_request=logical_request
    )
    verified_result = source_resolution.validate_formal_source_resolution_result(
        source_result, verified_plan
    )
    reveal_entries = _validated_reveal(
        logical_request=logical_request,
        verified_selection=verified_selection,
        reveal=cache_reveal,
    )
    physical = _validated_physical_plan(
        logical_request=logical_request,
        verified_selection=verified_selection,
        reveal=cache_reveal,
        reveal_entries=reveal_entries,
        physical_plan=physical_plan,
    )
    admission_sha = _validated_admission(
        executor_admission,
        logical_request_sha256=logical_request["measurement_request_sha256"],
        physical_request_sha256=physical["physical_request_sha256"],
        physical_row_count=physical["physical_row_count"],
    )
    deployment_sha = _validated_deployment_bundle(
        deployment_manifest,
        deployment_root=deployment_root,
        frozen_repo_root=frozen_repo_root,
        expected_release_sha256=expected_release_sha256,
        expected_manifest_file_sha256=expected_manifest_file_sha256,
    )
    lineage = {
        "logical_request_sha256": logical_request["measurement_request_sha256"],
        "selection_binding_sha256": verified_selection["selection_binding_sha256"],
        "cache_snapshot_sha256": cache_reveal["cache_snapshot_sha256"],
        "cache_reveal_sha256": cache_reveal["cache_reveal_sha256"],
        "physical_request_sha256": physical["physical_request_sha256"],
        "source_resolution_plan_sha256": verified_plan["source_resolution_plan_sha256"],
        "source_resolution_result_sha256": verified_result[
            "source_resolution_result_sha256"
        ],
        "resolved_source_lock_sha256": resolved_source_lock_sha256(
            physical, verified_result
        ),
        "executor_admission_sha256": admission_sha,
        "deployment_bundle_sha256": deployment_sha,
        "rows": _source_lineage(physical, verified_result, verified_selection),
    }
    if physical["physical_row_count"] == 0:
        payload = {
            "schema_version": EMPTY_TERMINAL_SCHEMA,
            "stage7_projection_lineage": lineage,
            "rows": [],
            "lineage_inputs": [],
            "gpu_subprocess_count": 0,
        }
        return {**payload, "empty_physical_terminal_sha256": _sha(payload)}
    rows = copy.deepcopy(physical["rows"])
    payload = {
        "schema_version": INDEPENDENT_REQUEST_SCHEMA,
        "task_id": logical_request["task_id"],
        "task_sha256": logical_request["task_sha256"],
        "round_index": logical_request.get("round_index"),
        "batch_size": len(rows),
        "required_metrics": copy.deepcopy(logical_request["required_metrics"]),
        "real_h800_measurement_required": True,
        "independent_from_search_measurement": True,
        "row_sha256": {
            _row_id(row): physical["row_sha256"][_row_id(row)] for row in rows
        },
        "rows": rows,
        "stage7_projection_lineage": lineage,
    }
    return {**payload, "measurement_request_sha256": _sha(payload)}


def _validate_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("independent projection is missing")
    copied = copy.deepcopy(dict(value))
    recorded = copied.pop("measurement_request_sha256", None)
    if (
        copied.get("schema_version") != INDEPENDENT_REQUEST_SCHEMA
        or recorded != _sha(copied)
        or not isinstance(copied.get("stage7_projection_lineage"), Mapping)
        or copied["stage7_projection_lineage"].get("rows") is None
    ):
        raise ValueError("independent projection authentication failed")
    measurement_plan_v2._validate_request(value)
    return {**copied, "measurement_request_sha256": recorded}


def _empty_performance_artifacts(
    terminal: Mapping[str, Any],
) -> dict[str, Any]:
    copied = copy.deepcopy(dict(terminal))
    recorded = copied.pop("empty_physical_terminal_sha256", None)
    if (
        copied.get("schema_version") != EMPTY_TERMINAL_SCHEMA
        or copied.get("rows") != []
        or copied.get("lineage_inputs") != []
        or copied.get("gpu_subprocess_count") != 0
        or recorded != _sha(copied)
    ):
        raise ValueError("empty physical terminal authentication failed")
    payload = {
        "schema_version": PERFORMANCE_ARTIFACT_SCHEMA,
        "empty_physical_terminal_sha256": recorded,
        "planned_candidate_ids": [],
        "manifest": None,
        "manifest_row_count": 0,
        "performance_jobs": [],
        "planner_call_count": 0,
        "source_binding_audit": {
            "source_evidence_paths": {},
            "source_evidence_file_sha256": {},
        },
    }
    return {**payload, "performance_artifacts_sha256": _sha(payload)}


def _rebuild_performance_jobs(
    manifest_rows: Sequence[Mapping[str, Any]],
    *,
    quant_contract_paths: Mapping[str, Path],
    remote_artifact_root: str | Path,
    gpus: Sequence[int],
) -> list[dict[str, Any]]:
    """Characterize the pinned Stage5 job builder without copying its semantics."""
    expected: list[dict[str, Any]] = []
    for index, manifest_row in enumerate(manifest_rows):
        row = copy.deepcopy(dict(manifest_row))
        row_id = _row_id(row)
        quant_path = quant_contract_paths.get(row_id)
        job = measurement_plan_v2._build_job(
            row,
            batch_index=1,
            row_index=index,
            remote_artifact_root=remote_artifact_root,
            gpus=gpus,
        )
        if quant_path is not None:
            command = list(job["command"])
            if "--tensor-quant-params-json" in command:
                raise ValueError(f"duplicate quant contract argument: {row_id}")
            job = {
                **job,
                "command": [
                    *command,
                    "--tensor-quant-params-json",
                    str(quant_path),
                ],
            }
        expected.append({**job, "schema_version": measurement_plan_v2.JOB_SCHEMA})
    return expected


def _validate_job_builder_import_chain() -> None:
    if (
        measurement_plan_v2._build_job is not measurement_plan_v1._build_job
        or measurement_plan_v1._build_job
        is not stage35_gold32_performance_plan_v1._build_job
    ):
        raise ValueError("Stage5 job builder import chain drift")


def _expected_prepared_rows(
    request_rows: Sequence[Mapping[str, Any]],
    *,
    source_paths: Mapping[str, Path],
    quant_contract_paths: Mapping[str, Path],
) -> list[dict[str, Any]]:
    required_quant_ids = {
        _row_id(row)
        for row in request_rows
        if row.get("dispatch_key") == "tvm_auto" and row.get("q_mode") == "int8"
    }
    if set(quant_contract_paths) != required_quant_ids:
        raise ValueError("quant contract paths drift from projected INT8 rows")
    evidence_by_group: dict[str, dict[str, Any]] = {}
    for row in request_rows:
        group_id = str(row["group_id"])
        if group_id not in evidence_by_group:
            evidence_by_group[group_id] = measurement_plan_v1._load_source_evidence(
                source_paths[group_id],
                group_id,
                str(row["source_evidence_sha256"]),
            )
    prepared_rows: list[dict[str, Any]] = []
    for request_row in request_rows:
        row = measurement_plan_v1._bind_verified_source(
            request_row,
            evidence_by_group[str(request_row["group_id"])],
            source_paths[str(request_row["group_id"])],
        )
        row = {
            **row,
            "source_plan_sha256": str(request_row["source_evidence_sha256"]),
        }
        row_id = _row_id(row)
        quant_path = quant_contract_paths.get(row_id)
        if quant_path is not None:
            if not quant_path.is_file():
                raise ValueError(f"quant contract missing: {row_id}")
            try:
                contract = json.loads(quant_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid quant contract: {row_id}") from error
            if (
                not isinstance(contract, Mapping)
                or contract.get("schema") != "stage3_tvm_int8_quant_contract_v3"
                or not isinstance(contract.get("params"), Mapping)
                or not contract["params"]
            ):
                raise ValueError(f"invalid quant contract: {row_id}")
            row = {
                **row,
                "source_contract": {
                    **row["source_contract"],
                    "tensor_quant_params_json": str(quant_path),
                    "tensor_quant_params_sha256": hashlib.sha256(
                        quant_path.read_bytes()
                    ).hexdigest(),
                },
            }
        prepared_rows.append(row)
    return prepared_rows


def build_authenticated_performance_artifacts(
    *,
    projection: Mapping[str, Any],
    source_plan: Mapping[str, Any],
    source_result: Mapping[str, Any],
    quant_contract_paths: Mapping[str, Path],
    remote_artifact_root: str | Path,
    gpus: Sequence[int],
) -> dict[str, Any]:
    """Call the frozen framework planner with authenticated source-result paths."""
    if projection.get("schema_version") == EMPTY_TERMINAL_SCHEMA:
        return _empty_performance_artifacts(projection)
    _validate_job_builder_import_chain()
    verified = _validate_projection(projection)
    plan = source_resolution.validate_source_resolution_plan(source_plan)
    try:
        result = source_resolution.validate_formal_source_resolution_result(
            source_result, plan
        )
    except OSError as error:
        raise ValueError("formal source-resolution evidence is unavailable") from error
    lineage = verified["stage7_projection_lineage"]
    if (
        lineage.get("source_resolution_plan_sha256")
        != plan["source_resolution_plan_sha256"]
        or lineage.get("source_resolution_result_sha256")
        != result["source_resolution_result_sha256"]
    ):
        raise ValueError("projection/source-resolution lineage drift")
    source_by_id = {row["candidate_id"]: row for row in result["rows"]}
    source_paths: dict[str, Path] = {}
    source_file_sha: dict[str, str] = {}
    projected_ids = [_row_id(row) for row in verified["rows"]]
    for row in verified["rows"]:
        candidate_id = _row_id(row)
        source = source_by_id.get(candidate_id)
        if source is None:
            raise ValueError("projection candidate has no formal source result")
        group_id = str(row["group_id"])
        path = Path(source["source_evidence_path"])
        digest = str(source["source_evidence_file_sha256"])
        if group_id in source_paths and (
            source_paths[group_id] != path or source_file_sha[group_id] != digest
        ):
            raise ValueError("shared source group has conflicting evidence")
        source_paths[group_id] = path
        source_file_sha[group_id] = digest
    normalized_quant_paths = {
        str(key): Path(value) for key, value in quant_contract_paths.items()
    }
    expected_manifest_rows = _expected_prepared_rows(
        verified["rows"],
        source_paths=source_paths,
        quant_contract_paths=normalized_quant_paths,
    )
    before = copy.deepcopy(verified["rows"])
    planned = measurement_plan_v2.build_performance_plan(
        verified,
        source_evidence_paths=source_paths,
        quant_contract_paths=normalized_quant_paths,
        remote_artifact_root=remote_artifact_root,
        gpus=gpus,
    )
    if verified["rows"] != before:
        raise ValueError("Stage5 planner mutated independent request rows")
    if not isinstance(planned, Mapping):
        raise ValueError("Stage5 planner returned an invalid result")
    manifest = planned.get("manifest")
    jobs = planned.get("performance_jobs")
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("schema_version") != measurement_plan_v2.MANIFEST_SCHEMA
        or manifest.get("source_request_schema") != INDEPENDENT_REQUEST_SCHEMA
        or manifest.get("source_request_sha256")
        != verified["measurement_request_sha256"]
        or manifest.get("row_count") != len(projected_ids)
        or not isinstance(manifest.get("jobs"), list)
        or len(manifest["jobs"]) != len(projected_ids)
        or [_row_id(row) for row in manifest["jobs"]] != projected_ids
        or not isinstance(jobs, list)
        or len(jobs) != len(projected_ids)
    ):
        raise ValueError("Stage5 performance manifest identity drift")
    execution_ids = [
        str(job.get("manifest_job_id") or "")
        for job in jobs
        if isinstance(job, Mapping)
    ]
    if execution_ids != projected_ids or len(execution_ids) != len(jobs):
        raise ValueError("Stage5 performance job identity/order drift")
    if manifest["jobs"] != expected_manifest_rows or canonical_sha256(
        manifest["jobs"]
    ) != canonical_sha256(expected_manifest_rows):
        raise ValueError("Stage5 performance manifest row canonical drift")
    expected_jobs = _rebuild_performance_jobs(
        expected_manifest_rows,
        quant_contract_paths=normalized_quant_paths,
        remote_artifact_root=remote_artifact_root,
        gpus=gpus,
    )
    if jobs != expected_jobs or canonical_sha256(jobs) != canonical_sha256(
        expected_jobs
    ):
        raise ValueError("Stage5 performance job canonical mapping drift")
    audit = {
        "source_resolution_plan_sha256": plan["source_resolution_plan_sha256"],
        "source_resolution_result_sha256": result["source_resolution_result_sha256"],
        "source_evidence_paths": {
            group: str(path) for group, path in sorted(source_paths.items())
        },
        "source_evidence_file_sha256": dict(sorted(source_file_sha.items())),
    }
    payload = {
        "schema_version": PERFORMANCE_ARTIFACT_SCHEMA,
        "independent_request_sha256": verified["measurement_request_sha256"],
        "planned_candidate_ids": projected_ids,
        "manifest": copy.deepcopy(dict(manifest)),
        "manifest_row_count": len(projected_ids),
        "performance_jobs": copy.deepcopy(jobs),
        "planner_call_count": 1,
        "source_binding_audit": audit,
    }
    return {**payload, "performance_artifacts_sha256": _sha(payload)}


__all__ = [
    "build_authenticated_performance_artifacts",
    "build_independent_projection",
]
