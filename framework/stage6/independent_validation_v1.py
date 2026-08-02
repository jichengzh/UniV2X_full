"""Arm-scoped independent-validation contracts for Stage6."""

from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence

from framework.stage6.evidence_bundle_v1 import payload_sha256


PIPELINE_IDS = {
    ("tvm", "compression_only"): "tvm:compression_only:max_trials=0",
    ("tvm", "compress_then_tune"): "tvm:compress_then_tune:max_trials=64",
    ("trt", "compression_only"): "trt:compression_only:standard_tactic_build",
    ("trt", "compress_then_tune"): "trt:compress_then_tune:standard_tactic_build",
    ("tvm", "joint_shcosearch"): "tvm:joint_shcosearch:max_trials=64",
    ("trt", "joint_shcosearch"): "trt:joint_shcosearch:standard_tactic_build",
}


def scope_passed_validation_tasks(
    audit: Mapping[str, Any], *, task_ids: Sequence[str]
) -> dict[str, Any]:
    """Extract a model-scoped, fully passed view from a broader Stage5 audit."""
    if audit.get("schema_version") != "stage5_independent_validation_audit_v1":
        raise ValueError("joint validation source must use the Stage5 audit schema")
    requested = [str(task_id) for task_id in task_ids]
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("requested validation task IDs are empty or duplicated")
    by_id = {
        str(task.get("task_id") or ""): task
        for task in audit.get("tasks") or []
        if isinstance(task, Mapping)
    }
    missing = [task_id for task_id in requested if task_id not in by_id]
    if missing:
        raise ValueError(f"requested validation tasks are missing: {missing}")
    selected = [copy.deepcopy(dict(by_id[task_id])) for task_id in requested]
    for task in selected:
        configurations = task.get("configurations") or []
        if task.get("passed") is not True:
            raise ValueError(f"requested validation task is not passed: {task['task_id']}")
        if not configurations or any(
            configuration.get("consistency", {}).get("passed") is not True
            for configuration in configurations
        ):
            raise ValueError(
                f"requested validation task has unpassed configurations: {task['task_id']}"
            )
    return {
        "schema_version": "stage5_independent_validation_audit_v1",
        "all_tasks_passed": True,
        "task_count": len(selected),
        "configuration_count": sum(
            len(task["configurations"]) for task in selected
        ),
        "tasks": selected,
    }


def build_scoped_validation_request(
    rows: Sequence[Mapping[str, Any]], *, arm_id: str, backend: str
) -> dict[str, Any]:
    """Build a signed independent request whose identity includes the arm pipeline."""
    key = (backend, arm_id)
    if key not in PIPELINE_IDS:
        raise ValueError(f"unsupported Stage6 validation scope: {backend}:{arm_id}")
    copied = [copy.deepcopy(dict(row)) for row in rows]
    if not 1 <= len(copied) <= 4:
        raise ValueError("Stage6 independent validation requires one to four rows")
    row_ids = [str(row.get("manifest_job_id") or row.get("row_id") or "") for row in copied]
    if any(not row_id for row_id in row_ids) or len(set(row_ids)) != len(copied):
        raise ValueError("Stage6 independent rows have empty or duplicate identities")
    task_ids = {str(row.get("task_id") or "") for row in copied}
    task_shas = {str(row.get("task_sha256") or "") for row in copied}
    dispatches = {str(row.get("dispatch_key") or "") for row in copied}
    expected_dispatch = "tvm_auto" if backend == "tvm" else "trt_engine"
    if len(task_ids) != 1 or len(task_shas) != 1 or dispatches != {expected_dispatch}:
        raise ValueError("Stage6 independent rows drift from one backend task")
    request = {
        "schema_version": "stage5_independent_validation_request_v1",
        "task_id": next(iter(task_ids)),
        "task_sha256": next(iter(task_shas)),
        "batch_size": len(copied),
        "real_h800_measurement_required": True,
        "required_metrics": ["latency_ms", "energy_j", "ap30", "ap50", "ap70"],
        "independent_from_search_measurement": True,
        "stage6_arm_id": arm_id,
        "stage6_pipeline_id": PIPELINE_IDS[key],
        "rows": copied,
        "row_sha256": {
            row_id: payload_sha256(row) for row_id, row in zip(row_ids, copied)
        },
    }
    return {
        **request,
        "measurement_request_sha256": payload_sha256(request),
    }
