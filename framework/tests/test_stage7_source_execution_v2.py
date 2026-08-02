from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from framework.stage5.measurement_plan_v1 import _source_plan_sha
from framework.stage7.source_resolution_v2 import build_source_resolution_plan
from framework.stage7.source_round_orchestration_v2 import freeze_selection_identity


def _sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _request(root: Path) -> dict[str, Any]:
    specs = (
        ("16x32x64", [16, 32, 64], "fp16"),
        ("16x32x64", [16, 32, 64], "int8"),
        ("24x48x96", [24, 48, 96], "fp16"),
        ("24x48x96", [24, 48, 96], "int8"),
    )
    rows = []
    for index, (tag, width, q_mode) in enumerate(specs):
        candidate_id = f"pyramid|{tag}|q={q_mode}|profile=h800-v3"
        source_root = root / tag
        row = {
            "schema_version": "stage5_candidate_row_v2",
            "task_id": "S7-PYR-TVM",
            "task_sha256": "ab" * 32,
            "row_id": candidate_id,
            "manifest_job_id": candidate_id,
            "strategy_id": f"candidate-{index}",
            "model": "pyramid",
            "group_id": f"pyramid|{tag}",
            "width": width,
            "genome": [*width, q_mode],
            "q_mode": q_mode,
            "hardware_id": "h800",
            "capability_profile_id": "h800-v3",
            "capability_digest": "cd" * 32,
            "dispatch_key": "tvm_auto",
            "source_status": "planned",
            "materialization_kind": "pyramid_checkpoint_export",
            "source_contract": {
                "checkpoint_path": str(source_root / "checkpoint.pth"),
                "checkpoint_sha256": None,
                "onnx_path": str(source_root / "model.onnx"),
                "onnx_sha256": None,
                "calibration_npz": str(source_root / "calibration.npz"),
                "calibration_summary": str(source_root / "summary.json"),
                "source_done_marker": str(source_root / "source.done"),
                "trt_calibration_dir": str(source_root / "trt"),
            },
            "graph_features": {"conv_count": 51},
        }
        row["source_evidence_sha256"] = _source_plan_sha(row)
        rows.append(row)
    payload = {
        "schema_version": "stage5_measurement_request_v2",
        "task_id": "S7-PYR-TVM",
        "task_sha256": "ab" * 32,
        "round_index": 0,
        "batch_size": 4,
        "sample_budget": 16,
        "required_metrics": ["latency_ms", "energy_j", "ap30", "ap50", "ap70"],
        "atomic_feedback": True,
        "real_h800_measurement_required": True,
        "row_sha256": {row["row_id"]: _sha(row) for row in rows},
        "rows": rows,
    }
    return {**payload, "measurement_request_sha256": _sha(payload)}


def _inputs(root: Path) -> tuple[bytes, dict[str, Any], dict[str, Any]]:
    request = _request(root)
    selected_ids = [row["row_id"] for row in request["rows"]]
    selection = {
        "measurement_request": request,
        "acquisition": {"selected_row_ids": selected_ids},
    }
    binding = freeze_selection_identity(selection)
    plan = build_source_resolution_plan(request)
    request_bytes = (
        json.dumps(request, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    return request_bytes, binding, plan


def _lease_args(execution_plan: dict[str, Any]) -> dict[str, Any]:
    groups = execution_plan["group_jobs"]
    inventory = {
        f"GPU-{index}": {"physical_index": index, "model": "NVIDIA H800"}
        for index in range(len(groups))
    }
    assignments = {
        group["source_group_execution_key"]: f"GPU-{index}"
        for index, group in enumerate(groups)
    }
    return {
        "hostname": "zs-nj-tap-gpu18",
        "inventory": inventory,
        "assignments": assignments,
        "attempt_index": 0,
        "lock_owner": {
            "pid": 1234,
            "owner": "jichengzhi",
            "start_time": "100.00",
        },
    }


def test_execution_plan_preserves_authenticated_request_and_deduplicates_groups(
    tmp_path: Path,
) -> None:
    from framework.stage7 import source_execution_v2 as module

    request_bytes, binding, source_plan = _inputs(tmp_path)

    result = module.build_source_execution_plan(request_bytes, binding, source_plan)

    assert result["schema_version"] == "stage7_source_execution_plan_v2"
    assert (
        result["logical_request_file_sha256"]
        == hashlib.sha256(request_bytes).hexdigest()
    )
    assert (
        result["logical_request_sha256"]
        == json.loads(request_bytes)["measurement_request_sha256"]
    )
    assert result["candidate_evidence_binding_count"] == 4
    assert len(result["candidate_evidence_bindings"]) == 4
    assert result["group_job_count"] == 2
    assert len(result["group_jobs"]) == 2
    assert all(len(group["candidate_ids"]) == 2 for group in result["group_jobs"])
    encoded = json.dumps(result, sort_keys=True).lower()
    assert "latency_ms" not in encoded
    assert "energy_j" not in encoded
    assert "terminal_status" not in encoded
    assert (
        module.validate_source_execution_plan(
            result, request_bytes, binding, source_plan
        )
        == result
    )


@pytest.mark.parametrize("mutation", ("bytes", "binding", "plan"))
def test_execution_plan_rejects_request_binding_or_plan_drift(
    mutation: str,
    tmp_path: Path,
) -> None:
    from framework.stage7 import source_execution_v2 as module

    request_bytes, binding, source_plan = _inputs(tmp_path)
    execution = module.build_source_execution_plan(request_bytes, binding, source_plan)
    if mutation == "bytes":
        request_bytes = request_bytes + b" "
    elif mutation == "binding":
        binding = copy.deepcopy(binding)
        binding["logical_request_sha256"] = "12" * 32
    else:
        source_plan = copy.deepcopy(source_plan)
        source_plan["rows"][0]["source_contract_sha256"] = "34" * 32

    with pytest.raises(ValueError):
        module.validate_source_execution_plan(
            execution, request_bytes, binding, source_plan
        )


def test_execution_plan_rejects_same_group_with_divergent_source_identity(
    tmp_path: Path,
) -> None:
    from framework.stage7 import source_execution_v2 as module

    request = _request(tmp_path)
    request["rows"][1]["source_contract"]["onnx_path"] = str(
        tmp_path / "different.onnx"
    )
    request["rows"][1]["source_evidence_sha256"] = _source_plan_sha(request["rows"][1])
    request["row_sha256"] = {row["row_id"]: _sha(row) for row in request["rows"]}
    unsigned = {
        key: value
        for key, value in request.items()
        if key != "measurement_request_sha256"
    }
    request["measurement_request_sha256"] = _sha(unsigned)
    selected_ids = [row["row_id"] for row in request["rows"]]
    binding = freeze_selection_identity(
        {
            "measurement_request": request,
            "acquisition": {"selected_row_ids": selected_ids},
        }
    )
    plan = build_source_resolution_plan(request)
    request_bytes = (json.dumps(request, sort_keys=True) + "\n").encode()

    with pytest.raises(ValueError, match="group"):
        module.build_source_execution_plan(request_bytes, binding, plan)


def test_formal_h800_lease_binds_uuid_index_locks_owner_and_attempt(
    tmp_path: Path,
) -> None:
    from framework.stage7 import source_execution_v2 as module

    request_bytes, binding, source_plan = _inputs(tmp_path)
    execution = module.build_source_execution_plan(request_bytes, binding, source_plan)
    args = _lease_args(execution)

    lease = module.build_source_gpu_lease(
        execution,
        **args,
        execution_mode="formal",
        source_locks_held=True,
    )

    assert lease["schema_version"] == "stage7_source_gpu_lease_v2"
    assert lease["execution_mode"] == "formal"
    assert lease["source_locks_held"] is True
    assert lease["group_binding_count"] == 2
    assert [row["physical_index"] for row in lease["group_bindings"]] == [0, 1]
    assert lease["source_lock_keys"] == sorted(lease["source_lock_keys"])
    assert len(lease["source_attempt_sha256"]) == 64
    assert module.validate_source_gpu_lease(lease, execution) == lease


def test_formal_lease_authenticates_complete_inventory_not_only_assignments(
    tmp_path: Path,
) -> None:
    from framework.stage7 import source_execution_v2 as module

    request_bytes, binding, source_plan = _inputs(tmp_path)
    execution = module.build_source_execution_plan(request_bytes, binding, source_plan)
    args = _lease_args(execution)
    args["inventory"]["GPU-unassigned"] = {
        "physical_index": 7,
        "model": "NVIDIA H800",
    }

    lease = module.build_source_gpu_lease(
        execution,
        **args,
        execution_mode="formal",
        source_locks_held=True,
    )

    assert lease["inventory"]["GPU-unassigned"]["physical_index"] == 7
    assert module.validate_source_gpu_lease(lease, execution) == lease


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("hostname", "hostname"),
        ("model", "H800"),
        ("duplicate_uuid", "assignment"),
        ("duplicate_index", "index"),
        ("locks", "lock"),
        ("attempt", "attempt"),
    ),
)
def test_formal_lease_rejects_runtime_or_identity_drift(
    mutation: str,
    message: str,
    tmp_path: Path,
) -> None:
    from framework.stage7 import source_execution_v2 as module

    request_bytes, binding, source_plan = _inputs(tmp_path)
    execution = module.build_source_execution_plan(request_bytes, binding, source_plan)
    args = _lease_args(execution)
    if mutation == "hostname":
        args["hostname"] = "other-host"
    elif mutation == "model":
        args["inventory"]["GPU-0"]["model"] = "NVIDIA RTX 4090"
    elif mutation == "duplicate_uuid":
        keys = list(args["assignments"])
        args["assignments"][keys[1]] = args["assignments"][keys[0]]
    elif mutation == "duplicate_index":
        args["inventory"]["GPU-1"]["physical_index"] = 0
    elif mutation == "attempt":
        args["attempt_index"] = -1

    with pytest.raises(ValueError, match=message):
        module.build_source_gpu_lease(
            execution,
            **args,
            execution_mode="formal",
            source_locks_held=mutation != "locks",
        )


def test_no_gpu_dryrun_lease_is_signed_but_does_not_claim_locks(
    tmp_path: Path,
) -> None:
    from framework.stage7 import source_execution_v2 as module

    request_bytes, binding, source_plan = _inputs(tmp_path)
    execution = module.build_source_execution_plan(request_bytes, binding, source_plan)
    lease = module.build_source_gpu_lease(
        execution,
        **_lease_args(execution),
        execution_mode="no_gpu_dryrun",
        source_locks_held=False,
    )

    assert lease["execution_mode"] == "no_gpu_dryrun"
    assert lease["source_locks_held"] is False
    assert module.validate_source_gpu_lease(lease, execution) == lease


def test_retry_maps_failed_groups_to_four_row_identity_and_zero_budget(
    tmp_path: Path,
) -> None:
    from framework.stage7 import source_execution_v2 as module

    request_bytes, binding, source_plan = _inputs(tmp_path)
    execution = module.build_source_execution_plan(request_bytes, binding, source_plan)
    failed_group = execution["group_jobs"][0]["source_group_execution_key"]

    retry = module.build_zero_budget_source_retry(
        source_plan,
        execution,
        failed_group_keys=[failed_group],
        reason_code="evidence_unavailable",
    )

    assert retry["selected_event_budget_delta"] == 0
    assert retry["partial_reveal_allowed"] is False
    assert retry["retry_candidate_ids"] == execution["group_jobs"][0]["candidate_ids"]


def test_public_api_is_explicit() -> None:
    from framework.stage7 import source_execution_v2 as module

    assert set(module.__all__) == {
        "build_source_execution_plan",
        "validate_source_execution_plan",
        "build_source_gpu_lease",
        "validate_source_gpu_lease",
        "build_zero_budget_source_retry",
        "candidate_evidence_paths",
    }
