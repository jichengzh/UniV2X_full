from framework.stage6.evidence_bundle_v1 import payload_sha256
import pytest

from framework.stage6.independent_validation_v1 import (
    build_scoped_validation_request,
    scope_passed_validation_tasks,
)


def _row(configuration_id, width):
    return {
        "manifest_job_id": configuration_id,
        "row_id": configuration_id,
        "task_id": "S5-PYR-TVM",
        "task_sha256": "a" * 64,
        "model": "pyramid",
        "hardware_id": "h800",
        "capability_profile_id": "h800-tvm-probe-conditioned-v3",
        "dispatch_key": "tvm_auto",
        "capability_digest": "b" * 64,
        "group_id": f"pyramid|{'x'.join(map(str, width))}",
        "width": width,
        "q_mode": "fp16",
        "genome": [*width, "fp16"],
        "source_evidence_sha256": "c" * 64,
    }


def test_scoped_request_binds_arm_and_hashes_without_mutating_rows() -> None:
    rows = [
        _row("pyramid|16x48x64|q=fp16|profile=h800-tvm-probe-conditioned-v3", [16, 48, 64]),
        _row("pyramid|32x48x128|q=fp16|profile=h800-tvm-probe-conditioned-v3", [32, 48, 128]),
    ]

    request = build_scoped_validation_request(
        rows, arm_id="compression_only", backend="tvm"
    )

    assert "row_sha256" not in rows[0]
    assert request["schema_version"] == "stage5_independent_validation_request_v1"
    assert request["stage6_arm_id"] == "compression_only"
    assert request["stage6_pipeline_id"] == "tvm:compression_only:max_trials=0"
    assert request["batch_size"] == 2
    assert request["row_sha256"][rows[0]["manifest_job_id"]] == payload_sha256(rows[0])
    unsigned = {key: value for key, value in request.items() if key != "measurement_request_sha256"}
    assert request["measurement_request_sha256"] == payload_sha256(unsigned)


def test_ctt_pipeline_identity_is_distinct_from_compression_only() -> None:
    rows = [_row("id", [16, 48, 64])]
    compression = build_scoped_validation_request(
        rows, arm_id="compression_only", backend="tvm"
    )
    ctt = build_scoped_validation_request(
        rows, arm_id="compress_then_tune", backend="tvm"
    )

    assert compression["stage6_pipeline_id"] != ctt["stage6_pipeline_id"]
    assert compression["measurement_request_sha256"] != ctt["measurement_request_sha256"]


def test_scope_joint_audit_excludes_unrelated_failed_model_tasks() -> None:
    audit = {
        "schema_version": "stage5_independent_validation_audit_v1",
        "all_tasks_passed": False,
        "tasks": [
            {"task_id": "S5-PYR-TVM", "passed": True, "configurations": [{"configuration_id": "pyr", "consistency": {"passed": True}}]},
            {"task_id": "S5-COD-TVM", "passed": False, "configurations": [{"configuration_id": "cod", "consistency": {"passed": False}}]},
            {"task_id": "S5-PYR-TRT", "passed": True, "configurations": [{"configuration_id": "pyr-trt", "consistency": {"passed": True}}]},
        ],
    }

    scoped = scope_passed_validation_tasks(
        audit, task_ids=["S5-PYR-TVM", "S5-PYR-TRT"]
    )

    assert scoped["all_tasks_passed"] is True
    assert scoped["task_count"] == 2
    assert scoped["configuration_count"] == 2
    assert [task["task_id"] for task in scoped["tasks"]] == ["S5-PYR-TVM", "S5-PYR-TRT"]
    assert audit["all_tasks_passed"] is False


def test_scope_joint_audit_rejects_failed_requested_task() -> None:
    audit = {
        "schema_version": "stage5_independent_validation_audit_v1",
        "tasks": [{"task_id": "S5-PYR-TVM", "passed": False, "configurations": []}],
    }
    with pytest.raises(ValueError, match="not passed"):
        scope_passed_validation_tasks(audit, task_ids=["S5-PYR-TVM"])
