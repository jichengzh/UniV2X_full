from __future__ import annotations

from pathlib import Path
import hashlib

import pytest

from scripts.fcooper_execute_measurement_row_v2 import (
    validate_recovery_training_evidence,
    validate_control_request,
    validate_formal_request,
)
from framework.stage2.canonical_search_v3 import build_capability_profile
from framework.stage5.single_target_search_v2 import (
    SearchTask,
    build_measurement_request,
    validate_search_task,
)


ROOT = Path(__file__).resolve().parents[2]


def test_formal_runner_requires_recovery_before_export_and_measurement() -> None:
    source = (ROOT / "scripts/fcooper_execute_measurement_row_v2.py").read_text()

    recovery = source.index("fcooper_recovery_train_v2.py")
    export = source.index("fcooper_export_source_v2.py")
    profile = source.index("trt_profile_v1.py")
    ap = source.index("fcooper_trt_ap_bridge_v1.py")
    assert recovery < export < profile < ap
    assert "fcooper_materialize_source_v1.py" not in source
    assert '"--checkpoint"' in source
    assert "recovery_training_report_sha256" in source


def test_reused_source_records_zero_work_for_skipped_phases() -> None:
    source = (ROOT / "scripts/fcooper_execute_measurement_row_v2.py").read_text()

    assert 'timings["recovery_initialization_seconds"] = 0.0' in source
    assert 'timings["recovery_training_seconds"] = 0.0' in source
    assert 'timings["onnx_export_seconds"] = 0.0' in source


def test_numeric_gate_uses_the_full_recovery_evidence_validator() -> None:
    source = (ROOT / "scripts/fcooper_gate_supervisor_v2.sh").read_text()

    assert "validate_recovery_training_evidence" in source
    assert "recovery_training_contract.json" in source
    assert "recovered_checkpoint.pth" in source


def _request() -> dict:
    profile = build_capability_profile(
        capability_profile_id="h800-trt",
        hardware_target="h800",
        compiler_fingerprint=hashlib.sha256(b"trt").hexdigest(),
        dispatch_key="trt_engine",
        features={"int8_propagation": 1.0, "qdq_fold": 1.0},
    )
    task = SearchTask("S5-FCO-TRT-V2", "fcooper", "h800", profile)
    task_contract = validate_search_task(task)
    row = {
        "row_id": "formal-row",
        "manifest_job_id": "formal-row",
        "group_id": "fcooper|formal",
        "task_id": task.task_id,
        "task_sha256": task_contract["task_sha256"],
        "model": "fcooper",
        "hardware_id": "h800",
        "capability_profile_id": "h800-trt",
        "width": [32, 64, 128, 64, 128],
        "width_schema": [
            "backbone.s0",
            "backbone.s1",
            "backbone.s2",
            "neck.deblock",
            "neck.output",
        ],
        "q_mode": "fp16",
        "source_status": "ready",
        "materialization_kind": "fcooper_scanner_materialize_export",
        "source_evidence_kind": "scanner_derived_plan",
        "source_contract": {},
        "source_evidence_sha256": "a" * 64,
        "graph_features": {"group_id": "fcooper|formal"},
    }
    rows = [
        {
            **row,
            "row_id": f"formal-row-{index}",
            "manifest_job_id": f"formal-row-{index}",
            "group_id": f"fcooper|formal-{index}",
            "graph_features": {"group_id": f"fcooper|formal-{index}"},
        }
        for index in range(4)
    ]
    return build_measurement_request(task=task, selected_rows=rows, round_index=0)


def test_formal_runner_rejects_old_task_or_request_sha_drift() -> None:
    request = _request()
    request["task_id"] = "S5-FCO-TRT"
    with pytest.raises(ValueError, match="task"):
        validate_formal_request(request, row_index=0)

    request = _request()
    request["round_index"] = 3
    with pytest.raises(ValueError, match="SHA"):
        validate_formal_request(request, row_index=0)


def test_control_runner_accepts_single_fixed_arm_row() -> None:
    row = {
        "row_id": "schedule-only-original",
        "manifest_job_id": "schedule-only-original",
        "task_id": "S6-FCO-TRT-SCHEDULE-V2",
        "task_sha256": "b" * 64,
        "model": "fcooper",
        "hardware_id": "h800",
        "width": [64, 128, 256, 128, 256],
        "q_mode": "fp32",
    }
    payload = {
        "schema_version": "stage6_fcooper_control_measurement_request_v2",
        "task_id": row["task_id"],
        "task_sha256": row["task_sha256"],
        "batch_size": 1,
        "real_h800_measurement_required": True,
        "row_sha256": {"schedule-only-original": hashlib.sha256(
            __import__("json").dumps(
                row, ensure_ascii=True, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()},
        "rows": [row],
    }
    payload["measurement_request_sha256"] = hashlib.sha256(
        __import__("json").dumps(
            payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()

    selected = validate_control_request(payload, row_index=0)

    assert selected["row_id"] == "schedule-only-original"


def test_recovery_evidence_rejects_minimal_forged_report(tmp_path: Path) -> None:
    json = __import__("json")
    contract = {
        "schema_version": "fcooper_recovery_training_contract_v2",
        "seed": 20260723,
        "start_epoch": 23,
        "minimum_epochs": 4,
        "recovery_epochs": 8,
        "full_train_split": True,
        "full_validation_split": True,
        "amp_fp16": True,
    }
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract))
    config = tmp_path / "config.yaml"
    initial = tmp_path / "initial.pth"
    recovered = tmp_path / "recovered.pth"
    config.write_text("model: fcooper\n")
    initial.write_bytes(b"initial")
    recovered.write_bytes(b"recovered")
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "schema_version": "fcooper_recovery_training_report_v2",
                "status": "success",
                "initialization_policy": "scanner_dependency_l1_v2",
                "epochs_completed": 4,
                "recovered_checkpoint_sha256": hashlib.sha256(
                    recovered.read_bytes()
                ).hexdigest(),
            }
        )
    )

    with pytest.raises(ValueError, match="recovery"):
        validate_recovery_training_evidence(
            report_path=report,
            recovery_contract_path=contract_path,
            config_path=config,
            initial_checkpoint_path=initial,
            recovered_checkpoint_path=recovered,
        )
