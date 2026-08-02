import hashlib
import json
from pathlib import Path

import pytest

from scripts.fcooper_tvm_repair_schedule_ap_binding_v1 import (
    repair_schedule_feedback,
)
from scripts.fcooper_tvm_evidence_pools_v1 import normalize_success_row
from scripts.stage6_finalize_fcooper_tvm_v1 import validate_success_evidence


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, bytes):
        path.write_bytes(payload)
    else:
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    execution = tmp_path / "source_execution"
    module = _write(execution / "route" / "model.so", b"module")
    _write(execution / "route" / "ms_work_dir" / "database_workload.json", b"[]\n")
    _write(
        execution / "route" / "ms_work_dir" / "database_tuning_record.json",
        b"[]\n",
    )
    checkpoint = _write(execution / "checkpoint.pth", b"checkpoint")
    onnx = _write(execution / "model.onnx", b"onnx")
    provenance = _write(
        execution / "source_reuse_audit.json",
        {
            "passed": True,
            "backend_neutral_only": True,
            "resolved_source_contract": {
                "checkpoint_path": str(checkpoint),
                "onnx_path": str(onnx),
            },
            "reused_artifacts": {
                "checkpoint": {"path": str(checkpoint), "sha256": _sha(checkpoint)},
                "onnx": {
                    "path": str(onnx),
                    "sha256": _sha(onnx),
                },
            },
        },
    )
    old_ap = _write(execution / "old_ap.json", {"stale": True})
    performance = _write(
        execution / "performance.json",
        {
            "schema": "route_b_fp16_auto_result_v1",
            "precision": "fp32",
            "status": "success",
            "build_success": True,
            "gold_measurement_complete": True,
            "artifact_path": str(module),
            "artifact_digest": _sha(module),
            "latency": {"latency_ms_p50": 100.0},
            "energy": {"joule_per_inference": 25.0},
        },
    )
    feedback = {
        "row_id": "S6-FCO-TVM-SCHEDULE-ONLY-MEASURE-V1-ROW-test",
        "task_id": "S6-FCO-TVM-SCHEDULE-ONLY-MEASURE-V1",
        "model": "fcooper",
        "backend": "tvm_auto",
        "hardware_id": "h800",
        "dispatch_key": "tvm_auto",
        "capability_profile_id": "h800-tvm-fcooper-probe-conditioned-v1",
        "training_source": "online_feedback",
        "terminal_status": "measured_success_gold",
        "width": [64, 128, 256, 128, 256],
        "q_mode": "fp32",
        "tvm_trials": 64,
        "ap30": 0.91,
        "ap50": 0.82,
        "ap70": 0.6328,
        "latency_ms": 100.0,
        "energy_j": 25.0,
        "tvm_artifact_path": str(module),
        "tvm_artifact_sha256": _sha(module),
        "checkpoint_sha256": _sha(checkpoint),
        "resolved_source_evidence_sha256": _sha(provenance),
        "performance_result_json": str(performance),
        "performance_result_sha256": _sha(performance),
        "graph_features": {
            "onnx_path": str(onnx),
            "onnx_sha256": _sha(onnx),
        },
        "ap_report_path": str(old_ap),
        "ap_report_sha256": _sha(old_ap),
    }
    feedback["actual_feedback_row_sha256"] = hashlib.sha256(
        json.dumps(feedback, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    feedback_path = _write(execution / "feedback_row.json", feedback)

    prediction = _write(tmp_path / "validation" / "predictions.jsonl", b"{}\n")
    full_ap_payload = {
        "schema_version": "fcooper_tvm_fp32_ap_report_v1",
        "status": "success_full",
        "dataset": "OPV2V",
        "split": "test",
        "requested_samples": 2170,
        "processed_samples": 2170,
        "dataset_samples": 2170,
        "backend_calls": 2170,
        "failed_samples": 0,
        "fallback_samples": 0,
        "gates": {"full": True, "sanity": True, "status": "success_full"},
        "execution_device": {
            "physical_gpu_id": 7,
            "cuda_visible_devices": "7",
        },
        "numerical_contract": {
            "silent_fallback_forbidden": True,
            "artifact_compute_dtype": "float32",
        },
        "artifact_path": str(module),
        "artifact_sha256": _sha(module),
        "checkpoint_sha256": _sha(checkpoint),
        "prediction_sha256": _sha(prediction),
        "sha256": {
            "artifact": _sha(module),
            "checkpoint": _sha(checkpoint),
            "prediction": _sha(prediction),
        },
        "ap30": 0.9101,
        "ap50": 0.8201,
        "ap70": 0.6329,
    }
    full_ap = _write(tmp_path / "validation" / "full_ap.json", full_ap_payload)
    validation = _write(
        tmp_path / "validation" / "validation_manifest.json",
        {
            "schema_version": "fcooper_tvm_gpu7_winner_validation_v1",
            "row_id": feedback["row_id"],
            "gpu_index": 7,
            "full_ap": {
                "gpu_index": 7,
                "ap30": full_ap_payload["ap30"],
                "ap50": full_ap_payload["ap50"],
                "ap70": full_ap_payload["ap70"],
                "report": {"path": str(full_ap), "sha256": _sha(full_ap)},
                "prediction": {
                    "path": str(prediction),
                    "sha256": _sha(prediction),
                },
                "prediction_sha256": _sha(prediction),
                "checkpoint_sha256": _sha(checkpoint),
                "tvm_module_sha256": _sha(module),
            },
        },
    )
    assert provenance.is_file()
    return feedback_path, validation


def test_creates_immutable_repaired_feedback_from_gpu7_full_ap(
    tmp_path: Path,
) -> None:
    feedback, validation = _fixture(tmp_path)

    result = repair_schedule_feedback(
        source_feedback=feedback,
        winner_validation=validation,
        output_root=tmp_path / "repair",
    )

    repaired = json.loads(Path(result["feedback_row"]["path"]).read_text())
    assert repaired["ap70"] == pytest.approx(0.6329)
    assert repaired["ap_report_path"].endswith("full_ap.json")
    assert repaired["source_actual_feedback_row_sha256"] == json.loads(
        feedback.read_text()
    )["actual_feedback_row_sha256"]
    canonical = {
        key: value
        for key, value in repaired.items()
        if key != "actual_feedback_row_sha256"
    }
    assert repaired["actual_feedback_row_sha256"] == hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert Path(result["source_reuse_audit"]["path"]).is_file()

    normalized = normalize_success_row(
        repaired,
        execution_dir=Path(result["feedback_row"]["path"]).parent,
    )
    validated = validate_success_evidence(normalized, root=tmp_path)
    assert validated["ap70"] == pytest.approx(0.6329)

    repeated = repair_schedule_feedback(
        source_feedback=feedback,
        winner_validation=validation,
        output_root=tmp_path / "repair",
    )
    assert repeated == result


def test_accepts_h800_gpu0_full_ap_validation(tmp_path: Path) -> None:
    feedback, validation = _fixture(tmp_path)
    payload = json.loads(validation.read_text())
    report_path = Path(payload["full_ap"]["report"]["path"])
    report = json.loads(report_path.read_text())
    report["execution_device"] = {
        "physical_gpu_id": 0,
        "cuda_visible_devices": "0",
    }
    _write(report_path, report)
    payload["gpu_index"] = 0
    payload["full_ap"]["gpu_index"] = 0
    payload["full_ap"]["report"]["sha256"] = _sha(report_path)
    _write(validation, payload)

    result = repair_schedule_feedback(
        source_feedback=feedback,
        winner_validation=validation,
        output_root=tmp_path / "repair",
    )

    assert result["passed"] is True


def test_rejects_gpu7_ap_drift_beyond_tolerance(tmp_path: Path) -> None:
    feedback, validation = _fixture(tmp_path)
    payload = json.loads(validation.read_text())
    report_path = Path(payload["full_ap"]["report"]["path"])
    report = json.loads(report_path.read_text())
    report["ap70"] = 0.60
    _write(report_path, report)
    payload["full_ap"]["ap70"] = 0.60
    payload["full_ap"]["report"]["sha256"] = _sha(report_path)
    _write(validation, payload)

    with pytest.raises(ValueError, match="AP drift"):
        repair_schedule_feedback(
            source_feedback=feedback,
            winner_validation=validation,
            output_root=tmp_path / "repair",
        )


def test_rejects_drifted_source_provenance(tmp_path: Path) -> None:
    feedback, validation = _fixture(tmp_path)
    provenance = feedback.parent / "source_reuse_audit.json"
    payload = json.loads(provenance.read_text())
    payload["unexpected_drift"] = True
    _write(provenance, payload)

    with pytest.raises(ValueError, match="provenance SHA"):
        repair_schedule_feedback(
            source_feedback=feedback,
            winner_validation=validation,
            output_root=tmp_path / "repair",
        )


def test_rejects_stale_redundant_winner_identity_sha(tmp_path: Path) -> None:
    feedback, validation = _fixture(tmp_path)
    payload = json.loads(validation.read_text())
    payload["full_ap"]["tvm_module_sha256"] = "0" * 64
    _write(validation, payload)

    with pytest.raises(ValueError, match="winner validation artifact SHA"):
        repair_schedule_feedback(
            source_feedback=feedback,
            winner_validation=validation,
            output_root=tmp_path / "repair",
        )
