import hashlib
import json
from pathlib import Path

from scripts.fcooper_tvm_independent_performance_v1 import resolve_paths


def _write(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def test_resolve_paths_binds_fp16_module_and_database(tmp_path: Path) -> None:
    label = tmp_path / "route/row_fp16"
    module = label / "route_b_fp16_auto.so"
    module_sha = _write(module, b"module")
    _write(label / "ms_work_dir/database_workload.json", b"workload")
    _write(label / "ms_work_dir/database_tuning_record.json", b"record")
    _write(label / "route_b_fp16_auto_result.json", b"{}")
    feedback = tmp_path / "feedback.json"
    feedback.write_text(
        json.dumps(
            {
                "row_id": "row",
                "q_mode": "fp16",
                "tvm_artifact_path": str(module),
                "tvm_artifact_sha256": module_sha,
            }
        )
    )

    paths = resolve_paths(feedback)

    assert paths["module"] == module
    assert paths["database"] == label / "ms_work_dir"
    assert paths["route_result"] == label / "route_b_fp16_auto_result.json"


def test_resolve_paths_accepts_zero_trial_database_contract(tmp_path: Path) -> None:
    label = tmp_path / "route/row_fp16"
    module = label / "route_b_fp16_auto.so"
    module_sha = _write(module, b"module")
    _write(label / "route_b_fp16_auto_result.json", b"{}")
    feedback = tmp_path / "feedback.json"
    feedback.write_text(
        json.dumps(
            {
                "row_id": "row",
                "q_mode": "fp16",
                "tvm_trials": 0,
                "tvm_artifact_path": str(module),
                "tvm_artifact_sha256": module_sha,
            }
        )
    )

    paths = resolve_paths(feedback)

    assert paths["database"] == label / "zero_trial_database_contract.json"
    assert paths["database"].is_file()


def test_resolve_paths_uses_fp32_route_result_for_schedule_only(tmp_path: Path) -> None:
    label = tmp_path / "route/row_fp32"
    module = label / "route_b_fp32_auto.so"
    module_sha = _write(module, b"module")
    _write(label / "ms_work_dir/database_workload.json", b"workload")
    _write(label / "ms_work_dir/database_tuning_record.json", b"record")
    route_result = label / "route_b_fp32_auto_result.json"
    _write(route_result, b"{}")
    feedback = tmp_path / "feedback.json"
    feedback.write_text(
        json.dumps(
            {
                "row_id": "row",
                "q_mode": "fp32",
                "tvm_trials": 64,
                "tvm_artifact_path": str(module),
                "tvm_artifact_sha256": module_sha,
            }
        )
    )

    paths = resolve_paths(feedback)

    assert paths["route_result"] == route_result


def test_resolve_paths_requires_int8_quant_artifacts(tmp_path: Path) -> None:
    label = tmp_path / "route/row_int8"
    module = label / "route_b_int8_auto_decomp.vmexec"
    module_sha = _write(module, b"module")
    _write(label / "tuning_database/database_workload.json", b"workload")
    _write(label / "tuning_database/database_tuning_record.json", b"record")
    _write(label / "route_b_int8_auto_decomp_result.json", b"{}")
    _write(label / "runtime_weights_int8.npz", b"weights")
    feedback = tmp_path / "feedback.json"
    feedback.write_text(
        json.dumps(
            {
                "row_id": "row",
                "q_mode": "int8",
                "tvm_artifact_path": str(module),
                "tvm_artifact_sha256": module_sha,
                "quant_contract_path": str(tmp_path / "missing_quant.json"),
            }
        )
    )

    try:
        resolve_paths(feedback)
    except FileNotFoundError as error:
        assert "missing_quant" in str(error)
    else:
        raise AssertionError("missing INT8 quant contract should fail")
