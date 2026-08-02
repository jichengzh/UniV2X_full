import hashlib
import json
from pathlib import Path

from scripts.fcooper_tvm_evidence_pools_v1 import (
    build_pool,
    feedback_index,
    normalize_automatic_selection,
    normalize_success_row,
)


def _write(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, bytes):
        path.write_bytes(payload)
    else:
        path.write_text(json.dumps(payload) + "\n")
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_normalize_success_row_binds_raw_measurements(tmp_path: Path) -> None:
    execution = tmp_path / "execution" / "abc"
    checkpoint = _write(tmp_path / "source/checkpoint.pth", b"checkpoint")
    onnx = _write(execution / "prepared/model.onnx", b"onnx")
    module = _write(execution / "route/model.so", b"module")
    database = execution / "route/ms_work_dir"
    _write(database / "database_tuning_record.json", b"record")
    _write(database / "database_workload.json", b"workload")
    performance = _write(
        execution / "route/result.json",
        {"latency": {"latency_ms_p50": 2.0}, "energy": {"joules_per_inference": 0.5}},
    )
    ap = _write(execution / "ap_report.json", {"ap30": 0.9, "ap50": 0.8, "ap70": 0.6})
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
                "checkpoint": {
                    "path": str(checkpoint),
                    "sha256": _sha(checkpoint),
                },
                "onnx": {"path": str(onnx), "sha256": _sha(onnx)},
            },
        },
    )
    row = {
        "row_id": "row-1",
        "terminal_status": "measured_success_gold",
        "width": [32, 64, 128, 64, 128],
        "q_mode": "fp16",
        "ap30": 0.9,
        "ap50": 0.8,
        "ap70": 0.6,
        "latency_ms": 2.0,
        "energy_j": 0.5,
        "tvm_max_trials": 64,
        "dispatch_key": "tvm_auto",
        "checkpoint_sha256": _sha(checkpoint),
        "source_contract": {
            "checkpoint_path": str(tmp_path / "stale_checkpoint.pth")
        },
        "graph_features": {"onnx_path": str(onnx), "onnx_sha256": _sha(onnx)},
        "tvm_artifact_path": str(module),
        "tvm_artifact_sha256": _sha(module),
        "performance_result_json": str(performance),
        "performance_result_sha256": _sha(performance),
        "ap_report_path": str(ap),
        "ap_report_sha256": _sha(ap),
    }
    row["actual_feedback_row_sha256"] = hashlib.sha256(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    feedback = _write(execution / "feedback_row.json", row)

    normalized = normalize_success_row(row, execution_dir=execution)

    assert normalized["tvm_trials"] == 64
    assert normalized["tvm_trials_source_field"] == "tvm_max_trials"
    assert normalized["backend"] == "tvm_auto"
    assert normalized["backend_source_field"] == "dispatch_key"
    assert normalized["artifacts"]["feedback_row"]["sha256"] == _sha(feedback)
    assert normalized["artifacts"]["checkpoint"]["sha256"] == _sha(checkpoint)
    assert normalized["artifacts"]["tvm_database"]["path"] == str(database)
    performance_wrapper = Path(normalized["artifacts"]["performance_report"]["path"])
    assert json.loads(performance_wrapper.read_text())["row_id"] == "row-1"
    assert normalized["artifacts"]["source_provenance"]["path"] == str(provenance)


def test_feedback_index_distinguishes_screen_and_tuned_trials(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "formal"
    for execution_id, trials in (("screen", 0), ("tuned", 64)):
        _write(
            artifact_root / "execution" / execution_id / "feedback_row.json",
            {
                "row_id": "shared-row",
                "tvm_trials": trials,
                "terminal_status": "measured_success_gold",
            },
        )

    indexed = feedback_index(artifact_root)

    assert set(indexed) == {("shared-row", 0), ("shared-row", 64)}
    assert indexed[("shared-row", 0)][1].name == "screen"
    assert indexed[("shared-row", 64)][1].name == "tuned"


def test_feedback_index_accepts_signed_legacy_tvm_max_trials(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "formal"
    execution = artifact_root / "execution" / "legacy"
    row = {
        "row_id": "gear-row",
        "tvm_max_trials": 64,
        "terminal_status": "measured_success_gold",
    }
    row["actual_feedback_row_sha256"] = hashlib.sha256(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    _write(execution / "feedback_row.json", row)

    indexed = feedback_index(artifact_root)

    assert set(indexed) == {("gear-row", 64)}


def test_normalizes_tuned_selection_to_sha_bound_screen_pool(
    tmp_path: Path,
) -> None:
    source_screen = _write(
        tmp_path / "source_screen.json",
        {"rows": [{"row_id": "a"}, {"row_id": "b"}]},
    )
    normalized_screen = _write(
        tmp_path / "normalized_screen.json",
        {
            "pool_name": "compress_then_tune_screen",
            "rows": [{"row_id": "a"}, {"row_id": "b"}],
        },
    )
    selection = _write(
        tmp_path / "selection.json",
        {
            "automatic": True,
            "selected_row_ids": ["b"],
            "source_pool_path": str(source_screen),
            "source_pool_sha256": _sha(source_screen),
        },
    )

    normalized = normalize_automatic_selection(
        selection_path=selection,
        normalized_screen_path=normalized_screen,
        tuned_row_ids=["b"],
    )

    assert normalized["automatic"]
    assert normalized["selected_row_ids"] == ["b"]
    assert normalized["source_pool_sha256"] == _sha(normalized_screen)
    assert (
        normalized["source_automatic_selection"]["sha256"] == _sha(selection)
    )


def test_build_pool_preserves_credible_terminal_failure(tmp_path: Path) -> None:
    artifact_root = tmp_path / "formal"
    execution = artifact_root / "execution" / "failed"
    feedback = _write(
        execution / "feedback_row.json",
        {
            "row_id": "failed-row",
            "terminal_status": "feasibility_failure",
            "failure_reason": "automatic TVM lowering rejected the shape",
            "ap30": None,
            "ap50": None,
            "ap70": None,
            "latency_ms": None,
            "energy_j": None,
            "tvm_max_trials": 64,
            "dispatch_key": "tvm_auto",
        },
    )
    payload = json.loads(feedback.read_text())
    payload["actual_feedback_row_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    feedback.write_text(json.dumps(payload) + "\n")
    request = _write(
        tmp_path / "request.json",
        {"rows": [{"row_id": "failed-row"}]},
    )

    pool = build_pool(
        name="gear",
        expected_path=request,
        artifact_root=artifact_root,
    )

    row = pool["rows"][0]
    assert row["terminal_status"] == "feasibility_failure"
    assert row["tvm_trials"] == 64
    assert row["tvm_trials_source_field"] == "tvm_max_trials"
    assert row["backend"] == "tvm_auto"
    assert row["artifacts"]["failure_contract"]["path"] == str(feedback.resolve())
    assert row["artifacts"]["failure_contract"]["sha256"] == _sha(feedback)
