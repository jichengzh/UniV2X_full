import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).parents[2] / "scripts/stage6_collect_paper_evidence_v1.py"
SPEC = importlib.util.spec_from_file_location("stage6_collect_paper_evidence_v1", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _write_attempts(root: Path, statuses: list[tuple[str, bool]]) -> None:
    output = root / "tvm/tune_then_compress/attempts"
    output.mkdir(parents=True)
    for index, (terminal_status, full_transfer) in enumerate(statuses):
        (output / f"attempt_{index:02d}.json").write_text(
            json.dumps(
                {
                    "terminal_status": terminal_status,
                    "full_transfer": full_transfer,
                    "q_dispatch_reached": False,
                    "compressed_shape_retuned": False,
                    "fallback_used": False,
                }
            ),
            encoding="utf-8",
        )


def test_tune_then_compress_all_failures_are_a_verified_terminal(tmp_path) -> None:
    _write_attempts(tmp_path, [("feasibility_failure", False)] * 16)

    arm = MODULE._tune_then_compress_arm(tmp_path, "tvm")

    assert arm["status"] == "complete_failure"
    assert arm["independent_validation_complete"] is True
    assert arm["failure_evidence_sha_verified"] is True
    assert arm["failure_count"] == 16
    assert arm["failure_reason"] == "frozen_backend_policy_transfer_failed_for_all_genomes"
    assert arm["unmeasured_transfer_success_count"] == 0


def test_tune_then_compress_transfer_success_remains_unmeasured(tmp_path) -> None:
    _write_attempts(
        tmp_path,
        [("transferred_success", True)]
        + [("feasibility_failure", False)] * 15,
    )

    arm = MODULE._tune_then_compress_arm(tmp_path, "tvm")

    assert arm["status"] == "running"
    assert arm["independent_validation_complete"] is False
    assert arm["failure_evidence_sha_verified"] is False
    assert arm["failure_count"] == 15
    assert arm["unmeasured_transfer_success_count"] == 1


def test_schedule_only_base_fp32_ap_collapse_is_a_numerical_failure(tmp_path) -> None:
    root = tmp_path / "trt/schedule_only"
    ap_root = root / "ap/full_1789"
    ap_root.mkdir(parents=True)
    (root / "performance_result.json").write_text(
        json.dumps({"build_success": True, "lat_p50_ms": 1.0, "energy_j": 0.2}),
        encoding="utf-8",
    )
    (ap_root / "full_ap_eval_report.json").write_text(
        json.dumps({"status": "success", "processed_samples": 1789, "ap70": 0.0}),
        encoding="utf-8",
    )

    arm = MODULE._schedule_arm(tmp_path, "trt", baseline_ap70=0.63)

    assert arm["status"] == "complete_failure"
    assert arm["failure_reason"] == "base_fp32_backend_numerical_contract_failure"
    assert arm["failure_evidence_sha_verified"] is True
    assert arm["failure_count"] == 1
    assert arm["points"] == []


def test_schedule_only_prefers_verified_epoch23_repair(tmp_path) -> None:
    root = tmp_path / "trt/schedule_only"
    ap_root = root / "ap/full_1789"
    ap_root.mkdir(parents=True)
    (root / "performance_result.json").write_text(
        json.dumps({"build_success": True, "lat_p50_ms": 1.0, "energy_j": 0.2}),
        encoding="utf-8",
    )
    (ap_root / "full_ap_eval_report.json").write_text(
        json.dumps({"status": "success", "processed_samples": 1789, "ap70": 0.0}),
        encoding="utf-8",
    )
    repair_root = tmp_path / "trt/schedule_only_epoch23_repair_v1"
    repair_ap = repair_root / "ap.json"
    repair_perf = repair_root / "performance.json"
    repair_root.mkdir(parents=True)
    repair_ap.write_text(json.dumps({"ap70": 0.631}), encoding="utf-8")
    repair_perf.write_text(
        json.dumps({"lat_p50_ms": 1.1, "energy_j": 0.4}), encoding="utf-8"
    )
    import hashlib

    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    (repair_root / "repair_audit.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "checkpoint_epoch": 23,
                "processed_samples": 1789,
                "ap_report_path": str(repair_ap),
                "ap_report_sha256": sha(repair_ap),
                "performance_repeats": [
                    {"path": str(repair_perf), "sha256": sha(repair_perf)}
                    for _ in range(3)
                ],
            }
        ),
        encoding="utf-8",
    )

    arm = MODULE._schedule_arm(tmp_path, "trt", baseline_ap70=0.63)

    assert arm["status"] == "complete"
    assert arm["points"][0]["AP70"] == 0.631
    assert arm["points"][0]["latency_ms"] == 1.1


def test_tvm_schedule_and_default_prefer_verified_epoch23_repair(tmp_path) -> None:
    import hashlib

    repair_root = tmp_path / "tvm/default_schedule_epoch23_repair_v1"
    repair_root.mkdir(parents=True)
    default_ap = repair_root / "default_ap.json"
    tuned_ap = repair_root / "tuned_ap.json"
    default_ap.write_text(json.dumps({"ap70": 0.631}), encoding="utf-8")
    tuned_ap.write_text(json.dumps({"ap70": 0.632}), encoding="utf-8")
    repeat_files = []
    for index in range(3):
        path = repair_root / f"repeat_{index}.json"
        path.write_text(json.dumps({"status": "success"}), encoding="utf-8")
        repeat_files.append(path)

    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    audit = {
        "status": "passed",
        "checkpoint_epoch": 23,
        "arms": {
            "default": {
                "ap70": 0.631,
                "latency_median_ms": 56.3,
                "energy_median_j": 20.2,
                "path": str(default_ap),
                "sha256": sha(default_ap),
                "tuning_trials": 0,
                "schedule_policy": "tvm_default_zero_trial",
            },
            "tuned": {
                "ap70": 0.632,
                "latency_median_ms": 56.2,
                "energy_median_j": 20.1,
                "path": str(tuned_ap),
                "sha256": sha(tuned_ap),
                "tuning_trials": 64,
                "schedule_policy": "tvm_metaschedule_64",
            },
        },
        "repeat_evidence": [
            {
                "repeat_index": index,
                "files": [{"path": str(path), "sha256": sha(path)}],
            }
            for index, path in enumerate(repeat_files)
        ],
    }
    (repair_root / "repair_audit.json").write_text(
        json.dumps(audit),
        encoding="utf-8",
    )

    arm = MODULE._schedule_arm(tmp_path, "tvm", baseline_ap70=0.63)
    baseline = MODULE._tvm_default_baseline(tmp_path)

    assert arm["status"] == "complete"
    assert arm["independent_validation_complete"] is True
    assert arm["points"][0]["AP70"] == 0.632
    assert arm["points"][0]["latency_ms"] == 56.2
    assert arm["points"][0]["independent_validation_passed"] is True
    assert baseline["backend"] == "tvm"
    assert baseline["latency_ms"] == 56.3
    assert baseline["schedule_policy"] == "tvm_default_zero_trial"

    applied = MODULE._apply_validation(
        {"schedule_only": arm},
        {},
        baseline_ap70=0.63,
    )
    assert applied["schedule_only"]["independent_validation_complete"] is True
    assert (
        applied["schedule_only"]["points"][0]["independent_validation_passed"]
        is True
    )
