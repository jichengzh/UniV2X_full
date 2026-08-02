from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.stage5_finalize_independent_validation_v1 import (
    _source_wrapper,
    evaluate_consistency,
    normalize_ap_report,
    normalize_performance_repeat,
    successful_state_for_configuration,
    unique_successful_ap_terminal,
)


def _write(path: Path, payload: object) -> str:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Stage5FinalizeIndependentValidationV1Tests(unittest.TestCase):
    def test_normalizes_real_performance_and_full_ap_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            performance = root / "performance.json"
            performance_sha = _write(
                performance,
                {
                    "status": "success",
                    "latency": {"latency_ms_p50": 1.25},
                    "energy": {"joules_per_inference": 0.75},
                },
            )
            repeat = normalize_performance_repeat(
                task_id="S5-PYR-TVM",
                configuration_id="config-1",
                repeat_index=1,
                state={
                    "status": "success",
                    "start_time_unix": 100.0,
                    "end_time_unix": 110.0,
                    "result_json": str(performance),
                    "result_sha256": performance_sha,
                },
            )
            ap = root / "ap.json"
            ap_sha = _write(
                ap,
                {
                    "status": "success",
                    "processed_samples": 1789,
                    "failed_samples": 0,
                    "ap30": 0.8,
                    "ap50": 0.7,
                    "ap70": 0.6,
                },
            )
            normalized_ap = normalize_ap_report(
                task_id="S5-PYR-TVM",
                configuration_id="config-1",
                terminal={
                    "stage": "full",
                    "status": "success",
                    "report_path": str(ap),
                    "report_sha256": ap_sha,
                },
            )

        self.assertEqual(repeat["latency_ms"], 1.25)
        self.assertEqual(repeat["energy_j"], 0.75)
        self.assertEqual(repeat["hardware_id"], "h800")
        self.assertEqual(normalized_ap["processed_samples"], 1789)
        self.assertEqual(normalized_ap["ap70"], 0.6)

    def test_rejects_non_full_or_incomplete_ap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ap.json"
            digest = _write(path, {"status": "success", "processed_samples": 16})
            with self.assertRaisesRegex(ValueError, "full AP"):
                normalize_ap_report(
                    task_id="task",
                    configuration_id="config",
                    terminal={
                        "stage": "sanity",
                        "status": "success",
                        "report_path": str(path),
                        "report_sha256": digest,
                    },
                )

    def test_source_wrapper_binds_manifest_sha_and_nested_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts = {}
            for name in ("checkpoint", "onnx", "calibration", "summary"):
                path = root / f"{name}.bin"
                path.write_bytes(name.encode())
                artifacts[name] = path
            source_path = root / "source.json"
            source_payload = {
                "status": "ready",
                "source_plan_sha256": "a" * 64,
                "checkpoint_path": str(artifacts["checkpoint"]),
                "checkpoint_sha256": hashlib.sha256(artifacts["checkpoint"].read_bytes()).hexdigest(),
                "onnx_path": str(artifacts["onnx"]),
                "onnx_sha256": hashlib.sha256(artifacts["onnx"].read_bytes()).hexdigest(),
                "calibration_path": str(artifacts["calibration"]),
                "calibration_sha256": hashlib.sha256(artifacts["calibration"].read_bytes()).hexdigest(),
                "calibration_summary_path": str(artifacts["summary"]),
                "calibration_summary_sha256": hashlib.sha256(artifacts["summary"].read_bytes()).hexdigest(),
            }
            source_sha = _write(source_path, source_payload)
            bound = {
                "source_evidence_path": str(source_path),
                "source_evidence_sha256": source_sha,
                "source_plan_sha256": "a" * 64,
            }

            wrapper = _source_wrapper("task", "config", bound)
            artifacts["onnx"].write_bytes(b"drift")
            with self.assertRaisesRegex(ValueError, "nested artifact SHA mismatch"):
                _source_wrapper("task", "config", bound)

        self.assertEqual(wrapper["raw_source_evidence_sha256"], source_sha)

    def test_maps_manifest_configuration_to_executor_job_id(self) -> None:
        state = successful_state_for_configuration(
            "config-id",
            jobs=[{"manifest_job_id": "config-id", "job_id": "group|tvm_int8"}],
            states=[{"job_id": "group|tvm_int8", "status": "success"}],
        )

        self.assertEqual(state["job_id"], "group|tvm_int8")

    def test_ap_terminal_accepts_exact_resume_duplicate(self) -> None:
        terminal = {
            "job_id": "config-id",
            "record_type": "job_terminal",
            "stage": "full",
            "status": "success",
            "report_path": "/tmp/ap.json",
            "report_sha256": "a" * 64,
        }
        selected = unique_successful_ap_terminal(
            "config-id", [terminal, dict(terminal)]
        )
        with self.assertRaisesRegex(ValueError, "one unique successful"):
            unique_successful_ap_terminal(
                "config-id",
                [terminal, {**terminal, "report_sha256": "b" * 64}],
            )

        self.assertEqual(selected["report_sha256"], "a" * 64)

    def test_consistency_gate_rejects_large_rerun_drift(self) -> None:
        repeats = [
            {"latency_ms": 20.0, "energy_j": 5.0},
            {"latency_ms": 21.0, "energy_j": 5.1},
            {"latency_ms": 22.0, "energy_j": 5.2},
        ]
        with self.assertRaisesRegex(ValueError, "independent validation drift"):
            evaluate_consistency(
                {"latency_ms": 10.0, "energy_j": 4.0, "ap70": 0.6},
                repeats,
                {"ap70": 0.5},
            )

        result = evaluate_consistency(
            {"latency_ms": 10.0, "energy_j": 4.0, "ap70": 0.6},
            repeats,
            {"ap70": 0.5},
            raise_on_failure=False,
        )
        self.assertFalse(result["passed"])


if __name__ == "__main__":
    unittest.main()
