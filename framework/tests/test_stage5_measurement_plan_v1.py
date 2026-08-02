from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from framework.stage5 import measurement_plan_v1 as plan
from framework.tests.test_stage5_materialize_round_sources_v1 import _request
from scripts.stage3_execute_performance_plan_v3 import prepare_job_command


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Stage5MeasurementPlanV1Tests(unittest.TestCase):
    def test_build_plan_requires_source_evidence_and_emits_eight_automatic_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = _request(root)
            evidence_paths = {}
            for model, width in (("pyramid", "32x64x96"), ("codriving", "16x48x64")):
                group_id = f"{model}|{width}"
                source_row = next(row for row in request["rows"] if row["group_id"] == group_id)
                trt_dir = Path(source_row["source_contract"]["trt_calibration_dir"])
                trt_dir.mkdir(parents=True, exist_ok=True)
                (trt_dir / "sample_000.npy").write_bytes(f"{model}-trt-calib".encode())
                files = {}
                for name in ("checkpoint", "onnx", "calibration", "summary"):
                    path = root / model / f"{name}.bin"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(f"{model}-{name}".encode())
                    files[name] = path
                evidence = {
                    "schema_version": "stage5_source_materialization_evidence_v1",
                    "group_id": group_id,
                    "source_plan_sha256": source_row["source_evidence_sha256"],
                    "status": "ready",
                    "checkpoint_path": str(files["checkpoint"]),
                    "checkpoint_sha256": _sha(files["checkpoint"]),
                    "onnx_path": str(files["onnx"]),
                    "onnx_sha256": _sha(files["onnx"]),
                    "calibration_path": str(files["calibration"]),
                    "calibration_sha256": _sha(files["calibration"]),
                    "calibration_summary_path": str(files["summary"]),
                    "calibration_summary_sha256": _sha(files["summary"]),
                }
                evidence_path = root / f"{model}_evidence.json"
                evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
                evidence_paths[group_id] = evidence_path

            result = plan.build_performance_plan(
                request,
                source_evidence_paths=evidence_paths,
                remote_artifact_root=root / "performance_execution",
                gpus=[6, 7],
            )

        self.assertEqual(result["manifest"]["group_count"], 2)
        self.assertEqual(result["manifest"]["row_count"], 8)
        self.assertEqual(len(result["performance_jobs"]), 8)
        self.assertEqual(
            {job["runner_key"] for job in result["performance_jobs"]},
            {"tvm_fp16", "tvm_int8", "trt_fp16", "trt_int8"},
        )
        commands = "\n".join(" ".join(job["command"]) for job in result["performance_jobs"])
        self.assertNotIn("im2col", commands.lower())
        self.assertNotIn("int8_tc", commands.lower())
        self.assertNotIn("mixed", commands.lower())
        trt_int8 = [
            job for job in result["performance_jobs"] if job["runner_key"] == "trt_int8"
        ]
        self.assertEqual(len(trt_int8), 2)
        for job in trt_int8:
            command = prepare_job_command(job)
            calibration = command[command.index("--calib-dir") + 1]
            self.assertEqual(calibration, job["source_contract"]["trt_calibration_dir"])

    def test_build_plan_rejects_tampered_source_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = _request(root)
            evidence_paths = {}
            for model, width in (("pyramid", "32x64x96"), ("codriving", "16x48x64")):
                source_row = next(
                    row for row in request["rows"] if row["group_id"] == f"{model}|{width}"
                )
                artifact = root / f"{model}.bin"
                artifact.write_bytes(b"artifact")
                evidence = {
                    "schema_version": "stage5_source_materialization_evidence_v1",
                    "group_id": f"{model}|{width}",
                    "source_plan_sha256": source_row["source_evidence_sha256"],
                    "status": "ready",
                    "checkpoint_path": str(artifact),
                    "checkpoint_sha256": "0" * 64,
                    "onnx_path": str(artifact),
                    "onnx_sha256": _sha(artifact),
                    "calibration_path": str(artifact),
                    "calibration_sha256": _sha(artifact),
                    "calibration_summary_path": str(artifact),
                    "calibration_summary_sha256": _sha(artifact),
                }
                evidence_path = root / f"{model}.json"
                evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
                evidence_paths[f"{model}|{width}"] = evidence_path

            with self.assertRaisesRegex(ValueError, "SHA mismatch"):
                plan.build_performance_plan(
                    request,
                    source_evidence_paths=evidence_paths,
                    remote_artifact_root=root / "performance",
                    gpus=[6, 7],
                )

    def test_build_plan_rejects_source_plan_or_group_identity_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = _request(root)
            request["rows"][0]["model"] = "codriving"
            with self.assertRaisesRegex(ValueError, "identity drift"):
                plan.build_performance_plan(
                    request,
                    source_evidence_paths={
                        "pyramid|32x64x96": root / "missing-pyramid.json",
                        "codriving|16x48x64": root / "missing-codriving.json",
                    },
                    remote_artifact_root=root / "performance",
                    gpus=[6, 7],
                )

            request = _request(root)
            request["rows"][0]["source_contract"]["trt_calibration_dir"] += "-drift"
            with self.assertRaisesRegex(ValueError, "source plan SHA mismatch"):
                plan.build_performance_plan(
                    request,
                    source_evidence_paths={
                        "pyramid|32x64x96": root / "missing-pyramid.json",
                        "codriving|16x48x64": root / "missing-codriving.json",
                    },
                    remote_artifact_root=root / "performance",
                    gpus=[6, 7],
                )


if __name__ == "__main__":
    unittest.main()
