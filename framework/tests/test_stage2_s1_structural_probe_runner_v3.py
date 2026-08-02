from __future__ import annotations

import argparse
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "stage2_s1_structural_probe_runner_v3.py"
SPEC = importlib.util.spec_from_file_location("stage2_s1_structural_probe_runner_v3", SCRIPT)
assert SPEC and SPEC.loader
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class Stage2S1StructuralProbeRunnerV3Test(unittest.TestCase):
    def test_compiler_fingerprint_is_content_addressed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first = root / "libfirst.so"
            second = root / "libsecond.so"
            first.write_bytes(b"alpha")
            second.write_bytes(b"beta")

            fingerprint = RUNNER.build_compiler_fingerprint("1.2.3", [first, second])
            same = RUNNER.build_compiler_fingerprint("1.2.3", [second, first])
            changed = RUNNER.build_compiler_fingerprint("1.2.4", [first, second])

        self.assertRegex(fingerprint, r"^[0-9a-f]{64}$")
        self.assertEqual(fingerprint, same)
        self.assertNotEqual(fingerprint, changed)

    def test_compiler_fingerprint_fails_when_file_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            present = root / "runtime.py"
            present.write_text("print('ok')\n", encoding="utf-8")
            missing = root / "missing.so"

            with self.assertRaises(FileNotFoundError):
                RUNNER.build_compiler_fingerprint("9.9.9", [present, missing])

    def test_fp16_marks_qdq_metrics_not_applicable(self) -> None:
        metrics = RUNNER._normalize_precision_metrics(
            {
                "qdq_pairs": 0,
                "qdq_folded_pairs": 0,
                "int8_propagated_ops": 0,
                "precision_eligible_ops": 1,
            },
            "fp16",
        )

        self.assertIsNone(metrics["qdq_pairs"])
        self.assertIsNone(metrics["qdq_folded_pairs"])
        self.assertIsNone(metrics["int8_propagated_ops"])
        self.assertIsNone(metrics["precision_eligible_ops"])

    def test_int8_preserves_structural_counts(self) -> None:
        source = {
            "qdq_pairs": 2,
            "qdq_folded_pairs": 1,
            "int8_propagated_ops": 1,
            "precision_eligible_ops": 1,
        }
        self.assertEqual(RUNNER._normalize_precision_metrics(source, "int8"), source)

    def test_run_emits_top_level_runtime_compiler_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            probe = root / "probe.onnx"
            probe.write_bytes(b"onnx-bytes")
            manifest = {
                "probes": [
                    {
                        "probe_id": "neutral_probe",
                        "precision": "fp16",
                        "artifact": {
                            "path": probe.name,
                            "sha256": RUNNER._sha256(probe),
                        },
                    }
                ]
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            output_dir = root / "out"
            args = argparse.Namespace(
                backend="tvm",
                manifest=str(manifest_path),
                output=str(output_dir),
            )
            compiler_fingerprint = "a" * 64

            with mock.patch.object(
                RUNNER,
                "_collect_backend_provenance",
                return_value={
                    "backend": "tvm",
                    "compiler_version": "0.20.dev0",
                    "compiler_files": ["/tmp/fake_tvm_runtime.so"],
                    "compiler_file_sha256": {"/tmp/fake_tvm_runtime.so": "b" * 64},
                    "compiler_fingerprint": compiler_fingerprint,
                },
            ) as provenance_mock, mock.patch.object(
                RUNNER,
                "_tvm_build",
                return_value=(RUNNER._empty_counts(), "tir"),
            ):
                report = RUNNER.run(args)

        provenance_mock.assert_called_once_with("tvm")
        self.assertEqual(report["provenance"]["compiler_fingerprint"], compiler_fingerprint)
        self.assertEqual(report["provenance"]["compiler_version"], "0.20.dev0")
        self.assertEqual(report["provenance"]["backend"], "tvm")
        self.assertEqual(report["records"][0]["build_success"], True)


if __name__ == "__main__":
    unittest.main()
