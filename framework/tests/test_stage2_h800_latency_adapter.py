from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/stage2_h800_b1_latency_command.py"


def _load_adapter_module():
    spec = importlib.util.spec_from_file_location("stage2_h800_b1_latency_command", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class Stage2H800LatencyAdapterTest(unittest.TestCase):
    def test_payload_from_b1_result_selects_tuned_latency(self):
        adapter = _load_adapter_module()
        result = {
            "label": "smoke_base",
            "onnx": "base_backbone.onnx",
            "trials": 2,
            "reps": 50,
            "default_us": 2200.0,
            "tuned_us": 1400.0,
            "e2e_ratio": 1.571,
            "tune_s": 12,
            "status": "DONE",
        }

        payload = adapter.payload_from_b1_result(
            result,
            schedule_policy="metaschedule_tuned",
            raw_artifact="/tmp/b1.json",
            log_artifact="/tmp/b1.log.json",
            b1_script="/exdata/jichengzhi/s2_tvm/b1_single_width.py",
            onnx_path="/exdata/jichengzhi/s2_tvm/models/base_backbone.onnx",
        )

        self.assertEqual(payload["latency_p50_us"], 1400.0)
        self.assertEqual(payload["latency_mean_us"], 1400.0)
        self.assertEqual(payload["latency_min_us"], 1400.0)
        self.assertEqual(payload["measure_iters"], 50)
        self.assertEqual(payload["repeat"], 5)
        self.assertEqual(payload["raw_artifact"], "/tmp/b1.json")
        self.assertIn("H800 TVM", payload["provenance"])

    def test_payload_from_b1_result_selects_default_latency(self):
        adapter = _load_adapter_module()
        result = {
            "label": "smoke_base",
            "onnx": "base_backbone.onnx",
            "trials": 0,
            "reps": 20,
            "default_us": 2100.0,
            "tuned_us": -1.0,
            "e2e_ratio": -1.0,
            "tune_s": 0,
            "status": "DONE",
        }

        payload = adapter.payload_from_b1_result(
            result,
            schedule_policy="default",
            raw_artifact="/tmp/b1.json",
            log_artifact="/tmp/b1.log.json",
            b1_script="/exdata/jichengzhi/s2_tvm/b1_single_width.py",
            onnx_path="/exdata/jichengzhi/s2_tvm/models/base_backbone.onnx",
        )

        self.assertEqual(payload["latency_p50_us"], 2100.0)
        self.assertEqual(payload["tvm_strategy"], "relax_default")

    def test_payload_rejects_failed_or_nonpositive_latency(self):
        adapter = _load_adapter_module()
        with self.assertRaisesRegex(adapter.AdapterError, "not DONE"):
            adapter.payload_from_b1_result(
                {"status": "FAILED", "default_us": 1.0, "tuned_us": 1.0},
                schedule_policy="metaschedule_tuned",
                raw_artifact="/tmp/b1.json",
                log_artifact="/tmp/b1.log.json",
                b1_script="/exdata/jichengzhi/s2_tvm/b1_single_width.py",
                onnx_path="/exdata/jichengzhi/s2_tvm/models/base_backbone.onnx",
            )

        with self.assertRaisesRegex(adapter.AdapterError, "positive latency"):
            adapter.payload_from_b1_result(
                {"status": "DONE", "default_us": 1.0, "tuned_us": -1.0},
                schedule_policy="metaschedule_tuned",
                raw_artifact="/tmp/b1.json",
                log_artifact="/tmp/b1.log.json",
                b1_script="/exdata/jichengzhi/s2_tvm/b1_single_width.py",
                onnx_path="/exdata/jichengzhi/s2_tvm/models/base_backbone.onnx",
            )

    def test_cli_prints_generator_json_from_existing_b1_result(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            result_path = tmp / "b1.json"
            result_path.write_text(
                json.dumps(
                    {
                        "label": "smoke_base",
                        "onnx": "base_backbone.onnx",
                        "trials": 1,
                        "reps": 10,
                        "default_us": 2000.0,
                        "tuned_us": 1500.0,
                        "e2e_ratio": 1.333,
                        "tune_s": 3,
                        "status": "DONE",
                    }
                ),
                encoding="utf-8",
            )

            proc = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--label",
                    "smoke_base",
                    "--onnx-file",
                    "base_backbone.onnx",
                    "--out-json",
                    str(result_path),
                    "--schedule-policy",
                    "metaschedule_tuned",
                    "--skip-run",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )

        self.assertEqual(proc.returncode, 0, proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertEqual(payload["latency_p50_us"], 1500.0)
        self.assertEqual(payload["build_status"], "success")


if __name__ == "__main__":
    unittest.main()
