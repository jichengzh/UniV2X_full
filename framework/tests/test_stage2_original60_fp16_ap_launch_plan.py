from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class Stage2Original60Fp16ApLaunchPlanTest(unittest.TestCase):
    def test_launch_plan_cli_builds_pilot_and_bulk_batches_from_fp16_ap_gaps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_root = tmp_path / "generated/original60_quant_20260627"
            jobs_dir = output_root / "jobs"
            jobs_dir.mkdir(parents=True, exist_ok=True)
            completion_queue = jobs_dir / "fp16_int8_original60_completion_queue_v1.jsonl"
            ap_queue = jobs_dir / "fp16_int8_original60_ap_true_eval_queue_v1.jsonl"

            completion_rows = [
                {
                    "schema": "original60_fp16_int8_completion_job_v1",
                    "job_id": f"original60_completion:label_{i:02d}:fp16",
                    "label": f"label_{i:02d}",
                    "precision": "fp16",
                    "width": [24 + i, 64, 128],
                    "onnx_backbone_path": f"/exdata/jichengzhi/s2_tvm/models/label_{i:02d}_backbone.onnx",
                    "workdir": f"/exdata/jichengzhi/s2_tvm/workdirs/label_{i:02d}",
                    "latency_status": "measured",
                    "energy_status": "measured",
                    "ap_status": "no_claim",
                    "required_actions": ["run_true_fp16_ap_eval"],
                    "full_network_claim": False,
                    "axis_rows": {},
                    "created_at": "2026-06-28T00:00:00Z",
                }
                for i in range(6)
            ]
            completion_queue.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in completion_rows),
                encoding="utf-8",
            )

            ap_rows = [
                {
                    "schema": "original60_fp16_int8_ap_true_eval_job_v1",
                    "job_id": f"original60_ap_true_eval:label_{i:02d}:fp16",
                    "completion_job_id": f"original60_completion:label_{i:02d}:fp16",
                    "label": f"label_{i:02d}",
                    "precision": "fp16",
                    "width": [24 + i, 64, 128],
                    "ap_eval_status": "blocked",
                    "next_action": "build_true_fp16_model_eval_backend_then_run_eval",
                    "blocker": "no_compliant_true_fp16_model_eval_source",
                    "created_at": "2026-06-28T00:00:00Z",
                }
                for i in range(6)
            ]
            ap_queue.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in ap_rows),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/stage2_generate_original60_fp16_ap_launch_plan.py"),
                    "--output-root",
                    str(output_root),
                    "--created-at",
                    "2026-06-28T00:00:00Z",
                ],
                cwd=ROOT,
                env={"PYTHONPATH": str(ROOT)},
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            plan_path = output_root / "exports/fp16_original60_ap_launch_plan_latest.json"
            plan = json.loads(plan_path.read_text(encoding="utf-8"))

            self.assertEqual(plan["summary"]["fp16_gap_labels"], 6)
            self.assertEqual(plan["summary"]["pilot_label_count"], 4)
            self.assertEqual(plan["summary"]["bulk_label_count"], 2)
            self.assertEqual(plan["pilot_batch"]["labels"], ["label_00", "label_01", "label_02", "label_03"])
            self.assertEqual(plan["pilot_batch"]["gpu_map"]["3"], ["label_00"])
            self.assertEqual(plan["pilot_batch"]["gpu_map"]["4"], ["label_01"])
            self.assertEqual(plan["pilot_batch"]["gpu_map"]["5"], ["label_02"])
            self.assertEqual(plan["pilot_batch"]["gpu_map"]["6"], ["label_03"])
            self.assertEqual(plan["bulk_batches"]["gpu_map"]["3"], ["label_04"])
            self.assertEqual(plan["bulk_batches"]["gpu_map"]["4"], ["label_05"])
            self.assertEqual(plan["bulk_batches"]["gpu_map"]["5"], [])
            self.assertEqual(plan["bulk_batches"]["gpu_map"]["6"], [])

            first = plan["labels"][0]
            self.assertEqual(first["label"], "label_00")
            self.assertEqual(first["width"], [24, 64, 128])
            self.assertEqual(first["gpu_id"], 3)
            self.assertEqual(first["master_port"], 29730)
            self.assertEqual(
                [item["master_port"] for item in plan["labels"]],
                [29730, 29731, 29732, 29733, 29734, 29735],
            )
            self.assertIn("stage2_original60_fp16_train_launcher.py", " ".join(first["train_command"]))
            self.assertIn("stage2_original60_fp16_eval_when_ready.py", " ".join(first["eval_watcher_command"]))


if __name__ == "__main__":
    unittest.main()
