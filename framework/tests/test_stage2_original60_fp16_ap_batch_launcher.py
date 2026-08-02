from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class Stage2Original60Fp16ApBatchLauncherTest(unittest.TestCase):
    def _write_plan(self, root: Path) -> Path:
        plan = {
            "schema": "stage2_original60_fp16_ap_launch_plan_v1",
            "created_at": "2026-06-28T00:00:00Z",
            "summary": {
                "fp16_gap_labels": 6,
                "pilot_label_count": 4,
                "bulk_label_count": 2,
                "gpus": [3, 4, 5, 6],
            },
            "pilot_batch": {
                "labels": ["frontier_01", "frontier_02", "frontier_03", "frontier_04"],
                "gpu_map": {"3": ["frontier_01"], "4": ["frontier_02"], "5": ["frontier_03"], "6": ["frontier_04"]},
            },
            "bulk_batches": {
                "labels": ["frontier_05", "frontier_06"],
                "gpu_map": {"3": ["frontier_05"], "4": ["frontier_06"], "5": [], "6": []},
            },
            "labels": [
                {
                    "label": "frontier_01",
                    "width": [24, 64, 128],
                    "gpu_id": 3,
                    "master_port": 29733,
                    "phase": "pilot",
                    "ckpt_dir": "/tmp/ckpts/frontier_01",
                    "raw_dir": "/tmp/raw/frontier_01",
                    "train_command": ["python", "train", "--label", "frontier_01"],
                    "eval_watcher_command": ["python", "watch", "--label", "frontier_01"],
                },
                {
                    "label": "frontier_02",
                    "width": [40, 64, 128],
                    "gpu_id": 4,
                    "master_port": 29734,
                    "phase": "pilot",
                    "ckpt_dir": "/tmp/ckpts/frontier_02",
                    "raw_dir": "/tmp/raw/frontier_02",
                    "train_command": ["python", "train", "--label", "frontier_02"],
                    "eval_watcher_command": ["python", "watch", "--label", "frontier_02"],
                },
                {
                    "label": "frontier_03",
                    "width": [32, 48, 128],
                    "gpu_id": 5,
                    "master_port": 29735,
                    "phase": "pilot",
                    "ckpt_dir": "/tmp/ckpts/frontier_03",
                    "raw_dir": "/tmp/raw/frontier_03",
                    "train_command": ["python", "train", "--label", "frontier_03"],
                    "eval_watcher_command": ["python", "watch", "--label", "frontier_03"],
                },
                {
                    "label": "frontier_04",
                    "width": [32, 80, 128],
                    "gpu_id": 6,
                    "master_port": 29736,
                    "phase": "pilot",
                    "ckpt_dir": "/tmp/ckpts/frontier_04",
                    "raw_dir": "/tmp/raw/frontier_04",
                    "train_command": ["python", "train", "--label", "frontier_04"],
                    "eval_watcher_command": ["python", "watch", "--label", "frontier_04"],
                },
                {
                    "label": "frontier_05",
                    "width": [32, 64, 96],
                    "gpu_id": 3,
                    "master_port": 29733,
                    "phase": "bulk",
                    "ckpt_dir": "/tmp/ckpts/frontier_05",
                    "raw_dir": "/tmp/raw/frontier_05",
                    "train_command": ["python", "train", "--label", "frontier_05"],
                    "eval_watcher_command": ["python", "watch", "--label", "frontier_05"],
                },
                {
                    "label": "frontier_06",
                    "width": [32, 64, 160],
                    "gpu_id": 4,
                    "master_port": 29734,
                    "phase": "bulk",
                    "ckpt_dir": "/tmp/ckpts/frontier_06",
                    "raw_dir": "/tmp/raw/frontier_06",
                    "train_command": ["python", "train", "--label", "frontier_06"],
                    "eval_watcher_command": ["python", "watch", "--label", "frontier_06"],
                },
            ],
        }
        path = root / "exports/fp16_original60_ap_launch_plan_latest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    def test_batch_launcher_dry_run_selects_only_pilot_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            plan_path = self._write_plan(tmp_path)
            gpu_snapshot = tmp_path / "gpu_snapshot.json"
            compute_apps = tmp_path / "compute_apps.json"
            gpu_snapshot.write_text(json.dumps({"gpus": [{"index": 3}, {"index": 4}, {"index": 5}, {"index": 6}]}), encoding="utf-8")
            compute_apps.write_text(json.dumps({"compute_apps": []}), encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/stage2_original60_fp16_ap_batch_launcher.py"),
                    "--launch-plan",
                    str(plan_path),
                    "--phase",
                    "pilot",
                    "--gpu-snapshot-json",
                    str(gpu_snapshot),
                    "--compute-apps-json",
                    str(compute_apps),
                    "--dry-run",
                ],
                cwd=ROOT,
                env={"PYTHONPATH": str(ROOT)},
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["selected_labels"], ["frontier_01", "frontier_02", "frontier_03", "frontier_04"])
            self.assertEqual(payload["deferred_labels"], [])

    def test_batch_launcher_defers_labels_on_busy_gpu_without_reassigning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            plan_path = self._write_plan(tmp_path)
            gpu_snapshot = tmp_path / "gpu_snapshot.json"
            compute_apps = tmp_path / "compute_apps.json"
            gpu_snapshot.write_text(json.dumps({"gpus": [{"index": 3}, {"index": 4}, {"index": 5}, {"index": 6}]}), encoding="utf-8")
            compute_apps.write_text(
                json.dumps(
                    {
                        "compute_apps": [
                            {"gpu_index": 4, "pid": 999999, "process_name": "python", "used_memory": "100 MiB"}
                        ]
                    }
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/stage2_original60_fp16_ap_batch_launcher.py"),
                    "--launch-plan",
                    str(plan_path),
                    "--phase",
                    "bulk",
                    "--gpu-snapshot-json",
                    str(gpu_snapshot),
                    "--compute-apps-json",
                    str(compute_apps),
                    "--dry-run",
                ],
                cwd=ROOT,
                env={"PYTHONPATH": str(ROOT)},
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["selected_labels"], ["frontier_05"])
            self.assertEqual(payload["deferred_labels"], ["frontier_06"])
            self.assertEqual(payload["busy_gpus"], [4])


if __name__ == "__main__":
    unittest.main()
