from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import stage2_original60_fp16_supervisor as supervisor


class Stage2Original60Fp16SupervisorTest(unittest.TestCase):
    def _write_plan(self, output_root: Path) -> Path:
        plan = {
            "schema": "stage2_original60_fp16_ap_launch_plan_v1",
            "summary": {"gpus": [3, 4, 5, 6]},
            "labels": [
                {"label": "frontier_01", "gpu_id": 3, "phase": "all"},
                {"label": "frontier_02", "gpu_id": 4, "phase": "all"},
                {"label": "frontier_03", "gpu_id": 5, "phase": "all"},
                {"label": "frontier_04", "gpu_id": 6, "phase": "all"},
                {"label": "frontier_05", "gpu_id": 3, "phase": "all"},
            ],
        }
        plan_path = output_root / "exports/fp16_original60_ap_launch_plan_latest.json"
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return plan_path

    def test_supervisor_dry_run_builds_lane_launches_for_available_gpus(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan_path = self._write_plan(root)
            gpu_snapshot = root / "gpu_snapshot.json"
            compute_apps = root / "compute_apps.json"
            gpu_snapshot.write_text(
                json.dumps(
                    {
                        "gpus": [
                            {"index": 3, "uuid": "gpu-3"},
                            {"index": 4, "uuid": "gpu-4"},
                            {"index": 5, "uuid": "gpu-5"},
                            {"index": 6, "uuid": "gpu-6"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            compute_apps.write_text(
                json.dumps({"compute_apps": [{"gpu_index": 4, "pid": 9001, "process_name": "python"}]}),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/stage2_original60_fp16_supervisor.py"),
                    "--launch-plan",
                    str(plan_path),
                    "--output-root",
                    str(root),
                    "--gpu-snapshot-json",
                    str(gpu_snapshot),
                    "--compute-apps-json",
                    str(compute_apps),
                    "--phase",
                    "all",
                    "--watcher-poll-seconds",
                    "60",
                    "--dry-run",
                ],
                cwd=ROOT,
                env={"PYTHONPATH": str(ROOT)},
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["busy_gpus"], [4])
            self.assertEqual(sorted(payload["available_gpus"]), [3, 5, 6])
            self.assertEqual(payload["lane_launches"][0]["gpu_id"], 3)
            self.assertEqual(payload["lane_launches"][0]["labels"], ["frontier_01", "frontier_05"])
            self.assertEqual(payload["lane_launches"][1]["gpu_id"], 5)
            self.assertEqual(payload["lane_launches"][1]["labels"], ["frontier_03"])
            self.assertEqual(payload["lane_launches"][2]["gpu_id"], 6)
            self.assertEqual(payload["lane_launches"][2]["labels"], ["frontier_04"])
            self.assertEqual(payload["deferred_labels"], ["frontier_02"])
            first_command = payload["lane_launches"][0]["command"]
            self.assertIn("--watcher-poll-seconds", first_command)
            watcher_poll_index = first_command.index("--watcher-poll-seconds")
            self.assertEqual(first_command[watcher_poll_index + 1], "60")
            state = json.loads((root / "exports/fp16_original60_supervisor_state_latest.json").read_text(encoding="utf-8"))
            self.assertEqual(state["status"], "dry_run")

    def test_build_lane_launches_skips_gpu_with_active_lane_runner(self) -> None:
        plan = {
            "labels": [
                {"label": "frontier_01", "gpu_id": 3, "phase": "all"},
                {"label": "frontier_02", "gpu_id": 4, "phase": "all"},
                {"label": "frontier_03", "gpu_id": 5, "phase": "all"},
            ]
        }
        launches, deferred = supervisor.build_lane_launches(
            plan=plan,
            phase="all",
            available_gpus=[3, 4, 5],
            active_lane_gpus=[4],
            output_root=Path("/tmp/out"),
            env_python="/usr/bin/python3",
            master_port_base=29730,
            poll_seconds=900,
            watcher_poll_seconds=60,
        )

        self.assertEqual([item["gpu_id"] for item in launches], [3, 5])
        self.assertEqual(deferred, ["frontier_02"])


if __name__ == "__main__":
    unittest.main()
