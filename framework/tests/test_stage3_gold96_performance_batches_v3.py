from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from framework.stage2.canonical_search_v3 import build_capability_profile


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "stage3_gold96_performance_batches_v3.py"
PLANNER_PATH = REPO_ROOT / "scripts" / "stage2_gold_coldstart96_plan_v3.py"

SPEC = importlib.util.spec_from_file_location("stage3_gold96_performance_batches_v3", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
PLANNER_SPEC = importlib.util.spec_from_file_location("stage2_gold_coldstart96_plan_v3", PLANNER_PATH)
assert PLANNER_SPEC and PLANNER_SPEC.loader
PLANNER = importlib.util.module_from_spec(PLANNER_SPEC)


def _load_module() -> None:
    SPEC.loader.exec_module(MODULE)


def _load_planner() -> None:
    PLANNER_SPEC.loader.exec_module(PLANNER)


def _profiles() -> list[dict]:
    return [
        build_capability_profile(
            capability_profile_id="h800-tvm-auto-v3",
            hardware_target="h800",
            compiler_fingerprint="a" * 64,
            dispatch_key="tvm_auto",
            features={"supports_int8_tensorcore": 1.0},
        ),
        build_capability_profile(
            capability_profile_id="h800-trt-v3",
            hardware_target="h800",
            compiler_fingerprint="b" * 64,
            dispatch_key="trt_engine",
            features={"supports_int8_tensorcore": 1.0},
        ),
    ]


def _manifest() -> dict:
    return PLANNER.build_gold_plan(_profiles())["manifest"]


class Stage3Gold96PerformanceBatchesV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _load_module()
        _load_planner()

    def test_build_batches_cover_88_non_pilot_rows_in_six_complete_group_shards(self) -> None:
        plan = MODULE.build_batch_plan(
            manifest=_manifest(),
            batch_index=1,
            remote_artifact_root="/remote/stage3",
            gpus=[0, 1],
        )
        batches = MODULE.build_all_batches(_manifest(), remote_artifact_root="/remote/stage3", gpus=[0, 1])

        self.assertEqual(plan["schema_version"], "stage3_gold96_performance_batch_plan_v3")
        self.assertEqual(len(batches), 6)
        self.assertEqual([len(batch["group_ids"]) for batch in batches], [4, 4, 4, 4, 3, 3])
        all_group_ids = [group_id for batch in batches for group_id in batch["group_ids"]]
        self.assertEqual(len(all_group_ids), 22)
        self.assertEqual(len(set(all_group_ids)), 22)
        self.assertFalse(set(_manifest()["pilot_group_ids"]) & set(all_group_ids))
        self.assertEqual(sum(batch["manifest_row_count"] for batch in batches), 88)
        self.assertTrue(all(batch["manifest_row_count"] == len(batch["manifest_rows"]) for batch in batches))
        self.assertTrue(all(len({row["group_id"] for row in batch["manifest_rows"]}) == len(batch["group_ids"]) for batch in batches))
        self.assertTrue(
            all(
                len([row for row in batch["manifest_rows"] if row["group_id"] == group_id]) == 4
                for batch in batches
                for group_id in batch["group_ids"]
            )
        )

    def test_jobs_expand_each_group_to_four_runners_with_correct_input_paths(self) -> None:
        batches = MODULE.build_all_batches(_manifest(), remote_artifact_root="/remote/stage3", gpus=[3, 5])
        jobs = [job for batch in batches for job in batch["jobs"]]

        self.assertEqual(len(jobs), 88)
        pyramid_fp16 = next(job for job in jobs if job["job_id"] == "pyramid|16x16x16|tvm_fp16")
        self.assertEqual(
            pyramid_fp16["onnx_path"],
            "/home/jichengzhi/V2X/results/stage3_gold96_v3_20260711/pyramid_sources/016x016x016/pyramid_016x016x016_multiscale.onnx",
        )
        self.assertEqual(
            pyramid_fp16["calibration_root"],
            "/home/jichengzhi/V2X/results/stage3_gold96_v3_20260711/pyramid_calibration/016x016x016",
        )
        codriving_int8 = next(job for job in jobs if job["job_id"] == "codriving|16x32x64|trt_int8")
        self.assertEqual(
            codriving_int8["onnx_path"],
            "/exdata/jichengzhi/V2Xverse_pyramid/output/codriving_v2_gold_ap_20260709/16x32x64/resnet_multiscale_16x32x64_final_fp32.onnx",
        )
        self.assertEqual(
            codriving_int8["calibration_root"],
            "/exdata/jichengzhi/V2Xverse_pyramid/output/codriving_v2_gold_ap_20260709/16x32x64/calibration_source",
        )

    def test_commands_use_only_approved_runner_entrypoints_and_no_hand_rewrite_or_mixed(self) -> None:
        batches = MODULE.build_all_batches(_manifest(), remote_artifact_root="/remote/stage3", gpus=[7])
        jobs = [job for batch in batches for job in batch["jobs"]]

        self.assertEqual({job["runner_key"] for job in jobs}, {"tvm_fp16", "tvm_int8", "trt_fp16", "trt_int8"})
        for job in jobs:
            cmd = " ".join(job["command"])
            self.assertNotIn("hand-rewrite", cmd)
            self.assertNotIn("hand_rewrite", cmd)
            self.assertNotIn("mixed", cmd.lower())
            if job["runner_key"] == "tvm_fp16":
                self.assertIn("scripts/stage2_route_b_fp16_auto_runner.py", cmd)
                self.assertIn("--fix none", cmd)
                self.assertIn("--max-trials 64", cmd)
                self.assertIn("--measure-energy", cmd)
            elif job["runner_key"] == "tvm_int8":
                self.assertIn("scripts/stage2_route_b_int8_auto_decomp.py", cmd)
                self.assertIn("--measure-energy", cmd)
            else:
                self.assertIn("framework/trt_baseline/trt_profile_v1.py", cmd)
                self.assertIn("--energy-secs 5.0", cmd)
            if job["runner_key"] == "trt_int8":
                self.assertIn("--calib-dir", cmd)

    def test_jobs_carry_runtime_gpu_pool_without_writing_gpu_into_manifest_rows(self) -> None:
        batches = MODULE.build_all_batches(_manifest(), remote_artifact_root="/remote/stage3", gpus=[2, 4, 6])
        jobs = [job for batch in batches for job in batch["jobs"]]

        self.assertEqual({job["gpu_pool"] for job in jobs}, {"2,4,6"})
        self.assertEqual({job["assigned_gpu"] for job in jobs[:6]}, {2, 4, 6})
        self.assertTrue(all("assigned_gpu" not in row for batch in batches for row in batch["manifest_rows"]))

    def test_local_state_aggregator_retries_once_then_confirms_failure_without_old_measurement_fill(self) -> None:
        batch = MODULE.build_batch_plan(_manifest(), batch_index=1, remote_artifact_root="/remote/stage3", gpus=[0])
        job_id = batch["jobs"][0]["job_id"]

        initial = MODULE.aggregate_local_state(batch["jobs"], state_rows=[])
        self.assertEqual(initial["by_job"][job_id]["terminal_status"], "pending")
        self.assertEqual(initial["ready_job_ids"], [job_id] + [job["job_id"] for job in batch["jobs"][1:]])

        after_fail_1 = MODULE.aggregate_local_state(
            batch["jobs"],
            state_rows=[{"job_id": job_id, "attempt": 1, "status": "failed"}],
        )
        self.assertEqual(after_fail_1["by_job"][job_id]["attempts_used"], 1)
        self.assertEqual(after_fail_1["by_job"][job_id]["terminal_status"], "pending")
        self.assertIn(job_id, after_fail_1["ready_job_ids"])

        after_fail_2 = MODULE.aggregate_local_state(
            batch["jobs"],
            state_rows=[
                {"job_id": job_id, "attempt": 1, "status": "failed"},
                {"job_id": job_id, "attempt": 2, "status": "failed"},
            ],
        )
        self.assertEqual(after_fail_2["by_job"][job_id]["attempts_used"], 2)
        self.assertEqual(after_fail_2["by_job"][job_id]["terminal_status"], "confirmed_failure")
        self.assertNotIn(job_id, after_fail_2["ready_job_ids"])

        measured_only = MODULE.aggregate_local_state(
            batch["jobs"],
            state_rows=[{"job_id": job_id, "attempt": 0, "status": "measured"}],
        )
        self.assertEqual(measured_only["by_job"][job_id]["terminal_status"], "pending")
        self.assertIn(job_id, measured_only["ready_job_ids"])

        success = MODULE.aggregate_local_state(
            batch["jobs"],
            state_rows=[{"job_id": job_id, "attempt": 1, "status": "success"}],
        )
        self.assertEqual(success["by_job"][job_id]["terminal_status"], "success")
        self.assertNotIn(job_id, success["ready_job_ids"])

    def test_cli_writes_plan_json_and_jsonl_for_selected_batch_in_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "manifest.json"
            out_dir = Path(tmp) / "out"
            manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--manifest-json",
                    str(manifest_path),
                    "--batch-index",
                    "6",
                    "--remote-artifact-root",
                    "/remote/stage3",
                    "--output-dir",
                    str(out_dir),
                    "--gpus",
                    "1,3",
                    "--dry-run",
                ],
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(completed.stdout)
            plan_json = Path(payload["plan_json"])
            jobs_jsonl = Path(payload["jobs_jsonl"])
            self.assertTrue(plan_json.exists())
            self.assertTrue(jobs_jsonl.exists())
            plan = json.loads(plan_json.read_text(encoding="utf-8"))
            jobs = [json.loads(line) for line in jobs_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(plan["batch_index"], 6)
            self.assertEqual(plan["group_count"], 3)
            self.assertEqual(len(jobs), 12)
            self.assertTrue(payload["dry_run"])


if __name__ == "__main__":
    unittest.main()
