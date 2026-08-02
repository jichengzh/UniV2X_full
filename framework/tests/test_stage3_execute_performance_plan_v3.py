from __future__ import annotations

import importlib.util
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "stage3_execute_performance_plan_v3.py"
SPEC = importlib.util.spec_from_file_location("stage3_execute_performance_plan_v3", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)


def _load_module() -> None:
    SPEC.loader.exec_module(MODULE)


def _job(
    *,
    job_id: str = "codriving|16x32x64|trt_int8",
    runner_key: str = "trt_int8",
    command: list[str] | None = None,
    calibration_root: str = "/tmp/calibration_source",
    assigned_gpu: int = 5,
    out_dir: str = "/tmp/out",
) -> dict:
    return {
        "job_id": job_id,
        "runner_key": runner_key,
        "command": command
        or [
            "python3",
            "framework/trt_baseline/trt_profile_v1.py",
            "--precision",
            "int8",
            "--gpu",
            str(assigned_gpu),
            "--artifact-dir",
            f"{out_dir}/artifacts",
            "--out",
            f"{out_dir}/trt_profile_result.json",
            "--calib-dir",
            calibration_root,
        ],
        "calibration_root": calibration_root,
        "assigned_gpu": assigned_gpu,
        "max_attempts": 2,
    }


class Stage3ExecutePerformancePlanV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _load_module()

    def test_prepare_job_command_rewrites_python_and_codriving_trt_int8_calibration_dir(self) -> None:
        job = _job(
            calibration_root="/exdata/jichengzhi/V2Xverse_pyramid/output/codriving_v2_gold_ap_20260709/16x32x64/calibration_source"
        )

        prepared = MODULE.prepare_job_command(job)

        self.assertEqual(prepared[0], str(MODULE.TRT_PYTHON))
        self.assertEqual(
            prepared[prepared.index("--calib-dir") + 1],
            "/exdata/jichengzhi/V2Xverse_pyramid/output/codriving_v2_gold_ap_20260709/16x32x64/trt_calibration_npy",
        )

    def test_prepare_job_command_applies_runtime_gpu_override(self) -> None:
        job = {**_job(assigned_gpu=5), "runtime_gpu": 7}

        prepared = MODULE.prepare_job_command(job)

        self.assertEqual(prepared[prepared.index("--gpu") + 1], "7")

    def test_prepare_job_command_rewrites_pyramid_trt_int8_calibration_dir(self) -> None:
        job = _job(
            job_id="pyramid|016x016x016|trt_int8",
            calibration_root="/home/jichengzhi/V2X/results/stage3_gold96_v3_20260711/pyramid_calibration/016x016x016",
        )

        prepared = MODULE.prepare_job_command(job)

        self.assertEqual(
            prepared[prepared.index("--calib-dir") + 1],
            "/home/jichengzhi/V2X/results/stage3_gold96_v3_20260711/pyramid_calibration/016x016x016/trt_npy",
        )

    def test_validate_result_payload_accepts_real_tvm_shape_and_ignores_wrong_expected_result_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            actual = root / "route_b_fp16_auto_result.json"
            actual.write_text(
                json.dumps(
                    {
                        "status": "success",
                        "build_success": True,
                        "latency": {"latency_ms_p50": 12.5},
                        "energy": {"status": "success", "energy_j": 1.75},
                        "correctness_vs_default_fp16": [{"max_abs_diff": 0.0}],
                    }
                ),
                encoding="utf-8",
            )
            wrong = root / "result.json"

            verdict = MODULE.validate_job_result(
                {
                    "job_id": "codriving|16x32x64|tvm_fp16",
                    "runner_key": "tvm_fp16",
                    "expected_result_json": str(wrong),
                    "command": ["python3", "scripts/stage2_route_b_fp16_auto_runner.py"],
                },
                returncode=0,
                work_dir=root,
            )

        self.assertTrue(verdict["success"])
        self.assertEqual(verdict["result_json"], str(actual))
        self.assertRegex(verdict["result_sha256"], r"^[0-9a-f]{64}$")

    def test_candidate_result_paths_include_absolute_out_file(self) -> None:
        job = _job(out_dir="/tmp/absolute-result")
        paths = MODULE._candidate_result_paths(job, Path("/tmp/attempt"))

        self.assertIn(Path("/tmp/absolute-result/trt_profile_result.json"), paths)

    def test_candidate_result_paths_include_tvm_label_subdirectory(self) -> None:
        job = {
            "runner_key": "tvm_int8",
            "command": [
                "python3",
                "scripts/stage2_route_b_int8_auto_decomp.py",
                "--label",
                "codriving_16x32x64",
                "--out-dir",
                "/tmp/tvm-output",
            ],
        }

        paths = MODULE._candidate_result_paths(job, Path("/tmp/attempt"))

        self.assertIn(
            Path("/tmp/tvm-output/codriving_16x32x64/route_b_int8_auto_decomp_result.json"),
            paths,
        )

    def test_tvm_subprocess_env_prepends_required_runtime_libraries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            nvlibs_file = Path(tmpdir) / "nvlibs.path"
            nvlibs_file.write_text("/tmp/nvlibs\n", encoding="utf-8")
            with mock.patch.object(MODULE, "TVM_NVLIBS_FILE", nvlibs_file):
                env = MODULE.subprocess_env_for_job({"runner_key": "tvm_int8"})

        self.assertTrue(env["LD_LIBRARY_PATH"].startswith(str(MODULE.TVM_CUDA_RUNTIME_LIB)))
        self.assertIn("/tmp/nvlibs", env["LD_LIBRARY_PATH"])

    def test_validate_result_payload_rejects_nonfinite_energy_or_missing_correctness(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bad = root / "trt_profile_result.json"
            bad.write_text(
                json.dumps(
                    {
                        "build_success": True,
                        "lat_p50_ms": 9.0,
                        "energy_j": "nan",
                        "numerical_finite": True,
                    }
                ),
                encoding="utf-8",
            )

            verdict = MODULE.validate_job_result(
                {
                    "job_id": "codriving|16x32x64|trt_int8",
                    "runner_key": "trt_int8",
                    "command": ["python3", "framework/trt_baseline/trt_profile_v1.py"],
                },
                returncode=0,
                work_dir=root,
            )

        self.assertFalse(verdict["success"])
        self.assertIn("energy", verdict["failure_reasons"][0])

    def test_extract_energy_accepts_both_real_tvm_energy_schemas(self) -> None:
        self.assertEqual(MODULE._extract_energy({"energy": {"joules_per_inference": 0.21}}), 0.21)
        self.assertEqual(MODULE._extract_energy({"energy": {"energy_J": 0.24}}), 0.24)

    def test_aggregate_resume_state_skips_success_and_reschedules_single_failure(self) -> None:
        jobs = [
            {"job_id": "a", "max_attempts": 2},
            {"job_id": "b", "max_attempts": 2},
            {"job_id": "c", "max_attempts": 2},
        ]
        state_rows = [
            {"job_id": "a", "attempt": 1, "status": "success"},
            {"job_id": "b", "attempt": 1, "status": "failed"},
        ]

        aggregated = MODULE.aggregate_state_rows(jobs, state_rows)

        self.assertEqual(aggregated["by_job"]["a"]["terminal_status"], "success")
        self.assertEqual(aggregated["by_job"]["b"]["terminal_status"], "pending")
        self.assertEqual(aggregated["ready_job_ids"], ["b", "c"])

    def test_run_job_retries_once_then_marks_confirmed_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            state_path = root / "state.jsonl"
            job = _job(out_dir=str(root / "artifacts"))
            attempt_dir = root / MODULE.job_slug(job["job_id"])
            attempt_dir.mkdir(parents=True, exist_ok=True)

            completed = mock.Mock(returncode=1, stdout="nope", stderr="fail")
            with mock.patch.object(MODULE, "prepare_job_command", return_value=job["command"]), mock.patch.object(
                MODULE.subprocess, "run", return_value=completed
            ), mock.patch.object(MODULE, "preflight_gpu_check", return_value=None):
                final_row = MODULE.run_single_job(
                    job=job,
                    state_jsonl=state_path,
                    attempt_root=root,
                )

            self.assertEqual(final_row["status"], "confirmed_failure")
            rows = MODULE.load_state_rows(state_path)
            self.assertEqual([row["status"] for row in rows], ["failed", "failed", "confirmed_failure"])

    def test_run_job_executes_relative_runner_from_repo_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            state_path = root / "state.jsonl"
            job = _job(out_dir=str(root / "artifacts"))
            completed = mock.Mock(returncode=1, stdout="", stderr="fail")
            with mock.patch.object(MODULE, "prepare_job_command", return_value=job["command"]), mock.patch.object(
                MODULE.subprocess, "run", return_value=completed
            ) as run_mock, mock.patch.object(MODULE, "preflight_gpu_check", return_value=None):
                MODULE.run_single_job(job=job, state_jsonl=state_path, attempt_root=root)

            self.assertEqual(Path(run_mock.call_args.kwargs["cwd"]), MODULE.REPO_ROOT)

    def test_execute_jobs_respects_gpu_mutex_even_with_multiple_workers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            state_path = root / "state.jsonl"
            jobs = [
                _job(job_id="job-a", assigned_gpu=5, out_dir=str(root / "a")),
                _job(job_id="job-b", assigned_gpu=5, out_dir=str(root / "b")),
            ]
            interval_lock = threading.Lock()
            intervals: list[tuple[float, float]] = []

            def fake_run(*args, **kwargs):
                started = time.monotonic()
                time.sleep(0.05)
                finished = time.monotonic()
                with interval_lock:
                    intervals.append((started, finished))
                return mock.Mock(returncode=1, stdout="", stderr="fail")

            with mock.patch.object(MODULE, "preflight_gpu_check", return_value=None), mock.patch.object(
                MODULE, "prepare_job_command", side_effect=lambda job: job["command"]
            ), mock.patch.object(MODULE.subprocess, "run", side_effect=fake_run):
                MODULE.execute_jobs(
                    jobs=jobs,
                    state_jsonl=state_path,
                    gpus=[5],
                    max_workers=2,
                    attempt_root=root,
                )

            self.assertEqual(len(intervals), 4)
            self.assertGreaterEqual(intervals[2][0], intervals[1][1] - 1e-6)

    def test_execute_jobs_remaps_planned_gpu_to_requested_pool(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            observed: list[int] = []

            def fake_run_single_job(*, job, state_jsonl, attempt_root):
                observed.append(job["runtime_gpu"])
                return {"job_id": job["job_id"], "status": "success"}

            with mock.patch.object(MODULE, "preflight_gpu_check", return_value=None), mock.patch.object(
                MODULE, "run_single_job", side_effect=fake_run_single_job
            ):
                MODULE.execute_jobs(
                    jobs=[_job(job_id="job-a", assigned_gpu=5), _job(job_id="job-b", assigned_gpu=6)],
                    state_jsonl=root / "state.jsonl",
                    gpus=[7],
                    max_workers=1,
                    attempt_root=root,
                )

        self.assertEqual(observed, [7, 7])


if __name__ == "__main__":
    unittest.main()
