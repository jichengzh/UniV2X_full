from __future__ import annotations

import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "stage3_execute_ap_plan_v3.py"
SPEC = importlib.util.spec_from_file_location("stage3_execute_ap_plan_v3", SCRIPT_PATH)
assert SPEC and SPEC.loader
EXECUTOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXECUTOR)


def _job(model: str = "pyramid") -> dict:
    return {
        "manifest_job_id": f"{model}-fp16",
        "model": model,
        "runner_key": f"{model}_tvm_fp16_bridge",
        "compiled_artifact": f"/artifacts/{model}.so",
        "compiled_artifact_digest": "a" * 64,
        "ap_terminal": "ready",
        "sanity_command": ["python3", "runner.py", "--report-json", "reports/sanity.json"],
        "full_command": ["python3", "runner.py", "--out-json", "reports/full.json"],
    }


def _codriving_report(*, processed: int = 16, stage: str = "sanity") -> dict:
    return {
        "status": "success",
        "processed_samples": processed,
        "engine_samples": processed,
        "fallback_samples": 0,
        "failed_samples": 0,
        "gates": {"sanity_16" if stage == "sanity" else "full_1789": True},
        "ap": {"ap30": 0.5, "ap50": 0.4, "ap70": 0.3},
    }


def _pyramid_report(*, processed: int = 16, claim: bool = False) -> dict:
    return {
        "status": "success",
        "processed_samples": processed,
        "fallback_samples": 0,
        "failed_samples": 0,
        "ap30": 0.8,
        "ap50": 0.7,
        "ap70": 0.6,
        "engine_ap_claim": claim,
        "output_error_summary": {"max_abs_err": 0.25, "mean_abs_err": 0.02},
    }


class Stage3ExecuteApPlanV3Tests(unittest.TestCase):
    def test_first_idle_gpu_selects_available_candidate(self) -> None:
        states = {
            "4": (False, "busy"),
            "5": (True, "idle"),
            "6": (True, "idle"),
        }

        gpu, evidence = EXECUTOR.first_idle_gpu(["4", "5", "6"], probe=lambda item: states[item])

        self.assertEqual(gpu, "5")
        self.assertEqual(evidence, "idle")

    def test_report_path_accepts_report_json_and_out_json(self) -> None:
        self.assertEqual(
            EXECUTOR.report_path_from_command(["python", "x.py", "--report-json", "a.json"]),
            Path("a.json"),
        )
        self.assertEqual(
            EXECUTOR.report_path_from_command(["python", "x.py", "--out-json=b.json"]),
            Path("b.json"),
        )
        self.assertEqual(
            EXECUTOR.report_path_from_command(["python", "x.py", "--export-report-json", "c.json"]),
            Path("c.json"),
        )

    def test_report_path_rejects_command_without_supported_output(self) -> None:
        with self.assertRaisesRegex(ValueError, "--export-report-json"):
            EXECUTOR.report_path_from_command(["python", "x.py"])

    def test_pyramid_sanity_requires_clean_finite_report(self) -> None:
        ok, reason, ap = EXECUTOR.validate_report("pyramid", "sanity", _pyramid_report())
        self.assertTrue(ok, reason)
        self.assertEqual(ap, {"ap30": 0.8, "ap50": 0.7, "ap70": 0.6})

        cases = [
            ({"status": "failed"}, "status"),
            ({"processed_samples": 15}, "processed_samples"),
            ({"fallback_samples": 1}, "fallback_samples"),
            ({"failed_samples": 1}, "failed_samples"),
            ({"ap50": math.nan}, "finite"),
            ({"output_error_summary": {"max_abs_err": math.inf}}, "finite"),
        ]
        for changes, expected in cases:
            with self.subTest(changes=changes):
                report = _pyramid_report()
                report.update(changes)
                valid, failure, _ = EXECUTOR.validate_report("pyramid", "sanity", report)
                self.assertFalse(valid)
                self.assertIn(expected, failure)

    def test_pyramid_full_requires_engine_ap_claim(self) -> None:
        valid, failure, _ = EXECUTOR.validate_report("pyramid", "full", _pyramid_report(processed=1789))
        self.assertFalse(valid)
        self.assertIn("engine_ap_claim", failure)
        valid, _, _ = EXECUTOR.validate_report("pyramid", "full", _pyramid_report(processed=1789, claim=True))
        self.assertTrue(valid)

    def test_pyramid_tvm_fp16_full_accepts_measured_smoke_gated_report(self) -> None:
        report = {
            "status": "success",
            "processed_samples": 1789,
            "fallback_samples": 0,
            "failed_samples": 0,
            "ap_measured": True,
            "smoke_gate_passed": True,
            "ap30": 0.8,
            "ap50": 0.7,
            "ap70": 0.5,
        }

        valid, reason, ap = EXECUTOR.validate_report(
            "pyramid", "full", report, runner_key="pyramid_tvm_fp16_bridge"
        )

        self.assertTrue(valid)
        self.assertIsNone(reason)
        self.assertEqual(ap["ap70"], 0.5)

    def test_pyramid_tvm_int8_full_accepts_repaired_numeric_gate_report(self) -> None:
        report = {
            "status": "success",
            "processed_samples": 1789,
            "fallback_samples": 0,
            "failed_samples": 0,
            "ap_measured": True,
            "ap_row_allowed": True,
            "gates": {"sanity_16": True, "full_1789": True},
            "ap30": 0.8,
            "ap50": 0.7,
            "ap70": 0.5,
        }

        valid, reason, ap = EXECUTOR.validate_report(
            "pyramid", "full", report, runner_key="pyramid_tvm_int8_numeric_gate"
        )

        self.assertTrue(valid)
        self.assertIsNone(reason)
        self.assertEqual(ap["ap70"], 0.5)

    def test_codriving_requires_complete_clean_stage_report(self) -> None:
        sanity = _codriving_report()
        full = _codriving_report(processed=1789, stage="full")
        self.assertTrue(EXECUTOR.validate_report("codriving", "sanity", sanity)[0])
        self.assertTrue(EXECUTOR.validate_report("codriving", "full", full)[0])

    def test_codriving_tvm_int8_accepts_explicit_numerical_sanity_status(self) -> None:
        report = {
            "status": "numerical_sanity_passed",
            "processed_samples": 16,
            "engine_samples": 16,
            "fallback_samples": 0,
            "failed_samples": 0,
            "gates": {"sanity_16": True, "full_1789": False},
        }

        valid, reason, _ = EXECUTOR.validate_report(
            "codriving", "sanity", report, runner_key="codriving_tvm_int8_numeric_gate"
        )

        self.assertTrue(valid)
        self.assertIsNone(reason)
        cases = [
            ({"status": "failed"}, "status"),
            ({"processed_samples": 15, "engine_samples": 15}, "processed_samples"),
            ({"engine_samples": 15}, "engine_samples"),
            ({"fallback_samples": 1}, "fallback_samples"),
            ({"failed_samples": 1}, "failed_samples"),
            ({"ap": {"ap30": math.nan, "ap50": 0.4, "ap70": 0.3}}, "finite"),
            ({"ap": {"ap30": "nan"}}, "finite"),
            ({"gates": {"sanity_16": False}}, "gates.sanity_16"),
        ]
        for changes, expected in cases:
            with self.subTest(changes=changes):
                report = {**_codriving_report(), **changes}
                valid, failure, _ = EXECUTOR.validate_report("codriving", "sanity", report)
                self.assertFalse(valid)
                self.assertIn(expected, failure)

    def test_codriving_full_requires_all_three_finite_ap_values(self) -> None:
        report = _codriving_report(processed=1789, stage="full")
        del report["ap"]["ap70"]

        valid, failure, _ = EXECUTOR.validate_report("codriving", "full", report)

        self.assertFalse(valid)
        self.assertIn("ap30_ap50_ap70", failure)

    def test_full_selects_only_ready_jobs_with_prior_sanity_success(self) -> None:
        jobs = [_job(), _job("codriving"), {**_job(), "manifest_job_id": "blocked", "ap_terminal": "blocked"}]
        state = [{
            "job_id": "pyramid-fp16",
            "stage": "sanity",
            "status": "success",
            "plan_fingerprint": EXECUTOR.plan_fingerprint(jobs[0], "sanity"),
        }]
        selected, skipped = EXECUTOR.select_jobs(jobs, stage="full", state_rows=state)
        self.assertEqual([row["manifest_job_id"] for row in selected], ["pyramid-fp16"])
        self.assertEqual(skipped["codriving-fp16"], "sanity_success_required")
        self.assertNotIn("blocked", skipped)

    def test_full_command_injects_sha_verified_sanity_state_binding(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            report = Path(tmpdir) / "sanity.json"
            report.write_text("{}", encoding="utf-8")
            report_digest = EXECUTOR.sha256_file(report)
            job = {
                **_job("codriving"),
                "runner_key": "codriving_tvm_int8_numeric_gate",
                "compiled_artifact_path": "/tmp/route_b_int8_auto_decomp.vmexec",
                "compiled_artifact_digest": "a" * 64,
                "full_command_state_bindings": {
                    "sanity_report": {
                        "command_option": "--sanity-report-json",
                        "sha256_command_option": "--sanity-report-sha256",
                        "state_stage": "sanity",
                        "state_status": "success",
                        "path_field": "report_path",
                        "sha256_field": "report_sha256",
                        "verify_sha256": True,
                    }
                },
            }
            state = [{
                "job_id": "codriving-fp16",
                "stage": "sanity",
                "status": "success",
                "plan_fingerprint": EXECUTOR.plan_fingerprint(job, "sanity"),
                "report_path": str(report),
                "report_sha256": report_digest,
            }]

            selected, skipped = EXECUTOR.select_jobs([job], stage="full", state_rows=state)

            self.assertEqual(skipped, {})
            command = selected[0]["full_command"]
            self.assertEqual(command[command.index("--sanity-report-json") + 1], str(report))
            self.assertEqual(command[command.index("--sanity-report-sha256") + 1], report_digest)

            state.append({
                "job_id": "codriving-fp16",
                "stage": "full",
                "status": "success",
                "plan_fingerprint": EXECUTOR.plan_fingerprint(selected[0], "full"),
            })
            resumed, resumed_skipped = EXECUTOR.select_jobs(
                [job], stage="full", state_rows=state
            )

        self.assertEqual(resumed, [])
        self.assertEqual(resumed_skipped["codriving-fp16"], "already_successful")

    def test_busy_gpu_does_not_consume_attempt_and_two_retries_are_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = root / "report.json"
            job = _job()
            job["sanity_command"] = ["python3", "runner.py", "--report-json", str(report)]
            states: list[dict] = []
            idle_results = iter([(False, "gpu_busy:utilization=80"), (True, "gpu_idle")])
            run_count = 0

            def fake_run(command, **kwargs):
                nonlocal run_count
                run_count += 1
                self.assertEqual(command[command.index("--gpu-id") + 1], "7")
                if run_count == 3:
                    report.write_text(json.dumps(_pyramid_report()), encoding="utf-8")
                    return EXECUTOR.CommandResult(0, "third stdout", "")
                return EXECUTOR.CommandResult(1, f"stdout {run_count}", f"stderr {run_count}")

            result = EXECUTOR.execute_job(
                job,
                stage="sanity",
                gpu="7",
                python=Path("/envs/UniV2X/bin/python"),
                artifact_root=root / "attempts",
                append_state=states.append,
                wait_for_gpu=lambda gpu: next(idle_results),
                run_command=fake_run,
                busy_poll_seconds=0,
            )

            self.assertEqual(result["status"], "success")
            self.assertEqual(run_count, 3)
            attempts = [row for row in states if row.get("record_type") == "attempt"]
            waits = [row for row in states if row.get("record_type") == "scheduler_wait"]
            self.assertEqual([row["attempt"] for row in attempts], [1, 2, 3])
            self.assertEqual(len(waits), 1)
            self.assertNotIn("attempt", waits[0])
            self.assertEqual(attempts[-1]["report_sha256"], EXECUTOR.sha256_file(report))
            self.assertEqual(attempts[-1]["ap"]["ap50"], 0.7)
            self.assertTrue(Path(attempts[0]["stdout_path"]).is_file())
            self.assertTrue(Path(attempts[0]["stderr_path"]).is_file())
            self.assertEqual(attempts[0]["failure_reason"], "command_exit_1")
            self.assertEqual(attempts[-1]["cuda_visible_devices"], "7")
            self.assertEqual(attempts[-1]["command"][0], "/envs/UniV2X/bin/python")

    def test_resume_skips_existing_stage_success(self) -> None:
        jobs = [_job()]
        fingerprint = EXECUTOR.plan_fingerprint(jobs[0], "sanity")
        state = [{"job_id": "pyramid-fp16", "stage": "sanity", "status": "success", "plan_fingerprint": fingerprint}]
        selected, skipped = EXECUTOR.select_jobs(jobs, stage="sanity", state_rows=state)
        self.assertEqual(selected, [])
        self.assertEqual(skipped["pyramid-fp16"], "already_successful")

    def test_resume_remeasures_terminal_success_with_missing_report(self) -> None:
        job = _job()
        state = [{
            "record_type": "job_terminal",
            "job_id": "pyramid-fp16",
            "stage": "sanity",
            "status": "success",
            "plan_fingerprint": EXECUTOR.plan_fingerprint(job, "sanity"),
            "report_path": "/missing/report.json",
            "report_sha256": "a" * 64,
        }]

        selected, skipped = EXECUTOR.select_jobs([job], stage="sanity", state_rows=state)

        self.assertEqual(selected, [job])
        self.assertNotIn("pyramid-fp16", skipped)

    def test_full_success_also_satisfies_sanity_resume_gate(self) -> None:
        jobs = [_job()]
        fingerprint = EXECUTOR.plan_fingerprint(jobs[0], "full")
        state = [{"job_id": "pyramid-fp16", "stage": "full", "status": "success", "plan_fingerprint": fingerprint}]

        selected, skipped = EXECUTOR.select_jobs(jobs, stage="sanity", state_rows=state)

        self.assertEqual(selected, [])
        self.assertEqual(skipped["pyramid-fp16"], "already_successful")

    def test_resume_rejects_stale_and_unbound_seed_success(self) -> None:
        job = _job()
        stale = [{"job_id": "pyramid-fp16", "stage": "sanity", "status": "success", "plan_fingerprint": "stale"}]
        selected, _ = EXECUTOR.select_jobs([job], stage="sanity", state_rows=stale)
        self.assertEqual([job], selected)

        seed = [{
            "schema_version": "stage3_gold96_ap_seed_state_v3",
            "job_id": "pyramid-fp16",
            "stage": "sanity",
            "status": "success",
        }]
        selected, skipped = EXECUTOR.select_jobs([job], stage="sanity", state_rows=seed)
        self.assertEqual(selected, [job])
        self.assertNotIn("pyramid-fp16", skipped)

    def test_fingerprint_changes_for_each_bound_plan_input(self) -> None:
        job = _job()
        baseline = EXECUTOR.plan_fingerprint(job, "sanity")
        variants = [
            {**job, "runner_key": "other_runner"},
            {**job, "compiled_artifact_digest": "b" * 64},
            {**job, "compiled_artifact": "/artifacts/other.so"},
            {**job, "sanity_command": ["python3", "other.py", "--report-json", "reports/sanity.json"]},
            {**job, "sanity_command": ["python3", "runner.py", "--report-json", "reports/other.json"]},
        ]
        self.assertTrue(all(EXECUTOR.plan_fingerprint(item, "sanity") != baseline for item in variants))

    def test_tvm_execution_binds_physical_gpu_and_records_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = root / "report.json"
            job = _job("codriving")
            job["sanity_command"] = ["python3", "runner.py", "--gpu-id", "0", "--report-json", str(report)]
            states: list[dict] = []

            def fake_run(command, **kwargs):
                report.write_text(json.dumps(_codriving_report()), encoding="utf-8")
                self.assertEqual(command[command.index("--gpu-id") + 1], "7")
                return EXECUTOR.CommandResult(0, "", "")

            EXECUTOR.execute_job(
                job, stage="sanity", gpu="7", python=Path("/env/python"),
                artifact_root=root / "attempts", append_state=states.append,
                wait_for_gpu=lambda gpu: (True, "gpu_idle"), run_command=fake_run,
            )

            expected = EXECUTOR.plan_fingerprint(job, "sanity")
            attempt = next(row for row in states if row["record_type"] == "attempt")
            terminal = next(row for row in states if row["record_type"] == "job_terminal")
            self.assertEqual(attempt["plan_fingerprint"], expected)
            self.assertEqual(terminal["plan_fingerprint"], expected)

    def test_nonzero_exit_still_records_generated_report_sha_and_ap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = root / "report.json"
            job = _job()
            job["sanity_command"] = ["python3", "runner.py", "--report-json", str(report)]
            states: list[dict] = []

            def failed_run(command, **kwargs):
                report.write_text(json.dumps(_pyramid_report()), encoding="utf-8")
                return EXECUTOR.CommandResult(2, "", "late failure")

            EXECUTOR.execute_job(
                job,
                stage="sanity",
                gpu="0",
                python=Path("/env/python"),
                artifact_root=root / "attempts",
                append_state=states.append,
                wait_for_gpu=lambda gpu: (True, "gpu_idle"),
                run_command=failed_run,
            )

            attempt = next(row for row in states if row.get("record_type") == "attempt")
            self.assertEqual(attempt["failure_reason"], "command_exit_2")
            self.assertEqual(attempt["report_sha256"], EXECUTOR.sha256_file(report))
            self.assertEqual(attempt["ap"]["ap70"], 0.6)

    def test_codriving_int8_numeric_failure_is_a_single_terminal_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = root / "report.json"
            job = _job("codriving")
            job["runner_key"] = "codriving_tvm_int8_numeric_gate"
            job["sanity_command"] = ["python3", "runner.py", "--report-json", str(report)]
            states: list[dict] = []
            numeric_failure = {
                "status": "numerical_feasibility_failure",
                "processed_samples": 16,
                "engine_samples": 16,
                "engine_calls": 32,
                "engine_calls_per_sample": 2.0,
                "engine_accounting_valid": True,
                "fallback_samples": 0,
                "failed_samples": 0,
                "ap_measured": False,
                "failure_reasons": ["output_correlation_failed:res1"],
                "gates": {"sanity_16": False, "full_1789": False},
                "numeric_outputs": {
                    "res0": {"passed": True},
                    "res1": {"passed": False},
                    "res2": {"passed": True},
                },
            }

            def failed_gate(command, **kwargs):
                report.write_text(json.dumps(numeric_failure), encoding="utf-8")
                return EXECUTOR.CommandResult(2, "", "")

            terminal = EXECUTOR.execute_job(
                job,
                stage="sanity",
                gpu="0",
                python=Path("/env/python"),
                artifact_root=root / "attempts",
                append_state=states.append,
                wait_for_gpu=lambda gpu: (True, "gpu_idle"),
                run_command=failed_gate,
            )

            attempts = [row for row in states if row["record_type"] == "attempt"]
            self.assertEqual(len(attempts), 1)
            self.assertEqual(terminal["status"], "failed")
            self.assertEqual(terminal["failure_reason"], "numerical_feasibility_failure")
            self.assertFalse(EXECUTOR._has_success(states, job, "sanity"))
            self.assertTrue(EXECUTOR._has_numerical_terminal(states, job))

            selected, skipped = EXECUTOR.select_jobs([job], stage="full", state_rows=states)
            self.assertEqual(selected, [])
            self.assertEqual(skipped["codriving-fp16"], "numerical_feasibility_terminal")


if __name__ == "__main__":
    unittest.main()
