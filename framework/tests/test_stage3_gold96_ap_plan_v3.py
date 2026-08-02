from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "stage3_gold96_ap_plan_v3.py"
SPEC = importlib.util.spec_from_file_location("stage3_gold96_ap_plan_v3", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)


def _manifest() -> dict:
    jobs = []
    for index in range(96):
        model = "pyramid" if index < 48 else "codriving"
        dispatch = "trt_engine" if index % 2 == 0 else "tvm_auto"
        q_mode = "fp16" if index % 4 < 2 else "int8"
        profile = f"h800-{dispatch}-v3"
        jobs.append(
            {
                "job_id": f"{model}|{index}x{index}x{index}|q={q_mode}|profile={profile}",
                "model": model,
                "width": [index, index, index],
                "width_key": f"{index}x{index}x{index}",
                "q_mode": q_mode,
                "capability_profile_id": profile,
                "dispatch_key": dispatch,
                "required_metrics": ["latency", "energy", "ap"],
                "terminal_status": "pending",
            }
        )
    return {"schema_version": "stage3_gold_coldstart96_manifest_v3", "jobs": jobs}


def _performance_job(row: dict) -> dict:
    runner = ("trt_" if row["dispatch_key"] == "trt_engine" else "tvm_") + row["q_mode"]
    return {
        "job_id": f"perf-{row['job_id']}",
        "manifest_job_id": row["job_id"],
        "model": row["model"],
        "runner_key": runner,
    }


class Stage3Gold96ApPlanV3Tests(unittest.TestCase):
    def test_compiled_artifact_resolves_real_trt_profile_sibling_engine(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            engine = root / "artifacts" / "compiled.engine"
            engine.parent.mkdir()
            engine.write_bytes(b"engine")
            result = root / "trt_profile_result.json"
            result.write_text(
                json.dumps({"artifact_sha256": {"compiled_engine": "a" * 64}}),
                encoding="utf-8",
            )

            artifact, digest = MODULE._compiled_artifact(result, expected_runner="trt_fp16")

        self.assertEqual(artifact, str(engine))
        self.assertEqual(digest, "a" * 64)

    def test_compiled_artifact_accepts_explicit_state_bound_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            engine = root / "compiled.engine"
            engine.write_bytes(b"engine")
            result = root / "pilot_result.json"
            result.write_text(json.dumps({"build_success": True}), encoding="utf-8")

            artifact, digest = MODULE._compiled_artifact(
                result,
                expected_runner="trt_int8",
                state_bound_artifact=str(engine),
                state_bound_digest="b" * 64,
            )

        self.assertEqual(artifact, str(engine))
        self.assertEqual(digest, "b" * 64)

    def test_compiled_artifact_requires_existing_runner_specific_file(self) -> None:
        cases = [
            ("trt_fp16", "compiled.so"),
            ("tvm_fp16", "compiled.engine"),
            ("tvm_int8", "model.so"),
            ("tvm_int8", "other.vmexec"),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            result = root / "result.json"
            for runner, filename in cases:
                with self.subTest(runner=runner, filename=filename):
                    artifact = root / filename
                    artifact.write_bytes(b"artifact")
                    result.write_text(json.dumps({"artifact_path": str(artifact)}), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "expected"):
                        MODULE._compiled_artifact(result, expected_runner=runner)

            missing = root / "missing.engine"
            result.write_text(json.dumps({"artifact_path": str(missing)}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not exist"):
                MODULE._compiled_artifact(result, expected_runner="trt_fp16")

    def test_state_bound_fallback_obeys_runner_specific_filename(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            result = root / "result.json"
            result.write_text(json.dumps({"build_success": True}), encoding="utf-8")
            wrong = root / "model.so"
            wrong.write_bytes(b"not-int8-ready")

            with self.assertRaisesRegex(ValueError, "route_b_int8_auto_decomp.vmexec"):
                MODULE._compiled_artifact(
                    result,
                    expected_runner="tvm_int8",
                    state_bound_artifact=str(wrong),
                )

    def test_trt_commands_use_model_specific_checkpoint_and_bridge_cli(self) -> None:
        pyramid = {
            "model": "pyramid",
            "width": [32, 32, 128],
            "width_key": "32x32x128",
            "q_mode": "fp16",
        }
        codriving = {**pyramid, "model": "codriving"}

        pyramid_cmd = MODULE._command(
            "scripts/stage3_trt_multiscale_ap_bridge_v3.py", pyramid, "/tmp/p.engine", "/tmp/ap", 16
        )
        codriving_cmd = MODULE._command(
            "scripts/stage3_codriving_trt_multiscale_ap_bridge_v3.py", codriving, "/tmp/c.engine", "/tmp/ap", 16
        )

        self.assertEqual(
            pyramid_cmd[pyramid_cmd.index("--ckpt-dir") + 1],
            "/home/jichengzhi/V2X/models/dataset_a_cache/ft_032_032_128",
        )
        self.assertIn("--raw-dir", pyramid_cmd)
        self.assertEqual(
            codriving_cmd[codriving_cmd.index("--model-dir") + 1],
            "/exdata/jichengzhi/V2Xverse_pyramid/output/codriving_v2_gold_ap_20260709/32x32x128",
        )
        self.assertIn("--gate", codriving_cmd)
        self.assertIn("--out-json", codriving_cmd)

    def test_pyramid_tvm_fp16_uses_existing_bridge_cli(self) -> None:
        row = {
            "model": "pyramid",
            "width": [16, 16, 64],
            "width_key": "16x16x64",
            "q_mode": "fp16",
        }

        command = MODULE._command(
            "scripts/stage2_h800_fp16_rewritten_activation_bridge.py",
            row,
            "/tmp/route_b_fp16_auto.so",
            "/tmp/ap",
            16,
        )

        self.assertIn("--artifact-path", command)
        self.assertIn("--persistent-worker", command)
        self.assertEqual(command[command.index("--artifact-input-dtype") + 1], "float32")
        self.assertEqual(command[command.index("--num-samples") + 1], "16")

    def test_ap_output_paths_are_isolated_by_capability_profile(self) -> None:
        base = {
            "model": "pyramid",
            "width": [16, 16, 64],
            "width_key": "16x16x64",
            "q_mode": "fp16",
        }
        tvm = {**base, "capability_profile_id": "h800-tvm-probe-conditioned-v3"}
        trt = {**base, "capability_profile_id": "h800-trt-probe-conditioned-v3"}

        tvm_command = MODULE._command(
            "scripts/stage2_h800_fp16_rewritten_activation_bridge.py", tvm, "/tmp/a.so", "/tmp/ap", 16
        )
        trt_command = MODULE._command(
            "scripts/stage3_trt_multiscale_ap_bridge_v3.py", trt, "/tmp/a.engine", "/tmp/ap", 16
        )

        tvm_report = Path(tvm_command[tvm_command.index("--export-report-json") + 1])
        trt_report = Path(trt_command[trt_command.index("--report-json") + 1])
        self.assertNotEqual(tvm_report, trt_report)
        self.assertIn("h800-tvm-probe-conditioned-v3", str(tvm_report))
        self.assertIn("h800-trt-probe-conditioned-v3", str(trt_report))

    def test_codriving_tvm_fp16_command_binds_model_dir_and_report_path(self) -> None:
        row = {
            "model": "codriving",
            "width": [24, 64, 128],
            "width_key": "24x64x128",
            "q_mode": "fp16",
            "capability_profile_id": "h800-tvm-probe-conditioned-v3",
        }

        command = MODULE._command(
            "scripts/stage3_codriving_tvm_fp16_ap_bridge_v3.py",
            row,
            "/tmp/route_b_fp16_auto.so",
            "/tmp/ap",
            16,
        )

        self.assertEqual(
            command[command.index("--model-dir") + 1],
            "/exdata/jichengzhi/V2Xverse_pyramid/output/codriving_v2_gold_ap_20260709/24x64x128",
        )
        self.assertIn("--report-json", command)

    def test_pyramid_tvm_int8_command_binds_checkpoint_and_report_path(self) -> None:
        row = {
            "model": "pyramid",
            "width": [24, 48, 192],
            "width_key": "24x48x192",
            "q_mode": "int8",
            "capability_profile_id": "h800-tvm-probe-conditioned-v3",
        }

        command = MODULE._command(
            "scripts/stage3_pyramid_tvm_int8_ap_numeric_gate_v3.py",
            row,
            "/tmp/route_b_int8_auto_decomp.vmexec",
            "/tmp/ap",
            16,
        )

        self.assertEqual(
            command[command.index("--model-dir") + 1],
            "/home/jichengzhi/V2X/models/dataset_a_cache/ft_024_048_192",
        )
        self.assertIn("--report-json", command)

    def test_codriving_tvm_int8_full_command_defers_sanity_binding_to_executor(self) -> None:
        row = {
            "model": "codriving",
            "width": [24, 64, 128],
            "width_key": "24x64x128",
            "q_mode": "int8",
            "capability_profile_id": "h800-tvm-probe-conditioned-v3",
            "source_contract": {"checkpoint_path": "/tmp/frozen_checkpoint.pth"},
        }

        command = MODULE._command(
            "scripts/stage3_codriving_tvm_int8_ap_numeric_gate_v3.py",
            row,
            "/tmp/route_b_int8_auto_decomp.vmexec",
            "/tmp/ap",
            1789,
        )

        self.assertEqual(
            command[command.index("--model-dir") + 1],
            "/exdata/jichengzhi/V2Xverse_pyramid/output/codriving_v2_gold_ap_20260709/24x64x128",
        )
        self.assertNotIn("--sanity-report-json", command)
        self.assertIn("--report-json", command)
        self.assertEqual(
            command[command.index("--checkpoint-path") + 1],
            "/tmp/frozen_checkpoint.pth",
        )

    @classmethod
    def setUpClass(cls) -> None:
        SPEC.loader.exec_module(MODULE)

    def test_builds_96_rows_preserving_manifest_and_terminal_contract(self) -> None:
        manifest = _manifest()
        jobs = [_performance_job(row) for row in manifest["jobs"]]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            states = []
            for index, job in enumerate(jobs):
                if index == 0:
                    states.append({"job_id": job["job_id"], "status": "confirmed_failure"})
                    continue
                result = root / f"result-{index}.json"
                runner = jobs[index]["runner_key"]
                artifact_name = (
                    f"artifact-{index}.engine" if runner.startswith("trt_")
                    else f"artifact-{index}.so" if runner == "tvm_fp16"
                    else f"route-{index}/route_b_int8_auto_decomp.vmexec"
                )
                artifact = root / artifact_name
                artifact.parent.mkdir(parents=True, exist_ok=True)
                artifact.write_bytes(b"artifact")
                result.write_text(json.dumps({"artifact_path": str(artifact)}), encoding="utf-8")
                states.append({"job_id": job["job_id"], "status": "success", "result_json": str(result)})

            rows = MODULE.build_ap_plan(manifest, performance_jobs=jobs, performance_state_rows=states, pilot_root=root)

        self.assertEqual(len(rows), 96)
        failed = rows[0]
        self.assertEqual(failed["ap_terminal"], "feasibility_failure")
        self.assertEqual(failed["performance_terminal"], "confirmed_failure")
        for source, row in zip(manifest["jobs"], rows):
            self.assertEqual(row["manifest_job_id"], source["job_id"])
            self.assertEqual(row["model"], source["model"])
            self.assertEqual(row["width"], source["width"])
            self.assertEqual(row["q"], source["q_mode"])
            self.assertEqual(row["profile"], source["capability_profile_id"])
            self.assertEqual(row["required_metrics"], source["required_metrics"])
        self.assertEqual(rows[1]["runner_key"], "pyramid_tvm_fp16_bridge")
        self.assertEqual(rows[2]["runner_key"], "pyramid_trt_multiscale")
        self.assertEqual(rows[3]["runner_key"], "pyramid_tvm_int8_numeric_gate")
        self.assertEqual(rows[49]["runner_key"], "codriving_tvm_fp16_bridge")

    def test_success_uses_only_state_result_json_and_never_cross_backend_or_expected_result(self) -> None:
        manifest = _manifest()
        row = manifest["jobs"][0]
        correct_job = _performance_job(row)
        wrong_job = {**correct_job, "job_id": "wrong", "runner_key": "tvm_fp16", "expected_result_json": "/old/result.json"}
        correct_job["expected_result_json"] = "/old/result.json"
        with tempfile.TemporaryDirectory() as tmp:
            result = Path(tmp) / "real.json"
            engine = Path(tmp) / "current.engine"
            engine.write_bytes(b"engine")
            result.write_text(json.dumps({"engine_path": str(engine)}), encoding="utf-8")
            states = [
                {"job_id": "wrong", "status": "success", "result_json": str(result)},
                {"job_id": correct_job["job_id"], "status": "success", "result_json": str(result)},
            ]
            rows = MODULE.build_ap_plan(manifest, performance_jobs=[wrong_job, correct_job], performance_state_rows=states, pilot_root=tmp)

        self.assertEqual(rows[0]["compiled_artifact"], str(engine))
        self.assertNotEqual(rows[0]["compiled_artifact"], "/old/result.json")
        self.assertEqual(rows[0]["runner_key"], "pyramid_trt_multiscale")
        self.assertIn("--engine", rows[0]["sanity_command"])
        self.assertIn("16", rows[0]["sanity_command"])
        self.assertIn("1789", rows[0]["full_command"])

    def test_codriving_tvm_int8_model_so_is_not_ready(self) -> None:
        manifest = _manifest()
        row = manifest["jobs"][51]
        job = _performance_job(row)
        with tempfile.TemporaryDirectory() as tmp:
            result = Path(tmp) / "real.json"
            model_so = Path(tmp) / "model.so"
            model_so.write_bytes(b"so")
            result.write_text(json.dumps({"artifact_path": str(model_so)}), encoding="utf-8")
            rows = MODULE.build_ap_plan(
                manifest,
                performance_jobs=[job],
                performance_state_rows=[{"job_id": job["job_id"], "status": "success", "result_json": str(result)}],
                pilot_root=tmp,
            )

        self.assertEqual(rows[51]["ap_terminal"], "blocked_compiled_artifact_missing")
        self.assertIsNone(rows[51]["sanity_command"])
        self.assertIsNone(rows[51]["full_command"])

    def test_codriving_tvm_int8_ready_row_declares_sanity_state_binding(self) -> None:
        manifest = _manifest()
        source = manifest["jobs"][51]
        job = _performance_job(source)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "route_b_int8_auto_decomp.vmexec"
            artifact.write_bytes(b"vmexec")
            result = root / "result.json"
            result.write_text(json.dumps({"artifact_path": str(artifact)}), encoding="utf-8")
            rows = MODULE.build_ap_plan(
                manifest,
                performance_jobs=[job],
                performance_state_rows=[{"job_id": job["job_id"], "status": "success", "result_json": str(result)}],
                pilot_root=root,
            )

        planned = rows[51]
        self.assertEqual(planned["ap_terminal"], "ready")
        self.assertNotIn("--sanity-report-json", planned["full_command"])
        self.assertEqual(
            planned["full_command_state_bindings"],
            {
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
        )


if __name__ == "__main__":
    unittest.main()
