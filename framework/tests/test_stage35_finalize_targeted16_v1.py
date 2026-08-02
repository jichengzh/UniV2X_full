from __future__ import annotations

import csv
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "stage35_finalize_targeted16_v1.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("stage35_finalize_targeted16_v1", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _manifest() -> dict:
    jobs = []
    for group_index in range(4):
        width = [16 + group_index, 32, 96]
        group_id = f"pyramid|{'x'.join(map(str, width))}"
        for dispatch_key, q_mode in (
            ("tvm_auto", "fp16"),
            ("tvm_auto", "int8"),
            ("trt_engine", "fp16"),
            ("trt_engine", "int8"),
        ):
            backend = "tvm" if dispatch_key == "tvm_auto" else "trt"
            profile = f"h800-{backend}-probe-conditioned-v3"
            jobs.append({
                "schema_version": "stage35_gold128_targeted_supplement_manifest_v1",
                "job_id": f"{group_id}|q={q_mode}|profile={profile}",
                "group_id": group_id,
                "model": "pyramid",
                "width": width,
                "q_mode": q_mode,
                "capability_profile_id": profile,
                "dispatch_key": dispatch_key,
                "split": "train",
                "width_stratum": "targeted_failure_boundary",
            })
    return {
        "schema_version": "stage35_gold128_targeted_supplement_manifest_v1",
        "locked_holdout_policy": "retain_the_original_six_groups_unchanged",
        "jobs": jobs,
    }


def _mixed_model_feedback_manifest() -> dict:
    manifest = _manifest()
    manifest["schema_version"] = "stage4_feedback16_manifest_v1"
    for index, job in enumerate(manifest["jobs"]):
        model = "codriving" if index < 8 else "pyramid"
        width_key = "x".join(map(str, job["width"]))
        old_group_id = job["group_id"]
        job["schema_version"] = "stage4_feedback16_manifest_v1"
        job["model"] = model
        job["group_id"] = f"{model}|{width_key}"
        job["job_id"] = job["job_id"].replace(old_group_id, job["group_id"])
        job["split"] = "online_feedback"
    return manifest


def _evidence(
    root: Path, manifest: dict
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    performance_jobs = []
    performance_rows = []
    ap_rows = []
    plan_rows = []
    for index, job in enumerate(manifest["jobs"]):
        performance_path = root / f"performance-{index}.json"
        performance_path.write_text(
            json.dumps({"lat_p50_ms": 4.0 + index / 10, "energy_j": 0.5 + index / 100}),
            encoding="utf-8",
        )
        report = {
            "status": "success",
            "processed_samples": 1789,
            "failed_samples": 0,
            "fallback_samples": 0,
            "ap30": 0.8,
            "ap50": 0.7,
            "ap70": 0.6,
        }
        if job["model"] == "codriving":
            report.update({"engine_samples": 1789, "gates": {"full_1789": True}})
        elif job["dispatch_key"] == "tvm_auto":
            report.update({"ap_measured": True, "smoke_gate_passed": True})
        else:
            report["engine_ap_claim"] = True
        report_path = root / f"ap-{index}.json"
        report_path.write_text(json.dumps(report), encoding="utf-8")
        performance_job_id = f"targeted-perf-{index}"
        performance_runner = f"{'tvm' if job['dispatch_key'] == 'tvm_auto' else 'trt'}_{job['q_mode']}"
        ap_runner = {
            ("pyramid", "tvm_fp16"): "pyramid_tvm_fp16_bridge",
            ("pyramid", "tvm_int8"): "pyramid_tvm_int8_numeric_gate",
            ("pyramid", "trt_fp16"): "pyramid_trt_multiscale",
            ("pyramid", "trt_int8"): "pyramid_trt_multiscale",
            ("codriving", "tvm_fp16"): "codriving_tvm_fp16_bridge",
            ("codriving", "tvm_int8"): "codriving_tvm_int8_numeric_gate",
            ("codriving", "trt_fp16"): "codriving_trt_multiscale",
            ("codriving", "trt_int8"): "codriving_trt_multiscale",
        }[(job["model"], performance_runner)]
        performance_jobs.append({
            "job_id": performance_job_id,
            "manifest_job_id": job["job_id"],
            "runner_key": performance_runner,
            "model": job["model"],
        })
        plan_rows.append({
            "manifest_job_id": job["job_id"],
            "performance_job_id": performance_job_id,
            "performance_result_json": str(performance_path),
            "runner_key": ap_runner,
            "full_command": ["python3", "runner.py", "--report-json", str(report_path)],
        })
        performance_rows.append({
            "job_id": performance_job_id,
            "status": "success",
            "result_json": str(performance_path),
        })
        ap_rows.append({
            "record_type": "job_terminal",
            "job_id": job["job_id"],
            "stage": "full",
            "status": "success",
            "report_path": str(report_path),
        })
    return performance_jobs, plan_rows, performance_rows, ap_rows


class Stage35FinalizeTargeted16V1Tests(unittest.TestCase):
    def test_cli_help_runs_from_repo_root(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_finalizes_exactly_four_train_groups_with_complete_evidence(self) -> None:
        module = _load_module()
        manifest = _manifest()
        with tempfile.TemporaryDirectory() as directory:
            performance_jobs, plan, performance, ap = _evidence(Path(directory), manifest)
            result = module.finalize_targeted16(
                manifest,
                performance_job_rows=performance_jobs,
                ap_plan_rows=plan,
                performance_state_rows=performance,
                ap_state_rows=ap,
            )

        self.assertEqual(result["summary"], {"measured": 16, "failure": 0, "pending": 0, "total": 16})
        self.assertEqual(len(result["rows"]), 16)
        self.assertEqual(len(result["group_audit"]), 4)
        self.assertIs(result["manifest"], manifest)
        for row in result["rows"]:
            self.assertEqual(row["schema_version"], "stage35_targeted16_final_v1")
            self.assertEqual(row["split"], "train")
            for key in ("latency_ms", "energy_j", "ap30", "ap50", "ap70"):
                self.assertTrue(math.isfinite(row[key]), (key, row))
            for key in ("performance_result_sha256", "ap_report_sha256"):
                self.assertRegex(row[key], r"^[0-9a-f]{64}$")

    def test_preserves_online_feedback_as_a_separate_batch(self) -> None:
        module = _load_module()
        manifest = _mixed_model_feedback_manifest()
        with tempfile.TemporaryDirectory() as directory:
            performance_jobs, plan, performance, ap = _evidence(Path(directory), manifest)
            result = module.finalize_targeted16(
                manifest,
                performance_job_rows=performance_jobs,
                ap_plan_rows=plan,
                performance_state_rows=performance,
                ap_state_rows=ap,
                manifest_schema="stage4_feedback16_manifest_v1",
                required_split="online_feedback",
            )

        self.assertEqual(len(result["rows"]), 16)
        self.assertEqual({row["split"] for row in result["rows"]}, {"online_feedback"})
        self.assertEqual(
            {row["schema_version"] for row in result["rows"]},
            {"stage4_feedback16_final_v1"},
        )
        self.assertEqual({row["model"] for row in result["rows"]}, {"codriving", "pyramid"})

    def test_stage4_schema_rejects_mismatched_output_schema(self) -> None:
        module = _load_module()
        manifest = _mixed_model_feedback_manifest()
        with self.assertRaisesRegex(ValueError, "output schema"):
            module.finalize_targeted16(
                manifest,
                performance_job_rows=[],
                ap_plan_rows=[],
                performance_state_rows=[],
                ap_state_rows=[],
                manifest_schema="stage4_feedback16_manifest_v1",
                output_schema="stage35_targeted16_final_v1",
            )

    def test_stage4_schema_rejects_train_split_even_when_requested(self) -> None:
        module = _load_module()
        manifest = _mixed_model_feedback_manifest()
        for job in manifest["jobs"]:
            job["split"] = "train"
        with self.assertRaisesRegex(ValueError, "online_feedback"):
            module.finalize_targeted16(
                manifest,
                performance_job_rows=[],
                ap_plan_rows=[],
                performance_state_rows=[],
                ap_state_rows=[],
                manifest_schema="stage4_feedback16_manifest_v1",
                output_schema="stage4_feedback16_final_v1",
                required_split="train",
            )

    def test_rejects_noncanonical_model_spelling_at_manifest_boundary(self) -> None:
        module = _load_module()
        manifest = _mixed_model_feedback_manifest()
        manifest["jobs"][0]["model"] = "CoDriving"
        with self.assertRaisesRegex(ValueError, "canonical model"):
            module.finalize_targeted16(
                manifest,
                performance_job_rows=[],
                ap_plan_rows=[],
                performance_state_rows=[],
                ap_state_rows=[],
                manifest_schema="stage4_feedback16_manifest_v1",
                output_schema="stage4_feedback16_final_v1",
            )

    def test_rejects_wrong_schema_non_train_and_incomplete_four_arm_groups(self) -> None:
        module = _load_module()
        cases = []
        wrong_schema = _manifest()
        wrong_schema["schema_version"] = "wrong"
        cases.append((wrong_schema, "stage35_gold128_targeted_supplement_manifest_v1"))
        non_train = _manifest()
        non_train["jobs"][0]["split"] = "locked_holdout"
        cases.append((non_train, "train"))
        incomplete = _manifest()
        incomplete["jobs"].pop()
        cases.append((incomplete, "exactly 16"))

        for manifest, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                module.finalize_targeted16(
                    manifest,
                    performance_job_rows=[],
                    ap_plan_rows=[],
                    performance_state_rows=[],
                    ap_state_rows=[],
                )

    def test_rejects_duplicate_manifest_id_and_duplicate_arm(self) -> None:
        module = _load_module()
        duplicate_id = _manifest()
        duplicate_id["jobs"][1]["job_id"] = duplicate_id["jobs"][0]["job_id"]
        duplicate_arm = _manifest()
        duplicate_arm["jobs"][1] = {
            **duplicate_arm["jobs"][1],
            "dispatch_key": duplicate_arm["jobs"][0]["dispatch_key"],
            "q_mode": duplicate_arm["jobs"][0]["q_mode"],
        }
        for manifest, message in ((duplicate_id, "unique"), (duplicate_arm, "four-arm")):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                module.finalize_targeted16(
                    manifest,
                    performance_job_rows=[],
                    ap_plan_rows=[],
                    performance_state_rows=[],
                    ap_state_rows=[],
                )

    def test_rejects_non_final_rows(self) -> None:
        module = _load_module()
        manifest = _manifest()
        with tempfile.TemporaryDirectory() as directory:
            performance_jobs, plan, performance, ap = _evidence(Path(directory), manifest)
            performance.pop()
            with self.assertRaisesRegex(ValueError, "non-final|non-terminal|measured_success_gold"):
                module.finalize_targeted16(
                    manifest,
                    performance_job_rows=performance_jobs,
                    ap_plan_rows=plan,
                    performance_state_rows=performance,
                    ap_state_rows=ap,
                )

    def test_writes_all_outputs_and_csv_uses_row_key_union(self) -> None:
        module = _load_module()
        manifest = _manifest()
        with tempfile.TemporaryDirectory() as evidence_directory:
            performance_jobs, plan, performance, ap = _evidence(Path(evidence_directory), manifest)
            result = module.finalize_targeted16(
                manifest,
                performance_job_rows=performance_jobs,
                ap_plan_rows=plan,
                performance_state_rows=performance,
                ap_state_rows=ap,
            )
            result["rows"][-1]["later_only"] = "kept"
            with tempfile.TemporaryDirectory() as output_directory:
                module.write_targeted16_outputs(result, output_directory)
                output = Path(output_directory)
                self.assertEqual(
                    {path.name for path in output.iterdir()},
                    {
                        "targeted16_final.json",
                        "targeted16_final.jsonl",
                        "targeted16_final.csv",
                        "targeted16_manifest.json",
                        "targeted16_audit.json",
                    },
                )
                with (output / "targeted16_final.csv").open(
                    "r", encoding="utf-8", newline=""
                ) as handle:
                    csv_rows = list(csv.DictReader(handle))
                written_manifest = json.loads(
                    (output / "targeted16_manifest.json").read_text(encoding="utf-8")
                )

        self.assertEqual(csv_rows[-1]["later_only"], "kept")
        self.assertEqual(written_manifest["schema_version"], manifest["schema_version"])

    def test_writes_feedback_batch_with_explicit_file_prefix(self) -> None:
        module = _load_module()
        manifest = _manifest()
        with tempfile.TemporaryDirectory() as evidence_directory:
            performance_jobs, plan, performance, ap = _evidence(Path(evidence_directory), manifest)
            result = module.finalize_targeted16(
                manifest,
                performance_job_rows=performance_jobs,
                ap_plan_rows=plan,
                performance_state_rows=performance,
                ap_state_rows=ap,
            )
            with tempfile.TemporaryDirectory() as output_directory:
                module.write_targeted16_outputs(
                    result,
                    output_directory,
                    file_prefix="feedback16",
                )
                names = {path.name for path in Path(output_directory).iterdir()}

        self.assertEqual(
            names,
            {
                "feedback16_final.json",
                "feedback16_final.jsonl",
                "feedback16_final.csv",
                "feedback16_manifest.json",
                "feedback16_audit.json",
            },
        )

    def test_rejects_swapped_or_stale_ap_performance_binding(self) -> None:
        module = _load_module()
        manifest = _manifest()
        with tempfile.TemporaryDirectory() as directory:
            performance_jobs, plan, performance, ap = _evidence(Path(directory), manifest)
            swapped = [dict(row) for row in plan]
            swapped[0]["performance_job_id"], swapped[1]["performance_job_id"] = (
                swapped[1]["performance_job_id"],
                swapped[0]["performance_job_id"],
            )
            with self.assertRaisesRegex(ValueError, "performance job binding"):
                module.finalize_targeted16(
                    manifest,
                    performance_job_rows=performance_jobs,
                    ap_plan_rows=swapped,
                    performance_state_rows=performance,
                    ap_state_rows=ap,
                )

            swapped_performance_evidence = [dict(row) for row in performance]
            swapped_performance_evidence[0]["result_json"], swapped_performance_evidence[1]["result_json"] = (
                swapped_performance_evidence[1]["result_json"],
                swapped_performance_evidence[0]["result_json"],
            )
            with self.assertRaisesRegex(ValueError, "performance evidence path"):
                module.finalize_targeted16(
                    manifest,
                    performance_job_rows=performance_jobs,
                    ap_plan_rows=plan,
                    performance_state_rows=swapped_performance_evidence,
                    ap_state_rows=ap,
                )

            swapped_ap_evidence = [dict(row) for row in ap]
            swapped_ap_evidence[0]["report_path"], swapped_ap_evidence[1]["report_path"] = (
                swapped_ap_evidence[1]["report_path"],
                swapped_ap_evidence[0]["report_path"],
            )
            with self.assertRaisesRegex(ValueError, "AP evidence path"):
                module.finalize_targeted16(
                    manifest,
                    performance_job_rows=performance_jobs,
                    ap_plan_rows=plan,
                    performance_state_rows=performance,
                    ap_state_rows=swapped_ap_evidence,
                )

            stale_runner = [dict(row) for row in plan]
            stale_runner[0]["runner_key"] = "pyramid_tvm_int8_numeric_gate"
            with self.assertRaisesRegex(ValueError, "runner"):
                module.finalize_targeted16(
                    manifest,
                    performance_job_rows=performance_jobs,
                    ap_plan_rows=stale_runner,
                    performance_state_rows=performance,
                    ap_state_rows=ap,
                )

    def test_rejects_performance_state_with_conflicting_explicit_manifest_binding(self) -> None:
        module = _load_module()
        manifest = _manifest()
        with tempfile.TemporaryDirectory() as directory:
            performance_jobs, plan, performance, ap = _evidence(Path(directory), manifest)
            performance[0] = {
                **performance[0],
                "manifest_job_id": manifest["jobs"][1]["job_id"],
            }
            with self.assertRaisesRegex(ValueError, "performance state binding"):
                module.finalize_targeted16(
                    manifest,
                    performance_job_rows=performance_jobs,
                    ap_plan_rows=plan,
                    performance_state_rows=performance,
                    ap_state_rows=ap,
                )

    def test_accepts_repair_importer_performance_id_alias(self) -> None:
        module = _load_module()
        manifest = _manifest()
        with tempfile.TemporaryDirectory() as directory:
            performance_jobs, plan, performance, ap = _evidence(Path(directory), manifest)
            performance = [
                {**row, "manifest_job_id": row["job_id"]}
                for row in performance
            ]
            result = module.finalize_targeted16(
                manifest,
                performance_job_rows=performance_jobs,
                ap_plan_rows=plan,
                performance_state_rows=performance,
                ap_state_rows=ap,
            )

        self.assertEqual(result["summary"]["measured"], 16)

    def test_repair_evidence_paths_are_bound_to_model_and_width(self) -> None:
        module = _load_module()
        manifest = _manifest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            performance_jobs, plan, performance, ap = _evidence(root, manifest)
            repair_index = 1
            source = manifest["jobs"][repair_index]
            width = "x".join(map(str, source["width"]))
            performance_path = (
                root
                / "tvm_int8_repair"
                / source["model"]
                / width
                / "build"
                / f"{source['model']}_{width}_scaleaware"
                / "route_b_int8_auto_decomp_result.json"
            )
            ap_path = (
                root
                / "tvm_int8_repair"
                / source["model"]
                / width
                / "ap_full"
                / "full_ap_eval_report.json"
            )
            performance_path.parent.mkdir(parents=True)
            ap_path.parent.mkdir(parents=True)
            performance_path.write_bytes(Path(performance[repair_index]["result_json"]).read_bytes())
            ap_path.write_bytes(Path(ap[repair_index]["report_path"]).read_bytes())
            repair_performance = [dict(row) for row in performance]
            repair_performance[repair_index] = {
                **repair_performance[repair_index],
                "manifest_job_id": repair_performance[repair_index]["job_id"],
                "result_json": str(performance_path),
                "source": "stage3_tvm_int8_repair_v3",
            }
            repair_ap = [dict(row) for row in ap]
            repair_ap[repair_index] = {
                **repair_ap[repair_index],
                "report_path": str(ap_path),
                "source": "stage3_tvm_int8_repair_v3",
            }
            result = module.finalize_targeted16(
                manifest,
                performance_job_rows=performance_jobs,
                ap_plan_rows=plan,
                performance_state_rows=repair_performance,
                ap_state_rows=repair_ap,
            )

            self.assertEqual(result["summary"]["measured"], 16)

            wrong_width_performance = [dict(row) for row in repair_performance]
            wrong_width_performance[repair_index] = {
                **wrong_width_performance[repair_index],
                "result_json": str(performance_path).replace(width, "99x99x99"),
            }
            with self.assertRaisesRegex(ValueError, "performance evidence path"):
                module.finalize_targeted16(
                    manifest,
                    performance_job_rows=performance_jobs,
                    ap_plan_rows=plan,
                    performance_state_rows=wrong_width_performance,
                    ap_state_rows=repair_ap,
                )

            wrong_width_ap = [dict(row) for row in repair_ap]
            wrong_width_ap[repair_index] = {
                **wrong_width_ap[repair_index],
                "report_path": str(ap_path).replace(width, "99x99x99"),
            }
            with self.assertRaisesRegex(ValueError, "AP evidence path"):
                module.finalize_targeted16(
                    manifest,
                    performance_job_rows=performance_jobs,
                    ap_plan_rows=plan,
                    performance_state_rows=repair_performance,
                    ap_state_rows=wrong_width_ap,
                )


if __name__ == "__main__":
    unittest.main()
