from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "stage3_finalize_gold96_v3.py"
SPEC = importlib.util.spec_from_file_location("stage3_finalize_gold96_v3", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _manifest() -> dict:
    jobs = []
    for group_index in range(24):
        model = "pyramid" if group_index < 12 else "codriving"
        width = [16 + group_index, 32, 64]
        group_id = f"{model}|{'x'.join(map(str, width))}"
        for backend, q in (("tvm", "fp16"), ("tvm", "int8"), ("trt", "fp16"), ("trt", "int8")):
            profile = f"h800-{backend}-probe-conditioned-v3"
            jobs.append({
                "job_id": f"{group_id}|q={q}|profile={profile}",
                "group_id": group_id,
                "model": model,
                "width": width,
                "q_mode": q,
                "capability_profile_id": profile,
                "dispatch_key": "tvm_auto" if backend == "tvm" else "trt_engine",
            })
    return {"schema_version": "stage3_gold_coldstart96_manifest_v3", "jobs": jobs}


def _ap_plan(manifest: dict) -> list[dict]:
    return [
        {
            "manifest_job_id": row["job_id"],
            "performance_job_id": f"perf-{index}",
            "model": row["model"],
            "runner_key": f"{row['model']}_{'tvm' if row['dispatch_key'] == 'tvm_auto' else 'trt'}_{row['q_mode']}",
        }
        for index, row in enumerate(manifest["jobs"])
    ]


def _write_result(root: Path, name: str, *, latency: float = 4.25, energy: float = 0.75) -> Path:
    path = root / name
    path.write_text(json.dumps({"lat_p50_ms": latency, "energy_j": energy}), encoding="utf-8")
    return path


def _write_ap_report(root: Path, name: str, *, model: str, backend: str, **overrides: object) -> Path:
    payload: dict[str, object] = {
        "status": "success",
        "processed_samples": 1789,
        "engine_samples": 1789,
        "failed_samples": 0,
        "fallback_samples": 0,
        "ap30": 0.8,
        "ap50": 0.7,
        "ap70": 0.6,
    }
    if model == "codriving":
        payload["gates"] = {"sanity_16": True, "full_1789": True, "blockers": []}
    elif backend == "tvm":
        payload.update({"ap_measured": True, "smoke_gate_passed": True})
    else:
        payload["engine_ap_claim"] = True
    payload.update(overrides)
    path = root / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class Stage3FinalizeGold96V3Tests(unittest.TestCase):
    def test_confirmed_performance_failure_normalizes_executor_reasons(self) -> None:
        row = MODULE._finalize_row(
            {
                "job_id": "pyramid-fp16",
                "group_id": "pyramid|24x128x256",
                "model": "pyramid",
                "width": [24, 128, 256],
                "q_mode": "fp16",
                "dispatch_key": "tvm_auto",
            },
            [{
                "status": "confirmed_failure",
                "failure_reasons": ["returncode=1"],
            }],
            [],
            output_schema="stage5_feedback_row_v2",
        )

        self.assertEqual(row["terminal_status"], "feasibility_failure")
        self.assertEqual(
            row["failure_reason"],
            "confirmed_backend_performance_failure:returncode=1",
        )

    def test_pyramid_tvm_accepts_full_numeric_gate_schema(self) -> None:
        source = {"model": "pyramid", "dispatch_key": "tvm_auto"}
        report = {
            "status": "success", "processed_samples": 1789,
            "ap_measured": True, "ap_row_allowed": True,
            "feasibility_blockers": [], "gates": {"sanity_16": True, "full_1789": True},
        }
        self.assertTrue(MODULE._valid_ap_report(source, report, "full"))
        report["gates"]["full_1789"] = False
        self.assertFalse(MODULE._valid_ap_report(source, report, "full"))

    def test_decision_priority_and_evidence_contract(self) -> None:
        manifest = _manifest()
        plan = _ap_plan(manifest)
        ids = [row["job_id"] for row in manifest["jobs"]]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = _write_result(root, "performance.json")
            failed_report = root / "failed.json"
            failed_report.write_text(json.dumps({"status": "blocked"}), encoding="utf-8")
            sanity_report = _write_ap_report(root, "sanity.json", model="pyramid", backend="tvm", processed_samples=16, engine_samples=16)
            report = _write_ap_report(root, "full.json", model="pyramid", backend="trt")
            invalid_report = _write_ap_report(root, "invalid.json", model="pyramid", backend="tvm", ap50=None)
            report_sha = MODULE.sha256_file(report)
            failed_report_sha = MODULE.sha256_file(failed_report)
            performance = [{"job_id": "perf-0", "status": "success", "result_json": str(result)}] + [
                {"job_id": f"perf-{index}", "status": status, **extra}
                for index, status, extra in (
                    (0, "confirmed_failure", {"failure_reason": "compile"}),
                    (1, "success", {"result_json": str(result)}),
                    (2, "success", {"result_json": str(result)}),
                    (3, "success", {"result_json": str(result)}),
                    (4, "success", {"result_json": str(result)}),
                    (5, "success", {"result_json": str(result)}),
                )
            ]
            ap = [
                {"record_type": "terminal_event", "job_id": ids[1], "stage": "sanity", "status": "failed", "report_path": str(failed_report), "report_sha256": "s" * 64},
                {"record_type": "terminal_event", "job_id": ids[2], "stage": "sanity", "status": "success", "report_path": str(sanity_report), "report_sha256": "a" * 64},
                {"record_type": "job_terminal", "job_id": ids[3], "stage": "sanity", "status": "failed", "report_path": str(failed_report), "report_sha256": "b" * 64},
                {"record_type": "job_terminal", "job_id": ids[3], "stage": "full", "status": "success", "report_path": str(report), "report_sha256": "c" * 64, "ap": {"ap30": 9.8, "ap50": 9.7, "ap70": 9.6}},
                {"record_type": "job_terminal", "job_id": ids[4], "stage": "full", "status": "success", "report_path": str(invalid_report), "report_sha256": "d" * 64, "ap": {"ap30": 0.8, "ap50": 0.7, "ap70": 0.6}},
                {"record_type": "job_terminal", "job_id": ids[5], "stage": "full", "status": "failed", "report_path": str(report), "report_sha256": "f" * 64, "failure_reason": "eval_failed"},
            ]

            output = MODULE.finalize_gold96(manifest, ap_plan_rows=plan, performance_state_rows=performance, ap_state_rows=ap)

        rows = output["rows"]
        self.assertEqual(rows[0]["terminal_status"], "feasibility_failure")
        self.assertEqual(rows[1]["terminal_status"], "numerical_feasibility_failure")
        self.assertEqual(rows[2]["terminal_status"], "pending_full_ap")
        self.assertEqual(rows[3]["terminal_status"], "measured_success_gold")
        self.assertEqual(rows[4]["terminal_status"], "blocked_ap_evidence")
        self.assertEqual(rows[5]["terminal_status"], "blocked_ap_evidence")
        self.assertEqual(rows[3]["latency_ms"], 4.25)
        self.assertEqual(rows[3]["energy_j"], 0.75)
        self.assertEqual(len(rows[3]["performance_result_sha256"]), 64)
        self.assertEqual((rows[3]["ap30"], rows[3]["ap50"], rows[3]["ap70"]), (0.8, 0.7, 0.6))
        self.assertEqual(rows[3]["ap_report_sha256"], report_sha)
        self.assertEqual(rows[1]["ap_report_sha256"], failed_report_sha)

    def test_stage5_strict_failure_rejects_infrastructure_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            perf = _write_result(root, "perf.json")
            report = root / "failed.json"
            report.write_text(json.dumps({"status": "blocked"}), encoding="utf-8")
            source = {
                "job_id": "codriving-int8",
                "group_id": "codriving|64x96x256",
                "model": "codriving",
                "width": [64, 96, 256],
                "q_mode": "int8",
                "dispatch_key": "tvm_auto",
            }

            row = MODULE._finalize_row(
                source,
                [{"status": "success", "result_json": str(perf)}],
                [{
                    "record_type": "job_terminal",
                    "stage": "sanity",
                    "status": "failed",
                    "failure_reason": "command_exit_2",
                    "report_path": str(report),
                }],
                output_schema="stage5_feedback_row_v2",
            )

        self.assertEqual(row["terminal_status"], "blocked_ap_evidence")

    def test_stage5_strict_numerical_failure_requires_matching_full_skip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            perf = _write_result(root, "perf.json")
            report = root / "numeric_failure.json"
            report.write_text(json.dumps({
                "status": "numerical_feasibility_failure",
                "processed_samples": 16,
                "engine_samples": 16,
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
            }), encoding="utf-8")
            digest = MODULE.sha256_file(report)
            source = {
                "job_id": "codriving-int8",
                "group_id": "codriving|64x96x256",
                "model": "codriving",
                "width": [64, 96, 256],
                "q_mode": "int8",
                "dispatch_key": "tvm_auto",
            }
            sanity = {
                "record_type": "job_terminal",
                "stage": "sanity",
                "status": "failed",
                "failure_reason": "numerical_feasibility_failure",
                "report_path": str(report),
                "report_sha256": digest,
            }

            blocked = MODULE._finalize_row(
                source,
                [{"status": "success", "result_json": str(perf)}],
                [sanity],
                output_schema="stage5_feedback_row_v2",
            )
            released = MODULE._finalize_row(
                source,
                [{"status": "success", "result_json": str(perf)}],
                [sanity, {
                    "record_type": "job_terminal",
                    "stage": "full",
                    "status": "skipped_numerical_feasibility",
                    "failure_reason": "numerical_feasibility_failure",
                    "report_path": str(report),
                    "report_sha256": digest,
                }],
                output_schema="stage5_feedback_row_v2",
            )

        self.assertEqual(blocked["terminal_status"], "blocked_ap_evidence")
        self.assertEqual(released["terminal_status"], "numerical_feasibility_failure")

    def test_write_dataset_outputs_supports_mixed_failure_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = {
                "rows": [{"row_id": "ok"}, {"row_id": "failed", "failure_evidence_path": "/x"}],
                "group_audit": [],
                "summary": {"total": 2},
            }

            MODULE.write_dataset_outputs(
                result,
                tmp,
                file_prefix="mixed",
                output_schema="test",
            )

            header = (Path(tmp) / "mixed_final.csv").read_text(encoding="utf-8").splitlines()[0]
            self.assertEqual(header, "row_id,failure_evidence_path")

    def test_full_report_contract_rejects_bad_gate_count_and_fallback_for_each_schema(self) -> None:
        cases = (
            ("pyramid", "trt", {"engine_ap_claim": False}),
            ("pyramid", "tvm", {"processed_samples": 1788}),
            ("codriving", "trt", {"fallback_samples": 1}),
            ("codriving", "tvm", {"gates": {"full_1789": False}}),
        )
        for index, (model, backend, override) in enumerate(cases):
            with self.subTest(model=model, backend=backend):
                manifest = _manifest()
                plan = _ap_plan(manifest)
                job_index = next(i for i, row in enumerate(manifest["jobs"]) if row["model"] == model and row["dispatch_key"] == ("tvm_auto" if backend == "tvm" else "trt_engine"))
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    perf = _write_result(root, "perf.json")
                    report = _write_ap_report(root, f"bad-{index}.json", model=model, backend=backend, **override)
                    output = MODULE.finalize_gold96(
                        manifest,
                        ap_plan_rows=plan,
                        performance_state_rows=[{"job_id": f"perf-{job_index}", "status": "success", "result_json": str(perf)}],
                        ap_state_rows=[{"record_type": "job_terminal", "job_id": manifest["jobs"][job_index]["job_id"], "stage": "full", "status": "success", "report_path": str(report), "report_sha256": "f" * 64, "ap": {"ap30": 1.0, "ap50": 1.0, "ap70": 1.0}}],
                    )
                self.assertEqual(output["rows"][job_index]["terminal_status"], "blocked_ap_evidence")

    def test_never_uses_ap_from_another_backend_and_never_promotes_sanity_ap(self) -> None:
        manifest = _manifest()
        plan = _ap_plan(manifest)
        first, second = manifest["jobs"][:2]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = _write_result(root, "result.json")
            report = root / "report.json"
            report.write_text("{}", encoding="utf-8")
            performance = [{"job_id": "perf-0", "status": "success", "result_json": str(result)}]
            ap = [{"record_type": "job_terminal", "job_id": second["job_id"], "stage": "full", "status": "success", "report_path": str(report), "report_sha256": "e" * 64, "ap": {"ap30": 0.9, "ap50": 0.8, "ap70": 0.7}}]
            output = MODULE.finalize_gold96(manifest, ap_plan_rows=plan, performance_state_rows=performance, ap_state_rows=ap)

        self.assertEqual(output["rows"][0]["terminal_status"], "pending_ap")
        self.assertIsNone(output["rows"][0]["ap30"])
        self.assertEqual(output["rows"][1]["terminal_status"], "pending_performance")

    def test_audit_has_24_groups_of_four_and_summary_partitions_all_rows(self) -> None:
        output = MODULE.finalize_gold96(_manifest(), ap_plan_rows=_ap_plan(_manifest()), performance_state_rows=[], ap_state_rows=[])

        self.assertEqual(len(output["rows"]), 96)
        self.assertEqual(len(output["group_audit"]), 24)
        self.assertTrue(all(group["row_count"] == 4 for group in output["group_audit"]))
        self.assertTrue(all(group["all_terminal"] for group in output["group_audit"]))
        self.assertEqual(output["summary"], {"measured": 0, "failure": 0, "pending": 96, "total": 96})

    def test_cli_writes_exactly_one_record_per_manifest_row_in_each_format(self) -> None:
        manifest = _manifest()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = {name: root / name for name in ("manifest.json", "plan.jsonl", "perf.jsonl", "ap.jsonl")}
            paths["manifest.json"].write_text(json.dumps(manifest), encoding="utf-8")
            paths["plan.jsonl"].write_text("\n".join(json.dumps(row) for row in _ap_plan(manifest)) + "\n", encoding="utf-8")
            paths["perf.jsonl"].write_text("", encoding="utf-8")
            paths["ap.jsonl"].write_text("", encoding="utf-8")
            out = root / "out"

            completed = subprocess.run([
                sys.executable, str(SCRIPT), "--manifest-json", str(paths["manifest.json"]),
                "--ap-plan-jsonl", str(paths["plan.jsonl"]), "--performance-state-jsonl", str(paths["perf.jsonl"]),
                "--ap-state-jsonl", str(paths["ap.jsonl"]), "--output-dir", str(out),
            ], text=True, capture_output=True, check=False)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            json_rows = json.loads((out / "gold96_final.json").read_text(encoding="utf-8"))
            jsonl_lines = (out / "gold96_final.jsonl").read_text(encoding="utf-8").splitlines()
            with (out / "gold96_final.csv").open(encoding="utf-8", newline="") as handle:
                csv_rows = list(csv.DictReader(handle))
            audit = json.loads((out / "gold96_audit.json").read_text(encoding="utf-8"))

        self.assertEqual(len(json_rows), 96)
        self.assertEqual(len(jsonl_lines), 96)
        self.assertEqual(len(csv_rows), 96)
        self.assertEqual(len(audit["groups"]), 24)
        self.assertEqual(audit["summary"]["pending"], 96)


if __name__ == "__main__":
    unittest.main()
