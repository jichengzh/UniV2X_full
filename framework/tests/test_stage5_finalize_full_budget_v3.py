from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/stage5_finalize_full_budget_v3.py"
TASKS = (
    ("S5-PYR-TVM", "pyramid", "tvm_auto"),
    ("S5-PYR-TRT", "pyramid", "trt_engine"),
    ("S5-COD-TVM", "codriving", "tvm_auto"),
    ("S5-COD-TRT", "codriving", "trt_engine"),
)


def _write_json(path: Path, payload: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity(task_id: str, model: str, dispatch: str, index: int) -> dict:
    profile = f"h800-{dispatch}-v3"
    width = [16 + index, 32 + index, 64 + index]
    q_mode = "fp16" if index % 2 == 0 else "int8"
    row_id = f"{task_id}|row-{index:02d}"
    source_plan_sha = hashlib.sha256(f"{model}:{width}".encode()).hexdigest()
    return {
        "task_id": task_id,
        "task_sha256": hashlib.sha256(task_id.encode()).hexdigest(),
        "row_id": row_id,
        "manifest_job_id": row_id,
        "group_id": f"{task_id}|group-{index:02d}",
        "model": model,
        "hardware_id": "h800",
        "capability_profile_id": profile,
        "dispatch_key": dispatch,
        "width": width,
        "q_mode": q_mode,
        "genome": [*width, q_mode],
        "source_evidence_sha256": source_plan_sha,
    }


def _build_tree(root: Path) -> tuple[Path, Path, Path, Path]:
    formal = root / "formal"
    output = root / "closure"
    coldstart_rows = root / "gold176_final.json"
    coldstart_graphs = root / "graph_features.json"
    initial_id = "gold-initial-000"
    gold = [
        {
            "manifest_job_id": initial_id,
            "terminal_status": "measured_success_gold",
            "latency_ms": 20.0,
            "energy_j": 10.0,
            "ap70": 0.5,
        },
        *[
            {
                "manifest_job_id": f"gold-{index:03d}",
                "terminal_status": "measured_success_gold",
            }
            for index in range(1, 176)
        ],
    ]
    _write_json(coldstart_rows, gold)
    _write_json(coldstart_graphs, {"graph_features": []})

    for task_id, model, dispatch in TASKS:
        rows = [_identity(task_id, model, dispatch, index) for index in range(16)]
        manifest = {
            "schema_version": "stage5_task_candidate_manifest_v2",
            "task_id": task_id,
            "task_sha256": rows[0]["task_sha256"],
            "target_model": model,
            "capability_profile_id": rows[0]["capability_profile_id"],
            "eligible_row_count": 16,
            "excluded": [{"row_id": initial_id, "reason": "already_measured"}],
            "rows": rows,
        }
        _write_json(formal / task_id / "candidate_manifest.json", manifest)
        for round_index in range(4):
            request_rows = copy.deepcopy(rows[round_index * 4 : (round_index + 1) * 4])
            request = {
                "schema_version": "stage5_measurement_request_v2",
                "task_id": task_id,
                "task_sha256": rows[0]["task_sha256"],
                "round_index": round_index,
                "batch_size": 4,
                "sample_budget": 16,
                "atomic_feedback": True,
                "rows": request_rows,
            }
            request["row_sha256"] = {
                row["row_id"]: hashlib.sha256(
                    json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
                for row in request_rows
            }
            request["measurement_request_sha256"] = hashlib.sha256(
                json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            round_dir = formal / task_id / f"round_{round_index:02d}"
            _write_json(round_dir / "measurement_request.json", request)
            feedback = []
            for offset, identity in enumerate(request_rows):
                artifact_dir = round_dir / "evidence" / str(offset)
                perf_path = artifact_dir / "performance.json"
                ap_path = artifact_dir / "ap.json"
                source_path = artifact_dir / "source.json"
                perf_sha = _write_json(
                    perf_path,
                    {
                        "status": "success",
                        "latency_ms": 10.0 + offset,
                        "energy_j": 4.0,
                    },
                )
                ap_sha = _write_json(
                    ap_path,
                    {
                        "status": "success",
                        "processed_samples": 1789,
                        "requested_num_samples": 1789,
                        "failed_samples": 0,
                        "fallback_samples": 0,
                        "ap30": 0.7 + offset / 100.0,
                        "ap50": 0.65 + offset / 100.0,
                        "ap70": 0.6 + offset / 100.0,
                    },
                )
                source_sha = _write_json(
                    source_path,
                    {
                        "status": "ready",
                        "source_plan_sha256": identity["source_evidence_sha256"],
                    },
                )
                row = {
                    **identity,
                    "terminal_status": "measured_success_gold",
                    "latency_ms": 10.0 + offset,
                    "energy_j": 4.0,
                    "ap30": 0.7 + offset / 100.0,
                    "ap50": 0.65 + offset / 100.0,
                    "ap70": 0.6 + offset / 100.0,
                    "performance_result_json": str(perf_path),
                    "performance_result_sha256": perf_sha,
                    "ap_report_path": str(ap_path),
                    "ap_report_sha256": ap_sha,
                    "materialized_source_evidence_path": str(source_path),
                    "materialized_source_evidence_sha256": source_sha,
                }
                if round_index == 3 and offset in {2, 3}:
                    failure_path = artifact_dir / "failure.json"
                    failure_sha = _write_json(failure_path, {"confirmed_failure": True})
                    row.update(
                        {
                            "terminal_status": (
                                "feasibility_failure"
                                if offset == 2
                                else "numerical_feasibility_failure"
                            ),
                            "failure_reason": "synthetic confirmed failure",
                            "failure_evidence_path": str(failure_path),
                            "failure_evidence_sha256": failure_sha,
                        }
                    )
                    row.pop("ap70")
                    if offset == 2:
                        for field in (
                            "latency_ms",
                            "energy_j",
                            "performance_result_json",
                            "performance_result_sha256",
                        ):
                            row.pop(field)
                    row["ap_report_path"] = str(failure_path)
                    row["ap_report_sha256"] = failure_sha
                feedback.append(row)
            final_dir = round_dir / "final"
            _write_json(final_dir / "stage5_feedback_v2_final.json", feedback)
            _write_json(
                final_dir / "atomic_batch_audit.json",
                {
                    "schema_version": "stage5_atomic_batch_audit_v2",
                    "feedback_released": True,
                    "batch_quarantined": False,
                    "budget_consumed": 4,
                    "released_feedback_rows": copy.deepcopy(feedback),
                },
            )
        _write_json(
            formal / task_id / "task_budget_terminal.json",
            {
                "schema_version": "stage5_task_budget_terminal_v3",
                "task_id": task_id,
                "status": "budget_exhausted",
                "budget_consumed": 16,
                "round_count": 4,
            },
        )
    _write_json(
        formal / "controller/full_budget_scheduler_terminal.json",
        {
            "schema_version": "stage5_full_budget_scheduler_terminal_v3",
            "status": "budget_exhausted",
            "task_count": 4,
            "round_count": 16,
            "formal_online_genomes": 64,
        },
    )
    return formal, output, coldstart_rows, coldstart_graphs


def _validation_audit(root: Path) -> Path:
    tasks = []
    for task_id, _, _ in TASKS:
        evidence_dir = root / "validation_evidence" / task_id
        configurations = []
        for config_index in range(4):
            config_id = f"{task_id}|row-{config_index:02d}"
            config_dir = evidence_dir / f"config_{config_index:02d}"
            repeats = []
            for repeat_index in range(3):
                repeat_id = f"repeat-{repeat_index}"
                path = config_dir / f"performance_repeat_{repeat_index}.json"
                digest = _write_json(
                    path,
                    {
                        "schema_version": "stage5_independent_performance_repeat_v1",
                        "status": "success",
                        "task_id": task_id,
                        "configuration_id": config_id,
                        "hardware_id": "h800",
                        "repeat_id": repeat_id,
                        "run_uuid": f"{task_id}-{config_index}-{repeat_index}",
                        "started_at_utc": f"2026-07-18T00:0{repeat_index}:00+00:00",
                        "ended_at_utc": f"2026-07-18T00:0{repeat_index}:10+00:00",
                        "latency_ms": 10.0 + config_index + repeat_index / 100.0,
                        "energy_j": 4.0 + repeat_index / 100.0,
                    },
                )
                repeats.append(
                    {
                        "repeat_id": repeat_id,
                        "performance_result_json": str(path),
                        "performance_result_sha256": digest,
                    }
                )
            ap_path = config_dir / "ap.json"
            source_path = config_dir / "source_evidence.json"
            configurations.append(
                {
                    "configuration_id": config_id,
                    "performance_repeats": repeats,
                    "ap_report_path": str(ap_path),
                    "ap_report_sha256": _write_json(
                        ap_path,
                        {
                            "schema_version": "stage5_independent_full_ap_v1",
                            "status": "success",
                            "task_id": task_id,
                            "configuration_id": config_id,
                            "processed_samples": 1789,
                            "requested_num_samples": 1789,
                            "failed_samples": 0,
                            "fallback_samples": 0,
                            "ap30": 0.7,
                            "ap50": 0.65,
                            "ap70": 0.6,
                        },
                    ),
                    "evidence_path": str(source_path),
                    "evidence_sha256": _write_json(
                        source_path,
                        {
                            "schema_version": "stage5_independent_source_evidence_v1",
                            "status": "ready",
                            "task_id": task_id,
                            "configuration_id": config_id,
                            "independent_from_search_measurement": True,
                        },
                    ),
                    "consistency": {
                        "passed": True,
                        "thresholds": {
                            "latency_relative": 0.15,
                            "energy_relative": 0.20,
                            "ap70_absolute": 0.01,
                            "latency_cv": 0.10,
                            "energy_cv": 0.15,
                        },
                        "deltas": {
                            "latency_relative": 0.01,
                            "energy_relative": 0.01,
                            "ap70_absolute": 0.0,
                        },
                        "rerun": {"latency_cv": 0.01, "energy_cv": 0.01},
                    },
                }
            )
        tasks.append(
            {
                "task_id": task_id,
                "passed": True,
                "configurations": configurations,
            }
        )
    path = root / "independent_validation_audit.json"
    _write_json(
        path,
        {
            "schema_version": "stage5_independent_validation_audit_v1",
            "all_tasks_passed": True,
            "tasks": tasks,
        },
    )
    return path


def _run(
    formal: Path,
    output: Path,
    rows: Path,
    graphs: Path,
    validation: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(SCRIPT),
        "--formal-root",
        str(formal),
        "--coldstart-rows-json",
        str(rows),
        "--coldstart-graph-features-json",
        str(graphs),
        "--expected-coldstart-rows-sha256",
        hashlib.sha256(rows.read_bytes()).hexdigest(),
        "--expected-coldstart-graph-features-sha256",
        hashlib.sha256(graphs.read_bytes()).hexdigest(),
        "--output-dir",
        str(output),
    ]
    if validation is not None:
        command.extend(["--independent-validation-audit", str(validation)])
    return subprocess.run(command, cwd=REPO, text=True, capture_output=True, check=False)


class Stage5FinalizeFullBudgetV3Tests(unittest.TestCase):
    def test_missing_validation_closes_engineering_but_not_phase6(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            formal, output, rows, graphs = _build_tree(Path(temporary))

            completed = _run(formal, output, rows, graphs)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            summary = json.loads(
                (output / "stage5_full_budget_closure_summary_v3.json").read_text()
            )
            self.assertTrue(summary["budget_search_closed"])
            self.assertFalse(summary["engineering_search_closed"])
            self.assertFalse(summary["phase6_entry_ready"])
            self.assertEqual(summary["formal_online_rows"], 64)
            self.assertEqual(len(summary["tasks"]), 4)
            evidence_manifest = Path(summary["formal_evidence_manifest_path"])
            self.assertTrue(evidence_manifest.is_file())
            evidence = json.loads(evidence_manifest.read_text())
            self.assertEqual(evidence["schema_version"], "stage5_evidence_manifest_v3")
            self.assertEqual(len(evidence["rounds"]), 16)
            self.assertEqual(
                evidence["verified_feedback_artifact_count"],
                summary["verified_formal_evidence_count"],
            )
            wall_clock = summary["wall_clock_audit"]
            self.assertEqual(wall_clock["round_count"], 16)
            self.assertEqual(len(wall_clock["tasks"]), 4)
            self.assertGreaterEqual(wall_clock["total_elapsed_s"], 0.0)
            for task_id, _, _ in TASKS:
                closure = output / task_id / "stage5_task_closure_audit_v3.json"
                self.assertTrue(closure.is_file())
                closure_payload = json.loads(closure.read_text())
                self.assertEqual(closure_payload["online_count"], 16)
                self.assertTrue(closure_payload["frontier_points"])

            first = (output / "stage5_full_budget_closure_summary_v3.json").read_bytes()
            rerun = _run(formal, output, rows, graphs)
            self.assertEqual(rerun.returncode, 0, rerun.stderr)
            self.assertEqual(
                first,
                (output / "stage5_full_budget_closure_summary_v3.json").read_bytes(),
            )

    def test_valid_independent_validation_enables_phase6(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            formal, output, rows, graphs = _build_tree(root)
            validation = _validation_audit(root)

            completed = _run(formal, output, rows, graphs, validation)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            summary = json.loads(
                (output / "stage5_full_budget_closure_summary_v3.json").read_text()
            )
            self.assertTrue(summary["engineering_search_closed"])
            self.assertTrue(summary["phase6_entry_ready"])
            self.assertEqual(summary["independent_validation"]["configuration_count"], 16)
            self.assertEqual(summary["independent_validation"]["performance_repeat_count"], 48)

    def test_complete_failed_validation_is_recorded_without_phase6_release(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            formal, output, rows, graphs = _build_tree(root)
            validation = _validation_audit(root)
            payload = json.loads(validation.read_text())
            payload["all_tasks_passed"] = False
            payload["tasks"][0]["passed"] = False
            consistency = payload["tasks"][0]["configurations"][0]["consistency"]
            consistency["passed"] = False
            consistency["deltas"]["latency_relative"] = 0.18
            _write_json(validation, payload)

            completed = _run(formal, output, rows, graphs, validation)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            summary = json.loads(
                (output / "stage5_full_budget_closure_summary_v3.json").read_text()
            )
            self.assertTrue(summary["budget_search_closed"])
            self.assertFalse(summary["engineering_search_closed"])
            self.assertFalse(summary["phase6_entry_ready"])
            self.assertFalse(summary["independent_validation"]["all_tasks_passed"])
            self.assertEqual(summary["independent_validation"]["configuration_count"], 16)

    def test_rejects_hash_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            formal, output, rows, graphs = _build_tree(Path(temporary))
            feedback_path = (
                formal
                / TASKS[0][0]
                / "round_00/final/stage5_feedback_v2_final.json"
            )
            feedback = json.loads(feedback_path.read_text())
            Path(feedback[0]["performance_result_json"]).write_text("drift\n", encoding="utf-8")

            completed = _run(formal, output, rows, graphs)

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("performance_result_json SHA mismatch", completed.stderr)
            self.assertFalse((output / "stage5_full_budget_closure_summary_v3.json").exists())

    def test_rejects_feedback_metric_drift_from_hashed_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            formal, output, rows, graphs = _build_tree(Path(temporary))
            final_dir = formal / TASKS[0][0] / "round_00/final"
            feedback_path = final_dir / "stage5_feedback_v2_final.json"
            audit_path = final_dir / "atomic_batch_audit.json"
            feedback = json.loads(feedback_path.read_text())
            feedback[0]["latency_ms"] = 999.0
            _write_json(feedback_path, feedback)
            audit = json.loads(audit_path.read_text())
            audit["released_feedback_rows"] = copy.deepcopy(feedback)
            _write_json(audit_path, audit)

            completed = _run(formal, output, rows, graphs)

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("feedback metric evidence mismatch", completed.stderr)
            self.assertFalse((output / "stage5_full_budget_closure_summary_v3.json").exists())

    def test_rejects_materialized_source_plan_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            formal, output, rows, graphs = _build_tree(Path(temporary))
            final_dir = formal / TASKS[0][0] / "round_00/final"
            feedback_path = final_dir / "stage5_feedback_v2_final.json"
            audit_path = final_dir / "atomic_batch_audit.json"
            feedback = json.loads(feedback_path.read_text())
            source_path = Path(feedback[0]["materialized_source_evidence_path"])
            feedback[0]["materialized_source_evidence_sha256"] = _write_json(
                source_path,
                {"status": "ready", "source_plan_sha256": "f" * 64},
            )
            _write_json(feedback_path, feedback)
            audit = json.loads(audit_path.read_text())
            audit["released_feedback_rows"] = copy.deepcopy(feedback)
            _write_json(audit_path, audit)

            completed = _run(formal, output, rows, graphs)

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("materialized source plan mismatch", completed.stderr)

    def test_rejects_placeholder_independent_validation_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            formal, output, rows, graphs = _build_tree(root)
            validation = _validation_audit(root)
            audit = json.loads(validation.read_text())
            repeat = audit["tasks"][0]["configurations"][0]["performance_repeats"][0]
            repeat_path = Path(repeat["performance_result_json"])
            repeat["performance_result_sha256"] = _write_json(
                repeat_path, {"repeat_index": 0}
            )
            _write_json(validation, audit)

            completed = _run(formal, output, rows, graphs, validation)

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("independent performance", completed.stderr)

    def test_rejects_incomplete_frozen_pareto_validation_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            formal, output, rows, graphs = _build_tree(root)
            validation = _validation_audit(root)
            audit = json.loads(validation.read_text())
            audit["tasks"][0]["configurations"].pop()
            _write_json(validation, audit)

            completed = _run(formal, output, rows, graphs, validation)

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("selection count mismatch", completed.stderr)

    def test_rejects_independent_validation_without_reproduction_consistency(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            formal, output, rows, graphs = _build_tree(root)
            validation = _validation_audit(root)
            audit = json.loads(validation.read_text())
            audit["tasks"][0]["configurations"][0]["consistency"]["passed"] = False
            _write_json(validation, audit)

            completed = _run(formal, output, rows, graphs, validation)

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("consistency contract mismatch", completed.stderr)


if __name__ == "__main__":
    unittest.main()
