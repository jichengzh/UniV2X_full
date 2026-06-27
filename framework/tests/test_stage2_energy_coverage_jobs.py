from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from framework.stage2.lut_productization import validate_job_plan_row


ROOT = Path(__file__).resolve().parents[2]


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _candidate(
    candidate_id: str,
    *,
    label: str,
    width: list[int],
    priority_selected: bool = False,
) -> dict[str, object]:
    row: dict[str, object] = {
        "schema": "stage2_candidate_row_v1",
        "candidate_id": candidate_id,
        "model": "pyramid_lidar",
        "label": label,
        "width": width,
        "quant_policy": "fp16",
        "schedule_policy": "metaschedule_tuned",
        "optimized_scope": "backbone_only",
        "axes_required": ["latency", "energy"],
        "priority": 80,
        "repeat_policy": "coverage",
    }
    if priority_selected:
        row["energy_priority_selected"] = True
    return row


def _artifact(candidate_id: str, status: str = "ready") -> dict[str, object]:
    safe = candidate_id.replace(":", "_")
    return {
        "schema": "stage2_artifact_state_v1",
        "candidate_id": candidate_id,
        "artifact_status": status,
        "quality_status": "ready" if status == "ready" else status,
        "onnx_path": f"/exdata/stage2/{safe}/model.onnx",
        "tvm_work_dir": f"/exdata/stage2/{safe}/tvm",
        "database_workload_path": f"/exdata/stage2/{safe}/tvm/database_workload.json",
        "database_tuning_record_path": (
            f"/exdata/stage2/{safe}/tvm/database_tuning_record.json"
        ),
        "vm_artifact_path": f"/exdata/stage2/{safe}/tvm/vm_exec.so",
        "failure_reason": None if status == "ready" else f"artifact_{status}",
    }


def _axis_row(
    schema: str,
    candidate: dict[str, object],
    *,
    backend: str,
    run_id: str,
    status: str = "measured",
) -> dict[str, object]:
    return {
        "schema": schema,
        "candidate_id": candidate["candidate_id"],
        "model": candidate["model"],
        "label": candidate["label"],
        "width": candidate["width"],
        "quant_policy": candidate["quant_policy"],
        "schedule_policy": candidate["schedule_policy"],
        "optimized_scope": candidate["optimized_scope"],
        "backend": backend,
        "measurement_status": status,
        "run_id": run_id,
    }


class Stage2EnergyCoverageJobsTest(unittest.TestCase):
    def _run_generator(
        self,
        tmp_path: Path,
        *,
        tag: str = "coverage",
        allow_repeat: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        command = [
            sys.executable,
            str(ROOT / "scripts/stage2_generate_energy_coverage_jobs.py"),
            "--candidates",
            str(tmp_path / "candidates/candidate_queue.jsonl"),
            "--artifact-registry",
            str(tmp_path / "artifacts/artifact_registry_v1.jsonl"),
            "--latency-rows",
            str(tmp_path / "rows/latency_lut_rows_v1.jsonl"),
            "--energy-rows",
            str(tmp_path / "rows/energy_lut_rows_v1.jsonl"),
            "--out-queue",
            str(tmp_path / "jobs/energy_job_queue.jsonl"),
            "--out-gap-csv",
            str(tmp_path / "exports/axis_gap_report.csv"),
            "--out-gap-json",
            str(tmp_path / "exports/axis_gap_report.json"),
            "--gpu-id",
            "2",
            "--run-id-prefix",
            "unit_energy",
            "--tag",
            tag,
        ]
        if allow_repeat:
            command.append("--allow-repeat")
        return subprocess.run(
            command,
            cwd=ROOT,
            env={"PYTHONPATH": str(ROOT)},
            capture_output=True,
            text=True,
            check=False,
        )

    def test_generates_energy_jobs_only_for_latency_success_or_priority_selected_ready_artifacts(
        self,
    ) -> None:
        latency_done = _candidate(
            "coverage:pyramid_lidar:w32x96x192:fp16:metaschedule_tuned",
            label="latency_done",
            width=[32, 96, 192],
        )
        priority_ready = _candidate(
            "coverage:pyramid_lidar:w40x112x224:fp16:metaschedule_tuned",
            label="priority_ready",
            width=[40, 112, 224],
            priority_selected=True,
        )
        duplicate_energy = _candidate(
            "coverage:pyramid_lidar:w48x128x256:fp16:metaschedule_tuned",
            label="duplicate_energy",
            width=[48, 128, 256],
        )
        waiting = _candidate(
            "coverage:pyramid_lidar:w56x144x288:fp16:metaschedule_tuned",
            label="waiting_for_latency",
            width=[56, 144, 288],
        )
        missing = _candidate(
            "coverage:pyramid_lidar:w64x160x320:fp16:metaschedule_tuned",
            label="missing_artifact",
            width=[64, 160, 320],
        )
        incomplete = _candidate(
            "coverage:pyramid_lidar:w68x168x336:fp16:metaschedule_tuned",
            label="incomplete_artifact",
            width=[68, 168, 336],
            priority_selected=True,
        )
        quarantined = _candidate(
            "coverage:pyramid_lidar:w72x176x352:fp16:metaschedule_tuned",
            label="quarantined_artifact",
            width=[72, 176, 352],
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            candidates = [
                latency_done,
                priority_ready,
                duplicate_energy,
                waiting,
                missing,
                incomplete,
                quarantined,
            ]
            incomplete_artifact = _artifact(str(incomplete["candidate_id"]))
            incomplete_artifact.pop("database_tuning_record_path")
            _write_jsonl(tmp_path / "candidates/candidate_queue.jsonl", candidates)
            _write_jsonl(
                tmp_path / "artifacts/artifact_registry_v1.jsonl",
                [
                    _artifact(str(latency_done["candidate_id"])),
                    _artifact(str(priority_ready["candidate_id"])),
                    _artifact(str(duplicate_energy["candidate_id"])),
                    _artifact(str(waiting["candidate_id"])),
                    _artifact(str(missing["candidate_id"]), "missing"),
                    incomplete_artifact,
                    _artifact(str(quarantined["candidate_id"]), "quarantined"),
                ],
            )
            _write_jsonl(
                tmp_path / "rows/latency_lut_rows_v1.jsonl",
                [
                    _axis_row(
                        "latency_lut_row_v1",
                        latency_done,
                        backend="h800_tvm",
                        run_id="latency_run_done",
                    ),
                    _axis_row(
                        "latency_lut_row_v1",
                        duplicate_energy,
                        backend="h800_tvm",
                        run_id="latency_run_duplicate",
                    ),
                ],
            )
            _write_jsonl(
                tmp_path / "rows/energy_lut_rows_v1.jsonl",
                [
                    _axis_row(
                        "energy_lut_row_v1",
                        duplicate_energy,
                        backend="h800_tvm_power_telemetry",
                        run_id="energy_run_existing",
                    ),
                ],
            )

            result = self._run_generator(tmp_path)

            self.assertEqual(result.returncode, 0, result.stderr)
            jobs = _read_jsonl(tmp_path / "jobs/energy_job_queue.jsonl")
            self.assertEqual(
                [job["candidate_id"] for job in jobs],
                [latency_done["candidate_id"], priority_ready["candidate_id"]],
            )

            latency_job = jobs[0]
            validate_job_plan_row(latency_job)
            self.assertEqual(latency_job["schema"], "lut_job_plan_row_v1")
            self.assertEqual(latency_job["job_type"], "generate_energy_lut")
            self.assertEqual(latency_job["label"], "latency_done")
            self.assertEqual(latency_job["width"], [32, 96, 192])
            self.assertEqual(latency_job["backend"], "h800_tvm_power_telemetry")
            self.assertEqual(latency_job["optimized_scope"], "backbone_only")
            self.assertEqual(latency_job["gpu_id"], 2)
            self.assertEqual(latency_job["latency_run_id"], "latency_run_done")
            self.assertTrue(str(latency_job["run_id"]).startswith("unit_energy:"))
            self.assertEqual(latency_job["quality_policy"], "energy_quality_gate_v1")
            self.assertIn("scripts/stage2_h800_run_measurement_job.py", latency_job["command"])
            self.assertEqual(
                latency_job["artifact_paths"],
                {
                    "database_tuning_record_path": (
                        "/exdata/stage2/coverage_pyramid_lidar_w32x96x192_fp16_"
                        "metaschedule_tuned/tvm/database_tuning_record.json"
                    ),
                    "database_workload_path": (
                        "/exdata/stage2/coverage_pyramid_lidar_w32x96x192_fp16_"
                        "metaschedule_tuned/tvm/database_workload.json"
                    ),
                    "onnx_path": (
                        "/exdata/stage2/coverage_pyramid_lidar_w32x96x192_fp16_"
                        "metaschedule_tuned/model.onnx"
                    ),
                    "tvm_work_dir": (
                        "/exdata/stage2/coverage_pyramid_lidar_w32x96x192_fp16_"
                        "metaschedule_tuned/tvm"
                    ),
                    "vm_artifact_path": (
                        "/exdata/stage2/coverage_pyramid_lidar_w32x96x192_fp16_"
                        "metaschedule_tuned/tvm/vm_exec.so"
                    ),
                },
            )

            priority_job = jobs[1]
            self.assertEqual(priority_job["selection_reason"], "priority_selected")
            self.assertIsNone(priority_job["latency_run_id"])

            gap_rows = json.loads(
                (tmp_path / "exports/axis_gap_report.json").read_text(encoding="utf-8")
            )
            gaps = {row["candidate_id"]: row for row in gap_rows}
            self.assertEqual(gaps[latency_done["candidate_id"]]["next_action"], "queue_energy")
            self.assertEqual(
                gaps[duplicate_energy["candidate_id"]]["next_action"],
                "skip_existing_energy_measurement",
            )
            self.assertEqual(
                gaps[waiting["candidate_id"]]["next_action"],
                "wait_for_latency_or_priority_selection",
            )
            self.assertEqual(gaps[missing["candidate_id"]]["energy_status"], "blocked")
            self.assertEqual(gaps[missing["candidate_id"]]["next_action"], "fix_missing_artifact")
            self.assertEqual(
                gaps[incomplete["candidate_id"]]["next_action"],
                "fix_missing_artifact",
            )
            self.assertIn(
                "database_tuning_record_path",
                gaps[incomplete["candidate_id"]]["gap_reason"],
            )
            self.assertEqual(
                gaps[quarantined["candidate_id"]]["next_action"],
                "resolve_quarantine",
            )

            with (tmp_path / "exports/axis_gap_report.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                csv_rows = list(csv.DictReader(handle))
            self.assertEqual(len(csv_rows), len(candidates))
            self.assertIn("latency_status", csv_rows[0])
            self.assertIn("energy_status", csv_rows[0])
            self.assertIn("next_action", csv_rows[0])

    def test_repeat_is_blocked_by_default_but_allowed_for_retest_tag_or_explicit_flag(
        self,
    ) -> None:
        candidate = _candidate(
            "coverage:pyramid_lidar:w80x192x384:fp16:metaschedule_tuned",
            label="repeat_candidate",
            width=[80, 192, 384],
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _write_jsonl(tmp_path / "candidates/candidate_queue.jsonl", [candidate])
            _write_jsonl(
                tmp_path / "artifacts/artifact_registry_v1.jsonl",
                [_artifact(str(candidate["candidate_id"]))],
            )
            _write_jsonl(
                tmp_path / "rows/latency_lut_rows_v1.jsonl",
                [
                    _axis_row(
                        "latency_lut_row_v1",
                        candidate,
                        backend="h800_tvm",
                        run_id="latency_repeat",
                    )
                ],
            )
            _write_jsonl(
                tmp_path / "rows/energy_lut_rows_v1.jsonl",
                [
                    _axis_row(
                        "energy_lut_row_v1",
                        candidate,
                        backend="h800_tvm_power_telemetry",
                        run_id="energy_existing",
                    )
                ],
            )

            default_result = self._run_generator(tmp_path)
            self.assertEqual(default_result.returncode, 0, default_result.stderr)
            self.assertEqual(_read_jsonl(tmp_path / "jobs/energy_job_queue.jsonl"), [])

            retest_result = self._run_generator(tmp_path, tag="paper_retest")
            self.assertEqual(retest_result.returncode, 0, retest_result.stderr)
            retest_jobs = _read_jsonl(tmp_path / "jobs/energy_job_queue.jsonl")
            self.assertEqual(len(retest_jobs), 1)
            self.assertEqual(retest_jobs[0]["tag"], "paper_retest")
            self.assertEqual(retest_jobs[0]["selection_reason"], "paper_retest")

            allow_result = self._run_generator(
                tmp_path,
                tag="coverage",
                allow_repeat=True,
            )
            self.assertEqual(allow_result.returncode, 0, allow_result.stderr)
            allow_jobs = _read_jsonl(tmp_path / "jobs/energy_job_queue.jsonl")
            self.assertEqual(len(allow_jobs), 1)
            self.assertEqual(allow_jobs[0]["selection_reason"], "allow_repeat")

    def test_energy_follows_candidate_schedule_instead_of_schedule_agnostic_candidate_id(self) -> None:
        candidate = _candidate(
            "coverage:pyramid_lidar:w32x96x192:fp16",
            label="schedule_guard",
            width=[32, 96, 192],
        )
        default_latency = dict(
            _axis_row(
                "latency_lut_row_v1",
                candidate,
                backend="h800_tvm",
                run_id="latency_default_only",
            )
        )
        default_latency["schedule_policy"] = "default"
        tuned_latency = dict(default_latency)
        tuned_latency["schedule_policy"] = "metaschedule_tuned"
        tuned_latency["run_id"] = "latency_tuned"
        default_energy = dict(
            _axis_row(
                "energy_lut_row_v1",
                candidate,
                backend="h800_tvm_power_telemetry",
                run_id="energy_default_only",
            )
        )
        default_energy["schedule_policy"] = "default"

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _write_jsonl(tmp_path / "candidates/candidate_queue.jsonl", [candidate])
            _write_jsonl(
                tmp_path / "artifacts/artifact_registry_v1.jsonl",
                [_artifact(str(candidate["candidate_id"]))],
            )
            _write_jsonl(tmp_path / "rows/energy_lut_rows_v1.jsonl", [default_energy])

            _write_jsonl(tmp_path / "rows/latency_lut_rows_v1.jsonl", [default_latency])
            default_only = self._run_generator(tmp_path)
            self.assertEqual(default_only.returncode, 0, default_only.stderr)
            self.assertEqual(_read_jsonl(tmp_path / "jobs/energy_job_queue.jsonl"), [])
            gap_rows = json.loads(
                (tmp_path / "exports/axis_gap_report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(gap_rows[0]["next_action"], "wait_for_latency_or_priority_selection")

            _write_jsonl(
                tmp_path / "rows/latency_lut_rows_v1.jsonl",
                [default_latency, tuned_latency],
            )
            tuned_ready = self._run_generator(tmp_path)
            self.assertEqual(tuned_ready.returncode, 0, tuned_ready.stderr)
            jobs = _read_jsonl(tmp_path / "jobs/energy_job_queue.jsonl")
            self.assertEqual(len(jobs), 1)
            self.assertEqual(jobs[0]["schedule_policy"], "metaschedule_tuned")
            self.assertEqual(jobs[0]["latency_run_id"], "latency_tuned")
            self.assertEqual(jobs[0]["latency_config_id"], tuned_latency.get("config_id"))


if __name__ == "__main__":
    unittest.main()
