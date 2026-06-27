import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _candidate(
    *,
    candidate_id: str,
    label: str,
    onnx_path: str,
    tvm_work_dir: str,
    ap_source_path: str,
) -> dict[str, object]:
    return {
        "schema": "stage2_candidate_row_v1",
        "candidate_id": candidate_id,
        "config_id": f"cfg_{label}",
        "model": "Pyramid-LiDAR",
        "label": label,
        "width": [32, 96, 192],
        "arm": "P",
        "quant_policy": "fp16",
        "schedule_policy": "metaschedule_tuned",
        "optimized_scope": "backbone_only",
        "priority": 80,
        "axes_required": ["latency", "energy", "ap"],
        "repeat_policy": "coverage",
        "max_latency_repeats": 1,
        "max_energy_repeats": 1,
        "ap_policy": "true_eval_or_true_import_only",
        "onnx_path": onnx_path,
        "tvm_work_dir": tvm_work_dir,
        "ap_source_path": ap_source_path,
        "created_at": "2026-06-26T00:00:00Z",
    }


def _make_latency_energy_artifacts(base_path: Path, *, onnx_path: str, tvm_work_dir: str) -> None:
    (base_path / onnx_path).parent.mkdir(parents=True, exist_ok=True)
    (base_path / onnx_path).write_bytes(b"onnx")
    work_dir = base_path / tvm_work_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "database_workload.json").write_text("{}\n", encoding="utf-8")
    (work_dir / "database_tuning_record.json").write_text("{}\n", encoding="utf-8")


def _make_ap_source(base_path: Path, ap_source_path: str) -> None:
    path = base_path / ap_source_path
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(path, [{"schema": "ap_anchor_row_v1", "config_id": "cfg_base"}])


class Stage2ArtifactTaskPlannerTest(unittest.TestCase):
    def test_cli_plans_artifact_tasks_and_keeps_missing_or_quarantined_out_of_measurement_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            base_path = tmp_path / "artifact_root"
            ready = _candidate(
                candidate_id="coverage:pyramid_lidar:base",
                label="base",
                onnx_path="models/base.onnx",
                tvm_work_dir="workdirs/base",
                ap_source_path="ap/base_ap.jsonl",
            )
            missing = _candidate(
                candidate_id="coverage:pyramid_lidar:p50b2_136",
                label="p50b2_136",
                onnx_path="models/p50b2_136.onnx",
                tvm_work_dir="workdirs/p50b2_136",
                ap_source_path="ap/p50b2_136_ap.jsonl",
            )
            quarantined = _candidate(
                candidate_id="coverage:pyramid_lidar:mix_a/pad64/s1_64",
                label="mix_a",
                onnx_path="models/mix_a.onnx",
                tvm_work_dir="workdirs/mix_a",
                ap_source_path="ap/mix_a_ap.jsonl",
            )

            _make_latency_energy_artifacts(
                base_path,
                onnx_path=str(ready["onnx_path"]),
                tvm_work_dir=str(ready["tvm_work_dir"]),
            )
            _make_ap_source(base_path, str(ready["ap_source_path"]))
            _make_latency_energy_artifacts(
                base_path,
                onnx_path=str(missing["onnx_path"]),
                tvm_work_dir=str(missing["tvm_work_dir"]),
            )
            (base_path / str(missing["onnx_path"])).unlink()
            _make_ap_source(base_path, str(missing["ap_source_path"]))
            _make_latency_energy_artifacts(
                base_path,
                onnx_path=str(quarantined["onnx_path"]),
                tvm_work_dir=str(quarantined["tvm_work_dir"]),
            )
            _make_ap_source(base_path, str(quarantined["ap_source_path"]))

            candidate_queue = tmp_path / "candidate_queue.jsonl"
            quarantine_file = tmp_path / "bad_db_quarantine_v1.jsonl"
            artifact_tasks = tmp_path / "artifact_tasks.jsonl"
            artifact_state = tmp_path / "artifact_state.jsonl"
            artifact_registry = tmp_path / "artifact_registry_v1.jsonl"
            _write_jsonl(candidate_queue, [ready, missing, quarantined])
            _write_jsonl(
                quarantine_file,
                [
                    {
                        "schema": "lut_bad_db_quarantine_row_v1",
                        "job_id": "latency:cfg_mix_a",
                        "config_id": "cfg_mix_a",
                        "model": "Pyramid-LiDAR",
                        "lut_kind": "latency",
                        "job_type": "generate_latency_lut",
                        "status": "active",
                        "failure_reason": "CUDA illegal memory access",
                        "created_at": "2026-06-26T01:00:00Z",
                    }
                ],
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/stage2_plan_artifact_tasks.py"),
                    "--candidate-queue",
                    str(candidate_queue),
                    "--quarantine-file",
                    str(quarantine_file),
                    "--artifact-tasks-out",
                    str(artifact_tasks),
                    "--artifact-state-out",
                    str(artifact_state),
                    "--artifact-registry-out",
                    str(artifact_registry),
                    "--artifact-root",
                    str(base_path),
                    "--created-at",
                    "2026-06-26T02:00:00Z",
                ],
                cwd=ROOT,
                env={"PYTHONPATH": str(ROOT)},
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            tasks_by_candidate = {row["candidate_id"]: row for row in _read_jsonl(artifact_tasks)}
            state_by_candidate = {row["candidate_id"]: row for row in _read_jsonl(artifact_state)}
            registry_by_candidate = {
                row["candidate_id"]: row for row in _read_jsonl(artifact_registry)
            }

            ready_task = tasks_by_candidate["coverage:pyramid_lidar:base"]
            self.assertEqual(ready_task["schema"], "stage2_artifact_task_v1")
            self.assertEqual(ready_task["artifact_status"], "ready")
            self.assertEqual(ready_task["measurement_jobs"], ["latency", "energy", "ap"])
            self.assertEqual(
                state_by_candidate["coverage:pyramid_lidar:base"]["artifact_status"],
                "ready",
            )
            self.assertEqual(
                registry_by_candidate["coverage:pyramid_lidar:base"]["artifact_status"],
                "ready",
            )

            missing_task = tasks_by_candidate["coverage:pyramid_lidar:p50b2_136"]
            self.assertEqual(missing_task["artifact_status"], "missing")
            self.assertEqual(missing_task["measurement_jobs"], [])
            self.assertIn("onnx_path", missing_task["missing_artifacts"])
            self.assertEqual(
                state_by_candidate["coverage:pyramid_lidar:p50b2_136"]["artifact_status"],
                "missing",
            )
            self.assertEqual(
                registry_by_candidate["coverage:pyramid_lidar:p50b2_136"]["artifact_status"],
                "missing",
            )

            quarantined_task = tasks_by_candidate[
                "coverage:pyramid_lidar:mix_a/pad64/s1_64"
            ]
            self.assertEqual(quarantined_task["artifact_status"], "quarantined")
            self.assertEqual(quarantined_task["measurement_jobs"], [])
            self.assertEqual(quarantined_task["quarantine_refs"], ["latency:cfg_mix_a"])
            self.assertEqual(
                state_by_candidate[
                    "coverage:pyramid_lidar:mix_a/pad64/s1_64"
                ]["artifact_status"],
                "quarantined",
            )
            self.assertEqual(
                registry_by_candidate[
                    "coverage:pyramid_lidar:mix_a/pad64/s1_64"
                ]["artifact_status"],
                "quarantined",
            )

    def test_cli_treats_missing_ap_source_as_ap_axis_only_blocker(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            base_path = tmp_path / "artifact_root"
            candidate = _candidate(
                candidate_id="coverage:pyramid_lidar:no_ap",
                label="no_ap",
                onnx_path="models/no_ap.onnx",
                tvm_work_dir="workdirs/no_ap",
                ap_source_path="ap/no_ap_rows.jsonl",
            )
            _make_latency_energy_artifacts(
                base_path,
                onnx_path=str(candidate["onnx_path"]),
                tvm_work_dir=str(candidate["tvm_work_dir"]),
            )
            candidate_queue = tmp_path / "candidate_queue.jsonl"
            artifact_tasks = tmp_path / "artifact_tasks.jsonl"
            artifact_state = tmp_path / "artifact_state.jsonl"
            artifact_registry = tmp_path / "artifact_registry_v1.jsonl"
            _write_jsonl(candidate_queue, [candidate])

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/stage2_plan_artifact_tasks.py"),
                    "--candidate-queue",
                    str(candidate_queue),
                    "--artifact-tasks-out",
                    str(artifact_tasks),
                    "--artifact-state-out",
                    str(artifact_state),
                    "--artifact-registry-out",
                    str(artifact_registry),
                    "--base-path",
                    str(base_path),
                    "--created-at",
                    "2026-06-26T02:00:00Z",
                ],
                cwd=ROOT,
                env={"PYTHONPATH": str(ROOT)},
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            task = _read_jsonl(artifact_tasks)[0]
            state = _read_jsonl(artifact_state)[0]
            registry = _read_jsonl(artifact_registry)[0]

            self.assertEqual(task["artifact_status"], "ready")
            self.assertEqual(task["ap_status"], "source_missing")
            self.assertEqual(task["measurement_jobs"], ["latency", "energy"])
            self.assertEqual(task["missing_artifacts"], [])
            self.assertEqual(state["artifact_status"], "ready")
            self.assertEqual(state["ap_status"], "source_missing")
            self.assertEqual(registry["artifact_status"], "ready")
            self.assertEqual(registry["ap_status"], "source_missing")

    def test_cli_quarantines_generated_candidate_when_any_config_id_is_quarantined(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            base_path = tmp_path / "artifact_root"
            candidate = {
                "schema": "stage2_candidate_row_v1",
                "candidate_id": "coverage:pyramid_lidar:w40x80x160:fp16",
                "model": "Pyramid-LiDAR",
                "label": "lhc_00",
                "width": [40, 80, 160],
                "arm": "P/S",
                "quant_policy": "fp16",
                "schedule_policy": "metaschedule_tuned",
                "optimized_scope": "backbone_only",
                "priority": 80,
                "axes_required": ["latency", "energy", "ap"],
                "repeat_policy": "coverage",
                "max_latency_repeats": 1,
                "max_energy_repeats": 1,
                "ap_policy": "true_eval_or_true_import_only",
                "config_id_default": "coverage_h800_tvm_pyramid_w40x80x160_fp16_default",
                "config_id_tuned": "coverage_h800_tvm_pyramid_w40x80x160_fp16_metaschedule_tuned",
                "config_ids": [
                    "coverage_h800_tvm_pyramid_w40x80x160_fp16_default",
                    "coverage_h800_tvm_pyramid_w40x80x160_fp16_metaschedule_tuned",
                ],
                "onnx_path": "models/lhc_00.onnx",
                "tvm_work_dir": "workdirs/lhc_00",
                "ap_source_path": "ap/lhc_00_ap.jsonl",
                "created_at": "2026-06-26T00:00:00Z",
            }
            _make_latency_energy_artifacts(
                base_path,
                onnx_path=str(candidate["onnx_path"]),
                tvm_work_dir=str(candidate["tvm_work_dir"]),
            )
            _make_ap_source(base_path, str(candidate["ap_source_path"]))
            candidate_queue = tmp_path / "candidate_queue.jsonl"
            quarantine_file = tmp_path / "bad_db_quarantine_v1.jsonl"
            artifact_tasks = tmp_path / "artifact_tasks.jsonl"
            artifact_state = tmp_path / "artifact_state.jsonl"
            artifact_registry = tmp_path / "artifact_registry_v1.jsonl"
            _write_jsonl(candidate_queue, [candidate])
            _write_jsonl(
                quarantine_file,
                [
                    {
                        "schema": "lut_bad_db_quarantine_row_v1",
                        "job_id": "latency:lhc_00:tuned",
                        "config_id": "coverage_h800_tvm_pyramid_w40x80x160_fp16_metaschedule_tuned",
                        "model": "Pyramid-LiDAR",
                        "lut_kind": "latency",
                        "job_type": "generate_latency_lut",
                        "status": "active",
                        "failure_reason": "CUDA illegal memory access",
                        "created_at": "2026-06-26T01:00:00Z",
                    }
                ],
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/stage2_plan_artifact_tasks.py"),
                    "--candidate-queue",
                    str(candidate_queue),
                    "--quarantine-file",
                    str(quarantine_file),
                    "--artifact-tasks-out",
                    str(artifact_tasks),
                    "--artifact-state-out",
                    str(artifact_state),
                    "--artifact-registry-out",
                    str(artifact_registry),
                    "--artifact-root",
                    str(base_path),
                    "--created-at",
                    "2026-06-26T02:00:00Z",
                ],
                cwd=ROOT,
                env={"PYTHONPATH": str(ROOT)},
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            task = _read_jsonl(artifact_tasks)[0]
            self.assertEqual(task["artifact_status"], "quarantined")
            self.assertEqual(task["measurement_jobs"], [])
            self.assertEqual(task["quarantine_refs"], ["latency:lhc_00:tuned"])


if __name__ == "__main__":
    unittest.main()
