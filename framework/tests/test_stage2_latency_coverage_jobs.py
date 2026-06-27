from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/stage2_generate_latency_coverage_jobs.py"
CREATED_AT = "2026-06-26T00:00:00Z"


def _load_job_module():
    spec = importlib.util.spec_from_file_location(
        "stage2_generate_latency_coverage_jobs",
        SCRIPT,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _config_id(width: list[int], schedule_policy: str) -> str:
    return (
        f"coverage_h800_tvm_pyramid_w{width[0]}x{width[1]}x{width[2]}"
        f"_fp16_{schedule_policy}"
    )


def _candidate(label: str, width: list[int], *, priority: int = 80) -> dict[str, object]:
    return {
        "schema": "stage2_candidate_row_v1",
        "candidate_id": f"coverage:pyramid_lidar:w{width[0]}x{width[1]}x{width[2]}:fp16",
        "model": "Pyramid-LiDAR",
        "label": label,
        "width": width,
        "arm": "P/S",
        "quant_policy": "fp16",
        "schedule_policy": "metaschedule_tuned",
        "optimized_scope": "backbone_only",
        "priority": priority,
        "axes_required": ["latency", "energy", "ap"],
        "repeat_policy": "coverage",
        "max_latency_repeats": 1,
        "max_energy_repeats": 1,
        "ap_policy": "true_eval_or_true_import_only",
        "created_at": CREATED_AT,
        "dense_stage": "backbone",
        "software_point_id": (
            f"pyramid_lidar:backbone:w{width[0]}x{width[1]}x{width[2]}:fp16"
        ),
        "config_id_default": _config_id(width, "default"),
        "config_id_tuned": _config_id(width, "metaschedule_tuned"),
    }


def _artifact(candidate: dict[str, object], status: str = "ready") -> dict[str, object]:
    label = str(candidate["label"])
    return {
        "schema": "stage2_artifact_state_v1",
        "candidate_id": candidate["candidate_id"],
        "label": label,
        "width": candidate["width"],
        "artifact_status": status,
        "onnx_path": f"/exdata/jichengzhi/s2_tvm/models/{label}_backbone.onnx",
        "tvm_work_dir": f"/exdata/jichengzhi/s2_tvm/ms_work_{label}",
        "database_workload_path": (
            f"/exdata/jichengzhi/s2_tvm/ms_work_{label}/database_workload.json"
        ),
        "database_tuning_record_path": (
            f"/exdata/jichengzhi/s2_tvm/ms_work_{label}/database_tuning_record.json"
        ),
        "updated_at": CREATED_AT,
    }


def _measured_latency(candidate: dict[str, object], schedule_policy: str) -> dict[str, object]:
    width = list(candidate["width"])  # type: ignore[arg-type]
    return {
        "schema": "latency_lut_row_v1",
        "config_id": _config_id(width, schedule_policy),
        "candidate_id": candidate["candidate_id"],
        "width": width,
        "quant_policy": "fp16",
        "schedule_policy": schedule_policy,
        "optimized_scope": "backbone_only",
        "measurement_status": "measured",
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


class Stage2LatencyCoverageJobGeneratorTest(unittest.TestCase):
    def test_generates_only_ready_missing_latency_cells_across_requested_gpus(self):
        generator = _load_job_module()
        ready_partial = _candidate("frontier_01", [32, 96, 192], priority=90)
        missing = _candidate("missing_onnx", [64, 32, 256], priority=89)
        ready_empty = _candidate("frontier_02", [40, 80, 160], priority=88)

        jobs_by_gpu = generator.generate_latency_coverage_jobs(
            candidates=[ready_partial, missing, ready_empty],
            artifact_rows=[
                _artifact(ready_partial),
                _artifact(missing, status="missing"),
                _artifact(ready_empty),
            ],
            latency_rows=[_measured_latency(ready_partial, "default")],
            gpus=[0, 1, 2, 4, 5],
            created_at=CREATED_AT,
            phase="coverage_pipeline_v1",
            tag="coverage",
            rows_out_jsonl="rows/latency_lut_rows_v1.jsonl",
            raw_root="raw/latency",
            manifest_path="manifest_placeholder.json",
            registry_path="artifacts/artifact_registry_v1.jsonl",
        )

        self.assertEqual(set(jobs_by_gpu), {"0", "1", "2", "4", "5"})
        jobs = [job for queue in jobs_by_gpu.values() for job in queue]
        self.assertEqual(len(jobs), 3)
        self.assertNotIn(
            ("frontier_01", "default"),
            {(job["label"], job["schedule_policy"]) for job in jobs},
        )
        self.assertIn(
            ("frontier_01", "metaschedule_tuned"),
            {(job["label"], job["schedule_policy"]) for job in jobs},
        )
        self.assertEqual({job["label"] for job in jobs}, {"frontier_01", "frontier_02"})
        self.assertEqual({job["schema"] for job in jobs}, {"lut_job_plan_row_v1"})
        self.assertEqual({job["job_type"] for job in jobs}, {"generate_latency_lut"})
        self.assertEqual({job["lut_kind"] for job in jobs}, {"latency"})
        self.assertTrue(
            all(job["resource"]["gpu"] in {0, 1, 2, 4, 5} for job in jobs)
        )
        self.assertTrue(all(job["resource"]["serial_queue"] is True for job in jobs))
        self.assertTrue(
            all("scripts/stage2_generate_latency_lut.py" in job["command"] for job in jobs)
        )
        self.assertNotIn("password", json.dumps(jobs).lower())

    def test_cli_retest_tag_writes_per_gpu_queues_even_for_measured_cells(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            candidate = _candidate("paper_retest_target", [56, 112, 224])
            candidates_path = tmp_path / "candidates.jsonl"
            artifact_path = tmp_path / "artifact_state.jsonl"
            latency_path = tmp_path / "latency_rows.jsonl"
            out_dir = tmp_path / "jobs"
            rows_out = tmp_path / "rows/latency_lut_rows_v1.jsonl"
            raw_root = tmp_path / "raw/latency"
            _write_jsonl(candidates_path, [candidate])
            _write_jsonl(artifact_path, [_artifact(candidate)])
            _write_jsonl(
                latency_path,
                [
                    _measured_latency(candidate, "default"),
                    _measured_latency(candidate, "metaschedule_tuned"),
                ],
            )

            proc = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--candidates",
                    str(candidates_path),
                    "--artifact-state",
                    str(artifact_path),
                    "--latency-rows",
                    str(latency_path),
                    "--out-dir",
                    str(out_dir),
                    "--rows-out-jsonl",
                    str(rows_out),
                    "--raw-root",
                    str(raw_root),
                    "--gpus",
                    "0,1,2,4,5",
                    "--tag",
                    "paper_retest",
                    "--created-at",
                    CREATED_AT,
                ],
                cwd=ROOT,
                env={"PYTHONPATH": str(ROOT)},
                capture_output=True,
                text=True,
            )

            self.assertEqual(proc.returncode, 0, proc.stderr)
            queue_paths = {
                gpu: out_dir / f"latency_job_queue_gpu{gpu}.jsonl"
                for gpu in ("0", "1", "2", "4", "5")
            }
            self.assertTrue(all(path.exists() for path in queue_paths.values()))
            jobs = []
            for path in queue_paths.values():
                jobs.extend(
                    json.loads(line)
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                )
            self.assertEqual(len(jobs), 2)
            self.assertEqual({job["tag"] for job in jobs}, {"paper_retest"})
            self.assertEqual(
                {job["schedule_policy"] for job in jobs},
                {"default", "metaschedule_tuned"},
            )
            self.assertEqual({job["resource"]["gpu"] for job in jobs}, {0, 1})


if __name__ == "__main__":
    unittest.main()
