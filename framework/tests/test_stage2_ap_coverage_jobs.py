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
SCRIPT = ROOT / "scripts/stage2_generate_ap_coverage_jobs.py"


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


class Stage2ApCoverageJobGeneratorTest(unittest.TestCase):
    def _run_generator(
        self,
        *,
        candidate_queue: Path,
        artifact_registry: Path,
        out_dir: Path,
        ap_rows: Path | None = None,
        allow_repeat: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        args = [
            sys.executable,
            str(SCRIPT),
            "--candidate-queue",
            str(candidate_queue),
            "--artifact-registry",
            str(artifact_registry),
            "--out-dir",
            str(out_dir),
            "--run-id",
            "unit_run",
        ]
        if ap_rows is not None:
            args.extend(["--ap-rows", str(ap_rows)])
        if allow_repeat:
            args.append("--allow-repeat")
        return subprocess.run(
            args,
            cwd=ROOT,
            env={"PYTHONPATH": str(ROOT)},
            capture_output=True,
            text=True,
        )

    def test_only_true_ready_sources_are_queued_and_blocked_candidates_are_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            true_source = tmp_path / "true_ap_source.jsonl"
            true_source.write_text('{"metric": "AP70"}\n', encoding="utf-8")
            missing_source = tmp_path / "missing_ap_source.jsonl"
            predicted_source = tmp_path / "predicted_ap_source.jsonl"
            predicted_source.write_text('{"metric_value": 0.62}\n', encoding="utf-8")

            candidates = [
                {
                    "schema": "stage2_candidate_row_v1",
                    "candidate_id": "coverage:pyramid_lidar:mix_a",
                    "config_id": "cfg_mix_a",
                    "model": "Pyramid-LiDAR",
                    "label": "mix_a",
                    "width": [32, 96, 192],
                    "axes_required": ["latency", "ap"],
                    "ap_policy": "true_eval_or_true_import_only",
                },
                {
                    "schema": "stage2_candidate_row_v1",
                    "candidate_id": "coverage:pyramid_lidar:mix_c",
                    "config_id": "cfg_mix_c",
                    "model": "Pyramid-LiDAR",
                    "label": "mix_c",
                    "width": [16, 128, 128],
                    "axes_required": ["latency", "ap"],
                    "ap_policy": "true_eval_or_true_import_only",
                },
                {
                    "schema": "stage2_candidate_row_v1",
                    "candidate_id": "coverage:pyramid_lidar:fake_ap",
                    "config_id": "cfg_fake",
                    "model": "Pyramid-LiDAR",
                    "label": "fake_ap",
                    "width": [64, 64, 64],
                    "axes_required": ["ap"],
                    "ap_policy": "true_eval_or_true_import_only",
                },
            ]
            artifacts = [
                {
                    "schema_version": "stage2_artifact_registry_v1",
                    "candidate_id": "coverage:pyramid_lidar:mix_a",
                    "config_id": "cfg_mix_a",
                    "model_name": "Pyramid-LiDAR",
                    "label": "mix_a",
                    "width": [32, 96, 192],
                    "artifact_status": "ready",
                    "quality_status": "ready",
                    "ap_source_kind": "true_import",
                    "ap_source_path": str(true_source),
                    "dataset": "DAIR-V2X",
                    "split": "val_1789",
                    "ckpt": "ckpts/mix_a.pth",
                    "protocol": "b2_mixed_epoch31",
                },
                {
                    "schema_version": "stage2_artifact_registry_v1",
                    "candidate_id": "coverage:pyramid_lidar:mix_c",
                    "config_id": "cfg_mix_c",
                    "model_name": "Pyramid-LiDAR",
                    "label": "mix_c",
                    "width": [16, 128, 128],
                    "artifact_status": "ready",
                    "quality_status": "ready",
                    "ap_source_kind": "true_import",
                    "ap_source_path": str(missing_source),
                    "dataset": "DAIR-V2X",
                    "split": "val_1789",
                    "ckpt": "ckpts/mix_c.pth",
                    "protocol": "b2_mixed_epoch31",
                },
                {
                    "schema_version": "stage2_artifact_registry_v1",
                    "candidate_id": "coverage:pyramid_lidar:fake_ap",
                    "config_id": "cfg_fake",
                    "model_name": "Pyramid-LiDAR",
                    "label": "fake_ap",
                    "width": [64, 64, 64],
                    "artifact_status": "ready",
                    "quality_status": "ready",
                    "ap_source_kind": "predicted",
                    "ap_source_path": str(predicted_source),
                    "dataset": "DAIR-V2X",
                    "split": "val_1789",
                    "ckpt": "ckpts/fake.pth",
                    "protocol": "ap70_model_prediction",
                },
            ]
            candidate_queue = tmp_path / "candidates/candidate_queue.jsonl"
            artifact_registry = tmp_path / "artifacts/artifact_registry_v1.jsonl"
            out_dir = tmp_path / "jobs"
            _write_jsonl(candidate_queue, candidates)
            _write_jsonl(artifact_registry, artifacts)

            result = self._run_generator(
                candidate_queue=candidate_queue,
                artifact_registry=artifact_registry,
                out_dir=out_dir,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            jobs = _read_jsonl(out_dir / "ap_job_queue.jsonl")
            self.assertEqual(len(jobs), 1)
            job = jobs[0]
            validate_job_plan_row(job)
            self.assertEqual(job["schema"], "lut_job_plan_row_v1")
            self.assertEqual(job["job_type"], "generate_ap_lut")
            self.assertEqual(job["candidate_id"], "coverage:pyramid_lidar:mix_a")
            self.assertEqual(job["label"], "mix_a")
            self.assertEqual(job["width"], [32, 96, 192])
            self.assertEqual(job["dataset"], "DAIR-V2X")
            self.assertEqual(job["split"], "val_1789")
            self.assertEqual(job["ckpt"], "ckpts/mix_a.pth")
            self.assertEqual(job["protocol"], "b2_mixed_epoch31")
            self.assertEqual(job["source_path"], str(true_source))
            self.assertEqual(job["ap_policy"], "true_eval_or_true_import_only")
            self.assertEqual(job["run_id"], "unit_run_ap_mix_a")
            self.assertIn("scripts/stage2_generate_ap_lut.py", job["command"])
            self.assertNotEqual(job.get("measurement_status"), "measured")
            self.assertNotIn("metric_value", job)

            report_json = json.loads(
                (out_dir / "axis_gap_report.json").read_text(encoding="utf-8")
            )
            by_candidate = {
                row["candidate_id"]: row for row in report_json["rows"]
            }
            self.assertEqual(
                by_candidate["coverage:pyramid_lidar:mix_a"]["ap_status"], "queued"
            )
            self.assertEqual(
                by_candidate["coverage:pyramid_lidar:mix_c"]["ap_status"], "blocked"
            )
            self.assertEqual(
                by_candidate["coverage:pyramid_lidar:mix_c"]["claim_status"],
                "no_claim",
            )
            self.assertIn(
                "missing",
                by_candidate["coverage:pyramid_lidar:mix_c"]["reason"],
            )
            self.assertEqual(
                by_candidate["coverage:pyramid_lidar:fake_ap"]["ap_status"],
                "blocked",
            )
            self.assertIn(
                "predicted",
                by_candidate["coverage:pyramid_lidar:fake_ap"]["reason"],
            )

            with (out_dir / "axis_gap_report.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                csv_rows = list(csv.DictReader(handle))
            self.assertEqual(len(csv_rows), 3)
            self.assertIn("next_action", csv_rows[0])

    def test_existing_measured_ap_is_not_repeated_except_retest_tag_or_allow_repeat(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            true_source = tmp_path / "true_ap_source.jsonl"
            true_source.write_text('{"metric": "AP70"}\n', encoding="utf-8")

            candidates = [
                {
                    "schema": "stage2_candidate_row_v1",
                    "candidate_id": "coverage:pyramid_lidar:base",
                    "config_id": "cfg_base",
                    "model": "Pyramid-LiDAR",
                    "label": "base",
                    "width": [64, 128, 256],
                    "ap_policy": "true_eval_or_true_import_only",
                },
                {
                    "schema": "stage2_candidate_row_v1",
                    "candidate_id": "coverage:pyramid_lidar:trap25",
                    "config_id": "cfg_trap25",
                    "model": "Pyramid-LiDAR",
                    "label": "trap25",
                    "width": [48, 96, 192],
                    "ap_policy": "true_eval_or_true_import_only",
                    "tag": "paper_retest",
                },
            ]
            artifacts = [
                {
                    "schema_version": "stage2_artifact_registry_v1",
                    "candidate_id": candidate["candidate_id"],
                    "config_id": candidate["config_id"],
                    "model_name": candidate["model"],
                    "label": candidate["label"],
                    "width": candidate["width"],
                    "artifact_status": "ready",
                    "quality_status": "ready",
                    "ap_source_kind": "true_import",
                    "ap_source_path": str(true_source),
                    "dataset": "DAIR-V2X",
                    "split": "val_1789",
                    "ckpt": f"ckpts/{candidate['label']}.pth",
                    "protocol": "stage_a_true_anchor",
                }
                for candidate in candidates
            ]
            ap_rows = [
                {
                    "schema": "ap_anchor_row_v1",
                    "candidate_id": candidate["candidate_id"],
                    "config_id": candidate["config_id"],
                    "label": candidate["label"],
                    "width": candidate["width"],
                    "dataset": "DAIR-V2X",
                    "eval_split": "val_1789",
                    "ckpt_path": f"ckpts/{candidate['label']}.pth",
                    "finetune_protocol": "stage_a_true_anchor",
                    "measurement_status": "measured",
                    "metric": "AP70",
                    "metric_value": 0.63,
                    "provenance": "true AP anchor import",
                }
                for candidate in candidates
            ]
            candidate_queue = tmp_path / "candidate_queue.jsonl"
            artifact_registry = tmp_path / "artifact_registry_v1.jsonl"
            ap_rows_path = tmp_path / "ap_anchor_rows_v1.jsonl"
            _write_jsonl(candidate_queue, candidates)
            _write_jsonl(artifact_registry, artifacts)
            _write_jsonl(ap_rows_path, ap_rows)

            out_dir = tmp_path / "first"
            result = self._run_generator(
                candidate_queue=candidate_queue,
                artifact_registry=artifact_registry,
                ap_rows=ap_rows_path,
                out_dir=out_dir,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            jobs = _read_jsonl(out_dir / "ap_job_queue.jsonl")
            self.assertEqual([job["candidate_id"] for job in jobs], [candidates[1]["candidate_id"]])
            report = json.loads(
                (out_dir / "axis_gap_report.json").read_text(encoding="utf-8")
            )
            by_candidate = {row["candidate_id"]: row for row in report["rows"]}
            self.assertEqual(
                by_candidate["coverage:pyramid_lidar:base"]["ap_status"],
                "measured_exists",
            )
            self.assertEqual(
                by_candidate["coverage:pyramid_lidar:base"]["next_action"],
                "skip_repeat",
            )
            self.assertEqual(
                by_candidate["coverage:pyramid_lidar:trap25"]["ap_status"],
                "queued_repeat",
            )

            repeat_out_dir = tmp_path / "allow_repeat"
            allow_repeat = self._run_generator(
                candidate_queue=candidate_queue,
                artifact_registry=artifact_registry,
                ap_rows=ap_rows_path,
                out_dir=repeat_out_dir,
                allow_repeat=True,
            )

            self.assertEqual(allow_repeat.returncode, 0, allow_repeat.stderr)
            repeated_jobs = _read_jsonl(repeat_out_dir / "ap_job_queue.jsonl")
            self.assertEqual(
                {job["candidate_id"] for job in repeated_jobs},
                {candidate["candidate_id"] for candidate in candidates},
            )

    def test_artifact_ap_row_ids_skip_repeat_without_requiring_source_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            candidate_queue = tmp_path / "candidate_queue.jsonl"
            artifact_registry = tmp_path / "artifact_registry_v1.jsonl"
            out_dir = tmp_path / "out"
            _write_jsonl(
                candidate_queue,
                [
                    {
                        "schema": "stage2_candidate_row_v1",
                        "candidate_id": "coverage:pyramid_lidar:base",
                        "config_id": "cfg_base",
                        "model": "Pyramid-LiDAR",
                        "label": "base",
                        "width": [64, 128, 256],
                        "ap_policy": "true_eval_or_true_import_only",
                    }
                ],
            )
            _write_jsonl(
                artifact_registry,
                [
                    {
                        "schema_version": "stage2_artifact_registry_v1",
                        "candidate_id": "coverage:pyramid_lidar:base",
                        "config_id": "cfg_base",
                        "model_name": "Pyramid-LiDAR",
                        "label": "base",
                        "width": [64, 128, 256],
                        "artifact_status": "ready",
                        "quality_status": "ready",
                        "ap_source_path": None,
                        "ap_row_ids": ["ap_anchor:Pyramid-LiDAR:cfg_base:model_eval:run_001"],
                    }
                ],
            )

            result = self._run_generator(
                candidate_queue=candidate_queue,
                artifact_registry=artifact_registry,
                out_dir=out_dir,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(_read_jsonl(out_dir / "ap_job_queue.jsonl"), [])
            report = json.loads(
                (out_dir / "axis_gap_report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["rows"][0]["ap_status"], "measured_exists")
            self.assertEqual(report["rows"][0]["next_action"], "skip_repeat")

    def test_unknown_non_whitelisted_ap_source_kind_is_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "manual_cache.jsonl"
            source.write_text('{"metric_value": 0.63}\n', encoding="utf-8")
            candidate_queue = tmp_path / "candidate_queue.jsonl"
            artifact_registry = tmp_path / "artifact_registry_v1.jsonl"
            out_dir = tmp_path / "out"
            _write_jsonl(
                candidate_queue,
                [
                    {
                        "schema": "stage2_candidate_row_v1",
                        "candidate_id": "coverage:pyramid_lidar:manual",
                        "config_id": "cfg_manual",
                        "model": "Pyramid-LiDAR",
                        "label": "manual",
                        "width": [40, 80, 160],
                        "ap_policy": "true_eval_or_true_import_only",
                    }
                ],
            )
            _write_jsonl(
                artifact_registry,
                [
                    {
                        "schema_version": "stage2_artifact_registry_v1",
                        "candidate_id": "coverage:pyramid_lidar:manual",
                        "config_id": "cfg_manual",
                        "model_name": "Pyramid-LiDAR",
                        "label": "manual",
                        "width": [40, 80, 160],
                        "artifact_status": "ready",
                        "quality_status": "ready",
                        "ap_source_kind": "manual_cache",
                        "ap_source_path": str(source),
                        "dataset": "DAIR-V2X",
                        "split": "val_1789",
                        "ckpt": "ckpts/manual.pth",
                        "protocol": "stage_a_true_anchor",
                    }
                ],
            )

            result = self._run_generator(
                candidate_queue=candidate_queue,
                artifact_registry=artifact_registry,
                out_dir=out_dir,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(_read_jsonl(out_dir / "ap_job_queue.jsonl"), [])
            report = json.loads(
                (out_dir / "axis_gap_report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["rows"][0]["ap_status"], "blocked")
            self.assertIn("not an allowed true AP source kind", report["rows"][0]["reason"])


if __name__ == "__main__":
    unittest.main()
