from __future__ import annotations

import csv
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


def _latency_row(run_id: str) -> dict[str, object]:
    return {
        "schema": "latency_lut_row_v1",
        "row_id": f"latency_lut:cfg_a:{run_id}",
        "config_id": "cfg_a",
        "model": "pyramid_lidar",
        "candidate_id": "coverage:pyramid_lidar:w32x96x192:fp16",
        "software_point_id": "pyramid_lidar:backbone:w32x96x192:fp16",
        "dense_stage": "backbone",
        "optimized_scope": "backbone_only",
        "width": [32, 96, 192],
        "quant_policy": "fp16",
        "schedule_policy": "metaschedule_tuned",
        "backend": "h800_tvm",
        "measurement_status": "measured",
        "run_id": run_id,
        "created_at": "2026-06-26T00:00:00Z",
    }


class Stage2SupervisorPollTest(unittest.TestCase):
    def test_cli_uses_latest_job_state_and_recommends_actions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            latency_path = tmp_path / "latency.jsonl"
            ap_path = tmp_path / "ap.jsonl"
            energy_path = tmp_path / "energy.jsonl"
            plan_path = tmp_path / "latency_job_plan.jsonl"
            state_path = tmp_path / "latency_job_state.jsonl"
            gap_path = tmp_path / "axis_gap_report.json"
            quarantine_path = tmp_path / "quarantine.jsonl"
            missing_artifact_path = tmp_path / "missing_artifacts.jsonl"
            previous_quarantine_path = tmp_path / "previous_quarantine.jsonl"
            out_dir = tmp_path / "exports"

            _write_jsonl(
                latency_path,
                [_latency_row(f"latency_repeat_{index}") for index in range(5)],
            )
            _write_jsonl(ap_path, [])
            _write_jsonl(energy_path, [])
            _write_jsonl(
                plan_path,
                [
                    {"job_id": f"latency:queued_{index}", "lut_kind": "latency"}
                    for index in range(7)
                ]
                + [
                    {"job_id": "latency:dup", "lut_kind": "latency"},
                    {"job_id": "latency:block", "lut_kind": "latency"},
                ],
            )
            _write_jsonl(
                state_path,
                [
                    {
                        "schema": "lut_job_state_row_v1",
                        "job_id": "latency:dup",
                        "status": "running",
                        "attempt": 1,
                    },
                    {
                        "schema": "lut_job_state_row_v1",
                        "job_id": "latency:dup",
                        "status": "succeeded",
                        "attempt": 1,
                    },
                    {
                        "schema": "lut_job_state_row_v1",
                        "job_id": "latency:block",
                        "status": "queued",
                        "attempt": 1,
                    },
                    {
                        "schema": "lut_job_state_row_v1",
                        "job_id": "latency:block",
                        "status": "preflight_blocked",
                        "attempt": 1,
                    },
                ],
            )
            gap_path.write_text(
                json.dumps(
                    {
                        "schema": "stage2_axis_gap_report_v1",
                        "rows": [
                            {
                                "candidate_id": "coverage:pyramid_lidar:a",
                                "ap_status": "blocked",
                                "energy_status": "queued",
                                "next_action": "block_ap_no_claim_missing_true_source",
                            },
                            {
                                "candidate_id": "coverage:pyramid_lidar:b",
                                "ap_status": "blocked",
                                "energy_status": "blocked",
                                "next_action": "fix_missing_artifact",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            _write_jsonl(
                previous_quarantine_path,
                [
                    {
                        "schema": "lut_bad_db_quarantine_row_v1",
                        "job_id": "latency:old",
                        "config_id": "cfg_old",
                        "model": "pyramid_lidar",
                        "lut_kind": "latency",
                        "job_type": "generate_latency_lut",
                        "status": "active",
                        "failure_reason": "old bad db",
                        "created_at": "2026-06-25T00:00:00Z",
                    }
                ],
            )
            _write_jsonl(
                quarantine_path,
                [
                    {
                        "schema": "lut_bad_db_quarantine_row_v1",
                        "job_id": "latency:old",
                        "config_id": "cfg_old",
                        "model": "pyramid_lidar",
                        "lut_kind": "latency",
                        "job_type": "generate_latency_lut",
                        "status": "active",
                        "failure_reason": "old bad db",
                        "created_at": "2026-06-25T00:00:00Z",
                    },
                    {
                        "schema": "lut_bad_db_quarantine_row_v1",
                        "job_id": "energy:new",
                        "config_id": "cfg_new",
                        "model": "pyramid_lidar",
                        "lut_kind": "energy",
                        "job_type": "generate_energy_lut",
                        "status": "active",
                        "failure_reason": "CUDA_ERROR_ILLEGAL_ADDRESS",
                        "created_at": "2026-06-26T00:00:00Z",
                    },
                ],
            )
            _write_jsonl(
                missing_artifact_path,
                [
                    {
                        "schema": "stage2_missing_artifact_v1",
                        "candidate_id": "coverage:pyramid_lidar:missing",
                        "config_id": "cfg_missing",
                        "missing_artifacts": ["onnx_path"],
                        "failure_reason": "missing_artifact",
                        "created_at": "2026-06-26T00:00:00Z",
                    }
                ],
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/stage2_supervisor_poll.py"),
                    "--latency-rows",
                    str(latency_path),
                    "--ap-rows",
                    str(ap_path),
                    "--energy-rows",
                    str(energy_path),
                    "--job-plan",
                    str(plan_path),
                    "--job-state",
                    str(state_path),
                    "--axis-gap-report",
                    str(gap_path),
                    "--quarantine-rows",
                    str(quarantine_path),
                    "--missing-artifact-rows",
                    str(missing_artifact_path),
                    "--previous-quarantine-rows",
                    str(previous_quarantine_path),
                    "--out-dir",
                    str(out_dir),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            gate = json.loads(
                (out_dir / "readiness_gate_latest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(gate["schema"], "stage2_supervisor_readiness_gate_v1")
            self.assertEqual(gate["decision"], "NO_GO")
            self.assertEqual(gate["latest_job_status"]["latency:dup"]["status"], "succeeded")
            self.assertNotIn("running", gate["latest_job_status_counts"])
            self.assertEqual(gate["latest_job_status_counts"]["preflight_blocked"], 1)
            self.assertEqual(gate["queue"]["latency_ready_jobs"], 7)
            self.assertAlmostEqual(gate["coverage_summary"]["repeat_ratio"], 4 / 5)
            self.assertEqual(gate["quarantine"]["growth"], 1)
            self.assertEqual(gate["coverage_summary"]["missing_artifact_config_count"], 1)
            self.assertEqual(gate["coverage_summary"]["blocked_config_count"], 3)
            self.assertEqual(
                gate["top_missing_axes"][0],
                {
                    "axis": "ap",
                    "status": "blocked",
                    "count": 2,
                    "examples": [
                        "coverage:pyramid_lidar:a",
                        "coverage:pyramid_lidar:b",
                    ],
                },
            )

            codes = {item["code"] for item in gate["recommendations"]}
            self.assertEqual(
                codes,
                {
                    "repeat_ratio_high",
                    "latency_queue_low",
                    "energy_axis_lag",
                    "ap_axis_lag",
                    "quarantine_growth",
                },
            )

            with (out_dir / "coverage_dashboard_latest.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                metrics = {row["metric"]: row["value"] for row in csv.DictReader(handle)}
            self.assertEqual(metrics["unique_latency_cells"], "1")
            self.assertEqual(metrics["latency_ready_jobs"], "7")
            self.assertEqual(metrics["blocked_config_count"], "3")

            report = (out_dir / "supervisor_report_latest.md").read_text(encoding="utf-8")
            self.assertIn("stop repeat", report)
            self.assertIn("prioritize energy subset", report)
            self.assertIn("pause related workdir/template", report)
            self.assertIn("Top Missing Axes", report)


if __name__ == "__main__":
    unittest.main()
