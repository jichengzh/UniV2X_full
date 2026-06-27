from __future__ import annotations

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


def _lut_row(schema: str, run_id: str) -> dict[str, object]:
    kind = schema.removesuffix("_row_v1")
    row: dict[str, object] = {
        "schema": schema,
        "row_id": f"{kind}:cfg_a:{run_id}",
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
    if schema == "ap_anchor_row_v1":
        row.update(
            {
                "backend": "model_eval",
                "schedule_policy": "not_applicable",
                "metric": "AP70",
                "dataset": "DAIR-V2X",
                "eval_split": "val",
            }
        )
    if schema == "energy_lut_row_v1":
        row.update({"backend": "h800_tvm_power_telemetry"})
    return row


class Stage2LutCoverageSummaryTest(unittest.TestCase):
    def test_cli_counts_unique_axis_cells_and_repeat_heavy_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            latency_rows = [
                _lut_row("latency_lut_row_v1", f"latency_repeat_{index:02d}")
                for index in range(25)
            ]
            ap_rows = [_lut_row("ap_anchor_row_v1", "ap_once")]
            energy_rows = [_lut_row("energy_lut_row_v1", "energy_once")]
            quarantine_rows = [
                {
                    "schema": "lut_bad_db_quarantine_row_v1",
                    "job_id": "latency:cfg_a",
                    "config_id": "cfg_a",
                    "model": "pyramid_lidar",
                    "lut_kind": "latency",
                    "job_type": "generate_latency_lut",
                    "status": "active",
                    "failure_reason": "cuda illegal memory access",
                    "created_at": "2026-06-26T00:00:00Z",
                }
            ]
            missing_artifact_rows = [
                {
                    "schema": "stage2_missing_artifact_v1",
                    "candidate_id": "coverage:pyramid_lidar:missing",
                    "config_id": "cfg_missing",
                    "missing_artifacts": ["onnx_path"],
                    "failure_reason": "missing_artifact",
                    "created_at": "2026-06-26T00:00:00Z",
                }
            ]
            state_rows = [
                {
                    "schema": "lut_job_state_row_v1",
                    "job_id": "latency:cfg_blocked",
                    "status": "preflight_blocked",
                    "attempt": 1,
                }
            ]
            latency_path = tmp_path / "latency.jsonl"
            ap_path = tmp_path / "ap.jsonl"
            energy_path = tmp_path / "energy.jsonl"
            quarantine_path = tmp_path / "quarantine.jsonl"
            missing_path = tmp_path / "missing_artifacts.jsonl"
            state_path = tmp_path / "job_state.jsonl"
            out_json = tmp_path / "exports/coverage_summary_seed.json"
            _write_jsonl(latency_path, latency_rows)
            _write_jsonl(ap_path, ap_rows)
            _write_jsonl(energy_path, energy_rows)
            _write_jsonl(quarantine_path, quarantine_rows)
            _write_jsonl(missing_path, missing_artifact_rows)
            _write_jsonl(state_path, state_rows)

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/stage2_summarize_lut_coverage.py"),
                    "--latency-rows",
                    str(latency_path),
                    "--ap-rows",
                    str(ap_path),
                    "--energy-rows",
                    str(energy_path),
                    "--quarantine-rows",
                    str(quarantine_path),
                    "--missing-artifact-rows",
                    str(missing_path),
                    "--job-state",
                    str(state_path),
                    "--out-json",
                    str(out_json),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema"], "stage2_lut_coverage_summary_v1")
            self.assertEqual(payload["raw_row_count"], 27)
            self.assertEqual(payload["unique_config_count"], 3)
            self.assertEqual(payload["unique_latency_cells"], 1)
            self.assertEqual(payload["unique_energy_cells"], 1)
            self.assertEqual(payload["unique_ap_cells"], 1)
            self.assertEqual(payload["repeat_rows"], 24)
            self.assertAlmostEqual(payload["repeat_ratio"], 24 / 27)
            self.assertEqual(payload["repeat_policy_label"], "repeat-heavy")
            self.assertEqual(payload["quarantined_config_count"], 1)
            self.assertEqual(payload["missing_artifact_config_count"], 1)
            self.assertEqual(payload["blocked_config_count"], 2)
            self.assertEqual(payload["blocked_job_count"], 1)


if __name__ == "__main__":
    unittest.main()
