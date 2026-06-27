from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from framework.stage2.artifact_registry import (
    build_artifact_registry_rows,
    validate_artifact_registry_row,
)
from framework.stage2.lut_productization import (
    ap_anchor_row,
    energy_lut_row,
    latency_lut_row,
    read_jsonl,
    write_jsonl,
)
from framework.stage2.outlier_policy import detect_latency_outliers


ROOT = Path(__file__).resolve().parents[2]


class Stage2ArtifactRegistryV1Test(unittest.TestCase):
    def _rows(self, raw_dir: Path) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
        onnx = raw_dir / "base_backbone.onnx"
        db_dir = raw_dir / "ms_work_base"
        onnx.write_text("onnx", encoding="utf-8")
        db_dir.mkdir()
        common = {
            "config_id": "calibration_h800_tvm_pyramid_base_fp16_metaschedule_tuned",
            "model": "pyramid_lidar",
            "manifest_digest": "a" * 64,
            "candidate_id": "calibration:pyramid_lidar:base",
            "software_point_id": "pyramid_lidar:backbone:w64x128x256:fp16",
            "dense_stage": "backbone",
            "optimized_scope": "backbone_only",
            "width": [64, 128, 256],
            "quant_policy": "fp16",
            "run_id": "latency_run",
            "created_at": "2026-06-25T00:00:00+08:00",
            "source_files": [str(onnx), str(db_dir)],
        }
        latency = latency_lut_row(
            **common,
            schedule_policy="metaschedule_tuned",
            backend="h800_tvm",
            measurement_status="measured",
            latency_p50_us=6209.998,
            latency_min_us=6209.0,
            latency_max_us=6210.0,
            warmup_iters=50,
            measure_iters=200,
            repeat=5,
            raw_artifact=str(raw_dir / "latency_raw"),
        )
        ap = ap_anchor_row(
            **{**common, "run_id": "ap_run", "source_files": []},
            schedule_policy="not_applicable",
            backend="model_eval",
            measurement_status="measured",
            metric="AP70",
            metric_value=0.63086,
            dataset="DAIR-V2X",
            eval_split="val",
            ckpt_path="checkpoints/pyramid.ckpt",
            finetune_protocol="none",
            raw_artifact=str(raw_dir / "ap_raw"),
        )
        energy = energy_lut_row(
            **{**common, "run_id": "energy_run"},
            schedule_policy="metaschedule_tuned",
            backend="h800_tvm_power_telemetry",
            measurement_status="measured",
            joule_per_inference=1.92,
            watt_avg=378.0,
            telemetry_source="nvidia_smi",
            idle_baseline_policy="subtract_idle_avg",
            sample_window_ms=5000,
            latency_run_id="latency_run",
            raw_artifact=str(raw_dir / "energy_raw"),
            measurement_run_id="energy_run",
            row_source="direct_generation",
        )
        return latency, ap, energy

    def test_artifact_registry_backfills_rows_and_validates_traceability(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            latency, ap, energy = self._rows(tmp_path)

            rows = build_artifact_registry_rows(
                latency_rows=[latency],
                ap_rows=[ap],
                energy_rows=[energy],
                created_at="2026-06-26T00:00:00Z",
            )

            self.assertEqual(len(rows), 1)
            row = rows[0]
            validate_artifact_registry_row(row)
            self.assertEqual(row["schema_version"], "stage2_artifact_registry_v1")
            self.assertEqual(row["label"], "base")
            self.assertEqual(row["artifact_status"], "ready")
            self.assertEqual(row["latency_row_ids"], [latency["row_id"]])
            self.assertEqual(row["ap_row_ids"], [ap["row_id"]])
            self.assertEqual(row["energy_row_ids"], [energy["row_id"]])
            self.assertTrue(str(row["onnx_path"]).endswith("base_backbone.onnx"))
            self.assertEqual(row["missing_artifacts"], [])

    def test_artifact_registry_builder_cli_writes_jsonl_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            latency, ap, energy = self._rows(tmp_path)
            latency_path = tmp_path / "latency.jsonl"
            ap_path = tmp_path / "ap.jsonl"
            energy_path = tmp_path / "energy.jsonl"
            registry_path = tmp_path / "artifact_registry_v1.jsonl"
            summary_path = tmp_path / "artifact_registry_summary_v1.json"
            write_jsonl(latency_path, [latency])
            write_jsonl(ap_path, [ap])
            write_jsonl(energy_path, [energy])

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/stage2_build_artifact_registry.py"),
                    "--latency-rows",
                    str(latency_path),
                    "--ap-rows",
                    str(ap_path),
                    "--energy-rows",
                    str(energy_path),
                    "--out-jsonl",
                    str(registry_path),
                    "--summary-json",
                    str(summary_path),
                ],
                cwd=ROOT,
                env={"PYTHONPATH": str(ROOT)},
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len(read_jsonl(registry_path)), 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["total_artifacts"], 1)
            self.assertEqual(summary["ready_artifacts"], 1)


class Stage2EnergySalvageTest(unittest.TestCase):
    def test_salvage_cli_rebuilds_energy_row_from_payload_and_latency_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_dir = tmp_path / "energy_raw"
            raw_dir.mkdir()
            latency_raw = tmp_path / "latency_raw"
            latency_raw.mkdir()
            latency_row = latency_lut_row(
                config_id="cfg_base_tuned",
                model="pyramid_lidar",
                manifest_digest="b" * 64,
                candidate_id="calibration:pyramid_lidar:base",
                software_point_id="pyramid_lidar:backbone:w64x128x256:fp16",
                dense_stage="backbone",
                optimized_scope="backbone_only",
                width=[64, 128, 256],
                quant_policy="fp16",
                schedule_policy="metaschedule_tuned",
                backend="h800_tvm",
                measurement_status="measured",
                latency_p50_us=6209.998,
                latency_min_us=6209.0,
                latency_max_us=6210.0,
                warmup_iters=50,
                measure_iters=200,
                repeat=5,
                raw_artifact=str(latency_raw),
                run_id="latency_run_001",
                created_at="2026-06-25T00:00:00+08:00",
            )
            latency_path = tmp_path / "latency.jsonl"
            out_path = tmp_path / "energy.jsonl"
            state_path = tmp_path / "salvage_state.jsonl"
            write_jsonl(latency_path, [latency_row])
            payload = {
                "run_id": "energy_run_001",
                "latency_run_id": "latency_run_001",
                "joule_per_inference": 1.9244,
                "watt_avg": 377.95,
                "telemetry_source": "nvidia-smi power.draw polling 50ms",
                "idle_baseline_policy": "subtract_idle_avg_5s_pre_window",
                "raw_artifact": str(raw_dir),
                "source_files": [str(raw_dir / "energy_result.json")],
            }
            (raw_dir / "telemetry_payload.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/stage2_salvage_energy_payloads.py"),
                    "--payload-root",
                    str(tmp_path),
                    "--latency-rows",
                    str(latency_path),
                    "--out-jsonl",
                    str(out_path),
                    "--job-state",
                    str(state_path),
                ],
                cwd=ROOT,
                env={"PYTHONPATH": str(ROOT)},
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            rows = read_jsonl(out_path)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["config_id"], "cfg_base_tuned")
            self.assertEqual(rows[0]["row_source"], "salvaged_from_payload")
            self.assertEqual(rows[0]["measurement_run_id"], "energy_run_001")
            self.assertEqual(rows[0]["latency_run_id"], "latency_run_001")
            state_statuses = [row["status"] for row in read_jsonl(state_path)]
            self.assertEqual(state_statuses, ["payload_ready", "salvaged", "succeeded"])


class Stage2LatencyOutlierPolicyTest(unittest.TestCase):
    def _latency_row(
        self,
        *,
        label: str,
        run_id: str,
        p50_us: float,
        min_us: float,
        max_us: float,
    ) -> dict[str, object]:
        return latency_lut_row(
            config_id=f"calibration_h800_tvm_pyramid_{label}_fp16_metaschedule_tuned",
            model="pyramid_lidar",
            manifest_digest="c" * 64,
            candidate_id=f"calibration:pyramid_lidar:{label}",
            software_point_id="pyramid_lidar:backbone:w64x128x256:fp16",
            dense_stage="backbone",
            optimized_scope="backbone_only",
            width=[64, 128, 256],
            quant_policy="fp16",
            schedule_policy="metaschedule_tuned",
            backend="h800_tvm",
            measurement_status="measured",
            latency_p50_us=p50_us,
            latency_min_us=min_us,
            latency_max_us=max_us,
            warmup_iters=50,
            measure_iters=200,
            repeat=5,
            raw_artifact=f"raw/{run_id}",
            run_id=run_id,
            created_at="2026-06-25T00:00:00+08:00",
        )

    def test_latency_detector_flags_repeat_outlier_and_passes_stable_anchor(self):
        stable = self._latency_row(
            label="base",
            run_id="base_clean",
            p50_us=6213.0,
            min_us=6210.0,
            max_us=6218.0,
        )
        unstable = self._latency_row(
            label="trap25",
            run_id="trap25_split",
            p50_us=30169.297,
            min_us=21621.723,
            max_us=46872.504,
        )

        report = detect_latency_outliers([stable, unstable], grade="calibration")
        by_run = {row["run_id"]: row for row in report["rows"]}

        self.assertEqual(by_run["base_clean"]["quality_flag"], "stable")
        self.assertEqual(by_run["base_clean"]["claim_status"], "claimable")
        self.assertEqual(by_run["trap25_split"]["quality_flag"], "unstable_repeat")
        self.assertEqual(by_run["trap25_split"]["claim_status"], "no_claim")
        self.assertIn("repeat_max_min_ratio", by_run["trap25_split"]["reasons"][0])

    def test_latency_detector_cli_writes_quality_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            rows_path = tmp_path / "latency.jsonl"
            out_json = tmp_path / "outlier_report.json"
            out_csv = tmp_path / "outlier_report.csv"
            write_jsonl(
                rows_path,
                [
                    self._latency_row(
                        label="trap25",
                        run_id="trap25_split",
                        p50_us=30169.297,
                        min_us=21621.723,
                        max_us=46872.504,
                    )
                ],
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/stage2_detect_latency_outliers.py"),
                    "--latency-rows",
                    str(rows_path),
                    "--out-json",
                    str(out_json),
                    "--out-csv",
                    str(out_csv),
                ],
                cwd=ROOT,
                env={"PYTHONPATH": str(ROOT)},
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(report["unstable_rows"], 1)
            self.assertIn("trap25_split", out_csv.read_text(encoding="utf-8"))


class Stage2ReadinessExportCliTest(unittest.TestCase):
    def test_quick_review_export_joins_registry_rows_and_quality(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_dir = tmp_path / "raw"
            raw_dir.mkdir()
            latency, ap, energy = Stage2ArtifactRegistryV1Test()._rows(raw_dir)
            ap = dict(ap)
            ap.update(
                {
                    "row_id": str(ap["row_id"]).replace(
                        "calibration_h800_tvm_pyramid_base_fp16_metaschedule_tuned",
                        "smoke_h800_tvm_pyramid_base_fp16_metaschedule_tuned",
                    ),
                    "config_id": "smoke_h800_tvm_pyramid_base_fp16_metaschedule_tuned",
                    "candidate_id": "smoke:pyramid_lidar:base",
                }
            )
            latency_path = tmp_path / "latency.jsonl"
            ap_path = tmp_path / "ap.jsonl"
            energy_path = tmp_path / "energy.jsonl"
            registry_path = tmp_path / "artifact_registry.jsonl"
            outlier_path = tmp_path / "outlier.json"
            out_csv = tmp_path / "quick_review.csv"
            out_json = tmp_path / "quick_review.json"
            write_jsonl(latency_path, [latency])
            write_jsonl(ap_path, [ap])
            write_jsonl(energy_path, [energy])
            registry_rows = build_artifact_registry_rows(
                latency_rows=[latency],
                ap_rows=[ap],
                energy_rows=[energy],
                created_at="2026-06-26T00:00:00Z",
            )
            registry_path.write_text(
                "\n".join(json.dumps(row) for row in registry_rows) + "\n",
                encoding="utf-8",
            )
            outlier_path.write_text(
                json.dumps(detect_latency_outliers([latency])),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/stage2_export_quick_review.py"),
                    "--artifact-registry",
                    str(registry_path),
                    "--latency-rows",
                    str(latency_path),
                    "--ap-rows",
                    str(ap_path),
                    "--energy-rows",
                    str(energy_path),
                    "--outlier-report",
                    str(outlier_path),
                    "--out-csv",
                    str(out_csv),
                    "--out-json",
                    str(out_json),
                ],
                cwd=ROOT,
                env={"PYTHONPATH": str(ROOT)},
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            text = out_csv.read_text(encoding="utf-8")
            self.assertIn("quality_flag", text)
            self.assertIn("claim_status", text)
            rows = json.loads(out_json.read_text(encoding="utf-8"))["rows"]
            latency_review = next(row for row in rows if row["latency_p50_ms"] is not None)
            self.assertEqual(latency_review["ap70"], 0.63086)
            self.assertEqual(latency_review["latency_claim_status"], "claimable")
            self.assertEqual(latency_review["energy_claim_status"], "claimable")

    def test_readiness_gate_reports_conditional_go_when_outlier_is_isolated(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_dir = tmp_path / "raw"
            raw_dir.mkdir()
            latency, ap, energy = Stage2ArtifactRegistryV1Test()._rows(raw_dir)
            unstable = dict(latency)
            unstable.update(
                {
                    "row_id": "latency_lut:pyramid_lidar:trap25:h800_tvm:unstable",
                    "config_id": "calibration_h800_tvm_pyramid_trap25_fp16_metaschedule_tuned",
                    "candidate_id": "calibration:pyramid_lidar:trap25",
                    "run_id": "trap25_unstable",
                    "latency_p50_us": 30169.297,
                    "latency_min_us": 21621.723,
                    "latency_max_us": 46872.504,
                }
            )
            latency_path = tmp_path / "latency.jsonl"
            ap_path = tmp_path / "ap.jsonl"
            energy_path = tmp_path / "energy.jsonl"
            registry_path = tmp_path / "artifact_registry.jsonl"
            evidence_registry_path = tmp_path / "evidence_registry.json"
            outlier_path = tmp_path / "outlier.json"
            report_path = tmp_path / "readiness.json"
            write_jsonl(latency_path, [latency, unstable])
            write_jsonl(ap_path, [ap])
            write_jsonl(energy_path, [energy])
            registry_rows = build_artifact_registry_rows(
                latency_rows=[latency, unstable],
                ap_rows=[ap],
                energy_rows=[energy],
                created_at="2026-06-26T00:00:00Z",
            )
            registry_path.write_text(
                "\n".join(json.dumps(row) for row in registry_rows) + "\n",
                encoding="utf-8",
            )
            outlier_path.write_text(
                json.dumps(detect_latency_outliers([latency, unstable])),
                encoding="utf-8",
            )
            evidence_registry_path.write_text(
                json.dumps(
                    {
                        "schema": "stage2_evidence_registry_v1",
                        "model": "pyramid_lidar",
                        "hardware_target": {
                            "name": "H800 Hopper",
                            "backend": "h800_tvm",
                        },
                        "manifest": {
                            "path": None,
                            "measurement_status": "not_available",
                            "backend": "not_available",
                            "scope": "rsu_dense_core",
                            "provenance": "test",
                        },
                        "latency_lut": {
                            "path": str(latency_path),
                            "measurement_status": "measured",
                            "backend": "h800_tvm",
                            "scope": "dense_core",
                            "provenance": "test",
                            "coverage": {
                                "expected_cells": 2,
                                "measured_cells": 2,
                                "failed_cells": 0,
                                "proxy_cells": 0,
                            },
                        },
                        "ap_anchors": {
                            "path": str(ap_path),
                            "measurement_status": "measured",
                            "backend": "model_eval",
                            "scope": "model_accuracy",
                            "provenance": "test",
                            "coverage": {
                                "expected_cells": 1,
                                "measured_cells": 1,
                                "failed_cells": 0,
                                "proxy_cells": 0,
                            },
                        },
                        "quant_evidence": {
                            "path": None,
                            "measurement_status": "not_done",
                            "backend": "not_done",
                            "scope": "quant",
                            "provenance": "test",
                            "coverage": {
                                "expected_cells": 0,
                                "measured_cells": 0,
                                "failed_cells": 0,
                                "proxy_cells": 0,
                            },
                        },
                        "energy_lut": {
                            "path": str(energy_path),
                            "measurement_status": "measured",
                            "backend": "h800_tvm_power_telemetry",
                            "scope": "dense_core",
                            "provenance": "test",
                            "coverage": {
                                "expected_cells": 1,
                                "measured_cells": 1,
                                "failed_cells": 0,
                                "proxy_cells": 0,
                            },
                        },
                        "downstream_objective": {
                            "path": None,
                            "measurement_status": "not_available",
                            "backend": "not_available",
                            "scope": {},
                            "provenance": "test",
                            "coverage": {
                                "expected_cells": 0,
                                "measured_cells": 0,
                                "failed_cells": 0,
                                "proxy_cells": 0,
                            },
                        },
                        "unsupported_conclusions": [],
                    }
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/stage2_readiness_gate.py"),
                    "--artifact-registry",
                    str(registry_path),
                    "--latency-rows",
                    str(latency_path),
                    "--ap-rows",
                    str(ap_path),
                    "--energy-rows",
                    str(energy_path),
                    "--evidence-registry",
                    str(evidence_registry_path),
                    "--outlier-report",
                    str(outlier_path),
                    "--out-json",
                    str(report_path),
                ],
                cwd=ROOT,
                env={"PYTHONPATH": str(ROOT)},
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["decision"], "CONDITIONAL_GO")
            self.assertEqual(report["gates"]["artifact_registry"]["status"], "pass")
            self.assertEqual(report["gates"]["evidence_registry"]["status"], "pass")
            self.assertEqual(report["gates"]["outlier"]["status"], "conditional")
