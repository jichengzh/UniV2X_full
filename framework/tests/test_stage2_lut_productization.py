from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from framework.stage2.evidence_registry import Stage2EvidenceRegistry
from framework.stage2.lut_productization import (
    LutProductizationError,
    TVM310_SITE_PACKAGES,
    ap_anchor_row,
    coverage_from_rows,
    build_tvm_runtime_env,
    energy_claim_allowed_from_rows,
    energy_lut_row,
    job_plan_row,
    latency_lut_row,
    latest_job_status,
    next_queued_jobs,
    read_jsonl,
    stable_config_id,
    validate_lut_row,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parents[2]


class Stage2LutProductizationTest(unittest.TestCase):
    def test_lut_productization_submodule_import_does_not_require_yaml(self):
        code = """
import importlib.abc
import sys

class BlockYaml(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'yaml' or fullname.startswith('yaml.'):
            raise ModuleNotFoundError('blocked yaml for lut-only import')
        return None

sys.meta_path.insert(0, BlockYaml())
from framework.stage2.lut_productization import energy_lut_row
print(energy_lut_row.__name__)
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            env={"PYTHONPATH": str(ROOT)},
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("energy_lut_row", result.stdout)

    def test_tvm_runtime_env_preserves_repo_and_tvm_pythonpath(self):
        env = build_tvm_runtime_env(
            {
                "PYTHONPATH": "/tmp/existing_python",
                "LD_LIBRARY_PATH": "/tmp/existing_ld",
                "PATH": "/tmp/existing_bin",
            }
        )

        pythonpath = env["PYTHONPATH"].split(":")
        self.assertEqual(pythonpath[:2], [str(ROOT), TVM310_SITE_PACKAGES])
        self.assertIn("/tmp/existing_python", pythonpath)
        self.assertTrue(env["LD_LIBRARY_PATH"].startswith(TVM310_SITE_PACKAGES))
        self.assertIn("/tmp/existing_ld", env["LD_LIBRARY_PATH"].split(":"))
        self.assertTrue(env["PATH"].startswith("/usr/local/cuda-12.2/bin:"))

    def test_stable_config_id_is_shared_across_latency_ap_and_energy(self):
        config_id = stable_config_id(
            model="pyramid_lidar",
            candidate_id="bev_encoder.s2",
            software_point_id="bev_encoder.s2:w128:fp16",
            quant_policy="fp16",
            schedule_policy="metaschedule_tuned",
        )

        self.assertEqual(
            config_id,
            "pyramid_lidar__bev_encoder.s2__bev_encoder.s2-w128-fp16__q_fp16__s_metaschedule_tuned",
        )

    def test_latency_ap_energy_rows_validate_required_fields(self):
        common = {
            "config_id": "pyramid_lidar__neck__neck-w64-fp16__q_fp16__s_default",
            "model": "pyramid_lidar",
            "manifest_digest": "a" * 64,
            "candidate_id": "neck",
            "software_point_id": "neck:w64:fp16",
            "dense_stage": "neck",
            "width": [64, 128, 256],
            "quant_policy": "fp16",
            "run_id": "run_001",
            "created_at": "2026-06-25T00:00:00+08:00",
        }
        latency = latency_lut_row(
            **common,
            schedule_policy="default",
            backend="h800_tvm",
            measurement_status="measured",
            latency_p50_us=123.4,
            warmup_iters=50,
            measure_iters=200,
            repeat=5,
        )
        ap = ap_anchor_row(
            **common,
            schedule_policy="not_applicable",
            backend="model_eval",
            measurement_status="measured",
            metric="AP70",
            metric_value=0.6369,
            dataset="DAIR-V2X",
            eval_split="val",
            ckpt_path="checkpoints/pyramid.ckpt",
            finetune_protocol="none",
        )
        energy = energy_lut_row(
            **common,
            schedule_policy="default",
            backend="h800_tvm_power_telemetry",
            measurement_status="measured",
            joule_per_inference=1.2,
            watt_avg=240.0,
            telemetry_source="nvidia_smi",
            idle_baseline_policy="subtract_idle_avg",
            sample_window_ms=5000,
            latency_run_id="run_001",
            raw_artifact="results/raw/energy.json",
            measurement_run_id="run_001",
            row_source="direct_generation",
        )

        for row in (latency, ap, energy):
            validate_lut_row(row)
            self.assertIn("precision", row)
            self.assertIn("quant_method", row)
            self.assertIn("quant_scope", row)
            self.assertIn("calibrator", row)
            self.assertIn("fallback_policy", row)
            self.assertIn("layer_precision_summary", row)
            self.assertFalse(row["full_network_claim"])
        self.assertEqual(latency["measurement_source"], "true_measurement_smoke")
        self.assertEqual(latency["claim_status"], "claimable_true_measurement_smoke")
        self.assertEqual(ap["measurement_source"], "true_eval")
        self.assertEqual(ap["claim_status"], "claimable_true_eval")
        self.assertEqual(energy["measurement_source"], "true_measurement_smoke")
        self.assertEqual(energy["claim_status"], "claimable_true_measurement_smoke")

    def test_quant_contract_defaults_to_h800_tvm_first_not_trt(self):
        row = latency_lut_row(
            config_id="cfg_int8",
            model="pyramid_lidar",
            manifest_digest="a" * 64,
            candidate_id="neck",
            software_point_id="neck:w64:int8",
            dense_stage="backbone",
            width=[64, 128, 256],
            quant_policy="int8",
            schedule_policy="metaschedule_tuned",
            backend="h800_tvm",
            measurement_status="failed",
            failure_reason="tvm_int8_backbone_subnet_not_ready",
            run_id="run_int8_gap",
            created_at="2026-06-27T00:00:00+08:00",
        )

        validate_lut_row(row)
        self.assertEqual(row["precision"], "int8")
        self.assertEqual(row["quant_method"], "h800_tvm_int8_backbone_subnet_experimental")
        self.assertEqual(row["engine_kind"], "tvm_vm")
        self.assertEqual(row["measurement_source"], "no_claim")
        self.assertFalse(row["full_network_claim"])

    def test_measured_h800_tvm_int8_rejects_trt_reference_method(self):
        row = latency_lut_row(
            config_id="bad_trt_int8",
            model="pyramid_lidar",
            manifest_digest="a" * 64,
            candidate_id="neck",
            software_point_id="neck:w64:int8",
            dense_stage="backbone",
            width=[64, 128, 256],
            quant_policy="int8",
            schedule_policy="metaschedule_tuned",
            backend="h800_tvm",
            measurement_status="measured",
            latency_p50_us=123.4,
            warmup_iters=1,
            measure_iters=1,
            repeat=1,
            run_id="run_bad_trt_reference",
            created_at="2026-06-27T00:00:00+08:00",
            quant_method="trt_ptq_minmax_wa_with_fp16_fallback",
            quant_scope="backbone_subnet",
            engine_kind="trt_reference",
            measurement_source="historical_reference_only",
            full_network_claim=False,
        )

        with self.assertRaisesRegex(LutProductizationError, "TRT reference"):
            validate_lut_row(row)

    def test_measured_latency_requires_h800_backend(self):
        row = latency_lut_row(
            config_id="bad",
            model="pyramid_lidar",
            manifest_digest="a" * 64,
            candidate_id="neck",
            software_point_id="neck:w64:fp16",
            dense_stage="neck",
            width=[64],
            quant_policy="fp16",
            schedule_policy="default",
            backend="historical_trt",
            measurement_status="measured",
            latency_p50_us=1.0,
            warmup_iters=1,
            measure_iters=1,
            repeat=1,
            run_id="run_bad",
            created_at="2026-06-25T00:00:00+08:00",
        )

        with self.assertRaisesRegex(LutProductizationError, "measured latency"):
            validate_lut_row(row)

    def test_jsonl_round_trip_preserves_rows(self):
        row = latency_lut_row(
            config_id="cfg",
            model="codriving",
            manifest_digest="b" * 64,
            candidate_id="backbone.s2",
            software_point_id="backbone.s2:w128:fp16",
            dense_stage="stage3",
            width=[32, 64, 128],
            quant_policy="fp16",
            schedule_policy="default",
            backend="h800_tvm",
            measurement_status="measured",
            latency_p50_us=1609.1,
            warmup_iters=50,
            measure_iters=200,
            repeat=5,
            run_id="run_jsonl",
            created_at="2026-06-25T00:00:00+08:00",
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "latency_lut_rows_v1.jsonl"
            write_jsonl(path, [row])
            self.assertEqual(read_jsonl(path), [row])


class Stage2LutJobPlanTest(unittest.TestCase):
    def test_job_plan_and_state_support_resume(self):
        plan = [
            job_plan_row(
                job_id="latency:cfg_a",
                model="pyramid_lidar",
                lut_kind="latency",
                job_type="generate_latency_lut",
                priority=10,
                config_id="cfg_a",
                manifest_path="framework/partitions/pyramid_lidar_partition.yaml",
                registry_path="results/stage2/pyramid/evidence_registry.json",
                expected_output="results/stage2/pyramid/latency/latency_lut_rows_v1.jsonl",
                command=[
                    "python",
                    "scripts/stage2_generate_latency_lut.py",
                    "--job-id",
                    "latency:cfg_a",
                ],
                max_attempts=2,
                timeout_s=21600,
            ),
            job_plan_row(
                job_id="ap:cfg_a",
                model="pyramid_lidar",
                lut_kind="ap",
                job_type="generate_ap_lut",
                priority=5,
                config_id="cfg_a",
                manifest_path="framework/partitions/pyramid_lidar_partition.yaml",
                registry_path="results/stage2/pyramid/evidence_registry.json",
                expected_output="results/stage2/pyramid/ap/ap_anchor_rows_v1.jsonl",
                command=[
                    "python",
                    "scripts/stage2_generate_ap_lut.py",
                    "--job-id",
                    "ap:cfg_a",
                ],
                max_attempts=1,
                timeout_s=21600,
            ),
        ]
        state = [
            {
                "schema": "lut_job_state_row_v1",
                "job_id": "latency:cfg_a",
                "status": "succeeded",
                "attempt": 1,
            },
        ]

        self.assertEqual(latest_job_status(state, "latency:cfg_a"), "succeeded")
        self.assertEqual([job["job_id"] for job in next_queued_jobs(plan, state)], ["ap:cfg_a"])

    def test_plan_cli_writes_only_generate_lut_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            out_jsonl = tmp_path / "jobs/lut_job_plan_v1.jsonl"
            command_json = json.dumps(
                [sys.executable, "-c", "import json; print(json.dumps({}))"]
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/stage2_plan_lut_jobs.py"),
                    "--model",
                    "pyramid_lidar",
                    "--manifest",
                    "framework/partitions/pyramid_lidar_partition.yaml",
                    "--registry",
                    "results/stage2/pyramid/evidence_registry.json",
                    "--config-id",
                    "cfg_shared",
                    "--candidate-id",
                    "smoke",
                    "--software-point-id",
                    "smoke:w64x128x256:fp16",
                    "--dense-stage",
                    "neck",
                    "--width",
                    "64,128,256",
                    "--quant-policy",
                    "fp16",
                    "--schedule-policy",
                    "default",
                    "--optimized-scope",
                    "backbone_only",
                    "--latency-measurement-command-json",
                    command_json,
                    "--ap-eval-command-json",
                    command_json,
                    "--energy-telemetry-command-json",
                    command_json,
                    "--out-jsonl",
                    str(out_jsonl),
                ],
                cwd=ROOT,
                env={"PYTHONPATH": str(ROOT)},
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            rows = read_jsonl(out_jsonl)
            self.assertEqual(
                [row["job_type"] for row in rows],
                ["generate_latency_lut", "generate_ap_lut", "generate_energy_lut"],
            )
            self.assertEqual({row["config_id"] for row in rows}, {"cfg_shared"})
            for row in rows:
                self.assertTrue(row["expected_output"].endswith("_rows_v1.jsonl"))
                self.assertIn("--optimized-scope", row["command"])
                self.assertIn("backbone_only", row["command"])
                self.assertIn("--quant-method", row["command"])
                self.assertIn("h800_tvm_relax_metaschedule_fp16", row["command"])
                self.assertIn("--quant-scope", row["command"])
                self.assertIn("backbone_only", row["command"])
                self.assertIn("--full-network-claim", row["command"])
                self.assertIn("false", row["command"])
                self.assertTrue(
                    any(
                        item.endswith(f"stage2_generate_{row['lut_kind']}_lut.py")
                        for item in row["command"]
                    )
                )
                self.assertFalse(str(row["job_type"]).startswith("import_existing"))


class Stage2LutGeneratorCliTest(unittest.TestCase):
    def test_latency_ap_energy_generators_append_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            latency_out = tmp_path / "latency_lut_rows_v1.jsonl"
            ap_out = tmp_path / "ap_anchor_rows_v1.jsonl"
            energy_out = tmp_path / "energy_lut_rows_v1.jsonl"

            latency_cmd = [
                sys.executable,
                "-c",
                (
                    "import json; "
                    "print(json.dumps({'latency_p50_us': 123.4, "
                    "'latency_p90_us': 140.0, 'latency_mean_us': 125.0}))"
                ),
            ]
            latency = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/stage2_generate_latency_lut.py"),
                    "--model",
                    "pyramid_lidar",
                    "--config-id",
                    "cfg_latency",
                    "--candidate-id",
                    "smoke",
                    "--software-point-id",
                    "smoke:w64x128x256:fp16",
                    "--dense-stage",
                    "neck",
                    "--width",
                    "64,128,256",
                    "--quant-policy",
                    "fp16",
                    "--quant-method",
                    "h800_tvm_relax_metaschedule_fp16",
                    "--quant-scope",
                    "backbone_only",
                    "--calibrator",
                    "none",
                    "--fallback-policy",
                    "none",
                    "--layer-precision-summary",
                    "not_applicable_fp16",
                    "--full-network-claim",
                    "false",
                    "--schedule-policy",
                    "default",
                    "--backend",
                    "h800_tvm",
                    "--optimized-scope",
                    "backbone_only",
                    "--measurement-command-json",
                    json.dumps(latency_cmd),
                    "--out-jsonl",
                    str(latency_out),
                ],
                cwd=ROOT,
                env={"PYTHONPATH": str(ROOT)},
                capture_output=True,
                text=True,
            )
            self.assertEqual(latency.returncode, 0, latency.stderr)
            latency_row = json.loads(latency_out.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(latency_row["schema"], "latency_lut_row_v1")
            self.assertEqual(latency_row["measurement_status"], "measured")
            self.assertEqual(latency_row["latency_p50_us"], 123.4)
            self.assertEqual(latency_row["optimized_scope"], "backbone_only")
            self.assertEqual(latency_row["precision"], "fp16")
            self.assertEqual(latency_row["quant_method"], "h800_tvm_relax_metaschedule_fp16")
            self.assertEqual(latency_row["quant_scope"], "backbone_only")
            self.assertFalse(latency_row["full_network_claim"])

            ap_cmd = [
                sys.executable,
                "-c",
                (
                    "import json; "
                    "print(json.dumps({'metric': 'AP70', 'metric_value': 0.612, "
                    "'dataset': 'DAIR-V2X', 'eval_split': 'val', "
                    "'ckpt_path': 'ckpts/smoke.pt'}))"
                ),
            ]
            ap = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/stage2_generate_ap_lut.py"),
                    "--model",
                    "pyramid_lidar",
                    "--config-id",
                    "cfg_latency",
                    "--candidate-id",
                    "smoke",
                    "--software-point-id",
                    "smoke:w64x128x256:fp16",
                    "--dense-stage",
                    "model",
                    "--width",
                    "64,128,256",
                    "--quant-policy",
                    "fp16",
                    "--quant-method",
                    "h800_tvm_relax_metaschedule_fp16",
                    "--quant-scope",
                    "backbone_only",
                    "--calibrator",
                    "none",
                    "--fallback-policy",
                    "none",
                    "--layer-precision-summary",
                    "not_applicable_fp16",
                    "--full-network-claim",
                    "false",
                    "--schedule-policy",
                    "not_applicable",
                    "--eval-command-json",
                    json.dumps(ap_cmd),
                    "--out-jsonl",
                    str(ap_out),
                ],
                cwd=ROOT,
                env={"PYTHONPATH": str(ROOT)},
                capture_output=True,
                text=True,
            )
            self.assertEqual(ap.returncode, 0, ap.stderr)
            ap_row = json.loads(ap_out.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(ap_row["schema"], "ap_anchor_row_v1")
            self.assertEqual(ap_row["config_id"], "cfg_latency")
            self.assertEqual(ap_row["metric_value"], 0.612)
            self.assertEqual(ap_row["quant_method"], "h800_tvm_relax_metaschedule_fp16")

            energy_cmd = [
                sys.executable,
                "-c",
                (
                    "import json; "
                    "print(json.dumps({'joule_per_inference': 1.2, "
                    "'watt_avg': 240.0, 'telemetry_source': 'nvidia_smi', "
                    "'idle_baseline_policy': 'subtract_idle_avg', "
                    "'raw_artifact': 'logs/energy_smoke.json'}))"
                ),
            ]
            energy = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/stage2_generate_energy_lut.py"),
                    "--model",
                    "pyramid_lidar",
                    "--config-id",
                    "cfg_latency",
                    "--candidate-id",
                    "smoke",
                    "--software-point-id",
                    "smoke:w64x128x256:fp16",
                    "--dense-stage",
                    "neck",
                    "--width",
                    "64,128,256",
                    "--quant-policy",
                    "fp16",
                    "--quant-method",
                    "h800_tvm_relax_metaschedule_fp16",
                    "--quant-scope",
                    "backbone_only",
                    "--calibrator",
                    "none",
                    "--fallback-policy",
                    "none",
                    "--layer-precision-summary",
                    "not_applicable_fp16",
                    "--full-network-claim",
                    "false",
                    "--schedule-policy",
                    "default",
                    "--backend",
                    "h800_tvm_power_telemetry",
                    "--telemetry-command-json",
                    json.dumps(energy_cmd),
                    "--out-jsonl",
                    str(energy_out),
                ],
                cwd=ROOT,
                env={"PYTHONPATH": str(ROOT)},
                capture_output=True,
                text=True,
            )
            self.assertEqual(energy.returncode, 0, energy.stderr)
            energy_row = json.loads(energy_out.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(energy_row["schema"], "energy_lut_row_v1")
            self.assertEqual(energy_row["measurement_status"], "measured")
            self.assertEqual(energy_row["telemetry_source"], "nvidia_smi")
            self.assertEqual(energy_row["row_source"], "direct_generation")
            self.assertEqual(energy_row["measurement_run_id"], energy_row["run_id"])
            self.assertEqual(energy_row["quant_method"], "h800_tvm_relax_metaschedule_fp16")


class Stage2QuantAnchorSmokeCliTest(unittest.TestCase):
    def test_quant_anchor_smoke_exports_tvm_first_state_rows_and_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "quant_smoke"
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/stage2_generate_quant_anchor_smoke.py"),
                    "--output-root",
                    str(output_root),
                ],
                cwd=ROOT,
                env={"PYTHONPATH": str(ROOT)},
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            latency_rows = read_jsonl(output_root / "rows/latency_quant_rows_v1.jsonl")
            energy_rows = read_jsonl(output_root / "rows/energy_quant_rows_v1.jsonl")
            ap_rows = read_jsonl(output_root / "rows/ap_quant_rows_v1.jsonl")
            expected_labels = {
                "base",
                "p50",
                "p75",
                "trap25",
                "s0_024",
                "s0_040",
                "s0_056",
                "s1_048",
            }
            expected_precisions = {"fp32", "fp16", "int8"}

            self.assertEqual(len(latency_rows), 24)
            self.assertEqual(len(energy_rows), 24)
            self.assertEqual(len(ap_rows), 24)
            for rows in (latency_rows, energy_rows, ap_rows):
                self.assertEqual({row["candidate_id"] for row in rows}, expected_labels)
                self.assertEqual({row["precision"] for row in rows}, expected_precisions)
                for row in rows:
                    validate_lut_row(row)
                    self.assertFalse(row["full_network_claim"])
                    measured_text = " ".join(
                        str(row.get(key, "")).lower()
                        for key in ("quant_method", "engine_kind", "measurement_source")
                    )
                    if row["measurement_status"] == "measured" and str(row["backend"]).startswith("h800_tvm"):
                        self.assertNotIn("trt", measured_text)

            int8_rows = [
                row
                for row in latency_rows + energy_rows + ap_rows
                if row["precision"] == "int8"
            ]
            self.assertTrue(int8_rows)
            self.assertFalse(any(row["measurement_status"] == "measured" for row in int8_rows))
            for row in int8_rows:
                self.assertEqual(row["quant_method"], "h800_tvm_int8_backbone_subnet_experimental")
                self.assertEqual(row["engine_kind"], "tvm_vm")
                self.assertIn("tvm_int8", row["failure_reason"])

            for relative in (
                "plans/quant_anchor_job_plan_v1.json",
                "quarantine/quant_unclaimable_v1.jsonl",
                "exports/quant_three_metric_summary_latest.md",
                "exports/quant_three_metric_summary_latest.csv",
                "exports/quant_three_metric_summary_latest.json",
                "exports/quant_gap_report_latest.json",
            ):
                self.assertTrue((output_root / relative).exists(), relative)

            summary = json.loads(
                (output_root / "exports/quant_three_metric_summary_latest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(summary["total_cells"], 24)
            self.assertEqual(summary["precision_counts"], {"fp16": 8, "fp32": 8, "int8": 8})
            self.assertEqual(summary["latency_status_counts"], {"no_claim": 24})

            gap_report = json.loads(
                (output_root / "exports/quant_gap_report_latest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertIn(
                "historical_fp16_tagged_latency_not_true_fp16_evidence",
                gap_report["failure_reason_counts"],
            )
            self.assertIn("tvm_int8_backbone_subnet_not_ready", gap_report["failure_reason_counts"])

    def test_quant_anchor_smoke_imports_measured_fp32_smoke_latency(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_root = tmp_path / "quant_smoke"
            smoke_rows = tmp_path / "fp32_smoke_rows.jsonl"
            smoke_rows.write_text(
                json.dumps(
                    {
                        "candidate_id": "base",
                        "software_point_id": "original60:base:64,128,256:fp32:latency_smoke",
                        "precision": "fp32",
                        "backend": "h800_tvm",
                        "measurement_status": "measured",
                        "latency_p50_us": 6208.528,
                        "warmup_iters": 1,
                        "measure_iters": 50,
                        "repeat": 3,
                        "tvm_target": "cuda",
                        "tvm_strategy": "relax_metaschedule_reuse_existing_ms_db",
                        "build_status": "success",
                        "run_id": "fp32_smoke_base_tuned",
                        "source_files": ["raw/fp32_smoke/base/latency_result.json"],
                        "raw_artifact": "raw/fp32_smoke/base",
                        "quant_method": "h800_tvm_relax_fp32",
                        "quant_scope": "backbone_only",
                        "engine_kind": "tvm_vm",
                        "measurement_source": "true_measurement",
                        "claim_status": "measured_smoke",
                        "quality_gate_status": "fp32_latency_smoke_only",
                        "full_network_claim": False,
                        "schedule_policy": "metaschedule_tuned",
                        "schedule_profile": "metaschedule_tuned",
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/stage2_generate_quant_anchor_smoke.py"),
                    "--output-root",
                    str(output_root),
                    "--fp32-latency-smoke-rows",
                    str(smoke_rows),
                ],
                cwd=ROOT,
                env={"PYTHONPATH": str(ROOT)},
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            latency_rows = read_jsonl(output_root / "rows/latency_quant_rows_v1.jsonl")
            fp32_base = next(
                row for row in latency_rows if row["candidate_id"] == "base" and row["precision"] == "fp32"
            )
            validate_lut_row(fp32_base)
            self.assertEqual(fp32_base["measurement_status"], "measured")
            self.assertEqual(fp32_base["latency_p50_us"], 6208.528)
            self.assertEqual(fp32_base["run_id"], "fp32_smoke_base_tuned")
            self.assertFalse(fp32_base["full_network_claim"])
            self.assertIsNone(fp32_base.get("failure_reason"))

            gap_report = json.loads(
                (output_root / "exports/quant_gap_report_latest.json").read_text(
                    encoding="utf-8"
                )
            )
            fp32_latency_gaps = [
                row
                for row in gap_report["rows"]
                if row["label"] == "base"
                and row["precision"] == "fp32"
                and row["axis"] == "latency"
            ]
            self.assertEqual(fp32_latency_gaps, [])

    def test_quant_anchor_smoke_imports_true_fp16_smoke_latency(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_root = tmp_path / "quant_smoke"
            fp16_rows = tmp_path / "fp16_true_rows.jsonl"
            fp16_rows.write_text(
                json.dumps(
                    {
                        "candidate_id": "base",
                        "software_point_id": "true_fp16_smoke:base:64x128x256:fp16:latency",
                        "precision": "fp16",
                        "quant_policy": "fp16",
                        "backend": "h800_tvm",
                        "measurement_status": "measured",
                        "latency_p50_us": 53277.002,
                        "warmup_iters": 1,
                        "measure_iters": 50,
                        "repeat": 3,
                        "tvm_target": "cuda",
                        "tvm_strategy": "relax_metaschedule_reuse_existing_ms_db",
                        "build_status": "success",
                        "run_id": "fp16_true_smoke_base_tuned",
                        "source_files": [
                            "/exdata/jichengzhi/s2_tvm/fp16_true_smoke/base/base_backbone_true_fp16.onnx",
                            "/exdata/jichengzhi/s2_tvm/fp16_true_smoke/base/layer_precision_summary.json",
                        ],
                        "raw_artifact": "raw/fp16_true_smoke/base",
                        "quant_method": "h800_tvm_true_fp16_onnx_relax",
                        "quant_scope": "backbone_only",
                        "layer_precision_summary": "raw/fp16_true_smoke/base/layer_precision_summary.json",
                        "engine_kind": "tvm_vm",
                        "measurement_source": "true_measurement",
                        "claim_status": "claimable_true_measurement",
                        "quality_gate_status": "true_fp16_onnx_smoke_only",
                        "full_network_claim": False,
                        "schedule_policy": "metaschedule_tuned",
                        "schedule_profile": "metaschedule_tuned",
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/stage2_generate_quant_anchor_smoke.py"),
                    "--output-root",
                    str(output_root),
                    "--fp16-latency-smoke-rows",
                    str(fp16_rows),
                ],
                cwd=ROOT,
                env={"PYTHONPATH": str(ROOT)},
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            latency_rows = read_jsonl(output_root / "rows/latency_quant_rows_v1.jsonl")
            fp16_base = next(
                row for row in latency_rows if row["candidate_id"] == "base" and row["precision"] == "fp16"
            )
            validate_lut_row(fp16_base)
            self.assertEqual(fp16_base["measurement_status"], "measured")
            self.assertEqual(fp16_base["latency_p50_us"], 53277.002)
            self.assertEqual(fp16_base["run_id"], "fp16_true_smoke_base_tuned")
            self.assertEqual(fp16_base["quality_gate_status"], "true_fp16_onnx_smoke_only")
            self.assertEqual(fp16_base["layer_precision_summary"], "raw/fp16_true_smoke/base/layer_precision_summary.json")
            self.assertFalse(fp16_base["full_network_claim"])
            self.assertIsNone(fp16_base.get("failure_reason"))

            summary = json.loads(
                (output_root / "exports/quant_three_metric_summary_latest.json").read_text(
                    encoding="utf-8"
                )
            )
            base_summary = next(
                row for row in summary["rows"] if row["label"] == "base" and row["precision"] == "fp16"
            )
            self.assertEqual(base_summary["latency_status"], "measured")
            self.assertEqual(base_summary["latency_ms"], 53.277002)
            self.assertIn("true_fp16_onnx_smoke_only", base_summary["quality_gate_status"])

    def test_quant_anchor_smoke_imports_measured_int8_smoke_latency(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_root = tmp_path / "quant_smoke"
            int8_rows = tmp_path / "int8_smoke_rows.jsonl"
            int8_rows.write_text(
                json.dumps(
                    {
                        "candidate_id": "base",
                        "software_point_id": "int8_smoke:base:64x128x256:int8:latency",
                        "precision": "int8",
                        "quant_policy": "int8",
                        "backend": "h800_tvm",
                        "measurement_status": "measured",
                        "latency_p50_us": 9100.5,
                        "warmup_iters": 1,
                        "measure_iters": 50,
                        "repeat": 3,
                        "tvm_target": "cuda",
                        "tvm_strategy": "tvm_vm_loaded_int8_qdq_artifact",
                        "build_status": "success",
                        "run_id": "int8_smoke_base_tvm_vm",
                        "source_files": [
                            "/exdata/jichengzhi/s2_tvm/int8_route/base/base_backbone_int8_qdq_direct_tvm_vm.so",
                            "/exdata/jichengzhi/s2_tvm/int8_route/base/layer_precision_summary.json",
                        ],
                        "raw_artifact": "raw/int8_latency_smoke/base",
                        "quant_scheme": "tvm_int8_static_qdq_synthetic_minmax",
                        "quant_method": "h800_tvm_int8_backbone_subnet_experimental",
                        "quant_scope": "backbone_only",
                        "calibration_source": "synthetic_shape_smoke",
                        "calibration_digest": "abc123",
                        "calibrator": "onnxruntime_static_qdq_minmax_synthetic",
                        "fallback_policy": "none",
                        "layer_precision_summary": "/exdata/jichengzhi/s2_tvm/int8_route/base/layer_precision_summary.json",
                        "engine_kind": "tvm_vm",
                        "engine_digest": "def456",
                        "measurement_source": "true_measurement_smoke",
                        "claim_status": "claimable_true_measurement_smoke",
                        "quality_gate_status": "int8_tvm_vm_latency_smoke_only",
                        "full_network_claim": False,
                        "schedule_policy": "int8_qdq_direct_tvm_vm",
                        "schedule_profile": "int8_qdq_direct_tvm_vm",
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/stage2_generate_quant_anchor_smoke.py"),
                    "--output-root",
                    str(output_root),
                    "--int8-latency-smoke-rows",
                    str(int8_rows),
                ],
                cwd=ROOT,
                env={"PYTHONPATH": str(ROOT)},
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            latency_rows = read_jsonl(output_root / "rows/latency_quant_rows_v1.jsonl")
            int8_base = next(
                row for row in latency_rows if row["candidate_id"] == "base" and row["precision"] == "int8"
            )
            validate_lut_row(int8_base)
            self.assertEqual(int8_base["measurement_status"], "measured")
            self.assertEqual(int8_base["latency_p50_us"], 9100.5)
            self.assertEqual(int8_base["run_id"], "int8_smoke_base_tvm_vm")
            self.assertEqual(int8_base["quality_gate_status"], "int8_tvm_vm_latency_smoke_only")
            self.assertFalse(int8_base["full_network_claim"])
            self.assertIsNone(int8_base.get("failure_reason"))

            summary = json.loads(
                (output_root / "exports/quant_three_metric_summary_latest.json").read_text(
                    encoding="utf-8"
                )
            )
            base_summary = next(
                row for row in summary["rows"] if row["label"] == "base" and row["precision"] == "int8"
            )
            self.assertEqual(base_summary["latency_status"], "measured")
            self.assertEqual(base_summary["latency_ms"], 9.1005)
            self.assertIn("int8_tvm_vm_latency_smoke_only", base_summary["quality_gate_status"])

    def test_quant_anchor_energy_smoke_import_downgrades_claim_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_root = tmp_path / "quant_smoke"
            fp16_energy_rows = tmp_path / "fp16_energy_rows.jsonl"
            fp16_energy_rows.write_text(
                json.dumps(
                    {
                        "candidate_id": "base",
                        "software_point_id": "true_fp16_smoke:base:64x128x256:fp16:energy",
                        "precision": "fp16",
                        "quant_policy": "fp16",
                        "backend": "h800_tvm_power_telemetry",
                        "measurement_status": "measured",
                        "joule_per_inference": 9.671489700245022,
                        "watt_avg": 240.0,
                        "telemetry_source": "nvidia_smi",
                        "idle_baseline_policy": "subtract_idle_avg",
                        "latency_run_id": "fp16_true_smoke_base_tuned",
                        "measurement_run_id": "fp16_true_energy_base_smoke",
                        "run_id": "fp16_true_energy_base_smoke",
                        "source_files": [
                            "raw/fp16_true_energy/base/idle_power_samples.csv",
                            "raw/fp16_true_energy/base/active_power_samples.csv",
                        ],
                        "raw_artifact": "raw/fp16_true_energy/base/energy_result.json",
                        "quant_method": "h800_tvm_true_fp16_onnx_relax",
                        "quant_scope": "backbone_only",
                        "engine_kind": "tvm_vm",
                        "measurement_source": "true_measurement",
                        "claim_status": "claimable_true_measurement",
                        "quality_gate_status": "true_fp16_onnx_energy",
                        "full_network_claim": False,
                        "schedule_policy": "metaschedule_tuned",
                        "schedule_profile": "metaschedule_tuned",
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/stage2_generate_quant_anchor_smoke.py"),
                    "--output-root",
                    str(output_root),
                    "--fp16-energy-smoke-rows",
                    str(fp16_energy_rows),
                ],
                cwd=ROOT,
                env={"PYTHONPATH": str(ROOT)},
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            energy_rows = read_jsonl(output_root / "rows/energy_quant_rows_v1.jsonl")
            fp16_base = next(
                row for row in energy_rows if row["candidate_id"] == "base" and row["precision"] == "fp16"
            )
            validate_lut_row(fp16_base)
            self.assertEqual(fp16_base["measurement_status"], "measured")
            self.assertEqual(fp16_base["measurement_source"], "true_measurement_smoke")
            self.assertEqual(fp16_base["claim_status"], "claimable_true_measurement_smoke")
            self.assertIn("true_fp16_onnx_energy", fp16_base["quality_gate_status"])


class Stage2Original60QuantCoverageCliTest(unittest.TestCase):
    def test_fp32_original60_remap_audit_reclassifies_suspect_fp16_backbone_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_root = tmp_path / "original60_quant"
            suspect_rows = tmp_path / "suspect_fp16_rows.jsonl"
            suspect_rows.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "candidate_id": "coverage:pyramid_lidar:w24x128x256:fp16",
                                "software_point_id": "original60:s0_024:24x128x256:fp16:latency",
                                "label": "s0_024",
                                "width": [24, 128, 256],
                                "quant_policy": "fp16",
                                "backend": "h800_tvm",
                                "measurement_status": "measured",
                                "schedule_policy": "default",
                                "latency_p50_us": 39483.718,
                                "warmup_iters": 1,
                                "measure_iters": 50,
                                "repeat": 3,
                                "run_id": "original60_s0_024_default",
                                "source_files": [
                                    "/exdata/jichengzhi/s2_tvm/models/s0_024_backbone.onnx"
                                ],
                                "raw_artifact": "raw/original60_latency/s0_024",
                                "full_network_claim": False,
                            },
                            sort_keys=True,
                        ),
                        json.dumps(
                            {
                                "candidate_id": "coverage:pyramid_lidar:w24x128x256:fp16",
                                "software_point_id": "original60:s0_024:24x128x256:fp16:latency",
                                "label": "s0_024",
                                "width": [24, 128, 256],
                                "quant_policy": "fp16",
                                "backend": "h800_tvm",
                                "measurement_status": "measured",
                                "schedule_policy": "metaschedule_tuned",
                                "latency_p50_us": 39503.066,
                                "warmup_iters": 1,
                                "measure_iters": 50,
                                "repeat": 3,
                                "run_id": "original60_s0_024_tuned",
                                "source_files": [
                                    "/exdata/jichengzhi/s2_tvm/models/s0_024_backbone.onnx"
                                ],
                                "raw_artifact": "raw/original60_latency/s0_024",
                                "full_network_claim": False,
                            },
                            sort_keys=True,
                        ),
                        json.dumps(
                            {
                                "candidate_id": "coverage:pyramid_lidar:w64x48x256:fp16",
                                "software_point_id": "original60:s1_048:64x48x256:fp16:latency",
                                "label": "s1_048",
                                "width": [64, 48, 256],
                                "quant_policy": "fp16",
                                "backend": "h800_tvm",
                                "measurement_status": "measured",
                                "schedule_policy": "metaschedule_tuned",
                                "latency_p50_us": 44452.754,
                                "run_id": "original60_s1_048_tuned",
                                "source_files": [
                                    "/exdata/jichengzhi/s2_tvm/models/s1_048_backbone_true_fp16.onnx"
                                ],
                                "raw_artifact": "raw/original60_latency/s1_048",
                                "full_network_claim": False,
                            },
                            sort_keys=True,
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/stage2_audit_fp32_original60_remap.py"),
                    "--input-rows",
                    str(suspect_rows),
                    "--output-root",
                    str(output_root),
                    "--created-at",
                    "2026-06-27T00:00:00+08:00",
                ],
                cwd=ROOT,
                env={"PYTHONPATH": str(ROOT)},
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            audit = json.loads(
                (
                    output_root / "exports/fp32_original60_remap_audit_latest.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(audit["classification_counts"]["remap_to_fp32_candidate"], 1)
            self.assertEqual(audit["classification_counts"]["needs_remeasure"], 1)
            self.assertTrue(
                (output_root / "exports/fp32_original60_remap_audit_latest.md").exists()
            )
            remapped = read_jsonl(
                output_root / "rows/fp32_latency_original60_remapped_rows_v1.jsonl"
            )
            self.assertEqual(len(remapped), 1)
            row = remapped[0]
            validate_lut_row(row)
            self.assertEqual(row["label"], "s0_024")
            self.assertEqual(row["candidate_id"], "coverage:pyramid_lidar:w24x128x256:fp32")
            self.assertEqual(row["original_candidate_id"], "coverage:pyramid_lidar:w24x128x256:fp16")
            self.assertEqual(row["precision"], "fp32")
            self.assertEqual(row["quant_policy"], "fp32")
            self.assertEqual(row["schedule_policy"], "metaschedule_tuned")
            self.assertEqual(row["latency_p50_us"], 39503.066)
            self.assertEqual(row["run_id"], "original60_s0_024_tuned")
            self.assertEqual(row["source_files"], ["/exdata/jichengzhi/s2_tvm/models/s0_024_backbone.onnx"])
            self.assertEqual(row["raw_artifact"], "raw/original60_latency/s0_024")
            self.assertEqual(row["measurement_source"], "historical_true_measurement_reclassified")
            self.assertEqual(row["quality_gate_status"], "fp32_reclassified_from_suspect_fp16_tagged_row")
            self.assertFalse(row["full_network_claim"])

    def test_original60_quant_state_coverage_exports_60_by_3_state_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "original60_quant"
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/stage2_generate_original60_quant_state_coverage.py"),
                    "--output-root",
                    str(output_root),
                ],
                cwd=ROOT,
                env={"PYTHONPATH": str(ROOT)},
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            latency_rows = read_jsonl(output_root / "rows/latency_original60_quant_rows_v1.jsonl")
            energy_rows = read_jsonl(output_root / "rows/energy_original60_quant_rows_v1.jsonl")
            ap_rows = read_jsonl(output_root / "rows/ap_original60_quant_rows_v1.jsonl")
            expected_precisions = {"fp32", "fp16", "int8"}

            self.assertEqual(len(latency_rows), 180)
            self.assertEqual(len(energy_rows), 180)
            self.assertEqual(len(ap_rows), 180)
            self.assertEqual(len({row["label"] for row in latency_rows}), 60)
            for rows in (latency_rows, energy_rows, ap_rows):
                self.assertEqual({row["precision"] for row in rows}, expected_precisions)
                for row in rows:
                    validate_lut_row(row)
                    self.assertFalse(row["full_network_claim"])
                    measured_text = " ".join(
                        str(row.get(key, "")).lower()
                        for key in ("quant_method", "engine_kind", "measurement_source")
                    )
                    if row["measurement_status"] == "measured" and str(row["backend"]).startswith("h800_tvm"):
                        self.assertNotIn("trt", measured_text)

            int8_rows = [
                row
                for row in latency_rows + energy_rows + ap_rows
                if row["precision"] == "int8"
            ]
            self.assertFalse(any(row["measurement_status"] == "measured" for row in int8_rows))
            for row in int8_rows:
                self.assertEqual(row["quant_method"], "h800_tvm_int8_backbone_subnet_experimental")
                self.assertFalse(row["full_network_claim"])

            summary = json.loads(
                (output_root / "exports/original60_quant_three_metric_summary_latest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(summary["total_cells"], 180)
            self.assertEqual(summary["precision_counts"], {"fp16": 60, "fp32": 60, "int8": 60})
            self.assertEqual(summary["candidate_count"], 60)
            self.assertEqual(summary["latency_status_counts"], {"no_claim": 180})
            self.assertEqual(summary["energy_status_counts"], {"measured": 56, "no_claim": 124})
            self.assertEqual(summary["ap_status_counts"], {"measured": 5, "no_claim": 175})

            gap_report = json.loads(
                (output_root / "exports/original60_quant_gap_report_latest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertIn(
                "historical_fp16_tagged_latency_not_true_fp16_evidence",
                gap_report["failure_reason_counts"],
            )
            self.assertIn(
                "fp16_energy_requires_true_fp16_h800_power_telemetry",
                gap_report["failure_reason_counts"],
            )
            self.assertIn(
                "fp32_energy_historical_remap_not_available",
                gap_report["failure_reason_counts"],
            )
            self.assertIn(
                "fp16_ap_requires_true_fp16_eval_source_revalidation",
                gap_report["failure_reason_counts"],
            )
            self.assertIn("tvm_int8_backbone_subnet_not_ready", gap_report["failure_reason_counts"])
            for relative in (
                "plans/original60_quant_state_plan_v1.json",
                "quarantine/original60_quant_unclaimable_v1.jsonl",
                "exports/original60_quant_three_metric_summary_latest.md",
                "exports/original60_quant_three_metric_summary_latest.csv",
                "exports/original60_quant_three_metric_summary_latest.json",
                "exports/original60_quant_gap_report_latest.json",
            ):
                self.assertTrue((output_root / relative).exists(), relative)

    def test_original60_quant_state_coverage_imports_measured_fp32_smoke_latency(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_root = tmp_path / "original60_quant"
            summary_csv = tmp_path / "original60_summary.csv"
            smoke_rows = tmp_path / "fp32_smoke_rows.jsonl"
            summary_csv.write_text(
                "\n".join(
                    [
                        "label,candidate_id,width,latency_status,latency_tuned_ms,energy_status,energy_j_per_inference,ap70,ap_source_kind,ap_status",
                        "base,base,64x128x256,,,,,,,",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            smoke_rows.write_text(
                json.dumps(
                    {
                        "candidate_id": "base",
                        "software_point_id": "original60:base:64,128,256:fp32:latency_smoke",
                        "precision": "fp32",
                        "backend": "h800_tvm",
                        "measurement_status": "measured",
                        "latency_p50_us": 12345.0,
                        "warmup_iters": 1,
                        "measure_iters": 50,
                        "repeat": 3,
                        "tvm_target": "cuda",
                        "tvm_strategy": "relax_metaschedule_reuse_existing_ms_db",
                        "build_status": "success",
                        "run_id": "fp32_smoke_base_tuned",
                        "source_files": ["raw/fp32_smoke/base/latency_result.json"],
                        "raw_artifact": "raw/fp32_smoke/base",
                        "quant_method": "h800_tvm_relax_fp32",
                        "quant_scope": "backbone_only",
                        "engine_kind": "tvm_vm",
                        "measurement_source": "true_measurement",
                        "claim_status": "measured_smoke",
                        "quality_gate_status": "fp32_latency_smoke_only",
                        "full_network_claim": False,
                        "schedule_policy": "metaschedule_tuned",
                        "schedule_profile": "metaschedule_tuned",
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/stage2_generate_original60_quant_state_coverage.py"),
                    "--output-root",
                    str(output_root),
                    "--original60-summary-csv",
                    str(summary_csv),
                    "--fp32-latency-smoke-rows",
                    str(smoke_rows),
                ],
                cwd=ROOT,
                env={"PYTHONPATH": str(ROOT)},
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            latency_rows = read_jsonl(output_root / "rows/latency_original60_quant_rows_v1.jsonl")
            fp32_base = next(row for row in latency_rows if row["label"] == "base" and row["precision"] == "fp32")
            validate_lut_row(fp32_base)
            self.assertEqual(fp32_base["measurement_status"], "measured")
            self.assertEqual(fp32_base["latency_p50_us"], 12345.0)
            self.assertEqual(fp32_base["run_id"], "fp32_smoke_base_tuned")
            self.assertEqual(fp32_base["measurement_source"], "true_measurement")
            self.assertFalse(fp32_base["full_network_claim"])
            self.assertIsNone(fp32_base.get("failure_reason"))

            summary = json.loads(
                (output_root / "exports/original60_quant_three_metric_summary_latest.json").read_text(
                    encoding="utf-8"
                )
            )
            base_summary = next(
                row for row in summary["rows"] if row["label"] == "base" and row["precision"] == "fp32"
            )
            self.assertEqual(base_summary["latency_status"], "measured")
            self.assertEqual(base_summary["latency_ms"], 12.345)
            self.assertEqual(base_summary["latency_schedule_policy"], "metaschedule_tuned")
            self.assertEqual(base_summary["latency_tvm_strategy"], "relax_metaschedule_reuse_existing_ms_db")

            gap_report = json.loads(
                (output_root / "exports/original60_quant_gap_report_latest.json").read_text(
                    encoding="utf-8"
                )
            )
            fp32_latency_gaps = [
                row
                for row in gap_report["rows"]
                if row["label"] == "base"
                and row["precision"] == "fp32"
                and row["axis"] == "latency"
            ]
            self.assertEqual(fp32_latency_gaps, [])

    def test_original60_quant_state_coverage_imports_true_fp16_smoke_latency(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_root = tmp_path / "original60_quant"
            summary_csv = tmp_path / "original60_summary.csv"
            fp16_rows = tmp_path / "fp16_smoke_rows.jsonl"
            summary_csv.write_text(
                "\n".join(
                    [
                        "label,candidate_id,width,latency_status,latency_tuned_ms,energy_status,energy_j_per_inference,ap70,ap_source_kind,ap_status",
                        "s0_024,coverage:pyramid_lidar:w24x128x256:fp16,24x128x256,measured,39.503066,,,,,",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            fp16_rows.write_text(
                json.dumps(
                    {
                        "candidate_id": "s0_024",
                        "software_point_id": "true_fp16_smoke:s0_024:24x128x256:fp16:latency",
                        "label": "s0_024",
                        "precision": "fp16",
                        "quant_policy": "fp16",
                        "backend": "h800_tvm",
                        "measurement_status": "measured",
                        "latency_p50_us": 37734.585,
                        "run_id": "true_fp16_smoke_s0_024_tuned",
                        "source_files": [
                            "/exdata/jichengzhi/s2_tvm/fp16_true_smoke/s0_024/s0_024_backbone_true_fp16.onnx"
                        ],
                        "raw_artifact": "raw/fp16_true_smoke/s0_024",
                        "quant_method": "h800_tvm_true_fp16_onnx_relax",
                        "quant_scope": "backbone_only",
                        "engine_kind": "tvm_vm",
                        "measurement_source": "true_measurement",
                        "claim_status": "claimable_true_measurement_smoke",
                        "quality_gate_status": "true_fp16_onnx_smoke_only",
                        "layer_precision_summary": "/exdata/jichengzhi/s2_tvm/fp16_true_smoke/s0_024/layer_precision_summary.json",
                        "full_network_claim": False,
                        "schedule_policy": "metaschedule_tuned",
                        "schedule_profile": "metaschedule_tuned",
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/stage2_generate_original60_quant_state_coverage.py"),
                    "--output-root",
                    str(output_root),
                    "--original60-summary-csv",
                    str(summary_csv),
                    "--fp16-latency-smoke-rows",
                    str(fp16_rows),
                ],
                cwd=ROOT,
                env={"PYTHONPATH": str(ROOT)},
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            latency_rows = read_jsonl(output_root / "rows/latency_original60_quant_rows_v1.jsonl")
            fp16_row = next(row for row in latency_rows if row["label"] == "s0_024" and row["precision"] == "fp16")
            validate_lut_row(fp16_row)
            self.assertEqual(fp16_row["measurement_status"], "measured")
            self.assertEqual(fp16_row["latency_p50_us"], 37734.585)
            self.assertEqual(fp16_row["run_id"], "true_fp16_smoke_s0_024_tuned")
            self.assertEqual(fp16_row["quality_gate_status"], "true_fp16_onnx_smoke_only")
            self.assertEqual(fp16_row["quant_method"], "h800_tvm_true_fp16_onnx_relax")
            self.assertFalse(fp16_row["full_network_claim"])
            self.assertIsNone(fp16_row.get("failure_reason"))

            summary = json.loads(
                (output_root / "exports/original60_quant_three_metric_summary_latest.json").read_text(
                    encoding="utf-8"
                )
            )
            fp16_summary = next(
                row for row in summary["rows"] if row["label"] == "s0_024" and row["precision"] == "fp16"
            )
            self.assertEqual(fp16_summary["latency_status"], "measured")
            self.assertEqual(fp16_summary["latency_ms"], 37.734585)
            self.assertIn("true_fp16_onnx_smoke_only", fp16_summary["quality_gate_status"])

    def test_original60_quant_state_coverage_prefers_fp32_remeasured_energy_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_root = tmp_path / "original60_quant"
            rows_dir = output_root / "rows"
            rows_dir.mkdir(parents=True)
            summary_csv = tmp_path / "original60_summary.csv"
            summary_csv.write_text(
                "\n".join(
                    [
                        "label,candidate_id,width,latency_status,latency_tuned_ms,latency_tuned_run_id,energy_status,energy_j_per_inference,energy_run_id,ap70,ap_source_kind,ap_status",
                        "s0_024,coverage:pyramid_lidar:w24x128x256:fp16,24x128x256,measured,39.522546,fp32_latency_run,measured,0.010000,historical_energy_run,,,",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            remeasured_row = {
                "candidate_id": "coverage:pyramid_lidar:w24x128x256:fp32",
                "label": "s0_024",
                "precision": "fp32",
                "quant_policy": "fp32",
                "backend": "h800_tvm_power_telemetry",
                "measurement_status": "measured",
                "joule_per_inference": 6.75,
                "watt_avg": 380.0,
                "idle_watt_avg": 145.0,
                "run_id": "fp32_energy_remeasure_s0_024",
                "latency_run_id": "fp32_latency_run",
                "measurement_run_id": "fp32_energy_remeasure_s0_024",
                "source_files": [
                    "raw/fp32_remeasure/s0_024/energy_result.json",
                    "raw/fp32_remeasure/s0_024/idle_power_samples.csv",
                    "raw/fp32_remeasure/s0_024/active_power_samples.csv",
                ],
                "raw_artifact": "raw/fp32_remeasure/s0_024",
                "quant_method": "h800_tvm_relax_fp32",
                "quant_scope": "backbone_only",
                "quant_scheme": "none",
                "engine_kind": "tvm_vm",
                "measurement_source": "true_measurement",
                "claim_status": "claimable_true_measurement",
                "quality_gate_status": "fp32_energy_remeasured_h800",
                "full_network_claim": False,
                "idle_baseline_policy": "subtract_idle_avg_5s_pre_window",
                "telemetry_source": "nvidia-smi power.draw polling 50ms",
                "schedule_policy": "metaschedule_tuned",
                "schedule_profile": "metaschedule_tuned",
            }
            (rows_dir / "fp32_original60_energy_remeasured_rows_v1.jsonl").write_text(
                json.dumps(remeasured_row, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/stage2_generate_original60_quant_state_coverage.py"),
                    "--output-root",
                    str(output_root),
                    "--original60-summary-csv",
                    str(summary_csv),
                ],
                cwd=ROOT,
                env={"PYTHONPATH": str(ROOT)},
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            energy_rows = [
                json.loads(line)
                for line in (rows_dir / "energy_original60_quant_rows_v1.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            ]
            fp32_energy = next(row for row in energy_rows if row["precision"] == "fp32")
            self.assertEqual(fp32_energy["measurement_status"], "measured")
            self.assertEqual(fp32_energy["joule_per_inference"], 6.75)
            self.assertEqual(fp32_energy["measurement_source"], "true_measurement")
            self.assertEqual(fp32_energy["claim_status"], "claimable_true_measurement")
            self.assertEqual(fp32_energy["quality_gate_status"], "fp32_energy_remeasured_h800")
            self.assertEqual(fp32_energy["schedule_policy"], "metaschedule_tuned")
            self.assertEqual(fp32_energy["raw_artifact"], "raw/fp32_remeasure/s0_024")

            summary = json.loads(
                (output_root / "exports/original60_quant_three_metric_summary_latest.json").read_text(
                    encoding="utf-8"
                )
            )
            fp32_summary = next(
                row for row in summary["rows"] if row["label"] == "s0_024" and row["precision"] == "fp32"
            )
            self.assertEqual(fp32_summary["energy_j_per_inference"], 6.75)
            self.assertEqual(fp32_summary["energy_measurement_source"], "true_measurement")
            self.assertEqual(fp32_summary["energy_claim_status"], "claimable_true_measurement")

            summary_md = (
                output_root / "exports/original60_quant_three_metric_summary_latest.md"
            ).read_text(encoding="utf-8")
            self.assertIn("energy schedule", summary_md)
            self.assertIn("| s0_024 | 24x128x256 | fp32 |", summary_md)
            self.assertIn("6.750000 | metaschedule_tuned | measured", summary_md)

    def test_original60_quant_state_coverage_prefers_fp32_threaded60_energy_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_root = tmp_path / "original60_quant"
            rows_dir = output_root / "rows"
            rows_dir.mkdir(parents=True)
            summary_csv = tmp_path / "original60_summary.csv"
            summary_csv.write_text(
                "\n".join(
                    [
                        "label,candidate_id,width,latency_status,latency_tuned_ms,latency_tuned_run_id,energy_status,energy_j_per_inference,energy_run_id,ap70,ap_source_kind,ap_status",
                        "s0_024,coverage:pyramid_lidar:w24x128x256:fp16,24x128x256,measured,39.522546,fp32_latency_run,measured,0.010000,historical_energy_run,,,",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            old_row = {
                "candidate_id": "coverage:pyramid_lidar:w24x128x256:fp32",
                "label": "s0_024",
                "precision": "fp32",
                "quant_policy": "fp32",
                "backend": "h800_tvm_power_telemetry",
                "measurement_status": "measured",
                "joule_per_inference": 6.75,
                "watt_avg": 380.0,
                "idle_watt_avg": 145.0,
                "run_id": "fp32_energy_remeasure_s0_024",
                "source_files": [
                    "raw/fp32_remeasure/s0_024/energy_result.json",
                    "raw/fp32_remeasure/s0_024/idle_power_samples.csv",
                    "raw/fp32_remeasure/s0_024/active_power_samples.csv",
                ],
                "raw_artifact": "raw/fp32_remeasure/s0_024",
                "measurement_source": "true_measurement",
                "claim_status": "claimable_true_measurement",
                "quality_gate_status": "fp32_energy_remeasured_h800",
                "schedule_policy": "metaschedule_tuned",
            }
            threaded_row = {
                **old_row,
                "joule_per_inference": 8.25,
                "watt_avg": 360.0,
                "idle_watt_avg": 140.0,
                "run_id": "fp32_energy_threaded60_s0_024",
                "source_files": [
                    "raw/fp32_threaded60/s0_024/energy_result.json",
                    "raw/fp32_threaded60/s0_024/idle_power_samples.csv",
                    "raw/fp32_threaded60/s0_024/active_power_samples.csv",
                ],
                "raw_artifact": "raw/fp32_threaded60/s0_024",
                "quality_gate_status": "fp32_energy_threaded_window_h800",
            }
            (rows_dir / "fp32_original60_energy_remeasured_rows_v1.jsonl").write_text(
                json.dumps(old_row, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (rows_dir / "fp32_original60_energy_threaded60_rows_v1.jsonl").write_text(
                json.dumps(threaded_row, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/stage2_generate_original60_quant_state_coverage.py"),
                    "--output-root",
                    str(output_root),
                    "--original60-summary-csv",
                    str(summary_csv),
                ],
                cwd=ROOT,
                env={"PYTHONPATH": str(ROOT)},
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            energy_rows = [
                json.loads(line)
                for line in (rows_dir / "energy_original60_quant_rows_v1.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            ]
            fp32_energy = next(row for row in energy_rows if row["precision"] == "fp32")
            self.assertEqual(fp32_energy["joule_per_inference"], 8.25)
            self.assertEqual(fp32_energy["quality_gate_status"], "fp32_energy_threaded_window_h800")
            self.assertEqual(fp32_energy["raw_artifact"], "raw/fp32_threaded60/s0_024")

    def test_original60_quant_state_coverage_defaults_to_authoritative_rows_under_output_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_root = tmp_path / "original60_quant"
            rows_dir = output_root / "rows"
            rows_dir.mkdir(parents=True)
            summary_csv = tmp_path / "original60_summary.csv"
            summary_csv.write_text(
                "\n".join(
                    [
                        "label,candidate_id,width,latency_status,latency_tuned_ms,energy_status,energy_j_per_inference,ap70,ap_source_kind,ap_status",
                        "s0_024,coverage:pyramid_lidar:w24x128x256:fp16,24x128x256,,,,,,,",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            def write_raw_rows(path: Path, rows: list[dict[str, object]]) -> None:
                path.write_text(
                    "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                    encoding="utf-8",
                )

            write_raw_rows(
                rows_dir / "fp16_true_original60_latency_rows_v1.jsonl",
                [
                    {
                        "candidate_id": "s0_024",
                        "label": "s0_024",
                        "width": [24, 128, 256],
                        "precision": "fp16",
                        "quant_policy": "fp16",
                        "backend": "h800_tvm",
                        "measurement_status": "measured",
                        "latency_p50_us": 37734.585,
                        "run_id": "true_fp16_original60_latency",
                        "source_files": ["raw/fp16/s0_024/latency_result.json"],
                        "raw_artifact": "raw/fp16/s0_024",
                        "quant_method": "h800_tvm_true_fp16_onnx_relax",
                        "quant_scope": "backbone_only",
                        "engine_kind": "tvm_vm",
                        "measurement_source": "true_measurement",
                        "claim_status": "claimable_true_measurement_smoke",
                        "quality_gate_status": "true_fp16_original60_latency",
                        "layer_precision_summary": "raw/fp16/s0_024/layer_precision_summary.json",
                        "full_network_claim": False,
                        "schedule_policy": "metaschedule_tuned",
                        "schedule_profile": "metaschedule_tuned",
                    }
                ],
            )
            write_raw_rows(
                rows_dir / "fp16_true_original60_energy_rows_v1.jsonl",
                [
                    {
                        "candidate_id": "s0_024",
                        "label": "s0_024",
                        "precision": "fp16",
                        "quant_policy": "fp16",
                        "backend": "h800_tvm_power_telemetry",
                        "measurement_status": "measured",
                        "joule_per_inference": 4.25,
                        "run_id": "true_fp16_original60_energy",
                        "source_files": [
                            "raw/fp16/s0_024/energy_result.json",
                            "raw/fp16/s0_024/idle_power_samples.csv",
                            "raw/fp16/s0_024/active_power_samples.csv",
                        ],
                        "raw_artifact": "raw/fp16/s0_024",
                        "quant_method": "h800_tvm_true_fp16_onnx_relax",
                        "quant_scope": "backbone_only",
                        "engine_kind": "tvm_vm",
                        "measurement_source": "true_measurement",
                        "claim_status": "claimable_true_measurement_smoke",
                        "quality_gate_status": "true_fp16_original60_energy",
                        "layer_precision_summary": "raw/fp16/s0_024/layer_precision_summary.json",
                        "full_network_claim": False,
                        "idle_baseline_policy": "subtract_idle_avg_5s_pre_window",
                        "telemetry_source": "nvidia_smi_power_draw",
                        "schedule_policy": "metaschedule_tuned",
                    }
                ],
            )
            write_raw_rows(
                rows_dir / "native_int8_full_onnx_original60_latency_rows_v1.jsonl",
                [
                    {
                        "candidate_id": "s0_024",
                        "label": "s0_024",
                        "precision": "int8",
                        "backend": "h800_tvm",
                        "measurement_status": "measured",
                        "latency_p50_us": 9210.731,
                        "run_id": "native_full_onnx_original60_latency",
                        "source_files": ["raw/native/s0_024/latency_result.json"],
                        "raw_artifact": "raw/native/s0_024",
                        "quant_method": "h800_tvm_native_int8_backbone_subnet",
                        "quant_scope": "backbone_subnet_native_int8",
                        "calibration_source": "none_direct_native_int8_synthetic_inputs",
                        "calibration_digest": "none",
                        "calibrator": "none",
                        "fallback_policy": "none",
                        "layer_precision_summary": "raw/native/s0_024/tvm_operator_inventory.json",
                        "engine_kind": "tvm_graph_executor",
                        "engine_digest": "native_engine_digest",
                        "measurement_source": "true_measurement_smoke",
                        "claim_status": "claimable_true_measurement_smoke",
                        "quality_gate_status": "native_int8_full_onnx_topology_latency",
                        "full_network_claim": False,
                        "schedule_policy": "native_int8_full_onnx_graph_executor",
                        "schedule_profile": "native_int8_full_onnx_graph_executor",
                    }
                ],
            )
            write_raw_rows(
                rows_dir / "native_int8_full_onnx_original60_energy_rows_v1.jsonl",
                [
                    {
                        "candidate_id": "s0_024",
                        "label": "s0_024",
                        "precision": "int8",
                        "backend": "h800_tvm_power_telemetry",
                        "measurement_status": "measured",
                        "joule_per_inference": 2.4527137319477,
                        "run_id": "native_full_onnx_original60_energy",
                        "latency_run_id": "native_full_onnx_original60_latency",
                        "measurement_run_id": "native_full_onnx_original60_energy",
                        "source_files": [
                            "raw/native/s0_024/energy_result.json",
                            "raw/native/s0_024/idle_power_samples.csv",
                            "raw/native/s0_024/active_power_samples.csv",
                        ],
                        "raw_artifact": "raw/native/s0_024",
                        "quant_method": "h800_tvm_native_int8_backbone_subnet",
                        "quant_scope": "backbone_subnet_native_int8",
                        "calibration_source": "none_direct_native_int8_synthetic_inputs",
                        "calibration_digest": "none",
                        "calibrator": "none",
                        "fallback_policy": "none",
                        "layer_precision_summary": "raw/native/s0_024/tvm_operator_inventory.json",
                        "engine_kind": "tvm_graph_executor",
                        "engine_digest": "native_engine_digest",
                        "measurement_source": "true_measurement_smoke",
                        "claim_status": "claimable_true_measurement_smoke",
                        "quality_gate_status": "native_int8_full_onnx_topology_energy",
                        "full_network_claim": False,
                        "idle_baseline_policy": "subtract_idle_avg_5s_pre_window",
                        "telemetry_source": "nvidia_smi_power_draw",
                        "schedule_policy": "native_int8_full_onnx_graph_executor",
                        "schedule_profile": "native_int8_full_onnx_graph_executor",
                    }
                ],
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/stage2_generate_original60_quant_state_coverage.py"),
                    "--output-root",
                    str(output_root),
                    "--original60-summary-csv",
                    str(summary_csv),
                ],
                cwd=ROOT,
                env={"PYTHONPATH": str(ROOT)},
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            latency_rows = read_jsonl(output_root / "rows/latency_original60_quant_rows_v1.jsonl")
            energy_rows = read_jsonl(output_root / "rows/energy_original60_quant_rows_v1.jsonl")
            fp16_latency = next(
                row for row in latency_rows if row["label"] == "s0_024" and row["precision"] == "fp16"
            )
            fp16_energy = next(
                row for row in energy_rows if row["label"] == "s0_024" and row["precision"] == "fp16"
            )
            int8_latency = next(
                row for row in latency_rows if row["label"] == "s0_024" and row["precision"] == "int8"
            )
            int8_energy = next(
                row for row in energy_rows if row["label"] == "s0_024" and row["precision"] == "int8"
            )
            self.assertEqual(fp16_latency["measurement_status"], "measured")
            self.assertEqual(fp16_latency["run_id"], "true_fp16_original60_latency")
            self.assertEqual(fp16_energy["measurement_status"], "measured")
            self.assertEqual(fp16_energy["joule_per_inference"], 4.25)
            self.assertEqual(int8_latency["measurement_status"], "measured")
            self.assertEqual(int8_latency["engine_kind"], "tvm_graph_executor")
            self.assertEqual(int8_energy["measurement_status"], "measured")
            self.assertEqual(int8_energy["joule_per_inference"], 2.4527137319477)

    def test_original60_quant_state_coverage_imports_remapped_rows_but_prefers_direct_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_root = tmp_path / "original60_quant"
            summary_csv = tmp_path / "original60_summary.csv"
            remap_rows = tmp_path / "fp32_remapped_rows.jsonl"
            smoke_rows = tmp_path / "fp32_smoke_rows.jsonl"
            summary_csv.write_text(
                "\n".join(
                    [
                        "label,candidate_id,width,latency_status,latency_tuned_ms,energy_status,energy_j_per_inference,ap70,ap_source_kind,ap_status",
                        "s0_024,coverage:pyramid_lidar:w24x128x256:fp16,24x128x256,,,,,,,",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            remap_rows.write_text(
                json.dumps(
                    {
                        "candidate_id": "coverage:pyramid_lidar:w24x128x256:fp16",
                        "software_point_id": "original60:s0_024:24x128x256:fp32:latency_remapped",
                        "label": "s0_024",
                        "precision": "fp32",
                        "quant_policy": "fp32",
                        "backend": "h800_tvm",
                        "measurement_status": "measured",
                        "latency_p50_us": 39503.066,
                        "run_id": "original60_s0_024_tuned",
                        "source_files": ["/exdata/jichengzhi/s2_tvm/models/s0_024_backbone.onnx"],
                        "raw_artifact": "raw/original60_latency/s0_024",
                        "quant_method": "h800_tvm_relax_fp32",
                        "quant_scope": "backbone_only",
                        "engine_kind": "tvm_vm",
                        "measurement_source": "historical_true_measurement_reclassified",
                        "claim_status": "claimable_true_measurement_remapped",
                        "quality_gate_status": "fp32_reclassified_from_suspect_fp16_tagged_row",
                        "full_network_claim": False,
                        "schedule_policy": "metaschedule_tuned",
                        "schedule_profile": "metaschedule_tuned",
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            smoke_rows.write_text(
                json.dumps(
                    {
                        "candidate_id": "s0_024",
                        "software_point_id": "original60:s0_024:24,128,256:fp32:latency_smoke",
                        "precision": "fp32",
                        "quant_policy": "fp32",
                        "backend": "h800_tvm",
                        "measurement_status": "measured",
                        "latency_p50_us": 39522.546,
                        "run_id": "fp32_smoke_s0_024_tuned",
                        "source_files": ["raw/fp32_smoke/s0_024/latency_result.json"],
                        "raw_artifact": "raw/fp32_smoke/s0_024",
                        "quant_method": "h800_tvm_relax_fp32",
                        "quant_scope": "backbone_only",
                        "engine_kind": "tvm_vm",
                        "measurement_source": "true_measurement",
                        "claim_status": "measured_smoke",
                        "quality_gate_status": "fp32_latency_smoke_only",
                        "full_network_claim": False,
                        "schedule_policy": "metaschedule_tuned",
                        "schedule_profile": "metaschedule_tuned",
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/stage2_generate_original60_quant_state_coverage.py"),
                    "--output-root",
                    str(output_root),
                    "--original60-summary-csv",
                    str(summary_csv),
                    "--fp32-latency-remap-rows",
                    str(remap_rows),
                    "--fp32-latency-smoke-rows",
                    str(smoke_rows),
                ],
                cwd=ROOT,
                env={"PYTHONPATH": str(ROOT)},
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            latency_rows = read_jsonl(output_root / "rows/latency_original60_quant_rows_v1.jsonl")
            fp32_row = next(row for row in latency_rows if row["label"] == "s0_024" and row["precision"] == "fp32")
            validate_lut_row(fp32_row)
            self.assertEqual(fp32_row["measurement_status"], "measured")
            self.assertEqual(fp32_row["latency_p50_us"], 39522.546)
            self.assertEqual(fp32_row["run_id"], "fp32_smoke_s0_024_tuned")
            self.assertEqual(fp32_row["quality_gate_status"], "fp32_latency_smoke_only")

            summary = json.loads(
                (output_root / "exports/original60_quant_three_metric_summary_latest.json").read_text(
                    encoding="utf-8"
                )
            )
            fp32_summary = next(
                row for row in summary["rows"] if row["label"] == "s0_024" and row["precision"] == "fp32"
            )
            self.assertEqual(fp32_summary["latency_status"], "measured")
            self.assertEqual(fp32_summary["latency_ms"], 39.522546)

    def test_original60_quant_state_coverage_imports_fp32_energy_and_ap_from_authoritative_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_root = tmp_path / "original60_quant"
            summary_csv = tmp_path / "original60_summary.csv"
            summary_csv.write_text(
                "\n".join(
                    [
                        "label,candidate_id,width,latency_status,latency_tuned_ms,energy_status,energy_j_per_inference,ap30,ap50,ap70,ap_source_kind,ap_status,energy_run_id,ap_source_path,ap_quality_gate_status",
                        "s0_024,coverage:pyramid_lidar:w24x128x256:fp16,24x128x256,measured,39.503066,measured,4.954709,0.833141389,0.791044145,0.632495206,true_eval,claimable_true_eval,original60_energy_v1:s0_024,raw/s0_024/ap_eval_report.json,trend_anomaly_pending_repeat",
                        "s2_096,coverage:pyramid_lidar:w64x128x96:fp16,64x128x96,quarantine_cuda_illegal_memory_access,,,,,,,no_claim,no_claim_missing_source,,,",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/stage2_generate_original60_quant_state_coverage.py"),
                    "--output-root",
                    str(output_root),
                    "--original60-summary-csv",
                    str(summary_csv),
                ],
                cwd=ROOT,
                env={"PYTHONPATH": str(ROOT)},
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            energy_rows = read_jsonl(output_root / "rows/energy_original60_quant_rows_v1.jsonl")
            ap_rows = read_jsonl(output_root / "rows/ap_original60_quant_rows_v1.jsonl")
            fp32_energy = next(row for row in energy_rows if row["label"] == "s0_024" and row["precision"] == "fp32")
            fp32_ap = next(row for row in ap_rows if row["label"] == "s0_024" and row["precision"] == "fp32")

            validate_lut_row(fp32_energy)
            validate_lut_row(fp32_ap)
            self.assertEqual(fp32_energy["measurement_status"], "measured")
            self.assertEqual(fp32_energy["joule_per_inference"], 4.954709)
            self.assertEqual(fp32_energy["measurement_source"], "historical_true_measurement_reclassified")
            self.assertEqual(fp32_energy["claim_status"], "claimable_true_measurement_remapped")
            self.assertIsNone(fp32_energy.get("failure_reason"))
            self.assertEqual(fp32_ap["measurement_status"], "measured")
            self.assertEqual(fp32_ap["metric_value"], 0.632495206)
            self.assertEqual(fp32_ap["measurement_source"], "stage2_original60_ap_source_map_reference")
            self.assertEqual(fp32_ap["claim_status"], "claimable_true_eval")
            self.assertIsNone(fp32_ap.get("failure_reason"))

            summary = json.loads(
                (output_root / "exports/original60_quant_three_metric_summary_latest.json").read_text(
                    encoding="utf-8"
                )
            )
            fp32_summary = next(
                row for row in summary["rows"] if row["label"] == "s0_024" and row["precision"] == "fp32"
            )
            self.assertEqual(fp32_summary["energy_status"], "measured")
            self.assertEqual(fp32_summary["energy_j_per_inference"], 4.954709)
            self.assertEqual(fp32_summary["ap_status"], "measured")
            self.assertEqual(fp32_summary["ap70"], 0.632495206)

            gap_report = json.loads(
                (output_root / "exports/original60_quant_gap_report_latest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                [
                    row
                    for row in gap_report["rows"]
                    if row["label"] == "s0_024"
                    and row["precision"] == "fp32"
                    and row["axis"] in {"energy", "ap"}
                ],
                [],
            )

    def test_original60_quant_state_coverage_prefers_true_fp32_ap_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_root = tmp_path / "original60_quant"
            summary_csv = tmp_path / "original60_summary.csv"
            fp32_ap_rows = tmp_path / "fp32_true_ap_rows.jsonl"
            summary_csv.write_text(
                "\n".join(
                    [
                        "label,candidate_id,width,latency_status,latency_tuned_ms,energy_status,energy_j_per_inference,ap30,ap50,ap70,ap_source_kind,ap_status,energy_run_id,ap_source_path,ap_quality_gate_status",
                        "s0_024,coverage:pyramid_lidar:w24x128x256:fp16,24x128x256,measured,39.503066,measured,4.954709,0.1,0.2,0.3,reference,claimable_true_eval,energy_run,raw/reference.json,reference_only",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            fp32_ap_rows.write_text(
                json.dumps(
                    {
                        "label": "s0_024",
                        "width": [24, 128, 256],
                        "precision": "fp32",
                        "backend": "model_eval",
                        "measurement_status": "measured",
                        "measurement_source": "true_eval",
                        "metric": "AP70",
                        "metric_value": 0.654321,
                        "secondary_metrics": {"AP30": 0.812345, "AP50": 0.765432},
                        "dataset": "DAIR-V2X",
                        "eval_split": "val",
                        "ckpt_path": "checkpoints/s0_024/net_epoch_bestval_at29.pth",
                        "ckpt_digest": "ckpt_digest",
                        "eval_command": "stage2_h800_true_fp32_ap_eval.py --execute",
                        "source_files": ["raw/fp32/s0_024/ap_eval_report.json"],
                        "raw_artifact": "raw/fp32/s0_024",
                        "quant_method": "h800_tvm_true_fp32_model_eval",
                        "quant_scope": "full_model_ap_eval_true_fp32",
                        "engine_kind": "model_eval",
                        "engine_digest": "report_digest",
                        "claim_status": "claimable_true_eval",
                        "quality_gate_status": "true_fp32_original60_ap_eval",
                        "full_network_claim": False,
                        "provenance": "stage2_original60_true_fp32_h800_model_eval",
                        "notes": "true_fp32 model_eval AP row from measured report",
                        "run_id": "true_fp32_original60_ap_s0_024",
                        "created_at": "2026-07-01T00:00:00Z",
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/stage2_generate_original60_quant_state_coverage.py"),
                    "--output-root",
                    str(output_root),
                    "--original60-summary-csv",
                    str(summary_csv),
                    "--fp32-ap-rows",
                    str(fp32_ap_rows),
                ],
                cwd=ROOT,
                env={"PYTHONPATH": str(ROOT)},
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            ap_rows = read_jsonl(output_root / "rows/ap_original60_quant_rows_v1.jsonl")
            fp32_ap = next(row for row in ap_rows if row["label"] == "s0_024" and row["precision"] == "fp32")
            validate_lut_row(fp32_ap)
            self.assertEqual(fp32_ap["measurement_status"], "measured")
            self.assertEqual(fp32_ap["metric_value"], 0.654321)
            self.assertEqual(fp32_ap["measurement_source"], "true_eval")
            self.assertEqual(fp32_ap["quant_method"], "h800_tvm_true_fp32_model_eval")
            self.assertEqual(fp32_ap["quality_gate_status"], "true_fp32_original60_ap_eval")

            summary = json.loads(
                (output_root / "exports/original60_quant_three_metric_summary_latest.json").read_text(
                    encoding="utf-8"
                )
            )
            fp32_summary = next(
                row for row in summary["rows"] if row["label"] == "s0_024" and row["precision"] == "fp32"
            )
            self.assertEqual(fp32_summary["ap70"], 0.654321)
            self.assertEqual(fp32_summary["ap_source_kind"], "true_eval")
            self.assertIn("fp32_true_original60_ap_rows", summary["update_note"])

    def test_original60_quant_state_coverage_imports_measured_int8_smoke_latency(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_root = tmp_path / "original60_quant"
            summary_csv = tmp_path / "original60_summary.csv"
            int8_rows = tmp_path / "int8_smoke_rows.jsonl"
            summary_csv.write_text(
                "\n".join(
                    [
                        "label,candidate_id,width,latency_status,latency_tuned_ms,energy_status,energy_j_per_inference,ap70,ap_source_kind,ap_status",
                        "s0_024,coverage:pyramid_lidar:w24x128x256:fp16,24x128x256,,,,,,,",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            int8_rows.write_text(
                json.dumps(
                    {
                        "candidate_id": "s0_024",
                        "software_point_id": "int8_smoke:s0_024:24x128x256:int8:latency",
                        "precision": "int8",
                        "quant_policy": "int8",
                        "backend": "h800_tvm",
                        "measurement_status": "measured",
                        "latency_p50_us": 41040.342,
                        "run_id": "int8_smoke_s0_024_tvm_vm",
                        "source_files": ["/exdata/jichengzhi/s2_tvm/int8_route/s0_024/s0_024_backbone_int8_qdq_direct_tvm_vm.so"],
                        "raw_artifact": "raw/int8_latency_smoke/s0_024",
                        "quant_scheme": "tvm_int8_static_qdq_synthetic_minmax",
                        "quant_method": "h800_tvm_int8_backbone_subnet_experimental",
                        "quant_scope": "backbone_only",
                        "calibration_source": "synthetic_shape_smoke",
                        "calibration_digest": "artifact_digest",
                        "calibrator": "onnxruntime_static_qdq_minmax_synthetic",
                        "fallback_policy": "none",
                        "layer_precision_summary": "/exdata/jichengzhi/s2_tvm/int8_route/s0_024/layer_precision_summary.json",
                        "engine_kind": "tvm_vm",
                        "engine_digest": "artifact_digest",
                        "measurement_source": "true_measurement_smoke",
                        "claim_status": "claimable_true_measurement_smoke",
                        "quality_gate_status": "int8_tvm_vm_latency_smoke_only",
                        "full_network_claim": False,
                        "schedule_policy": "int8_qdq_direct_tvm_vm",
                        "schedule_profile": "int8_qdq_direct_tvm_vm",
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/stage2_generate_original60_quant_state_coverage.py"),
                    "--output-root",
                    str(output_root),
                    "--original60-summary-csv",
                    str(summary_csv),
                    "--int8-latency-smoke-rows",
                    str(int8_rows),
                ],
                cwd=ROOT,
                env={"PYTHONPATH": str(ROOT)},
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            latency_rows = read_jsonl(output_root / "rows/latency_original60_quant_rows_v1.jsonl")
            int8_row = next(row for row in latency_rows if row["label"] == "s0_024" and row["precision"] == "int8")
            validate_lut_row(int8_row)
            self.assertEqual(int8_row["measurement_status"], "measured")
            self.assertEqual(int8_row["latency_p50_us"], 41040.342)
            self.assertEqual(int8_row["run_id"], "int8_smoke_s0_024_tvm_vm")
            self.assertEqual(int8_row["quality_gate_status"], "int8_tvm_vm_latency_smoke_only")
            self.assertEqual(int8_row["candidate_id"], "coverage:pyramid_lidar:w24x128x256:int8")
            self.assertEqual(int8_row["original60_candidate_id"], "coverage:pyramid_lidar:w24x128x256:fp16")
            self.assertFalse(int8_row["full_network_claim"])
            self.assertIsNone(int8_row.get("failure_reason"))

            summary = json.loads(
                (output_root / "exports/original60_quant_three_metric_summary_latest.json").read_text(
                    encoding="utf-8"
                )
            )
            int8_summary = next(
                row for row in summary["rows"] if row["label"] == "s0_024" and row["precision"] == "int8"
            )
            self.assertEqual(int8_summary["latency_status"], "measured")
            self.assertEqual(int8_summary["latency_ms"], 41.040342)
            self.assertEqual(int8_summary["candidate_id"], "coverage:pyramid_lidar:w24x128x256:int8")
            self.assertEqual(
                int8_summary["original60_candidate_id"],
                "coverage:pyramid_lidar:w24x128x256:fp16",
            )

    def test_original60_quant_state_coverage_prefers_native_int8_full_onnx_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_root = tmp_path / "original60_quant"
            summary_csv = tmp_path / "original60_summary.csv"
            qdq_latency_rows = tmp_path / "int8_qdq_latency_rows.jsonl"
            native_latency_rows = tmp_path / "native_int8_latency_rows.jsonl"
            native_energy_rows = tmp_path / "native_int8_energy_rows.jsonl"
            summary_csv.write_text(
                "\n".join(
                    [
                        "label,candidate_id,width,latency_status,latency_tuned_ms,energy_status,energy_j_per_inference,ap70,ap_source_kind,ap_status",
                        "s0_024,coverage:pyramid_lidar:w24x128x256:fp16,24x128x256,,,,,,,",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            qdq_latency_rows.write_text(
                json.dumps(
                    {
                        "candidate_id": "s0_024",
                        "label": "s0_024",
                        "precision": "int8",
                        "backend": "h800_tvm",
                        "measurement_status": "measured",
                        "latency_p50_us": 41040.342,
                        "run_id": "old_qdq",
                        "quant_method": "h800_tvm_int8_backbone_subnet_experimental",
                        "quant_scope": "backbone_only",
                        "engine_kind": "tvm_vm",
                        "full_network_claim": False,
                        "schedule_policy": "int8_qdq_direct_tvm_vm",
                        "schedule_profile": "int8_qdq_direct_tvm_vm",
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            native_latency_rows.write_text(
                json.dumps(
                    {
                        "candidate_id": "s0_024",
                        "label": "s0_024",
                        "precision": "int8",
                        "backend": "h800_tvm",
                        "measurement_status": "measured",
                        "latency_p50_us": 9210.731,
                        "run_id": "native_full_onnx_latency",
                        "source_files": ["raw/native/s0_024/latency_result.json"],
                        "raw_artifact": "raw/native/s0_024",
                        "quant_method": "h800_tvm_native_int8_backbone_subnet",
                        "quant_scope": "backbone_subnet_native_int8",
                        "calibration_source": "none_direct_native_int8_synthetic_inputs",
                        "calibration_digest": "none",
                        "calibrator": "none",
                        "fallback_policy": "none",
                        "layer_precision_summary": "raw/native/s0_024/tvm_operator_inventory.json",
                        "layer_precision_summary_digest": "digest",
                        "engine_kind": "tvm_graph_executor",
                        "engine_digest": "native_engine_digest",
                        "measurement_source": "true_measurement_smoke",
                        "claim_status": "claimable_true_measurement_smoke",
                        "quality_gate_status": "native_int8_full_onnx_topology_latency",
                        "full_network_claim": False,
                        "schedule_policy": "native_int8_full_onnx_graph_executor",
                        "schedule_profile": "native_int8_full_onnx_graph_executor",
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            native_energy_rows.write_text(
                json.dumps(
                    {
                        "candidate_id": "s0_024",
                        "label": "s0_024",
                        "precision": "int8",
                        "backend": "h800_tvm_power_telemetry",
                        "measurement_status": "measured",
                        "joule_per_inference": 2.4527137319477,
                        "run_id": "native_full_onnx_energy",
                        "latency_run_id": "native_full_onnx_latency",
                        "measurement_run_id": "native_full_onnx_energy",
                        "source_files": [
                            "raw/native/s0_024/energy_result.json",
                            "raw/native/s0_024/idle_power_samples.csv",
                            "raw/native/s0_024/active_power_samples.csv",
                        ],
                        "raw_artifact": "raw/native/s0_024",
                        "quant_method": "h800_tvm_native_int8_backbone_subnet",
                        "quant_scope": "backbone_subnet_native_int8",
                        "calibration_source": "none_direct_native_int8_synthetic_inputs",
                        "calibration_digest": "none",
                        "calibrator": "none",
                        "fallback_policy": "none",
                        "layer_precision_summary": "raw/native/s0_024/tvm_operator_inventory.json",
                        "layer_precision_summary_digest": "digest",
                        "engine_kind": "tvm_graph_executor",
                        "engine_digest": "native_engine_digest",
                        "measurement_source": "true_measurement_smoke",
                        "claim_status": "claimable_true_measurement_smoke",
                        "quality_gate_status": "native_int8_full_onnx_topology_energy",
                        "full_network_claim": False,
                        "idle_baseline_policy": "subtract_idle_avg_5s_pre_window",
                        "telemetry_source": "nvidia_smi_power_draw",
                        "schedule_policy": "native_int8_full_onnx_graph_executor",
                        "schedule_profile": "native_int8_full_onnx_graph_executor",
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/stage2_generate_original60_quant_state_coverage.py"),
                    "--output-root",
                    str(output_root),
                    "--original60-summary-csv",
                    str(summary_csv),
                    "--int8-latency-smoke-rows",
                    str(qdq_latency_rows),
                    "--native-int8-full-onnx-latency-rows",
                    str(native_latency_rows),
                    "--native-int8-full-onnx-energy-rows",
                    str(native_energy_rows),
                ],
                cwd=ROOT,
                env={"PYTHONPATH": str(ROOT)},
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            latency_rows = read_jsonl(output_root / "rows/latency_original60_quant_rows_v1.jsonl")
            energy_rows = read_jsonl(output_root / "rows/energy_original60_quant_rows_v1.jsonl")
            int8_latency = next(
                row for row in latency_rows if row["label"] == "s0_024" and row["precision"] == "int8"
            )
            int8_energy = next(
                row for row in energy_rows if row["label"] == "s0_024" and row["precision"] == "int8"
            )
            validate_lut_row(int8_latency)
            validate_lut_row(int8_energy)
            self.assertEqual(int8_latency["measurement_status"], "measured")
            self.assertEqual(int8_latency["latency_p50_us"], 9210.731)
            self.assertEqual(int8_latency["run_id"], "native_full_onnx_latency")
            self.assertEqual(int8_latency["engine_kind"], "tvm_graph_executor")
            self.assertEqual(int8_latency["quant_scope"], "backbone_subnet_native_int8")
            self.assertFalse(int8_latency["full_network_claim"])
            self.assertEqual(int8_energy["measurement_status"], "measured")
            self.assertEqual(int8_energy["joule_per_inference"], 2.4527137319477)
            self.assertEqual(int8_energy["run_id"], "native_full_onnx_energy")
            self.assertEqual(int8_energy["engine_kind"], "tvm_graph_executor")
            self.assertFalse(int8_energy["full_network_claim"])

            summary = json.loads(
                (output_root / "exports/original60_quant_three_metric_summary_latest.json").read_text(
                    encoding="utf-8"
                )
            )
            int8_summary = next(
                row for row in summary["rows"] if row["label"] == "s0_024" and row["precision"] == "int8"
            )
            self.assertEqual(int8_summary["latency_status"], "measured")
            self.assertEqual(int8_summary["latency_ms"], 9.210731)
            self.assertEqual(int8_summary["energy_status"], "measured")
            self.assertEqual(int8_summary["energy_j_per_inference"], 2.4527137319477)
            self.assertIn("native_int8_full_onnx", int8_summary["quality_gate_status"])

    def test_original60_quant_state_coverage_imports_compliant_fp16_and_int8_ap_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_root = tmp_path / "original60_quant"
            summary_csv = tmp_path / "original60_summary.csv"
            fp16_ap_rows = tmp_path / "fp16_ap_rows.jsonl"
            int8_ap_rows = tmp_path / "int8_ap_rows.jsonl"
            summary_csv.write_text(
                "\n".join(
                    [
                        "label,candidate_id,width,latency_status,latency_tuned_ms,energy_status,energy_j_per_inference,ap70,ap_source_kind,ap_status",
                        "s0_024,coverage:pyramid_lidar:w24x128x256:fp16,24x128x256,,,,,,,",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            fp16_ap_rows.write_text(
                json.dumps(
                    {
                        "schema": "ap_anchor_row_v1",
                        "candidate_id": "s0_024",
                        "label": "s0_024",
                        "width": [24, 128, 256],
                        "precision": "fp16",
                        "quant_policy": "fp16",
                        "backend": "model_eval",
                        "measurement_status": "measured",
                        "metric": "AP70",
                        "metric_value": 0.612345,
                        "secondary_metrics": {"AP30": 0.812345, "AP50": 0.712345},
                        "dataset": "DAIR-V2X",
                        "eval_split": "val",
                        "ckpt_path": "checkpoints/pyramid_lidar_true_fp16_s0_024.ckpt",
                        "ckpt_digest": "fp16_ckpt_digest",
                        "eval_command": "python tools/eval.py --precision true_fp16 --label s0_024",
                        "run_id": "fp16_true_ap_s0_024",
                        "source_files": [
                            "raw/ap_eval_original60/fp16_true_s0_024/eval_stdout.json",
                            "raw/ap_eval_original60/fp16_true_s0_024/metrics.json",
                        ],
                        "raw_artifact": "raw/ap_eval_original60/fp16_true_s0_024",
                        "quant_method": "h800_tvm_true_fp16_onnx_relax",
                        "quant_scope": "backbone_only",
                        "quant_scheme": "fp16_onnx_cast",
                        "calibration_source": "none",
                        "calibration_digest": "none",
                        "calibrator": "none",
                        "fallback_policy": "none",
                        "layer_precision_summary": "raw/ap_eval_original60/fp16_true_s0_024/layer_precision_summary.json",
                        "engine_kind": "model_eval",
                        "engine_digest": "fp16_eval_digest",
                        "measurement_source": "true_eval",
                        "claim_status": "claimable_true_eval",
                        "quality_gate_status": "true_fp16_original60_ap_eval",
                        "full_network_claim": False,
                        "schedule_policy": "not_applicable",
                        "schedule_profile": "not_applicable",
                        "tune_budget": "not_applicable",
                        "created_at": "2026-06-28T00:00:00Z",
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            int8_ap_rows.write_text(
                json.dumps(
                    {
                        "schema": "ap_anchor_row_v1",
                        "candidate_id": "s0_024",
                        "label": "s0_024",
                        "width": [24, 128, 256],
                        "precision": "int8",
                        "quant_policy": "int8",
                        "backend": "model_eval",
                        "measurement_status": "measured",
                        "metric": "AP70",
                        "metric_value": 0.598765,
                        "secondary_metrics": {"AP30": 0.798765, "AP50": 0.698765},
                        "dataset": "DAIR-V2X",
                        "eval_split": "val",
                        "ckpt_path": "checkpoints/pyramid_lidar_native_int8_s0_024.ckpt",
                        "ckpt_digest": "int8_ckpt_digest",
                        "eval_command": "python tools/eval.py --precision native_int8 --label s0_024",
                        "run_id": "native_int8_ap_s0_024",
                        "source_files": [
                            "raw/ap_eval_original60/native_int8_s0_024/eval_stdout.json",
                            "raw/ap_eval_original60/native_int8_s0_024/metrics.json",
                        ],
                        "raw_artifact": "raw/ap_eval_original60/native_int8_s0_024",
                        "quant_method": "h800_tvm_native_int8_backbone_subnet",
                        "quant_scope": "backbone_subnet_native_int8",
                        "quant_scheme": "tvm_native_int8_full_onnx_topology",
                        "calibration_source": "none_direct_native_int8_synthetic_inputs",
                        "calibration_digest": "none",
                        "calibrator": "none",
                        "fallback_policy": "none",
                        "layer_precision_summary": "raw/native/s0_024/tvm_operator_inventory.json",
                        "engine_kind": "model_eval",
                        "engine_digest": "native_int8_eval_digest",
                        "measurement_source": "true_eval",
                        "claim_status": "claimable_true_eval",
                        "quality_gate_status": "native_int8_full_onnx_original60_ap_eval",
                        "full_network_claim": False,
                        "schedule_policy": "not_applicable",
                        "schedule_profile": "not_applicable",
                        "tune_budget": "not_applicable",
                        "created_at": "2026-06-28T00:00:00Z",
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/stage2_generate_original60_quant_state_coverage.py"),
                    "--output-root",
                    str(output_root),
                    "--original60-summary-csv",
                    str(summary_csv),
                    "--fp16-ap-rows",
                    str(fp16_ap_rows),
                    "--int8-ap-rows",
                    str(int8_ap_rows),
                ],
                cwd=ROOT,
                env={"PYTHONPATH": str(ROOT)},
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            ap_rows = read_jsonl(output_root / "rows/ap_original60_quant_rows_v1.jsonl")
            fp16_ap = next(row for row in ap_rows if row["label"] == "s0_024" and row["precision"] == "fp16")
            int8_ap = next(row for row in ap_rows if row["label"] == "s0_024" and row["precision"] == "int8")
            validate_lut_row(fp16_ap)
            validate_lut_row(int8_ap)
            self.assertEqual(fp16_ap["measurement_status"], "measured")
            self.assertEqual(fp16_ap["metric_value"], 0.612345)
            self.assertEqual(fp16_ap["secondary_metrics"]["AP50"], 0.712345)
            self.assertEqual(fp16_ap["quant_method"], "h800_tvm_true_fp16_onnx_relax")
            self.assertEqual(fp16_ap["quality_gate_status"], "true_fp16_original60_ap_eval")
            self.assertEqual(int8_ap["measurement_status"], "measured")
            self.assertEqual(int8_ap["metric_value"], 0.598765)
            self.assertEqual(int8_ap["quant_method"], "h800_tvm_native_int8_backbone_subnet")
            self.assertEqual(int8_ap["quant_scope"], "backbone_subnet_native_int8")
            self.assertNotIn("predicted", fp16_ap["notes"].lower())
            self.assertNotIn("predicted", int8_ap["notes"].lower())
            self.assertFalse(fp16_ap["full_network_claim"])
            self.assertFalse(int8_ap["full_network_claim"])

            summary = json.loads(
                (output_root / "exports/original60_quant_three_metric_summary_latest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(summary["ap_status_counts"], {"measured": 2, "no_claim": 1})
            fp16_summary = next(
                row for row in summary["rows"] if row["label"] == "s0_024" and row["precision"] == "fp16"
            )
            int8_summary = next(
                row for row in summary["rows"] if row["label"] == "s0_024" and row["precision"] == "int8"
            )
            self.assertEqual(fp16_summary["ap_status"], "measured")
            self.assertEqual(fp16_summary["ap70"], 0.612345)
            self.assertEqual(int8_summary["ap_status"], "measured")
            self.assertEqual(int8_summary["ap70"], 0.598765)

    def test_original60_quant_state_coverage_rejects_non_compliant_ap_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_root = tmp_path / "original60_quant"
            summary_csv = tmp_path / "original60_summary.csv"
            fp16_ap_rows = tmp_path / "bad_fp16_ap_rows.jsonl"
            int8_ap_rows = tmp_path / "bad_int8_ap_rows.jsonl"
            summary_csv.write_text(
                "\n".join(
                    [
                        "label,candidate_id,width,latency_status,latency_tuned_ms,energy_status,energy_j_per_inference,ap70,ap_source_kind,ap_status",
                        "s0_024,coverage:pyramid_lidar:w24x128x256:fp16,24x128x256,,,,,,,",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            fp16_ap_rows.write_text(
                json.dumps(
                    {
                        "label": "s0_024",
                        "precision": "fp16",
                        "backend": "model_eval",
                        "measurement_status": "measured",
                        "metric": "AP70",
                        "metric_value": 0.9,
                        "secondary_metrics": {"AP30": 0.9, "AP50": 0.9},
                        "dataset": "DAIR-V2X",
                        "eval_split": "val",
                        "ckpt_path": "checkpoints/trt_reference.ckpt",
                        "ckpt_digest": "digest",
                        "eval_command": "python tools/eval.py --backend trt",
                        "source_files": ["raw/trt_reference/metrics.json"],
                        "raw_artifact": "raw/trt_reference",
                        "quant_method": "trt_fp16_reference",
                        "quant_scope": "backbone_only",
                        "engine_kind": "trt_reference",
                        "measurement_source": "true_eval",
                        "claim_status": "claimable_true_eval",
                        "quality_gate_status": "trt_reference_ap",
                        "full_network_claim": False,
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            int8_ap_rows.write_text(
                json.dumps(
                    {
                        "label": "s0_024",
                        "precision": "int8",
                        "backend": "model_eval",
                        "measurement_status": "measured",
                        "metric": "AP70",
                        "metric_value": 0.88,
                        "secondary_metrics": {"AP30": 0.88, "AP50": 0.88},
                        "dataset": "DAIR-V2X",
                        "eval_split": "val",
                        "ckpt_path": "checkpoints/native_int8.ckpt",
                        "ckpt_digest": "digest",
                        "eval_command": "python tools/eval.py --use-predicted-ap",
                        "source_files": ["raw/predicted/metrics.json"],
                        "raw_artifact": "raw/predicted",
                        "quant_method": "h800_tvm_native_int8_backbone_subnet",
                        "quant_scope": "backbone_subnet_native_int8",
                        "engine_kind": "model_eval",
                        "measurement_source": "predicted",
                        "claim_status": "claimable_true_eval",
                        "quality_gate_status": "predicted_model_fit_ap",
                        "full_network_claim": False,
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/stage2_generate_original60_quant_state_coverage.py"),
                    "--output-root",
                    str(output_root),
                    "--original60-summary-csv",
                    str(summary_csv),
                    "--fp16-ap-rows",
                    str(fp16_ap_rows),
                    "--int8-ap-rows",
                    str(int8_ap_rows),
                ],
                cwd=ROOT,
                env={"PYTHONPATH": str(ROOT)},
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            ap_rows = read_jsonl(output_root / "rows/ap_original60_quant_rows_v1.jsonl")
            fp16_ap = next(row for row in ap_rows if row["label"] == "s0_024" and row["precision"] == "fp16")
            int8_ap = next(row for row in ap_rows if row["label"] == "s0_024" and row["precision"] == "int8")
            validate_lut_row(fp16_ap)
            validate_lut_row(int8_ap)
            self.assertEqual(fp16_ap["measurement_status"], "no_claim")
            self.assertEqual(int8_ap["measurement_status"], "no_claim")
            self.assertEqual(fp16_ap["failure_reason"], "fp16_ap_requires_true_fp16_eval_source_revalidation")
            self.assertEqual(int8_ap["failure_reason"], "tvm_int8_ap_eval_backend_missing")
            self.assertIn("no-claim", fp16_ap["notes"])
            self.assertIn("no-claim", int8_ap["notes"])
            self.assertNotIn("imported from compliant", fp16_ap["notes"])
            self.assertNotIn("imported from compliant", int8_ap["notes"])

            summary = json.loads(
                (output_root / "exports/original60_quant_three_metric_summary_latest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(summary["ap_status_counts"], {"no_claim": 3})


class Stage2TvmInt8ArtifactRouteCliTest(unittest.TestCase):
    def test_int8_artifact_route_writes_base_s0_024_and_s1_048_missing_or_ready_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "int8_route"
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/stage2_probe_tvm_int8_artifact_route.py"),
                    "--output-root",
                    str(output_root),
                ],
                cwd=ROOT,
                env={"PYTHONPATH": str(ROOT)},
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            registry = read_jsonl(output_root / "artifacts/tvm_int8_artifact_registry_v1.jsonl")
            self.assertEqual({row["label"] for row in registry}, {"base", "s0_024", "s1_048"})
            for row in registry:
                self.assertEqual(row["precision"], "int8")
                self.assertEqual(row["quant_method"], "h800_tvm_int8_backbone_subnet_experimental")
                self.assertFalse(row["full_network_claim"])
                self.assertIn(row["artifact_status"], {"ready", "missing", "quarantine"})
                self.assertIn("artifact_path", row)
                self.assertIn("artifact_digest", row)
                self.assertIn("build_status", row)
                self.assertIn("validation_status", row)
                if row["artifact_status"] != "ready":
                    self.assertTrue(row["blocker"])

            summary = json.loads(
                (output_root / "exports/tvm_int8_artifact_status_latest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(summary["labels"], ["base", "s0_024", "s1_048"])
            self.assertEqual(summary["total_artifacts"], 3)
            self.assertTrue((output_root / "exports/tvm_int8_artifact_status_latest.md").exists())
            self.assertTrue((output_root / "exports/tvm_int8_artifact_status_latest.csv").exists())

    def test_int8_artifact_route_marks_global_h800_probe_failure_as_quarantine(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_root = tmp_path / "int8_route"
            probe = tmp_path / "probe_failed.json"
            probe.write_text(
                json.dumps(
                    {
                        "probe_source": "h800_readonly_probe_20260627",
                        "status": "ssh_failed",
                        "stderr": "kex_exchange_identification: Connection closed",
                    }
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/stage2_probe_tvm_int8_artifact_route.py"),
                    "--output-root",
                    str(output_root),
                    "--probe-json",
                    str(probe),
                ],
                cwd=ROOT,
                env={"PYTHONPATH": str(ROOT)},
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            registry = read_jsonl(output_root / "artifacts/tvm_int8_artifact_registry_v1.jsonl")
            self.assertEqual({row["artifact_status"] for row in registry}, {"quarantine"})
            self.assertTrue(all("ssh_failed" in row["blocker"] for row in registry))


class Stage2Fp32LatencySmokePreflightCliTest(unittest.TestCase):
    def test_fp32_latency_smoke_preflight_records_three_anchor_statuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "fp32_preflight"
            probe = Path(tmp) / "probe_failed.json"
            probe.write_text(
                json.dumps({"status": "ssh_failed", "stderr": "rate limited"}),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/stage2_record_fp32_latency_smoke_preflight.py"),
                    "--output-root",
                    str(output_root),
                    "--probe-json",
                    str(probe),
                ],
                cwd=ROOT,
                env={"PYTHONPATH": str(ROOT)},
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads(
                (output_root / "exports/fp32_latency_smoke_preflight_latest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(summary["labels"], ["base", "s0_024", "s1_048"])
            self.assertEqual(summary["total_preflight_rows"], 3)
            self.assertEqual(summary["status_counts"], {"ssh_failed_preflight_blocked": 3})
            self.assertTrue((output_root / "exports/fp32_latency_smoke_preflight_latest.md").exists())
            self.assertTrue((output_root / "exports/fp32_latency_smoke_preflight_latest.csv").exists())


class Stage2H800MeasurementJobQuantArgsTest(unittest.TestCase):
    def test_h800_measurement_job_passes_tvm_first_quant_contract_to_generators(self):
        from scripts import stage2_h800_run_measurement_job as runner

        args = SimpleNamespace(
            quant_policy="int8",
            optimized_scope="backbone_subnet",
            precision=None,
            quant_scheme=None,
            quant_method=None,
            quant_scope=None,
            calibration_source=None,
            calibration_digest=None,
            calibrator=None,
            calibration_inputs="spatial_features",
            fallback_policy=None,
            layer_precision_summary=None,
            full_network_claim="false",
            engine_kind=None,
            engine_digest=None,
            measurement_source=None,
            claim_status=None,
            quality_gate_status=None,
            tune_budget="unknown",
        )

        quant_args = runner.generator_quant_args(args, schedule_policy="metaschedule_tuned")

        self.assertIn("--precision", quant_args)
        self.assertIn("int8", quant_args)
        self.assertIn("--quant-method", quant_args)
        self.assertIn("h800_tvm_int8_backbone_subnet_experimental", quant_args)
        self.assertIn("--quant-scope", quant_args)
        self.assertIn("backbone_subnet", quant_args)
        self.assertIn("--full-network-claim", quant_args)
        self.assertIn("false", quant_args)
        self.assertIn("--engine-kind", quant_args)
        self.assertIn("tvm_vm", quant_args)
        self.assertIn("--layer-precision-summary", quant_args)
        self.assertIn("unknown_pending_tvm_inventory", quant_args)

    def test_h800_measurement_job_accepts_default_energy_schedule_policy(self):
        from scripts import stage2_h800_run_measurement_job as runner

        old_argv = sys.argv
        sys.argv = [
            "stage2_h800_run_measurement_job.py",
            "--kind",
            "energy",
            "--label",
            "frontier_25",
            "--gpu",
            "0",
            "--onnx",
            "/exdata/models/frontier_25_backbone.onnx",
            "--work-dir",
            "/exdata/workdirs/frontier_25",
            "--width",
            "32,64,96",
            "--candidate-id",
            "coverage:pyramid_lidar:w32x64x96:fp32",
            "--software-point-id",
            "original60:frontier_25:32x64x96:fp32:energy",
            "--config-id-tuned",
            "frontier_25_fp32_energy_default",
            "--run-id",
            "frontier_25_default_energy",
            "--raw-root",
            "/tmp/raw",
            "--out-jsonl",
            "/tmp/energy.jsonl",
            "--energy-schedule-policy",
            "default",
        ]
        try:
            args = runner.parse_args()
        finally:
            sys.argv = old_argv

        self.assertEqual(args.energy_schedule_policy, "default")

        payload = runner.energy_payload(
            {
                "joule_per_inference": 1.25,
                "watt_avg": 300.0,
                "watt_p50": 301.0,
                "watt_p90": 305.0,
                "idle_watt_avg": 120.0,
                "sample_window_ms": 5000,
            },
            Path("/tmp/raw/frontier_25_default_energy"),
            args,
        )

        self.assertIn("default", payload["provenance"])
        self.assertIn("default", payload["notes"])

    def test_h800_measurement_job_accepts_threaded_energy_sampling_mode(self):
        from scripts import stage2_h800_run_measurement_job as runner

        old_argv = sys.argv
        sys.argv = [
            "stage2_h800_run_measurement_job.py",
            "--kind",
            "energy",
            "--label",
            "s0_024",
            "--gpu",
            "0",
            "--onnx",
            "/exdata/models/s0_024_backbone.onnx",
            "--work-dir",
            "/exdata/workdirs/s0_024",
            "--width",
            "24,128,256",
            "--candidate-id",
            "coverage:pyramid_lidar:w24x128x256:fp32",
            "--software-point-id",
            "original60:s0_024:24x128x256:fp32:energy",
            "--config-id-tuned",
            "s0_024_fp32_energy_threaded",
            "--run-id",
            "s0_024_threaded_energy",
            "--raw-root",
            "/tmp/raw",
            "--out-jsonl",
            "/tmp/energy.jsonl",
            "--energy-sampling-mode",
            "threaded_window",
        ]
        try:
            args = runner.parse_args()
        finally:
            sys.argv = old_argv

        self.assertEqual(args.energy_sampling_mode, "threaded_window")

        payload = runner.energy_payload(
            {
                "joule_per_inference": 8.25,
                "watt_avg": 360.0,
                "watt_p50": 361.0,
                "watt_p90": 365.0,
                "idle_watt_avg": 140.0,
                "sample_window_ms": 5000,
                "energy_sampling_mode": "threaded_window",
                "completed_measure_iters": 300,
                "min_active_s": 5.0,
            },
            Path("/tmp/raw/s0_024_threaded_energy"),
            args,
        )

        self.assertEqual(payload["energy_sampling_mode"], "threaded_window")
        self.assertEqual(payload["completed_measure_iters"], 300)
        self.assertIn("threaded_window", payload["notes"])


class Stage2EnergyClaimGateTest(unittest.TestCase):
    def _registry_dict(self, energy_path: Path, *, measured: bool = True) -> dict[str, object]:
        status = "measured" if measured else "not_done"
        return {
            "schema": "stage2_evidence_registry_v1",
            "model": "pyramid_lidar",
            "hardware_target": {"name": "H800 Hopper", "backend": "h800_tvm"},
            "manifest": {
                "path": None,
                "measurement_status": "not_available",
                "backend": "not_available",
                "scope": "rsu_dense_core",
                "provenance": "test",
            },
            "latency_lut": {
                "path": None,
                "measurement_status": "not_available",
                "backend": "not_available",
                "scope": "dense_core",
                "provenance": "test",
            },
            "ap_anchors": {
                "path": None,
                "measurement_status": "not_available",
                "backend": "not_available",
                "scope": "model_accuracy",
                "provenance": "test",
            },
            "quant_evidence": {
                "path": None,
                "measurement_status": "not_available",
                "backend": "not_available",
                "scope": "quant",
                "provenance": "test",
            },
            "energy_lut": {
                "path": str(energy_path),
                "measurement_status": status,
                "backend": "h800_tvm_power_telemetry",
                "scope": "dense_core",
                "provenance": "test energy telemetry",
                "coverage": {
                    "expected_cells": 1,
                    "measured_cells": 1 if measured else 0,
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
            },
            "unsupported_conclusions": [],
        }

    def _measured_energy_row(self) -> dict[str, object]:
        return energy_lut_row(
            config_id="cfg_energy",
            model="pyramid_lidar",
            manifest_digest="c" * 64,
            candidate_id="neck",
            software_point_id="neck:w64:fp16",
            dense_stage="neck",
            width=[64, 128, 256],
            quant_policy="fp16",
            schedule_policy="default",
            backend="h800_tvm_power_telemetry",
            measurement_status="measured",
            joule_per_inference=1.2,
            watt_avg=240.0,
            telemetry_source="nvidia_smi",
            idle_baseline_policy="subtract_idle_avg",
            sample_window_ms=5000,
            latency_run_id="run_001",
            raw_artifact="results/raw/energy.json",
            run_id="energy_001",
            measurement_run_id="energy_001",
            row_source="direct_generation",
            created_at="2026-06-25T00:00:00+08:00",
        )

    def test_energy_claim_gate_requires_measured_telemetry_row(self):
        measured = self._measured_energy_row()
        failed = dict(measured)
        failed.update(
            {
                "row_id": "energy:failed",
                "measurement_status": "failed",
                "joule_per_inference": None,
                "failure_reason": "telemetry_failed",
            }
        )

        self.assertFalse(energy_claim_allowed_from_rows([]))
        self.assertFalse(energy_claim_allowed_from_rows([failed]))
        self.assertTrue(
            energy_claim_allowed_from_rows([measured], telemetry_source="nvidia_smi")
        )
        self.assertFalse(
            energy_claim_allowed_from_rows([measured], telemetry_source="nvml")
        )

    def test_registry_energy_claim_fails_closed_without_measured_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            empty = tmp_path / "energy_lut_rows_v1.jsonl"
            empty.write_text("", encoding="utf-8")

            registry = Stage2EvidenceRegistry.from_dict(
                self._registry_dict(empty),
                base_dir=tmp_path,
            )

            self.assertFalse(registry.energy_claim_allowed)
            self.assertIn(
                "energy_improvement_without_energy_lut",
                registry.unsupported_conclusions,
            )

    def test_registry_energy_claim_allows_measured_telemetry_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            energy_path = tmp_path / "energy_lut_rows_v1.jsonl"
            write_jsonl(energy_path, [self._measured_energy_row()])

            registry = Stage2EvidenceRegistry.from_dict(
                self._registry_dict(energy_path),
                base_dir=tmp_path,
            )

            self.assertTrue(registry.energy_claim_allowed)


class Stage2LutImporterCliTest(unittest.TestCase):
    def test_latency_and_ap_importers_write_canonical_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            latency_out = tmp_path / "latency_lut_rows_v1.jsonl"
            ap_out = tmp_path / "ap_anchor_rows_v1.jsonl"

            latency = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/stage2_import_latency_lut.py"),
                    "--model",
                    "pyramid_lidar",
                    "--source-json",
                    str(ROOT / "results/latency_lut_pyramid.json"),
                    "--out-jsonl",
                    str(latency_out),
                    "--backend",
                    "h800_tvm",
                    "--measurement-status",
                    "measured",
                ],
                cwd=ROOT,
                env={"PYTHONPATH": str(ROOT)},
                capture_output=True,
                text=True,
            )
            self.assertEqual(latency.returncode, 0, latency.stderr)
            first_latency = json.loads(latency_out.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(first_latency["schema"], "latency_lut_row_v1")
            self.assertEqual(first_latency["backend"], "h800_tvm")

            ap = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/stage2_import_ap_anchors.py"),
                    "--model",
                    "pyramid_lidar",
                    "--source-json",
                    str(ROOT / "results/ap70_model_pyramid.json"),
                    "--out-jsonl",
                    str(ap_out),
                    "--metric",
                    "AP70",
                    "--dataset",
                    "DAIR-V2X",
                    "--eval-split",
                    "val",
                    "--finetune-protocol",
                    "mixed_existing_anchors",
                ],
                cwd=ROOT,
                env={"PYTHONPATH": str(ROOT)},
                capture_output=True,
                text=True,
            )
            self.assertEqual(ap.returncode, 0, ap.stderr)
            first_ap = json.loads(ap_out.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(first_ap["schema"], "ap_anchor_row_v1")
            self.assertEqual(first_ap["metric"], "AP70")

    def test_energy_importer_writes_measured_telemetry_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_csv = tmp_path / "energy.csv"
            out_jsonl = tmp_path / "energy_lut_rows_v1.jsonl"
            source_csv.write_text(
                "\n".join(
                    [
                        "config_id,model,candidate_id,software_point_id,dense_stage,width,quant_policy,schedule_policy,joule_per_inference,watt_avg,telemetry_source,latency_run_id",
                        "cfg,pyramid_lidar,neck,neck:w64:fp16,neck,64|128|256,fp16,default,1.2,240.0,nvidia_smi,run_001",
                    ]
                ),
                encoding="utf-8",
            )

            energy = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/stage2_import_energy_lut.py"),
                    "--source-csv",
                    str(source_csv),
                    "--out-jsonl",
                    str(out_jsonl),
                ],
                cwd=ROOT,
                env={"PYTHONPATH": str(ROOT)},
                capture_output=True,
                text=True,
            )

            self.assertEqual(energy.returncode, 0, energy.stderr)
            row = json.loads(out_jsonl.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(row["schema"], "energy_lut_row_v1")
            self.assertEqual(row["measurement_status"], "measured")
            self.assertEqual(row["backend"], "h800_tvm_power_telemetry")


class Stage2LutRegistryUpdateTest(unittest.TestCase):
    def test_coverage_counts_measured_failed_and_proxy_rows(self):
        rows = [
            {"schema": "latency_lut_row_v1", "measurement_status": "measured"},
            {"schema": "latency_lut_row_v1", "measurement_status": "proxy"},
            {"schema": "latency_lut_row_v1", "measurement_status": "estimated"},
            {
                "schema": "latency_lut_row_v1",
                "measurement_status": "failed",
                "failure_reason": "build_failed",
            },
        ]

        self.assertEqual(
            coverage_from_rows(rows, expected_cells=6),
            {
                "expected_cells": 6,
                "measured_cells": 1,
                "failed_cells": 1,
                "proxy_cells": 2,
            },
        )

    def test_registry_updater_refreshes_lut_sources_and_claim_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            registry_path = tmp_path / "evidence_registry.json"
            latency_rows = tmp_path / "latency_lut_rows_v1.jsonl"
            ap_rows = tmp_path / "ap_anchor_rows_v1.jsonl"
            energy_rows = tmp_path / "energy_lut_rows_v1.jsonl"
            out_registry = tmp_path / "updated_registry.json"

            common = {
                "config_id": "cfg_shared",
                "model": "pyramid_lidar",
                "manifest_digest": "d" * 64,
                "candidate_id": "neck",
                "software_point_id": "neck:w64:fp16",
                "dense_stage": "neck",
                "width": [64, 128, 256],
                "quant_policy": "fp16",
                "run_id": "run_001",
                "created_at": "2026-06-25T00:00:00+08:00",
            }
            write_jsonl(
                latency_rows,
                [
                    latency_lut_row(
                        **common,
                        schedule_policy="default",
                        backend="h800_tvm",
                        measurement_status="measured",
                        latency_p50_us=123.4,
                        warmup_iters=50,
                        measure_iters=200,
                        repeat=5,
                    )
                ],
            )
            write_jsonl(
                ap_rows,
                [
                    ap_anchor_row(
                        **common,
                        schedule_policy="not_applicable",
                        backend="model_eval",
                        measurement_status="measured",
                        metric="AP70",
                        metric_value=0.63,
                        dataset="DAIR-V2X",
                        eval_split="val",
                        ckpt_path="ckpts/smoke.pt",
                        finetune_protocol="none",
                    )
                ],
            )
            write_jsonl(
                energy_rows,
                [
                    energy_lut_row(
                        **common,
                        schedule_policy="default",
                        backend="h800_tvm_power_telemetry",
                        measurement_status="measured",
                        joule_per_inference=1.2,
                        watt_avg=240.0,
                        telemetry_source="nvidia_smi",
                        idle_baseline_policy="subtract_idle_avg",
                        sample_window_ms=5000,
                        latency_run_id="run_001",
                        raw_artifact="energy_raw.json",
                        measurement_run_id="run_001",
                        row_source="direct_generation",
                    )
                ],
            )
            registry_path.write_text(
                json.dumps(
                    {
                        "schema": "stage2_evidence_registry_v1",
                        "model": "pyramid_lidar",
                        "hardware_target": {"name": "H800 Hopper", "backend": "h800_tvm"},
                        "manifest": {
                            "path": None,
                            "measurement_status": "not_available",
                            "backend": "not_available",
                            "scope": "rsu_dense_core",
                            "provenance": "test",
                        },
                        "latency_lut": {
                            "path": None,
                            "measurement_status": "not_done",
                            "backend": "not_available",
                            "scope": "dense_core",
                            "provenance": "test",
                        },
                        "ap_anchors": {
                            "path": None,
                            "measurement_status": "not_done",
                            "backend": "not_available",
                            "scope": "model_accuracy",
                            "provenance": "test",
                        },
                        "quant_evidence": {
                            "path": None,
                            "measurement_status": "not_available",
                            "backend": "not_available",
                            "scope": "quant",
                            "provenance": "test",
                        },
                        "energy_lut": {
                            "path": None,
                            "measurement_status": "not_done",
                            "backend": "not_available",
                            "scope": "dense_core",
                            "provenance": "test",
                        },
                        "downstream_objective": {
                            "path": None,
                            "measurement_status": "not_available",
                            "backend": "not_available",
                            "scope": {},
                            "provenance": "test",
                        },
                        "unsupported_conclusions": [],
                    }
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/stage2_update_registry_from_luts.py"),
                    "--registry",
                    str(registry_path),
                    "--latency-rows",
                    str(latency_rows),
                    "--ap-rows",
                    str(ap_rows),
                    "--energy-rows",
                    str(energy_rows),
                    "--out-json",
                    str(out_registry),
                ],
                cwd=ROOT,
                env={"PYTHONPATH": str(ROOT)},
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            updated = json.loads(out_registry.read_text(encoding="utf-8"))
            self.assertTrue(updated["latency_lut"]["path"].endswith("latency_lut_rows_v1.jsonl"))
            self.assertEqual(updated["latency_lut"]["coverage"]["measured_cells"], 1)
            self.assertEqual(updated["ap_anchors"]["coverage"]["measured_cells"], 1)
            self.assertEqual(updated["energy_lut"]["coverage"]["measured_cells"], 1)
            self.assertEqual(updated["energy_lut"]["measurement_status"], "measured")
            self.assertEqual(updated["energy_lut"]["backend"], "h800_tvm_power_telemetry")


class Stage2LutWorkerTest(unittest.TestCase):
    def test_worker_resume_skips_succeeded_jobs_and_runs_next_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            plan_path = tmp_path / "lut_job_plan_v1.jsonl"
            state_path = tmp_path / "lut_job_state_v1.jsonl"
            marker = tmp_path / "worker_marker.txt"
            command = [
                sys.executable,
                "-c",
                (
                    "from pathlib import Path; "
                    f"Path({str(marker)!r}).write_text('new', encoding='utf-8')"
                ),
            ]
            write_jsonl(
                plan_path,
                [
                    job_plan_row(
                        job_id="latency:done",
                        model="pyramid_lidar",
                        lut_kind="latency",
                        job_type="generate_latency_lut",
                        priority=10,
                        config_id="done",
                        manifest_path="m",
                        registry_path="r",
                        expected_output=str(tmp_path / "out.jsonl"),
                        command=[
                            sys.executable,
                            "-c",
                            "raise SystemExit('should not rerun')",
                        ],
                        max_attempts=1,
                        timeout_s=30,
                    ),
                    job_plan_row(
                        job_id="latency:new",
                        model="pyramid_lidar",
                        lut_kind="latency",
                        job_type="generate_latency_lut",
                        priority=9,
                        config_id="new",
                        manifest_path="m",
                        registry_path="r",
                        expected_output=str(tmp_path / "out.jsonl"),
                        command=command,
                        max_attempts=1,
                        timeout_s=30,
                    ),
                ],
            )
            write_jsonl(
                state_path,
                [
                    {
                        "schema": "lut_job_state_row_v1",
                        "job_id": "latency:done",
                        "status": "succeeded",
                        "attempt": 1,
                    }
                ],
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/stage2_lut_worker.py"),
                    "--job-plan",
                    str(plan_path),
                    "--job-state",
                    str(state_path),
                    "--max-jobs",
                    "1",
                    "--resume",
                ],
                cwd=ROOT,
                env={"PYTHONPATH": str(ROOT)},
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(marker.read_text(encoding="utf-8"), "new")
            state_rows = read_jsonl(state_path)
            done_successes = [
                row
                for row in state_rows
                if row["job_id"] == "latency:done" and row["status"] == "succeeded"
            ]
            new_successes = [
                row
                for row in state_rows
                if row["job_id"] == "latency:new" and row["status"] == "succeeded"
            ]
            self.assertEqual(len(done_successes), 1)
            self.assertEqual(len(new_successes), 1)

    def test_worker_continues_after_cuda_failure_and_quarantines_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            plan_path = tmp_path / "lut_job_plan_v1.jsonl"
            state_path = tmp_path / "lut_job_state_v1.jsonl"
            quarantine_path = tmp_path / "bad_db_quarantine_v1.jsonl"
            marker = tmp_path / "continued.txt"
            bad_command = [
                sys.executable,
                "-c",
                (
                    "import sys; "
                    "print('CUDA_ERROR_ILLEGAL_ADDRESS in tuned VM', file=sys.stderr); "
                    "raise SystemExit(17)"
                ),
            ]
            good_command = [
                sys.executable,
                "-c",
                (
                    "from pathlib import Path; "
                    f"Path({str(marker)!r}).write_text('continued', encoding='utf-8')"
                ),
            ]
            write_jsonl(
                plan_path,
                [
                    job_plan_row(
                        job_id="latency:s1_64_tuned",
                        model="pyramid_lidar",
                        lut_kind="latency",
                        job_type="generate_latency_lut",
                        priority=20,
                        config_id="s1_64",
                        manifest_path="m",
                        registry_path="r",
                        expected_output=str(tmp_path / "latency.jsonl"),
                        command=bad_command,
                        max_attempts=1,
                        timeout_s=30,
                    ),
                    job_plan_row(
                        job_id="latency:mix_b_tuned",
                        model="pyramid_lidar",
                        lut_kind="latency",
                        job_type="generate_latency_lut",
                        priority=10,
                        config_id="mix_b",
                        manifest_path="m",
                        registry_path="r",
                        expected_output=str(tmp_path / "latency.jsonl"),
                        command=good_command,
                        max_attempts=1,
                        timeout_s=30,
                    ),
                ],
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/stage2_lut_worker.py"),
                    "--job-plan",
                    str(plan_path),
                    "--job-state",
                    str(state_path),
                    "--quarantine-db",
                    str(quarantine_path),
                    "--max-jobs",
                    "2",
                ],
                cwd=ROOT,
                env={"PYTHONPATH": str(ROOT)},
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(marker.read_text(encoding="utf-8"), "continued")
            quarantine_rows = read_jsonl(quarantine_path)
            self.assertEqual(len(quarantine_rows), 1)
            self.assertEqual(quarantine_rows[0]["config_id"], "s1_64")
            self.assertEqual(quarantine_rows[0]["job_id"], "latency:s1_64_tuned")
            self.assertIn("CUDA_ERROR_ILLEGAL_ADDRESS", quarantine_rows[0]["failure_reason"])

    def test_worker_salvages_energy_payload_after_generator_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            plan_path = tmp_path / "lut_job_plan_v1.jsonl"
            state_path = tmp_path / "lut_job_state_v1.jsonl"
            latency_path = tmp_path / "latency_rows.jsonl"
            energy_out = tmp_path / "energy_rows.jsonl"
            raw_dir = tmp_path / "raw/energy_payload_job"
            raw_dir.mkdir(parents=True)
            latency_row = latency_lut_row(
                config_id="cfg_energy_salvage",
                model="pyramid_lidar",
                manifest_digest="e" * 64,
                candidate_id="calibration:pyramid_lidar:base",
                software_point_id="pyramid_lidar:backbone:w64x128x256:fp16",
                dense_stage="backbone",
                optimized_scope="backbone_only",
                width=[64, 128, 256],
                quant_policy="fp16",
                schedule_policy="metaschedule_tuned",
                backend="h800_tvm",
                measurement_status="measured",
                latency_p50_us=6210.0,
                latency_min_us=6209.0,
                latency_max_us=6211.0,
                warmup_iters=50,
                measure_iters=200,
                repeat=5,
                raw_artifact=str(tmp_path / "raw/latency"),
                run_id="latency_salvage_run",
                created_at="2026-06-25T00:00:00+08:00",
            )
            write_jsonl(latency_path, [latency_row])
            payload = {
                "run_id": "energy_salvage_run",
                "latency_run_id": "latency_salvage_run",
                "joule_per_inference": 1.5,
                "watt_avg": 300.0,
                "telemetry_source": "nvidia_smi",
                "idle_baseline_policy": "subtract_idle_avg",
                "raw_artifact": str(raw_dir),
            }
            command = [
                sys.executable,
                "-c",
                (
                    "import json, pathlib; "
                    f"pathlib.Path({str(raw_dir / 'telemetry_payload.json')!r}).write_text("
                    f"json.dumps({payload!r}), encoding='utf-8'); "
                    "raise SystemExit(17)"
                ),
            ]
            write_jsonl(
                plan_path,
                [
                    job_plan_row(
                        job_id="energy:cfg_energy_salvage",
                        model="pyramid_lidar",
                        lut_kind="energy",
                        job_type="generate_energy_lut",
                        priority=10,
                        config_id="cfg_energy_salvage",
                        manifest_path="m",
                        registry_path="r",
                        expected_output=str(energy_out),
                        command=command,
                        max_attempts=1,
                        timeout_s=30,
                        resource={
                            "energy_salvage": {
                                "payload_root": str(tmp_path / "raw"),
                                "latency_rows": [str(latency_path)],
                                "out_jsonl": str(energy_out),
                            }
                        },
                    )
                ],
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/stage2_lut_worker.py"),
                    "--job-plan",
                    str(plan_path),
                    "--job-state",
                    str(state_path),
                    "--max-jobs",
                    "1",
                ],
                cwd=ROOT,
                env={"PYTHONPATH": str(ROOT)},
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                latest_job_status(read_jsonl(state_path), "energy:cfg_energy_salvage"),
                "succeeded",
            )
            rows = read_jsonl(energy_out)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["row_source"], "salvaged_from_payload")

    def test_worker_skips_quarantined_job_without_running_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            plan_path = tmp_path / "lut_job_plan_v1.jsonl"
            state_path = tmp_path / "lut_job_state_v1.jsonl"
            quarantine_path = tmp_path / "bad_db_quarantine_v1.jsonl"
            blocked_marker = tmp_path / "blocked.txt"
            allowed_marker = tmp_path / "allowed.txt"
            write_jsonl(
                quarantine_path,
                [
                    {
                        "schema": "lut_bad_db_quarantine_row_v1",
                        "job_id": "energy:pad64_tuned",
                        "config_id": "pad64",
                        "model": "pyramid_lidar",
                        "lut_kind": "energy",
                        "job_type": "generate_energy_lut",
                        "status": "active",
                        "failure_reason": "CUDA illegal memory access",
                        "created_at": "2026-06-25T00:00:00+08:00",
                    }
                ],
            )
            write_jsonl(
                plan_path,
                [
                    job_plan_row(
                        job_id="energy:pad64_tuned",
                        model="pyramid_lidar",
                        lut_kind="energy",
                        job_type="generate_energy_lut",
                        priority=20,
                        config_id="pad64",
                        manifest_path="m",
                        registry_path="r",
                        expected_output=str(tmp_path / "energy.jsonl"),
                        command=[
                            sys.executable,
                            "-c",
                            (
                                "from pathlib import Path; "
                                f"Path({str(blocked_marker)!r}).write_text('bad')"
                            ),
                        ],
                        max_attempts=1,
                        timeout_s=30,
                    ),
                    job_plan_row(
                        job_id="energy:p75_tuned",
                        model="pyramid_lidar",
                        lut_kind="energy",
                        job_type="generate_energy_lut",
                        priority=10,
                        config_id="p75",
                        manifest_path="m",
                        registry_path="r",
                        expected_output=str(tmp_path / "energy.jsonl"),
                        command=[
                            sys.executable,
                            "-c",
                            (
                                "from pathlib import Path; "
                                f"Path({str(allowed_marker)!r}).write_text('ok')"
                            ),
                        ],
                        max_attempts=1,
                        timeout_s=30,
                    ),
                ],
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/stage2_lut_worker.py"),
                    "--job-plan",
                    str(plan_path),
                    "--job-state",
                    str(state_path),
                    "--quarantine-db",
                    str(quarantine_path),
                ],
                cwd=ROOT,
                env={"PYTHONPATH": str(ROOT)},
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(blocked_marker.exists())
            self.assertEqual(allowed_marker.read_text(encoding="utf-8"), "ok")
            skipped = [
                row
                for row in read_jsonl(state_path)
                if row["job_id"] == "energy:pad64_tuned" and row["status"] == "skipped"
            ]
            self.assertEqual(len(skipped), 1)
            self.assertEqual(skipped[0]["failure_reason"], "quarantined_bad_db")

    def test_worker_injects_tvm_runtime_environment_into_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            plan_path = tmp_path / "lut_job_plan_v1.jsonl"
            state_path = tmp_path / "lut_job_state_v1.jsonl"
            env_path = tmp_path / "env.json"
            write_jsonl(
                plan_path,
                [
                    job_plan_row(
                        job_id="latency:env",
                        model="pyramid_lidar",
                        lut_kind="latency",
                        job_type="generate_latency_lut",
                        priority=10,
                        config_id="env",
                        manifest_path="m",
                        registry_path="r",
                        expected_output=str(tmp_path / "latency.jsonl"),
                        command=[
                            sys.executable,
                            "-c",
                            (
                                "import json, os; "
                                f"open({str(env_path)!r}, 'w', encoding='utf-8').write("
                                "json.dumps({'ld': os.environ.get('LD_LIBRARY_PATH', ''), "
                                "'path': os.environ.get('PATH', '')}))"
                            ),
                        ],
                        max_attempts=1,
                        timeout_s=30,
                    )
                ],
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/stage2_lut_worker.py"),
                    "--job-plan",
                    str(plan_path),
                    "--job-state",
                    str(state_path),
                    "--max-jobs",
                    "1",
                ],
                cwd=ROOT,
                env={"PYTHONPATH": str(ROOT), "PATH": os.environ.get("PATH", "")},
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            child_env = json.loads(env_path.read_text(encoding="utf-8"))
            self.assertTrue(
                child_env["ld"].startswith(
                    "/exdata/jichengzhi/tvm310/lib/python3.10/site-packages/"
                    "nvidia/cuda_runtime/lib:"
                    "/exdata/jichengzhi/tvm310/lib/python3.10/site-packages/tvm/lib"
                )
            )
            self.assertTrue(child_env["path"].startswith("/usr/local/cuda-12.2/bin:"))

    def test_worker_preflight_blocks_busy_numeric_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fake_bin = tmp_path / "bin"
            fake_bin.mkdir()
            fake_nvidia_smi = fake_bin / "nvidia-smi"
            fake_nvidia_smi.write_text(
                "\n".join(
                    [
                        "#!/usr/bin/env python3",
                        "import sys",
                        "args = ' '.join(sys.argv[1:])",
                        "if 'pmon' in args:",
                        "    print('# gpu pid type sm mem enc dec command')",
                        "    print('1 12345 C 0 0 0 0 python')",
                        "else:",
                        "    print('index, name, utilization.gpu, memory.used, "
                        "memory.total, power.draw, pstate')",
                        "    print('1, NVIDIA H800, 9, 2048, 81559, 150, P0')",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            fake_nvidia_smi.chmod(0o755)
            plan_path = tmp_path / "lut_job_plan_v1.jsonl"
            state_path = tmp_path / "lut_job_state_v1.jsonl"
            marker = tmp_path / "should_not_run.txt"
            write_jsonl(
                plan_path,
                [
                    job_plan_row(
                        job_id="latency:busy_gpu",
                        model="pyramid_lidar",
                        lut_kind="latency",
                        job_type="generate_latency_lut",
                        priority=10,
                        config_id="busy_gpu",
                        manifest_path="m",
                        registry_path="r",
                        expected_output=str(tmp_path / "latency.jsonl"),
                        command=[
                            sys.executable,
                            "-c",
                            (
                                "from pathlib import Path; "
                                f"Path({str(marker)!r}).write_text('bad')"
                            ),
                        ],
                        max_attempts=1,
                        timeout_s=30,
                        resource={"gpu": "1", "exclusive": True},
                    )
                ],
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/stage2_lut_worker.py"),
                    "--job-plan",
                    str(plan_path),
                    "--job-state",
                    str(state_path),
                    "--max-jobs",
                    "1",
                    "--require-gpu-idle",
                ],
                cwd=ROOT,
                env={
                    "PYTHONPATH": str(ROOT),
                    "PATH": f"{fake_bin}:{os.environ.get('PATH', '')}",
                },
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(marker.exists())
            rows = read_jsonl(state_path)
            self.assertEqual(rows[-1]["job_id"], "latency:busy_gpu")
            self.assertEqual(rows[-1]["status"], "preflight_blocked")
            self.assertIn("gpu_not_idle", rows[-1]["failure_reason"])
