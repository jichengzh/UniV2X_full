from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

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
