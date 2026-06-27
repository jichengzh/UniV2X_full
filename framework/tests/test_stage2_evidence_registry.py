from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from framework.stage2.evidence_registry import (
    EvidenceRegistryError,
    Stage2EvidenceRegistry,
)


ROOT = Path(__file__).resolve().parents[2]
PYRAMID_MANIFEST = ROOT / "framework/partitions/pyramid_lidar_partition.yaml"
CODRIVING_MANIFEST = ROOT / "framework/partitions/codriving_partition.yaml"
PYRAMID_LATENCY = ROOT / "results/latency_lut_pyramid.json"
PYRAMID_AP = ROOT / "results/ap70_model_pyramid.json"
PYRAMID_Q = ROOT / "results/latency_lut_pyramid_q.json"
CODRIVING_LATENCY = ROOT / "results/latency_lut_codriving.json"
CODRIVING_AP = ROOT / "results/ap70_model_codriving.json"
DS_MAP = ROOT / "multi_agent/real_test/ds_ap_latency_all_measured.csv"


class Stage2EvidenceRegistryTest(unittest.TestCase):
    def _write_registry(
        self,
        tmp_path: Path,
        *,
        model: str = "pyramid_lidar",
        manifest: Path = PYRAMID_MANIFEST,
        latency: Path | None = PYRAMID_LATENCY,
        ap: Path | None = PYRAMID_AP,
        q: Path | None = PYRAMID_Q,
        latency_backend: str = "h800_tvm",
        latency_status: str = "measured",
        q_backend: str = "historical_trt",
        q_status: str = "historical",
        energy_status: str = "not_available",
        downstream_model: str = "codriving",
    ) -> Path:
        data = {
            "schema": "stage2_evidence_registry_v1",
            "model": model,
            "hardware_target": {
                "name": "H800 Hopper",
                "backend": "h800_tvm",
            },
            "manifest": {
                "path": str(manifest),
                "scope": "rsu_dense_core",
                "provenance": "stage1_partition_manifest",
            },
            "latency_lut": {
                "path": None if latency is None else str(latency),
                "measurement_status": latency_status,
                "backend": latency_backend,
                "scope": "dense_core",
                "provenance": "H800 TVM MetaSchedule measured LUT",
                "coverage": {
                    "expected_cells": 24,
                    "measured_cells": 24,
                    "failed_cells": 0,
                    "proxy_cells": 0,
                },
            },
            "ap_anchors": {
                "path": None if ap is None else str(ap),
                "measurement_status": "measured",
                "backend": "model_eval",
                "scope": "model_accuracy",
                "provenance": "DAIR validation AP anchors",
                "metric": "AP70",
                "coverage": {
                    "expected_cells": 10,
                    "measured_cells": 10,
                    "failed_cells": 0,
                    "proxy_cells": 0,
                },
            },
            "quant_evidence": {
                "path": None if q is None else str(q),
                "measurement_status": q_status,
                "backend": q_backend,
                "scope": "backbone_only",
                "provenance": "historical TRT context evidence",
                "coverage": {
                    "expected_cells": 10,
                    "measured_cells": 0,
                    "failed_cells": 0,
                    "proxy_cells": 10,
                },
            },
            "energy_lut": {
                "path": None,
                "measurement_status": energy_status,
                "backend": "not_available",
                "scope": "dense_core",
                "provenance": "not_collected_yet",
                "coverage": {
                    "expected_cells": 24,
                    "measured_cells": 0,
                    "failed_cells": 0,
                    "proxy_cells": 0,
                },
            },
            "downstream_objective": {
                "path": str(DS_MAP),
                "measurement_status": "measured",
                "backend": "closed_loop_sim",
                "scope": {
                    "model": downstream_model,
                    "town": "Town05",
                    "routes": "clean6",
                    "traffic": "full_traffic_1",
                },
                "provenance": "CoDriving AP x latency measured DS map v2",
                "mode_policy": "report_only_unless_measured_inputs",
                "coverage": {
                    "expected_cells": 65,
                    "measured_cells": 65,
                    "failed_cells": 0,
                    "proxy_cells": 0,
                },
                "cliff_band_ms": [600, 650],
            },
            "unsupported_conclusions": [],
        }
        path = tmp_path / f"{model}_evidence_registry.json"
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return path

    def test_pyramid_and_codriving_fixtures_load_cost_model_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pyramid = Stage2EvidenceRegistry.from_file(
                self._write_registry(tmp_path, model="pyramid_lidar")
            )
            codriving = Stage2EvidenceRegistry.from_file(
                self._write_registry(
                    tmp_path,
                    model="codriving",
                    manifest=CODRIVING_MANIFEST,
                    latency=CODRIVING_LATENCY,
                    ap=CODRIVING_AP,
                    q=None,
                    q_status="not_done",
                    q_backend="not_done",
                )
            )

            self.assertEqual(pyramid.schema, "stage2_evidence_registry_v1")
            self.assertEqual(codriving.model, "codriving")
            self.assertGreater(pyramid.load_latency_lut().latency((64, 128, 256), "tuned"), 0)
            self.assertGreater(pyramid.load_ap_model().ap70((64, 128, 256)), 0)
            self.assertEqual(
                pyramid.load_q_lookup().speedup((64, 128, 256)),
                pyramid.load_q_lookup().speedup((64, 128, 256)),
            )
            cost_inputs = pyramid.load_cost_inputs()
            self.assertGreater(
                cost_inputs.latency_lut.latency((64, 128, 256), "tuned"),
                0,
            )
            self.assertGreater(cost_inputs.ap_model.ap70((64, 128, 256)), 0)
            self.assertFalse(cost_inputs.energy_claim_allowed)
            self.assertEqual(
                cost_inputs.downstream_objective.measurement_status,
                "measured",
            )

    def test_measured_historical_trt_is_rejected_and_historical_trt_is_not_promoted(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            bad = self._write_registry(
                tmp_path,
                latency_backend="historical_trt",
                latency_status="measured",
            )
            with self.assertRaisesRegex(EvidenceRegistryError, "measured evidence"):
                Stage2EvidenceRegistry.from_file(bad)

            ok = Stage2EvidenceRegistry.from_file(
                self._write_registry(
                    tmp_path,
                    q_backend="historical_trt",
                    q_status="historical",
                )
            )
            self.assertFalse(ok.source("quant_evidence").promotable_to_measured)

    def test_missing_required_latency_or_ap_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            missing_latency = Stage2EvidenceRegistry.from_file(
                self._write_registry(tmp_path, latency=None, latency_status="not_available")
            )
            with self.assertRaisesRegex(EvidenceRegistryError, "latency_lut"):
                missing_latency.require_search_ready()

            missing_ap = Stage2EvidenceRegistry.from_file(
                self._write_registry(tmp_path, ap=None)
            )
            with self.assertRaisesRegex(EvidenceRegistryError, "ap_anchors"):
                missing_ap.require_search_ready()

    def test_energy_missing_forces_no_claim_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = Stage2EvidenceRegistry.from_file(self._write_registry(Path(tmp)))

            self.assertFalse(registry.energy_claim_allowed)
            self.assertIn(
                "energy_improvement_without_energy_lut",
                registry.unsupported_conclusions,
            )

    def test_downstream_ds_lookup_is_scope_limited_report_only_and_cliff_aware(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = Stage2EvidenceRegistry.from_file(self._write_registry(Path(tmp)))

            result = registry.query_downstream_ds(
                ap50=0.559,
                latency_ms=620.0,
                ap_input_status="predicted",
                latency_input_status="predicted",
                ap_interval=(0.54, 0.58),
                latency_interval_ms=(590.0, 660.0),
            )

            self.assertEqual(result.mode, "report_only")
            self.assertIn("predicted_ds_report_only", result.flags)
            self.assertIn("uncertain_due_to_cliff", result.flags)
            self.assertLessEqual(result.ds_low, result.ds_mid)
            self.assertLessEqual(result.ds_mid, result.ds_high)

            with self.assertRaisesRegex(EvidenceRegistryError, "outside measured"):
                registry.query_downstream_ds(
                    ap50=0.9,
                    latency_ms=620.0,
                    ap_input_status="measured",
                    latency_input_status="measured",
                )

            wrong_scope = Stage2EvidenceRegistry.from_file(
                self._write_registry(Path(tmp), downstream_model="pyramid_lidar")
            )
            with self.assertRaisesRegex(EvidenceRegistryError, "CoDriving/Town05/clean6"):
                wrong_scope.query_downstream_ds(
                    ap50=0.559,
                    latency_ms=620.0,
                    ap_input_status="measured",
                    latency_input_status="measured",
                )

    def test_coverage_summary_reports_expected_measured_failed_and_proxy_cells(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = Stage2EvidenceRegistry.from_file(self._write_registry(Path(tmp)))

            coverage = registry.coverage_summary()

            self.assertEqual(coverage["latency_lut"]["expected_cells"], 24)
            self.assertEqual(coverage["energy_lut"]["measured_cells"], 0)
            self.assertEqual(coverage["quant_evidence"]["proxy_cells"], 10)
            self.assertEqual(coverage["total_expected_cells"], 133)
            self.assertEqual(coverage["total_measured_cells"], 99)

    def test_prepare_evidence_cli_writes_pyramid_and_codriving_registry_fixtures(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pyramid_path = tmp_path / "pyramid_evidence_registry.json"
            codriving_path = tmp_path / "codriving_evidence_registry.json"

            for model, out_path in (
                ("pyramid_lidar", pyramid_path),
                ("codriving", codriving_path),
            ):
                result = subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "scripts/stage2_prepare_evidence.py"),
                        "--model",
                        model,
                        "--out-json",
                        str(out_path),
                    ],
                    cwd=ROOT,
                    env={"PYTHONPATH": str(ROOT)},
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

            pyramid = Stage2EvidenceRegistry.from_file(pyramid_path)
            codriving = Stage2EvidenceRegistry.from_file(codriving_path)

            self.assertEqual(pyramid.latency_lut.backend, "h800_tvm")
            self.assertEqual(pyramid.latency_lut.measurement_status, "measured")
            self.assertEqual(pyramid.quant_evidence.backend, "historical_trt")
            self.assertFalse(pyramid.quant_evidence.promotable_to_measured)
            self.assertFalse(pyramid.energy_claim_allowed)
            self.assertEqual(codriving.downstream_objective.measurement_status, "measured")
            self.assertEqual(
                codriving.downstream_objective.scope["model"],
                "codriving",
            )
            self.assertEqual(
                codriving.coverage_summary()["downstream_objective"]["measured_cells"],
                65,
            )
            pyramid.require_search_ready()
            codriving.require_search_ready()


if __name__ == "__main__":
    unittest.main()
