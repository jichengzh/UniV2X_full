from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import fields
from pathlib import Path

from framework.stage2.contracts import (
    Stage2EvidenceDelta,
    Stage2EvidenceRecord,
    Stage2Input,
    apply_stage1_gate,
)
from framework.stage1.model_classifier import build_classification_report


ROOT = Path(__file__).resolve().parents[2]
CLASSIFICATION = (
    ROOT
    / "results/stage1_model_predict/model_classifier/stage1_model_classification_v1.json"
)
PYRAMID = ROOT / "framework/partitions/pyramid_lidar_partition.yaml"
CODRIVING = ROOT / "framework/partitions/codriving_partition.yaml"
WHERE2COMM = ROOT / "results/autoscan_where2comm_partition.yaml"


class Stage2IntegrationContractTest(unittest.TestCase):
    def _run_stage2_cli(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(ROOT / "scripts/stage2_optimize_model.py"), *args],
            cwd=ROOT,
            env={"PYTHONPATH": str(ROOT)},
            capture_output=True,
            text=True,
        )

    def test_stage2_input_is_thin_and_derives_hardware_context_from_manifest(self):
        self.assertEqual(
            [field.name for field in fields(Stage2Input)],
            ["manifest_path", "model_classification_path"],
        )

        stage2_input = Stage2Input(
            manifest_path=PYRAMID,
            model_classification_path=CLASSIFICATION,
        )
        context = stage2_input.to_runtime_context()

        self.assertEqual(
            set(context["public_input"]),
            {"manifest_path", "model_classification_path"},
        )
        self.assertNotIn("evidence_registry_path", context["public_input"])
        self.assertNotIn("hardware_target", context["public_input"])
        self.assertNotIn("search_policy", context["public_input"])
        self.assertEqual(context["hardware_context"]["source"], "manifest.hw_capability")
        self.assertIs(context["hardware_context"]["read_only"], True)
        self.assertEqual(
            context["hardware_context"]["name"], "NVIDIA GeForce RTX 4090"
        )

        with self.assertRaises(TypeError):
            Stage2Input(  # type: ignore[call-arg]
                manifest_path=PYRAMID,
                model_classification_path=CLASSIFICATION,
                hardware_target="h800",
            )

    def test_classifier_gate_maps_three_classes_to_runtime_modes(self):
        pyramid = apply_stage1_gate(PYRAMID, CLASSIFICATION)
        self.assertIs(pyramid.allowed, True)
        self.assertEqual(pyramid.runtime_mode, "joint")
        self.assertEqual(
            pyramid.source_classification["acceleration_class"],
            "CO_ACCELERATION_REQUIRED",
        )

        codriving = apply_stage1_gate(CODRIVING, CLASSIFICATION)
        self.assertIs(codriving.allowed, True)
        self.assertEqual(codriving.runtime_mode, "serial")
        self.assertEqual(
            codriving.source_classification["acceleration_class"],
            "SEPARABLE_ACCELERATION",
        )

        where2comm = apply_stage1_gate(WHERE2COMM, CLASSIFICATION)
        self.assertIs(where2comm.allowed, False)
        self.assertEqual(where2comm.runtime_mode, "fail_closed")
        self.assertEqual(where2comm.reason, "scan_failed")
        self.assertEqual(
            where2comm.source_classification["acceleration_class"],
            "SCAN_FAILED",
        )

    def test_evidence_delta_validation_blocks_overpromotion(self):
        measured = Stage2EvidenceRecord(
            backend="h800_tvm",
            hardware="H800 Hopper",
            scope="rsu_dense_core",
            evidence_kind="measured",
            provenance="tvm_metaschedule_smoke",
            candidate_config={"width": [64, 128, 128], "quant": "fp16"},
            metric={"latency_us": 123.4},
        )
        self.assertIs(measured.promotable_to_classifier, True)

        historical_trt = Stage2EvidenceRecord(
            backend="trt",
            hardware="RTX 4090",
            scope="rsu_dense_core",
            evidence_kind="historical",
            provenance="legacy_context",
            candidate_config={"width": [64, 128, 128], "quant": "int8"},
            metric={"latency_us": 99.0},
        )
        self.assertIs(historical_trt.promotable_to_classifier, False)

        demo = Stage2EvidenceRecord(
            backend="h800_tvm",
            hardware="H800 Hopper",
            scope="rsu_dense_core",
            evidence_kind="demo",
            provenance="fixture",
            candidate_config={"width": [64, 128, 128], "quant": "fp16"},
            metric={"latency_us": 1.0},
        )
        self.assertIs(demo.promotable_to_classifier, False)

        with self.assertRaisesRegex(ValueError, "Only h800_tvm measured evidence"):
            Stage2EvidenceRecord(
                backend="trt",
                hardware="RTX 4090",
                scope="rsu_dense_core",
                evidence_kind="measured",
                provenance="bad_new_measurement",
                candidate_config={"width": [64, 128, 128], "quant": "int8"},
                metric={"latency_us": 99.0},
            )

        delta = Stage2EvidenceDelta(
            model="pyramid_lidar",
            records=[measured, historical_trt, demo],
        )
        data = delta.to_dict()
        self.assertEqual(data["schema"], "stage2_evidence_delta_v1")
        self.assertIs(data["no_overpromotion"], True)
        self.assertEqual(data["promotable_record_count"], 1)

    def test_stage2_cli_writes_scoped_output_without_user_level_policy_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            out_json = tmp_path / "stage2_optimization_v1.json"
            evidence_delta = tmp_path / "stage2_evidence_delta_v1.json"

            result = self._run_stage2_cli(
                [
                    "--manifest",
                    str(PYRAMID),
                    "--classification",
                    str(CLASSIFICATION),
                    "--out-json",
                    str(out_json),
                    "--evidence-delta-out",
                    str(evidence_delta),
                ]
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            data = json.loads(out_json.read_text())
            self.assertEqual(data["schema"], "stage2_output_v1")
            self.assertEqual(data["model"], "pyramid_lidar")
            self.assertIs(data["optimization_status"]["allowed"], True)
            self.assertEqual(data["optimization_status"]["mode"], "joint")
            self.assertEqual(
                data["search_space_summary"]["model_search_policy"]["selected"],
                "joint",
            )
            self.assertEqual(data["optimized_scope"], "rsu_dense_core")
            self.assertIs(data["claim_boundaries"]["full_model_claim_allowed"], False)
            self.assertTrue(data["recommended_configs"])
            self.assertNotIn("dispatch_plan", data)
            self.assertEqual(
                set(data["public_input"]),
                {"manifest_path", "model_classification_path"},
            )
            self.assertNotIn("evidence_registry_path", repr(data))
            self.assertNotIn("hardware_target", data["public_input"])
            self.assertNotIn("search_policy", data["public_input"])
            self.assertEqual(data["hardware_context"]["source"], "manifest.hw_capability")
            self.assertIs(data["hardware_context"]["read_only"], True)
            self.assertIn("cost_evidence", data)
            self.assertEqual(
                data["cost_evidence"]["latency_lut"]["measurement_status"],
                "measured",
            )
            self.assertEqual(data["cost_evidence"]["latency_lut"]["backend"], "h800_tvm")
            self.assertEqual(
                data["cost_evidence"]["quant_evidence"]["backend"],
                "historical_trt",
            )
            self.assertFalse(data["cost_evidence"]["energy_lut"]["claim_allowed"])
            self.assertIn(
                "energy_improvement_without_energy_lut",
                data["cost_evidence"]["unsupported_conclusions"],
            )
            self.assertGreater(
                data["cost_evidence"]["coverage_summary"]["total_expected_cells"],
                0,
            )

            delta = json.loads(evidence_delta.read_text())
            self.assertEqual(delta["schema"], "stage2_evidence_delta_v1")
            self.assertEqual(delta["records"][0]["evidence_kind"], "demo")
            self.assertIs(delta["records"][0]["promotable_to_classifier"], False)

    def test_stage2_cli_maps_codriving_to_serial_and_scan_failed_to_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            codriving_json = tmp_path / "codriving.json"
            codriving = self._run_stage2_cli(
                [
                    "--manifest",
                    str(CODRIVING),
                    "--classification",
                    str(CLASSIFICATION),
                    "--out-json",
                    str(codriving_json),
                ]
            )
            self.assertEqual(codriving.returncode, 0, codriving.stderr)
            codriving_data = json.loads(codriving_json.read_text())
            self.assertEqual(codriving_data["optimization_status"]["mode"], "serial")
            self.assertEqual(
                codriving_data["search_space_summary"]["model_search_policy"]["selected"],
                "serial",
            )
            self.assertIn(
                "groups1_not_model_separable_proof",
                codriving_data["claim_boundaries"]["unsupported_conclusions"],
            )

            where2comm_json = tmp_path / "where2comm.json"
            where2comm = self._run_stage2_cli(
                [
                    "--manifest",
                    str(WHERE2COMM),
                    "--classification",
                    str(CLASSIFICATION),
                    "--out-json",
                    str(where2comm_json),
                ]
            )
            self.assertEqual(where2comm.returncode, 0, where2comm.stderr)
            where2comm_data = json.loads(where2comm_json.read_text())
            self.assertIs(where2comm_data["optimization_status"]["allowed"], False)
            self.assertEqual(
                where2comm_data["optimization_status"]["mode"],
                "fail_closed",
            )
            self.assertEqual(where2comm_data["recommended_configs"], [])

    def test_stage2_cli_missing_classification_fails_closed_without_demo(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_json = Path(tmp) / "missing_classification.json"
            result = self._run_stage2_cli(
                [
                    "--manifest",
                    str(PYRAMID),
                    "--out-json",
                    str(out_json),
                ]
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            data = json.loads(out_json.read_text())
            self.assertIs(data["optimization_status"]["allowed"], False)
            self.assertEqual(data["optimization_status"]["mode"], "fail_closed")
            self.assertEqual(data["optimization_status"]["reason"], "missing_classification")
            self.assertEqual(data["recommended_configs"], [])

    def test_stage2_cli_rejects_user_level_hardware_evidence_and_policy_flags(self):
        for forbidden in (
            ["--hardware-target", "h800"],
            ["--evidence-registry", "results/evidence_registry.json"],
            ["--search-policy", "joint"],
        ):
            result = self._run_stage2_cli(
                [
                    "--manifest",
                    str(PYRAMID),
                    "--classification",
                    str(CLASSIFICATION),
                    "--out-json",
                    "unused.json",
                    *forbidden,
                ]
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("unrecognized arguments", result.stderr)

    def test_stage2_update_evidence_validates_and_archives_delta(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            out_json = tmp_path / "stage2_optimization_v1.json"
            evidence_delta = tmp_path / "stage2_evidence_delta_v1.json"
            archive = tmp_path / "stage2_evidence_delta"

            result = self._run_stage2_cli(
                [
                    "--manifest",
                    str(PYRAMID),
                    "--classification",
                    str(CLASSIFICATION),
                    "--out-json",
                    str(out_json),
                    "--evidence-delta-out",
                    str(evidence_delta),
                ]
            )
            self.assertEqual(result.returncode, 0, result.stderr)

            update = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/stage2_update_evidence.py"),
                    "--delta",
                    str(evidence_delta),
                    "--out-dir",
                    str(archive),
                ],
                cwd=ROOT,
                env={"PYTHONPATH": str(ROOT)},
                capture_output=True,
                text=True,
            )

            self.assertEqual(update.returncode, 0, update.stderr)
            archived = archive / "pyramid_lidar_stage2_evidence_delta_v1.json"
            self.assertTrue(archived.exists())
            archived_data = json.loads(archived.read_text())
            self.assertEqual(archived_data["schema"], "stage2_evidence_delta_v1")
            self.assertTrue(archived_data["no_overpromotion"])

            report = build_classification_report(evidence_dir=archive.parent)
            self.assertIn("stage2_evidence_delta", report["evidence_inputs"])
            self.assertTrue(
                report["evidence_inputs"]["stage2_evidence_delta"].endswith(
                    "stage2_evidence_delta"
                )
            )


if __name__ == "__main__":
    unittest.main()
