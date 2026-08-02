from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "fcooper_finalize_tvm_capability_v1.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "fcooper_finalize_tvm_capability_v1", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class FCooperFinalizeTvmCapabilityV1Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_module()
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.output = self.root / "out"
        self.scanner = self.root / "fcooper_partition.yaml"
        self.scanner.write_text("width_schema: [s0, s1, s2, neck, output]\n")
        self.recovery_contract = self.root / "recovery_contract.json"
        write_json(self.recovery_contract, {"schema_version": "recovery_v2"})
        self.formal = self.root / "formal_contract.json"
        write_json(
            self.formal,
            {
                "schema_version": "fcooper_formal_v2_contract",
                "model": "fcooper",
                "partition_path": str(self.scanner),
                "partition_sha256": sha256(self.scanner),
                "recovery_contract_path": str(self.recovery_contract),
                "recovery_contract_sha256": sha256(self.recovery_contract),
                "probe_labels_allowed_in_training": False,
                "probe_rows_allowed_as_winner": False,
                "search_budget": {"batch_size": 4, "rounds": 4, "total": 16},
            },
        )
        self.recovery_reports = [
            self._recovery_report("base"),
            self._recovery_report("boundary"),
        ]
        self.base_fp16 = self._fp_result("base_fp16", "fp16")
        self.boundary_fp16 = self._fp_result("boundary_fp16", "fp16")
        self.fp32 = self._fp_result("base_fp32", "fp32")
        self.base_quant = self._quant_contract("base")
        self.boundary_quant = self._quant_contract("boundary")
        self.base_int8 = self._int8_result("base", self.base_quant, tensorized=23)
        self.boundary_int8 = self._int8_result(
            "boundary", self.boundary_quant, tensorized=20
        )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _artifact(self, name: str) -> Path:
        path = self.root / f"{name}.so"
        path.write_bytes(f"artifact:{name}".encode())
        return path

    def _onnx(self, name: str) -> Path:
        path = self.root / f"{name}.onnx"
        path.write_bytes(f"onnx:{name}".encode())
        return path

    def _recovery_report(self, name: str) -> Path:
        checkpoint = self.root / f"{name}.pth"
        checkpoint.write_bytes(name.encode())
        config = self.root / f"{name}.yaml"
        config.write_text(f"name: {name}\n")
        report = self.root / f"{name}_recovery.json"
        write_json(
            report,
            {
                "schema_version": "fcooper_recovery_training_report_v2",
                "status": "success",
                "recovery_contract_path": str(self.recovery_contract),
                "recovery_contract_sha256": sha256(self.recovery_contract),
                "recovered_checkpoint_path": str(checkpoint),
                "recovered_checkpoint_sha256": sha256(checkpoint),
                "config_path": str(config),
                "config_sha256": sha256(config),
                "best_validation_loss": 0.5,
                "elapsed_seconds": 123.0,
            },
        )
        return report

    def _fp_result(self, name: str, precision: str) -> Path:
        artifact = self._artifact(name)
        onnx = self._onnx(name)
        result = self.root / f"{name}_result.json"
        write_json(
            result,
            {
                "schema": "route_b_fp16_auto_result_v1",
                "status": "success",
                "build_success": True,
                "precision": precision,
                "host": "h800",
                "method": f"Route B {precision} automatic measurement",
                "artifact_path": str(artifact),
                "artifact_digest": sha256(artifact),
                "onnx_path": str(onnx),
                f"correctness_vs_default_{precision}": [
                    {
                        "shape_match": True,
                        "max_abs_err": 0.001,
                        "p99_rel_err": 0.002,
                    }
                ],
                "latency": {"latency_ms_p50": 1.0},
                "energy": {"joules_per_inference": 1.0},
            },
        )
        return result

    def _quant_contract(self, name: str) -> Path:
        onnx = self.root / f"{name}_int8.onnx"
        onnx.write_bytes(f"onnx:{name}:int8".encode())
        calibration = self.root / f"{name}_calibration.json"
        write_json(calibration, {"schema_version": "fcooper_calibration_manifest_v1"})
        contract = self.root / f"{name}_quant.json"
        params = {key: {"scale": 0.1, "zero_point": 128} for key in "abcdef"}
        write_json(
            contract,
            {
                "schema_version": "fcooper_tvm_int8_quant_contract_v1",
                "status": "success",
                "quantization": {
                    "dtype": "uint8",
                    "semantics": "static_symmetric_uint8_centered_128",
                },
                "onnx": {"path": str(onnx), "sha256": sha256(onnx)},
                "calibration": {
                    "summary_path": str(calibration),
                    "summary_sha256": sha256(calibration),
                    "sample_count": 16,
                    "method": "absmax",
                },
                "sample_count": 16,
                "coverage": {
                    "required": list(params),
                    "observed": list(params),
                    "missing": [],
                    "required_count": len(params),
                    "observed_count": len(params),
                },
                "concat_scale_normalization": [
                    {"members": ["d", "e", "f"], "common_absmax": 12.0}
                ],
                "params": params,
            },
        )
        return contract

    def _int8_result(
        self, name: str, quant_contract: Path, *, tensorized: int
    ) -> Path:
        artifact = self._artifact(f"{name}_int8")
        contract = json.loads(quant_contract.read_text())
        result = self.root / f"{name}_int8_result.json"
        write_json(
            result,
            {
                "schema": "route_b_int8_auto_decomp_result_v1",
                "status": "success",
                "build_success": True,
                "host": "h800",
                "method": "per-Conv Relax call_tir automatic tensorization",
                "route_spec": "route_b_int8_auto_decomp_per_block_matmul_tensorization_v1",
                "not_used": [
                    "hand-written full-engine im2col+MMA as final route",
                    "silent fallback",
                ],
                "artifact_path": str(artifact),
                "artifact_digest": sha256(artifact),
                "onnx_path": contract["onnx"]["path"],
                "tensor_quant_params_path": str(quant_contract),
                "tensor_quant_params_sha256": sha256(quant_contract),
                "tensor_quant_params_count": 6,
                "correctness_all_exact": True,
                "correctness_vs_native_direct": [
                    {"shape": [5, 256, 256, 256], "exact_equal": True}
                ],
                "n_conv_blocks": 24,
                "n_tensorized_conv_blocks": tensorized,
                "n_schedule_failures": 0,
                "op_counts": {
                    "Conv": 24,
                    "Relu": 24,
                    "DepthToSpace": 3,
                    "Concat": 1,
                },
                "block_reports": [
                    *[
                        {"op_type": "Conv", "tensorization_status": "tensorized"}
                        for _ in range(tensorized)
                    ],
                    *[
                        {
                            "op_type": "Conv",
                            "tensorization_status": "native_fallback_uint8_input_not_s8_tensorcore_eligible",
                        }
                        for _ in range(24 - tensorized)
                    ],
                ],
                "latency": {"latency_ms_p50": 1.0},
            },
        )
        return result

    def _kwargs(self) -> dict:
        return {
            "base_fp16_result": self.base_fp16,
            "boundary_fp16_result": self.boundary_fp16,
            "base_int8_result": self.base_int8,
            "boundary_int8_result": self.boundary_int8,
            "fp32_schedule_result": self.fp32,
            "base_int8_quant_contract": self.base_quant,
            "boundary_int8_quant_contract": self.boundary_quant,
            "recovery_reports": self.recovery_reports,
            "scanner_manifest": self.scanner,
            "formal_contract": self.formal,
        }

    def test_builds_label_free_profile_and_all_audits(self) -> None:
        outputs = self.module.finalize_capability(**self._kwargs())
        profile = outputs["capability_profiles"]["capability_profiles"][0]

        self.assertEqual(
            profile["capability_profile_id"],
            "h800-tvm-fcooper-probe-conditioned-v1",
        )
        self.assertEqual(profile["dispatch_key"], "tvm_auto")
        self.assertAlmostEqual(
            profile["features"]["int8_tensorized_conv_coverage"],
            43 / 48,
        )
        self.assertEqual(profile["features"]["depth_to_space_count"], 6)
        self.assertEqual(profile["features"]["concat_count"], 2)
        self.assertEqual(profile["features"]["int8_fallback_conv_count"], 5)
        serialized = json.dumps(profile).lower()
        for forbidden in ("latency", "energy", "ap70", "accuracy"):
            self.assertNotIn(forbidden, serialized)

        isolation = outputs["probe_isolation_audit"]
        self.assertTrue(isolation["passed"])
        self.assertEqual(isolation["status"], "passed")
        self.assertTrue(isolation["probe_row_ids"])
        self.assertFalse(isolation["probe_metrics_allowed_as_cost_model_labels"])
        self.assertFalse(isolation["probe_rows_allowed_in_t16_budget"])
        self.assertFalse(isolation["probe_rows_allowed_as_winner"])
        self.assertFalse(isolation["policy"]["probe_labels_allowed_in_training"])
        self.assertFalse(isolation["policy"]["probe_rows_allowed_in_t16"])
        self.assertFalse(isolation["policy"]["probe_rows_allowed_as_winner"])

        recovery = outputs["recovery_numeric_gate_summary"]
        self.assertEqual(recovery["status"], "passed")
        self.assertTrue(recovery["t16_search_allowed"])
        self.assertTrue(recovery["passed"])
        self.assertEqual(recovery["report_count"], 2)
        serialized_recovery = json.dumps(recovery).lower()
        self.assertNotIn("elapsed_seconds", serialized_recovery)
        self.assertNotIn("best_validation_loss", serialized_recovery)
        self.assertNotIn("trt", json.dumps(recovery["reports"]).lower())

    def test_inherits_common_h800_tvm_scan_features(self) -> None:
        from framework.stage2.canonical_search_v3 import build_capability_profile

        base = self.root / "base_profiles.json"
        base.write_text(
            json.dumps(
                [
                    build_capability_profile(
                        capability_profile_id="h800-tvm-generic",
                        hardware_target="h800",
                        compiler_fingerprint="a" * 64,
                        dispatch_key="tvm_auto",
                        features={"s1q_qdq_fold_ratio_mean": 0.25},
                    )
                ]
            )
        )

        outputs = self.module.finalize_capability(
            **self._kwargs(),
            base_capability_profile=base,
        )

        profile = outputs["capability_profiles"]["capability_profiles"][0]
        self.assertEqual(profile["features"]["s1q_qdq_fold_ratio_mean"], 0.25)

    def test_write_outputs_uses_fixed_atomic_filenames(self) -> None:
        outputs = self.module.finalize_capability(**self._kwargs())
        self.module.write_outputs(self.output, outputs)

        expected = {
            "capability_profiles.json",
            "probe_audit.json",
            "probe_isolation_audit.json",
            "recovery_numeric_gate_summary.json",
        }
        self.assertEqual({path.name for path in self.output.iterdir()}, expected)

    def test_rejects_artifact_sha_drift(self) -> None:
        payload = json.loads(self.base_fp16.read_text())
        payload["artifact_digest"] = "0" * 64
        write_json(self.base_fp16, payload)

        with self.assertRaisesRegex(ValueError, "artifact.*SHA256"):
            self.module.finalize_capability(**self._kwargs())

    def test_rejects_quant_coverage_or_route_binding_drift(self) -> None:
        payload = json.loads(self.base_quant.read_text())
        payload["coverage"]["missing"] = ["relu"]
        payload["coverage"]["observed_count"] = 5
        write_json(self.base_quant, payload)

        with self.assertRaisesRegex(ValueError, "quant coverage"):
            self.module.finalize_capability(**self._kwargs())

    def test_rejects_nonautomatic_or_inexact_int8_route(self) -> None:
        payload = json.loads(self.base_int8.read_text())
        payload["route_spec"] = "hand_written_im2col_mma"
        payload["correctness_all_exact"] = False
        write_json(self.base_int8, payload)

        with self.assertRaisesRegex(ValueError, "automatic Route B"):
            self.module.finalize_capability(**self._kwargs())

    def test_rejects_formal_probe_leakage_and_scanner_sha_drift(self) -> None:
        payload = json.loads(self.formal.read_text())
        payload["probe_labels_allowed_in_training"] = True
        write_json(self.formal, payload)
        with self.assertRaisesRegex(ValueError, "probe labels"):
            self.module.finalize_capability(**self._kwargs())

        payload["probe_labels_allowed_in_training"] = False
        payload["partition_sha256"] = "0" * 64
        write_json(self.formal, payload)
        with self.assertRaisesRegex(ValueError, "scanner.*SHA256"):
            self.module.finalize_capability(**self._kwargs())


if __name__ == "__main__":
    unittest.main()
