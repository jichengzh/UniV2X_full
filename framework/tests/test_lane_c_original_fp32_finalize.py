from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tools.orin_deploy.lane_c_original_fp32_finalize import (
    EvidenceValidationError,
    finalize_evidence,
    main,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


class OriginalFp32Fixture:
    def __init__(self, root: Path, baseline_root: Path) -> None:
        self.root = root
        self.baseline_root = baseline_root
        self.checkpoint = root / "source/net_epoch_bestval_at23.pth"
        self.config = root / "source/original_config.yaml"
        self.onnx = root / "source/pyramid_064x128x256_multiscale.onnx"
        self.checkpoint.parent.mkdir(parents=True)
        self.checkpoint.write_bytes(b"original-checkpoint")
        self.config.write_bytes(b"original-config")
        self.onnx.write_bytes(b"original-onnx")
        self.onnx_export_report = root / "source/onnx_export_report.json"
        _write_json(
            self.onnx_export_report,
            {
                "onnx_digest": _sha256(self.onnx),
                "input_shape": [2, 64, 128, 256],
                "output_names": [
                    "pyramid_level0",
                    "pyramid_level1",
                    "pyramid_level2",
                ],
                "output_shapes": {
                    "pyramid_level0": [2, 64, 128, 256],
                    "pyramid_level1": [2, 128, 64, 128],
                    "pyramid_level2": [2, 256, 32, 64],
                },
            },
        )
        self.expected_source = {
            "checkpoint_sha256": _sha256(self.checkpoint),
            "config_sha256": _sha256(self.config),
            "onnx_sha256": _sha256(self.onnx),
        }
        self._write_contract()
        self._write_heldout()
        self._write_build()
        self._write_numerical()
        self._write_latency()
        self._write_ap()
        self._write_baseline()

    def _write_contract(self) -> None:
        _write_json(
            self.root / "contracts/experiment_contract.json",
            {
                "source": {
                    **self.expected_source,
                    "onnx_export_report_sha256": _sha256(
                        self.onnx_export_report
                    ),
                    "input": {
                        "name": "spatial_features",
                        "dtype": "float32",
                        "shape": [2, 64, 128, 256],
                    },
                    "outputs": [
                        {
                            "name": "pyramid_level0",
                            "dtype": "float32",
                            "shape": [2, 64, 128, 256],
                        },
                        {
                            "name": "pyramid_level1",
                            "dtype": "float32",
                            "shape": [2, 128, 64, 128],
                        },
                        {
                            "name": "pyramid_level2",
                            "dtype": "float32",
                            "shape": [2, 256, 32, 64],
                        },
                    ],
                },
                "scope": {
                    "model": "Pyramid",
                    "structure": [64, 128, 256],
                    "subgraph": "get_multiscale_feature_multiscale_backbone_only",
                },
                "reporting": {
                    "comparison": (
                        "combined_structure_and_precision_deployment_comparison"
                    ),
                    "single_factor_causal_claim_allowed": False,
                    "unsupported_cross_hardware_speedup_allowed": False,
                },
            },
        )

    def _write_heldout(self) -> None:
        output_path = self.root / "heldout/original_heldout_batch2_fp32.npy"
        source_path = (
            self.root / "heldout/original_train32_spatial_features_fp32.npz"
        )
        summary_path = self.root / "heldout/original_train32_export_summary.json"
        output_path.parent.mkdir(parents=True)
        np.save(output_path, np.ones((1, 2, 64, 1, 1), dtype=np.float32))
        np.savez(
            source_path,
            spatial_features=np.ones((2, 64, 1, 1), dtype=np.float32),
        )
        _write_json(summary_path, {"source": "DAIR", "dtype": "float32"})
        _write_json(
            self.root / "heldout/original_heldout_batch2_audit.json",
            {
                "checkpoint_sha256": self.expected_source["checkpoint_sha256"],
                "config_sha256": self.expected_source["config_sha256"],
                "output_dtype": "float32",
                "source_dtype": "float32",
                "output_all_finite": True,
                "source_all_finite": True,
                "output_npy_sha256": _sha256(output_path),
                "source_export_npz_sha256": _sha256(source_path),
                "source_export_summary_sha256": _sha256(summary_path),
                "split_source_sha256": "a" * 64,
                "calibration_used_for_engine": False,
            },
        )

    def _write_build(self) -> None:
        engine_path = self.root / "orin/fp32/original_strict_fp32.engine"
        inspector_path = self.root / "orin/fp32/engine_inspector.json"
        engine_path.parent.mkdir(parents=True)
        engine_path.write_bytes(b"strict-fp32-engine")
        _write_json(
            inspector_path,
            {
                "Bindings": [
                    "spatial_features",
                    "pyramid_level0",
                    "pyramid_level1",
                    "pyramid_level2",
                ],
                "Layers": [
                    {
                        "Name": "layer0",
                        "Inputs": [{"Format/Datatype": "Row major linear FP32"}],
                        "Outputs": [{"Format/Datatype": "Row major linear FP32"}],
                    }
                ],
            },
        )
        self.engine_sha256 = _sha256(engine_path)
        _write_json(
            self.root / "orin/fp32/build_receipt.json",
            {
                "precision": "fp32",
                "strict_fp32": True,
                "tf32_allowed": False,
                "builder_flags": ["fp32", "tf32_disabled"],
                "layer_precision_counts": {
                    "fp32": 1,
                    "fp16": 0,
                    "int8": 0,
                    "other": 0,
                },
                "forbidden_inspector_matches": [],
                "external_fallback_count": 0,
                "onnx_sha256": self.expected_source["onnx_sha256"],
                "engine_sha256": self.engine_sha256,
                "inspector_json_sha256": _sha256(inspector_path),
                "output_channel_signature": [64, 128, 256],
                "output_shapes": [
                    [2, 64, 128, 256],
                    [2, 128, 64, 128],
                    [2, 256, 32, 64],
                ],
            },
        )

    def _write_numerical(self) -> None:
        heldout_path = self.root / "heldout/original_heldout_batch2_fp32.npy"
        candidate_path = self.root / "orin/fp32/heldout_outputs_fp32.npz"
        reference_path = self.root / "reference/server_ort_fp32_outputs.npz"
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        reference_path.parent.mkdir(parents=True, exist_ok=True)
        outputs = {
            "pyramid_level0": np.ones((1, 2, 64, 1, 1), dtype=np.float32),
            "pyramid_level1": np.ones((1, 2, 128, 1, 1), dtype=np.float32),
            "pyramid_level2": np.ones((1, 2, 256, 1, 1), dtype=np.float32),
        }
        np.savez(candidate_path, **outputs)
        np.savez(reference_path, **outputs)
        _write_json(
            self.root / "reference/server_ort_fp32_reference_report.json",
            {
                "all_finite": True,
                "inputs_sha256": _sha256(heldout_path),
                "onnx_sha256": self.expected_source["onnx_sha256"],
                "output_npz_sha256": _sha256(reference_path),
            },
        )
        _write_json(
            self.root / "orin/fp32/numerical_report.json",
            {
                "all_finite": True,
                "engine_sha256": self.engine_sha256,
                "inputs_sha256": _sha256(heldout_path),
                "output_sha256": _sha256(candidate_path),
                "external_fallback_count": 0,
                "expected_output_channels": [64, 128, 256],
                "output_channel_signature": [64, 128, 256],
                "output_shapes": {
                    name: list(value.shape) for name, value in outputs.items()
                },
            },
        )
        _write_json(
            self.root / "orin/fp32/numerical_comparison.json",
            {
                "precision": "fp32",
                "reference_npz_sha256": _sha256(reference_path),
                "candidate_npz_sha256": _sha256(candidate_path),
                "per_output": {
                    name: {
                        "cosine": 1.0 - index * 0.0001,
                        "nrmse": index * 0.001,
                        "mae": index * 0.0001,
                        "finite": True,
                    }
                    for index, name in enumerate(outputs)
                },
            },
        )

    def _write_latency(self) -> None:
        _write_json(
            self.root / "orin/fp32/primary_latency_power.json",
            {
                "engine_sha256": self.engine_sha256,
                "inputs_sha256": _sha256(
                    self.root / "heldout/original_heldout_batch2_fp32.npy"
                ),
                "median_ms": 12.5,
                "p90_ms": 12.9,
                "p99_ms": 13.4,
                "mean_ms": 12.6,
                "sample_count": 1500,
                "per_repeat": [
                    {"repeat_index": index, "sample_count": 300}
                    for index in range(5)
                ],
                "protocol": {
                    "agent_batch": 2,
                    "warmup": 20,
                    "iters": 300,
                    "repeat": 5,
                    "timing": "CUDA_event",
                    "data_transfer_inside_timed_region": False,
                    "boundary": "engine_compute_no_data_transfer",
                },
                "expected_output_channels": [64, 128, 256],
                "output_channel_signature": [64, 128, 256],
                "power_measurement": {
                    "measurement_status": "unavailable",
                    "blocker": "missing_tegrastats_power_rails",
                    "used_sudo": False,
                    "rail_semantics": (
                        "Orin tegrastats named rails; not H800 NVML board power"
                    ),
                },
            },
        )

    def _write_ap(self) -> None:
        derived_config = (
            self.root
            / "ap_source/Pyramid_DAIR_m1_base_2023_08_14_11_42_29/config.yaml"
        )
        derived_config.parent.mkdir(parents=True)
        derived_config.write_bytes(b"derived-original-config-with-full-data-paths")
        derived_sha256 = _sha256(derived_config)
        self.derived_config_sha256 = derived_sha256
        _write_json(
            self.root / "ap_source/config_derivation_audit.json",
            {
                "original_config_sha256": self.expected_source["config_sha256"],
                "derived_config_sha256": derived_sha256,
                "allowed_changes": [
                    "data_dir",
                    "root_dir",
                    "test_dir",
                    "validate_dir",
                ],
                "unexpected_changes": [],
            },
        )
        _write_json(
            self.root
            / "ap_full/orin/fp32/stage3_trt_multiscale_ap_bridge_report.json",
            {
                "status": "success",
                "precision_tag": "fp32",
                "checkpoint_sha256": self.expected_source["checkpoint_sha256"],
                "config_sha256": derived_sha256,
                "engine_sha256": self.engine_sha256,
                "processed_samples": 1789,
                "failed_samples": 0,
                "fallback_samples": 0,
                "ap50": 0.81,
                "ap70": 0.66,
                "engine_ap_claim": True,
                "engine_ap_claim_blockers": [],
                "protocol": {
                    "eval_range": "102.4,51.2",
                    "expected_output_channels": [64, 128, 256],
                    "voxelizer": "pure_torch",
                },
                "output_error_summary": {
                    "num_records": 5367,
                    "num_compared": 5367,
                    "shape_mismatch_count": 0,
                    "nonfinite_count": 0,
                    "all_finite": True,
                },
            },
        )

    def _write_baseline(self) -> None:
        engine_path = self.baseline_root / "orin/fp16/backbone_fp16.engine"
        onnx_path = (
            self.baseline_root / "source/pyramid_016x032x064_multiscale.onnx"
        )
        engine_path.parent.mkdir(parents=True)
        onnx_path.parent.mkdir(parents=True)
        engine_path.write_bytes(b"compact-fp16-engine")
        onnx_path.write_bytes(b"compact-fp16-onnx")
        engine_sha256 = _sha256(engine_path)
        self.baseline_onnx_sha256 = _sha256(onnx_path)
        _write_json(
            self.baseline_root / "orin/fp16/build_report.json",
            {
                "precision": "fp16",
                "scope": "pyramid_get_multiscale_feature_16x32x64_only",
                "engine_sha256": engine_sha256,
                "onnx_sha256": self.baseline_onnx_sha256,
            },
        )
        _write_json(
            self.baseline_root / "orin/fp16/numerical_report.json",
            {
                "engine_sha256": engine_sha256,
                "all_finite": True,
                "external_fallback_count": 0,
                "output_shapes": {
                    "pyramid_level0": [16, 2, 16, 128, 256],
                    "pyramid_level1": [16, 2, 32, 64, 128],
                    "pyramid_level2": [16, 2, 64, 32, 64],
                },
            },
        )
        _write_json(
            self.baseline_root / "orin/fp16/numerical_comparison.json",
            {
                "precision": "fp16",
                "per_output": {
                    f"pyramid_level{index}": {
                        "cosine": 0.999 - index * 0.001,
                        "nrmse": 0.001 + index * 0.001,
                        "mae": 0.0009 + index * 0.0001,
                        "finite": True,
                    }
                    for index in range(3)
                },
            },
        )
        _write_json(
            self.baseline_root / "orin/fp16/primary_latency_power.json",
            {
                "engine_sha256": engine_sha256,
                "median_ms": 10.53929615020752,
                "p90_ms": 10.55900764465332,
                "p99_ms": 10.728863716125488,
                "mean_ms": 10.541547538757325,
                "sample_count": 1500,
                "protocol": {
                    "agent_batch": 2,
                    "warmup": 20,
                    "iters": 300,
                    "repeat": 5,
                    "timing": "CUDA_event",
                    "data_transfer_inside_timed_region": False,
                },
            },
        )
        _write_json(
            self.baseline_root
            / "ap_full/orin/fp16/stage3_trt_multiscale_ap_bridge_report.json",
            {
                "engine_sha256": engine_sha256,
                "checkpoint_sha256": (
                    "d08fb16e778c6701aef7172e8d9609f1e4d73f20c419ee27bb8f158acc24a279"
                ),
                "processed_samples": 1789,
                "failed_samples": 0,
                "fallback_samples": 0,
                "engine_ap_claim": True,
                "ap50": 0.7854796006193402,
                "ap70": 0.6210292430415968,
                "output_error_summary": {
                    "num_records": 5367,
                    "num_compared": 5367,
                    "all_finite": True,
                },
            },
        )

    def mutate_json(self, relative_path: str, **updates: object) -> None:
        path = self.root / relative_path
        payload = _read_json(path)
        payload.update(updates)
        _write_json(path, payload)


class LaneCOriginalFp32FinalizeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        base = Path(self.temporary_directory.name)
        self.root = base / "original"
        self.baseline_root = base / "baseline"
        self.fixture = OriginalFp32Fixture(self.root, self.baseline_root)
        self.ap_config_patcher = patch(
            "tools.orin_deploy.lane_c_original_fp32_finalize."
            "LOCKED_AP_CONFIG_SHA256",
            self.fixture.derived_config_sha256,
        )
        self.ap_config_patcher.start()
        self.baseline_onnx_patcher = patch(
            "tools.orin_deploy.lane_c_original_fp32_finalize."
            "COMPACT_ONNX_SHA256",
            self.fixture.baseline_onnx_sha256,
            create=True,
        )
        self.baseline_onnx_patcher.start()

    def tearDown(self) -> None:
        self.baseline_onnx_patcher.stop()
        self.ap_config_patcher.stop()
        self.temporary_directory.cleanup()

    def test_finalizer_emits_raw_value_outputs_and_required_caveats(self) -> None:
        summary = finalize_evidence(
            self.root,
            self.baseline_root,
            expected_source=self.fixture.expected_source,
        )
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(
            summary["comparison_semantics"],
            "deployed_model_variant_checkpoint_and_structure_plus_precision",
        )
        with (self.root / "comparison.csv").open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        self.assertEqual([row["structure"] for row in rows], ["64,128,256", "16,32,64"])
        self.assertEqual(rows[0]["median_ms"], "12.5")
        self.assertEqual(rows[1]["median_ms"], "10.53929615020752")
        self.assertEqual(rows[1]["ap50"], "0.7854796006193402")
        self.assertEqual(rows[1]["ap70"], "0.6210292430415968")
        self.assertEqual(
            summary["rows"][0]["power"]["semantics"],
            "Orin tegrastats named rails; not H800 NVML board power",
        )
        rendered = (
            (self.root / "final_summary.json").read_text(encoding="utf-8")
            + (self.root / "comparison.csv").read_text(encoding="utf-8")
            + (self.root / "summary.md").read_text(encoding="utf-8")
        ).lower()
        normalized_rendered = " ".join(rendered.split())
        self.assertNotIn("speedup_ratio", rendered)
        self.assertIn("structure + precision", rendered)
        self.assertIn(
            "deployed model variants (checkpoint + structure)",
            normalized_rendered,
        )
        self.assertIn(
            "same-precision comparison would not establish pure channel-count causality",
            normalized_rendered,
        )
        self.assertIn("power is not comparable", rendered)
        self.assertIn("pure-torch voxelizer", rendered)

    def test_cli_requires_and_uses_explicit_baseline_root(self) -> None:
        with patch(
            "tools.orin_deploy.lane_c_original_fp32_finalize.LOCKED_SOURCE",
            dict(self.fixture.expected_source),
        ):
            result = main(
                [
                    "--artifact-root",
                    str(self.root),
                    "--baseline-root",
                    str(self.baseline_root),
                ]
            )
        self.assertEqual(result, 0)

    def test_rejects_wrong_original_output_channels(self) -> None:
        path = self.root / "orin/fp32/numerical_report.json"
        payload = _read_json(path)
        payload["output_channel_signature"] = [64, 128, 255]
        _write_json(path, payload)
        with self.assertRaisesRegex(EvidenceValidationError, "output channels"):
            finalize_evidence(
                self.root,
                self.baseline_root,
                expected_source=self.fixture.expected_source,
            )

    def test_rejects_wrong_source_onnx_output_contract(self) -> None:
        path = self.root / "contracts/experiment_contract.json"
        payload = _read_json(path)
        payload["source"]["outputs"][2]["shape"] = [2, 255, 32, 64]
        _write_json(path, payload)
        with self.assertRaisesRegex(EvidenceValidationError, "ONNX output contract"):
            finalize_evidence(
                self.root,
                self.baseline_root,
                expected_source=self.fixture.expected_source,
            )

    def test_rejects_incomplete_ap_output_accounting(self) -> None:
        path = (
            self.root
            / "ap_full/orin/fp32/stage3_trt_multiscale_ap_bridge_report.json"
        )
        payload = _read_json(path)
        payload["output_error_summary"]["num_compared"] = 5366
        _write_json(path, payload)
        with self.assertRaisesRegex(EvidenceValidationError, "AP num_compared"):
            finalize_evidence(
                self.root,
                self.baseline_root,
                expected_source=self.fixture.expected_source,
            )

    def test_rejects_injected_speedup_claim(self) -> None:
        self.fixture.mutate_json(
            "orin/fp32/primary_latency_power.json", speedup_ratio=2.5
        )
        with self.assertRaisesRegex(EvidenceValidationError, "forbidden claim"):
            finalize_evidence(
                self.root,
                self.baseline_root,
                expected_source=self.fixture.expected_source,
            )

    def test_rejects_non_float32_heldout(self) -> None:
        path = self.root / "heldout/original_heldout_batch2_fp32.npy"
        np.save(path, np.ones((1, 2, 64, 1, 1), dtype=np.float16))
        audit_path = self.root / "heldout/original_heldout_batch2_audit.json"
        audit = _read_json(audit_path)
        audit["output_npy_sha256"] = _sha256(path)
        _write_json(audit_path, audit)
        with self.assertRaisesRegex(EvidenceValidationError, "held-out.*float32"):
            finalize_evidence(
                self.root,
                self.baseline_root,
                expected_source=self.fixture.expected_source,
            )

    def test_rejects_non_strict_fp32_build_counts(self) -> None:
        path = self.root / "orin/fp32/build_receipt.json"
        payload = _read_json(path)
        payload["layer_precision_counts"]["fp16"] = 1
        _write_json(path, payload)
        with self.assertRaisesRegex(EvidenceValidationError, "FP16 count"):
            finalize_evidence(
                self.root,
                self.baseline_root,
                expected_source=self.fixture.expected_source,
            )

    def test_rejects_forged_empty_receipt_for_unsafe_inspector(self) -> None:
        inspector_path = self.root / "orin/fp32/engine_inspector.json"
        build_path = self.root / "orin/fp32/build_receipt.json"
        unsafe_layers = (
            {
                "Name": "conv",
                "LayerType": "Convolution",
                "Outputs": [{"Format/Datatype": "BFloat16"}],
            },
            {
                "Name": "conv",
                "LayerType": "Convolution",
            },
        )
        for layer in unsafe_layers:
            with self.subTest(layer=layer):
                _write_json(inspector_path, {"Layers": [layer]})
                build = _read_json(build_path)
                build["inspector_json_sha256"] = _sha256(inspector_path)
                build["forbidden_inspector_matches"] = []
                build["layer_precision_counts"] = {
                    "fp32": 1,
                    "fp16": 0,
                    "int8": 0,
                    "other": 0,
                }
                _write_json(build_path, build)
                with self.assertRaisesRegex(
                    EvidenceValidationError, "strict FP32 inspector"
                ):
                    finalize_evidence(
                        self.root,
                        self.baseline_root,
                        expected_source=self.fixture.expected_source,
                    )

    def test_rejects_receipt_other_precision_count(self) -> None:
        path = self.root / "orin/fp32/build_receipt.json"
        payload = _read_json(path)
        payload["layer_precision_counts"]["other"] = 1
        _write_json(path, payload)
        with self.assertRaisesRegex(EvidenceValidationError, "other count"):
            finalize_evidence(
                self.root,
                self.baseline_root,
                expected_source=self.fixture.expected_source,
            )

    def test_rejects_incomplete_latency_protocol(self) -> None:
        self.fixture.mutate_json(
            "orin/fp32/primary_latency_power.json", sample_count=1499
        )
        with self.assertRaisesRegex(EvidenceValidationError, "sample_count"):
            finalize_evidence(
                self.root,
                self.baseline_root,
                expected_source=self.fixture.expected_source,
            )

    def test_rejects_watts_when_power_is_unavailable(self) -> None:
        path = self.root / "orin/fp32/primary_latency_power.json"
        payload = _read_json(path)
        payload["power_measurement"]["vdd_gpu_soc_mean_w"] = 7.9
        _write_json(path, payload)
        with self.assertRaisesRegex(EvidenceValidationError, "must not contain watts"):
            finalize_evidence(
                self.root,
                self.baseline_root,
                expected_source=self.fixture.expected_source,
            )

    def test_rejects_nested_power_units_and_numeric_samples_when_unavailable(
        self,
    ) -> None:
        path = self.root / "orin/fp32/primary_latency_power.json"
        payload = _read_json(path)
        payload["power_measurement"]["details"] = {
            "rail": {"unit": "W", "samples": [7.8, 7.9]}
        }
        _write_json(path, payload)
        with self.assertRaisesRegex(
            EvidenceValidationError, "unavailable power.*payload"
        ):
            finalize_evidence(
                self.root,
                self.baseline_root,
                expected_source=self.fixture.expected_source,
            )

    def test_rejects_power_values_hidden_in_allowed_string_metadata(self) -> None:
        path = self.root / "orin/fp32/primary_latency_power.json"
        malicious_strings = (
            "forged mean 7.9 W on VDD_GPU_SOC",
            "forged 7 watts on VDD_GPU_SOC",
            "mean_w=7.9",
            "vdd_gpu_soc_w: 7.9",
        )
        for value in malicious_strings:
            with self.subTest(value=value):
                payload = _read_json(path)
                payload["power_measurement"]["rail_semantics"] = value
                _write_json(path, payload)
                with self.assertRaisesRegex(
                    EvidenceValidationError,
                    "unavailable power.*value or unit",
                ):
                    finalize_evidence(
                        self.root,
                        self.baseline_root,
                        expected_source=self.fixture.expected_source,
                    )

    def test_rejects_ap_engine_sha_mismatch(self) -> None:
        path = (
            self.root
            / "ap_full/orin/fp32/stage3_trt_multiscale_ap_bridge_report.json"
        )
        payload = _read_json(path)
        payload["engine_sha256"] = "f" * 64
        _write_json(path, payload)
        with self.assertRaisesRegex(EvidenceValidationError, "AP engine SHA"):
            finalize_evidence(
                self.root,
                self.baseline_root,
                expected_source=self.fixture.expected_source,
            )

    def test_rejects_unaudited_ap_config_derivation(self) -> None:
        path = self.root / "ap_source/config_derivation_audit.json"
        payload = _read_json(path)
        payload["allowed_changes"] = [
            "data_dir",
            "root_dir",
            "test_dir",
            "validate_dir",
            "model",
        ]
        _write_json(path, payload)
        with self.assertRaisesRegex(EvidenceValidationError, "allowed changes"):
            finalize_evidence(
                self.root,
                self.baseline_root,
                expected_source=self.fixture.expected_source,
            )

    def test_rejects_wrong_compact_baseline_checkpoint(self) -> None:
        path = (
            self.baseline_root
            / "ap_full/orin/fp16/stage3_trt_multiscale_ap_bridge_report.json"
        )
        payload = _read_json(path)
        payload["checkpoint_sha256"] = "f" * 64
        _write_json(path, payload)
        with self.assertRaisesRegex(
            EvidenceValidationError, "baseline AP checkpoint SHA"
        ):
            finalize_evidence(
                self.root,
                self.baseline_root,
                expected_source=self.fixture.expected_source,
            )


if __name__ == "__main__":
    unittest.main()
