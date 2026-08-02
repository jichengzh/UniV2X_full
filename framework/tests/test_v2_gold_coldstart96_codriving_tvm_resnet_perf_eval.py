from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import stage2_codriving_int8_provenance as provenance  # noqa: E402
import stage2_v2_gold_coldstart96_codriving_tvm_resnet_perf_eval as perf  # noqa: E402


def valid_manifest() -> dict:
    return {
        "schema": provenance.INT8_CALIBRATION_SCHEMA,
        "quantization_semantics": provenance.INT8_QUANTIZATION_SEMANTICS,
        "calibration_source": "/tmp/calib.npz",
        "calibration_summary": "/tmp/calib_summary.json",
        "calibration_split": "train",
        "calibration_split_source": "/data/train.json",
        "calibration_source_sha256": "a" * 64,
        "calibration_summary_sha256": "b" * 64,
        "calibration_samples": 16,
        "spatial_features_shape": [16, 2, 64, 256, 512],
        "qmin": -127,
        "qmax": 127,
        "scales_by_signature": {
            "sig": {"input_scale": 0.1, "weight_scale": 0.01},
        },
    }


class RouteBResnetPerfEvalTests(unittest.TestCase):
    def test_load_manifest_requires_matching_valid_ap_report(self) -> None:
        manifest = valid_manifest()
        payload = {
            "width": "32x64x128",
            "onnx": "/tmp/model.onnx",
            "onnx_sha256": "c" * 64,
            "mode": "mixed_top25_flops",
            "precision": "mixed",
            "mixed_policy": "top25_flops",
            "pipeline_scope": "tvm_routeb_resnet_in_full_pytorch_eval",
            "compile_summary": {
                "quantization_semantics": provenance.INT8_QUANTIZATION_SEMANTICS,
                "conv_precision_plan": {
                    "conv0": "int8",
                    "conv1": "fp16",
                    "conv2": "fp16",
                    "conv3": "fp16",
                },
                "int8_signature_plan": {"conv0": "sig"},
                "int8_calibration": manifest,
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ap.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(
                perf.load_calibration_manifest_from_ap_report(
                    path,
                    expected_mode="mixed_top25_flops",
                    expected_width="32x64x128",
                    expected_onnx=Path("/tmp/model.onnx"),
                    expected_onnx_sha256="c" * 64,
                ),
                manifest,
            )
            with self.assertRaisesRegex(ValueError, "mode"):
                perf.load_calibration_manifest_from_ap_report(
                    path,
                    expected_mode="int8_all",
                    expected_width="32x64x128",
                    expected_onnx=Path("/tmp/model.onnx"),
                    expected_onnx_sha256="c" * 64,
                )

    def test_load_manifest_rejects_dummy_scale_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ap.json"
            path.write_text(
                json.dumps(
                    {
                        "width": "32x64x128",
                        "onnx": "/tmp/model.onnx",
                        "onnx_sha256": "c" * 64,
                        "mode": "int8_all",
                        "precision": "int8",
                        "mixed_policy": "all",
                        "pipeline_scope": "tvm_routeb_resnet_in_full_pytorch_eval",
                        "compile_summary": {
                            "quantization_semantics": "dummy_scale_1_latency_tensorization_probe_only"
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "calibrated"):
                perf.load_calibration_manifest_from_ap_report(
                    path,
                    expected_mode="int8_all",
                    expected_width="32x64x128",
                    expected_onnx=Path("/tmp/model.onnx"),
                    expected_onnx_sha256="c" * 64,
                )

    def test_int8_all_rejects_partial_precision_plan(self) -> None:
        summary = {
            "quantization_semantics": provenance.INT8_QUANTIZATION_SEMANTICS,
            "conv_precision_plan": {"conv0": "int8", "conv1": "fp16"},
            "int8_signature_plan": {"conv0": "sig"},
            "int8_calibration": valid_manifest(),
        }
        with self.assertRaisesRegex(ValueError, "all Conv"):
            perf.validate_compile_summary_for_mode(summary, "int8_all")

    def test_gpu_mapping_requires_energy_to_follow_visible_device(self) -> None:
        self.assertEqual(
            perf.validate_gpu_mapping(tvm_gpu=0, physical_gpu=3, cuda_visible_devices="3"),
            {"tvm_logical_gpu": 0, "physical_gpu": 3, "cuda_visible_devices": "3"},
        )
        with self.assertRaisesRegex(ValueError, "maps"):
            perf.validate_gpu_mapping(tvm_gpu=0, physical_gpu=4, cuda_visible_devices="3")

    def test_energy_failure_is_not_accepted(self) -> None:
        with self.assertRaisesRegex(ValueError, "energy sampling failed"):
            perf.validate_energy_result({"status": "no_power_samples"})

    def test_energy_requires_idle_active_and_completed_samples(self) -> None:
        invalid = {
            "status": "success",
            "joule_per_inference": 0.1,
            "idle_sample_count": 0,
            "active_sample_count": 10,
            "completed_measure_iters": 300,
            "requested_measure_iters": 300,
            "elapsed_s": 5.0,
            "min_active_s": 5.0,
        }
        with self.assertRaisesRegex(ValueError, "sample counts"):
            perf.validate_energy_result(invalid)

    def test_runtime_plan_must_exactly_match_ap_report_plan(self) -> None:
        expected = {"conv0": "int8", "conv1": "fp16"}
        with self.assertRaisesRegex(ValueError, "precision plan"):
            perf.validate_runtime_plan(expected, {"conv0": "fp16", "conv1": "int8"})

    def test_summarize_latencies_uses_linear_percentiles(self) -> None:
        summary = perf.summarize_latencies([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(summary["samples"], 4)
        self.assertAlmostEqual(summary["p50_ms"], 2.5)
        self.assertAlmostEqual(summary["p90_ms"], 3.7)
        self.assertAlmostEqual(summary["mean_ms"], 2.5)

    def test_build_report_separates_backbone_scope_from_full_eval(self) -> None:
        summary = {
            "quantization_semantics": provenance.INT8_QUANTIZATION_SEMANTICS,
            "conv_precision_plan": {
                "conv0": "int8",
                "conv1": "fp16",
                "conv2": "fp16",
                "conv3": "fp16",
            },
            "int8_signature_plan": {"conv0": "sig"},
            "int8_calibration": valid_manifest(),
            "n_conv_replaced_tensorized": 4,
        }
        report = perf.build_report(
            width="32x64x128",
            mode="mixed_top25_flops",
            graph_io_dtype="fp32",
            onnx=Path("/tmp/model.onnx"),
            onnx_sha256="c" * 64,
            input_sha256="d" * 64,
            independent_run_id=2,
            latency={"p50_ms": 2.0, "p90_ms": 2.2, "samples": 100},
            energy={
                "status": "success",
                "joule_per_inference": 0.2,
                "idle_sample_count": 10,
                "active_sample_count": 10,
                "completed_measure_iters": 300,
                "requested_measure_iters": 300,
                "elapsed_s": 5.1,
                "min_active_s": 5.0,
            },
            compile_summary=summary,
            measurement_config={
                "tvm_logical_gpu": 0,
                "physical_gpu": 3,
                "cuda_visible_devices": "3",
                "warmup": 20,
                "reps": 100,
                "measure_energy": True,
                "energy_iters": 300,
                "energy_min_active_s": 5.0,
                "tvm_version": "test",
                "timing_method": "host_perf_counter_vm_call_plus_device_sync",
            },
        )
        self.assertEqual(report["pipeline_scope"], "tvm_routeb_resnet_backbone_only_vm")
        self.assertEqual(report["measurement_semantics"], "scale_aware_paired_backbone_v1")
        self.assertEqual(report["independent_run_id"], 2)
        self.assertEqual(report["latency"]["p50_ms"], 2.0)
        self.assertEqual(report["measurement_config"]["physical_gpu"], 3)


if __name__ == "__main__":
    unittest.main()
