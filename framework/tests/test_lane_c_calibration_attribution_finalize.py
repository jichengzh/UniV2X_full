from __future__ import annotations

import json
import unittest

from tools.orin_deploy.lane_c_calibration_attribution_finalize import (
    attribute_ap_outcomes,
    build_attribution_summary,
    classify_device_ap,
)


class LaneCCalibrationAttributionFinalizeTest(unittest.TestCase):
    def test_device_ap_classification_is_preregistered(self) -> None:
        self.assertEqual(
            classify_device_ap(
                fp16={"ap50": 0.78, "ap70": 0.62},
                int8={"ap50": 0.0, "ap70": 0.0},
            ),
            "collapsed",
        )
        self.assertEqual(
            classify_device_ap(
                fp16={"ap50": 0.78, "ap70": 0.62},
                int8={"ap50": 0.779, "ap70": 0.615},
            ),
            "maintained",
        )
        self.assertEqual(
            classify_device_ap(
                fp16={"ap50": 0.78, "ap70": 0.62},
                int8={"ap50": 0.60, "ap70": 0.30},
            ),
            "mixed",
        )

    def test_attribution_matrix_distinguishes_calibration_from_runtime(self) -> None:
        self.assertEqual(
            attribute_ap_outcomes("collapsed", "collapsed"),
            "shared_rebuilt_calibration_supported_as_common_trigger",
        )
        self.assertEqual(
            attribute_ap_outcomes("maintained", "collapsed"),
            "tensorrt_runtime_build_difference_audit_required",
        )

    def test_summary_keeps_power_quantities_separate_and_has_no_cross_device_ratio(self) -> None:
        devices = {
            "h800": self.device(
                trt="10.13.0.35",
                fp16_ap=(0.78, 0.62),
                int8_ap=(0.0, 0.0),
                power_source="H800 NVML board power",
            ),
            "orin": self.device(
                trt="8.5.2.2",
                fp16_ap=(0.78, 0.62),
                int8_ap=(0.0, 0.0),
                power_source="Orin tegrastats rail",
            ),
        }
        summary = build_attribution_summary(
            receipts={
                name: self.receipt(name)
                for name in ("local", "h800", "orin")
            },
            devices=devices,
            execution_audits={
                device: self.execution_audit(device, devices[device])
                for device in ("h800", "orin")
            },
        )

        self.assertEqual(
            summary["attribution"]["decision"],
            "shared_rebuilt_calibration_supported_as_common_trigger",
        )
        self.assertEqual(
            summary["causal_evidence_grade"],
            "degraded_for_strict_binary_causality",
        )
        self.assertEqual(
            summary["devices"]["h800"]["power"]["physical_quantity"],
            "NVML board power",
        )
        self.assertEqual(
            summary["devices"]["orin"]["power"]["physical_quantity"],
            "Jetson tegrastats rails",
        )
        serialized = json.dumps(summary).lower()
        self.assertNotIn("cross_hardware_ratio", serialized)
        self.assertNotIn("speedup", serialized)

    def test_incomplete_receipt_cannot_be_promoted(self) -> None:
        devices = {
            "h800": self.device(
                trt="10.13.0.35",
                fp16_ap=(0.78, 0.62),
                int8_ap=(0.0, 0.0),
                power_source="H800 NVML board power",
            ),
            "orin": self.device(
                trt="8.5.2.2",
                fp16_ap=(0.78, 0.62),
                int8_ap=(0.0, 0.0),
                power_source="Orin tegrastats rail",
            ),
        }
        receipts = {name: self.receipt(name) for name in ("local", "h800", "orin")}
        receipts["orin"]["files"] = []
        with self.assertRaisesRegex(ValueError, "receipt files"):
            build_attribution_summary(
                receipts=receipts,
                devices=devices,
                execution_audits={
                    device: self.execution_audit(device, devices[device])
                    for device in ("h800", "orin")
                },
            )

    @staticmethod
    def receipt(name):
        host = {
            "local": ("bm-2s55he", "x86_64", "unavailable:ModuleNotFoundError"),
            "h800": ("zs-nj-tap-gpu18", "x86_64", "10.13.0.35"),
            "orin": ("ubuntu", "aarch64", "8.5.2.2"),
        }[name]
        return {
            "schema_version": "lane_c_calibration_receive_receipt_v1",
            "status": "byte_identical_15_of_15",
            "scope": "pyramid_get_multiscale_feature_16x32x64_only",
            "manifest_sha256": "m" * 64,
            "payload_id": "p" * 64,
            "expected_file_count": 15,
            "matched_file_count": 15,
            "missing_files": [],
            "extra_files": [],
            "host": {
                "hostname": host[0],
                "architecture": host[1],
                "tensorrt_version": host[2],
            },
            "files": [
                {
                    "filename": f"batch2_{index:03d}.npy",
                    "bytes": 16777344,
                    "sha256": f"{index:064x}",
                    "shape": [2, 64, 128, 256],
                    "dtype": "float32",
                    "c_contiguous": True,
                }
                for index in range(15)
            ],
        }

    @staticmethod
    def device(*, trt, fp16_ap, int8_ap, power_source):
        manifest_sha = "m" * 64
        def build(precision):
            return {
                "onnx_sha256": (
                    "8f09b5256f1856cc79ebbebf0d6c26e6dc63fba2ac3994552a690eaacd3be2a5"
                ),
                "precision": precision,
                "tensorrt_version": trt,
                "calibration_manifest_sha256": (
                    manifest_sha if precision == "int8" else None
                ),
                "calibration_verified_file_count": 15 if precision == "int8" else 0,
                "calibration_consumed_file_count": 15 if precision == "int8" else 0,
                "fresh_build": {
                    "preexisting_output_count": 0,
                    "calibration_cache_read": False,
                },
                "engine_sha256": precision[0] * 64,
                "layer_precision_counts": {
                    "fp16": 1,
                    "fp32": 1,
                    "int8": 1 if precision == "int8" else 0,
                    "other": 0,
                },
            }

        def latency(precision):
            return {
                "engine_sha256": precision[0] * 64,
                "scope": "pyramid_get_multiscale_feature_engine_compute_no_data_transfer",
                "protocol": {
                    "agent_batch": 2,
                    "warmup": 20,
                    "iters": 300,
                    "repeat": 5,
                    "timing": "CUDA_event",
                    "data_transfer_inside_timed_region": False,
                },
                "sample_count": 1500,
                "median_ms": 1.0,
                "p90_ms": 1.1,
                "p99_ms": 1.2,
                "mean_ms": 1.05,
                "power_measurement": {
                    "source": power_source,
                    "sample_count": 2,
                    "board_power_mean_w": 70.0,
                    "vin_sys_5v0_mean_w": 8.0,
                    "vdd_gpu_soc_mean_w": 5.0,
                },
            }

        def ap(precision, values):
            return {
                "schema": "stage3_trt_multiscale_ap_bridge_v3",
                "status": "success",
                "precision_tag": precision,
                "checkpoint_sha256": (
                    "d08fb16e778c6701aef7172e8d9609f1e4d73f20c419ee27bb8f158acc24a279"
                ),
                "engine_sha256": precision[0] * 64,
                "dataset_split_sha256": (
                    "f6805e26aec6af0f994395ad8c74105e65c88770bc7261bd7e995e30d47e2814"
                ),
                "processed_samples": 1789,
                "failed_samples": 0,
                "fallback_samples": 0,
                "full_ap_min_samples": 1789,
                "engine_ap_claim": True,
                "protocol": {
                    "bridge": "stage3_trt_multiscale_ap_bridge_v3",
                    "engine_batch": 2,
                    "eval_range": "102.4,51.2",
                    "full_ap_min_samples": 1789,
                    "fallback_policy": "forbidden_for_engine_ap_claim",
                    "input_route": (
                        "HEAL spatial_features -> TensorRT multiscale outputs -> "
                        "PyTorch fusion/head/postprocess"
                    ),
                },
                "ap50": values[0],
                "ap70": values[1],
                "_report_sha256": precision[0] * 64,
            }

        return {
            "build": {"fp16": build("fp16"), "int8": build("int8")},
            "latency": {"fp16": latency("fp16"), "int8": latency("int8")},
            "ap": {
                "fp16": ap("fp16", fp16_ap),
                "int8": ap("int8", int8_ap),
            },
            "numerical": {
                "fp16": {"all_required_metrics_present": True},
                "int8": {"all_required_metrics_present": True},
            },
        }

    @staticmethod
    def execution_audit(device, evidence):
        host = {
            "h800": ("zs-nj-tap-gpu18", "x86_64"),
            "orin": ("ubuntu", "aarch64"),
        }[device]
        return {
            "schema_version": "lane_c_direct_device_ap_execution_audit_v1",
            "device": device,
            "execution_mode": "direct_same_process",
            "replacement_scope": "get_multiscale_feature_only",
            "rpc_used": False,
            "host": {
                "hostname": host[0],
                "architecture": host[1],
            },
            "ap_reports": {
                precision: {
                    "report_sha256": evidence["ap"][precision]["_report_sha256"],
                    "engine_sha256": evidence["build"][precision]["engine_sha256"],
                }
                for precision in ("fp16", "int8")
            },
        }


if __name__ == "__main__":
    unittest.main()
