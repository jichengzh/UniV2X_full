from __future__ import annotations

import math
import unittest

import numpy as np

from tools.orin_deploy.lane_c_contract import (
    assess_deployment_numerical_gate,
    assess_no_execution_fallback,
    assess_server_reference_sanity,
    compare_outputs,
    parse_tegrastats,
    parse_trtexec,
    summarize_repeat_medians,
)


class LaneCOrinContractTest(unittest.TestCase):
    def test_parse_trtexec_requires_all_latency_statistics(self) -> None:
        output = """
        [I] Throughput: 47.25 qps
        [I] GPU Compute Time: min = 20.1 ms, max = 21.7 ms,
            mean = 20.8 ms, median = 20.7 ms,
            percentile(90%) = 21.1 ms, percentile(95%) = 21.3 ms,
            percentile(99%) = 21.6 ms
        [I] Total Host Walltime: 10.52 s
        [I] Total GPU Compute Time: 10.41 s
        """
        parsed = parse_trtexec(output)
        self.assertEqual(
            parsed,
            {
                "median_ms": 20.7,
                "p90_ms": 21.1,
                "p99_ms": 21.6,
                "mean_ms": 20.8,
                "throughput_qps": 47.25,
                "host_walltime_s": 10.52,
                "gpu_compute_total_s": 10.41,
            },
        )

    def test_parse_trtexec_rejects_incomplete_output(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing trtexec metrics"):
            parse_trtexec("Throughput: 12.0 qps")

    def test_parse_tegrastats_uses_instantaneous_vin_sys_power(self) -> None:
        raw = "\n".join(
            [
                "RAM 100/1000MB GR3D_FREQ 50%@[611,0] GPU@51.0C "
                "VIN_SYS_5V0 7000mW/6500mW VDD_GPU_SOC 4000mW/3900mW",
                "RAM 100/1000MB GR3D_FREQ 75%@[611,0] GPU@52.0C "
                "VIN_SYS_5V0 9000mW/7500mW VDD_GPU_SOC 5000mW/4400mW",
            ]
        )
        parsed = parse_tegrastats(raw)
        self.assertEqual(parsed["sample_count"], 2)
        self.assertEqual(parsed["vin_sys_5v0_w"], [7.0, 9.0])
        self.assertEqual(parsed["vdd_gpu_soc_w"], [4.0, 5.0])
        self.assertEqual(parsed["gr3d_percent"], [50.0, 75.0])
        self.assertEqual(parsed["gpu_temperature_c"], [51.0, 52.0])
        self.assertEqual(parsed["vin_sys_5v0_mean_w"], 8.0)

    def test_compare_outputs_accepts_fp32_sanity_and_rejects_nonfinite(self) -> None:
        reference = {"cls": np.array([1.0, 2.0, 3.0], dtype=np.float32)}
        actual = {"cls": np.array([1.001, 1.999, 3.002], dtype=np.float32)}
        passed = compare_outputs(reference, actual, precision="fp32")
        self.assertTrue(passed["passed"])
        self.assertGreaterEqual(passed["per_output"]["cls"]["cosine_similarity"], 0.999)

        bad = {"cls": np.array([1.0, math.nan, 3.0], dtype=np.float32)}
        failed = compare_outputs(reference, bad, precision="fp32")
        self.assertFalse(failed["passed"])
        self.assertFalse(failed["per_output"]["cls"]["finite"])

    def test_compare_outputs_int8_uses_explicit_relaxed_thresholds(self) -> None:
        reference = {"reg": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)}
        actual = {"reg": np.array([1.05, 1.9, 3.1, 3.95], dtype=np.float32)}
        report = compare_outputs(reference, actual, precision="int8")
        self.assertTrue(report["passed"])
        self.assertEqual(report["thresholds"]["cosine_similarity_min"], 0.99)
        self.assertEqual(report["thresholds"]["normalized_rmse_max"], 0.1)

    def test_server_reference_sanity_is_relative_and_keeps_strict_diagnostic(self) -> None:
        reference = {"cls": np.array([3.0, 4.0, 5.0], dtype=np.float32)}
        actual = {"cls": np.array([3.02, 3.98, 5.02], dtype=np.float32)}
        strict = compare_outputs(reference, actual, precision="fp32")
        sanity = assess_server_reference_sanity(
            reference, actual, precision="fp32"
        )
        self.assertFalse(strict["passed"])
        self.assertTrue(sanity["passed"])
        self.assertEqual(sanity["thresholds"]["normalized_rmse_max"], 0.02)

    def test_deployment_gate_cannot_be_rescued_by_same_device_reference(self) -> None:
        server_sanity = {"passed": False}
        same_device = {
            "passed": True,
            "per_output": {
                "cls": {
                    "finite": True,
                    "cosine_similarity": 1.0,
                    "normalized_rmse": 0.0,
                }
            },
        }
        report = assess_deployment_numerical_gate(
            cross_backend_report=server_sanity,
            same_device_report=same_device,
            current_onnx_sha256="abc123",
            reference_onnx_sha256="abc123",
        )
        self.assertFalse(report["passed"])
        self.assertEqual(report["basis"], "server_reference_sanity_failed")

        passed = assess_deployment_numerical_gate(
            cross_backend_report={"passed": True},
            same_device_report=same_device,
            current_onnx_sha256="abc123",
            reference_onnx_sha256="abc123",
        )
        self.assertTrue(passed["passed"])
        self.assertEqual(passed["basis"], "server_reference_sanity")

    def test_no_fallback_contract_distinguishes_trt_mixed_precision_from_external_fallback(self) -> None:
        passed = assess_no_execution_fallback(
            requested_precision="int8",
            builder_flags=["int8", "fp16"],
            execution_provider="tensorrt_engine",
            external_fallback_count=0,
            layer_precision_counts={"int8": 51, "fp16": 7, "fp32": 3, "other": 0},
        )
        self.assertTrue(passed["passed"])
        self.assertTrue(passed["mixed_precision_inside_engine"])

        failed = assess_no_execution_fallback(
            requested_precision="int8",
            builder_flags=["int8", "fp16"],
            execution_provider="pytorch",
            external_fallback_count=1,
            layer_precision_counts={"int8": 51, "fp16": 7, "fp32": 3, "other": 0},
        )
        self.assertFalse(failed["passed"])

    def test_repeat_summary_reports_cv_across_independent_processes(self) -> None:
        report = summarize_repeat_medians([20.0, 20.4, 19.6])
        self.assertEqual(report["repeat_count"], 3)
        self.assertAlmostEqual(report["median_of_medians_ms"], 20.0)
        self.assertAlmostEqual(report["mean_of_medians_ms"], 20.0)
        self.assertGreater(report["cv_percent"], 0.0)


if __name__ == "__main__":
    unittest.main()
