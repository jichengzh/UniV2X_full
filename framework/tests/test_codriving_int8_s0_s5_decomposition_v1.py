from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import stage2_codriving_int8_s0_s5_decomposition_v1 as decomposition  # noqa: E402


class CoDrivingInt8S0S5DecompositionV1Tests(unittest.TestCase):
    def test_stage_latency_sums_per_conv_kernel_medians(self) -> None:
        profiles = {
            "fp16": [{"latency_ms_p50": 0.4}, {"latency_ms_p50": 0.6}],
            "int8_core": [{"latency_ms_p50": 0.2}, {"latency_ms_p50": 0.3}],
            "int8_materialized": [{"latency_ms_p50": 0.35}, {"latency_ms_p50": 0.45}],
            "int8_epilogue": [{"latency_ms_p50": 0.38}, {"latency_ms_p50": 0.48}],
            "quantize": [{"latency_ms_p50": 0.05}, {"latency_ms_p50": 0.06}],
            "bridge": [{"latency_ms_p50": 0.02}],
        }

        stages = decomposition.summarize_micro_stages(profiles)

        self.assertAlmostEqual(stages["S0"]["latency_ms_p50"], 1.0)
        self.assertAlmostEqual(stages["S1"]["latency_ms_p50"], 0.5)
        self.assertAlmostEqual(stages["S2"]["latency_ms_p50"], 0.8)
        self.assertAlmostEqual(stages["S3"]["latency_ms_p50"], 0.88)
        self.assertAlmostEqual(stages["S4"]["latency_ms_p50"], 0.99)

    def test_reversal_classifier_finds_first_stage_losing_fp16_advantage(self) -> None:
        stages = {
            "S0": {"latency_ms_p50": 1.0},
            "S1": {"latency_ms_p50": 0.6},
            "S2": {"latency_ms_p50": 0.9},
            "S3": {"latency_ms_p50": 1.1},
            "S4": {"latency_ms_p50": 1.2},
        }
        self.assertEqual(decomposition.classify_first_reversal(stages), "S3_epilogue")

    def test_reversal_classifier_reports_core_when_s1_is_not_faster(self) -> None:
        stages = {
            "S0": {"latency_ms_p50": 1.0},
            "S1": {"latency_ms_p50": 1.01},
            "S2": {"latency_ms_p50": 1.2},
            "S3": {"latency_ms_p50": 1.3},
            "S4": {"latency_ms_p50": 1.4},
        }
        self.assertEqual(decomposition.classify_first_reversal(stages), "S1_core")

    def test_linked_full_results_reject_mismatched_gpu(self) -> None:
        fp16 = {
            "status": "success",
            "onnx_sha256": "a" * 64,
            "width": "32x64x128",
            "gpu": "7",
            "selected_int8_node_ids": [],
            "numerical_gate": {"passed": True},
        }
        int8 = {
            "status": "diagnostic_only_accuracy_matrix_calibration",
            "onnx_sha256": "a" * 64,
            "width": "32x64x128",
            "gpu": "6",
            "selected_int8_node_ids": ["first", "second"],
            "numerical_gate": {"passed": True},
            "calibration_binding": "diagnostic_accuracy_matrix_node_id_and_tensor_names",
            "calibration_evidence": {"manifest_sha256": "b" * 64},
            "retain_int8_regions": True,
            "fuse_int8_epilogue": True,
            "region_formation": {"regions": [{"node_ids": ["first", "second"]}]},
        }

        with self.assertRaisesRegex(ValueError, "GPU mismatch"):
            decomposition.validate_linked_full_results(
                fp16,
                int8,
                onnx_sha256="a" * 64,
                width="32x64x128",
                gpu="7",
                calibration_sha256="b" * 64,
                selected_node_ids=["first", "second"],
            )


if __name__ == "__main__":
    unittest.main()
