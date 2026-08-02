from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import stage2_codriving_int8_accuracy_matrix_v1 as matrix  # noqa: E402


class CoDrivingInt8AccuracyMatrixV1Tests(unittest.TestCase):
    def test_activation_scale_supports_absmax_and_percentile(self) -> None:
        values = [np.asarray([-100.0, -2.0, 0.0, 1.0, 2.0], dtype="float32")]

        absmax = matrix.activation_scale(values, strategy="absmax", percentile=99.0)
        percentile = matrix.activation_scale(values, strategy="percentile", percentile=80.0)

        self.assertAlmostEqual(absmax, 100.0 / 127.0)
        self.assertLess(percentile, absmax)
        self.assertGreater(percentile, 0.0)

    def test_weight_scales_and_quantization_support_output_channels(self) -> None:
        weight = np.asarray([[[[1.0, -2.0]]], [[[10.0, -20.0]]]], dtype="float32")

        tensor_scales = matrix.weight_scales(weight, granularity="per_tensor")
        channel_scales = matrix.weight_scales(weight, granularity="per_output_channel")
        quantized = matrix.quantize_weight(weight, channel_scales)

        self.assertEqual(tensor_scales.shape, ())
        np.testing.assert_allclose(channel_scales, np.asarray([2.0 / 127.0, 20.0 / 127.0]))
        np.testing.assert_array_equal(
            quantized,
            np.asarray([[[[64, -127]]], [[[64, -127]]]], dtype="int8"),
        )

    def test_quantize_dequantize_reports_finite_error_and_saturation(self) -> None:
        values = np.asarray([-20.0, -0.1, 0.0, 0.1, 20.0], dtype="float32")

        dequantized, metrics = matrix.quantize_dequantize_activation(values, scale=0.1)

        self.assertTrue(np.isfinite(dequantized).all())
        self.assertEqual(metrics["saturation_count"], 2)
        self.assertAlmostEqual(metrics["saturation_ratio"], 0.4)
        self.assertGreater(metrics["mse"], 0.0)
        self.assertLessEqual(metrics["cosine_similarity"], 1.0)

    def test_build_tvm_candidate_calibration_is_explicitly_diagnostic(self) -> None:
        base = {
            "schema": "codriving_raw_onnx_per_node_calibration_v1",
            "quantization_semantics": "symmetric_absmax_int8_per_raw_onnx_node",
            "nodes": {
                "first": {"input_scale": 1.0, "weight_scale": 0.1},
                "second": {"input_scale": 2.0, "weight_scale": 0.2},
                "other": {"input_scale": 3.0, "weight_scale": 0.3},
            },
        }
        row = {
            "point_id": "candidate",
            "weight_scale_granularity": "per_tensor",
            "first_conv_input_scale": 0.5,
            "second_conv_input_scale": 0.25,
        }

        result = matrix.build_tvm_candidate_calibration(
            base,
            row=row,
            selected_node_ids=["first", "second"],
            parent_sha256="a" * 64,
        )

        self.assertEqual(result["schema"], "codriving_int8_accuracy_matrix_calibration_v1")
        self.assertEqual(result["nodes"]["first"]["input_scale"], 0.5)
        self.assertEqual(result["nodes"]["second"]["input_scale"], 0.25)
        self.assertEqual(result["nodes"]["other"]["input_scale"], 3.0)
        self.assertEqual(base["nodes"]["first"]["input_scale"], 1.0)

    def test_qdq_model_clips_signed_int8_to_negative_127(self) -> None:
        import importlib

        saved_path = list(sys.path)
        sys.modules.pop("onnx", None)
        sys.path = [item for item in sys.path if item not in ("", str(REPO_ROOT))]
        try:
            onnx = importlib.import_module("onnx")
        except ModuleNotFoundError:
            sys.path = saved_path
            self.skipTest("real ONNX package is unavailable in the local test interpreter")
        sys.path = saved_path
        from onnx import TensorProto, helper, numpy_helper

        model = helper.make_model(
            helper.make_graph(
                [helper.make_node("Conv", ["x", "w"], ["y"], name="conv")],
                "clip_test",
                [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 1, 1])],
                [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 1, 1])],
                [numpy_helper.from_array(np.ones((1, 1, 1, 1), dtype="float32"), name="w")],
            ),
            opset_imports=[helper.make_opsetid("", 13)],
        )

        rewritten = matrix._qdq_model(
            model,
            [{"rank": 1, "node_index": 0, "activation_scale": 0.1, "dequantized_weight": np.ones((1, 1, 1, 1))}],
        )

        self.assertEqual([node.op_type for node in rewritten.graph.node], ["QuantizeLinear", "Clip", "DequantizeLinear", "Conv"])
        initializers = {item.name: numpy_helper.to_array(item) for item in rewritten.graph.initializer}
        self.assertIn(np.asarray(-127, dtype="int8"), list(initializers.values()))
        self.assertIn(np.asarray(127, dtype="int8"), list(initializers.values()))

    def test_multisample_gate_uses_worst_sample_not_only_first(self) -> None:
        reference = [
            [np.asarray([0.0, 0.0], dtype="float32")],
            [np.asarray([0.0, 0.0], dtype="float32")],
        ]
        candidate = [
            [np.asarray([0.01, 0.01], dtype="float32")],
            [np.asarray([1.0, 1.0], dtype="float32")],
        ]

        errors, passed = matrix.aggregate_multisample_output_errors(
            reference,
            candidate,
            max_mean_abs_error=0.05,
        )

        self.assertFalse(passed)
        self.assertAlmostEqual(errors[0]["mean_abs_error"], 0.505)
        self.assertAlmostEqual(errors[0]["worst_sample_mean_abs_error"], 1.0)


if __name__ == "__main__":
    unittest.main()
