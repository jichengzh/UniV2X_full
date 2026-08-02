from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import stage3_tvm_int8_quant_contract_v3 as contract  # noqa: E402
import numpy as np


class Stage3TvmInt8QuantContractV3Tests(unittest.TestCase):
    def test_quant_param_is_centered_static_uint8(self) -> None:
        value = contract.quant_param(12.7)
        self.assertAlmostEqual(value["scale"], 0.1)
        self.assertEqual(value["zero_point"], 128)
        self.assertEqual(value["source"], "train16_onnxruntime_static_calibration")

    def test_observed_names_cover_supported_intermediates_and_outputs(self) -> None:
        model = SimpleNamespace(graph=SimpleNamespace(
            input=[SimpleNamespace(name="input")],
            node=[
                SimpleNamespace(op_type="Conv", output=["conv_out"]),
                SimpleNamespace(op_type="Relu", output=["relu_out"]),
                SimpleNamespace(op_type="Unsupported", output=["ignored"]),
            ],
            output=[SimpleNamespace(name="relu_out")],
        ))
        self.assertEqual(contract.observed_tensor_names(model), ["input", "conv_out", "relu_out"])

    def test_nonpositive_absmax_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            contract.quant_param(0.0)

    def test_percentile_clip_rejects_a_large_outlier(self) -> None:
        samples = np.concatenate([
            np.linspace(-1.0, 1.0, 10000, dtype=np.float32),
            np.asarray([100.0], dtype=np.float32),
        ])
        threshold = contract.percentile_clip_threshold(samples)
        self.assertLess(threshold, 10.0)
        self.assertGreaterEqual(threshold, 0.9)

    def test_percentile_clip_keeps_dense_range_without_outliers(self) -> None:
        samples = np.linspace(-2.0, 2.0, 10001, dtype=np.float32)
        self.assertGreater(contract.percentile_clip_threshold(samples), 1.99)

    def test_flattened_pyramid_agents_are_zero_padded_per_scene(self) -> None:
        spatial = np.arange(29 * 2, dtype=np.float32).reshape(29, 1, 1, 2)
        result = contract.normalize_calibration_batch(
            spatial, {"scene_record_lens": [1, 1, 1, *([2] * 13)]},
        )
        self.assertEqual(result.shape, (16, 2, 1, 1, 2))
        np.testing.assert_array_equal(result[0, 0], spatial[0])
        np.testing.assert_array_equal(result[0, 1], np.zeros((1, 1, 2), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
