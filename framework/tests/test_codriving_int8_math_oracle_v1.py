from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import stage2_codriving_int8_math_oracle_v1 as oracle  # noqa: E402


class CoDrivingInt8MathOracleV1Tests(unittest.TestCase):
    def test_quantize_s8_matches_round_and_signed_clip(self) -> None:
        values = np.asarray([-20.0, -0.15, -0.05, 0.05, 0.15, 20.0], dtype="float32")

        quantized, stats = oracle.quantize_s8_reference(values, scale=0.1)

        np.testing.assert_array_equal(
            quantized,
            np.asarray([-127, -2, 0, 0, 2, 127], dtype="int8"),
        )
        self.assertAlmostEqual(stats["saturation_ratio"], 2.0 / 6.0)

    def test_grouped_conv_reference_keeps_groups_separate(self) -> None:
        activation = np.asarray(
            [[[[1, 2], [3, 4]], [[10, 20], [30, 40]]]],
            dtype="int8",
        )
        weight = np.asarray([[[[2]]], [[[3]]]], dtype="int8")

        output = oracle.conv2d_nchw_int32_reference(
            activation,
            weight,
            strides=[1, 1],
            pads=[0, 0, 0, 0],
            group=2,
        )

        expected = np.asarray(
            [[[[2, 4], [6, 8]], [[30, 60], [90, 120]]]],
            dtype="int32",
        )
        np.testing.assert_array_equal(output, expected)

    def test_conv_reference_honors_padding_and_stride(self) -> None:
        activation = np.arange(1, 10, dtype="int8").reshape(1, 1, 3, 3)
        weight = np.ones((1, 1, 2, 2), dtype="int8")

        output = oracle.conv2d_nchw_int32_reference(
            activation,
            weight,
            strides=[2, 2],
            pads=[1, 1, 0, 0],
            group=1,
        )

        np.testing.assert_array_equal(
            output,
            np.asarray([[[[1, 5], [11, 28]]]], dtype="int32"),
        )

    def test_requant_bias_uses_accumulator_scale_and_clips(self) -> None:
        accum = np.asarray([[[[-1000, -10, 0, 10, 1000]]]], dtype="int32")
        bias = np.asarray([0.5], dtype="float16")

        output = oracle.requantize_s8_reference(
            accum,
            bias,
            input_scale=0.2,
            weight_scale=0.1,
            output_scale=0.1,
        )

        np.testing.assert_array_equal(
            output,
            np.asarray([[[[-127, 3, 5, 7, 127]]]], dtype="int8"),
        )

    def test_sample_coordinates_are_deterministic_and_include_boundaries(self) -> None:
        first = oracle.deterministic_output_samples([2, 4, 5, 6], count=12, seed=7)
        second = oracle.deterministic_output_samples([2, 4, 5, 6], count=12, seed=7)

        self.assertEqual(first, second)
        self.assertIn((0, 0, 0, 0), first)
        self.assertIn((1, 3, 4, 5), first)
        self.assertEqual(len(first), len(set(first)))

    def test_sampled_accumulator_matches_full_reference(self) -> None:
        activation = np.arange(-8, 10, dtype="int8").reshape(1, 2, 3, 3)
        weight = np.asarray(
            [
                [[[1, 0], [0, -1]], [[2, 0], [0, -2]]],
                [[[-1, 1], [1, -1]], [[1, 1], [1, 1]]],
            ],
            dtype="int8",
        )
        full = oracle.conv2d_nchw_int32_reference(
            activation,
            weight,
            strides=[1, 1],
            pads=[0, 0, 0, 0],
            group=1,
        )
        samples = [(0, 0, 0, 0), (0, 1, 1, 1)]

        sampled = oracle.sample_conv2d_accumulators(
            activation,
            weight,
            samples=samples,
            strides=[1, 1],
            pads=[0, 0, 0, 0],
            group=1,
        )

        self.assertEqual(sampled, [int(full[index]) for index in samples])

    def test_extract_tvm_accumulator_samples_maps_grouped_padded_layout(self) -> None:
        tvm_accumulator = np.zeros((2, 4, 128), dtype="int32")
        tvm_accumulator[0, 0, 1] = 11
        tvm_accumulator[1, 3, 2] = 22

        extracted = oracle.extract_tvm_accumulator_samples(
            tvm_accumulator,
            samples=[(0, 1, 0, 0), (0, 5, 1, 1)],
            output_shape=[1, 6, 2, 2],
            group=2,
        )

        self.assertEqual(extracted, [11, 22])

    def test_compare_sampled_accumulators_reports_exact_mismatches(self) -> None:
        comparison = oracle.compare_sampled_accumulators(
            samples=[(0, 0, 0, 0), (0, 1, 0, 0), (0, 2, 0, 0)],
            reference=[7, -2, 100],
            candidate=[7, 3, 98],
        )

        self.assertFalse(comparison["exact_match"])
        self.assertEqual(comparison["mismatch_count"], 2)
        self.assertEqual(comparison["max_abs_error"], 5)
        self.assertEqual(comparison["mismatches"][0]["coordinate"], [0, 1, 0, 0])

    def test_sampled_epilogue_matches_full_dequant_and_requant(self) -> None:
        samples = [(0, 0, 0, 0), (0, 1, 0, 0)]
        accumulators = [10, -20]
        bias = np.asarray([0.5, -1.0], dtype="float16")

        dequantized = oracle.dequantize_sampled_fp16_reference(
            accumulators,
            samples=samples,
            bias=bias,
            accumulator_scale=0.02,
        )
        requantized = oracle.requantize_sampled_s8_reference(
            accumulators,
            samples=samples,
            bias=bias,
            accumulator_scale=0.02,
            output_scale=0.1,
        )

        np.testing.assert_array_equal(dequantized, np.asarray([0.7, -1.4], dtype="float16"))
        np.testing.assert_array_equal(requantized, np.asarray([7, -14], dtype="int8"))


if __name__ == "__main__":
    unittest.main()
