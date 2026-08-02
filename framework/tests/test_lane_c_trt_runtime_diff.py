from __future__ import annotations

import unittest

from tools.orin_deploy.lane_c_trt_runtime_diff import (
    build_runtime_diff,
    parse_calibration_cache,
)


class LaneCTrtRuntimeDiffTest(unittest.TestCase):
    def test_cache_parser_separates_header_from_tensor_scales(self) -> None:
        parsed = parse_calibration_cache(
            b"TRT-8502-EntropyCalibration2\ninput: 3f800000\nout: 40000000\n"
        )
        self.assertEqual(parsed["header"], "TRT-8502-EntropyCalibration2")
        self.assertEqual(parsed["entry_count"], 2)
        self.assertEqual(parsed["entries"]["input"]["hex"], "3f800000")
        self.assertEqual(parsed["entries"]["input"]["float32_be"], 1.0)

    def test_cache_parser_rejects_duplicate_tensor_rows(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse_calibration_cache(
                b"TRT-8502-EntropyCalibration2\n"
                b"input: 3f800000\n"
                b"input: 40000000\n"
            )

    def test_runtime_diff_detects_identical_scales_but_precision_changes(self) -> None:
        h800 = {
            "Bindings": [],
            "Layers": [
                {
                    "Name": "conv",
                    "Inputs": [{"Format/Datatype": "Half"}],
                    "Outputs": [{"Format/Datatype": "Half"}],
                    "TacticName": "h800_fp16",
                }
            ],
        }
        orin = {
            "Bindings": [],
            "Layers": [
                {
                    "Name": "conv",
                    "Inputs": [{"Format/Datatype": "Int8"}],
                    "Outputs": [{"Format/Datatype": "Int8"}],
                    "TacticName": "orin_int8",
                }
            ],
        }
        h800_cache = parse_calibration_cache(
            b"TRT-101300-EntropyCalibration2\ninput: 3f800000\n"
        )
        orin_cache = parse_calibration_cache(
            b"TRT-8502-EntropyCalibration2\ninput: 3f800000\n"
        )
        report = build_runtime_diff(
            h800_inspector=h800,
            orin_inspector=orin,
            h800_cache=h800_cache,
            orin_cache=orin_cache,
        )
        self.assertTrue(report["calibration_cache"]["tensor_scales_identical"])
        self.assertEqual(
            report["inspector"]["shared_layers_with_precision_difference"], 1
        )
        self.assertEqual(report["inspector"]["shared_layers_with_tactic_difference"], 1)
        self.assertEqual(
            report["dynamic_range_visibility"],
            "cache_tensor_scales_visible_internal_fused_layer_ranges_not_exposed",
        )


if __name__ == "__main__":
    unittest.main()
