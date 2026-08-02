from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import stage2_codriving_mixed_auto_lowering_v1 as lowering  # noqa: E402


class CoDrivingMixedAutoLoweringTests(unittest.TestCase):
    @staticmethod
    def _strict_calibration(records: list[dict[str, object]], root: Path) -> dict[str, object]:
        source = root / "calibration.npz"
        summary = root / "calibration_summary.json"
        split_source = root / "train.json"
        source.write_bytes(b"calibration")
        summary.write_text("{}", encoding="utf-8")
        split_source.write_text("[]", encoding="utf-8")
        return {
            "schema": "codriving_raw_onnx_per_node_calibration_v1",
            "quantization_semantics": "symmetric_absmax_int8_per_raw_onnx_node",
            "calibration_split": "train",
            "sample_count": 16,
            "onnx_sha256": "a" * 64,
            "calibration_source": str(source),
            "calibration_source_sha256": lowering.sha256_file(source),
            "calibration_summary": str(summary),
            "calibration_summary_sha256": lowering.sha256_file(summary),
            "calibration_split_source": str(split_source),
            "node_count": len(records),
            "nodes": {
                str(record["node_id"]): {
                    "input_name": record["input_name"],
                    "weight_name": record["weight_name"],
                    "output_name": record["output_name"],
                    "input_scale": 0.1,
                    "weight_scale": 0.01,
                }
                for record in records
            },
        }

    def test_select_topk_uses_stable_mac_then_node_id_order(self) -> None:
        records = [
            {"node_id": "conv_b", "macs": 100},
            {"node_id": "conv_a", "macs": 100},
            {"node_id": "conv_c", "macs": 50},
            {"node_id": "conv_d", "macs": 25},
        ]
        selected = lowering.select_conv_ids(records, "top50_flops")
        self.assertEqual(selected, ["conv_a", "conv_b"])

    def test_select_topk_rounds_coverage_up(self) -> None:
        records = [{"node_id": f"conv_{index}", "macs": 100 - index} for index in range(7)]
        self.assertEqual(len(lowering.select_conv_ids(records, "top25_flops")), 2)
        self.assertEqual(len(lowering.select_conv_ids(records, "top10_flops")), 1)
        self.assertEqual(len(lowering.select_conv_ids(records, "top75_flops")), 6)

    def test_selection_rejects_duplicate_node_ids(self) -> None:
        records = [{"node_id": "conv", "macs": 100}, {"node_id": "conv", "macs": 50}]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            lowering.select_conv_ids(records, "top25_flops")

    def test_quantize_weight_s8_uses_scale_round_and_signed_clip(self) -> None:
        value = np.asarray([-20.0, -0.14, 0.0, 0.16, 20.0], dtype="float32")
        got = lowering.quantize_weight_s8(value, scale=0.1)
        np.testing.assert_array_equal(got, np.asarray([-127, -1, 0, 2, 127], dtype="int8"))

    def test_conv_signature_is_shape_stable(self) -> None:
        self.assertEqual(
            lowering.conv_signature([2, 64, 256, 512], [32, 64, 3, 3]),
            "input=2x64x256x512|weight=32x64x3x3",
        )

    def test_node_scale_binding_requires_exact_node_and_tensor_names(self) -> None:
        calibration = {
            "schema": "codriving_raw_onnx_per_node_calibration_v1",
            "onnx_sha256": "a" * 64,
            "nodes": {
                "node0": {
                    "input_name": "x0",
                    "weight_name": "w0",
                    "input_scale": 0.1,
                    "weight_scale": 0.01,
                }
            },
        }
        scales = lowering.resolve_node_scales(
            calibration,
            node_id="node0",
            input_name="x0",
            weight_name="w0",
            onnx_sha256="a" * 64,
        )
        self.assertEqual(scales, {"input_scale": 0.1, "weight_scale": 0.01})
        with self.assertRaisesRegex(ValueError, "weight_name"):
            lowering.resolve_node_scales(
                calibration,
                node_id="node0",
                input_name="x0",
                weight_name="wrong",
                onnx_sha256="a" * 64,
            )

    def test_formal_lowering_calibration_requires_train16_and_full_coverage(self) -> None:
        records = [
            {"node_id": "node0", "input_name": "x0", "weight_name": "w0", "output_name": "y0"},
            {"node_id": "node1", "input_name": "x1", "weight_name": "w1", "output_name": "y1"},
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            calibration = self._strict_calibration(records, Path(temp_dir))
            lowering.validate_per_node_calibration_for_lowering(
                calibration,
                records=records,
                onnx_sha256="a" * 64,
            )

            wrong_count = {**calibration, "sample_count": 15}
            with self.assertRaisesRegex(ValueError, "sample_count"):
                lowering.validate_per_node_calibration_for_lowering(
                    wrong_count,
                    records=records,
                    onnx_sha256="a" * 64,
                )

            partial = {
                **calibration,
                "node_count": 1,
                "nodes": {"node0": calibration["nodes"]["node0"]},
            }
            with self.assertRaisesRegex(ValueError, "coverage"):
                lowering.validate_per_node_calibration_for_lowering(
                    partial,
                    records=records,
                    onnx_sha256="a" * 64,
                )

            missing_provenance = {**calibration, "calibration_split_source": ""}
            with self.assertRaisesRegex(ValueError, "calibration_split_source"):
                lowering.validate_per_node_calibration_for_lowering(
                    missing_provenance,
                    records=records,
                    onnx_sha256="a" * 64,
                )

            Path(str(calibration["calibration_source"])).write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                lowering.validate_per_node_calibration_for_lowering(
                    calibration,
                    records=records,
                    onnx_sha256="a" * 64,
                )

    def test_accuracy_matrix_calibration_requires_explicit_diagnostic_opt_in(self) -> None:
        records = [
            {"node_id": "node0", "input_name": "x0", "weight_name": "w0", "output_name": "y0"},
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            base = self._strict_calibration(records, Path(temp_dir))
            diagnostic = {
                **base,
                "schema": "codriving_int8_accuracy_matrix_calibration_v1",
                "quantization_semantics": "symmetric_percentile_int8_per_raw_onnx_node",
                "parent_calibration_sha256": "b" * 64,
            }
            with self.assertRaisesRegex(ValueError, "diagnostic"):
                lowering.validate_per_node_calibration_for_lowering(
                    diagnostic,
                    records=records,
                    onnx_sha256="a" * 64,
                )
            lowering.validate_per_node_calibration_for_lowering(
                diagnostic,
                records=records,
                onnx_sha256="a" * 64,
                allow_diagnostic_accuracy_matrix=True,
            )

    def test_formal_int8_cli_rejects_legacy_calibration(self) -> None:
        with self.assertRaisesRegex(ValueError, "legacy"):
            lowering.validate_calibration_mode(
                int8_conv_limit=1,
                per_node_calibration=Path("strict.json"),
                calibration_ap_report=Path("legacy.json"),
                allow_legacy_shape_calibration=True,
            )
        with self.assertRaisesRegex(ValueError, "per-node"):
            lowering.validate_calibration_mode(
                int8_conv_limit=1,
                per_node_calibration=None,
                calibration_ap_report=None,
                allow_legacy_shape_calibration=False,
            )

    def test_non_success_measurement_status_returns_nonzero(self) -> None:
        self.assertEqual(lowering.result_exit_code({"status": "success"}), 0)
        self.assertNotEqual(lowering.result_exit_code({"status": "failed_numerical_gate"}), 0)
        self.assertNotEqual(
            lowering.result_exit_code({"status": "diagnostic_only_legacy_shape_calibration"}),
            0,
        )

    def test_full_output_comparison_rejects_output_count_mismatch(self) -> None:
        reference = [np.zeros((1,), dtype="float32"), np.zeros((1,), dtype="float32")]
        candidate = [np.zeros((1,), dtype="float32")]

        with self.assertRaisesRegex(ValueError, "output count mismatch"):
            lowering.compare_full_outputs(reference, candidate)

    def test_region_formation_merges_unique_conv_relu_conv_chain(self) -> None:
        nodes = [
            {"node_index": 0, "op_type": "Conv", "node_id": "c0", "inputs": ["x"], "outputs": ["a"]},
            {"node_index": 1, "op_type": "Relu", "inputs": ["a"], "outputs": ["b"]},
            {"node_index": 2, "op_type": "Conv", "node_id": "c1", "inputs": ["b"], "outputs": ["c"]},
            {"node_index": 3, "op_type": "Add", "inputs": ["c", "skip"], "outputs": ["d"]},
            {"node_index": 4, "op_type": "Conv", "node_id": "c2", "inputs": ["d"], "outputs": ["e"]},
        ]
        result = lowering.form_int8_regions(nodes, {"c0", "c1", "c2"})
        self.assertEqual([item["node_ids"] for item in result["regions"]], [["c0", "c1"], ["c2"]])
        self.assertEqual(result["region_count"], 2)
        self.assertEqual(result["quantize_boundary_count"], 2)
        self.assertEqual(result["dequantize_boundary_count"], 2)
        self.assertEqual(result["per_conv_quantize_boundary_count"], 3)
        self.assertEqual(result["regions"][0]["bridges"][0]["op_types"], ["Relu"])
        self.assertIn("Add", result["regions"][0]["exit_break_reason"])

    def test_region_formation_breaks_on_fanout_without_model_specific_rules(self) -> None:
        nodes = [
            {"node_index": 0, "op_type": "Conv", "node_id": "c0", "inputs": ["x"], "outputs": ["a"]},
            {"node_index": 1, "op_type": "Relu", "inputs": ["a"], "outputs": ["b"]},
            {"node_index": 2, "op_type": "Conv", "node_id": "c1", "inputs": ["b"], "outputs": ["c"]},
            {"node_index": 3, "op_type": "Identity", "inputs": ["a"], "outputs": ["side"]},
        ]
        result = lowering.form_int8_regions(nodes, {"c0", "c1"})
        self.assertEqual([item["node_ids"] for item in result["regions"]], [["c0"], ["c1"]])
        self.assertEqual(result["regions"][0]["exit_break_reason"], "fanout")

    def test_region_formation_breaks_when_conv_tensor_is_graph_output(self) -> None:
        nodes = [
            {"node_index": 0, "op_type": "Conv", "node_id": "c0", "inputs": ["x"], "outputs": ["a"]},
            {"node_index": 1, "op_type": "Relu", "inputs": ["a"], "outputs": ["b"]},
            {"node_index": 2, "op_type": "Conv", "node_id": "c1", "inputs": ["b"], "outputs": ["c"]},
        ]
        for graph_output_name in ("a", "b"):
            with self.subTest(graph_output_name=graph_output_name):
                result = lowering.form_int8_regions(
                    nodes,
                    {"c0", "c1"},
                    graph_output_names={graph_output_name},
                )
                self.assertEqual([item["node_ids"] for item in result["regions"]], [["c0"], ["c1"]])
                self.assertEqual(result["regions"][0]["exit_break_reason"], "graph_output")

    def test_unsupported_conv_attributes_fail_closed(self) -> None:
        lowering.validate_supported_conv_attrs(
            {"dilations": [1, 1], "strides": [1, 1], "pads": [1, 1, 1, 1], "group": 1},
            node_id="c0",
        )
        with self.assertRaisesRegex(ValueError, "dilations"):
            lowering.validate_supported_conv_attrs({"dilations": [2, 1]}, node_id="c0")
        with self.assertRaisesRegex(ValueError, "auto_pad"):
            lowering.validate_supported_conv_attrs({"auto_pad": b"SAME_UPPER"}, node_id="c0")


if __name__ == "__main__":
    unittest.main()
