from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from framework.stage2.s1_neutral_probes_v3 import (
    PROBE_SPECS,
    build_probe_model,
    generate_neutral_probes,
    load_onnx_package,
)


class Stage2S1NeutralProbesV3Test(unittest.TestCase):
    def test_builds_fixed_shape_fp16_topologies(self) -> None:
        onnx = load_onnx_package()
        expected_ops = {
            "P1": ["Conv"],
            "P2": ["Conv"],
            "P3": ["Conv", "Conv"],
            "P4": ["Conv", "Add"],
            "P5": ["Conv"],
            "P6": ["Conv"],
        }

        for probe_id, ops in expected_ops.items():
            with self.subTest(probe_id=probe_id):
                model = build_probe_model(probe_id, "fp16")
                onnx.checker.check_model(model)
                self.assertEqual([node.op_type for node in model.graph.node], ops)
                self.assertEqual(model.graph.input[0].type.tensor_type.elem_type, onnx.TensorProto.FLOAT16)
                self.assertEqual(
                    [dim.dim_value for dim in model.graph.input[0].type.tensor_type.shape.dim],
                    PROBE_SPECS[probe_id]["input_shape"],
                )

        p1_conv = build_probe_model("P1", "fp16").graph.node[0]
        p2_conv = build_probe_model("P2", "fp16").graph.node[0]
        p5_conv = build_probe_model("P5", "fp16").graph.node[0]
        self.assertEqual(self._attribute(p1_conv, "strides").ints, [1, 1])
        self.assertEqual(self._attribute(p2_conv, "strides").ints, [2, 2])
        self.assertEqual(self._attribute(p5_conv, "group").i, PROBE_SPECS["P5"]["group"])

    def test_int8_variants_use_explicit_qdq_around_each_conv(self) -> None:
        onnx = load_onnx_package()
        for probe_id in PROBE_SPECS:
            with self.subTest(probe_id=probe_id):
                model = build_probe_model(probe_id, "int8")
                onnx.checker.check_model(model)
                ops = [node.op_type for node in model.graph.node]
                conv_count = ops.count("Conv")
                expected_qdq = conv_count * 2 + (1 if probe_id == "P6" else 0)
                self.assertEqual(ops.count("QuantizeLinear"), expected_qdq)
                self.assertEqual(ops.count("DequantizeLinear"), expected_qdq)
                self.assertEqual(model.graph.input[0].type.tensor_type.elem_type, onnx.TensorProto.FLOAT)
                zero_points = [item for item in model.graph.initializer if item.name.endswith("zero_point")]
                self.assertTrue(zero_points)
                self.assertTrue(all(item.data_type == onnx.TensorProto.INT8 for item in zero_points))

    def test_generates_twelve_checked_models_and_sha_manifest(self) -> None:
        onnx = load_onnx_package()
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = generate_neutral_probes(Path(tmp))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

            self.assertEqual(manifest["schema"], "stage2_s1_neutral_probes_v3")
            self.assertEqual(len(manifest["probes"]), 12)
            self.assertEqual({record["precision"] for record in manifest["probes"]}, {"fp16", "int8"})
            self.assertEqual(manifest_path.name, "probe_manifest.json")
            for record in manifest["probes"]:
                model_path = Path(tmp) / record["path"]
                self.assertEqual(hashlib.sha256(model_path.read_bytes()).hexdigest(), record["sha256"])
                self.assertEqual(record["checker_status"], "passed")
                self.assertEqual(record["input_shape"], PROBE_SPECS[record["probe_id"]]["input_shape"])
                onnx.checker.check_model(onnx.load(str(model_path)))

    @staticmethod
    def _attribute(node, name):
        return next(attribute for attribute in node.attribute if attribute.name == name)


if __name__ == "__main__":
    unittest.main()
