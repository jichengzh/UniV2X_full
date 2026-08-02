from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from framework.stage2.s1_pruning_shape_probes_v3 import (
    PROBE_SPECS,
    build_probe_model,
    check_probe_manifest,
    generate_pruning_shape_probes,
    load_onnx_package,
)


class Stage2S1PruningShapeProbesV3Test(unittest.TestCase):
    def test_builds_six_fixed_shape_neutral_topologies(self) -> None:
        onnx = load_onnx_package()
        expected = {
            "aligned_channels": ["Conv"],
            "misaligned_channels": ["Conv"],
            "small_channels": ["Conv"],
            "group_packed": ["Conv"],
            "group_unpacked": ["Conv"],
            "off_diagonal_two_stage_boundary": ["Conv", "Relu", "Conv"],
        }

        self.assertEqual(set(PROBE_SPECS), set(expected))
        for probe_id, operations in expected.items():
            with self.subTest(probe_id=probe_id):
                model = build_probe_model(probe_id, "fp16")
                onnx.checker.check_model(model)
                self.assertEqual([node.op_type for node in model.graph.node], operations)
                self.assertEqual(len(model.graph.input[0].type.tensor_type.shape.dim), 4)
                self.assertEqual(
                    model.graph.input[0].type.tensor_type.elem_type,
                    onnx.TensorProto.FLOAT16,
                )

        packed = build_probe_model("group_packed", "fp16").graph.node[0]
        unpacked = build_probe_model("group_unpacked", "fp16").graph.node[0]
        self.assertEqual(self._attribute(packed, "group").i, PROBE_SPECS["group_packed"]["group"])
        self.assertEqual(self._attribute(unpacked, "group").i, PROBE_SPECS["group_unpacked"]["group"])

    def test_int8_variants_are_explicit_signed_qdq_graphs(self) -> None:
        onnx = load_onnx_package()
        for probe_id in PROBE_SPECS:
            with self.subTest(probe_id=probe_id):
                model = build_probe_model(probe_id, "int8_qdq")
                onnx.checker.check_model(model)
                operations = [node.op_type for node in model.graph.node]
                conv_count = operations.count("Conv")
                self.assertEqual(operations.count("QuantizeLinear"), conv_count * 2)
                self.assertEqual(operations.count("DequantizeLinear"), conv_count * 2)
                self.assertEqual(
                    model.graph.input[0].type.tensor_type.elem_type,
                    onnx.TensorProto.FLOAT,
                )
                zero_points = [
                    value for value in model.graph.initializer if value.name.endswith("zero_point")
                ]
                self.assertEqual(len(zero_points), conv_count * 2)
                self.assertTrue(
                    all(value.data_type == onnx.TensorProto.INT8 for value in zero_points)
                )

    def test_manifest_is_content_addressed_and_contains_shape_alignment_metadata(self) -> None:
        onnx = load_onnx_package()
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = generate_pruning_shape_probes(tmp)
            manifest_bytes = manifest_path.read_bytes()
            manifest = json.loads(manifest_bytes)

            self.assertRegex(manifest_path.name, r"^manifest-[0-9a-f]{64}\.json$")
            self.assertEqual(
                manifest_path.stem.removeprefix("manifest-"),
                hashlib.sha256(manifest_bytes).hexdigest(),
            )
            self.assertEqual(manifest["schema"], "stage2_s1_pruning_shape_probes_v3")
            self.assertEqual(manifest["probe_count"], 12)
            self.assertEqual(
                {(item["probe_id"], item["precision"]) for item in manifest["probes"]},
                {(probe_id, precision) for probe_id in PROBE_SPECS for precision in ("fp16", "int8_qdq")},
            )

            for item in manifest["probes"]:
                model_path = Path(tmp) / item["artifact"]["path"]
                self.assertIn(item["artifact"]["sha256"], model_path.name)
                self.assertEqual(
                    hashlib.sha256(model_path.read_bytes()).hexdigest(),
                    item["artifact"]["sha256"],
                )
                self.assertEqual(item["checker"], {"name": "onnx.checker", "status": "passed"})
                self.assertEqual(item["shape"]["input"], PROBE_SPECS[item["probe_id"]]["input_shape"])
                self.assertEqual(item["shape"]["output"], PROBE_SPECS[item["probe_id"]]["output_shape"])
                self.assertIn("channel_multiple", item["alignment"])
                self.assertIn("input_channels_aligned", item["alignment"])
                self.assertIn("output_channels_aligned", item["alignment"])
                onnx.checker.check_model(onnx.load(str(model_path)))

            boundary = next(
                item
                for item in manifest["probes"]
                if item["probe_id"] == "off_diagonal_two_stage_boundary"
                and item["precision"] == "fp16"
            )
            self.assertEqual(boundary["shape"]["stage_boundary"], [1, 48, 8, 8])
            self.assertEqual(boundary["alignment"]["stage_boundary_channels_aligned"], True)
            self.assertEqual(boundary["alignment"]["off_diagonal"], True)

            packed_fp16 = next(
                item for item in manifest["probes"]
                if item["probe_id"] == "group_packed" and item["precision"] == "fp16"
            )
            packed_int8 = next(
                item for item in manifest["probes"]
                if item["probe_id"] == "group_packed" and item["precision"] == "int8_qdq"
            )
            self.assertEqual(packed_fp16["alignment"]["channel_multiple"], 8)
            self.assertTrue(packed_fp16["alignment"]["output_per_group_aligned"])
            self.assertEqual(packed_int8["alignment"]["channel_multiple"], 16)
            self.assertFalse(packed_int8["alignment"]["output_per_group_aligned"])
            self.assertTrue(check_probe_manifest(manifest_path))

    def test_checker_rejects_an_artifact_modified_after_generation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = generate_pruning_shape_probes(tmp)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            artifact = Path(tmp) / manifest["probes"][0]["artifact"]["path"]
            artifact.write_bytes(artifact.read_bytes() + b"tampered")

            with self.assertRaisesRegex(ValueError, "sha256 mismatch"):
                check_probe_manifest(manifest_path)

    @staticmethod
    def _attribute(node, name):
        return next(attribute for attribute in node.attribute if attribute.name == name)


if __name__ == "__main__":
    unittest.main()
