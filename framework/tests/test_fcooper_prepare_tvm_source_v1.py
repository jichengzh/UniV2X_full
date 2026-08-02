from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "fcooper_prepare_tvm_source_v1.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "fcooper_prepare_tvm_source_v1", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_identity_model(path: Path) -> None:
    graph = helper.make_graph(
        [helper.make_node("Identity", ["input"], ["output"], name="identity")],
        "identity",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 2, 2])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 2, 2])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    onnx.save(model, path)


def save_convtranspose_bn_model(
    path: Path, *, strides: tuple[int, int] = (2, 2)
) -> None:
    weights = np.asarray(
        [
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[-1.0, 0.5], [2.5, -3.0]],
            ]
        ],
        dtype=np.float32,
    )
    initializers = [
        numpy_helper.from_array(weights, "deconv.weight"),
        numpy_helper.from_array(np.asarray([0.25, -0.75], np.float32), "deconv.bias"),
        numpy_helper.from_array(np.asarray([1.5, 0.75], np.float32), "bn.scale"),
        numpy_helper.from_array(np.asarray([0.1, -0.2], np.float32), "bn.bias"),
        numpy_helper.from_array(np.asarray([0.5, -1.0], np.float32), "bn.mean"),
        numpy_helper.from_array(np.asarray([4.0, 0.25], np.float32), "bn.var"),
    ]
    nodes = [
        helper.make_node(
            "ConvTranspose",
            ["input", "deconv.weight", "deconv.bias"],
            ["deconv.output"],
            name="deconv",
            kernel_shape=[2, 2],
            strides=list(strides),
            pads=[0, 0, 0, 0],
            dilations=[1, 1],
            group=1,
        ),
        helper.make_node(
            "BatchNormalization",
            [
                "deconv.output",
                "bn.scale",
                "bn.bias",
                "bn.mean",
                "bn.var",
            ],
            ["output"],
            name="bn",
            epsilon=1e-5,
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "deconv_bn",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 2, 2])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2, 4, 4])],
        initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    onnx.save(model, path)


class FCooperPrepareTvmSourceV1Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_module()
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.source = self.root / "source.onnx"
        self.source_report = self.root / "source_export_report.json"
        self.output = self.root / "prepared.onnx"
        self.report = self.root / "prepare_report.json"

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def write_formal_report(self) -> str:
        payload = {
            "schema_version": "fcooper_source_export_v2",
            "status": "success",
            "source_kind": "formal_recovered",
            "formal_measurement_eligible": True,
            "onnx_path": str(self.source),
            "onnx_sha256": sha256_file(self.source),
        }
        self.source_report.write_text(json.dumps(payload))
        return sha256_file(self.source_report)

    def args(self, report_sha: str, *, decompose: bool = False) -> argparse.Namespace:
        return argparse.Namespace(
            source_onnx=self.source,
            source_export_report=self.source_report,
            source_export_report_sha256=report_sha,
            output_onnx=self.output,
            report=self.report,
            decompose_convtranspose=decompose,
            seed=7,
            samples=3,
            rtol=1e-5,
            atol=1e-5,
        )

    def test_rejects_source_export_report_sha_mismatch(self) -> None:
        save_identity_model(self.source)
        report_sha = self.write_formal_report()

        with self.assertRaisesRegex(ValueError, "source export report SHA256 mismatch"):
            self.module.prepare_source(self.args("0" * 64))

        self.assertNotEqual(report_sha, "0" * 64)
        self.assertFalse(self.output.exists())
        failure = json.loads(self.report.read_text())
        self.assertEqual(failure["status"], "failed")

    def test_rejects_nonformal_source_contract(self) -> None:
        save_identity_model(self.source)
        report_sha = self.write_formal_report()
        payload = json.loads(self.source_report.read_text())
        payload["formal_measurement_eligible"] = False
        self.source_report.write_text(json.dumps(payload))

        with self.assertRaisesRegex(ValueError, "formal backend-neutral source"):
            self.module.prepare_source(self.args(sha256_file(self.source_report)))

        self.assertNotEqual(report_sha, sha256_file(self.source_report))
        self.assertFalse(self.output.exists())

    def test_rejects_source_onnx_sha_mismatch(self) -> None:
        save_identity_model(self.source)
        self.write_formal_report()
        payload = json.loads(self.source_report.read_text())
        payload["onnx_sha256"] = "f" * 64
        self.source_report.write_text(json.dumps(payload))

        with self.assertRaisesRegex(ValueError, "source ONNX SHA256 mismatch"):
            self.module.prepare_source(self.args(sha256_file(self.source_report)))

        self.assertFalse(self.output.exists())

    def test_prepares_without_optional_decomposition(self) -> None:
        save_identity_model(self.source)
        report_sha = self.write_formal_report()

        result = self.module.prepare_source(self.args(report_sha))

        onnx.checker.check_model(onnx.load(self.output))
        persisted = json.loads(self.report.read_text())
        self.assertEqual(persisted, result)
        self.assertFalse(
            result["transformations"]["convtranspose_bn_decomposition_enabled"]
        )
        self.assertEqual(result["transformations"]["convtranspose_bn_decomposed"], 0)
        self.assertEqual(result["numerical_equivalence"]["status"], "passed")

    def test_decomposes_no_overlap_convtranspose_bn_equivalently(self) -> None:
        save_convtranspose_bn_model(self.source)
        source_sha = sha256_file(self.source)
        report_sha = self.write_formal_report()

        result = self.module.prepare_source(self.args(report_sha, decompose=True))

        prepared = onnx.load(self.output)
        onnx.checker.check_model(prepared)
        op_types = [node.op_type for node in prepared.graph.node]
        self.assertNotIn("ConvTranspose", op_types)
        self.assertNotIn("BatchNormalization", op_types)
        self.assertIn("Conv", op_types)
        self.assertIn("DepthToSpace", op_types)

        sample = np.random.default_rng(11).normal(size=(1, 1, 2, 2)).astype(np.float32)
        original_output = ort.InferenceSession(
            str(self.source), providers=["CPUExecutionProvider"]
        ).run(None, {"input": sample})[0]
        prepared_output = ort.InferenceSession(
            str(self.output), providers=["CPUExecutionProvider"]
        ).run(None, {"input": sample})[0]
        np.testing.assert_allclose(
            prepared_output, original_output, rtol=1e-5, atol=1e-5
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["input_sha256"], source_sha)
        self.assertEqual(result["output_sha256"], sha256_file(self.output))
        self.assertEqual(result["source_export_report_sha256"], report_sha)
        self.assertEqual(result["transformations"]["convtranspose_bn_decomposed"], 1)
        self.assertEqual(result["numerical_equivalence"]["samples"], 3)
        self.assertIn("input", result["node_counts"])
        self.assertIn("output", result["op_counts"])
        self.assertGreaterEqual(result["elapsed_seconds"], 0)

    def test_rejects_unsupported_convtranspose_when_decomposition_requested(
        self,
    ) -> None:
        save_convtranspose_bn_model(self.source, strides=(1, 1))
        report_sha = self.write_formal_report()

        with self.assertRaisesRegex(
            ValueError,
            "unsupported ConvTranspose 'deconv'.*kernel_shape must equal strides",
        ):
            self.module.prepare_source(self.args(report_sha, decompose=True))

        self.assertFalse(self.output.exists())
        failure = json.loads(self.report.read_text())
        self.assertEqual(failure["status"], "failed")
        self.assertIn("kernel_shape must equal strides", failure["error"])


if __name__ == "__main__":
    unittest.main()
