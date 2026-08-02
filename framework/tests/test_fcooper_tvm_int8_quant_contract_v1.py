from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "fcooper_tvm_int8_quant_contract_v1.py"
TINY_SHAPE = (1, 1, 2, 2)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "fcooper_tvm_int8_quant_contract_v1", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_concat_model(path: Path) -> None:
    initializers = [
        numpy_helper.from_array(
            np.asarray([[[[1.0]]]], dtype=np.float32), "small.weight"
        ),
        numpy_helper.from_array(
            np.asarray([[[[4.0]]]], dtype=np.float32), "large.weight"
        ),
    ]
    nodes = [
        helper.make_node(
            "Conv", ["input", "small.weight"], ["small.conv"], name="small"
        ),
        helper.make_node("Relu", ["small.conv"], ["small.relu"], name="relu"),
        helper.make_node(
            "Conv", ["input", "large.weight"], ["large.conv"], name="large"
        ),
        helper.make_node(
            "Concat",
            ["small.relu", "large.conv"],
            ["output"],
            name="merge",
            axis=1,
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "concat_contract",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, list(TINY_SHAPE))],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2, 2, 2])],
        initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    onnx.save(model, path)


class FCooperTvmInt8QuantContractV1Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_module()
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.onnx_path = self.root / "prepared.onnx"
        self.calibration_dir = self.root / "calibration_validate"
        self.calibration_dir.mkdir()
        self.summary_path = self.calibration_dir / "calibration_summary.json"
        self.output_path = self.root / "quant_contract.json"
        save_concat_model(self.onnx_path)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def write_calibration(self, *, count: int = 16) -> dict:
        records = []
        for index in range(count):
            path = self.calibration_dir / f"sample_{index:05d}.npy"
            value = np.full(TINY_SHAPE, index + 1, dtype=np.float32)
            np.save(path, value)
            records.append(
                {
                    "sample_index": index,
                    "agents": 1,
                    "shape": list(TINY_SHAPE),
                    "path": str(path.resolve()),
                    "sha256": sha256_file(path),
                }
            )
        summary = {
            "schema_version": "fcooper_calibration_manifest_v1",
            "dataset": "OPV2V",
            "split": "validate",
            "sample_count": count,
            "engine_agent_batch": 1,
            "records": records,
        }
        self.summary_path.write_text(json.dumps(summary), encoding="utf-8")
        return summary

    def build(self, **overrides):
        options = {
            "onnx_path": self.onnx_path,
            "calibration_summary": self.summary_path,
            "calibration_dir": self.calibration_dir,
            "expected_shape": TINY_SHAPE,
            "max_outputs_per_run": 2,
            "sample_values_per_tensor_per_sample": 8,
        }
        options.update(overrides)
        return self.module.build_contract(**options)

    def test_module_import_does_not_require_onnx_runtime(self) -> None:
        self.assertNotIn("onnxruntime", self.module.__dict__)

    def test_observed_tensors_cover_required_ops_and_concat_members(self) -> None:
        model = onnx.shape_inference.infer_shapes(onnx.load(self.onnx_path))

        observed, concat_groups = self.module.observation_plan(model)

        self.assertEqual(
            observed,
            ["input", "small.conv", "small.relu", "large.conv", "output"],
        )
        self.assertEqual(
            concat_groups,
            [
                {
                    "node_name": "merge",
                    "members": ["small.relu", "large.conv", "output"],
                }
            ],
        )

    def test_output_chunks_are_strictly_bounded(self) -> None:
        chunks = self.module.bounded_chunks(["a", "b", "c", "d", "e"], 2)

        self.assertEqual(chunks, [("a", "b"), ("c", "d"), ("e",)])
        self.assertTrue(all(len(chunk) <= 2 for chunk in chunks))

    def test_subnormal_range_uses_float32_safe_scale_floor(self) -> None:
        param = self.module._quant_param(
            float(np.nextafter(np.float32(0), np.float32(1))),
            observed_absmax=1.0e-39,
            method="absmax",
        )

        self.assertGreaterEqual(param["scale"], float(np.finfo(np.float32).tiny))
        self.assertTrue(np.isfinite(np.float32(1.0) / np.float32(param["scale"])))
        self.assertTrue(param["range_floored"])

    def test_rejects_manifest_record_path_outside_calibration_directory(self) -> None:
        summary = self.write_calibration()
        other = self.root / "sample_00000.npy"
        other.write_bytes((self.calibration_dir / "sample_00000.npy").read_bytes())
        summary["records"][0]["path"] = str(other)
        summary["records"][0]["sha256"] = sha256_file(other)
        self.summary_path.write_text(json.dumps(summary), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "record path"):
            self.build()

    def test_accepts_summary_in_parent_of_verified_sample_directory(self) -> None:
        self.write_calibration()
        sample_dir = self.calibration_dir / "npy"
        sample_dir.mkdir()
        summary = json.loads(self.summary_path.read_text(encoding="utf-8"))
        for record in summary["records"]:
            source = Path(record["path"])
            target = sample_dir / source.name
            source.rename(target)
            record["path"] = str(target.resolve())
        self.summary_path.write_text(json.dumps(summary), encoding="utf-8")

        contract = self.build(calibration_dir=sample_dir)

        self.assertEqual(contract["status"], "success")
        self.assertEqual(contract["sample_count"], 16)

    def test_rejects_manifest_sha_and_npy_shape_or_dtype_mismatches(self) -> None:
        summary = self.write_calibration()
        summary["records"][0]["sha256"] = "0" * 64
        self.summary_path.write_text(json.dumps(summary), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.build()

        self.write_calibration()
        bad_path = self.calibration_dir / "sample_00000.npy"
        np.save(bad_path, np.ones((1, 1, 1, 1), dtype=np.float64))
        summary = json.loads(self.summary_path.read_text(encoding="utf-8"))
        summary["records"][0]["sha256"] = sha256_file(bad_path)
        summary["records"][0]["shape"] = [1, 1, 1, 1]
        self.summary_path.write_text(json.dumps(summary), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "expected shape"):
            self.build()

    def test_requires_exactly_16_samples(self) -> None:
        self.write_calibration(count=15)

        with self.assertRaisesRegex(ValueError, "exactly 16"):
            self.build()

    def test_absmax_contract_streams_all_samples_and_normalizes_concat(self) -> None:
        self.write_calibration()

        contract = self.build(calibration_method="absmax")

        self.assertEqual(contract["status"], "success")
        self.assertEqual(contract["sample_count"], 16)
        self.assertEqual(contract["calibration"]["method"], "absmax")
        self.assertEqual(contract["onnx"]["sha256"], sha256_file(self.onnx_path))
        self.assertEqual(
            contract["calibration"]["summary_sha256"], sha256_file(self.summary_path)
        )
        self.assertEqual(len(contract["calibration"]["samples"]), 16)
        self.assertEqual(
            contract["calibration"]["samples"][0],
            {
                "path": str(
                    (self.calibration_dir / "sample_00000.npy").resolve()
                ),
                "sha256": sha256_file(
                    self.calibration_dir / "sample_00000.npy"
                ),
                "shape": list(TINY_SHAPE),
            },
        )
        self.assertEqual(contract["coverage"]["missing"], [])
        self.assertEqual(contract["coverage"]["observed_count"], 5)
        group = contract["concat_scale_normalization"][0]
        self.assertEqual(group["members"], ["small.relu", "large.conv", "output"])
        self.assertEqual(group["common_absmax"], 64.0)
        for name in group["members"]:
            self.assertEqual(contract["params"][name]["absmax"], 64.0)
            self.assertEqual(contract["params"][name]["zero_point"], 128)
            self.assertAlmostEqual(contract["params"][name]["scale"], 64.0 / 127.0)
        self.assertEqual(contract["params"]["small.relu"]["observed_absmax"], 16.0)

    def test_percentile_method_records_sampling_parameters(self) -> None:
        self.write_calibration()

        contract = self.build(
            calibration_method="percentile_99_99",
            percentile=99.99,
            sample_values_per_tensor_per_sample=2,
        )

        calibration = contract["calibration"]
        self.assertEqual(calibration["method"], "percentile_99_99")
        self.assertEqual(calibration["parameters"]["percentile"], 99.99)
        self.assertEqual(
            calibration["parameters"]["sample_values_per_tensor_per_sample"], 2
        )
        self.assertEqual(
            contract["params"]["input"]["calibration_method"],
            "percentile_99_99",
        )

    def test_failure_report_is_persisted_with_provenance(self) -> None:
        self.write_calibration(count=15)

        with self.assertRaisesRegex(ValueError, "exactly 16"):
            self.module.write_contract(
                output_path=self.output_path,
                onnx_path=self.onnx_path,
                calibration_summary=self.summary_path,
                calibration_dir=self.calibration_dir,
                expected_shape=TINY_SHAPE,
            )

        report = json.loads(self.output_path.read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "failed")
        self.assertIn("exactly 16", report["failure"]["message"])
        self.assertEqual(report["calibration"]["method"], "absmax")
        self.assertEqual(report["calibration"]["expected_sample_count"], 16)
        self.assertEqual(report["calibration"]["expected_sample_shape"], list(TINY_SHAPE))
        self.assertEqual(report["onnx"]["sha256"], sha256_file(self.onnx_path))
        self.assertEqual(
            report["calibration"]["summary_sha256"], sha256_file(self.summary_path)
        )


if __name__ == "__main__":
    unittest.main()
