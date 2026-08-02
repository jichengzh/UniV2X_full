from __future__ import annotations

import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

bridge = importlib.import_module("scripts.fcooper_tvm_int8_ap_bridge_v1")


def _runtime_plan() -> list[dict[str, object]]:
    return [
        {
            "arg_name": "spatial_features",
            "role": "graph_input",
            "shape": [5, 2, 2, 3],
            "dtype": "uint8",
        },
        {
            "arg_name": "conv_weight",
            "initializer_name": "onnx::Conv_1",
            "role": "weight_input",
            "shape": [3, 2, 1, 1],
            "dtype": "int8",
        },
        {
            "arg_name": "conv_bias",
            "initializer_name": "onnx::Conv_2",
            "role": "bias_input",
            "shape": [3],
            "dtype": "int32",
        },
        {
            "arg_name": "spatial_features_2d",
            "role": "graph_output",
            "shape": [5, 3, 2, 3],
            "dtype": "uint8",
        },
    ]


def _quant_payload() -> dict[str, object]:
    return {
        "params": {
            "spatial_features": {"scale": 0.25, "zero_point": 120, "source": "static_calibration"},
            "spatial_features_2d": {
                "scale": 0.5,
                "zero_point": 100,
                "source": "static_calibration",
            },
        }
    }


class _FakeDevice:
    def __init__(self) -> None:
        self.sync_count = 0

    def sync(self) -> None:
        self.sync_count += 1


class _FakeRunner:
    input_shape = (5, 2, 2, 3)
    output_shape = (5, 3, 2, 3)

    def __init__(self) -> None:
        self.call_count = 0
        self.inputs: list[torch.Tensor] = []

    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        self.call_count += 1
        self.inputs.append(value.clone())
        return torch.full(self.output_shape, 2.5, dtype=torch.float32, device=value.device)


class FCooperTvmInt8ApBridgeTests(unittest.TestCase):
    def test_import_does_not_require_tvm(self) -> None:
        self.assertNotIn("tvm", bridge.__dict__)

    def test_quantize_clips_rounds_and_dequantizes_static_uint8(self) -> None:
        source = np.asarray([-40.0, -0.24, 0.0, 0.26, 40.0], dtype=np.float32)

        quantized = bridge.quantize_uint8(source, scale=0.25, zero_point=120)
        restored = bridge.dequantize_uint8(quantized, scale=0.25, zero_point=120)

        np.testing.assert_array_equal(quantized, np.asarray([0, 119, 120, 121, 255], dtype=np.uint8))
        np.testing.assert_allclose(
            restored,
            np.asarray([-30.0, -0.25, 0.0, 0.25, 33.75], dtype=np.float32),
        )
        self.assertEqual(restored.dtype, np.float32)

    def test_runtime_contract_requires_one_input_one_output_and_batch_five(self) -> None:
        contract = bridge.RuntimeContract.from_payloads(_runtime_plan(), _quant_payload())

        self.assertEqual(contract.input_shape, (5, 2, 2, 3))
        self.assertEqual(contract.output_shape, (5, 3, 2, 3))
        self.assertEqual(contract.input_quant.scale, 0.25)
        self.assertEqual(contract.output_quant.zero_point, 100)

        two_outputs = [*_runtime_plan(), dict(_runtime_plan()[-1], arg_name="extra")]
        with self.assertRaisesRegex(ValueError, "exactly one graph input and one graph output"):
            bridge.RuntimeContract.from_payloads(two_outputs, _quant_payload())
        bad_batch = [dict(item) for item in _runtime_plan()]
        bad_batch[0]["shape"] = [4, 2, 2, 3]
        bad_batch[-1]["shape"] = [4, 3, 2, 3]
        with self.assertRaisesRegex(ValueError, "artifact batch must be 5"):
            bridge.RuntimeContract.from_payloads(bad_batch, _quant_payload())

    def test_runtime_contract_rejects_dynamic_or_missing_boundary_quantization(self) -> None:
        dynamic = _quant_payload()
        dynamic["params"]["spatial_features"]["dynamic"] = True
        with self.assertRaisesRegex(ValueError, "dynamic quantization is forbidden"):
            bridge.RuntimeContract.from_payloads(_runtime_plan(), dynamic)

        missing = _quant_payload()
        del missing["params"]["spatial_features_2d"]
        with self.assertRaisesRegex(ValueError, "missing static quant params"):
            bridge.RuntimeContract.from_payloads(_runtime_plan(), missing)

    def test_weight_bindings_validate_mapping_shape_dtype_and_sha(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive_path = root / "weights.npz"
            np.savez(
                archive_path,
                w=np.zeros((3, 2, 1, 1), dtype=np.int8),
                b=np.zeros((3,), dtype=np.int32),
            )
            digest = bridge.sha256_path(archive_path)
            result = {
                "runtime_arg_plan": _runtime_plan(),
                "runtime_weight_archive_keys": {"conv_weight": "w", "conv_bias": "b"},
                "runtime_weight_archive_sha256": digest,
            }

            arrays, bindings = bridge.load_runtime_weights(result, archive_path)

            self.assertEqual([array.dtype for array in arrays], [np.dtype("int8"), np.dtype("int32")])
            self.assertEqual(bindings, {"conv_weight": "w", "conv_bias": "b"})
            result["runtime_weight_archive_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "runtime weight SHA256 mismatch"):
                bridge.load_runtime_weights(result, archive_path)

    def test_weight_bindings_fail_closed_without_declared_archive_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            archive_path = Path(tmp) / "weights.npz"
            np.savez(archive_path, unrelated=np.zeros((1,), dtype=np.int8))
            result = {"runtime_arg_plan": _runtime_plan(), "runtime_weight_archive_keys": {}}

            with self.assertRaisesRegex(ValueError, "missing archive binding"):
                bridge.load_runtime_weights(result, archive_path)

    def test_runner_quantizes_once_binds_weights_and_dequantizes_one_output(self) -> None:
        contract = bridge.RuntimeContract.from_payloads(_runtime_plan(), _quant_payload())
        runner = bridge.TvmRelaxVmInt8Runner.__new__(bridge.TvmRelaxVmInt8Runner)
        runner.contract = contract
        runner.input_shape = contract.input_shape
        runner.output_shape = contract.output_shape
        runner.call_count = 0
        runner._device = _FakeDevice()
        runner._make_tvm_array = lambda value: value
        runner._runtime_weights = (
            np.zeros((3, 2, 1, 1), dtype=np.int8),
            np.zeros((3,), dtype=np.int32),
        )
        calls: list[tuple[np.ndarray, ...]] = []

        def fake_main(*args: np.ndarray) -> np.ndarray:
            calls.append(args)
            return np.full(contract.output_shape, 102, dtype=np.uint8)

        runner._main = fake_main
        output = runner._execute_host_input(np.zeros(contract.input_shape, dtype=np.float32))

        self.assertEqual(runner.call_count, 1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(calls[0]), 3)
        self.assertEqual(calls[0][0].dtype, np.uint8)
        self.assertEqual(runner._device.sync_count, 1)
        self.assertEqual(output.dtype, np.float32)
        np.testing.assert_array_equal(output, np.ones(contract.output_shape, dtype=np.float32))

    def test_dense_body_zero_pads_to_five_unpads_and_returns_float32(self) -> None:
        runner = _FakeRunner()
        module = bridge.TvmInt8DenseBody(runner)
        source = torch.full((2, 2, 2, 3), 7.0)

        result = module({"spatial_features": source})

        self.assertEqual(runner.call_count, 1)
        self.assertEqual(tuple(runner.inputs[0].shape), runner.input_shape)
        torch.testing.assert_close(runner.inputs[0][:2], source)
        torch.testing.assert_close(runner.inputs[0][2:], torch.zeros_like(runner.inputs[0][2:]))
        self.assertEqual(tuple(result["spatial_features_2d"].shape), (2, 3, 2, 3))
        self.assertEqual(result["spatial_features_2d"].dtype, torch.float32)

    def test_shape_failures_happen_before_vm_call(self) -> None:
        runner = _FakeRunner()
        module = bridge.TvmInt8DenseBody(runner)
        with self.assertRaisesRegex(ValueError, "input shape drift"):
            module({"spatial_features": torch.zeros((1, 9, 2, 3))})
        with self.assertRaisesRegex(ValueError, "artifact supports 5"):
            module({"spatial_features": torch.zeros((6, 2, 2, 3))})
        self.assertEqual(runner.call_count, 0)

    def test_fallback_and_call_gates_suppress_ap(self) -> None:
        passing = bridge.evaluate_execution_gates(
            mode="full",
            requested_samples=2170,
            processed_samples=2170,
            failed_samples=0,
            vm_calls=2170,
            fallback_samples=0,
            numerical_sanity_passed=True,
        )
        blocked = bridge.evaluate_execution_gates(
            mode="full",
            requested_samples=2170,
            processed_samples=2170,
            failed_samples=0,
            vm_calls=2169,
            fallback_samples=1,
            numerical_sanity_passed=True,
        )

        self.assertTrue(passing["publish_ap"])
        self.assertFalse(blocked["publish_ap"])
        self.assertIn("fallback_forbidden", blocked["blockers"])
        self.assertIn("vm_calls_mismatch", blocked["blockers"])
        with mock.patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "7"}):
            report = bridge.build_report(gates=blocked, ap=(0.7, 0.6, 0.5))
        self.assertNotIn("ap30", report)
        self.assertNotIn("ap", report)
        self.assertEqual(report["execution_device"]["physical_gpu_id"], 7)
        self.assertEqual(report["execution_device"]["cuda_visible_devices"], "7")

    def test_full_gate_requires_bound_passing_sanity_report(self) -> None:
        bindings = {"artifact_sha256": "a", "result_sha256": "b"}
        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "sanity.json"
            report_path.write_text(
                json.dumps(
                    {
                        "gates": {"sanity_16": True},
                        "processed_samples": 16,
                        "vm_calls": 16,
                        **bindings,
                    }
                )
            )
            digest = bridge.sha256_path(report_path)

            bridge.require_bound_sanity_report(report_path, digest, bindings)
            with self.assertRaisesRegex(ValueError, "sanity binding mismatch"):
                bridge.require_bound_sanity_report(report_path, digest, {"artifact_sha256": "wrong"})


if __name__ == "__main__":
    unittest.main()
