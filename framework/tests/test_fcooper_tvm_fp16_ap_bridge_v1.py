from __future__ import annotations

import importlib
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

bridge = importlib.import_module("scripts.fcooper_tvm_fp16_ap_bridge_v1")


class _FakeRunner:
    def __init__(
        self,
        input_shape: tuple[int, ...] = (2, 4, 3, 5),
        output_shape: tuple[int, ...] = (2, 6, 3, 5),
    ) -> None:
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.call_count = 0
        self.inputs: list[torch.Tensor] = []

    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        self.inputs.append(value.clone())
        self.call_count += 1
        return torch.ones(self.output_shape, device=value.device, dtype=torch.float16)


class _FakeDevice:
    def __init__(self) -> None:
        self.sync_count = 0

    def sync(self) -> None:
        self.sync_count += 1


class FCooperTvmFp16ApBridgeTests(unittest.TestCase):
    def test_single_file_sha_is_standard_content_digest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.so"
            path.write_bytes(b"compiled")

            actual = bridge.sha256_path(path)

        self.assertEqual(actual, hashlib.sha256(b"compiled").hexdigest())

    def test_module_import_does_not_require_tvm(self) -> None:
        self.assertNotIn("tvm", bridge.__dict__)

    def test_static_contract_accepts_exactly_one_positive_input_and_output(self) -> None:
        contract = bridge.StaticTensorContract.from_shapes(
            input_shapes=[(2, 64, 128, 256)],
            output_shapes=[(2, 128, 128, 256)],
        )

        self.assertEqual(contract.input_shape, (2, 64, 128, 256))
        self.assertEqual(contract.output_shape, (2, 128, 128, 256))

    def test_static_contract_rejects_multiple_or_dynamic_tensors(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly one input and one output"):
            bridge.StaticTensorContract.from_shapes(
                input_shapes=[(2, 4, 3, 5), (2, 1)],
                output_shapes=[(2, 6, 3, 5)],
            )
        with self.assertRaisesRegex(ValueError, "positive static dimensions"):
            bridge.StaticTensorContract.from_shapes(
                input_shapes=[(2, -1, 3, 5)],
                output_shapes=[(2, 6, 3, 5)],
            )

    def test_dense_body_pads_agent_batch_and_unpads_float_output(self) -> None:
        runner = _FakeRunner()
        module = bridge.TvmDenseBody(runner)
        source = torch.full((1, 4, 3, 5), 7.0)

        result = module({"spatial_features": source})

        self.assertEqual(runner.call_count, 1)
        self.assertEqual(tuple(runner.inputs[0].shape), runner.input_shape)
        torch.testing.assert_close(runner.inputs[0][0], source[0])
        torch.testing.assert_close(runner.inputs[0][1], torch.zeros_like(source[0]))
        self.assertEqual(tuple(result["spatial_features_2d"].shape), (1, 6, 3, 5))
        self.assertEqual(result["spatial_features_2d"].dtype, torch.float32)

    def test_dense_body_rejects_shape_drift_before_backend_call(self) -> None:
        runner = _FakeRunner()
        module = bridge.TvmDenseBody(runner)

        with self.assertRaisesRegex(ValueError, "dense-body input shape drift"):
            module({"spatial_features": torch.zeros((1, 5, 3, 5))})
        with self.assertRaisesRegex(ValueError, "artifact supports 2"):
            module({"spatial_features": torch.zeros((3, 4, 3, 5))})

        self.assertEqual(runner.call_count, 0)

    def test_runner_counts_each_fake_vm_dispatch_without_tvm_installed(self) -> None:
        runner = bridge.TvmRelaxVmRunner.__new__(bridge.TvmRelaxVmRunner)
        runner.input_shape = (2, 4, 3, 5)
        runner.output_shape = (2, 6, 3, 5)
        runner.call_count = 0
        runner._device = _FakeDevice()
        runner._make_tvm_array = lambda value: value
        runner._main = lambda value: np.ones(runner.output_shape, dtype=np.float16)
        host_input = np.zeros(runner.input_shape, dtype=np.float32)

        output = runner._execute_host_input(host_input)

        self.assertEqual(runner.call_count, 1)
        self.assertEqual(runner._device.sync_count, 1)
        self.assertEqual(output.dtype, np.float16)
        self.assertEqual(output.shape, runner.output_shape)

    def test_execution_gates_require_one_backend_call_per_processed_sample(self) -> None:
        sanity = bridge.evaluate_execution_gates(
            dataset_samples=100,
            requested_samples=16,
            processed_samples=16,
            failed_samples=0,
            backend_calls=16,
            fallback_samples=0,
        )
        full = bridge.evaluate_execution_gates(
            dataset_samples=100,
            requested_samples=100,
            processed_samples=100,
            failed_samples=0,
            backend_calls=100,
            fallback_samples=0,
        )

        self.assertEqual(sanity["status"], "success_sanity")
        self.assertTrue(sanity["sanity"])
        self.assertFalse(sanity["full"])
        self.assertEqual(full["status"], "success_full")
        self.assertTrue(full["full"])

    def test_execution_gates_forbid_fallback_and_call_mismatch(self) -> None:
        fallback = bridge.evaluate_execution_gates(
            dataset_samples=16,
            requested_samples=16,
            processed_samples=16,
            failed_samples=0,
            backend_calls=16,
            fallback_samples=1,
        )
        mismatch = bridge.evaluate_execution_gates(
            dataset_samples=16,
            requested_samples=16,
            processed_samples=16,
            failed_samples=0,
            backend_calls=15,
            fallback_samples=0,
        )

        self.assertEqual(fallback["status"], "failure")
        self.assertIn("fallback_forbidden", fallback["blockers"])
        self.assertEqual(mismatch["status"], "failure")
        self.assertIn("backend_calls_mismatch", mismatch["blockers"])

    def test_report_records_metrics_provenance_and_explicit_numerics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = {}
            for name in ("artifact", "checkpoint", "config", "prediction"):
                path = root / name
                path.write_bytes(name.encode("ascii"))
                paths[name] = path

            with mock.patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "7"}):
                report = bridge.build_report(
                    dataset_samples=16,
                    requested_samples=16,
                    processed_samples=16,
                    failed_samples=0,
                    backend_calls=16,
                    fallback_samples=0,
                    ap30=0.7,
                    ap50=0.6,
                    ap70=0.5,
                    elapsed_seconds=1.25,
                    artifact_path=paths["artifact"],
                    checkpoint_path=paths["checkpoint"],
                    config_path=paths["config"],
                    prediction_path=paths["prediction"],
                    input_shape=(2, 64, 128, 256),
                    output_shape=(2, 128, 128, 256),
                )

        self.assertEqual(report["status"], "success_full")
        self.assertEqual(report["backend_calls"], 16)
        self.assertEqual(report["ap30"], 0.7)
        self.assertEqual(report["ap50"], 0.6)
        self.assertEqual(report["ap70"], 0.5)
        self.assertEqual(set(report["sha256"]), {"artifact", "checkpoint", "config", "prediction"})
        for name in ("artifact", "checkpoint", "config", "prediction"):
            self.assertEqual(report[f"{name}_sha256"], report["sha256"][name])
        self.assertEqual(report["artifact_path"], str(paths["artifact"]))
        self.assertEqual(report["numerical_contract"]["artifact_input_dtype"], "float32")
        self.assertEqual(report["numerical_contract"]["artifact_compute_dtype"], "float16")
        self.assertEqual(report["numerical_contract"]["model_boundary_output_dtype"], "float32")
        self.assertEqual(report["numerical_contract"]["input_shape"], [2, 64, 128, 256])
        self.assertTrue(report["numerical_contract"]["silent_fallback_forbidden"])
        self.assertEqual(report["execution_device"]["physical_gpu_id"], 7)
        self.assertEqual(report["execution_device"]["cuda_visible_devices"], "7")

    def test_checkpoint_provenance_matches_implicit_loader_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp)
            selected = checkpoint_dir / "net_epoch_bestval_at17.pth"
            selected.write_bytes(b"best")
            (checkpoint_dir / "net_epoch99.pth").write_bytes(b"latest")
            args = SimpleNamespace(checkpoint=None, checkpoint_dir=checkpoint_dir)

            resolved = bridge._checkpoint_for_hash(args)

        self.assertEqual(resolved, selected)

    def test_fp32_artifact_dtype_is_explicit(self) -> None:
        contract = bridge.StaticTensorContract.from_shapes(
            input_shapes=[(5, 64, 512, 512)],
            output_shapes=[(5, 256, 256, 256)],
        )
        runner = object.__new__(bridge.TvmRelaxVmRunner)
        runner.input_shape = contract.input_shape
        runner.output_shape = contract.output_shape
        runner.output_dtype = "float32"
        runner.call_count = 0
        runner._make_tvm_array = lambda value: value
        runner._main = lambda value: np.zeros(contract.output_shape, dtype=np.float32)
        runner._device = _FakeDevice()
        source = np.zeros(contract.input_shape, dtype=np.float32)

        output = runner._execute_host_input(source)

        self.assertEqual(output.dtype, np.float32)

    def test_fp32_report_uses_fp32_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "model.so"
            config = root / "config.yaml"
            artifact.write_bytes(b"module")
            config.write_bytes(b"config")

            report = bridge.build_report(
                dataset_samples=1,
                requested_samples=1,
                processed_samples=1,
                failed_samples=0,
                backend_calls=1,
                fallback_samples=0,
                ap30=0.7,
                ap50=0.6,
                ap70=0.5,
                elapsed_seconds=1.0,
                artifact_path=artifact,
                checkpoint_path=None,
                config_path=config,
                prediction_path=None,
                input_shape=(5, 64, 512, 512),
                output_shape=(5, 256, 256, 256),
                artifact_compute_dtype="float32",
                artifact_output_dtype="float32",
            )

        self.assertEqual(
            report["schema_version"], "fcooper_tvm_fp32_ap_report_v1"
        )


if __name__ == "__main__":
    unittest.main()
