import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import stage2_native_int8_tvm_worker as worker  # noqa: E402


class FakeTvmArray:
    def __init__(self, value: np.ndarray):
        self.value = np.array(value, copy=True)

    def numpy(self) -> np.ndarray:
        return np.array(self.value, copy=True)

    def asnumpy(self) -> np.ndarray:
        return self.numpy()


class FakeDevice:
    def __init__(self) -> None:
        self.sync_calls = 0

    def sync(self) -> None:
        self.sync_calls += 1


class FakeNdModule:
    @staticmethod
    def array(value: np.ndarray, dev: object) -> FakeTvmArray:
        del dev
        return FakeTvmArray(value)


class FakeRuntimeModule:
    @staticmethod
    def load_module(path: str) -> object:
        return FakeRuntimeModule.loader(path)


class FakeRelaxVmResult:
    def __init__(self, *items: FakeTvmArray) -> None:
        self._items = list(items)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> FakeTvmArray:
        return self._items[idx]


class Stage2NativeInt8TvmWorkerTests(unittest.TestCase):
    def _write_request(
        self,
        tmp: Path,
        *,
        runtime_arg_plan: list[dict[str, object]],
        execution_abi: str | None = None,
    ) -> Path:
        artifact_path = tmp / "artifact.so"
        artifact_path.write_bytes(b"fake")
        activation_path = tmp / "activation.npy"
        np.save(activation_path, np.zeros((1, 2), dtype=np.uint8))
        weights_path = tmp / "weights.npz"
        np.savez(
            weights_path,
            weight_0=np.array([[3, 4]], dtype=np.int8),
            bias_0=np.array([7, 9], dtype=np.int32),
        )
        request = {
            "schema": "native_int8_tvm_worker_request_v1",
            "label": "sample0",
            "run_id": "sample0_run",
            "gpu": 0,
            "artifact_path": str(artifact_path),
            "runtime_weight_archive_path": str(weights_path),
            "activation_npy_path": str(activation_path),
            "output_dir": str(tmp / "outputs"),
            "runtime_arg_plan": runtime_arg_plan,
            "expected_output_shapes": {
                str(item["arg_name"]): list(item["shape"])
                for item in runtime_arg_plan
                if item["role"] == "graph_output"
            },
        }
        if execution_abi is not None:
            request["execution_abi"] = execution_abi
        request_path = tmp / "request.json"
        request_path.write_text(json.dumps(request), encoding="utf-8")
        return request_path

    def _install_fake_tvm(self, *, loader: object, vm_factory: object | None = None) -> dict[str, object]:
        fake_tvm = types.ModuleType("tvm")
        fake_tvm.cuda = lambda gpu: FakeDevice()
        fake_tvm.nd = FakeNdModule()
        fake_tvm.runtime = types.SimpleNamespace(load_module=loader)
        if vm_factory is not None:
            relax_module = types.ModuleType("tvm.relax")
            relax_module.VirtualMachine = vm_factory
            fake_tvm.relax = relax_module
            return {"tvm": fake_tvm, "tvm.relax": relax_module}
        return {"tvm": fake_tvm}

    def test_run_worker_supports_direct_packed_function_abi(self) -> None:
        runtime_arg_plan = [
            {"role": "graph_input", "arg_name": "graph_input", "shape": [1, 2], "dtype": "uint8"},
            {"role": "weight_input", "arg_name": "weight_0", "shape": [1, 2], "dtype": "int8"},
            {"role": "bias_input", "arg_name": "bias_0", "shape": [2], "dtype": "int32"},
            {"role": "graph_output", "arg_name": "graph_output", "shape": [1, 2], "dtype": "uint8"},
        ]
        call_args: list[list[FakeTvmArray]] = []

        class FakeDirectLib:
            def __getitem__(self, key: str) -> object:
                if key != "main":
                    raise KeyError(key)

                def main(*args: FakeTvmArray) -> None:
                    call_args.append(list(args))
                    args[3].value = np.array([[11, 12]], dtype=np.uint8)

                return main

        with tempfile.TemporaryDirectory() as tmpdir:
            request_path = self._write_request(Path(tmpdir), runtime_arg_plan=runtime_arg_plan)
            fake_modules = self._install_fake_tvm(loader=lambda path: FakeDirectLib())
            with mock.patch.dict(sys.modules, fake_modules, clear=False):
                worker._load_context.__defaults__[0].clear()
                response = worker.run_worker(request_path)
                saved_output = np.load(response["outputs"][0]["path"])

        self.assertEqual(response["status"], "success")
        self.assertEqual(len(call_args), 1)
        self.assertEqual(len(call_args[0]), 4)
        np.testing.assert_array_equal(call_args[0][0].numpy(), np.zeros((1, 2), dtype=np.uint8))
        np.testing.assert_array_equal(call_args[0][1].numpy(), np.array([[3, 4]], dtype=np.int8))
        np.testing.assert_array_equal(call_args[0][2].numpy(), np.array([7, 9], dtype=np.int32))
        self.assertEqual(response["outputs"][0]["arg_name"], "graph_output")
        self.assertEqual(response["outputs"][0]["shape"], [1, 2])
        self.assertEqual(response["outputs"][0]["dtype"], "uint8")
        np.testing.assert_array_equal(saved_output, np.array([[11, 12]], dtype=np.uint8))

    def test_run_worker_supports_relax_vm_return_abi(self) -> None:
        runtime_arg_plan = [
            {"role": "graph_input", "arg_name": "graph_input", "shape": [1, 2], "dtype": "uint8"},
            {"role": "weight_input", "arg_name": "weight_0", "shape": [1, 2], "dtype": "int8"},
            {"role": "bias_input", "arg_name": "bias_0", "shape": [2], "dtype": "int32"},
            {"role": "graph_output", "arg_name": "graph_output", "shape": [1, 2], "dtype": "uint8"},
        ]
        vm_calls: list[list[FakeTvmArray]] = []

        class FakeVm:
            def __init__(self, lib: object, dev: object) -> None:
                self.lib = lib
                self.dev = dev

            def __getitem__(self, key: str) -> object:
                if key != "main":
                    raise KeyError(key)

                def main(*args: FakeTvmArray) -> FakeRelaxVmResult:
                    vm_calls.append(list(args))
                    return FakeRelaxVmResult(FakeTvmArray(np.array([[21, 22]], dtype=np.uint8)))

                return main

        class FakeVmExecutable:
            pass

        with tempfile.TemporaryDirectory() as tmpdir:
            request_path = self._write_request(
                Path(tmpdir),
                runtime_arg_plan=runtime_arg_plan,
                execution_abi="relax_vm_return",
            )
            fake_modules = self._install_fake_tvm(loader=lambda path: FakeVmExecutable(), vm_factory=FakeVm)
            with mock.patch.dict(sys.modules, fake_modules, clear=False):
                worker._load_context.__defaults__[0].clear()
                response = worker.run_worker(request_path)
                saved_output = np.load(response["outputs"][0]["path"])

        self.assertEqual(response["status"], "success")
        self.assertEqual(len(vm_calls), 1)
        self.assertEqual(len(vm_calls[0]), 3)
        np.testing.assert_array_equal(vm_calls[0][0].numpy(), np.zeros((1, 2), dtype=np.uint8))
        np.testing.assert_array_equal(vm_calls[0][1].numpy(), np.array([[3, 4]], dtype=np.int8))
        np.testing.assert_array_equal(vm_calls[0][2].numpy(), np.array([7, 9], dtype=np.int32))
        np.testing.assert_array_equal(saved_output, np.array([[21, 22]], dtype=np.uint8))

    def test_run_worker_rejects_relax_vm_output_contract_mismatch(self) -> None:
        runtime_arg_plan = [
            {"role": "graph_input", "arg_name": "graph_input", "shape": [1, 2], "dtype": "uint8"},
            {"role": "weight_input", "arg_name": "weight_0", "shape": [1, 2], "dtype": "int8"},
            {"role": "graph_output", "arg_name": "graph_output", "shape": [1, 2], "dtype": "uint8"},
        ]

        class FakeVm:
            def __init__(self, lib: object, dev: object) -> None:
                self.lib = lib
                self.dev = dev

            def __getitem__(self, key: str) -> object:
                if key != "main":
                    raise KeyError(key)

                def main(*args: FakeTvmArray) -> tuple[FakeTvmArray]:
                    del args
                    return (FakeTvmArray(np.array([[1, 2]], dtype=np.int16)),)

                return main

        class FakeVmExecutable:
            pass

        with tempfile.TemporaryDirectory() as tmpdir:
            request_path = self._write_request(
                Path(tmpdir),
                runtime_arg_plan=runtime_arg_plan,
                execution_abi="relax_vm_return",
            )
            fake_modules = self._install_fake_tvm(loader=lambda path: FakeVmExecutable(), vm_factory=FakeVm)
            with mock.patch.dict(sys.modules, fake_modules, clear=False):
                worker._load_context.__defaults__[0].clear()
                with self.assertRaisesRegex(ValueError, "array dtype mismatch"):
                    worker.run_worker(request_path)


if __name__ == "__main__":
    unittest.main()
