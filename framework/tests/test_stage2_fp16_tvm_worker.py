import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import stage2_fp16_tvm_worker as worker  # noqa: E402


class Stage2Fp16TvmWorkerTests(unittest.TestCase):
    def test_output_filename_sanitizes_arg_name(self):
        self.assertEqual(worker.output_filename_for_arg("out:0/feature"), "out_out_0_feature.npy")
        self.assertEqual(worker.output_filename_for_arg("123"), "out_tensor_123.npy")

    def test_worker_request_validation_requires_artifact_and_activation(self):
        with tempfile.TemporaryDirectory() as tmp:
            request_path = Path(tmp) / "request.json"
            request_path.write_text(json.dumps({"output_dir": tmp}), encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                worker.validate_request(worker._read_json(request_path))
            self.assertIn("artifact_path", str(ctx.exception))

    def test_build_output_record_has_shape_and_dtype(self):
        import numpy as np

        array = np.zeros((2, 3), dtype="float16")
        record = worker.build_output_record(
            arg_name="feature0",
            output_path=Path("/tmp/feature0.npy"),
            array=array,
        )
        self.assertEqual(record["arg_name"], "feature0")
        self.assertEqual(record["shape"], [2, 3])
        self.assertEqual(record["dtype"], "float16")

    def test_validate_request_rejects_unsupported_activation_dtype(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact_path = root / "artifact.so"
            artifact_path.write_bytes(b"stub")
            activation_path = root / "activation.npy"
            activation_path.write_bytes(b"stub")
            request = {
                "artifact_path": str(artifact_path),
                "activation_npy_path": str(activation_path),
                "output_dir": str(root / "out"),
                "activation_dtype": "bfloat16",
            }
            with self.assertRaises(ValueError) as ctx:
                worker.validate_request(request)
        self.assertIn("activation_dtype", str(ctx.exception))

    def test_execute_uses_requested_float32_activation_dtype(self):
        import numpy as np

        class FakeDevice:
            def sync(self) -> None:
                return None

        class FakeTensor:
            def __init__(self, array: np.ndarray):
                self._array = array

            def numpy(self) -> np.ndarray:
                return self._array

        class FakeVm:
            def __init__(self) -> None:
                self.last_input = None

            def __getitem__(self, name: str):
                self_name = self

                def _main(arg):
                    self_name.last_input = arg
                    return [FakeTensor(np.zeros((2, 24, 128, 256), dtype=np.float32))]

                return _main

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact_path = root / "artifact.so"
            artifact_path.write_bytes(b"stub")
            activation_path = root / "activation.npy"
            np.save(activation_path, np.ones((2, 64, 128, 256), dtype=np.float16))
            request_json = root / "request.json"
            request = {
                "artifact_path": str(artifact_path),
                "activation_npy_path": str(activation_path),
                "output_dir": str(root / "out"),
                "activation_dtype": "float32",
                "output_names": ["output0"],
                "expected_output_shapes": {"output0": [2, 24, 128, 256]},
            }
            request_json.write_text(json.dumps(request), encoding="utf-8")
            fake_vm = FakeVm()
            ctx = {
                "tvm": object(),
                "dev": FakeDevice(),
                "vm": fake_vm,
            }

            with mock.patch.object(worker, "make_tvm_array", side_effect=lambda _tvm, value, _dev: value):
                response = worker._execute(ctx, request, request_json)

        self.assertEqual(str(fake_vm.last_input.dtype), "float32")
        self.assertEqual(response["activation_dtype"], "float32")

    def test_execute_defaults_activation_dtype_to_float16_for_compatibility(self):
        import numpy as np

        class FakeDevice:
            def sync(self) -> None:
                return None

        class FakeTensor:
            def __init__(self, array: np.ndarray):
                self._array = array

            def numpy(self) -> np.ndarray:
                return self._array

        class FakeVm:
            def __init__(self) -> None:
                self.last_input = None

            def __getitem__(self, name: str):
                self_name = self

                def _main(arg):
                    self_name.last_input = arg
                    return [FakeTensor(np.zeros((2, 24, 128, 256), dtype=np.float16))]

                return _main

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact_path = root / "artifact.so"
            artifact_path.write_bytes(b"stub")
            activation_path = root / "activation.npy"
            np.save(activation_path, np.ones((2, 64, 128, 256), dtype=np.float32))
            request_json = root / "request.json"
            request = {
                "artifact_path": str(artifact_path),
                "activation_npy_path": str(activation_path),
                "output_dir": str(root / "out"),
                "output_names": ["output0"],
                "expected_output_shapes": {"output0": [2, 24, 128, 256]},
            }
            request_json.write_text(json.dumps(request), encoding="utf-8")
            fake_vm = FakeVm()
            ctx = {
                "tvm": object(),
                "dev": FakeDevice(),
                "vm": fake_vm,
            }

            with mock.patch.object(worker, "make_tvm_array", side_effect=lambda _tvm, value, _dev: value):
                response = worker._execute(ctx, request, request_json)

        self.assertEqual(str(fake_vm.last_input.dtype), "float16")
        self.assertEqual(response["activation_dtype"], "float16")


if __name__ == "__main__":
    unittest.main()
