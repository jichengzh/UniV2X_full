from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts import fcooper_tvm_relax_worker_v1 as worker


class FCooperTvmRelaxWorkerTests(unittest.TestCase):
    def test_quant_params_reject_dynamic_or_fallback_contracts(self) -> None:
        for payload in (
            {"params": {"x": {"scale": 0.25, "zero_point": 3, "dynamic": True}}},
            {
                "params": {
                    "x": {
                        "scale": 0.25,
                        "zero_point": 3,
                        "source": "dynamic_per_chunk",
                    }
                }
            },
        ):
            with self.assertRaisesRegex(ValueError, "dynamic quantization"):
                worker.quant_params(payload, "x")

    def test_runtime_tensor_contract_rejects_unknown_role_or_dynamic_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "runtime role"):
            worker.validate_runtime_tensor(
                {"arg_name": "x", "role": "mystery", "shape": [5, 2], "dtype": "uint8"}
            )
        with self.assertRaisesRegex(ValueError, "positive static"):
            worker.validate_runtime_tensor(
                {"arg_name": "x", "role": "graph_input", "shape": [5, -1], "dtype": "uint8"}
            )

    def test_vmexec_elf_uses_explicit_shared_library_loader(self) -> None:
        calls = []
        tvm = SimpleNamespace(
            get_global_func=lambda name: (
                lambda path, fmt: calls.append((name, path, fmt)) or "module"
            ),
            runtime=SimpleNamespace(load_module=lambda path: "default"),
        )

        loaded = worker.load_compiled_module(tvm, Path("/tmp/model.vmexec"))

        self.assertEqual(loaded, "module")
        self.assertEqual(
            calls,
            [("ffi.Module.load_from_file.so", "/tmp/model.vmexec", "so")],
        )

    def test_runtime_weight_binding_prefers_arg_name_before_initializer(self) -> None:
        item = {
            "arg_name": "weight_1",
            "initializer_name": "onnx::Conv_1",
        }

        key = worker.runtime_weight_archive_key(
            item,
            declared={},
            archive_keys={"weight_1"},
        )

        self.assertEqual(key, "weight_1")


if __name__ == "__main__":
    unittest.main()
