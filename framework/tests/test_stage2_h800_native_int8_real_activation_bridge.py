import argparse
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import stage2_h800_native_int8_real_activation_bridge as bridge_mod  # noqa: E402


class Stage2H800NativeInt8RealActivationBridgeTests(unittest.TestCase):
    def test_exact_checkpoint_loader_uses_selected_file_not_config_resume_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "net_epoch31.pth"
            source = torch.nn.Linear(2, 1)
            with torch.no_grad():
                source.weight.fill_(3.0)
                source.bias.fill_(4.0)
            torch.save(source.state_dict(), checkpoint)
            target = torch.nn.Linear(2, 1)

            epoch = bridge_mod.load_exact_checkpoint(
                target, checkpoint, torch_module=torch, train_utils_module=None
            )

        self.assertEqual(epoch, 31)
        self.assertTrue(torch.equal(target.weight, source.weight))
        self.assertTrue(torch.equal(target.bias, source.bias))

    def _make_args(self, root: Path) -> argparse.Namespace:
        return argparse.Namespace(
            label="s0_024",
            artifact_path=str(root / "artifact.vmexec"),
            inventory_path=str(root / "inventory.json"),
            runtime_weight_archive_path=str(root / "weights.npz"),
            keep_detailed_samples=1,
            numeric_sanity_only=False,
            persistent_worker=False,
            tensor_quant_params_path=None,
            tvm_python="/tmp/fake-python",
            worker_script=str(REPO_ROOT / "scripts/stage2_native_int8_tvm_worker.py"),
            gpu_id=0,
            tvm_ld_library_path="/tmp/fake-ld",
        )

    def _make_inventory(self, *, engine_batch: int = 2) -> dict[str, object]:
        return {
            "runtime_arg_plan": [
                {
                    "role": "graph_input",
                    "arg_name": "graph_input",
                    "shape": [engine_batch, 1, 1, 1],
                    "dtype": "uint8",
                },
                {
                    "role": "graph_output",
                    "arg_name": "graph_output",
                    "shape": [engine_batch, 1, 1, 1],
                    "dtype": "uint8",
                },
            ],
            "execution_abi": "relax_vm_return",
        }

    def _install_fake_dispatch(self, bridge: bridge_mod.NativeInt8BackboneBridge) -> dict[str, list[object]]:
        captured: dict[str, list[object]] = {"requests": [], "activations": []}

        def fake_dispatch(request_path: Path, chunk_dir: Path) -> dict[str, object]:
            request = bridge_mod._load_json(request_path)
            activation = np.load(request["activation_npy_path"])
            captured["requests"].append(request)
            captured["activations"].append(np.array(activation, copy=True))
            chunk_idx = len(captured["requests"]) - 1
            values = np.array([10 * (chunk_idx + 1) + 1, 10 * (chunk_idx + 1) + 2], dtype=np.uint8)
            output = values.reshape(2, 1, 1, 1)
            output_path = chunk_dir / "out_graph_output.npy"
            np.save(output_path, output)
            response = {
                "schema": "native_int8_tvm_worker_response_v1",
                "status": "success",
                "outputs": [
                    {
                        "arg_name": "graph_output",
                        "path": str(output_path),
                        "shape": [2, 1, 1, 1],
                        "dtype": "uint8",
                    }
                ],
            }
            return response

        bridge._dispatch_worker = fake_dispatch  # type: ignore[method-assign]
        return captured

    def test_bridge_pads_single_agent_to_engine_batch_and_records_padding(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bridge = bridge_mod.NativeInt8BackboneBridge(
                args=self._make_args(root),
                raw_dir=root / "raw",
                inventory=self._make_inventory(engine_batch=2),
            )
            bridge.tensor_quant_params = {"graph_output": {"scale": 1.0, "zero_point": 0, "source": "test"}}
            captured = self._install_fake_dispatch(bridge)

            outputs = bridge(torch.ones((1, 1, 1, 1), dtype=torch.float32))

        self.assertEqual(len(captured["requests"]), 1)
        self.assertEqual(captured["requests"][0]["execution_abi"], "relax_vm_return")
        self.assertEqual(list(captured["activations"][0].shape), [2, 1, 1, 1])
        self.assertEqual(int(captured["activations"][0][1, 0, 0, 0]), 0)
        self.assertEqual([list(item.shape) for item in outputs], [[1, 1, 1, 1]])
        self.assertEqual(float(outputs[0][0, 0, 0, 0]), 11.0)
        self.assertEqual(bridge.worker_responses[0]["padding_agents"], 1)
        self.assertEqual(bridge.worker_responses[0]["original_agents"], 1)

    def test_bridge_uses_full_engine_batch_without_padding_for_two_agents(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bridge = bridge_mod.NativeInt8BackboneBridge(
                args=self._make_args(root),
                raw_dir=root / "raw",
                inventory=self._make_inventory(engine_batch=2),
            )
            bridge.tensor_quant_params = {"graph_output": {"scale": 1.0, "zero_point": 0, "source": "test"}}
            captured = self._install_fake_dispatch(bridge)

            outputs = bridge(torch.ones((2, 1, 1, 1), dtype=torch.float32))

        self.assertEqual(len(captured["requests"]), 1)
        self.assertEqual(list(captured["activations"][0].shape), [2, 1, 1, 1])
        self.assertEqual(bridge.worker_responses[0]["padding_agents"], 0)
        self.assertEqual([list(item.shape) for item in outputs], [[2, 1, 1, 1]])
        np.testing.assert_array_equal(
            outputs[0].numpy().reshape(-1),
            np.array([11.0, 12.0], dtype=np.float32),
        )

    def test_bridge_slices_each_chunk_back_to_real_agents_and_concats_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bridge = bridge_mod.NativeInt8BackboneBridge(
                args=self._make_args(root),
                raw_dir=root / "raw",
                inventory=self._make_inventory(engine_batch=2),
            )
            bridge.tensor_quant_params = {"graph_output": {"scale": 1.0, "zero_point": 0, "source": "test"}}
            captured = self._install_fake_dispatch(bridge)

            outputs = bridge(torch.ones((3, 1, 1, 1), dtype=torch.float32))

        self.assertEqual(len(captured["requests"]), 2)
        self.assertEqual([list(arr.shape) for arr in captured["activations"]], [[2, 1, 1, 1], [2, 1, 1, 1]])
        self.assertEqual(bridge.worker_responses[0]["padding_agents"], 0)
        self.assertEqual(bridge.worker_responses[1]["padding_agents"], 1)
        self.assertEqual([list(item.shape) for item in outputs], [[3, 1, 1, 1]])
        np.testing.assert_array_equal(
            outputs[0].numpy().reshape(-1),
            np.array([11.0, 12.0, 21.0], dtype=np.float32),
        )


if __name__ == "__main__":
    unittest.main()
