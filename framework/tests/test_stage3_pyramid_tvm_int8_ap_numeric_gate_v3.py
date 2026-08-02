from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import stage3_pyramid_tvm_int8_ap_numeric_gate_v3 as gate_v3  # noqa: E402


def _runtime_arg_plan(*, valid_quant: bool = True) -> list[dict[str, object]]:
    scale = 0.125 if valid_quant else 0.0
    return [
        {
            "role": "graph_input",
            "arg_name": "spatial_features",
            "shape": [2, 1, 1, 1],
            "dtype": "uint8",
            "quantization": {"scale": scale, "zero_point": 3, "source": "calibration"},
        },
        {
            "role": "weight_input",
            "arg_name": "weight_1",
            "initializer_name": "conv.weight",
            "shape": [1, 1, 1, 1],
            "dtype": "int8",
        },
        {
            "role": "graph_output",
            "arg_name": "pyramid_0",
            "shape": [2, 1, 1, 1],
            "dtype": "uint8",
            "quantization": {"scale": scale, "zero_point": 0, "source": "calibration"},
        },
    ]


class Stage3PyramidTvmInt8ApNumericGateV3Tests(unittest.TestCase):
    def test_cli_matches_planner_and_defaults_to_numeric_sanity(self) -> None:
        args = gate_v3.parse_args([
            "--compiled-artifact", "/tmp/model.vmexec",
            "--precision-tag", "int8",
            "--num-samples", "16",
            "--full-ap-min-samples", "1789",
            "--output-dir", "/tmp/out",
            "--model-dir", "/tmp/model",
            "--report-json", "/tmp/report.json",
        ])

        self.assertEqual(args.model_dir, Path("/tmp/model"))
        self.assertEqual(args.report_json, Path("/tmp/report.json"))
        self.assertTrue(args.numeric_sanity_only)

    def test_inventory_is_derived_from_route_runtime_plan_and_npz(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            weights = root / "runtime_weights_int8.npz"
            np.savez(weights, **{"conv.weight": np.ones((1, 1, 1, 1), dtype=np.int8)})

            inventory = gate_v3.build_audit_inventory(
                runtime_arg_plan=_runtime_arg_plan(),
                runtime_weight_archive_path=weights,
                route_report_path=root / "route_b_int8_auto_decomp_result.json",
            )

        self.assertEqual(inventory["execution_abi"], "relax_vm_return")
        self.assertEqual(inventory["runtime_arg_plan"], _runtime_arg_plan())
        self.assertEqual(inventory["runtime_weight_archive_keys"], {"weight_1": "conv.weight"})

    def test_vmexec_is_copied_to_output_as_so(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "compiled.vmexec"
            artifact.write_bytes(b"TVM")

            prepared = gate_v3.prepare_compiled_artifact(artifact, root / "output")

            self.assertEqual(prepared.suffix, ".so")
            self.assertEqual(prepared.parent, root / "output")
            self.assertEqual(prepared.read_bytes(), b"TVM")

    def test_full_run_with_invalid_quant_params_writes_feasibility_blocker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact_dir = root / "artifact"
            native_dir = artifact_dir / "native_direct_reference"
            native_dir.mkdir(parents=True)
            artifact = artifact_dir / "route.vmexec"
            artifact.write_bytes(b"TVM")
            np.savez(native_dir / "runtime_weights_int8.npz", **{"conv.weight": np.ones(1, dtype=np.int8)})
            (artifact_dir / "route_b_int8_auto_decomp_result.json").write_text(json.dumps({
                "runtime_arg_plan": _runtime_arg_plan(valid_quant=False),
            }))
            args = gate_v3.parse_args([
                "--compiled-artifact", str(artifact), "--precision-tag", "int8",
                "--num-samples", "1789", "--full-ap-min-samples", "1789",
                "--output-dir", str(root / "out"), "--model-dir", str(root / "model"),
            ])

            with mock.patch.object(gate_v3.stage2_bridge, "run_bridge") as run_stage2:
                report = gate_v3.run_gate(args)

            run_stage2.assert_not_called()
            self.assertEqual(report["status"], "blocked")
            self.assertFalse(report["ap_row_allowed"])
            self.assertFalse(report["gates"]["full_1789"])
            self.assertIn("invalid_tensor_quant_params", report["feasibility_blockers"])
            self.assertTrue((root / "out" / "feasibility_blocker.json").is_file())

    def test_full_measured_requires_stage2_ap_row_and_clean_full_gate(self) -> None:
        stage2_report = {
            "status": "success", "processed_samples": 1789, "failed_samples": 0,
            "ap_row_allowed": True, "ap30": 0.7, "ap50": 0.6, "ap70": 0.5,
        }

        gates = gate_v3.evaluate_result_gates(
            stage2_report, tensor_quant_params_valid=True, requested_samples=1789,
            full_ap_min_samples=1789,
        )

        self.assertTrue(gates["ap_row_allowed"])
        self.assertTrue(gates["full_1789"])
        self.assertEqual(gates["feasibility_blockers"], [])

    def test_sixteen_samples_invokes_existing_stage2_real_activation_bridge(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact_dir = root / "artifact"
            native_dir = artifact_dir / "native_direct_reference"
            native_dir.mkdir(parents=True)
            artifact = artifact_dir / "route.vmexec"
            artifact.write_bytes(b"TVM")
            np.savez(native_dir / "runtime_weights_int8.npz", **{"conv.weight": np.ones(1, dtype=np.int8)})
            (artifact_dir / "route_b_int8_auto_decomp_result.json").write_text(json.dumps({
                "runtime_arg_plan": _runtime_arg_plan(),
            }))
            args = gate_v3.parse_args([
                "--compiled-artifact", str(artifact), "--precision-tag", "int8",
                "--num-samples", "16", "--output-dir", str(root / "out"),
                "--model-dir", str(root / "model"),
            ])
            stage2_result = {
                "status": "pass", "processed_samples": 16, "failed_samples": 0,
                "ap_measured": False,
            }

            with mock.patch.object(gate_v3.stage2_bridge, "run_bridge", return_value=stage2_result) as run_stage2:
                report = gate_v3.run_gate(args)

            bridge_args = run_stage2.call_args.args[0]
            self.assertTrue(bridge_args.numeric_sanity_only)
            self.assertEqual(bridge_args.execution_abi, "relax_vm_return")
            self.assertEqual(Path(bridge_args.runtime_weight_archive_path), native_dir / "runtime_weights_int8.npz")
            self.assertTrue(report["gates"]["sanity_16"])
            self.assertFalse(report["gates"]["full_1789"])
            self.assertFalse(report["ap_measured"])

    def test_relative_output_dir_is_resolved_before_persistent_worker_dispatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact_dir = root / "artifact"
            native_dir = artifact_dir / "native_direct_reference"
            native_dir.mkdir(parents=True)
            artifact = artifact_dir / "route.vmexec"
            artifact.write_bytes(b"TVM")
            np.savez(native_dir / "runtime_weights_int8.npz", **{"conv.weight": np.ones(1, dtype=np.int8)})
            (artifact_dir / "route_b_int8_auto_decomp_result.json").write_text(json.dumps({
                "runtime_arg_plan": _runtime_arg_plan(),
            }))
            args = gate_v3.parse_args([
                "--compiled-artifact", str(artifact), "--precision-tag", "int8",
                "--num-samples", "16", "--output-dir", "relative/out",
                "--model-dir", str(root / "model"),
            ])
            stage2_result = {
                "status": "pass", "processed_samples": 16, "failed_samples": 0,
                "ap_measured": False,
            }

            with mock.patch.object(gate_v3.stage2_bridge, "run_bridge", return_value=stage2_result) as run_stage2:
                with mock.patch("pathlib.Path.cwd", return_value=root):
                    gate_v3.run_gate(args)

            bridge_args = run_stage2.call_args.args[0]
            self.assertTrue(Path(bridge_args.raw_dir).is_absolute())
            self.assertTrue(args.report_json.is_absolute())


if __name__ == "__main__":
    unittest.main()
