from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts import stage2_h800_native_int8_real_activation_bridge as native_bridge  # noqa: E402
from scripts import stage3_codriving_tvm_int8_ap_numeric_gate_v3 as gate_v3  # noqa: E402


class _FakeBridge:
    def __init__(self) -> None:
        self.reference_get_multiscale = None
        self.numeric_sanity_records = []
        self.call_index = 0
        self.closed = False

    def __call__(self, value):
        self.call_index += 1
        return (value, value, value)

    def close(self) -> None:
        self.closed = True


class _Backbone:
    def __init__(self, resnet) -> None:
        self.resnet = resnet


class _Model:
    def __init__(self, resnet) -> None:
        self.backbone = _Backbone(resnet)


class Stage3CoDrivingTvmInt8NumericGateV3Tests(unittest.TestCase):
    def test_engine_accounting_separates_samples_from_backbone_calls(self) -> None:
        accounting = gate_v3.engine_accounting(processed_samples=16, engine_calls=32)

        self.assertEqual(accounting["engine_samples"], 16)
        self.assertEqual(accounting["engine_calls"], 32)
        self.assertEqual(accounting["engine_calls_per_sample"], 2.0)
        self.assertTrue(accounting["engine_accounting_valid"])

        non_divisible = gate_v3.engine_accounting(processed_samples=16, engine_calls=31)
        self.assertFalse(non_divisible["engine_accounting_valid"])
        self.assertIn("not_divisible", non_divisible["engine_accounting_failure_reason"])

        no_calls = gate_v3.engine_accounting(processed_samples=16, engine_calls=0)
        self.assertFalse(no_calls["engine_accounting_valid"])

        drift = gate_v3.engine_accounting(
            processed_samples=16, engine_calls=16, expected_calls_per_sample=2.0
        )
        self.assertFalse(drift["engine_accounting_valid"])
        self.assertIn("pattern_mismatch", drift["engine_accounting_failure_reason"])

    def test_place_model_on_cuda_binds_requested_device(self) -> None:
        model = Mock()
        model.to.return_value.eval.return_value = "ready"
        expected = torch.device("cuda", 7)
        with patch.object(gate_v3.torch.cuda, "set_device") as set_device:
            result = gate_v3.place_model_on_cuda(model, 7)
        set_device.assert_called_once_with(expected)
        model.to.assert_called_once_with(expected)
        self.assertEqual(result, "ready")

    def test_place_model_on_cuda_maps_visible_physical_gpu_to_logical_zero(self) -> None:
        model = Mock()
        model.to.return_value.eval.return_value = "ready"
        expected = torch.device("cuda", 0)
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "6"}), patch.object(
            gate_v3.torch.cuda, "set_device"
        ) as set_device:
            result = gate_v3.place_model_on_cuda(model, 6)
        set_device.assert_called_once_with(expected)
        model.to.assert_called_once_with(expected)
        self.assertEqual(result, "ready")

    def test_runtime_paths_can_be_resolved_before_repo_chdir(self) -> None:
        args = gate_v3.parse_args([
            "--compiled-artifact", "relative/route_b_int8_auto_decomp.vmexec",
            "--precision-tag", "int8", "--num-samples", "16",
            "--output-dir", "relative/out", "--model-dir", "relative/model",
            "--report-json", "relative/report.json",
        ])
        with unittest.mock.patch.object(gate_v3, "preflight_numerical_feasibility", return_value={
            "status": "numerical_feasibility_failure",
        }) as preflight, unittest.mock.patch.object(gate_v3, "write_json"):
            gate_v3.run_bridge(args)
        self.assertTrue(Path(preflight.call_args.args[0]).is_absolute())
        self.assertTrue(args.output_dir.is_absolute())
        self.assertTrue(args.report_json.is_absolute())
    def _route(self, root: Path, *, include_all_params: bool = True) -> Path:
        route = root / "route"
        route.mkdir()
        vmexec = route / "route_b_int8_auto_decomp.vmexec"
        vmexec.write_bytes(b"vmexec")
        weights = route / "runtime_weights_int8.npz"
        weights.write_bytes(b"weights")
        direct = route / "native_direct_reference" / "reference.so"
        direct.parent.mkdir()
        direct.write_bytes(b"reference")
        plan = [
            {
                "arg_name": "spatial_features", "role": "graph_input",
                "shape": [2, 64, 8, 8], "dtype": "uint8",
            },
            *[
                {"arg_name": name, "role": "graph_output", "shape": [2, 64, 8, 8], "dtype": "uint8"}
                for name in ("out0", "out1", "out2")
            ],
        ]
        result = {
            "status": "success",
            "artifact_path": str(vmexec),
            "runtime_arg_plan": plan,
            "runtime_weight_archive_path": str(weights),
            "runtime_weight_archive_keys": {"weight": "weight"},
            "native_direct_reference": {"artifact_path": str(direct)},
        }
        (route / "route_b_int8_auto_decomp_result.json").write_text(json.dumps(result))
        params = {
            name: {"scale": 0.25, "zero_point": 3, "source": "calibration"}
            for name in ("spatial_features", "out0", "out1", "out2")
        }
        if not include_all_params:
            params.pop("out2")
        (route / "tensor_quant_params.json").write_text(json.dumps({"params": params}))
        return vmexec

    def test_reuses_native_int8_bridge_and_patches_only_resnet(self) -> None:
        self.assertIs(gate_v3.NativeInt8BackboneBridge, native_bridge.NativeInt8BackboneBridge)
        original = torch.nn.Identity()
        model = _Model(original)
        shared = _FakeBridge()

        module = gate_v3.patch_model_resnet(model, shared)

        self.assertIs(model.backbone.resnet, module)
        self.assertIs(shared.reference_get_multiscale, original)
        module.close()
        self.assertTrue(shared.closed)

    def test_materializes_relax_inventory_and_so_copy_from_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vmexec = self._route(root)
            bundle = gate_v3.materialize_runtime_bundle(vmexec, root / "run")
            inventory = json.loads(bundle.inventory_path.read_text())

            self.assertEqual(bundle.artifact_path.suffix, ".so")
            self.assertEqual(bundle.artifact_path.read_bytes(), b"vmexec")
            self.assertEqual(inventory["execution_abi"], "relax_vm_return")
            self.assertEqual(inventory["runtime_arg_plan"][0]["role"], "graph_input")
            self.assertEqual(bundle.runtime_weights_path.read_bytes(), b"weights")
            self.assertEqual(bundle.native_direct_reference_path.read_bytes(), b"reference")
            self.assertEqual(
                bundle.source_tensor_quant_params_path,
                (vmexec.parent / "tensor_quant_params.json").resolve(),
            )

    def test_runtime_weights_fall_back_to_native_direct_reference_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vmexec = self._route(root)
            vmexec.parent.joinpath("runtime_weights_int8.npz").unlink()
            result_path = vmexec.parent / "route_b_int8_auto_decomp_result.json"
            result = json.loads(result_path.read_text())
            result.pop("runtime_weight_archive_path")
            result_path.write_text(json.dumps(result))
            nested = vmexec.parent / "native_direct_reference" / "runtime_weights_int8.npz"
            nested.parent.mkdir(exist_ok=True)
            nested.write_bytes(b"nested-weights")

            _, _, weights, _, _ = gate_v3.inspect_route_artifact(vmexec)

            self.assertEqual(weights, nested.resolve())

    def test_materialization_rebases_stale_native_reference_to_same_route_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vmexec = self._route(root)
            result_path = vmexec.parent / "route_b_int8_auto_decomp_result.json"
            result = json.loads(result_path.read_text())
            result["native_direct_reference"]["artifact_path"] = "/stale/host/reference.so"
            result_path.write_text(json.dumps(result))

            bundle = gate_v3.materialize_runtime_bundle(vmexec, root / "run")

            self.assertEqual(bundle.native_direct_reference_path.read_bytes(), b"reference")

    def test_missing_output_dequant_param_is_credible_feasibility_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vmexec = self._route(root, include_all_params=False)

            report = gate_v3.preflight_numerical_feasibility(vmexec)

            self.assertEqual(report["status"], "numerical_feasibility_failure")
            self.assertFalse(report["ap_measured"])
            self.assertNotIn("ap", report)
            self.assertIn("missing_output_dequant_params:out2", report["failure_reasons"])

    def test_preflight_requires_static_spatial_features_quant_params(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            vmexec = self._route(Path(tmp))
            params_path = vmexec.parent / "tensor_quant_params.json"
            payload = json.loads(params_path.read_text())
            payload["params"].pop("spatial_features")
            params_path.write_text(json.dumps(payload))

            report = gate_v3.preflight_numerical_feasibility(vmexec)

            self.assertEqual(report["status"], "numerical_feasibility_failure")
            self.assertIn(
                "missing_static_quant_params:spatial_features",
                report["failure_reasons"],
            )

    def test_preflight_rejects_implicit_zero_point_and_dynamic_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            vmexec = self._route(Path(tmp))
            params_path = vmexec.parent / "tensor_quant_params.json"
            payload = json.loads(params_path.read_text())
            payload["params"]["out1"].pop("zero_point")
            payload["params"]["out1"]["source"] = "dynamic_per_chunk_fallback"
            params_path.write_text(json.dumps(payload))

            report = gate_v3.preflight_numerical_feasibility(vmexec)

            self.assertIn("missing_static_quant_params:out1", report["failure_reasons"])
            self.assertIn("dynamic_quant_fallback_forbidden:out1", report["failure_reasons"])

    def test_correlation_failure_blocks_ap_without_fabricated_values(self) -> None:
        records = [
            {"tensor_name": name, "alignment_error": {"corrcoef": corr}, "passed": corr >= 0.5}
            for name, corr in (("out0", 0.9), ("out1", 0.8), ("out2", 0.1))
        ]

        report = gate_v3.build_numerical_gate_report(
            records, processed_samples=16, engine_calls=16
        )

        self.assertEqual(report["status"], "numerical_feasibility_failure")
        self.assertFalse(report["gates"]["sanity_16"])
        self.assertFalse(report["ap_measured"])
        self.assertNotIn("ap30", report)

    def test_sanity_requires_numeric_records_for_every_engine_call(self) -> None:
        records = [
            {"tensor_name": name, "alignment_error": {"corrcoef": 0.9}, "passed": True}
            for name in ("out0", "out1", "out2")
            for _ in range(16)
        ]

        report = gate_v3.build_numerical_gate_report(
            records, processed_samples=16, engine_calls=32
        )

        self.assertEqual(report["status"], "numerical_feasibility_failure")
        self.assertFalse(report["gates"]["sanity_16"])
        self.assertTrue(any("record_coverage_failed" in item for item in report["failure_reasons"]))

    def test_sanity_accepts_complete_numeric_records_for_every_engine_call(self) -> None:
        records = [
            {
                "tensor_name": name,
                "sample_index": call_index,
                "alignment_error": {"corrcoef": 0.9},
                "passed": True,
            }
            for name in ("out0", "out1", "out2")
            for call_index in range(32)
        ]

        report = gate_v3.build_numerical_gate_report(
            records, processed_samples=16, engine_calls=32
        )

        self.assertEqual(report["status"], "numerical_sanity_passed")
        self.assertTrue(report["gates"]["sanity_16"])
        self.assertTrue(report["engine_accounting_valid"])

    def test_sanity_rejects_duplicate_records_that_hide_a_missing_call(self) -> None:
        call_indexes = list(range(31)) + [0]
        records = [
            {
                "tensor_name": name,
                "sample_index": call_index,
                "alignment_error": {"corrcoef": 0.9},
                "passed": True,
            }
            for name in ("out0", "out1", "out2")
            for call_index in call_indexes
        ]

        report = gate_v3.build_numerical_gate_report(
            records, processed_samples=16, engine_calls=32
        )

        self.assertEqual(report["status"], "numerical_feasibility_failure")
        self.assertTrue(any("missing_call_ids=31" in reason for reason in report["failure_reasons"]))

    def test_sanity_accepts_multiple_chunks_per_engine_call(self) -> None:
        records = [
            {
                "tensor_name": name,
                "sample_index": call_index,
                "chunk_index": chunk_index,
                "alignment_error": {"corrcoef": 0.9},
                "passed": True,
            }
            for name in ("out0", "out1", "out2")
            for call_index in range(32)
            for chunk_index in (0, 1)
        ]

        report = gate_v3.build_numerical_gate_report(
            records, processed_samples=16, engine_calls=32
        )

        self.assertEqual(report["status"], "numerical_sanity_passed")
        self.assertTrue(report["gates"]["sanity_16"])

    def test_full_gate_reports_short_dataset_without_claiming_ap(self) -> None:
        accounting = gate_v3.engine_accounting(
            processed_samples=1788,
            engine_calls=3576,
            expected_calls_per_sample=2.0,
        )

        decision = gate_v3.build_full_gate_decision(
            processed_samples=1788,
            minimum_samples=1789,
            accounting=accounting,
        )

        self.assertFalse(decision["full_gate"])
        self.assertFalse(decision["ap_measured"])
        self.assertEqual(decision["failure_reasons"], ["full_ap_samples_1788_lt_1789"])

    def test_full_run_requires_persisted_passing_sanity_report_and_sha(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "sanity.json"
            with self.assertRaisesRegex(ValueError, "passing 16-sample numerical sanity"):
                gate_v3.require_full_run_sanity(1789, missing, None, {})
            missing.write_text(json.dumps({"gates": {"sanity_16": True}, "processed_samples": 16}))
            with self.assertRaisesRegex(ValueError, "sanity report SHA256 is required"):
                gate_v3.require_full_run_sanity(1789, missing, None, {})

    def test_full_run_rejects_sanity_sha_or_binding_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "route_b_int8_auto_decomp.vmexec"
            artifact.write_bytes(b"artifact")
            checkpoint = root / "net_epoch_bestval_at1.pth"
            checkpoint.write_bytes(b"checkpoint")
            params = root / "tensor_quant_params.json"
            params.write_text("{}")
            protocol = gate_v3.protocol_payload(precision_tag="int8", full_ap_min_samples=1789)
            bindings = gate_v3.build_run_bindings(
                source_compiled_artifact=artifact,
                model_dir=root / "model",
                checkpoint_path=checkpoint,
                tensor_quant_params_path=params,
                protocol=protocol,
            )
            sanity = root / "sanity.json"
            sanity.write_text(json.dumps({
                "gates": {"sanity_16": True}, "processed_samples": 16,
                "engine_samples": 16, "engine_calls": 32,
                "engine_calls_per_sample": 2.0, "engine_accounting_valid": True,
                **bindings,
            }, sort_keys=True))
            sanity_sha = hashlib.sha256(sanity.read_bytes()).hexdigest()

            self.assertEqual(
                gate_v3.require_full_run_sanity(1789, sanity, sanity_sha, bindings), 2.0
            )
            with self.assertRaisesRegex(ValueError, "sanity report SHA256 mismatch"):
                gate_v3.require_full_run_sanity(1789, sanity, "0" * 64, bindings)
            wrong = dict(bindings)
            wrong["model_dir"] = str(root / "other-model")
            with self.assertRaisesRegex(ValueError, "sanity binding mismatch:model_dir"):
                gate_v3.require_full_run_sanity(1789, sanity, sanity_sha, wrong)

    def test_report_binding_fields_include_source_params_checkpoint_and_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "route_b_int8_auto_decomp.vmexec"
            checkpoint = root / "checkpoint.pth"
            params = root / "tensor_quant_params.json"
            artifact.write_bytes(b"artifact")
            checkpoint.write_bytes(b"checkpoint")
            params.write_text("{}")
            protocol = gate_v3.protocol_payload(precision_tag="int8", full_ap_min_samples=1789)

            report = gate_v3.build_run_bindings(
                source_compiled_artifact=artifact, model_dir=root / "model",
                checkpoint_path=checkpoint, tensor_quant_params_path=params,
                protocol=protocol,
            )

            self.assertEqual(report["source_compiled_artifact"], str(artifact.resolve()))
            self.assertEqual(report["source_compiled_artifact_sha256"], gate_v3.sha256_path(artifact))
            self.assertEqual(report["checkpoint_sha256"], gate_v3.sha256_path(checkpoint))
            self.assertEqual(report["tensor_quant_params_sha256"], gate_v3.sha256_path(params))
            self.assertEqual(report["protocol"], protocol)
            self.assertEqual(report["protocol_sha256"], gate_v3.sha256_json(protocol))

    def test_cli_matches_generic_planner_and_binds_codriving_paths(self) -> None:
        args = gate_v3.parse_args([
            "--compiled-artifact", "/tmp/route_b_int8_auto_decomp.vmexec",
            "--precision-tag", "int8", "--num-samples", "16",
            "--full-ap-min-samples", "1789", "--output-dir", "/tmp/out",
            "--model-dir", "/tmp/model", "--repo-root", "/tmp/repo",
            "--report-json", "/tmp/report.json",
            "--sanity-report-sha256", "a" * 64,
        ])

        self.assertEqual(args.model_dir, Path("/tmp/model"))
        self.assertEqual(args.repo_root, Path("/tmp/repo"))
        self.assertEqual(args.report_json, Path("/tmp/report.json"))
        self.assertTrue(args.persistent_worker)
        self.assertTrue(args.numeric_sanity_only)
        self.assertEqual(args.sanity_report_sha256, "a" * 64)


if __name__ == "__main__":
    unittest.main()
