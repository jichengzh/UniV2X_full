from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import stage3_codriving_tvm_fp16_ap_bridge_v3 as bridge_v3  # noqa: E402
from scripts import stage2_h800_fp16_rewritten_activation_bridge as shared_bridge  # noqa: E402


class _FakeSharedBridge:
    def __init__(self) -> None:
        self.reference_get_multiscale = None
        self.output_error_records = [{"max_abs_err": 0.125}]
        self.calls: list[torch.Tensor] = []
        self.closed = False

    def __call__(self, value: torch.Tensor):
        self.calls.append(value)
        return (value + 1, value + 2, value + 3)

    def close(self) -> None:
        self.closed = True


class _Backbone:
    def __init__(self, resnet: object) -> None:
        self.resnet = resnet


class _Model:
    def __init__(self, resnet: object) -> None:
        self.backbone = _Backbone(resnet)


class Stage3CoDrivingTvmFp16ApBridgeV3Tests(unittest.TestCase):
    def test_output_paths_are_resolved_before_changing_to_codriving_repo(self) -> None:
        args = SimpleNamespace(output_dir=Path("relative/out"), report_json=Path("relative/report.json"))

        bridge_v3.resolve_output_paths(args)

        self.assertTrue(args.output_dir.is_absolute())
        self.assertTrue(args.report_json.is_absolute())
    def test_parse_args_accepts_explicit_report_json(self) -> None:
        args = bridge_v3.parse_args(
            [
                "--compiled-artifact", "/tmp/model.so",
                "--precision-tag", "fp16",
                "--num-samples", "16",
                "--output-dir", "/tmp/output",
                "--model-dir", "/tmp/model",
                "--report-json", "/tmp/report.json",
            ]
        )

        self.assertEqual(args.report_json, Path("/tmp/report.json"))
    def test_reuses_stage2_fp16_rewritten_bridge_class(self) -> None:
        self.assertIs(
            bridge_v3.Fp16RewrittenBackboneBridge,
            shared_bridge.Fp16RewrittenBackboneBridge,
        )

    def test_thin_module_delegates_to_shared_bridge_and_keeps_reference(self) -> None:
        shared = _FakeSharedBridge()
        reference = lambda value: (value, value, value)
        module = bridge_v3.CoDrivingTvmResnetModule(
            reference_resnet=reference,
            shared_bridge=shared,
        )
        value = torch.ones((1, 4, 2, 2))

        outputs = module(value)

        self.assertIs(shared.reference_get_multiscale, reference)
        self.assertEqual(len(shared.calls), 1)
        self.assertEqual(len(outputs), 3)
        self.assertIs(module.output_error_records, shared.output_error_records)
        module.close()
        self.assertTrue(shared.closed)

    def test_patch_replaces_only_model_backbone_resnet(self) -> None:
        original = torch.nn.Identity()
        model = _Model(original)
        shared = _FakeSharedBridge()

        module = bridge_v3.patch_model_resnet(model, shared)

        self.assertIs(model.backbone.resnet, module)
        self.assertIs(module.reference_resnet, original)

    def test_protocol_fixes_float32_input_and_forbids_fallback(self) -> None:
        protocol = bridge_v3.protocol_payload(
            precision_tag="fp16",
            full_ap_min_samples=1789,
        )

        self.assertEqual(protocol["artifact_input_dtype"], "float32")
        self.assertEqual(protocol["patch_target"], "model.backbone.resnet")
        self.assertEqual(protocol["fallback_policy"], "forbidden")
        self.assertEqual(protocol["sanity_samples"], 16)
        self.assertEqual(protocol["full_samples"], 1789)

    def test_gates_require_clean_16_and_1789_engine_samples(self) -> None:
        sanity = bridge_v3.evaluate_gates(
            processed_samples=16, engine_samples=16, fallback_samples=0, failed_samples=0,
            full_ap_min_samples=1789,
        )
        full = bridge_v3.evaluate_gates(
            processed_samples=1789, engine_samples=1789, fallback_samples=0, failed_samples=0,
            full_ap_min_samples=1789,
        )
        failed = bridge_v3.evaluate_gates(
            processed_samples=1789, engine_samples=1788, fallback_samples=1, failed_samples=0,
            full_ap_min_samples=1789,
        )

        self.assertTrue(sanity["sanity_16"])
        self.assertFalse(sanity["full_1789"])
        self.assertTrue(full["full_1789"])
        self.assertIn("fallback_forbidden", failed["blockers"])
        self.assertIn("engine_samples_mismatch", failed["blockers"])

    def test_report_binds_all_sources_ap_and_output_error(self) -> None:
        protocol = bridge_v3.protocol_payload(precision_tag="fp16", full_ap_min_samples=1789)
        report = bridge_v3.build_report(
            model_dir=Path("/model"), artifact_path=Path("/route_b_fp16_auto.so"),
            checkpoint_path=Path("/model/net_epoch20.pth"), config_path=Path("/model/config.yaml"),
            dataset_path=Path("/data/val.txt"),
            shas={"artifact": "a" * 64, "checkpoint": "b" * 64, "config": "c" * 64, "dataset": "d" * 64},
            protocol=protocol, processed_samples=1789, engine_samples=1789, engine_calls=3578,
            fallback_samples=0, failed_samples=0, ap30=0.7, ap50=0.6, ap70=0.5,
            output_error_summary={"max_abs_err": 0.125}, elapsed_secs=12.0,
            full_ap_min_samples=1789,
        )

        self.assertEqual(report["ap"], {"ap30": 0.7, "ap50": 0.6, "ap70": 0.5})
        self.assertEqual(report["output_error_summary"]["max_abs_err"], 0.125)
        for key in ("artifact", "checkpoint", "config", "dataset"):
            self.assertEqual(report[f"{key}_sha256"], report["sha256"][key])
        self.assertEqual(report["protocol_sha256"], bridge_v3.sha256_json(protocol))
        self.assertTrue(report["gates"]["full_1789"])
        self.assertEqual(report["engine_calls"], 3578)
        self.assertEqual(report["engine_calls_per_sample"], 2.0)

    def test_cli_matches_stage3_planner_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = bridge_v3.parse_args([
                "--compiled-artifact", "/tmp/route_b_fp16_auto.so",
                "--precision-tag", "fp16",
                "--num-samples", "16",
                "--full-ap-min-samples", "1789",
                "--output-dir", tmp,
                "--model-dir", "/tmp/model",
                "--repo-root", "/tmp/repo",
            ])

        self.assertEqual(args.artifact_input_dtype, "float32")
        self.assertEqual(args.num_samples, 16)
        self.assertTrue(args.persistent_worker)


if __name__ == "__main__":
    unittest.main()
