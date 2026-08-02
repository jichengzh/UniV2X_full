from __future__ import annotations

import sys
import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import stage3_codriving_trt_multiscale_ap_bridge_v3 as bridge_v3  # noqa: E402
import stage3_trt_multiscale_ap_bridge_v3 as shared_bridge_v3  # noqa: E402


class _FakeRunner:
    def __init__(self) -> None:
        self.calls: list[np.ndarray] = []

    def __call__(self, spatial_features: np.ndarray):
        self.calls.append(np.asarray(spatial_features))
        batch = spatial_features.shape[0]
        return [
            ("res2", np.full((batch, 128, 1, 1), 3.25, dtype=np.float32)),
            ("res0", np.full((batch, 16, 4, 4), 1.25, dtype=np.float32)),
            ("res1", np.full((batch, 32, 2, 2), 2.25, dtype=np.float32)),
        ]


class _Backbone:
    def __init__(self, resnet: object) -> None:
        self.resnet = resnet


class _Model:
    def __init__(self, resnet: object) -> None:
        self.backbone = _Backbone(resnet)


class Stage3CoDrivingTrtMultiscaleApBridgeV3Tests(unittest.TestCase):
    def test_file_sha256_uses_raw_bytes_without_filename_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "first.engine"
            second = Path(temporary) / "renamed.engine"
            first.write_bytes(b"same engine bytes")
            second.write_bytes(first.read_bytes())
            expected = hashlib.sha256(first.read_bytes()).hexdigest()

            self.assertEqual(bridge_v3.sha256_path(first), expected)
            self.assertEqual(bridge_v3.sha256_path(second), expected)

    def test_output_paths_are_resolved_before_changing_to_codriving_repo(self) -> None:
        args = SimpleNamespace(
            eval_dir=Path("relative/eval"), out_json=Path("relative/report.json"),
            engine=Path("relative/model.engine"), model_dir=Path("relative/model"),
        )

        bridge_v3.resolve_output_paths(args)

        self.assertTrue(args.eval_dir.is_absolute())
        self.assertTrue(args.out_json.is_absolute())
        self.assertTrue(args.engine.is_absolute())
        self.assertTrue(args.model_dir.is_absolute())
    def test_default_repo_root_matches_h800_checkout(self) -> None:
        self.assertEqual(
            bridge_v3.DEFAULT_REPO_ROOT,
            Path("/exdata/jichengzhi/V2Xverse_pyramid"),
        )

    def test_reuses_shared_trt_multiscale_runner(self) -> None:
        self.assertIs(bridge_v3.TrtMultiscaleRunner, shared_bridge_v3.TrtMultiscaleRunner)

    def test_resnet_bridge_pads_to_two_slices_outputs_and_records_reference_error(self) -> None:
        runner = _FakeRunner()
        reference = lambda _x: (
            torch.full((1, 16, 4, 4), 1.0),
            torch.full((1, 32, 2, 2), 2.0),
            torch.full((1, 128, 1, 1), 3.0),
        )
        module = bridge_v3.CoDrivingTrtResnetBridge(reference_resnet=reference, trt_runner=runner)

        outputs = module(torch.ones((1, 64, 4, 4), dtype=torch.float16))

        self.assertEqual(list(runner.calls[0].shape), [2, 64, 4, 4])
        self.assertTrue(np.allclose(runner.calls[0][1], 0.0))
        self.assertEqual([list(item.shape) for item in outputs], [[1, 16, 4, 4], [1, 32, 2, 2], [1, 128, 1, 1]])
        self.assertTrue(all(item.dtype == torch.float32 for item in outputs))
        self.assertEqual(len(module.output_error_records), 3)
        self.assertAlmostEqual(module.output_error_records[0]["max_abs_err"], 0.25)

    def test_patch_model_replaces_only_backbone_resnet(self) -> None:
        original = object()
        model = _Model(original)
        runner = _FakeRunner()

        patched = bridge_v3.patch_model_resnet(model, runner)

        self.assertIs(model.backbone.resnet, patched)
        self.assertIs(patched.reference_resnet, original)

    def test_gate_distinguishes_sanity_16_from_full_1789_and_forbids_fallback(self) -> None:
        def clean_output_summary(samples: int) -> dict:
            expected_records = samples * 6
            return {
                "all_finite": True,
                "shape_mismatch_count": 0,
                "nonfinite_count": 0,
                "num_records": expected_records,
                "num_compared": expected_records,
            }

        sanity = bridge_v3.evaluate_gates(
            processed_samples=16,
            engine_samples=16,
            fallback_samples=0,
            failed_samples=0,
            output_error_summary=clean_output_summary(16),
        )
        full = bridge_v3.evaluate_gates(
            processed_samples=1789,
            engine_samples=1789,
            fallback_samples=0,
            failed_samples=0,
            output_error_summary=clean_output_summary(1789),
        )
        fallback = bridge_v3.evaluate_gates(
            processed_samples=1789,
            engine_samples=1788,
            fallback_samples=1,
            failed_samples=0,
            output_error_summary=clean_output_summary(1789),
        )
        nonfinite = bridge_v3.evaluate_gates(
            processed_samples=1789,
            engine_samples=1789,
            fallback_samples=0,
            failed_samples=0,
            output_error_summary={
                **clean_output_summary(1789),
                "all_finite": False,
                "nonfinite_count": 1,
            },
        )

        self.assertEqual(sanity, {"sanity_16": True, "full_1789": False, "blockers": ["full_samples_below_1789"]})
        self.assertTrue(full["sanity_16"])
        self.assertTrue(full["full_1789"])
        self.assertEqual(full["blockers"], [])
        self.assertFalse(fallback["full_1789"])
        self.assertIn("fallback_forbidden", fallback["blockers"])
        self.assertIn("engine_samples_mismatch", fallback["blockers"])
        self.assertFalse(nonfinite["full_1789"])
        self.assertIn("output_nonfinite", nonfinite["blockers"])

    def test_report_binds_engine_checkpoint_config_dataset_and_protocol_shas(self) -> None:
        report = bridge_v3.build_report(
            model_dir=Path("/exdata/V2Xverse_pyramid/model"),
            engine_path=Path("/tmp/model.engine"),
            checkpoint_path=Path("/tmp/net_epoch20.pth"),
            config_path=Path("/tmp/config.yaml"),
            dataset_path=Path("/tmp/validate.txt"),
            shas={"engine": "a" * 64, "checkpoint": "b" * 64, "config": "c" * 64, "dataset": "d" * 64},
            protocol={"engine_batch": 2, "fallback_policy": "forbidden"},
            processed_samples=1789,
            engine_samples=1789,
            engine_calls=3578,
            fallback_samples=0,
            failed_samples=0,
            ap30=0.7,
            ap50=0.6,
            ap70=0.5,
            output_error_summary={
                "all_finite": True,
                "shape_mismatch_count": 0,
                "nonfinite_count": 0,
                "num_records": 1789 * 6,
                "num_compared": 1789 * 6,
                "max_abs_err": 0.25,
            },
            elapsed_secs=12.0,
        )

        self.assertEqual(report["pipeline_scope"], "codriving_resnet_trt_in_full_pytorch_fusion_head_postprocess")
        for key in ("engine", "checkpoint", "config", "dataset", "protocol"):
            self.assertEqual(len(report["sha256"][key]), 64)
        self.assertTrue(report["gates"]["full_1789"])
        self.assertEqual(report["engine_calls"], 3578)
        self.assertEqual(report["engine_calls_per_sample"], 2.0)
        self.assertEqual(report["output_vs_reference_error"]["max_abs_err"], 0.25)


if __name__ == "__main__":
    unittest.main()
