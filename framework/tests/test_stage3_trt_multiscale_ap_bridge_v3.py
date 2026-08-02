from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import stage3_trt_multiscale_ap_bridge_v3 as bridge_v3  # noqa: E402


class _FakeRunner:
    def __init__(self) -> None:
        self.calls: list[np.ndarray] = []

    def __call__(self, spatial_features: np.ndarray):
        self.calls.append(np.asarray(spatial_features))
        batch = spatial_features.shape[0]
        return [
            ("out_small", np.full((batch, 256, 1, 1), 3.25, dtype=np.float32)),
            ("out_large", np.full((batch, 64, 4, 4), 1.25, dtype=np.float32)),
            ("out_mid", np.full((batch, 128, 2, 2), 2.25, dtype=np.float32)),
        ]


class Stage3TrtMultiscaleApBridgeV3Tests(unittest.TestCase):
    def test_prepare_spatial_features_pads_single_agent_to_fixed_engine_batch(self) -> None:
        spatial = np.ones((1, 64, 8, 8), dtype=np.float32)

        padded, meta = bridge_v3.prepare_spatial_features_for_engine(
            spatial,
            record_len=1,
            engine_batch=2,
        )

        self.assertEqual(list(padded.shape), [2, 64, 8, 8])
        self.assertEqual(meta["original_agents"], 1)
        self.assertEqual(meta["engine_batch"], 2)
        self.assertEqual(meta["padding_agents"], 1)
        self.assertTrue(np.allclose(padded[1], 0.0))

    def test_sort_output_specs_orders_by_spatial_resolution_then_name(self) -> None:
        specs = bridge_v3.sort_output_specs(
            [
                {"name": "z_small", "shape": [2, 128, 1, 1]},
                {"name": "a_large", "shape": [2, 24, 4, 4]},
                {"name": "m_mid", "shape": [2, 48, 2, 2]},
                {"name": "b_large", "shape": [2, 24, 4, 4]},
            ]
        )

        self.assertEqual([item["name"] for item in specs], ["a_large", "b_large", "m_mid", "z_small"])

    def test_bridge_records_reference_error_and_slices_back_to_original_agent_count(self) -> None:
        runner = _FakeRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir)
            bridge = bridge_v3.TrtMultiscaleBackboneBridge(
                raw_dir=raw_dir,
                precision_tag="fp16",
                engine_path=Path("/tmp/fake.engine"),
                trt_runner=runner,
                expected_output_channels=(64, 128, 256),
            )
            bridge.reference_get_multiscale = lambda x: (
                torch.full((1, 64, 4, 4), 1.0, dtype=torch.float32),
                torch.full((1, 128, 2, 2), 2.0, dtype=torch.float32),
                torch.full((1, 256, 1, 1), 3.0, dtype=torch.float32),
            )
            bridge.current_record_len = 1

            spatial = torch.ones((1, 64, 4, 4), dtype=torch.float16)
            outputs = bridge(spatial)

            self.assertEqual(len(runner.calls), 1)
            self.assertEqual(list(runner.calls[0].shape), [2, 64, 4, 4])
            self.assertEqual(
                [list(item.shape) for item in outputs],
                [[1, 64, 4, 4], [1, 128, 2, 2], [1, 256, 1, 1]],
            )
            self.assertTrue(all(item.dtype == torch.float32 for item in outputs))
            self.assertEqual(len(bridge.output_error_records), 3)
            self.assertTrue(all(item["candidate_all_finite"] for item in bridge.output_error_records))
            self.assertAlmostEqual(bridge.output_error_records[0]["max_abs_err"], 0.25, places=6)

    def test_bridge_fails_closed_when_ordered_channels_violate_contract(self) -> None:
        runner = _FakeRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            bridge = bridge_v3.TrtMultiscaleBackboneBridge(
                raw_dir=Path(tmpdir),
                precision_tag="fp32",
                engine_path=Path("/tmp/fake.engine"),
                trt_runner=runner,
                expected_output_channels=(64, 128, 255),
            )
            bridge.reference_get_multiscale = lambda x: (
                torch.zeros((1, 64, 4, 4)),
                torch.zeros((1, 128, 2, 2)),
                torch.zeros((1, 256, 1, 1)),
            )
            bridge.current_record_len = 1

            with self.assertRaisesRegex(RuntimeError, "channel"):
                bridge(torch.ones((1, 64, 4, 4)))

    def test_bridge_fails_closed_when_reference_shape_differs(self) -> None:
        runner = _FakeRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            bridge = bridge_v3.TrtMultiscaleBackboneBridge(
                raw_dir=Path(tmpdir),
                precision_tag="fp32",
                engine_path=Path("/tmp/fake.engine"),
                trt_runner=runner,
                expected_output_channels=(64, 128, 256),
            )
            bridge.reference_get_multiscale = lambda x: (
                torch.zeros((1, 64, 4, 4)),
                torch.zeros((1, 128, 3, 2)),
                torch.zeros((1, 256, 1, 1)),
            )
            bridge.current_record_len = 1

            with self.assertRaisesRegex(RuntimeError, "reference output shape"):
                bridge(torch.ones((1, 64, 4, 4)))

    def test_cli_expected_output_channels_is_optional_and_parses_contract(self) -> None:
        required = [
            "bridge.py",
            "--ckpt-dir",
            "/tmp/ckpt",
            "--engine",
            "/tmp/model.engine",
            "--precision-tag",
            "fp32",
            "--raw-dir",
            "/tmp/raw",
        ]
        with mock.patch.object(sys, "argv", required):
            compatible_args = bridge_v3.parse_args()
        self.assertIsNone(compatible_args.expected_output_channels)
        self.assertEqual(compatible_args.num_workers, 0)

        with mock.patch.object(
            sys,
            "argv",
            [*required, "--expected-output-channels", "64,128,256"],
        ):
            strict_args = bridge_v3.parse_args()
        self.assertEqual(strict_args.expected_output_channels, (64, 128, 256))

        with mock.patch.object(
            sys,
            "argv",
            [*required, "--num-workers", "2"],
        ):
            parallel_loader_args = bridge_v3.parse_args()
        self.assertEqual(parallel_loader_args.num_workers, 2)

        protocol = bridge_v3.protocol_payload(
            precision_tag="fp32",
            eval_range="102.4,102.4",
            full_ap_min_samples=1789,
            expected_output_channels=strict_args.expected_output_channels,
        )
        self.assertEqual(protocol["expected_output_channels"], [64, 128, 256])

    def test_summarize_error_records_reports_shape_and_nonfinite_failures(self) -> None:
        summary = bridge_v3.summarize_error_records(
            [
                {
                    "status": "compared",
                    "reference_all_finite": True,
                    "candidate_all_finite": True,
                    "diff_all_finite": True,
                    "max_abs_err": 0.1,
                    "mean_abs_err": 0.01,
                },
                {
                    "status": "shape_mismatch",
                    "reference_all_finite": True,
                    "candidate_all_finite": True,
                },
                {
                    "status": "compared",
                    "reference_all_finite": True,
                    "candidate_all_finite": False,
                    "diff_all_finite": False,
                    "max_abs_err": float("nan"),
                    "mean_abs_err": float("nan"),
                },
            ]
        )

        self.assertEqual(summary["num_records"], 3)
        self.assertEqual(summary["num_compared"], 2)
        self.assertEqual(summary["shape_mismatch_count"], 1)
        self.assertEqual(summary["nonfinite_count"], 1)
        self.assertFalse(summary["all_finite"])

    def test_engine_ap_gate_rejects_partial_or_fallback_runs(self) -> None:
        healthy_summary = {
            "num_records": 1789 * 3,
            "num_compared": 1789 * 3,
            "shape_mismatch_count": 0,
            "nonfinite_count": 0,
            "all_finite": True,
        }
        eligible, reasons = bridge_v3.engine_ap_gate(
            processed_samples=1789,
            min_samples=1789,
            fallback_samples=0,
            failed_samples=0,
            output_error_summary=healthy_summary,
        )
        self.assertTrue(eligible)
        self.assertEqual(reasons, [])

        partial_ok, partial_reasons = bridge_v3.engine_ap_gate(
            processed_samples=16,
            min_samples=1789,
            fallback_samples=0,
            failed_samples=0,
            output_error_summary={
                **healthy_summary,
                "num_records": 16 * 3,
                "num_compared": 16 * 3,
            },
        )
        self.assertFalse(partial_ok)
        self.assertIn("processed_samples_below_min", partial_reasons)

        fallback_ok, fallback_reasons = bridge_v3.engine_ap_gate(
            processed_samples=1789,
            min_samples=1789,
            fallback_samples=1,
            failed_samples=0,
            output_error_summary=healthy_summary,
        )
        self.assertFalse(fallback_ok)
        self.assertIn("fallback_samples_present", fallback_reasons)

        bad_evidence_ok, bad_evidence_reasons = bridge_v3.engine_ap_gate(
            processed_samples=1789,
            min_samples=1789,
            fallback_samples=0,
            failed_samples=0,
            output_error_summary={
                **healthy_summary,
                "num_records": 1789 * 3 - 1,
                "num_compared": 1789 * 3 - 1,
                "shape_mismatch_count": 1,
                "all_finite": False,
            },
        )
        self.assertFalse(bad_evidence_ok)
        self.assertIn("output_num_records_mismatch", bad_evidence_reasons)
        self.assertIn("output_num_compared_mismatch", bad_evidence_reasons)
        self.assertIn("output_shape_mismatch_present", bad_evidence_reasons)
        self.assertIn("output_nonfinite_present", bad_evidence_reasons)

    def test_build_report_binds_provenance_shas_and_engine_ap_claim(self) -> None:
        protocol = bridge_v3.protocol_payload(
            precision_tag="fp32",
            eval_range="102.4,102.4",
            full_ap_min_samples=1789,
            expected_output_channels=(64, 128, 256),
        )
        report = bridge_v3.build_report(
            label="pyramid",
            precision_tag="fp16",
            ckpt_dir=Path("/tmp/ckpt"),
            checkpoint_path=Path("/tmp/ckpt/net_epoch24.pth"),
            checkpoint_sha256="a" * 64,
            checkpoint_epoch=24,
            config_path=Path("/tmp/ckpt/config.yaml"),
            config_sha256="b" * 64,
            engine_path=Path("/tmp/model.engine"),
            engine_sha256="c" * 64,
            dataset_split_file=Path("/tmp/split.json"),
            dataset_split_sha256="d" * 64,
            protocol=protocol,
            processed_samples=1789,
            failed_samples=0,
            fallback_samples=0,
            ap30=0.8,
            ap50=0.7,
            ap70=0.6,
            pred_nonempty_count=1700,
            pred_total_count=4000,
            model_dtype_counts={"torch.float32": 10},
            output_error_summary={
                "num_records": 1789 * 3,
                "num_compared": 1789 * 3,
                "shape_mismatch_count": 0,
                "nonfinite_count": 0,
                "all_finite": True,
                "max_abs_err": 0.25,
            },
            elapsed_secs=12.0,
        )

        self.assertEqual(report["engine_sha256"], "c" * 64)
        self.assertEqual(report["protocol_sha256"], bridge_v3.sha256_json(protocol))
        self.assertEqual(report["dataset_split_sha256"], "d" * 64)
        self.assertTrue(report["engine_ap_claim"])

        report["output_error_summary"]["all_finite"] = False
        blocked = bridge_v3.build_report(
            label="pyramid",
            precision_tag="fp32",
            ckpt_dir=Path("/tmp/ckpt"),
            checkpoint_path=Path("/tmp/ckpt/net_epoch24.pth"),
            checkpoint_sha256="a" * 64,
            checkpoint_epoch=24,
            config_path=Path("/tmp/ckpt/config.yaml"),
            config_sha256="b" * 64,
            engine_path=Path("/tmp/model.engine"),
            engine_sha256="c" * 64,
            dataset_split_file=Path("/tmp/split.json"),
            dataset_split_sha256="d" * 64,
            protocol=protocol,
            processed_samples=1789,
            failed_samples=0,
            fallback_samples=0,
            ap30=0.8,
            ap50=0.7,
            ap70=0.6,
            pred_nonempty_count=1700,
            pred_total_count=4000,
            model_dtype_counts={"torch.float32": 10},
            output_error_summary=report["output_error_summary"],
            elapsed_secs=12.0,
        )
        self.assertFalse(blocked["engine_ap_claim"])
        self.assertIn("output_nonfinite_present", blocked["engine_ap_claim_blockers"])


if __name__ == "__main__":
    unittest.main()
