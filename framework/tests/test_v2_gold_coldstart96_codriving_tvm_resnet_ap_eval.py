from __future__ import annotations

import json
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import stage2_v2_gold_coldstart96_codriving_tvm_resnet_ap_eval as tvm_ap  # noqa: E402
import stage2_codriving_whole_engine_tc_v1 as whole_engine  # noqa: E402
import stage2_codriving_int8_provenance as provenance  # noqa: E402


class V2GoldColdstart96CoDrivingTvmResnetApEvalTests(unittest.TestCase):
    def test_routeb_mode_maps_to_precision_policy_and_tag(self) -> None:
        self.assertEqual(
            tvm_ap.routeb_mode_spec("fp16"),
            {"precision": "fp16", "mixed_policy": "none", "tag": "tvm_routeb_fp16"},
        )
        self.assertEqual(
            tvm_ap.routeb_mode_spec("int8_all"),
            {"precision": "int8", "mixed_policy": "all", "tag": "tvm_routeb_int8_all"},
        )
        self.assertEqual(
            tvm_ap.routeb_mode_spec("mixed_top25_flops"),
            {"precision": "mixed", "mixed_policy": "top25_flops", "tag": "tvm_routeb_int8_top25"},
        )

    def test_build_report_records_resnet_scope_not_full_collab(self) -> None:
        report = tvm_ap.build_report(
            width="16x32x64",
            mode="int8_all",
            model_dir=Path("/tmp/model"),
            onnx=Path("/tmp/resnet.onnx"),
            onnx_sha256="c" * 64,
            ap30=0.1,
            ap50=0.2,
            ap70=0.3,
            n_done=20,
            n_tvm_path=20,
            n_fallback_path=0,
            n_skipped=1,
            elapsed_secs=2.5,
            compile_summary={"n_conv_replaced_tensorized": 12},
        )

        self.assertEqual(report["schema"], "v2_gold_coldstart_96_codriving_tvm_resnet_ap_eval_v1")
        self.assertEqual(report["tag"], "tvm_routeb_int8_all")
        self.assertEqual(report["pipeline_scope"], "tvm_routeb_resnet_in_full_pytorch_eval")
        self.assertEqual(report["ap70"], 0.3)
        self.assertEqual(report["onnx_sha256"], "c" * 64)
        self.assertEqual(report["n_tvm_path"], 20)
        self.assertEqual(report["n_fallback_path"], 0)

    def test_report_complete_requires_no_fallback_and_min_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.json"
            path.write_text(
                json.dumps(
                    {
                        "precision": "fp16",
                        "pipeline_scope": tvm_ap.PIPELINE_SCOPE,
                        "ap70": 0.2,
                        "n_done": 8,
                        "n_tvm_path": 8,
                        "n_fallback_path": 0,
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(tvm_ap.report_complete(path, min_samples=8))

            path.write_text(
                json.dumps(
                    {
                        "precision": "fp16",
                        "pipeline_scope": tvm_ap.PIPELINE_SCOPE,
                        "ap70": 0.2,
                        "n_done": 8,
                        "n_tvm_path": 7,
                        "n_fallback_path": 1,
                    }
                ),
                encoding="utf-8",
            )
            self.assertFalse(tvm_ap.report_complete(path, min_samples=8))

    def test_report_complete_rejects_legacy_dummy_scale_int8_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.json"
            path.write_text(
                json.dumps(
                    {
                        "precision": "int8",
                        "ap70": 0.0,
                        "n_done": 8,
                        "n_tvm_path": 8,
                        "n_fallback_path": 0,
                        "compile_summary": {},
                    }
                ),
                encoding="utf-8",
            )

            self.assertFalse(tvm_ap.report_complete(path, min_samples=8))

    def test_build_int8_scale_manifest_aggregates_real_calibration_by_signature(self) -> None:
        layer_records = [
            {
                "input_shape": [2, 32, 128, 256],
                "weight_shape": [32, 32, 3, 3],
                "input_absmax": 12.7,
            },
            {
                "input_shape": [2, 32, 128, 256],
                "weight_shape": [32, 32, 3, 3],
                "input_absmax": 25.4,
            },
        ]
        weight_absmax = {(32, 32, 3, 3): 0.508}

        manifest = tvm_ap.build_int8_scale_manifest(
            layer_records,
            weight_absmax,
            calibration_source=Path("/tmp/calib.npz"),
            calibration_summary=Path("/tmp/calib_summary.json"),
            calibration_split="train",
            calibration_split_source=Path("/data/train.json"),
            calibration_source_sha256="a" * 64,
            calibration_summary_sha256="b" * 64,
            calibration_samples=16,
        )

        key = tvm_ap.conv_signature((2, 32, 128, 256), (32, 32, 3, 3))
        self.assertEqual(manifest["quantization_semantics"], "symmetric_absmax_int8_dequant_fp32")
        self.assertEqual(manifest["calibration_samples"], 16)
        self.assertEqual(manifest["calibration_source"], "/tmp/calib.npz")
        self.assertEqual(manifest["calibration_split"], "train")
        self.assertAlmostEqual(manifest["scales_by_signature"][key]["input_scale"], 0.2)
        self.assertAlmostEqual(manifest["scales_by_signature"][key]["weight_scale"], 0.004)

    def test_int8_same_signature_rewrite_requires_positive_calibration_scales(self) -> None:
        spec = {
            "input_nchw": (2, 32, 16, 16),
            "weight_oihw": (32, 32, 3, 3),
            "strides": (1, 1),
            "padding": (1, 1, 1, 1),
            "relu": True,
            "residual": False,
        }

        with self.assertRaisesRegex(ValueError, "requires positive input_scale and weight_scale"):
            whole_engine.make_std_conv_im2col_same_signature_primfunc(
                spec,
                "fused_conv2d",
                "int32",
                input_scale=None,
                weight_scale=None,
            )

    def test_round_te_expr_uses_te_round_on_h800_tvm_fork(self) -> None:
        fake_tvm = SimpleNamespace()
        fake_te = SimpleNamespace(round=lambda value: ("te.round", value))

        self.assertEqual(
            whole_engine.round_te_expr(fake_tvm, fake_te, "value"),
            ("te.round", "value"),
        )

    def test_validate_manifest_rejects_truthy_stub(self) -> None:
        with self.assertRaisesRegex(ValueError, "schema"):
            provenance.validate_int8_calibration_manifest({"calibration_samples": 16})

    def test_validate_manifest_rejects_validation_split(self) -> None:
        manifest = {
            "schema": provenance.INT8_CALIBRATION_SCHEMA,
            "quantization_semantics": provenance.INT8_QUANTIZATION_SEMANTICS,
            "calibration_source": "/tmp/calib.npz",
            "calibration_summary": "/tmp/calib_summary.json",
            "calibration_split": "val",
            "calibration_samples": 16,
            "spatial_features_shape": [16, 2, 64, 256, 512],
            "qmin": -127,
            "qmax": 127,
            "scales_by_signature": {
                "sig": {"input_scale": 0.1, "weight_scale": 0.01},
            },
        }

        with self.assertRaisesRegex(ValueError, "calibration_split"):
            provenance.validate_int8_calibration_manifest(manifest)

    def test_validate_manifest_rejects_missing_required_signature(self) -> None:
        manifest = {
            "schema": provenance.INT8_CALIBRATION_SCHEMA,
            "quantization_semantics": provenance.INT8_QUANTIZATION_SEMANTICS,
            "calibration_source": "/tmp/calib.npz",
            "calibration_summary": "/tmp/calib_summary.json",
            "calibration_split": "train",
            "calibration_split_source": "/data/train.json",
            "calibration_source_sha256": "a" * 64,
            "calibration_summary_sha256": "b" * 64,
            "calibration_samples": 16,
            "spatial_features_shape": [16, 2, 64, 256, 512],
            "qmin": -127,
            "qmax": 127,
            "scales_by_signature": {
                "present": {"input_scale": 0.1, "weight_scale": 0.01},
            },
        }

        with self.assertRaisesRegex(ValueError, "missing calibrated signatures"):
            provenance.validate_int8_calibration_manifest(manifest, required_signatures={"missing"})

    def test_validate_calibration_tensor_shape_rejects_eval_sized_npz(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly 16"):
            provenance.validate_calibration_tensor_shape((1789, 2, 64, 256, 512))

    def test_validate_manifest_rejects_fractional_integer_fields(self) -> None:
        manifest = {
            "schema": provenance.INT8_CALIBRATION_SCHEMA,
            "quantization_semantics": provenance.INT8_QUANTIZATION_SEMANTICS,
            "calibration_source": "/tmp/calib.npz",
            "calibration_summary": "/tmp/calib_summary.json",
            "calibration_split": "train",
            "calibration_samples": 16.9,
            "spatial_features_shape": [16, 2, 64, 256, 512],
            "qmin": -127.9,
            "qmax": 127.9,
            "scales_by_signature": {
                "sig": {"input_scale": 0.1, "weight_scale": 0.01},
            },
        }

        with self.assertRaisesRegex(ValueError, "calibration_samples"):
            provenance.validate_int8_calibration_manifest(manifest)

    def test_calibration_summary_sample_count_requires_exact_json_integer(self) -> None:
        self.assertEqual(tvm_ap.validate_calibration_summary_sample_count(16), 16)
        for invalid in (16.0, 16.9, "16", True):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "collected_samples"):
                    tvm_ap.validate_calibration_summary_sample_count(invalid)

    def test_validate_manifest_rejects_missing_artifact_digests(self) -> None:
        manifest = {
            "schema": provenance.INT8_CALIBRATION_SCHEMA,
            "quantization_semantics": provenance.INT8_QUANTIZATION_SEMANTICS,
            "calibration_source": "/tmp/calib.npz",
            "calibration_summary": "/tmp/calib_summary.json",
            "calibration_split": "train",
            "calibration_split_source": "/data/train.json",
            "calibration_samples": 16,
            "spatial_features_shape": [16, 2, 64, 256, 512],
            "qmin": -127,
            "qmax": 127,
            "scales_by_signature": {
                "sig": {"input_scale": 0.1, "weight_scale": 0.01},
            },
        }

        with self.assertRaisesRegex(ValueError, "calibration_source_sha256"):
            provenance.validate_int8_calibration_manifest(manifest)

    def test_export_resnet_onnx_uses_backbone_parameter_device_for_dummy(self) -> None:
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("torch is not installed in this Python environment")

        class FakeResnet(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.empty(1, device="meta"))

            def forward(self, spatial_features):
                self.seen_device = spatial_features.device
                return (spatial_features, spatial_features, spatial_features)

        class FakeBackbone(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.resnet = FakeResnet()

        model = SimpleNamespace(backbone=FakeBackbone())

        def fake_export(wrapper, dummy, *_args, **_kwargs):
            wrapper(dummy)

        with tempfile.TemporaryDirectory() as tmpdir, mock.patch("torch.onnx.export", side_effect=fake_export):
            tvm_ap.export_resnet_onnx(model, Path(tmpdir) / "resnet.onnx")

        self.assertEqual(model.backbone.resnet.seen_device.type, "meta")


if __name__ == "__main__":
    unittest.main()
