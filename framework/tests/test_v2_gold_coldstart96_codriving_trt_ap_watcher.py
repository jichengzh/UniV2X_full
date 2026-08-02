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

import stage2_v2_gold_coldstart96_codriving_trt_ap_watcher as watcher  # noqa: E402
import stage2_codriving_int8_provenance as provenance  # noqa: E402


def valid_int8_compile_summary() -> dict:
    signature = "input=2x64x256x512|weight=32x64x3x3"
    return {
        "quantization_semantics": provenance.INT8_QUANTIZATION_SEMANTICS,
        "conv_precision_plan": {"fused_conv2d": "int8"},
        "int8_signature_plan": {"fused_conv2d": signature},
        "int8_calibration": {
            "schema": provenance.INT8_CALIBRATION_SCHEMA,
            "quantization_semantics": provenance.INT8_QUANTIZATION_SEMANTICS,
            "calibration_source": "/tmp/calib.npz",
            "calibration_summary": "/tmp/calib_summary.json",
            "calibration_split": "train",
            "calibration_split_source": "/data/train.json",
            "calibration_samples": 16,
            "calibration_source_sha256": "a" * 64,
            "calibration_summary_sha256": "b" * 64,
            "spatial_features_shape": [16, 2, 64, 256, 512],
            "qmin": -127,
            "qmax": 127,
            "scales_by_signature": {
                signature: {"input_scale": 0.1, "weight_scale": 0.01},
            },
        },
    }


class V2GoldColdstart96CoDrivingTrtApWatcherTests(unittest.TestCase):
    def test_latest_bestval_ckpt_prefers_highest_epoch_number(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            for name in ("net_epoch_bestval_at2.pth", "net_epoch_bestval_at11.pth", "net_epoch_bestval_at9.pth"):
                (model_dir / name).write_text("x", encoding="utf-8")

            self.assertEqual(watcher.latest_bestval_ckpt(model_dir).name, "net_epoch_bestval_at11.pth")

    def test_trt_ap_raw_path_uses_importer_contract_tags(self) -> None:
        root = Path("/tmp/results")

        self.assertEqual(
            watcher.trt_ap_raw_path(root, "16x32x64", "fp16"),
            Path("/tmp/results/codriving_trt_hybrid_ap_raw/16x32x64/trt_fp16_final.json"),
        )
        self.assertEqual(
            watcher.trt_ap_raw_path(root, "16x32x64", "int8"),
            Path("/tmp/results/codriving_trt_hybrid_ap_raw/16x32x64/trt_int8_all_final.json"),
        )

    def test_tvm_routeb_ap_raw_path_uses_importer_contract_tags(self) -> None:
        root = Path("/tmp/results")

        self.assertEqual(
            watcher.tvm_routeb_ap_raw_path(root, "16x32x64", "mixed_top50_flops"),
            Path("/tmp/results/codriving_tvm_routeb_resnet_ap_raw/16x32x64/tvm_routeb_int8_top50_final.json"),
        )

    def test_hybrid_report_complete_requires_no_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            report = Path(tmpdir) / "report.json"
            report.write_text(
                json.dumps({"ap70": 0.3, "n_done": 20, "n_trt_path": 20, "n_fallback_path": 0}),
                encoding="utf-8",
            )

            self.assertTrue(watcher.hybrid_report_complete(report, min_samples=20))

            report.write_text(
                json.dumps({"ap70": 0.3, "n_done": 20, "n_trt_path": 19, "n_fallback_path": 1}),
                encoding="utf-8",
            )
            self.assertFalse(watcher.hybrid_report_complete(report, min_samples=20))

    def test_tvm_report_complete_requires_calibrated_int8_and_no_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            report = Path(tmpdir) / "report.json"
            report.write_text(
                json.dumps(
                    {
                        "precision": "int8",
                        "ap70": 0.0,
                        "pipeline_scope": "tvm_routeb_resnet_in_full_pytorch_eval",
                        "n_done": 20,
                        "n_tvm_path": 20,
                        "n_fallback_path": 0,
                        "compile_summary": {},
                    }
                ),
                encoding="utf-8",
            )

            self.assertFalse(watcher.tvm_report_complete(report, min_samples=20))

            report.write_text(
                json.dumps(
                    {
                        "precision": "int8",
                        "ap70": 0.2,
                        "pipeline_scope": "tvm_routeb_resnet_in_full_pytorch_eval",
                        "n_done": 20,
                        "n_tvm_path": 20,
                        "n_fallback_path": 0,
                        "compile_summary": valid_int8_compile_summary(),
                    }
                ),
                encoding="utf-8",
            )

            self.assertTrue(watcher.tvm_report_complete(report, min_samples=20))

            report.write_text(
                json.dumps(
                    {
                        "precision": "int8",
                        "ap70": 0.0,
                        "pipeline_scope": "tvm_routeb_resnet_in_full_pytorch_eval",
                        "n_done": 20,
                        "n_tvm_path": 19,
                        "n_fallback_path": 1,
                        "compile_summary": valid_int8_compile_summary(),
                    }
                ),
                encoding="utf-8",
            )
            self.assertFalse(watcher.tvm_report_complete(report, min_samples=20))

    def test_run_width_uses_conda_python_for_opencood_fp_ap_and_onnx_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            out_root = root / "out"
            results_root = root / "results"
            width = "24x32x96"
            model_dir = out_root / width
            model_dir.mkdir(parents=True)
            (model_dir / "config.yaml").write_text("train_params: {}\n", encoding="utf-8")
            (model_dir / "net_epoch_bestval_at9.pth").write_text("ckpt", encoding="utf-8")
            (model_dir / "collab_calib_train_final_n16_float16.npz").write_text("calib", encoding="utf-8")
            (model_dir / "collab_calib_train_final_n16_float16_summary.json").write_text(
                "{}",
                encoding="utf-8",
            )
            engine_root = out_root / "trt_collab_engines_final" / width
            engine_root.mkdir(parents=True)
            for mode in ("fp16", "int8"):
                (engine_root / f"collab_{width}_{mode}.engine").write_text("engine", encoding="utf-8")
                (model_dir / f"trt_hybrid_ap_final_{mode}.json").write_text(
                    json.dumps({"ap70": 0.1, "n_done": 1, "n_trt_path": 1, "n_fallback_path": 0}),
                    encoding="utf-8",
                )
            for mode in ("fp16", "int8_all", "mixed_top25_flops", "mixed_top50_flops"):
                precision = "fp16" if mode == "fp16" else "int8" if mode == "int8_all" else "mixed"
                (model_dir / f"tvm_resnet_ap_final_{mode}.json").write_text(
                    json.dumps(
                        {
                            "precision": precision,
                            "ap70": 0.1,
                            "pipeline_scope": "tvm_routeb_resnet_in_full_pytorch_eval",
                            "n_done": 1,
                            "n_tvm_path": 1,
                            "n_fallback_path": 0,
                            "compile_summary": (
                                {}
                                if precision == "fp16"
                                else valid_int8_compile_summary()
                            ),
                        }
                    ),
                    encoding="utf-8",
                )

            args = SimpleNamespace(
                out_root=out_root,
                results_root=results_root,
                repo_root=root / "repo",
                local_v2x=root / "local_v2x",
                gpu="4",
                conda_python="/conda/bin/python",
                system_python="/usr/bin/python3",
                calib_samples=16,
                num_workers=0,
                progress_every=100,
                workspace_mb=4096,
                ap_samples=1,
            )
            recorded: list[tuple[list[str], dict[str, str]]] = []

            def fake_run_logged(cmd, *, log, cwd, env):
                recorded.append((cmd, env))
                if log.name == "fp_ap.log":
                    (model_dir / "eval_intermediate_epoch9.yaml").write_text(
                        "ap30: 0.1\nap_50: 0.1\nap_70: 0.1\n",
                        encoding="utf-8",
                    )
                if log.name == "onnx_export.log":
                    (model_dir / f"collab_{width}_final_fp32.onnx").write_text("onnx", encoding="utf-8")

            with mock.patch.object(watcher, "run_logged", side_effect=fake_run_logged):
                watcher.run_width(args, width)

        self.assertGreaterEqual(len(recorded), 2)
        self.assertEqual(recorded[0][0][0], "/conda/bin/python")
        self.assertEqual(recorded[1][0][0], "/conda/bin/python")
        self.assertNotIn("/exdata/jichengzhi/tvm310/lib/python3.10/site-packages", recorded[0][1]["PYTHONPATH"])
        self.assertNotIn("/exdata/jichengzhi/tvm310/lib/python3.10/site-packages", recorded[1][1]["PYTHONPATH"])
        self.assertEqual(recorded[1][1]["CODRIVING_EXPORT_DISABLE_LEGACY_T2LIB"], "1")


if __name__ == "__main__":
    unittest.main()
