import argparse
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import stage2_h800_fp16_rewritten_activation_bridge as bridge  # noqa: E402


class Stage2H800Fp16RewrittenActivationBridgeTests(unittest.TestCase):
    def test_build_worker_request_uses_full_batch_activation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            request = bridge.build_fp16_worker_request(
                label="lhc_07",
                run_id="smoke",
                artifact_path=root / "rewritten_full_engine.so",
                activation_npy_path=root / "activation_float16.npy",
                output_dir=root / "call_000",
                gpu=0,
            )
        self.assertEqual(request["label"], "lhc_07")
        self.assertEqual(request["run_id"], "smoke")
        self.assertEqual(request["activation_dtype"], "float16")
        self.assertEqual(request["output_names"], ["output0", "output1", "output2"])
        self.assertEqual(request["expected_output_shapes"]["output2"], [2, 128, 32, 64])

    def test_build_worker_request_accepts_float32_activation_dtype(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            request = bridge.build_fp16_worker_request(
                label="lhc_07",
                run_id="smoke",
                artifact_path=root / "rewritten_full_engine.so",
                activation_npy_path=root / "activation_float32.npy",
                output_dir=root / "call_000",
                gpu=0,
                activation_dtype="float32",
            )
        self.assertEqual(request["activation_dtype"], "float32")

    def test_expected_output_shapes_from_rewrite_report_uses_rewritten_shapes(self):
        report = {
            "rewritten_full_engine": {
                "output_compare": [
                    {"output_index": 0, "rewritten_shape": [2, 24, 256, 256]},
                    {"output_index": 1, "rewritten_shape": [2, 48, 128, 128]},
                    {"output_index": 2, "rewritten_shape": [2, 128, 64, 64]},
                ]
            }
        }
        shapes = bridge.expected_output_shapes_from_rewrite_report(report)
        self.assertEqual(shapes["output0"], [2, 24, 256, 256])
        self.assertEqual(shapes["output2"], [2, 128, 64, 64])

    def test_expected_output_shapes_from_reference_outputs_uses_dynamic_width_32_32_128(self):
        import numpy as np

        outputs = (
            np.zeros((1, 32, 128, 256), dtype=np.float32),
            np.zeros((1, 32, 64, 128), dtype=np.float32),
            np.zeros((1, 128, 32, 64), dtype=np.float32),
        )
        shapes = bridge.expected_output_shapes_from_reference_outputs(outputs, engine_batch=2)
        self.assertEqual(shapes["output0"], [2, 32, 128, 256])
        self.assertEqual(shapes["output1"], [2, 32, 64, 128])
        self.assertEqual(shapes["output2"], [2, 128, 32, 64])

    def test_load_configured_expected_output_shapes_uses_explicit_rewrite_report(self):
        import json

        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "rewrite_report.json"
            report_path.write_text(
                json.dumps(
                    {
                        "rewritten_full_engine": {
                            "output_compare": [
                                {"output_index": 0, "rewritten_shape": [2, 32, 128, 256]},
                                {"output_index": 1, "rewritten_shape": [2, 32, 64, 128]},
                                {"output_index": 2, "rewritten_shape": [2, 128, 32, 64]},
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )
            args = argparse.Namespace(rewrite_report=str(report_path), rewrite_report_explicit=True)
            shapes, source = bridge.load_configured_expected_output_shapes(args)
        self.assertEqual(source, "explicit_rewrite_report")
        self.assertEqual(shapes["output1"], [2, 32, 64, 128])

    def test_load_configured_expected_output_shapes_ignores_nonexplicit_rewrite_report(self):
        import json

        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "rewrite_report.json"
            report_path.write_text(
                json.dumps(
                    {
                        "rewritten_full_engine": {
                            "output_compare": [
                                {"output_index": 0, "rewritten_shape": [2, 24, 128, 256]},
                                {"output_index": 1, "rewritten_shape": [2, 48, 64, 128]},
                                {"output_index": 2, "rewritten_shape": [2, 128, 32, 64]},
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )
            args = argparse.Namespace(rewrite_report=str(report_path), rewrite_report_explicit=False)
            shapes, source = bridge.load_configured_expected_output_shapes(args)
        self.assertIsNone(shapes)
        self.assertEqual(source, "reference_outputs_dynamic")

    def test_attach_expected_output_shape_metadata_records_source_in_report(self):
        report = {"status": "success"}
        updated = bridge.attach_expected_output_shape_metadata(
            report,
            source="reference_outputs_dynamic",
            shapes={"output0": [2, 32, 128, 256]},
        )
        self.assertEqual(updated["expected_output_shapes_source"], "reference_outputs_dynamic")
        self.assertEqual(updated["expected_output_shapes"]["output0"], [2, 32, 128, 256])

    def test_tvm_env_prepends_repo_and_tvm_ld_path(self):
        args = argparse.Namespace(gpu_id=4, tvm_ld_library_path="/tmp/tvm_lib")
        env = bridge._tvm_env(args)
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "4")
        self.assertTrue(env["PYTHONPATH"].startswith(str(REPO_ROOT)))
        self.assertTrue(env["LD_LIBRARY_PATH"].startswith("/tmp/tvm_lib"))

    def test_prepare_worker_activation_pads_single_batch_to_engine_batch_default_float16(self):
        import numpy as np

        activation = np.ones((1, 64, 128, 256), dtype=np.float32)
        padded, meta = bridge.prepare_worker_activation(activation, engine_batch=2)
        self.assertEqual(list(padded.shape), [2, 64, 128, 256])
        self.assertEqual(str(padded.dtype), "float16")
        self.assertEqual(meta["original_batch"], 1)
        self.assertEqual(meta["engine_batch"], 2)
        self.assertEqual(meta["padding_batch"], 1)
        self.assertTrue(np.all(padded[1] == 0))

    def test_prepare_worker_activation_preserves_requested_float32_dtype(self):
        import numpy as np

        activation = np.ones((1, 64, 128, 256), dtype=np.float16)
        padded, meta = bridge.prepare_worker_activation(
            activation,
            engine_batch=2,
            activation_dtype="float32",
        )
        self.assertEqual(list(padded.shape), [2, 64, 128, 256])
        self.assertEqual(str(padded.dtype), "float32")
        self.assertEqual(meta["requested_activation_dtype"], "float32")
        self.assertTrue(np.all(padded[1] == 0))

    def test_slice_worker_output_to_original_batch(self):
        import numpy as np

        output = np.ones((2, 24, 128, 256), dtype=np.float16)
        sliced = bridge.slice_worker_output_to_original_batch(output, original_batch=1)
        self.assertEqual(list(sliced.shape), [1, 24, 128, 256])

    def test_resolve_paths_makes_rewrite_report_absolute(self):
        args = argparse.Namespace(
            rewrite_report="multi_agent/report.json",
            artifact_path="/tmp/rewrite.so",
            worker_script="scripts/stage2_fp16_tvm_worker.py",
            ckpt_dir="checkpoints/lhc_07",
            raw_dir="raw/lhc_07",
            heal_root="/home/jichengzhi/heal_research/HEAL",
            export_report_json="exports/report.json",
        )
        bridge._resolve_paths(args)
        self.assertTrue(Path(args.rewrite_report).is_absolute())
        self.assertTrue(Path(args.worker_script).is_absolute())
        self.assertTrue(Path(args.raw_dir).is_absolute())

    def test_smoke_gate_status_returns_builtin_bool(self):
        value = bridge.smoke_gate_status(pred_nonempty_count=1, ap_values=[0.7, 0.6, 0.5])
        self.assertIs(type(value), bool)
        self.assertTrue(value)

    def test_streaming_tensor_distribution_accumulates_all_values(self):
        import numpy as np

        stats = bridge.StreamingTensorDistribution("pred_score")
        stats.update(np.array([0.2, 0.8], dtype=np.float32))
        stats.update(np.array([0.5], dtype=np.float32))

        summary = stats.to_summary()

        self.assertEqual(summary["tensor_name"], "pred_score")
        self.assertEqual(summary["samples"], 2)
        self.assertEqual(summary["size"], 3)
        self.assertAlmostEqual(summary["min"], 0.2, places=6)
        self.assertAlmostEqual(summary["max"], 0.8, places=6)
        self.assertAlmostEqual(summary["mean"], 0.5, places=6)
        self.assertAlmostEqual(summary["std"], float(np.std([0.2, 0.8, 0.5])), places=6)

    def test_full_prediction_distribution_uses_all_samples_and_tensor_stats(self):
        import numpy as np

        aggregate = bridge.FullPostprocessDistribution()
        aggregate.update(
            pred_count=2,
            pred_score=np.array([0.3, 0.7], dtype=np.float32),
            pred_box_tensor=np.ones((2, 8, 3), dtype=np.float32),
            gt_box_tensor=np.zeros((1, 8, 3), dtype=np.float32),
        )
        aggregate.update(
            pred_count=0,
            pred_score=np.empty((0,), dtype=np.float32),
            pred_box_tensor=np.empty((0, 8, 3), dtype=np.float32),
            gt_box_tensor=np.ones((2, 8, 3), dtype=np.float32) * 2,
        )

        summary = aggregate.to_summary()

        self.assertEqual(summary["prediction_distribution"]["samples"], 2)
        self.assertEqual(summary["prediction_distribution"]["nonempty"], 1)
        self.assertEqual(summary["prediction_distribution"]["total_predictions"], 2)
        self.assertEqual(summary["pred_score"]["size"], 2)
        self.assertEqual(summary["pred_box_tensor"]["size"], 48)
        self.assertEqual(summary["gt_box_tensor"]["size"], 72)

    def test_parse_args_defaults_to_full_validation(self):
        argv = [
            "prog",
            "--ckpt-dir",
            "/tmp/ckpt",
            "--raw-dir",
            "/tmp/raw",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = bridge.parse_args()
        self.assertIsNone(args.num_samples)
        self.assertEqual(args.artifact_input_dtype, "float16")

    def test_parse_args_accepts_float32_artifact_input_dtype(self):
        argv = [
            "prog",
            "--ckpt-dir",
            "/tmp/ckpt",
            "--raw-dir",
            "/tmp/raw",
            "--artifact-input-dtype",
            "float32",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = bridge.parse_args()
        self.assertEqual(args.artifact_input_dtype, "float32")

    def test_resolve_checkpoint_prefers_explicit_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint_dir = root / "checkpoints"
            checkpoint_dir.mkdir()
            (checkpoint_dir / "net_epoch1.pth").write_bytes(b"epoch1")
            epoch23 = checkpoint_dir / "net_epoch_bestval_at23.pth"
            epoch23.write_bytes(b"epoch23")

            selected = bridge.resolve_checkpoint_path(
                checkpoint_dir,
                epoch23,
            )

        self.assertEqual(selected, epoch23)

    def test_resolve_checkpoint_rejects_missing_explicit_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint_dir = root / "checkpoints"
            checkpoint_dir.mkdir()
            with self.assertRaises(FileNotFoundError):
                bridge.resolve_checkpoint_path(
                    checkpoint_dir,
                    checkpoint_dir / "net_epoch_bestval_at23.pth",
                )

    def test_parse_args_accepts_explicit_checkpoint_path(self):
        argv = [
            "prog",
            "--ckpt-dir",
            "/tmp/ckpt",
            "--checkpoint-path",
            "/tmp/ckpt/net_epoch_bestval_at23.pth",
            "--raw-dir",
            "/tmp/raw",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = bridge.parse_args()
        self.assertEqual(
            args.checkpoint_path,
            "/tmp/ckpt/net_epoch_bestval_at23.pth",
        )


if __name__ == "__main__":
    unittest.main()
