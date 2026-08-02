from __future__ import annotations

import argparse
import json
import os
import tempfile
import unittest
from pathlib import Path

from scripts import stage2_original60_fp16_rewritten_ap_worker as worker


class Stage2Original60Fp16RewrittenApWorkerTest(unittest.TestCase):
    def _args(self, root: Path) -> argparse.Namespace:
        return argparse.Namespace(
            gpu_id=2,
            labels="lhc_07",
            queue=str(root / "queue.jsonl"),
            raw_root=str(root / "raw"),
            ckpt_root=str(root / "ckpts"),
            rows_out=str(root / "rows.jsonl"),
            env_python="/env/python",
            tvm_python="/tvm/python",
            tvm_ld_library_path="/tvm/cuda:/tvm/lib",
            heal_root="/heal",
            master_port_base=29830,
            poll_seconds=1,
            min_epoch=25,
            num_samples=1,
            full_reps=30,
            input_shape="2,64,256,256",
            mode="smoke",
        )

    def test_builds_rewritten_resource_commands_for_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self._args(root)
            raw_dir = root / "raw/lhc_07_gpu2"
            ckpt_dir = root / "ckpts/Pyramid_DAIR_m1_stage2_ap_lhc_07_2026_06_28"

            export_cmd, onnx_path = worker.build_export_command(args, "lhc_07", ckpt_dir, raw_dir)
            self.assertIn("stage2_h800_export_checkpoint_multiscale_onnx.py", " ".join(export_cmd))
            self.assertIn("--input-shape", export_cmd)
            self.assertEqual(export_cmd[export_cmd.index("--input-shape") + 1], "2,64,256,256")
            self.assertEqual(onnx_path.name, "lhc_07_apshape_multiscale.onnx")

            rewrite_cmd, rewrite_report = worker.build_rewrite_command(args, "lhc_07", onnx_path, raw_dir)
            self.assertEqual(rewrite_cmd[0], "/tvm/python")
            self.assertIn("full-engine-group-conv-rewrite", rewrite_cmd)
            self.assertIn("--cast-fp16-source", rewrite_cmd)
            self.assertEqual(rewrite_report.name, "fp16_lhc_07_full_engine_group_conv_rewrite_latest.json")

            bridge_cmd, bridge_report = worker.build_bridge_command(args, "lhc_07", ckpt_dir, rewrite_report, raw_dir)
            self.assertIn("stage2_h800_fp16_rewritten_activation_bridge.py", " ".join(bridge_cmd))
            self.assertIn("--persistent-worker", bridge_cmd)
            self.assertIn("--num-samples", bridge_cmd)
            self.assertEqual(bridge_cmd[bridge_cmd.index("--num-samples") + 1], "1")
            self.assertEqual(bridge_report.name, "fp16_lhc_07_rewritten_ap_smoke_latest.json")

    def test_rewrite_gate_requires_tensorcore_and_export_library(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "rewrite.json"
            report.write_text(
                json.dumps(
                    {
                        "status": "success",
                        "rewritten_full_engine": {
                            "tensorcore_gate": True,
                            "export_library": {"status": "success", "path": "/tmp/engine.so"},
                        },
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(worker.assert_rewrite_gate(report)["status"], "success")

            report.write_text(
                json.dumps({"status": "success", "rewritten_full_engine": {"tensorcore_gate": False}}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "tensorcore_gate"):
                worker.assert_rewrite_gate(report)

    def test_bridge_gate_requires_processed_smoke_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "bridge.json"
            report.write_text(
                json.dumps({"status": "success", "processed_samples": 1, "smoke_gate_passed": True}),
                encoding="utf-8",
            )
            self.assertEqual(worker.assert_bridge_gate(report)["processed_samples"], 1)

            report.write_text(
                json.dumps({"status": "success", "processed_samples": 1, "smoke_gate_passed": False}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "smoke gate"):
                worker.assert_bridge_gate(report)

    def test_completion_jobs_keeps_fp16_even_when_old_ap_status_is_measured(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            queue = Path(tmp) / "queue.jsonl"
            queue.write_text(
                json.dumps({"precision": "fp16", "label": "frontier_01", "ap_status": "measured", "width": [24, 64, 128]})
                + "\n"
                + json.dumps({"precision": "int8", "label": "frontier_01", "width": [24, 64, 128]})
                + "\n",
                encoding="utf-8",
            )
            jobs = worker.completion_jobs(queue)
            self.assertEqual(list(jobs), ["frontier_01"])

    def test_worker_env_does_not_mask_physical_gpu_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = self._args(Path(tmp))
            old_value = os.environ.get("CUDA_VISIBLE_DEVICES")
            os.environ["CUDA_VISIBLE_DEVICES"] = "7"
            try:
                env = worker.build_env(args)
            finally:
                if old_value is None:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = old_value
            self.assertNotIn("CUDA_VISIBLE_DEVICES", env)
            self.assertTrue(env["PYTHONPATH"].startswith("/heal"))

    def test_tvm_env_adds_ld_library_path_without_gpu_mask(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = self._args(Path(tmp))
            env = worker.build_tvm_env(args)
            self.assertNotIn("CUDA_VISIBLE_DEVICES", env)
            self.assertTrue(env["LD_LIBRARY_PATH"].startswith("/tvm/cuda:/tvm/lib"))


if __name__ == "__main__":
    unittest.main()
