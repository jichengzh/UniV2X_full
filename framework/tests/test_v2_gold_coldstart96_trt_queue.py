from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import stage2_v2_gold_coldstart96_launch_trt_queue as trt_queue  # noqa: E402


class V2GoldColdstart96TrtQueueTests(unittest.TestCase):
    def test_build_jobs_emits_fp16_and_int8_files_importer_expects(self) -> None:
        jobs = trt_queue.build_jobs(["24x64x128", "40x64x128"], gpu=6, warmup=20, iters=300, repeat=5, energy_secs=5.0)

        self.assertEqual(len(jobs), 4)
        self.assertEqual([job["precision"] for job in jobs], ["fp16", "int8", "fp16", "int8"])
        self.assertEqual(
            [Path(job["out_json"]).name for job in jobs],
            [
                "codriving_trt_24x64x128_fp16_20260708.json",
                "codriving_trt_24x64x128_int8_20260708.json",
                "codriving_trt_40x64x128_fp16_20260708.json",
                "codriving_trt_40x64x128_int8_20260708.json",
            ],
        )
        self.assertTrue(jobs[0]["onnx"].endswith("/24x64x128/backbone_only.onnx"))
        self.assertEqual(str(trt_queue.CALIB_DIR), jobs[1]["calib_dir"])

    def test_command_for_int8_uses_existing_trt_profiler_and_calib_dir(self) -> None:
        job = trt_queue.build_jobs(["24x64x128"], gpu=5, warmup=20, iters=300, repeat=5, energy_secs=5.0)[1]

        cmd = trt_queue.command_for(job)

        self.assertIn("framework/trt_baseline/trt_profile_v1.py", cmd)
        self.assertIn("--precision", cmd)
        self.assertIn("int8", cmd)
        self.assertIn("--calib-dir", cmd)
        self.assertIn(job["calib_dir"], cmd)
        self.assertIn("--out", cmd)
        self.assertIn(job["out_json"], cmd)


if __name__ == "__main__":
    unittest.main()
