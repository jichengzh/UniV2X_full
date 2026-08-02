from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import stage2_v2_gold_coldstart96_codriving_fp_ap_eval_queue as launcher  # noqa: E402


class V2GoldColdstart96CoDrivingFpApEvalQueueTests(unittest.TestCase):
    def test_build_jobs_maps_width_to_training_log_and_raw_ap_dir(self) -> None:
        jobs = launcher.build_jobs(["16x32x64", "24x32x96"], eval_gpu=4)

        self.assertEqual(len(jobs), 2)
        self.assertEqual(jobs[0]["width"], "16x32x64")
        self.assertEqual(jobs[0]["train_gpu"], 5)
        self.assertTrue(jobs[0]["train_log"].endswith("/logs/train_16x32x64_gpu5.log"))
        self.assertTrue(jobs[0]["model_dir"].endswith("/16x32x64"))
        self.assertTrue(jobs[0]["raw_ap_dir"].endswith("/codriving_ap_raw/16x32x64"))
        self.assertEqual(jobs[1]["train_gpu"], 6)

    def test_command_waits_for_training_and_uses_python3_inference(self) -> None:
        job = launcher.build_jobs(["32x32x128"], eval_gpu=4)[0]

        cmd = launcher.command_for(job)

        self.assertIn("grep -q 'Training Finished'", cmd)
        self.assertIn("CUDA_VISIBLE_DEVICES=4", cmd)
        self.assertIn("python3 -u opencood/tools/inference.py", cmd)
        self.assertIn("--model_dir", cmd)
        self.assertIn(job["model_dir"], cmd)
        self.assertIn("cp -f", cmd)
        self.assertIn("eval_intermediate_epoch*.yaml", cmd)
        self.assertIn(job["raw_ap_dir"], cmd)


if __name__ == "__main__":
    unittest.main()
