from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/stage35_codriving_source_queue_v1.sh"


class Stage35CodrivingSourceQueueV1Tests(unittest.TestCase):
    def test_dry_run_has_four_checkpoint_consistent_source_jobs_without_mixed_policy(self) -> None:
        completed = subprocess.run(
            ["bash", str(SCRIPT), "--dry-run", "--gpu", "7", "--wait-pid", "0"],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.count("stage2_v2_gold_coldstart96_codriving_calib_export.py"), 4)
        self.assertEqual(completed.stdout.count("stage2_v2_gold_coldstart96_codriving_tvm_resnet_ap_eval.py"), 4)
        self.assertEqual(completed.stdout.count("--force-fallback"), 4)
        self.assertNotIn("mixed", completed.stdout.lower())
        self.assertIn("resnet_multiscale_40x80x160_final_fp32.onnx", completed.stdout)


if __name__ == "__main__":
    unittest.main()
