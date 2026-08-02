from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "stage35_pyramid_source_queue_v1.sh"


class Stage35PyramidSourceQueueV1Tests(unittest.TestCase):
    def test_dry_run_binds_four_explicit_checkpoints_and_absolute_outputs(self) -> None:
        completed = subprocess.run(
            ["bash", str(SCRIPT), "--dry-run", "--gpu", "7", "--wait-pid", "0"],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.count("stage2_h800_export_checkpoint_multiscale_onnx.py"), 4)
        self.assertEqual(completed.stdout.count("stage3_pyramid_calibration_export_v3.py"), 4)
        self.assertEqual(completed.stdout.count("--checkpoint-path"), 8)
        self.assertEqual(completed.stdout.count(r"--input-shape 2\,64\,128\,256"), 4)
        self.assertEqual(completed.stdout.count("stage35_prepare_pyramid_trt_calibration_v1.py"), 4)
        self.assertIn("Pyramid_DAIR_m1_stage2_ap_frontier_01_2026_06_28/net_epoch31.pth", completed.stdout)
        self.assertIn("Pyramid_DAIR_m1_base_2023_08_14_11_42_29/net_epoch_bestval_at23.pth", completed.stdout)
        self.assertIn("/home/jichengzhi/V2X/results/stage35_gold32_supplement_v1_20260713", completed.stdout)


if __name__ == "__main__":
    unittest.main()
