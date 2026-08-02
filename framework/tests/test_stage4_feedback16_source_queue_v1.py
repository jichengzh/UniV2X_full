from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/stage4_feedback16_source_queue_v1.sh"


class Stage4Feedback16SourceQueueTests(unittest.TestCase):
    def test_dry_run_reads_candidate_plan_and_emits_four_group_source_commands(self) -> None:
        completed = subprocess.run(
            ["bash", str(SCRIPT), "--gpu", "6", "--dry-run"],
            cwd=REPO,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(len([line for line in completed.stdout.splitlines() if line.strip()]), 10)
        self.assertEqual(completed.stdout.count("stage2_h800_export_checkpoint_multiscale_onnx.py"), 2)
        self.assertEqual(completed.stdout.count("stage3_pyramid_calibration_export_v3.py"), 2)
        self.assertEqual(completed.stdout.count("stage35_prepare_pyramid_trt_calibration_v1.py"), 2)
        self.assertEqual(completed.stdout.count("stage2_v2_gold_coldstart96_codriving_calib_export.py"), 2)
        self.assertEqual(completed.stdout.count("stage2_v2_gold_coldstart96_codriving_tvm_resnet_ap_eval.py"), 2)
        for width in ("48x32x64", "16x64x256"):
            self.assertIn(width, completed.stdout)
        self.assertIn("net_epoch_bestval_at35.pth", completed.stdout)
        self.assertIn("net_epoch_bestval_at33.pth", completed.stdout)
        self.assertIn("results/stage4_feedback16_v1_20260716", completed.stdout)
        self.assertIn(
            "/exdata/jichengzhi/V2Xverse_pyramid/output/codriving_v2_gold_ap_20260709/48x32x64",
            completed.stdout,
        )
        self.assertIn(
            "/exdata/jichengzhi/V2Xverse_pyramid/output/codriving_v2_gold_ap_20260709/16x64x256",
            completed.stdout,
        )
        self.assertNotIn("--model-dir ''", completed.stdout)
        self.assertNotIn("--output /stage3_calib_train_n16_float32.npz", completed.stdout)

    def test_source_queue_is_marker_and_sha_fail_closed(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")

        self.assertIn("stage4_feedback16_training_complete_v1", source)
        self.assertIn("sha256sum -c", source)
        self.assertIn("source_done_marker", source)
        self.assertIn("training_marker_valid", source)
        self.assertNotIn("grep -q 'Training Finished'", source)


if __name__ == "__main__":
    unittest.main()
