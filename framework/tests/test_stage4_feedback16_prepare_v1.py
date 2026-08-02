from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
CANDIDATE = REPO / "results/stage4_feedback16_v1_20260716/candidate_plan.json"
GOLD176 = (
    REPO
    / "results/stage35_gold144_targeted_supplement_v2_20260714"
    / "final_gold176_v1/gold176_final.json"
)
GOLD176_MANIFEST = GOLD176.with_name("gold176_manifest.json")
QUEUE = REPO / "scripts/stage4_feedback16_prepare_v1.sh"


class Stage4Feedback16PrepareV1Tests(unittest.TestCase):
    def test_candidate_is_four_complete_groups_frozen_against_gold176(self) -> None:
        candidate = json.loads(CANDIDATE.read_text(encoding="utf-8"))
        gold_manifest = json.loads(GOLD176_MANIFEST.read_text(encoding="utf-8"))

        self.assertEqual(candidate["schema_version"], "stage4_feedback16_candidate_v1")
        self.assertEqual(candidate["group_count"], 4)
        self.assertEqual(candidate["row_count"], 16)
        self.assertEqual(
            candidate["source_gold176_sha256"],
            hashlib.sha256(GOLD176.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            candidate["source_gold176_manifest_sha256"],
            hashlib.sha256(GOLD176_MANIFEST.read_bytes()).hexdigest(),
        )

        groups = candidate["groups"]
        group_ids = {group["group_id"] for group in groups}
        existing = {row["group_id"] for row in gold_manifest["jobs"]}
        self.assertEqual(len(group_ids), 4)
        self.assertFalse(group_ids & existing)
        self.assertEqual({group["model"] for group in groups}, {"pyramid", "codriving"})
        self.assertEqual(
            {
                tuple(group["width"])
                for group in groups
                if group["model"] == "pyramid"
            },
            {
                tuple(group["width"])
                for group in groups
                if group["model"] == "codriving"
            },
        )
        self.assertEqual(
            candidate["required_arm_product"],
            [
                ["tvm_auto", "fp16"],
                ["tvm_auto", "int8"],
                ["trt_engine", "fp16"],
                ["trt_engine", "int8"],
            ],
        )

    def test_dry_run_prepares_and_trains_only_two_codriving_groups(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_root = root / "models"
            completed = subprocess.run(
                [
                    "bash",
                    str(QUEUE),
                    "--candidate",
                    str(CANDIDATE),
                    "--gpu",
                    "6",
                    "--dry-run",
                ],
                check=True,
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "REPO": str(REPO),
                    "V2X_ROOT": str(root / "v2x"),
                    "MODEL_ROOT": str(model_root),
                    "PYTHON_BIN": "/test/python",
                },
            )

        lines = completed.stdout.splitlines()
        self.assertEqual(sum("prepare-one" in line for line in lines), 2)
        self.assertEqual(sum("opencood/tools/train.py" in line for line in lines), 2)
        self.assertTrue(all("CUDA_VISIBLE_DEVICES=6" in line for line in lines))
        for width in ("48x32x64", "16x64x256"):
            self.assertIn(width, completed.stdout)

    def test_queue_has_lock_atomic_markers_and_no_fixed_width_array(self) -> None:
        text = QUEUE.read_text(encoding="utf-8")
        self.assertIn("flock", text)
        self.assertIn("training_complete.json", text)
        self.assertIn('mv "$marker.tmp" "$marker"', text)
        self.assertIn("source_gold176_sha256", text)
        self.assertIn("source_graph_features_sha256", text)
        self.assertIn("prepare_report_valid", text)
        self.assertIn('if ! prepare_report_valid "$model_dir/prepare_report.json" "$width"; then', text)
        self.assertNotIn("WIDTHS=(", text)


if __name__ == "__main__":
    unittest.main()
