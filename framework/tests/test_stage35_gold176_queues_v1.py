from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
TRAIN_QUEUE = REPO / "scripts/stage35_gold176_codriving_train_queue_v1.sh"
SOURCE_QUEUE = REPO / "scripts/stage35_gold176_source_queue_v1.sh"
CANDIDATE_PLAN = (
    REPO
    / "results/stage35_gold144_targeted_supplement_v2_20260714/candidate_plan.json"
)
GOLD144 = (
    REPO
    / "results/stage35_gold128_targeted_supplement_v1_20260714/final_gold144_v1/gold144_final.json"
)
GOLD144_MANIFEST = GOLD144.with_name("gold144_manifest.json")


class Stage35Gold176QueueTests(unittest.TestCase):
    def test_training_dry_run_contains_each_frozen_codriving_width(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_root = root / "models"
            for width in ("48x64x192", "48x96x128", "48x32x128", "64x64x192"):
                (model_root / width).mkdir(parents=True)
                (model_root / width / "config.yaml").write_text("name: test\n")
            completed = subprocess.run(
                ["bash", str(TRAIN_QUEUE), "--gpu", "6", "--dry-run"],
                check=True,
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "V2X_ROOT": str(root / "v2x"),
                    "MODEL_ROOT": str(model_root),
                    "PYTHON_BIN": "/test/python",
                },
            )
        self.assertEqual(completed.stdout.count("CUDA_VISIBLE_DEVICES=6"), 4)
        for width in ("48x64x192", "48x96x128", "48x32x128", "64x64x192"):
            self.assertIn(width, completed.stdout)

    def test_source_dry_run_covers_four_pyramid_and_four_codriving_groups(self) -> None:
        completed = subprocess.run(
            ["bash", str(SOURCE_QUEUE), "--gpu", "6", "--dry-run"],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(len(completed.stdout.splitlines()), 20)
        for width in (
            "24x32x64", "40x96x192", "56x128x256", "64x128x224",
            "48x64x192", "48x96x128", "48x32x128", "64x64x192",
        ):
            self.assertIn(width, completed.stdout)

    def test_candidate_plan_is_frozen_to_gold144_without_group_overlap(self) -> None:
        candidate = json.loads(CANDIDATE_PLAN.read_text())
        manifest = json.loads(GOLD144_MANIFEST.read_text())
        digest = hashlib.sha256(GOLD144.read_bytes()).hexdigest()
        manifest_digest = hashlib.sha256(GOLD144_MANIFEST.read_bytes()).hexdigest()
        groups = candidate["groups"]
        candidate_ids = {row["group_id"] for row in groups}
        existing_ids = {row["group_id"] for row in manifest["jobs"]}
        locked_ids = {
            row["group_id"] for row in manifest["jobs"]
            if row["split"] == "locked_holdout"
        }

        self.assertEqual(candidate["source_gold144_sha256"], digest)
        self.assertEqual(
            candidate["source_gold144_manifest_sha256"], manifest_digest
        )
        self.assertEqual(candidate["group_count"], 8)
        self.assertEqual(candidate["row_count"], 32)
        self.assertEqual(len(candidate_ids), 8)
        self.assertFalse(candidate_ids & existing_ids)
        self.assertFalse(candidate_ids & locked_ids)
        self.assertEqual(
            {row["model"] for row in groups}, {"pyramid", "codriving"}
        )

    def test_queues_use_shared_gpu_lock_and_artifact_completion_markers(self) -> None:
        training = TRAIN_QUEUE.read_text()
        sources = SOURCE_QUEUE.read_text()

        self.assertIn("flock", training)
        self.assertIn("flock", sources)
        self.assertIn("training_complete.json", training)
        self.assertIn("training_complete.json", sources)
        self.assertNotIn('grep -q "Training Finished" "$train_log"', sources)
        self.assertNotIn('grep -q "Training Finished"', training)
        self.assertIn('bestval_count', training)
        self.assertIn('bestval_count', sources)
        self.assertIn('config" == "$model_dir/config.yaml', training)
        self.assertIn('dirname "$checkpoint")" == "$model_dir', training)
        self.assertIn("sha256sum -c", sources)


if __name__ == "__main__":
    unittest.main()
