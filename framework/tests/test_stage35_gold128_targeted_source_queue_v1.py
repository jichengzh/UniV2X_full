from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "stage35_gold128_targeted_source_queue_v1.sh"


class Stage35Gold128TargetedSourceQueueV1Tests(unittest.TestCase):
    def test_dry_run_shards_cover_frozen_candidates_without_overlap(self) -> None:
        outputs = []
        for shard, gpu in ((0, 6), (1, 7)):
            completed = subprocess.run(
                ["bash", str(SCRIPT), "--shard", str(shard), "--gpu", str(gpu), "--dry-run"],
                check=True,
                capture_output=True,
                text=True,
            )
            outputs.append(completed.stdout)

        self.assertIn("16x32x96", outputs[0])
        self.assertIn("40x64x128", outputs[0])
        self.assertNotIn("24x32x96", outputs[0])
        self.assertIn("24x32x96", outputs[1])
        self.assertIn("56x112x224", outputs[1])
        self.assertNotIn("16x32x96", outputs[1])


if __name__ == "__main__":
    unittest.main()
