from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

from scripts import stage35_gold176_merge_v1 as merge


ROOT = Path(__file__).resolve().parents[2]
GOLD144_ROOT = ROOT / "results/stage35_gold128_targeted_supplement_v1_20260714/final_gold144_v1"
TARGET_MANIFEST = ROOT / "results/stage35_gold144_targeted_supplement_v2_20260714/plan/targeted_supplement_manifest.json"


class Stage35Gold176MergeTests(unittest.TestCase):
    def test_cli_help_runs_from_repo_root(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(ROOT / "scripts/stage35_gold176_merge_v1.py"), "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_merges_eight_train_groups_without_changing_locked_holdout(self) -> None:
        gold_rows = json.loads((GOLD144_ROOT / "gold144_final.json").read_text())
        gold_manifest = json.loads((GOLD144_ROOT / "gold144_manifest.json").read_text())
        target_manifest = json.loads(TARGET_MANIFEST.read_text())
        target_rows = []
        for job in target_manifest["jobs"]:
            target_rows.append(
                {
                    **job,
                    "schema_version": merge.TARGETED32_OUTPUT_SCHEMA,
                    "manifest_job_id": job["job_id"],
                    "terminal_status": "measured_success_gold",
                    "latency_ms": 1.0,
                    "energy_j": 0.5,
                    "ap30": 0.6,
                    "ap50": 0.5,
                    "ap70": 0.4,
                    "performance_result_sha256": "a" * 64,
                    "ap_report_sha256": "b" * 64,
                }
            )
        result = merge.merge_gold176(
            gold_rows,
            gold_manifest,
            target_rows,
            target_manifest,
            gold144_sha256="c" * 64,
        )
        self.assertEqual(result["audit"]["rows"], 176)
        self.assertEqual(result["audit"]["groups"], 44)
        self.assertEqual(result["audit"]["split_groups"], {"train": 38, "locked_holdout": 6})
        self.assertEqual(len(result["audit"]["locked_holdout_groups"]), 6)


if __name__ == "__main__":
    unittest.main()
