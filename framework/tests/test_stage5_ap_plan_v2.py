from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path

from scripts.stage5_ap_plan_v2 import expected_row_count, write_stage5_outputs


class Stage5ApPlanV2Tests(unittest.TestCase):
    def test_accepts_one_to_four_rows_for_online_or_independent_manifest(self) -> None:
        for count in range(1, 5):
            with self.subTest(count=count):
                self.assertEqual(expected_row_count({"row_count": count}), count)

    def test_rejects_out_of_range_or_inconsistent_counts(self) -> None:
        for payload in (
            {"row_count": 0},
            {"row_count": 5},
            {"row_count": 2, "jobs": [{}, {}, {}]},
        ):
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    expected_row_count(payload)

    def test_writes_stage5_schema_at_top_level_and_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_stage5_outputs(
                [{"job_id": "job-1"}], root / "plan.json", root / "plan.jsonl"
            )
            payload = json.loads((root / "plan.json").read_text())
            row = json.loads((root / "plan.jsonl").read_text())

        self.assertEqual(payload["schema_version"], "stage5_ap_plan_v2")
        self.assertEqual(payload["jobs"][0]["schema_version"], "stage5_ap_plan_v2")
        self.assertEqual(row["schema_version"], "stage5_ap_plan_v2")


if __name__ == "__main__":
    unittest.main()
