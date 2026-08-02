from __future__ import annotations

import unittest
from unittest.mock import patch

from scripts import stage4_feedback16_ap_plan_v1 as planner


class Stage4Feedback16ApPlanV1Tests(unittest.TestCase):
    def test_delegates_to_stage3_ap_contract_for_exactly_16_rows(self) -> None:
        manifest = {
            "schema_version": "stage4_feedback16_manifest_v1",
            "jobs": [{"job_id": f"job-{index}"} for index in range(16)],
        }
        delegated = [{"manifest_job_id": f"job-{index}", "ap_terminal": "ready"} for index in range(16)]

        with patch.object(planner.gold96, "build_ap_plan", return_value=delegated) as build:
            rows = planner.build_feedback16_ap_plan(
                manifest,
                performance_jobs=[{"job_id": "performance"}],
                performance_state_rows=[{"status": "success"}],
                output_root="/tmp/feedback16",
            )

        self.assertEqual(len(rows), 16)
        self.assertEqual({row["schema_version"] for row in rows}, {"stage4_feedback16_ap_plan_v1"})
        build.assert_called_once_with(
            manifest,
            performance_jobs=[{"job_id": "performance"}],
            performance_state_rows=[{"status": "success"}],
            pilot_root="/tmp/feedback16",
            manifest_schema="stage4_feedback16_manifest_v1",
            expected_row_count=16,
        )


if __name__ == "__main__":
    unittest.main()
