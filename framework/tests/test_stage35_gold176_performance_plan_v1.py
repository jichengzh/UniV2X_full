from __future__ import annotations

import json
import unittest
from pathlib import Path

from scripts import stage35_gold176_performance_plan_v1 as plan
from scripts import stage3_execute_performance_plan_v3 as executor


ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT / "results/stage35_gold144_targeted_supplement_v2_20260714/candidate_plan.json"
GOLD = ROOT / "results/stage35_gold128_targeted_supplement_v1_20260714/final_gold144_v1/gold144_final.json"
MANIFEST = GOLD.with_name("gold144_manifest.json")
CAPABILITIES = ROOT / "results/s1_profile_final_v3_20260711/capability_profiles_v3.json"


class Stage35Gold176PerformancePlanTests(unittest.TestCase):
    def test_builds_eight_complete_four_arm_groups_without_holdout_overlap(self) -> None:
        candidate = json.loads(CANDIDATE.read_text())
        base_manifest = json.loads(MANIFEST.read_text())
        capabilities = json.loads(CAPABILITIES.read_text())
        result = plan.build_plan(
            candidate,
            base_manifest=base_manifest,
            capability_profiles=capabilities,
            remote_result_root="/remote/results/gold176",
            gpus=[6, 7],
        )

        jobs = result["manifest"]["jobs"]
        performance = result["performance_jobs"]
        self.assertEqual(len(jobs), 32)
        self.assertEqual(len(performance), 32)
        self.assertEqual({row["split"] for row in jobs}, {"train"})
        self.assertEqual({row["model"] for row in jobs}, {"pyramid", "codriving"})
        self.assertEqual({row["assigned_gpu"] for row in performance}, {6, 7})
        self.assertFalse(
            {row["group_id"] for row in jobs}
            & {row["group_id"] for row in base_manifest["jobs"]}
        )
        grouped: dict[str, set[tuple[str, str]]] = {}
        for row in jobs:
            grouped.setdefault(row["group_id"], set()).add(
                (row["dispatch_key"], row["q_mode"])
            )
        self.assertEqual(len(grouped), 8)
        self.assertTrue(all(arms == plan.EXPECTED_ARMS for arms in grouped.values()))

        pyramid_row = next(row for row in jobs if row["model"] == "pyramid")
        pyramid_source = pyramid_row["source_contract"]
        self.assertEqual(
            pyramid_source["checkpoint_dir"],
            str(Path(pyramid_source["checkpoint_path"]).parent),
        )
        self.assertTrue(pyramid_source["calibration_summary"].endswith("/summary.json"))

        codriving_row = next(row for row in jobs if row["model"] == "codriving")
        self.assertTrue(
            codriving_row["source_contract"]["calibration_summary"].endswith(
                "/stage3_calib_train_n16_float32_summary.json"
            )
        )

        codriving_int8 = next(
            row
            for row in performance
            if row["model"] == "codriving" and row["runner_key"] == "trt_int8"
        )
        model_dir = Path(codriving_int8["source_contract"]["model_dir"])
        self.assertEqual(
            executor._derive_trt_calibration_dir(codriving_int8),
            str(model_dir / "trt_calibration_npy"),
        )

    def test_rejects_changed_gold_or_manifest_sha(self) -> None:
        candidate = json.loads(CANDIDATE.read_text())
        candidate["source_gold144_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "Gold144 SHA"):
            plan.validate_source_hashes(
                candidate,
                gold_path=GOLD,
                manifest_path=MANIFEST,
                report_path=ROOT / "results/stage35_gold144_sufficiency_final_v2_20260715/gold144_sufficiency_report.json",
                capabilities_path=CAPABILITIES,
            )


if __name__ == "__main__":
    unittest.main()
