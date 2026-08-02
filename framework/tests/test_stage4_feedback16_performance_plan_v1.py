from __future__ import annotations

import json
import unittest
from pathlib import Path

from scripts import stage4_feedback16_performance_plan_v1 as plan
from scripts import stage3_execute_performance_plan_v3 as executor


ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT / "results/stage4_feedback16_v1_20260716/candidate_plan.json"
GOLD176 = ROOT / "results/stage35_gold144_targeted_supplement_v2_20260714/final_gold176_v1/gold176_final.json"
MANIFEST176 = GOLD176.with_name("gold176_manifest.json")
GRAPH176 = GOLD176.with_name("graph_features.json")
CAPABILITIES = ROOT / "results/s1_profile_final_v3_20260711/capability_profiles_v3.json"


class Stage4Feedback16PerformancePlanTests(unittest.TestCase):
    def test_builds_four_complete_online_feedback_groups(self) -> None:
        candidate = json.loads(CANDIDATE.read_text(encoding="utf-8"))
        gold176 = json.loads(GOLD176.read_text(encoding="utf-8"))
        manifest176 = json.loads(MANIFEST176.read_text(encoding="utf-8"))
        capabilities = json.loads(CAPABILITIES.read_text(encoding="utf-8"))

        result = plan.build_plan(
            candidate,
            gold176_rows=gold176,
            gold176_manifest=manifest176,
            capability_profiles=capabilities,
            remote_result_root="/remote/results/stage4_feedback16",
            gpus=[6, 7],
            graph_features_sha256=plan.sha256_file(GRAPH176),
            capability_profiles_sha256=plan.sha256_file(CAPABILITIES),
        )

        manifest = result["manifest"]
        jobs = manifest["jobs"]
        performance = result["performance_jobs"]
        self.assertEqual(manifest["schema_version"], "stage4_feedback16_manifest_v1")
        self.assertEqual(manifest["source_pool"], "online_feedback")
        self.assertEqual(manifest["group_count"], 4)
        self.assertEqual(manifest["row_count"], 16)
        self.assertEqual(len(jobs), 16)
        self.assertEqual(len(performance), 16)
        self.assertEqual({row["split"] for row in jobs}, {"online_feedback"})
        self.assertEqual({row["source_pool"] for row in jobs}, {"stage4_feedback16_online_feedback"})
        self.assertFalse({row["group_id"] for row in jobs} & {row["group_id"] for row in manifest176["jobs"]})
        self.assertEqual(manifest["source_sha256"]["gold176"], candidate["source_gold176_sha256"])
        self.assertEqual(manifest["source_sha256"]["gold176_manifest"], candidate["source_gold176_manifest_sha256"])
        self.assertEqual(manifest["source_sha256"]["graph_features"], candidate["source_graph_features_sha256"])
        self.assertEqual(manifest["source_sha256"]["capability_profiles"], plan.sha256_file(CAPABILITIES))

        grouped: dict[str, set[tuple[str, str]]] = {}
        for row in jobs:
            grouped.setdefault(row["group_id"], set()).add((row["dispatch_key"], row["q_mode"]))
        self.assertEqual(set(grouped), {row["group_id"] for row in candidate["groups"]})
        self.assertTrue(all(arms == plan.EXPECTED_ARMS for arms in grouped.values()))

        codriving_int8 = next(
            row for row in performance
            if row["model"] == "codriving" and row["runner_key"] == "trt_int8"
        )
        model_dir = Path(codriving_int8["source_contract"]["model_dir"])
        self.assertEqual(
            executor._derive_trt_calibration_dir(codriving_int8),
            str(model_dir / "trt_calibration_npy"),
        )

    def test_rejects_changed_gold176_or_graph_sha(self) -> None:
        candidate = json.loads(CANDIDATE.read_text(encoding="utf-8"))
        candidate["source_graph_features_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "graph features SHA"):
            plan.validate_source_hashes(
                candidate,
                gold176_path=GOLD176,
                manifest176_path=MANIFEST176,
                graph_features_path=GRAPH176,
            )


if __name__ == "__main__":
    unittest.main()
