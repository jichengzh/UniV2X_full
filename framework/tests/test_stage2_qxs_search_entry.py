from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import stage2_smbo_qxs_entry_v1 as qxs  # noqa: E402


def _manifest() -> dict:
    return {
        "arms": [
            {
                "search_arm_id": "fp32__baseline_reference",
                "q_axis": "fp32",
                "backend": "h800_tvm_relax_fp32",
                "can_enter_final_frontier": True,
            },
            {
                "search_arm_id": "fp16__route_b_auto_tc",
                "q_axis": "fp16",
                "backend": "route_b_auto_tune_split_fix",
                "can_enter_final_frontier": True,
            },
            {
                "search_arm_id": "int8__route_b_auto_int8_tc",
                "q_axis": "int8",
                "backend": "route_b_auto_decomp_int8_tensorcore",
                "can_enter_final_frontier": True,
            },
            {
                "search_arm_id": "int8__historical_hand_rewrite_prior",
                "q_axis": "int8",
                "backend": "h800_tvm_int8_rewritten_tensorcore",
                "can_enter_final_frontier": False,
            },
        ]
    }


class Stage2QxSSearchEntryTests(unittest.TestCase):
    def test_feature_encoding_uses_width_q_backend_and_source_confidence(self) -> None:
        rows = [
            {
                "width": "16x32x64",
                "search_arm_id": "fp16__route_b_auto_tc",
                "q_axis": "fp16",
                "backend": "route_b_auto_tune_split_fix",
                "confidence_weight": 0.8,
            }
        ]

        spec = qxs.build_feature_spec(_manifest(), rows)
        encoded = qxs.encode_training_row(rows[0], spec)

        self.assertEqual(spec["genome_schema"], ["w0", "w1", "w2", "search_arm_id"])
        self.assertEqual(spec["feature_order"][:3], ["w0", "w1", "w2"])
        self.assertIn("q_axis=fp16", spec["feature_order"])
        self.assertIn("backend=route_b_auto_tune_split_fix", spec["feature_order"])
        self.assertEqual(spec["feature_order"][-1], "source_confidence")
        self.assertEqual(encoded["genome"], [16, 32, 64, "fp16__route_b_auto_tc"])
        self.assertEqual(encoded["features"][spec["feature_order"].index("q_axis=fp16")], 1.0)
        self.assertEqual(encoded["features"][spec["feature_order"].index("source_confidence")], 0.8)

    def test_final_frontier_filter_excludes_historical_prior_even_if_row_is_marked_trusted(self) -> None:
        rows = [
            {
                "label": "fp32_ok",
                "width": "16x32x64",
                "search_arm_id": "fp32__baseline_reference",
                "trusted_for_final_frontier": "conditional",
            },
            {
                "label": "routeb_ok",
                "width": "16x32x64",
                "search_arm_id": "int8__route_b_auto_int8_tc",
                "trusted_for_final_frontier": True,
            },
            {
                "label": "historical_not_gold",
                "width": "16x32x64",
                "search_arm_id": "int8__historical_hand_rewrite_prior",
                "trusted_for_final_frontier": True,
            },
            {
                "label": "legacy_not_gold",
                "width": "24x32x96",
                "search_arm_id": "fp16__legacy_prior_needs_routeb_remeasure",
                "trusted_for_final_frontier": False,
            },
        ]

        final_rows = qxs.filter_final_frontier_rows(rows, _manifest())

        self.assertEqual([row["label"] for row in final_rows], ["fp32_ok", "routeb_ok"])

    def test_disagreement_trigger_emits_center_and_axis_neighbors(self) -> None:
        keypoints = [
            {"width": "16x32x64", "routeb_div_old_int8tc_prior": 2.47},
            {"width": "24x32x96", "routeb_div_old_int8tc_prior": 2.02},
            {"width": "32x64x128", "routeb_div_old_int8tc_prior": 1.10},
        ]

        plan = qxs.build_disagreement_remeasure_plan(
            keypoints,
            width_grids=([16, 24, 32], [32, 48, 64], [64, 96, 128]),
            threshold=1.5,
        )

        self.assertEqual([item["width"] for item in plan["centers"]], ["16x32x64", "24x32x96"])
        queue_widths = {item["width"] for item in plan["remeasure_queue"]}
        self.assertTrue({"16x32x64", "24x32x96"} <= queue_widths)
        self.assertTrue({"24x32x64", "16x48x64", "16x32x96"} <= queue_widths)
        self.assertTrue({"32x32x96", "24x48x96", "24x32x128"} <= queue_widths)


if __name__ == "__main__":
    unittest.main()
