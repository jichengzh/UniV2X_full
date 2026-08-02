from __future__ import annotations

import copy
import unittest

from scripts import stage4_merge_feedback_view_v1 as merge


def _rows(source: str, groups: tuple[str, ...]) -> list[dict[str, object]]:
    split = "train" if source == "initial_coldstart" else "online_feedback"
    output = []
    for group_id in groups:
        for dispatch_key, q_mode in (
            ("tvm_auto", "fp16"),
            ("tvm_auto", "int8"),
            ("trt_engine", "fp16"),
            ("trt_engine", "int8"),
        ):
            output.append(
                {
                    "manifest_job_id": f"{group_id}|{dispatch_key}|{q_mode}",
                    "group_id": group_id,
                    "model": group_id.split("|", 1)[0],
                    "dispatch_key": dispatch_key,
                    "q_mode": q_mode,
                    "split": split,
                }
            )
    return output


class Stage4MergeFeedbackViewV1Tests(unittest.TestCase):
    def test_merges_disjoint_complete_groups_and_marks_training_source(self) -> None:
        cold = _rows("initial_coldstart", ("pyramid|16x32x64",))
        feedback = _rows("online_feedback", ("codriving|48x32x64",))
        result = merge.merge_feedback_view(
            cold,
            feedback,
            [{"group_id": "pyramid|16x32x64", "conv_count": 1}],
            [{"group_id": "codriving|48x32x64", "conv_count": 2}],
        )

        self.assertEqual(len(result["rows"]), 8)
        self.assertEqual(len(result["graph_features"]), 2)
        self.assertEqual(
            result["audit"]["training_source_rows"],
            {"initial_coldstart": 4, "online_feedback": 4},
        )
        self.assertEqual(
            {row["training_source"] for row in result["rows"][:4]},
            {"initial_coldstart"},
        )
        self.assertEqual(
            {row["training_source"] for row in result["rows"][4:]},
            {"online_feedback"},
        )

    def test_does_not_mutate_inputs(self) -> None:
        cold = _rows("initial_coldstart", ("pyramid|16x32x64",))
        for row in cold:
            row["split"] = "locked_holdout"
        before = copy.deepcopy(cold)
        result = merge.merge_feedback_view(
            cold,
            _rows("online_feedback", ("codriving|48x32x64",)),
            [{"group_id": "pyramid|16x32x64"}],
            [{"group_id": "codriving|48x32x64"}],
        )
        self.assertEqual(cold, before)
        self.assertEqual({row["split"] for row in result["rows"][:4]}, {"locked_holdout"})

    def test_rejects_overlap_wrong_split_and_missing_graph_context(self) -> None:
        cold = _rows("initial_coldstart", ("pyramid|16x32x64",))
        overlap = _rows("online_feedback", ("pyramid|16x32x64",))
        with self.assertRaisesRegex(ValueError, "overlap"):
            merge.merge_feedback_view(cold, overlap, [], [])

        feedback = _rows("online_feedback", ("codriving|48x32x64",))
        feedback[0]["split"] = "train"
        with self.assertRaisesRegex(ValueError, "online_feedback"):
            merge.merge_feedback_view(cold, feedback, [], [])

        feedback[0]["split"] = "online_feedback"
        with self.assertRaisesRegex(ValueError, "graph feature coverage"):
            merge.merge_feedback_view(
                cold,
                feedback,
                [{"group_id": "pyramid|16x32x64"}],
                [],
            )

    def test_rejects_duplicate_rows_and_incomplete_groups(self) -> None:
        cold = _rows("initial_coldstart", ("pyramid|16x32x64",))
        feedback = _rows("online_feedback", ("codriving|48x32x64",))
        with self.assertRaisesRegex(ValueError, "duplicate manifest_job_id"):
            merge.merge_feedback_view(
                [*cold, dict(cold[0])], feedback, [], []
            )

        with self.assertRaisesRegex(ValueError, "complete four-arm"):
            merge.merge_feedback_view(cold[:-1], feedback, [], [])


if __name__ == "__main__":
    unittest.main()
