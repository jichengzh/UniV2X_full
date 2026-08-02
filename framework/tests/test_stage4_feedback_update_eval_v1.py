from __future__ import annotations

import unittest

from framework.stage4.feedback_update_eval_v1 import run_feedback_update_evaluation


ARMS = (
    ("tvm_auto", "fp16"),
    ("tvm_auto", "int8"),
    ("trt_engine", "fp16"),
    ("trt_engine", "int8"),
)


def _rows() -> list[dict[str, object]]:
    rows = []
    specifications = [
        *(('pyramid', index, 'train', 'initial_coldstart') for index in range(6)),
        *(('codriving', index + 6, 'locked_holdout', 'initial_coldstart') for index in range(2)),
        *(('codriving', index + 8, 'online_feedback', 'online_feedback') for index in range(4)),
    ]
    for model, index, split, source in specifications:
        width = [16 + 4 * index, 32 + 4 * index, 64 + 8 * index]
        group_id = f"{model}|{'x'.join(map(str, width))}"
        for arm_index, (dispatch_key, q_mode) in enumerate(ARMS):
            rows.append(
                {
                    "manifest_job_id": f"{group_id}|{dispatch_key}|{q_mode}",
                    "group_id": group_id,
                    "model": model,
                    "width": width,
                    "dispatch_key": dispatch_key,
                    "q_mode": q_mode,
                    "capability_profile_id": f"h800-{dispatch_key}",
                    "split": split,
                    "training_source": source,
                    "latency_ms": 2.0 + 0.2 * index + 0.05 * arm_index,
                    "energy_j": 0.4 + 0.03 * index + 0.01 * arm_index,
                    "ap70": 0.3 + 0.02 * index - 0.005 * arm_index,
                }
            )
    return rows


def _graph(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {"group_id": group_id, "conv_count": index + 1}
        for index, group_id in enumerate(sorted({str(row["group_id"]) for row in rows}))
    ]


def _profiles() -> list[dict[str, object]]:
    return [
        {"capability_profile_id": "h800-tvm_auto", "features": {"tc": 1.0}},
        {"capability_profile_id": "h800-trt_engine", "features": {"tc": 2.0}},
    ]


def _cost_report() -> dict[str, object]:
    return {
        "targets": {
            target: {
                "selected_candidate_counts": {
                    "extra_trees_raw": 4,
                    "lgbm_l1_raw": 1,
                }
            }
            for target in ("latency_ms", "energy_j", "ap70")
        }
    }


class Stage4FeedbackUpdateEvalV1Tests(unittest.TestCase):
    def test_leave_one_feedback_group_out_preserves_temporal_roles(self) -> None:
        rows = _rows()
        report = run_feedback_update_evaluation(
            rows, _graph(rows), _profiles(), _cost_report(), seed=7
        )

        self.assertEqual(report["schema_version"], "stage4_feedback_update_eval_v1")
        self.assertEqual(report["canonical_value_heads"], {
            "latency_ms": "extra_trees_raw",
            "energy_j": "extra_trees_raw",
            "ap70": "extra_trees_raw",
        })
        self.assertEqual(len(report["folds"]), 4)
        for fold in report["folds"]:
            heldout = fold["heldout_feedback_group"]
            self.assertNotIn(heldout, fold["baseline_fit_groups"])
            self.assertNotIn(heldout, fold["updated_fit_groups"])
            self.assertFalse(set(fold["calibration_groups"]) & set(fold["updated_fit_groups"]))
            self.assertEqual(len(fold["added_feedback_groups"]), 3)
        for mode in ("before_feedback", "after_feedback"):
            self.assertEqual(report[mode]["row_count"], 16)
            self.assertEqual(report[mode]["group_count"], 4)
            self.assertGreaterEqual(report[mode]["simultaneous_group_coverage"], 0.0)
            self.assertLessEqual(report[mode]["simultaneous_group_coverage"], 1.0)

    def test_rejects_missing_source_role_or_calibration_split(self) -> None:
        rows = _rows()
        rows[0].pop("training_source")
        with self.assertRaisesRegex(ValueError, "training_source"):
            run_feedback_update_evaluation(rows, _graph(rows), _profiles(), _cost_report())

        rows = _rows()
        for row in rows:
            if row["split"] == "locked_holdout":
                row["split"] = "train"
        with self.assertRaisesRegex(ValueError, "locked_holdout"):
            run_feedback_update_evaluation(rows, _graph(rows), _profiles(), _cost_report())


if __name__ == "__main__":
    unittest.main()
