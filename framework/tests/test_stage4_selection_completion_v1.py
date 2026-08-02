from __future__ import annotations

import math
import unittest

from framework.stage4 import selection_completion_v1 as completion


ARMS = (
    ("tvm_auto", "fp16"),
    ("tvm_auto", "int8"),
    ("trt_engine", "fp16"),
    ("trt_engine", "int8"),
)


def _rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for group_index in range(7):
        model = "codriving" if group_index % 2 == 0 else "pyramid"
        width = [16 + 8 * group_index, 32 + 4 * group_index, 64 + 16 * group_index]
        group_id = f"{model}|{'x'.join(str(value) for value in width)}"
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
                    "terminal_status": "measured_success_gold",
                    "latency_ms": 1.0 + group_index + 0.2 * arm_index,
                    "energy_j": 0.5 + 0.1 * group_index + 0.03 * arm_index,
                    "ap70": 0.4 + 0.02 * group_index - 0.005 * arm_index,
                }
            )
    rows[-1]["latency_ms"] = None
    rows[-1]["terminal_status"] = "feasibility_failure"
    return rows


def _graph(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    for group_id in sorted({str(row["group_id"]) for row in rows}):
        row = next(item for item in rows if item["group_id"] == group_id)
        width = [float(value) for value in row["width"]]
        output.append(
            {
                "group_id": group_id,
                "conv_count": width[0] / 4,
                "group_conv_count": width[1] / 16,
                "log_macs": sum(width),
            }
        )
    return output


def _profiles() -> list[dict[str, object]]:
    return [
        {
            "capability_profile_id": "h800-tvm_auto",
            "features": {"int8_propagation": 0.3, "qdq_folding": 0.2},
        },
        {
            "capability_profile_id": "h800-trt_engine",
            "features": {"int8_propagation": 0.9, "qdq_folding": 0.8},
        },
    ]


def _perfect_cost_report(rows: list[dict[str, object]]) -> dict[str, object]:
    groups = sorted({str(row["group_id"]) for row in rows})
    fold_by_group = {group_id: index % 3 for index, group_id in enumerate(groups)}
    folds = [
        {
            "outer_fold": fold_index,
            "train_groups": sorted(group for group in groups if fold_by_group[group] != fold_index),
            "test_groups": sorted(group for group in groups if fold_by_group[group] == fold_index),
        }
        for fold_index in range(3)
    ]
    return {
        "schema_version": "stage4_cost_model_selection_v1",
        "split_protocol": "nested_grouped_cv_by_model_width",
        "targets": {
            target: {
                "outer_folds": folds,
                "oof_predictions": [
                    {
                        "row_id": row["manifest_job_id"],
                        "group_id": row["group_id"],
                        "outer_fold": fold_by_group[str(row["group_id"])],
                        "truth": row[target],
                        "prediction": row[target],
                    }
                    for row in rows
                    if isinstance(row[target], (int, float)) and math.isfinite(float(row[target]))
                ]
            }
            for target in ("latency_ms", "energy_j", "ap70")
        },
    }


class Stage4SelectionCompletionV1Tests(unittest.TestCase):
    def test_complete_group_filter_excludes_whole_group_for_one_bad_arm(self) -> None:
        selected, audit = completion.select_complete_four_arm_groups(_rows())

        self.assertEqual(len(selected), 24)
        self.assertEqual(audit["selected_group_count"], 6)
        self.assertEqual(audit["excluded_group_count"], 1)
        self.assertEqual(len(audit["excluded_groups"]), 1)
        self.assertIn("non_finite_latency_ms", audit["excluded_groups"][0]["reasons"])

    def test_completion_runs_all_stage4_evaluations_on_same_rows(self) -> None:
        rows = _rows()
        report = completion.run_stage4_completion(
            rows,
            _graph(rows),
            _profiles(),
            _perfect_cost_report(rows),
            outer_splits=3,
            uncertainty_splits=3,
            replay_initial_groups=2,
            replay_budget_groups=6,
            replay_seeds=(3,),
            seed=17,
        )

        self.assertEqual(report["summary"]["evaluation_row_count"], 24)
        self.assertEqual(report["summary"]["evaluation_group_count"], 6)
        self.assertEqual(report["ranking_pareto"]["row_count"], 24)
        self.assertEqual(report["uncertainty"]["row_count"], 24)
        self.assertEqual(report["replay"]["budget_groups"], 6)
        self.assertEqual(
            report["summary"]["generalization_scope"],
            "within_model_grouped_interpolation_not_model_holdout",
        )
        selected_ids = set(report["summary"]["evaluation_row_ids"])
        for target in ("latency_ms", "energy_j", "ap70"):
            predictions = report["filtered_cost_model_report"]["targets"][target]["oof_predictions"]
            self.assertEqual({item["row_id"] for item in predictions}, selected_ids)


if __name__ == "__main__":
    unittest.main()
