from __future__ import annotations

import copy
import unittest

from framework.stage4.closure_audit_v1 import build_stage4_closure_audit


def _cost() -> dict:
    return {
        "targets": {
            "latency_ms": {"selected_candidate_counts": {"extra_trees_log": 5}},
            "energy_j": {"selected_candidate_counts": {"extra_trees_log": 4}},
            "ap70": {"selected_candidate_counts": {"lgbm_huber_residual": 3}},
        }
    }


def _completion(random_groups: int, selected_groups: int) -> dict:
    return {
        "summary": {"retain_ranker": False, "selected_uncertainty_method": "lgbm_quantile"},
        "replay": {
            "policies": {
                "predicted_frontier_diversity": {
                    "groups_to_95pct_oracle_HV_median": selected_groups,
                    "success_rate": 1.0,
                },
                "random": {
                    "groups_to_95pct_oracle_HV_median": random_groups,
                    "success_rate": 1.0,
                },
            }
        },
    }


def _feedback() -> dict:
    feedback_groups = ["pyramid|f0", "pyramid|f1", "codriving|f2", "codriving|f3"]
    records = {
        target: [
            {"manifest_job_id": f"{group}|row{arm}", "group_id": group}
            for group in feedback_groups
            for arm in range(4)
        ]
        for target in ("latency_ms", "energy_j", "ap70")
    }
    return {
        "schema_version": "stage4_feedback_update_eval_v1",
        "canonical_value_heads": {
            "latency_ms": "extra_trees_log",
            "energy_j": "extra_trees_log",
            "ap70": "lgbm_huber_residual",
        },
        "uncertainty_method": "lgbm_quantile_plus_group_conformal",
        "folds": [
            {
                "heldout_feedback_group": heldout,
                "baseline_fit_groups": ["pyramid|train", "codriving|train"],
                "updated_fit_groups": [
                    "pyramid|train", "codriving|train",
                    *(group for group in feedback_groups if group != heldout),
                ],
                "added_feedback_groups": [group for group in feedback_groups if group != heldout],
                "calibration_groups": ["pyramid|cal", "codriving|cal"],
            }
            for heldout in feedback_groups
        ],
        "before_feedback": {
            "row_count": 16,
            "records": copy.deepcopy(records),
            "targets": {
                target: {"point_metrics": {"mae": 0.04}, "interval": {"mean_width": 0.5, "row_coverage": 1.0}}
                for target in ("latency_ms", "energy_j", "ap70")
            },
        },
        "after_feedback": {
            "row_count": 16,
            "records": copy.deepcopy(records),
            "targets": {
                target: {
                    "point_metrics": {"mae": 0.03 if target == "ap70" else 0.05},
                    "interval": {"mean_width": 0.45 if target == "ap70" else 0.55, "row_coverage": 1.0},
                }
                for target in ("latency_ms", "energy_j", "ap70")
            },
        },
    }


def _holdout() -> dict:
    return {
        "schema_version": "stage4_independent_holdout_v1",
        "status": "frozen_before_source_materialization_and_labels",
        "group_count": 4,
        "row_count_after_four_arm_measurement": 16,
        "required_arm_product": [
            ["tvm_auto", "fp16"], ["tvm_auto", "int8"],
            ["trt_engine", "fp16"], ["trt_engine", "int8"],
        ],
        "groups": [
            {"group_id": "pyramid|new0", "model": "pyramid", "width": [1, 2, 3], "label_state_at_freeze": "unavailable"},
            {"group_id": "pyramid|new1", "model": "pyramid", "width": [2, 3, 4], "label_state_at_freeze": "unavailable"},
            {"group_id": "codriving|new0", "model": "codriving", "width": [1, 2, 3], "label_state_at_freeze": "unavailable"},
            {"group_id": "codriving|new1", "model": "codriving", "width": [2, 3, 4], "label_state_at_freeze": "unavailable"},
        ],
        "freeze_audit": {
            "overlap_with_gold176_or_feedback16_groups": [],
            "pre_freeze_results_or_script_reference_count_by_width": {
                "1x2x3": 0,
                "2x3x4": 0,
            },
            "forbidden_until_measurement": [
                "latency_ms", "energy_j", "ap30", "ap50", "ap70", "terminal_status"
            ],
        },
    }


def _reproducibility(exact: bool = True) -> dict:
    names = {
        "gold176_baseline_completion",
        "feedback_cost_model",
        "feedback_update_eval",
        "feedback_completion_ranking_uncertainty_pareto_replay",
    }
    return {
        "schema_version": "stage4_p1_p3_reproducibility_audit_v1",
        "all_exact_match": exact,
        "artifacts": {
            name: {"original_sha256": "a" * 64, "rerun_sha256": "a" * 64, "byte_exact": exact}
            for name in names
        },
    }


def _merged_rows() -> list[dict[str, object]]:
    roles = [
        ("pyramid|train", "initial_coldstart", "train"),
        ("codriving|train", "initial_coldstart", "train"),
        ("pyramid|cal", "initial_coldstart", "locked_holdout"),
        ("codriving|cal", "initial_coldstart", "locked_holdout"),
        ("pyramid|f0", "online_feedback", "online_feedback"),
        ("pyramid|f1", "online_feedback", "online_feedback"),
        ("codriving|f2", "online_feedback", "online_feedback"),
        ("codriving|f3", "online_feedback", "online_feedback"),
    ]
    return [
        {
            "manifest_job_id": f"{group_id}|row{arm}",
            "group_id": group_id,
            "training_source": source,
            "split": split,
            "latency_ms": 1.0,
            "energy_j": 1.0,
            "ap70": 0.5,
        }
        for group_id, source, split in roles
        for arm in range(4)
    ]


class Stage4ClosureAuditV1Tests(unittest.TestCase):
    def test_closes_when_all_six_gates_pass(self) -> None:
        report = build_stage4_closure_audit(
            _cost(),
            _completion(10, 8),
            _completion(13, 10),
            _feedback(),
            _holdout(),
            reproducibility_audit=_reproducibility(),
            merged_rows=_merged_rows(),
        )
        self.assertTrue(report["stage4_closed"])
        self.assertTrue(report["stage5_search_ready"])
        self.assertEqual(report["selected_acquisition_policy"], "predicted_frontier_diversity")
        self.assertTrue(all(report["gates"].values()))

    def test_rejects_holdout_labels_and_fails_slow_acquisition(self) -> None:
        holdout = copy.deepcopy(_holdout())
        holdout["groups"][0]["latency_ms"] = 1.0
        with self.assertRaisesRegex(ValueError, "label"):
            build_stage4_closure_audit(
                _cost(), _completion(10, 12), _completion(13, 15), _feedback(), holdout,
                reproducibility_audit=_reproducibility(),
                merged_rows=_merged_rows(),
            )

        report = build_stage4_closure_audit(
            _cost(), _completion(10, 12), _completion(13, 15), _feedback(), _holdout(),
            reproducibility_audit=_reproducibility(),
            merged_rows=_merged_rows(),
        )
        self.assertFalse(report["gates"]["acquisition_not_worse_than_random"])
        self.assertFalse(report["stage4_closed"])

    def test_requires_exact_reproducibility(self) -> None:
        report = build_stage4_closure_audit(
            _cost(), _completion(10, 8), _completion(13, 10), _feedback(), _holdout(),
            reproducibility_audit=_reproducibility(False),
            merged_rows=_merged_rows(),
        )
        self.assertFalse(report["gates"]["feedback_update_reproducible"])

    def test_rejects_incomplete_feedback_protocol_and_holdout_contract(self) -> None:
        feedback = _feedback()
        feedback["folds"] = feedback["folds"][:1]
        report = build_stage4_closure_audit(
            _cost(), _completion(10, 8), _completion(13, 10), feedback, _holdout(),
            reproducibility_audit=_reproducibility(),
            merged_rows=_merged_rows(),
        )
        self.assertFalse(report["gates"]["feedback_update_reproducible"])

        holdout = _holdout()
        holdout["groups"] = holdout["groups"][:2]
        report = build_stage4_closure_audit(
            _cost(), _completion(10, 8), _completion(13, 10), _feedback(), holdout,
            reproducibility_audit=_reproducibility(),
            merged_rows=_merged_rows(),
        )
        self.assertFalse(report["gates"]["independent_holdout_frozen_before_labels"])

    def test_feedback_protocol_must_bind_to_merged_source_roles(self) -> None:
        rows = _merged_rows()
        for row in rows:
            if row["group_id"] == "pyramid|cal":
                row["split"] = "train"
        report = build_stage4_closure_audit(
            _cost(), _completion(10, 8), _completion(13, 10), _feedback(), _holdout(),
            reproducibility_audit=_reproducibility(), merged_rows=rows,
        )
        self.assertFalse(report["gates"]["feedback_update_reproducible"])


if __name__ == "__main__":
    unittest.main()
