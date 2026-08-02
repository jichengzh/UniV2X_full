from __future__ import annotations

import unittest

from framework.stage2 import canonical_search_v3 as canonical
from framework.stage2 import cost_model_bundle_v3 as cost_model
from framework.stage2 import search_loop_v3 as search
from framework.tests.test_stage2_cost_model_bundle_v3 import _profiles, _rows


class Stage2SearchLoopV3Tests(unittest.TestCase):
    def test_pareto_uncertainty_acquisition_and_dispatch_keep_backend_out_of_genome(self) -> None:
        bundle = cost_model.fit_model_bundle(_rows(), _profiles(), ridge=1e-6)
        candidates = _rows()[:4]
        predicted = cost_model.predict_rows(bundle, candidates, _profiles())

        selected = search.select_candidates(predicted, measured_row_ids=set(), budget=2)
        request = search.build_measurement_request(selected[0], _profiles())

        self.assertEqual(len(selected), 2)
        self.assertIn("pareto_rank", selected[0])
        self.assertIn("acquisition_uncertainty", selected[0])
        self.assertIn(request["strategy_id"], {"q=fp16", "q=int8"})
        self.assertNotIn("tvm", request["strategy_id"])
        self.assertNotIn("trt", request["strategy_id"])
        self.assertIn(request["dispatch_key"], {"tvm_auto", "trt_engine"})

    def test_feedback_returns_retrained_bundle_without_mutating_previous_bundle(self) -> None:
        profiles = _profiles()
        rows = _rows()
        first = cost_model.fit_model_bundle(rows[:8], profiles, ridge=1e-6)

        updated = search.apply_measurement_feedback(first, rows[8:], profiles)

        self.assertEqual(first["training_row_count"], 8)
        self.assertEqual(updated["training_row_count"], 12)

    def test_conditioned_model_selects_fp16_for_tvm_and_int8_for_trt(self) -> None:
        profiles = _profiles()
        rows = _rows()
        bundle = cost_model.fit_model_bundle(rows, profiles, ridge=1e-6)
        candidates = [row for row in rows if row["width"] == [32, 64, 128]]
        predicted = cost_model.predict_rows(bundle, candidates, profiles)

        decisions = search.best_q_by_profile(predicted, objective="latency_ms")

        self.assertEqual(decisions["tvm-profile"]["q_mode"], "fp16")
        self.assertEqual(decisions["trt-profile"]["q_mode"], "int8")

    def test_unconditional_int8_baseline_reports_profile_specific_regret(self) -> None:
        profiles = _profiles()
        rows = _rows()
        bundle = cost_model.fit_model_bundle(rows, profiles, ridge=1e-6)
        candidates = [row for row in rows if row["width"] == [32, 64, 128]]
        predicted = cost_model.predict_rows(bundle, candidates, profiles)

        report = search.unconditional_int8_regret(predicted, objective="latency_ms")

        self.assertGreater(report["tvm-profile"]["relative_regret"], 0.0)
        self.assertAlmostEqual(report["trt-profile"]["relative_regret"], 0.0, places=5)

    def test_ap_helpers_maximize_accuracy_and_report_loss_as_regret(self) -> None:
        predicted = [
            {
                "row_id": "fp16",
                "capability_profile_id": "profile",
                "q_mode": "fp16",
                "predictions": {"ap70": 0.8},
            },
            {
                "row_id": "int8",
                "capability_profile_id": "profile",
                "q_mode": "int8",
                "predictions": {"ap70": 0.6},
            },
        ]

        decision = search.best_q_by_profile(predicted, objective="ap70")
        regret = search.unconditional_int8_regret(predicted, objective="ap70")

        self.assertEqual(decision["profile"]["q_mode"], "fp16")
        self.assertAlmostEqual(regret["profile"]["relative_regret"], 0.25)

    def test_acquisition_supports_metric_specific_training_views(self) -> None:
        rows = [
            {
                "row_id": "a",
                "predictions": {"latency_ms": 1.0, "energy_j": 0.4, "feasibility": 1.0},
                "uncertainty_p90": {"latency_ms": 0.1, "energy_j": 0.02},
            },
            {
                "row_id": "b",
                "predictions": {"latency_ms": 1.1, "energy_j": 0.3, "feasibility": 1.0},
                "uncertainty_p90": {"latency_ms": 0.1, "energy_j": 0.02},
            },
        ]

        selected = search.select_candidates(
            rows, measured_row_ids=set(), budget=1, objectives=("latency_ms", "energy_j")
        )

        self.assertEqual(len(selected), 1)

    def test_group_acquisition_returns_complete_paired_groups(self) -> None:
        rows = []
        for group_id in ("g1", "g2", "g3"):
            for profile in ("tvm", "trt"):
                for q_mode in ("fp16", "int8"):
                    rows.append(
                        {
                            "row_id": f"{group_id}-{profile}-{q_mode}",
                            "group_id": group_id,
                            "predictions": {
                                "latency_ms": 1.0 + 0.1 * (q_mode == "int8"),
                                "energy_j": 0.4,
                                "feasibility": 1.0,
                            },
                            "uncertainty_p90": {"latency_ms": 0.1, "energy_j": 0.02},
                        }
                    )

        selected = search.select_candidate_groups(
            rows,
            measured_group_ids={"g1"},
            group_budget=1,
            objectives=("latency_ms", "energy_j"),
        )

        self.assertEqual(len(selected), 4)
        self.assertEqual(len({row["group_id"] for row in selected}), 1)
        self.assertNotEqual(selected[0]["group_id"], "g1")

    def test_group_acquisition_rejects_incomplete_group(self) -> None:
        rows = [
            {
                "row_id": f"g1-{index}",
                "group_id": "g1",
                "predictions": {"latency_ms": 1.0, "energy_j": 0.4, "feasibility": 1.0},
                "uncertainty_p90": {"latency_ms": 0.1, "energy_j": 0.02},
            }
            for index in range(3)
        ]

        with self.assertRaisesRegex(ValueError, "complete groups of 4 rows"):
            search.select_candidate_groups(rows, measured_group_ids=set(), group_budget=1)


if __name__ == "__main__":
    unittest.main()
