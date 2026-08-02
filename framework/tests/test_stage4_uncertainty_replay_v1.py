from __future__ import annotations

import math
import unittest

from framework.stage4 import uncertainty_replay_v1 as replay


ARMS = (
    ("tvm_auto", "fp16"),
    ("tvm_auto", "int8"),
    ("trt_engine", "fp16"),
    ("trt_engine", "int8"),
)


def _rows(group_count: int = 10) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(group_count):
        model = "codriving" if index % 2 == 0 else "pyramid"
        width = [24 + index * 4, 32 + index * 3, 64 + index * 8]
        group_id = f"{model}|g{index:02d}"
        for arm_index, (backend, q_mode) in enumerate(ARMS):
            latency = 9.0 + index * 0.9 + arm_index * 0.35
            if backend == "trt_engine":
                latency -= 0.55
            if q_mode == "int8":
                latency -= 0.3
            energy = 1.8 + index * 0.13 + arm_index * 0.05
            ap70 = 0.46 + index * 0.018 - arm_index * 0.004
            if q_mode == "int8":
                ap70 -= 0.012
            rows.append(
                {
                    "manifest_job_id": f"{group_id}|{backend}|{q_mode}",
                    "group_id": group_id,
                    "model": model,
                    "width": width,
                    "dispatch_key": backend,
                    "q_mode": q_mode,
                    "capability_profile_id": f"h800-{backend}",
                    "latency_ms": latency,
                    "energy_j": energy,
                    "ap70": ap70,
                }
            )
    return rows


def _graph_features(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    for group_id in sorted({str(row["group_id"]) for row in rows}):
        first = next(row for row in rows if row["group_id"] == group_id)
        width = [float(value) for value in first["width"]]
        output.append(
            {
                "group_id": group_id,
                "conv_count": int(width[0] // 4),
                "group_conv_count": int(width[1] // 16),
                "log_macs": sum(width) / 100.0,
            }
        )
    return output


def _profiles() -> list[dict[str, object]]:
    return [
        {
            "capability_profile_id": "h800-tvm_auto",
            "features": {"throughput": 1.0, "memory_bw": 0.8},
        },
        {
            "capability_profile_id": "h800-trt_engine",
            "features": {"throughput": 1.2, "memory_bw": 1.0},
        },
    ]


class Stage4UncertaintyReplayV1Tests(unittest.TestCase):
    def test_hypervolume_ignores_dominated_points(self) -> None:
        frontier = [(0.2, 0.4, 0.5), (0.4, 0.2, 0.3)]
        dominated = (0.8, 0.8, 0.8)

        self.assertAlmostEqual(
            replay._hypervolume(frontier),
            replay._hypervolume([*frontier, dominated]),
        )

    def test_ehvi_uses_one_shared_bound_set_per_model(self) -> None:
        rows = _rows(10)
        encoded = replay.encode_rows(rows, _graph_features(rows), _profiles())
        measured = {"codriving|g00", "pyramid|g01"}
        candidates = sorted({str(row["group_id"]) for row in rows} - measured)
        context = replay._candidate_acquisition_context(
            rows, encoded.matrix, measured, candidates
        )

        first = replay._acquisition_bounds(rows, context, "codriving")
        second = replay._acquisition_bounds(rows, context, "codriving")
        self.assertEqual(first, second)
        for target in replay.TARGETS:
            candidate_values = [
                float(context["means"][target][local_index])
                for local_index, row_index in enumerate(context["candidate_indices"])
                if rows[row_index]["model"] == "codriving"
            ]
            self.assertLessEqual(first[target][0], min(candidate_values))
            self.assertGreaterEqual(first[target][1], max(candidate_values))

    def test_conformal_split_uses_about_twenty_percent_groups_per_model(self) -> None:
        rows = _rows(20)
        groups = sorted({str(row["group_id"]) for row in rows})

        fit, calibration = replay._split_fit_calibration_groups(groups, rows, seed=5)

        self.assertFalse(set(fit) & set(calibration))
        self.assertEqual(set(fit) | set(calibration), set(groups))
        for model in ("codriving", "pyramid"):
            model_calibration = [group for group in calibration if group.startswith(model)]
            self.assertEqual(len(model_calibration), 2)

    def test_grouped_oof_uncertainty_has_all_rows_once_and_group_conformal_metrics(self) -> None:
        rows = _rows()

        report = replay.run_grouped_oof_uncertainty(
            rows,
            _graph_features(rows),
            _profiles(),
            n_splits=5,
            seed=7,
        )

        self.assertEqual(report["schema_version"], "stage4_uncertainty_oof_v1")
        self.assertEqual(report["mode"], "grouped_outer_oof_uncertainty")
        for method in ("lgbm_quantile", "extra_trees_ensemble"):
            method_report = report["methods"][method]
            self.assertEqual(method_report["oof_row_count"], len(rows))
            self.assertGreaterEqual(method_report["simultaneous_all_targets_group_coverage"], 0.0)
            self.assertLessEqual(method_report["simultaneous_all_targets_group_coverage"], 1.0)
            self.assertEqual(
                sorted(method_report["oof_manifest_job_ids"]),
                sorted(row["manifest_job_id"] for row in rows),
            )
            for target in ("latency_ms", "energy_j", "ap70"):
                metrics = method_report["targets"][target]
                self.assertGreaterEqual(metrics["row_coverage"], 0.0)
                self.assertLessEqual(metrics["row_coverage"], 1.0)
                self.assertGreaterEqual(metrics["fully_covered_group_coverage"], 0.0)
                self.assertLessEqual(metrics["fully_covered_group_coverage"], 1.0)
                self.assertGreater(metrics["mean_width"], 0.0)
                self.assertGreater(metrics["calibration_group_count"], 0)

        test_groups = []
        for fold in report["folds"]:
            self.assertFalse(set(fold["train_groups"]) & set(fold["test_groups"]))
            self.assertFalse(set(fold["fit_groups"]) & set(fold["calibration_groups"]))
            self.assertEqual(set(fold["train_groups"]), set(fold["fit_groups"]) | set(fold["calibration_groups"]))
            test_groups.extend(fold["test_groups"])
        self.assertEqual(sorted(test_groups), sorted({str(row["group_id"]) for row in rows}))

    def test_simultaneous_group_coverage_requires_every_target_and_arm(self) -> None:
        intervals = {
            "latency_ms": [
                {"group_id": "g1", "truth": 1.0, "lower": 0.0, "upper": 2.0},
                {"group_id": "g2", "truth": 1.0, "lower": 0.0, "upper": 2.0},
            ],
            "energy_j": [
                {"group_id": "g1", "truth": 1.0, "lower": 0.0, "upper": 2.0},
                {"group_id": "g2", "truth": 3.0, "lower": 0.0, "upper": 2.0},
            ],
            "ap70": [
                {"group_id": "g1", "truth": 1.0, "lower": 0.0, "upper": 2.0},
                {"group_id": "g2", "truth": 1.0, "lower": 0.0, "upper": 2.0},
            ],
        }

        self.assertEqual(replay._simultaneous_group_coverage(intervals), 0.5)

    def test_grouped_oof_uncertainty_is_deterministic(self) -> None:
        rows = _rows()

        first = replay.run_grouped_oof_uncertainty(
            rows, _graph_features(rows), _profiles(), n_splits=5, seed=11
        )
        second = replay.run_grouped_oof_uncertainty(
            rows, _graph_features(rows), _profiles(), n_splits=5, seed=11
        )

        self.assertEqual(first["methods"], second["methods"])
        self.assertEqual(first["folds"], second["folds"])

    def test_offline_replay_samples_complete_groups_without_repeats_and_threshold_is_correct(self) -> None:
        rows = _rows(9)

        report = replay.run_offline_closed_loop_replay(
            rows,
            _graph_features(rows),
            _profiles(),
            initial_group_count=2,
            budget_groups=6,
            seeds=(3, 4),
        )

        self.assertEqual(report["schema_version"], "stage4_offline_closed_loop_replay_v2")
        self.assertEqual(report["mode"], "offline_replay")
        self.assertEqual(
            set(report["policies"]),
            {
                "pareto_uncertainty",
                "uncertainty_only",
                "expected_hv_improvement",
                "predicted_frontier_diversity",
                "ehvi_uncertainty_diversity",
                "random",
            },
        )
        self.assertGreater(report["oracle_hv"], 0.0)
        self.assertAlmostEqual(report["threshold_hv"], report["oracle_hv"] * 0.95)
        self.assertEqual(report["hv_aggregation"], "mean_model_internal_oracle_fraction")
        self.assertEqual(set(report["oracle_hv_by_model"]), {"codriving", "pyramid"})
        for policy, policy_report in report["policies"].items():
            self.assertIn("samples_to_95pct_oracle_HV", policy_report)
            self.assertIn("groups_to_95pct_oracle_HV_median", policy_report)
            self.assertIn("samples_to_95pct_oracle_HV_median", policy_report)
            self.assertGreaterEqual(policy_report["success_rate"], 0.0)
            self.assertLessEqual(policy_report["success_rate"], 1.0)
            for trace in policy_report["traces"]:
                sampled = trace["sampled_groups"]
                self.assertEqual(len(sampled), len(set(sampled)))
                self.assertLessEqual(len(sampled), 6)
                groups_to_threshold = trace["groups_to_95pct_oracle_HV"]
                samples_to_threshold = trace["samples_to_95pct_oracle_HV"]
                if groups_to_threshold is None:
                    self.assertIsNone(samples_to_threshold)
                else:
                    self.assertEqual(samples_to_threshold, groups_to_threshold * len(ARMS))
                for step in trace["steps"]:
                    self.assertEqual(len(step["measured_manifest_job_ids"]), len(ARMS))
                    self.assertEqual(len(set(step["measured_manifest_job_ids"])), len(ARMS))
                    self.assertTrue(step["candidate_features_visible"])
                    self.assertFalse(step["candidate_labels_visible_before_measurement"])
                    self.assertEqual(step["initial_groups"], trace["initial_groups"])

        again = replay.run_offline_closed_loop_replay(
            rows,
            _graph_features(rows),
            _profiles(),
            initial_group_count=2,
            budget_groups=6,
            seeds=(3, 4),
        )
        self.assertEqual(report, again)

    def test_offline_replay_requires_graph_and_capability_context(self) -> None:
        rows = _rows(9)
        incomplete_graph = _graph_features(rows)[1:]

        with self.assertRaisesRegex(ValueError, "missing feature context"):
            replay.run_offline_closed_loop_replay(
                rows,
                incomplete_graph,
                _profiles(),
                initial_group_count=2,
                budget_groups=3,
                seeds=(3,),
            )

    def test_unmeasured_candidate_label_changes_do_not_change_next_selection(self) -> None:
        rows = _rows(9)
        baseline = replay.run_offline_closed_loop_replay(
            rows,
            _graph_features(rows),
            _profiles(),
            initial_group_count=2,
            budget_groups=3,
            seeds=(13,),
        )
        for policy in sorted(set(baseline["policies"]) - {"random"}):
            with self.subTest(policy=policy):
                first_trace = baseline["policies"][policy]["traces"][0]
                baseline_choice = first_trace["steps"][0]["selected_group"]
                initial_groups = set(first_trace["initial_groups"])
                candidate_groups = {str(row["group_id"]) for row in rows} - initial_groups

                changed = [dict(row) for row in rows]
                for row in changed:
                    if str(row["group_id"]) in candidate_groups:
                        row["ap70"] = 10.0 if str(row["group_id"]) != baseline_choice else -10.0
                        row["latency_ms"] = 0.01 if str(row["group_id"]) != baseline_choice else 999.0
                        row["energy_j"] = 0.01 if str(row["group_id"]) != baseline_choice else 999.0

                replayed = replay.run_offline_closed_loop_replay(
                    changed,
                    _graph_features(changed),
                    _profiles(),
                    initial_group_count=2,
                    budget_groups=3,
                    seeds=(13,),
                )

                self.assertEqual(
                    replayed["policies"][policy]["traces"][0]["steps"][0]["selected_group"],
                    baseline_choice,
                )

    def test_new_acquisition_policies_emit_leakage_free_diagnostics(self) -> None:
        rows = _rows(9)
        report = replay.run_offline_closed_loop_replay(
            rows,
            _graph_features(rows),
            _profiles(),
            initial_group_count=2,
            budget_groups=3,
            seeds=(17,),
        )

        expected_fields = {
            "uncertainty_only": "selected_group_uncertainty",
            "expected_hv_improvement": "selected_group_expected_hv_improvement",
            "predicted_frontier_diversity": "selected_group_feature_diversity",
            "ehvi_uncertainty_diversity": "selected_group_combined_score",
        }
        for policy, field in expected_fields.items():
            step = report["policies"][policy]["traces"][0]["steps"][0]
            self.assertIn(field, step)
            self.assertTrue(math.isfinite(float(step[field])))
            self.assertEqual(step["acquisition_fit_scope"], "measured_complete_groups_only")
            self.assertFalse(step["candidate_labels_visible_before_measurement"])


if __name__ == "__main__":
    unittest.main()
