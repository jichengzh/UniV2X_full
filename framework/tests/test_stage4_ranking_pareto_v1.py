from __future__ import annotations

import unittest

import numpy as np

from framework.stage4 import cost_model_selection_v1 as selection
from framework.stage4 import ranking_pareto_v1 as ranking


def _profiles() -> list[dict]:
    return [
        {"capability_profile_id": "tvm-profile", "features": {"int8": 0.2}},
        {"capability_profile_id": "trt-profile", "features": {"int8": 0.8}},
    ]


def _rows(group_count: int = 12) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    graph: list[dict] = []
    for group_index in range(group_count):
        model = "pyramid" if group_index % 2 == 0 else "codriving"
        width = [16 + group_index * 4, 32 + group_index * 4, 64 + group_index * 8]
        group_id = f"{model}|g{group_index}"
        graph.append(
            {
                "group_id": group_id,
                "conv_flops": float(np.prod(width)),
                "conv_output_elements": float(sum(width)),
            }
        )
        for backend, profile in (("tvm_auto", "tvm-profile"), ("trt_engine", "trt-profile")):
            for q_mode in ("fp16", "int8"):
                row_index = len(rows)
                is_best = backend == "trt_engine" and q_mode == "fp16"
                rows.append(
                    {
                        "manifest_job_id": f"{group_id}|{backend}|{q_mode}",
                        "group_id": group_id,
                        "model": model,
                        "width": width,
                        "q_mode": q_mode,
                        "dispatch_key": backend,
                        "capability_profile_id": profile,
                        "latency_ms": 1.0 + group_index * 0.1 + (0.0 if is_best else 0.8),
                        "energy_j": 0.2 + group_index * 0.02 + (0.0 if is_best else 0.2),
                        "ap70": 0.9 - group_index * 0.01 - (0.0 if is_best else 0.12),
                        "row_order": row_index,
                    }
                )
    return rows, graph


def _perfect_value_report(rows: list[dict]) -> dict:
    groups = sorted({str(row["group_id"]) for row in rows})
    fold_by_group = {group_id: index % min(3, len(groups)) for index, group_id in enumerate(groups)}
    folds = []
    for fold_index in sorted(set(fold_by_group.values())):
        test_groups = sorted(group for group, fold in fold_by_group.items() if fold == fold_index)
        train_groups = sorted(set(groups) - set(test_groups))
        folds.append(
            {
                "outer_fold": fold_index,
                "train_groups": train_groups,
                "test_groups": test_groups,
            }
        )
    targets = {}
    for target in ("latency_ms", "energy_j", "ap70"):
        targets[target] = {
            "outer_folds": folds,
            "oof_predictions": [
                {
                    "row_id": row["manifest_job_id"],
                    "group_id": row["group_id"],
                    "model": row["model"],
                    "truth": row[target],
                    "prediction": row[target],
                    "outer_fold": fold_by_group[str(row["group_id"])],
                }
                for row in rows
            ]
        }
    return {
        "schema_version": "stage4_cost_model_selection_v1",
        "split_protocol": "nested_grouped_cv_by_model_width",
        "targets": targets,
    }


class Stage4RankingParetoV1Tests(unittest.TestCase):
    def test_hv3_matches_single_box_known_answer(self) -> None:
        value = ranking._hv3([(0.2, 0.3, 0.4)], (1.05, 1.05, 1.05))

        self.assertAlmostEqual(value, 0.85 * 0.75 * 0.65)

    def test_pareto_rejects_predictions_without_grouped_oof_provenance(self) -> None:
        rows, _ = _rows(6)
        report = _perfect_value_report(rows)
        report.pop("split_protocol")

        with self.assertRaisesRegex(ValueError, "OOF provenance"):
            ranking.evaluate_pareto_from_value_oof(rows, report)

    def test_ranker_oof_uses_grouped_outer_folds_and_unique_rows(self) -> None:
        rows, graph = _rows()
        encoded = selection.encode_rows(rows, graph, _profiles())

        report = ranking.run_ranker_oof(
            rows,
            encoded_rows=encoded,
            outer_splits=3,
            seed=17,
        )

        self.assertEqual(report["schema_version"], "stage4_ranking_pareto_ranker_v1")
        self.assertEqual(len(report["oof_scores"]), len(rows))
        self.assertEqual(
            len({row["row_id"] for row in report["oof_scores"]}),
            len(rows),
        )
        for fold in report["outer_folds"]:
            self.assertFalse(set(fold["train_groups"]) & set(fold["test_groups"]))
            self.assertGreater(fold["test_rows"], 0)
        self.assertEqual(set(report["top10_recall_by_model"]), {"codriving", "pyramid"})

    def test_pareto_metrics_use_value_oof_coordinates_not_ranker_score(self) -> None:
        rows = [
            {
                "manifest_job_id": "a",
                "group_id": "ga",
                "model": "pyramid",
                "latency_ms": 1.0,
                "energy_j": 1.0,
                "ap70": 0.9,
            },
            {
                "manifest_job_id": "b",
                "group_id": "gb",
                "model": "pyramid",
                "latency_ms": 2.0,
                "energy_j": 2.0,
                "ap70": 0.8,
            },
            {
                "manifest_job_id": "c",
                "group_id": "gc",
                "model": "pyramid",
                "latency_ms": 3.0,
                "energy_j": 3.0,
                "ap70": 0.7,
            },
        ]
        value_report = _perfect_value_report(rows)
        ranker_report = {
            "oof_scores": [
                {"row_id": "a", "score": -100.0},
                {"row_id": "b", "score": 100.0},
                {"row_id": "c", "score": 200.0},
            ]
        }

        report = ranking.evaluate_pareto_from_value_oof(
            rows,
            value_report,
            ranker_report=ranker_report,
        )

        metrics = report["by_model"]["pyramid"]
        self.assertEqual(metrics["true_frontier"], ["a"])
        self.assertEqual(metrics["predicted_frontier"], ["a"])
        self.assertEqual(metrics["pareto_recall"], 1.0)
        self.assertEqual(metrics["pareto_precision"], 1.0)
        self.assertEqual(metrics["hv_regret"], 0.0)

    def test_pareto_known_example_reports_missed_frontier_and_hv_regret(self) -> None:
        rows = [
            {
                "manifest_job_id": "fast",
                "group_id": "g1",
                "model": "codriving",
                "latency_ms": 1.0,
                "energy_j": 1.0,
                "ap70": 0.7,
            },
            {
                "manifest_job_id": "accurate",
                "group_id": "g2",
                "model": "codriving",
                "latency_ms": 2.0,
                "energy_j": 2.0,
                "ap70": 0.9,
            },
            {
                "manifest_job_id": "dominated",
                "group_id": "g3",
                "model": "codriving",
                "latency_ms": 3.0,
                "energy_j": 3.0,
                "ap70": 0.6,
            },
        ]
        value_report = _perfect_value_report(rows)
        for target, target_payload in value_report["targets"].items():
            for prediction in target_payload["oof_predictions"]:
                if prediction["row_id"] == "fast":
                    prediction["prediction"] = 0.1 if target == "ap70" else 9.0

        report = ranking.evaluate_pareto_from_value_oof(rows, value_report)

        metrics = report["by_model"]["codriving"]
        self.assertEqual(metrics["true_frontier"], ["accurate", "fast"])
        self.assertEqual(metrics["predicted_frontier"], ["accurate"])
        self.assertEqual(metrics["pareto_recall"], 0.5)
        self.assertEqual(metrics["pareto_precision"], 1.0)
        self.assertGreater(metrics["hv_regret"], 0.0)

    def test_build_report_combines_ranker_and_pareto_outputs(self) -> None:
        rows, graph = _rows()
        report = ranking.build_stage4_ranking_pareto_report(
            rows,
            graph_features=graph,
            capability_profiles=_profiles(),
            cost_model_selection_report=_perfect_value_report(rows),
            outer_splits=3,
            seed=19,
        )

        self.assertEqual(report["schema_version"], "stage4_ranking_pareto_v1")
        self.assertIn("ranker", report)
        self.assertIn("pareto", report)
        self.assertEqual(report["ranker"]["row_count"], len(rows))
        self.assertEqual(set(report["pareto"]["by_model"]), {"codriving", "pyramid"})
        comparison = report["ranker_value_comparison"]
        self.assertEqual(
            set(comparison["value_top10_recall_by_model"]),
            {"codriving", "pyramid"},
        )
        self.assertEqual(
            set(comparison["ranker_delta_by_model"]),
            {"codriving", "pyramid"},
        )
        self.assertIsInstance(comparison["retain_ranker"], bool)


if __name__ == "__main__":
    unittest.main()
