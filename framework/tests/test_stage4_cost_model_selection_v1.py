from __future__ import annotations

import copy
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from framework.stage4 import cost_model_selection_v1 as selection


REPO = Path(__file__).resolve().parents[2]
CLI = REPO / "scripts/stage4_cost_model_selection_v1.py"


def _profiles() -> list[dict]:
    return [
        {
            "capability_profile_id": "tvm-profile",
            "features": {"int8_propagation": 0.0, "qdq_fold": 0.0},
        },
        {
            "capability_profile_id": "trt-profile",
            "features": {"int8_propagation": 1.0, "qdq_fold": 0.8},
        },
    ]


def _dataset(group_count: int = 12) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    graph: list[dict] = []
    for group_index in range(group_count):
        model = "pyramid" if group_index % 2 == 0 else "codriving"
        width = [16 + 8 * group_index, 32 + 8 * group_index, 64 + 16 * group_index]
        group_id = f"{model}|{'x'.join(map(str, width))}"
        graph.append(
            {
                "group_id": group_id,
                "conv_flops": float(np.prod(width) * 1000),
                "conv_output_elements": float(sum(width) * 100),
                "arithmetic_intensity_proxy": float(width[-1] / width[0]),
            }
        )
        for backend, profile in (("tvm_auto", "tvm-profile"), ("trt_engine", "trt-profile")):
            for q_mode in ("fp16", "int8"):
                scale = 1.0 + group_index / 10.0
                int8_ratio = 1.25 if backend == "tvm_auto" else 0.72
                latency = scale * (int8_ratio if q_mode == "int8" else 1.0)
                rows.append(
                    {
                        "manifest_job_id": f"{group_id}|{backend}|{q_mode}",
                        "group_id": group_id,
                        "model": model,
                        "width": width,
                        "q_mode": q_mode,
                        "dispatch_key": backend,
                        "capability_profile_id": profile,
                        "terminal_status": "measured_success_gold",
                        "latency_ms": latency,
                        "energy_j": latency * 0.2,
                        "ap70": 0.65 - 0.01 * group_index - (0.015 if q_mode == "int8" else 0.0),
                    }
                )
    return rows, graph


class Stage4CostModelSelectionV1Tests(unittest.TestCase):
    def test_encoder_has_stable_four_axis_and_fcooper_model_context(self) -> None:
        rows, graph = _dataset(group_count=2)
        fcooper = {
            **rows[0],
            "manifest_job_id": "fcooper-row",
            "row_id": "fcooper-row",
            "group_id": (
                "fcooper|backbone.s0=64|backbone.s1=128|backbone.s2=256|"
                "neck.deblock=128|neck.output=256"
            ),
            "model": "fcooper",
            "width": [64, 128, 256, 128, 256],
            "width_schema": [
                "backbone.s0",
                "backbone.s1",
                "backbone.s2",
                "neck.deblock",
                "neck.output",
            ],
        }
        combined = [*rows, fcooper]
        graph = [
            *graph,
            {
                "group_id": fcooper["group_id"],
                "model": "fcooper",
                "width": fcooper["width"],
                "conv_count": 31,
            },
        ]

        encoded = selection.encode_rows(combined, graph, _profiles())

        self.assertIn("width:axis4", encoded.feature_names)
        self.assertIn("model:fcooper", encoded.feature_names)
        self.assertEqual(encoded.matrix.shape[0], len(combined))
        self.assertEqual(encoded.matrix.shape[1], len(encoded.feature_names))

    def test_grouped_folds_never_split_four_arm_groups(self) -> None:
        rows, _ = _dataset()
        folds = selection.grouped_folds(rows, n_splits=4, seed=7)

        self.assertEqual(len(folds), 4)
        all_groups = {row["group_id"] for row in rows}
        seen_test: set[str] = set()
        for fold in folds:
            train_groups = set(fold.train_groups)
            test_groups = set(fold.test_groups)
            self.assertFalse(train_groups & test_groups)
            self.assertEqual(train_groups | test_groups, all_groups)
            self.assertEqual(sum(row["group_id"] in test_groups for row in rows) % 4, 0)
            seen_test |= test_groups
        self.assertEqual(seen_test, all_groups)

    def test_feature_encoder_uses_numeric_capability_not_backend_name(self) -> None:
        rows, graph = _dataset()
        encoded = selection.encode_rows(rows, graph, _profiles())

        self.assertEqual(encoded.matrix.shape[0], len(rows))
        self.assertIn("cap:int8_propagation", encoded.feature_names)
        self.assertIn("q:int8", encoded.feature_names)
        self.assertFalse(any("tvm" in name or "trt" in name for name in encoded.feature_names))

    def test_feature_encoder_rejects_label_like_context_fields(self) -> None:
        rows, graph = _dataset()
        graph[0]["latency_ms"] = 1.0
        with self.assertRaisesRegex(ValueError, "label-like"):
            selection.encode_rows(rows, graph, _profiles())

        rows, graph = _dataset()
        profiles = _profiles()
        profiles[0]["features"]["ap70"] = 0.5
        with self.assertRaisesRegex(ValueError, "label-like"):
            selection.encode_rows(rows, graph, profiles)

        rows, graph = _dataset()
        graph[0]["terminal_status"] = 1
        with self.assertRaisesRegex(ValueError, "label-like"):
            selection.encode_rows(rows, graph, _profiles())

    def test_nested_selection_is_grouped_and_does_not_mutate_input(self) -> None:
        rows, graph = _dataset()
        original = copy.deepcopy(rows)

        report = selection.run_nested_selection(
            rows,
            graph,
            _profiles(),
            outer_splits=3,
            inner_splits=2,
            seed=11,
            candidates=("extra_trees_raw", "extra_trees_log", "lgbm_l1_raw"),
        )

        self.assertEqual(rows, original)
        self.assertEqual(report["schema_version"], "stage4_cost_model_selection_v1")
        self.assertEqual(set(report["targets"]), {"latency_ms", "energy_j", "ap70"})
        for target, payload in report["targets"].items():
            self.assertEqual(len(payload["outer_folds"]), 3)
            self.assertGreaterEqual(payload["summary"]["spearman"], -1.0)
            self.assertEqual(len(payload["oof_predictions"]), len(rows))
            self.assertEqual(
                len({row["row_id"] for row in payload["oof_predictions"]}),
                len(rows),
            )
            for fold in payload["outer_folds"]:
                self.assertFalse(set(fold["train_groups"]) & set(fold["test_groups"]))
                self.assertIn(fold["selected_candidate"], fold["inner_candidate_scores"])
                self.assertGreater(fold["test_rows"], 0)

    def test_huber_and_median_quantile_losses_are_valid_candidates(self) -> None:
        rows, graph = _dataset()

        report = selection.run_nested_selection(
            rows,
            graph,
            _profiles(),
            outer_splits=3,
            inner_splits=2,
            seed=13,
            candidates=("lgbm_huber_raw", "lgbm_quantile_raw"),
        )

        self.assertEqual(report["candidate_specs"]["lgbm_huber_raw"]["objective"], "huber")
        self.assertEqual(
            report["candidate_specs"]["lgbm_quantile_raw"]["objective"],
            "quantile_p50",
        )
        for payload in report["targets"].values():
            for fold in payload["outer_folds"]:
                self.assertEqual(
                    set(fold["inner_candidate_scores"]),
                    {"lgbm_huber_raw", "lgbm_quantile_raw"},
                )

    def test_quantile_interval_is_ordered_after_inverse_transform(self) -> None:
        rows, graph = _dataset()
        encoded = selection.encode_rows(rows, graph, _profiles())
        y = np.asarray([row["latency_ms"] for row in rows], dtype=float)

        interval = selection.fit_predict_quantile_interval(
            encoded.matrix[:32],
            y[:32],
            encoded.matrix[32:],
            transform="log1p",
            seed=3,
        )

        self.assertTrue(np.all(interval.lower <= interval.median))
        self.assertTrue(np.all(interval.median <= interval.upper))
        self.assertTrue(np.all(interval.lower >= 0.0))

    def test_cli_writes_machine_readable_report_and_summary(self) -> None:
        rows, graph = _dataset()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gold_path = root / "gold.json"
            graph_path = root / "graph.json"
            profile_path = root / "profiles.json"
            output_dir = root / "output"
            gold_path.write_text(json.dumps(rows), encoding="utf-8")
            graph_path.write_text(json.dumps(graph), encoding="utf-8")
            profile_path.write_text(json.dumps(_profiles()), encoding="utf-8")

            subprocess.run(
                [
                    sys.executable,
                    str(CLI),
                    "--gold-json",
                    str(gold_path),
                    "--graph-features-json",
                    str(graph_path),
                    "--capability-profiles-json",
                    str(profile_path),
                    "--output-dir",
                    str(output_dir),
                    "--outer-splits",
                    "3",
                    "--inner-splits",
                    "2",
                    "--candidates",
                    "extra_trees_raw,lgbm_l1_raw",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            report = json.loads((output_dir / "stage4_cost_model_selection_report.json").read_text())
            with (output_dir / "model_family_summary.csv").open(newline="") as handle:
                summary = list(csv.DictReader(handle))
        self.assertEqual(report["group_count"], 12)
        self.assertEqual({row["target"] for row in summary}, set(selection.TARGETS))


if __name__ == "__main__":
    unittest.main()
