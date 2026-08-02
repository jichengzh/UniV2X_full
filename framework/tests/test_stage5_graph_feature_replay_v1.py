from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import stage5_graph_feature_replay_v1 as replay  # noqa: E402


class Stage5GraphFeatureReplayV1Tests(unittest.TestCase):
    @staticmethod
    def _actual_view_rows():
        historical = []
        actual = []
        round_by_id = {}
        for task_index, task_id in enumerate(replay.TASK_SPECS):
            for round_index in range(4):
                for offset in range(4):
                    row_id = f"{task_id}-{round_index}-{offset}"
                    graph = {
                        "group_id": f"group-{task_index}-{round_index}-{offset}",
                        "graph_feature_provenance": "coldstart_width_conditioned_surrogate_v1",
                        "conv_count": 20.5,
                    }
                    source = {
                        "task_id": task_id,
                        "manifest_job_id": row_id,
                        "row_id": row_id,
                        "group_id": graph["group_id"],
                        "model": "pyramid" if "PYR" in task_id else "codriving",
                        "width": [16, 32, 64],
                        "q_mode": "fp16",
                        "dispatch_key": "tvm_auto" if "TVM" in task_id else "trt_engine",
                        "capability_profile_id": f"profile-{task_index}",
                        "terminal_status": "measured_success_gold",
                        "latency_ms": 1.0,
                        "energy_j": 0.5,
                        "ap30": 0.7,
                        "ap50": 0.6,
                        "ap70": 0.4,
                        "graph_features": graph,
                    }
                    historical.append(source)
                    derived = {
                        **source,
                        "candidate_graph_features": graph,
                        "candidate_graph_features_sha256": replay._sha_payload(graph),
                        "graph_features": {
                            **graph,
                            "conv_count": 21,
                            "graph_feature_provenance": "materialized_onnx_extracted_v1",
                        },
                        "materialized_graph_features_sha256": hashlib.sha256(
                            row_id.encode()
                        ).hexdigest(),
                        "graph_feature_backfill_round_index": round_index,
                        "derived_view_only": True,
                        "historical_feedback_row_sha256": replay._sha_payload(source),
                        "historical_request_identity_matches_visible_row": False,
                    }
                    derived["derived_view_row_sha256"] = replay._sha_payload(derived)
                    actual.append(derived)
                    round_by_id[row_id] = (task_id, round_index)
        return historical, actual, round_by_id

    def test_selection_comparison_is_set_based_and_deterministic(self) -> None:
        result = replay.selection_comparison(
            ["a", "b", "c", "d"], ["b", "c", "e", "f"]
        )
        self.assertFalse(result["exact_match"])
        self.assertEqual(result["overlap_count"], 2)
        self.assertEqual(result["jaccard"], 2 / 6)
        self.assertEqual(result["historical_only"], ["a", "d"])
        self.assertEqual(result["actual_replay_only"], ["e", "f"])

    def test_objective_projection_is_unchanged_by_graph_replacement(self) -> None:
        source = {
            "manifest_job_id": "row",
            "terminal_status": "measured_success_gold",
            "latency_ms": 1.2,
            "energy_j": 0.4,
            "ap30": 0.7,
            "ap50": 0.6,
            "ap70": 0.5,
            "graph_features": {"conv_count": 20.5},
        }
        actual = copy.deepcopy(source)
        actual["graph_features"] = {"conv_count": 21}
        self.assertEqual(
            replay.objective_projection_sha256([source]),
            replay.objective_projection_sha256([actual]),
        )

    def test_prediction_delta_reports_relative_change(self) -> None:
        result = replay.prediction_delta(
            {
                "predictions": {
                    "latency_ms": 2.0,
                    "energy_j": 1.0,
                    "ap70": 0.5,
                }
            },
            {
                "predictions": {
                    "latency_ms": 2.2,
                    "energy_j": 0.9,
                    "ap70": 0.55,
                }
            },
        )
        self.assertAlmostEqual(result["latency_ms"]["relative_change"], 0.1)
        self.assertAlmostEqual(result["energy_j"]["relative_change"], -0.1)
        self.assertAlmostEqual(result["ap70"]["absolute_change"], 0.05)

    def test_actual_view_round_is_bound_to_historical_directory(self) -> None:
        historical, actual, round_by_id = self._actual_view_rows()
        replay._validate_actual_view(actual, historical, round_by_id)
        actual[4]["graph_feature_backfill_round_index"] = 0
        visible = {
            key: value
            for key, value in actual[4].items()
            if key != "derived_view_row_sha256"
        }
        actual[4]["derived_view_row_sha256"] = replay._sha_payload(visible)
        with self.assertRaisesRegex(ValueError, "invalid actual graph feedback view"):
            replay._validate_actual_view(actual, historical, round_by_id)

    def test_online_pareto_is_grouped_by_task_identity(self) -> None:
        rows = [
            {
                "task_id": "S5-PYR-TVM",
                "manifest_job_id": "dominant",
                "terminal_status": "measured_success_gold",
                "latency_ms": 1.0,
                "energy_j": 1.0,
                "ap70": 0.8,
            },
            {
                "task_id": "S5-PYR-TVM",
                "manifest_job_id": "dominated",
                "terminal_status": "measured_success_gold",
                "latency_ms": 2.0,
                "energy_j": 2.0,
                "ap70": 0.7,
            },
        ]
        frontiers = replay.online_pareto_ids_by_task(rows)
        self.assertEqual(frontiers["S5-PYR-TVM"], ["dominant"])

    def test_run_replay_uses_only_completed_historical_prefix(self) -> None:
        historical, actual, _ = self._actual_view_rows()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            formal = root / "formal"
            for task_id in replay.TASK_SPECS:
                for round_index in range(4):
                    rows = [
                        row
                        for row in historical
                        if row["task_id"] == task_id
                        and row["manifest_job_id"].startswith(
                            f"{task_id}-{round_index}-"
                        )
                    ]
                    round_root = formal / task_id / f"round_{round_index:02d}"
                    (round_root / "final").mkdir(parents=True, exist_ok=True)
                    (round_root / "measurement_request.json").write_text(
                        json.dumps({"rows": rows})
                    )
                    (round_root / "final/stage5_feedback_v2_final.json").write_text(
                        json.dumps(rows)
                    )
                    predicted = [
                        {
                            **row,
                            "predictions": {
                                "latency_ms": 1.0,
                                "energy_j": 0.5,
                                "ap70": 0.4,
                            },
                        }
                        for row in historical
                        if row["task_id"] == task_id
                    ]
                    (round_root / "predicted_candidates.json").write_text(
                        json.dumps({"rows": predicted})
                    )
            cold_rows = root / "gold.json"
            cold_rows.write_text(
                json.dumps(
                    [
                        {
                            "manifest_job_id": f"cold-{index}",
                            "group_id": f"cold-group-{index}",
                        }
                        for index in range(176)
                    ]
                )
            )
            cold_graphs = root / "graphs.json"
            cold_graphs.write_text(json.dumps({"graph_features": []}))
            profiles = root / "profiles.json"
            profiles.write_text(
                json.dumps(
                    {
                        "capability_profiles": [
                            {"dispatch_key": "tvm_auto"},
                            {"dispatch_key": "trt_engine"},
                        ]
                    }
                )
            )
            registry = root / "registry.json"
            registry.write_text(json.dumps({}))
            actual_path = root / "actual.json"
            actual_path.write_text(json.dumps({"rows": actual}))

            def fake_manifest(_registry, *, task, measured_row_ids):
                candidates = [
                    {
                        **row,
                        "graph_features": row["graph_features"],
                    }
                    for row in historical
                    if row["task_id"] == task.task_id
                    and row["manifest_job_id"] not in measured_row_ids
                ]
                return {"rows": candidates}

            def fake_predict(_bundle, candidates, _profiles):
                return [
                    {
                        **row,
                        "predictions": {
                            "latency_ms": 1.0,
                            "energy_j": 0.5,
                            "ap70": 0.4,
                        },
                    }
                    for row in candidates
                ]

            def fake_select(predicted, *_args, **_kwargs):
                selected = sorted(predicted, key=lambda row: row["manifest_job_id"])[:4]
                return {
                    "selected_row_ids": [row["manifest_job_id"] for row in selected]
                }

            with (
                patch.object(replay, "verify_frozen_coldstart_artifacts", return_value={}),
                patch.object(replay, "freeze_initial_coldstart", side_effect=lambda rows: rows),
                patch.object(replay, "validate_full_source_registry", return_value={}),
                patch.object(replay, "validate_task_feedback_history", return_value={}),
                patch.object(replay, "fit_online_bundle", return_value=object()),
                patch.object(replay, "build_task_candidate_manifest", side_effect=fake_manifest),
                patch.object(replay, "predict_candidate_rows", side_effect=fake_predict),
                patch.object(replay, "select_task_batch", side_effect=fake_select),
            ):
                report = replay.run_replay(
                    formal_root=formal,
                    coldstart_rows_path=cold_rows,
                    coldstart_graphs_path=cold_graphs,
                    profiles_path=profiles,
                    registry_path=registry,
                    actual_feedback_view_path=actual_path,
                )

            self.assertEqual(report["summary"]["teacher_forced_replay_round_count"], 12)
            self.assertEqual(report["summary"]["exact_match_round_count"], 12)
            self.assertEqual(report["summary"]["selection_divergence_round_count"], 0)


if __name__ == "__main__":
    unittest.main()
