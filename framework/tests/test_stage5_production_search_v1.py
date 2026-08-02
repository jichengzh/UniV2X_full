from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from framework.stage2.canonical_search_v3 import build_capability_profile
from framework.stage5 import production_search_v1 as stage5


def _profiles() -> list[dict]:
    return [
        build_capability_profile(
            capability_profile_id="h800-tvm-probe-conditioned-v3",
            hardware_target="h800",
            compiler_fingerprint=hashlib.sha256(b"tvm-fingerprint").hexdigest(),
            dispatch_key="tvm_auto",
            features={"int8_propagation": 0.0, "qdq_fold": 0.0},
        ),
        build_capability_profile(
            capability_profile_id="h800-trt-probe-conditioned-v3",
            hardware_target="h800",
            compiler_fingerprint=hashlib.sha256(b"trt-fingerprint").hexdigest(),
            dispatch_key="trt_engine",
            features={"int8_propagation": 1.0, "qdq_fold": 0.8},
        ),
    ]


def _graph(group_id: str, model: str, width: list[int]) -> dict:
    return {
        "group_id": group_id,
        "model": model,
        "width": width,
        "conv_count": 24 if model == "codriving" else 27,
        "conv_macs": float(width[0] * width[1] * width[2]),
        "group_conv_count": 0 if model == "codriving" else 3,
    }


def _training_rows() -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    graphs: list[dict] = []
    widths = ([16, 32, 64], [24, 48, 96], [32, 64, 128])
    for model_index, model in enumerate(("pyramid", "codriving")):
        for width_index, width in enumerate(widths):
            group_id = f"{model}|{'x'.join(map(str, width))}"
            graphs.append(_graph(group_id, model, list(width)))
            for profile in _profiles():
                for q_mode in ("fp16", "int8"):
                    tvm = profile["dispatch_key"] == "tvm_auto"
                    int8 = q_mode == "int8"
                    base = 1.0 + model_index + 0.2 * width_index
                    latency = base * (1.25 if tvm and int8 else 0.75 if not tvm and int8 else 1.0)
                    energy = latency * 0.25
                    ap70 = 0.72 - 0.03 * width_index - (0.01 if int8 else 0.0)
                    row_id = f"{group_id}|q={q_mode}|profile={profile['capability_profile_id']}"
                    rows.append(
                        {
                            "manifest_job_id": row_id,
                            "group_id": group_id,
                            "model": model,
                            "width": list(width),
                            "dispatch_key": profile["dispatch_key"],
                            "capability_profile_id": profile["capability_profile_id"],
                            "q_mode": q_mode,
                            "latency_ms": latency,
                            "energy_j": energy,
                            "ap30": min(1.0, ap70 + 0.2),
                            "ap50": min(1.0, ap70 + 0.1),
                            "ap70": ap70,
                            "terminal_status": "measured_success_gold",
                            "training_source": (
                                "initial_coldstart" if width_index < 2 else "online_feedback"
                            ),
                        }
                    )
    return rows, graphs


def _closure() -> dict:
    return {
        "schema_version": "stage4_p1_p3_closure_audit_v1",
        "stage4_closed": True,
        "stage5_search_ready": True,
        "canonical_value_heads": {
            "latency_ms": "extra_trees_log",
            "energy_j": "extra_trees_log",
            "ap70": "lgbm_huber_residual",
        },
        "ranker_policy": "rejected_use_value_heads_only",
        "uncertainty_policy": "lgbm_quantile_plus_group_conformal",
        "selected_acquisition_policy": "predicted_frontier_diversity",
        "training_source_rows": {"initial_coldstart": 16, "online_feedback": 8},
    }


def _source_registry() -> dict:
    groups = []
    for model, widths in {
        "pyramid": ([40, 80, 160], [48, 96, 192]),
        "codriving": ([40, 80, 160], [48, 96, 192]),
    }.items():
        for width in widths:
            group_id = f"{model}|{'x'.join(map(str, width))}"
            groups.append(
                {
                    "group_id": group_id,
                    "model": model,
                    "width": width,
                    "source_status": "ready",
                    "source_evidence_sha256": "a" * 64,
                    "source_contract": {"onnx_path": f"/remote/{group_id}.onnx"},
                    "graph_features": _graph(group_id, model, width),
                }
            )
    return {"schema_version": "stage5_candidate_source_registry_v1", "groups": groups}


class Stage5ProductionSearchV1Tests(unittest.TestCase):
    def test_contract_rejects_drift_from_stage4_frozen_policies(self) -> None:
        rows, graphs = _training_rows()
        closure = {**_closure(), "selected_acquisition_policy": "uncertainty_only"}

        with self.assertRaisesRegex(ValueError, "predicted_frontier_diversity"):
            stage5.validate_stage5_contract(closure, rows, graphs, _profiles())

    def test_contract_preserves_gold_and_feedback_source_roles(self) -> None:
        rows, graphs = _training_rows()

        audit = stage5.validate_stage5_contract(_closure(), rows, graphs, _profiles())

        self.assertEqual(audit["training_source_rows"], {"initial_coldstart": 16, "online_feedback": 8})
        self.assertEqual(audit["group_count"], 6)
        self.assertTrue(audit["four_arm_groups_complete"])

    def test_contract_can_freeze_stage5_to_initial_coldstart_only(self) -> None:
        rows, graphs = _training_rows()
        gold = [row for row in rows if row["training_source"] == "initial_coldstart"]
        gold_groups = {row["group_id"] for row in gold}
        gold_graphs = [graph for graph in graphs if graph["group_id"] in gold_groups]

        audit = stage5.validate_stage5_contract(
            _closure(),
            gold,
            gold_graphs,
            _profiles(),
            training_view_policy="initial_coldstart_only",
        )

        self.assertEqual(audit["training_source_rows"], {"initial_coldstart": 16})
        self.assertEqual(audit["training_view_policy"], "initial_coldstart_only")
        self.assertEqual(audit["excluded_stage4_feedback_rows"], 8)

    def test_contract_allows_append_only_online_feedback_but_not_coldstart_drift(self) -> None:
        rows, graphs = _training_rows()
        source_group = [row for row in rows if row["group_id"] == "codriving|32x64x128"]
        appended = []
        for row in source_group:
            clone = dict(row)
            clone["group_id"] = "codriving|40x80x160"
            clone["width"] = [40, 80, 160]
            clone["manifest_job_id"] = clone["manifest_job_id"].replace(
                "codriving|32x64x128", "codriving|40x80x160"
            )
            clone["training_source"] = "online_feedback"
            appended.append(clone)
        graphs.append(_graph("codriving|40x80x160", "codriving", [40, 80, 160]))

        audit = stage5.validate_stage5_contract(
            _closure(), [*rows, *appended], graphs, _profiles()
        )
        self.assertEqual(audit["training_source_rows"]["initial_coldstart"], 16)
        self.assertEqual(audit["training_source_rows"]["online_feedback"], 12)

        drifted = [dict(row) for row in rows]
        drifted[0]["training_source"] = "online_feedback"
        with self.assertRaisesRegex(ValueError, "initial_coldstart"):
            stage5.validate_stage5_contract(_closure(), drifted, graphs[:-1], _profiles())

    def test_contract_retains_feasibility_rows_but_excludes_their_incomplete_groups_from_value_fit(self) -> None:
        rows, graphs = _training_rows()
        failed_group = "pyramid|16x32x64"
        failed_row = next(
            row
            for row in rows
            if row["group_id"] == failed_group
            and row["dispatch_key"] == "tvm_auto"
            and row["q_mode"] == "fp16"
        )
        for field in ("latency_ms", "energy_j", "ap30", "ap50", "ap70"):
            failed_row[field] = None
        failed_row["terminal_status"] = "feasibility_failure"

        audit = stage5.validate_stage5_contract(_closure(), rows, graphs, _profiles())
        bundle = stage5.fit_production_bundle(rows, graphs, _profiles(), _closure(), seed=7)

        self.assertEqual(audit["training_row_count"], 24)
        self.assertEqual(audit["value_training_row_count"], 20)
        self.assertEqual(audit["excluded_value_groups"], [failed_group])
        self.assertEqual(bundle.manifest["value_training_row_count"], 20)

    def test_candidate_manifest_excludes_measured_and_frozen_holdout_groups(self) -> None:
        rows, _ = _training_rows()
        registry = _source_registry()
        holdout = {
            "groups": [
                {"group_id": "pyramid|48x96x192"},
                {"group_id": "codriving|48x96x192"},
            ]
        }

        manifest = stage5.build_candidate_manifest(
            registry,
            measured_group_ids={row["group_id"] for row in rows},
            frozen_holdout=holdout,
            capability_profiles=_profiles(),
        )

        self.assertEqual(manifest["eligible_group_count"], 2)
        self.assertEqual(len(manifest["rows"]), 8)
        self.assertEqual({row["q_mode"] for row in manifest["rows"]}, {"fp16", "int8"})
        self.assertEqual({row["dispatch_key"] for row in manifest["rows"]}, {"tvm_auto", "trt_engine"})
        self.assertTrue(all(len(row["genome"]) == 4 for row in manifest["rows"]))
        self.assertTrue(all(row["genome"][-1] in {"fp16", "int8"} for row in manifest["rows"]))
        self.assertTrue(all("backend" not in row["genome"] for row in manifest["rows"]))

    def test_candidate_manifest_accepts_materializable_sources_without_claiming_ready(self) -> None:
        registry = _source_registry()
        registry["groups"][0]["source_status"] = "materializable"

        manifest = stage5.build_candidate_manifest(
            registry,
            measured_group_ids=set(),
            frozen_holdout={"groups": []},
            capability_profiles=_profiles(),
        )

        rows = [
            row for row in manifest["rows"]
            if row["group_id"] == registry["groups"][0]["group_id"]
        ]
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(row["source_status"] == "materializable" for row in rows))

    def test_fit_predict_uses_frozen_heads_and_group_conformal_intervals(self) -> None:
        rows, graphs = _training_rows()
        bundle = stage5.fit_production_bundle(rows, graphs, _profiles(), _closure(), seed=7)
        manifest = stage5.build_candidate_manifest(
            _source_registry(),
            measured_group_ids={row["group_id"] for row in rows},
            frozen_holdout={"groups": []},
            capability_profiles=_profiles(),
        )

        predicted = stage5.predict_candidate_rows(bundle, manifest["rows"], _profiles())

        self.assertEqual(bundle.manifest["canonical_value_heads"], _closure()["canonical_value_heads"])
        self.assertEqual(bundle.manifest["uncertainty_policy"], "lgbm_quantile_plus_group_conformal")
        self.assertEqual(len(predicted), 16)
        for row in predicted:
            self.assertEqual(set(row["predictions"]), {"latency_ms", "energy_j", "ap70"})
            self.assertEqual(set(row["prediction_intervals"]), {"latency_ms", "energy_j", "ap70"})
            for interval in row["prediction_intervals"].values():
                self.assertLessEqual(interval["lower"], interval["median"])
                self.assertLessEqual(interval["median"], interval["upper"])

    def test_predicted_frontier_diversity_selects_one_complete_group_per_model(self) -> None:
        rows, graphs = _training_rows()
        bundle = stage5.fit_production_bundle(rows, graphs, _profiles(), _closure(), seed=11)
        manifest = stage5.build_candidate_manifest(
            _source_registry(),
            measured_group_ids={row["group_id"] for row in rows},
            frozen_holdout={"groups": []},
            capability_profiles=_profiles(),
        )
        predicted = stage5.predict_candidate_rows(bundle, manifest["rows"], _profiles())

        selected = stage5.select_predicted_frontier_diversity(
            predicted, rows, graphs, group_budget_by_model={"pyramid": 1, "codriving": 1}
        )

        self.assertEqual(selected["policy"], "predicted_frontier_diversity")
        self.assertEqual(len(selected["selected_group_ids"]), 2)
        self.assertEqual(len(selected["selected_rows"]), 8)
        self.assertEqual({row["model"] for row in selected["selected_rows"]}, {"pyramid", "codriving"})
        self.assertTrue(selected["candidate_labels_visible_before_measurement"] is False)
        self.assertTrue(all(len(group["rows"]) == 4 for group in selected["groups"]))

    def test_round_zero_checkpoint_is_idempotent_and_records_real_measurement_boundary(self) -> None:
        rows, graphs = _training_rows()
        with tempfile.TemporaryDirectory() as temporary:
            result = stage5.initialize_round_zero(
                closure=_closure(),
                training_rows=rows,
                graph_features=graphs,
                capability_profiles=_profiles(),
                source_registry=_source_registry(),
                frozen_holdout={"groups": []},
                output_dir=Path(temporary),
                seed=19,
            )
            repeated = stage5.initialize_round_zero(
                closure=_closure(),
                training_rows=rows,
                graph_features=graphs,
                capability_profiles=_profiles(),
                source_registry=_source_registry(),
                frozen_holdout={"groups": []},
                output_dir=Path(temporary),
                seed=19,
            )

            state = json.loads((Path(temporary) / "round_00/round_state.json").read_text())
            requests = json.loads(
                (Path(temporary) / "round_00/measurement_request.json").read_text()
            )

        self.assertEqual(result["round_state_sha256"], repeated["round_state_sha256"])
        self.assertEqual(state["round_index"], 0)
        self.assertEqual(state["status"], "awaiting_real_measurement")
        self.assertFalse(state["offline_replay"])
        self.assertEqual(state["selected_group_count"], 2)
        self.assertEqual(requests["row_count"], 8)
        self.assertEqual(requests["required_metrics"], ["latency_ms", "energy_j", "ap30", "ap50", "ap70"])

    def test_feedback_validation_requires_real_evidence_sha_and_builds_round1_audits(self) -> None:
        rows, graphs = _training_rows()
        feedback: list[dict] = []
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for model in ("pyramid", "codriving"):
                width = [40, 80, 160]
                group_id = f"{model}|40x80x160"
                for profile in _profiles():
                    for q_mode in ("fp16", "int8"):
                        performance = root / f"{model}-{profile['dispatch_key']}-{q_mode}-perf.json"
                        ap = root / f"{model}-{profile['dispatch_key']}-{q_mode}-ap.json"
                        performance.write_text(
                            json.dumps({"lat_p50_ms": 1.0, "energy_j": 0.2}) + "\n",
                            encoding="utf-8",
                        )
                        ap.write_text(
                            json.dumps(
                                {
                                    "status": "success",
                                    "processed_samples": 1789,
                                    "fallback_samples": 0,
                                    "failed_samples": 0,
                                    "ap30": 0.8,
                                    "ap50": 0.7,
                                    "ap70": 0.6,
                                }
                            )
                            + "\n",
                            encoding="utf-8",
                        )
                        source_plan_sha = hashlib.sha256(
                            f"{model}:{width}".encode()
                        ).hexdigest()
                        source_evidence = root / (
                            f"{model}-{profile['dispatch_key']}-{q_mode}-source.json"
                        )
                        source_evidence.write_text(
                            json.dumps({"source_plan_sha256": source_plan_sha}) + "\n",
                            encoding="utf-8",
                        )
                        feedback.append(
                            {
                                "manifest_job_id": f"{group_id}|q={q_mode}|profile={profile['capability_profile_id']}",
                                "group_id": group_id,
                                "model": model,
                                "width": width,
                                "dispatch_key": profile["dispatch_key"],
                                "capability_profile_id": profile["capability_profile_id"],
                                "q_mode": q_mode,
                                "genome": [*width, q_mode],
                                "strategy_id": f"q={q_mode}",
                                "capability_digest": profile["capability_digest"],
                                "source_status": "ready",
                                "source_contract": {"model": model, "width": width},
                                "source_evidence_sha256": source_plan_sha,
                                "materialized_source_evidence_path": str(source_evidence),
                                "materialized_source_evidence_sha256": hashlib.sha256(
                                    source_evidence.read_bytes()
                                ).hexdigest(),
                                "latency_ms": 1.0,
                                "energy_j": 0.2,
                                "ap30": 0.8,
                                "ap50": 0.7,
                                "ap70": 0.6,
                                "terminal_status": "measured_success_gold",
                                "performance_result_json": str(performance),
                                "performance_result_sha256": hashlib.sha256(performance.read_bytes()).hexdigest(),
                                "ap_report_path": str(ap),
                                "ap_report_sha256": hashlib.sha256(ap.read_bytes()).hexdigest(),
                            }
                        )

            measurement_request = {
                "schema_version": "stage5_measurement_request_v1",
                "rows": [
                    {
                        key: row[key]
                        for key in (
                            "manifest_job_id",
                            "group_id",
                            "model",
                            "width",
                            "q_mode",
                            "capability_profile_id",
                            "dispatch_key",
                            "genome",
                            "strategy_id",
                            "capability_digest",
                            "source_status",
                            "source_contract",
                            "source_evidence_sha256",
                        )
                    }
                    for row in feedback
                ],
            }
            request_by_id = {
                row["manifest_job_id"]: row for row in measurement_request["rows"]
            }
            feedback = [
                {
                    **row,
                    "measurement_request_row_sha256": stage5._sha(
                        request_by_id[row["manifest_job_id"]]
                    ),
                }
                for row in feedback
            ]
            audit = stage5.build_feedback_round_audit(
                initial_rows=rows,
                feedback_rows=feedback,
                selected_group_ids={"pyramid|40x80x160", "codriving|40x80x160"},
                measurement_request_rows=measurement_request["rows"],
                round_index=1,
            )
            numerical_feedback = [dict(row) for row in feedback]
            failed = numerical_feedback[0]
            failure_report = root / "numerical-failure.json"
            failure_report.write_text(
                json.dumps({"status": "blocked", "feasibility_blockers": ["invalid_tensor_quant_params"]})
                + "\n",
                encoding="utf-8",
            )
            failed.update(
                {
                    "terminal_status": "numerical_feasibility_failure",
                    "ap30": None,
                    "ap50": None,
                    "ap70": None,
                    "failure_reason": "invalid_tensor_quant_params",
                    "ap_report_path": str(failure_report),
                    "ap_report_sha256": hashlib.sha256(failure_report.read_bytes()).hexdigest(),
                    "failure_evidence_path": str(failure_report),
                    "failure_evidence_sha256": hashlib.sha256(failure_report.read_bytes()).hexdigest(),
                }
            )
            numerical_audit = stage5.build_feedback_round_audit(
                initial_rows=rows,
                feedback_rows=numerical_feedback,
                selected_group_ids={"pyramid|40x80x160", "codriving|40x80x160"},
                measurement_request_rows=measurement_request["rows"],
                round_index=1,
            )
            numerical_advanced = stage5.advance_search_round(
                closure=_closure(),
                initial_training_rows=rows,
                initial_graph_features=graphs,
                capability_profiles=_profiles(),
                source_registry=_source_registry(),
                frozen_holdout={"groups": []},
                feedback_rows=numerical_feedback,
                measurement_request=measurement_request,
                selected_group_ids={"pyramid|40x80x160", "codriving|40x80x160"},
                output_dir=root / "numerical-search",
                round_index=1,
                seed=23,
            )
            advanced = stage5.advance_search_round(
                closure=_closure(),
                initial_training_rows=rows,
                initial_graph_features=graphs,
                capability_profiles=_profiles(),
                source_registry=_source_registry(),
                frozen_holdout={"groups": []},
                feedback_rows=feedback,
                measurement_request=measurement_request,
                selected_group_ids={"pyramid|40x80x160", "codriving|40x80x160"},
                output_dir=root / "search",
                round_index=1,
                seed=23,
            )
            reordered = stage5.advance_search_round(
                closure=_closure(),
                initial_training_rows=rows,
                initial_graph_features=list(reversed(graphs)),
                capability_profiles=_profiles(),
                source_registry=_source_registry(),
                frozen_holdout={"groups": []},
                feedback_rows=feedback,
                measurement_request=measurement_request,
                selected_group_ids={"pyramid|40x80x160", "codriving|40x80x160"},
                output_dir=root / "reordered-search",
                round_index=1,
                seed=23,
            )
            round_state = json.loads(
                (root / "search/round_01/round_state.json").read_text(encoding="utf-8")
            )
            reordered_state = json.loads(
                (root / "reordered-search/round_01/round_state.json").read_text(encoding="utf-8")
            )
            drifted_request = json.loads(json.dumps(measurement_request))
            drifted_request["rows"][0]["q_mode"] = "int8"
            with self.assertRaisesRegex(ValueError, "measurement request|identity drift"):
                stage5.advance_search_round(
                    closure=_closure(),
                    initial_training_rows=rows,
                    initial_graph_features=graphs,
                    capability_profiles=_profiles(),
                    source_registry=_source_registry(),
                    frozen_holdout={"groups": []},
                    feedback_rows=feedback,
                    measurement_request=drifted_request,
                    selected_group_ids={"pyramid|40x80x160", "codriving|40x80x160"},
                    output_dir=root / "drifted",
                    round_index=1,
                    seed=23,
                )
            for field, drifted_value in (
                ("genome", [999, 80, 160, "fp16"]),
                ("strategy_id", "q=drifted"),
                ("capability_digest", "f" * 64),
                ("source_status", "materializable"),
                ("source_contract", {"model": "drifted"}),
                ("source_evidence_sha256", "e" * 64),
            ):
                drifted_feedback = json.loads(json.dumps(feedback))
                drifted_feedback[0][field] = drifted_value
                with self.assertRaisesRegex(ValueError, "identity drift"):
                    stage5.build_feedback_round_audit(
                        initial_rows=rows,
                        feedback_rows=drifted_feedback,
                        selected_group_ids={"pyramid|40x80x160", "codriving|40x80x160"},
                        measurement_request_rows=measurement_request["rows"],
                        round_index=1,
                    )

        self.assertEqual(audit["feedback_row_count"], 8)
        self.assertEqual(audit["feedback_group_count"], 2)
        self.assertTrue(audit["evidence_sha_verified"])
        self.assertEqual(audit["budget"]["new_complete_groups"], 2)
        self.assertIn("measured_pareto", audit)
        self.assertIn("measured_hv", audit)
        self.assertEqual(numerical_audit["budget"]["feasibility_terminal_rows"], 1)
        self.assertEqual(numerical_advanced["round_index"], 1)
        self.assertEqual(advanced["round_index"], 1)
        self.assertEqual(
            advanced["selected_group_ids"],
            ["codriving|48x96x192", "pyramid|48x96x192"],
        )
        self.assertTrue(round_state["previous_round_feedback_verified"])
        self.assertFalse(round_state["offline_replay"])
        self.assertEqual(
            round_state["previous_measurement_request_sha256"],
            stage5._sha(measurement_request),
        )
        self.assertEqual(
            round_state["model_bundle_config_sha256"],
            reordered_state["model_bundle_config_sha256"],
        )
        self.assertEqual(round_state["selection_sha256"], reordered_state["selection_sha256"])


if __name__ == "__main__":
    unittest.main()
