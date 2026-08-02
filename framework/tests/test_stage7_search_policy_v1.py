from __future__ import annotations

import copy
import hashlib
import unittest
from unittest import mock

from framework.stage2.canonical_search_v3 import build_capability_profile
from framework.stage5 import single_target_search_v2 as stage5
from framework.stage7 import search_policy_v1 as policy


def _profile() -> dict:
    return build_capability_profile(
        capability_profile_id="h800-tvm-auto-formal-v3",
        hardware_target="h800",
        compiler_fingerprint=hashlib.sha256(b"compiler").hexdigest(),
        dispatch_key="tvm_auto",
        features={
            "s1p_fp16_build_success_coverage": 1.0,
            "s1q_fp16_build_success_coverage": 1.0,
            "s1p_int8_build_success_coverage": 1.0,
            "s1q_int8_build_success_coverage": 1.0,
            "operator_coverage": 0.75,
        },
    )


def _registry() -> dict:
    groups = []
    for index, width in enumerate(
        ([16, 32, 64], [24, 48, 96], [32, 64, 128], [40, 80, 160])
    ):
        group_id = "pyramid|" + "x".join(map(str, width))
        groups.append(
            {
                "group_id": group_id,
                "model": "pyramid",
                "width": list(width),
                "source_status": "ready",
                "source_evidence_sha256": f"{index + 1:064x}",
                "source_contract": {
                    "checkpoint_path": f"/frozen/{group_id}.pth",
                    "onnx_path": f"/frozen/{group_id}.onnx",
                },
                "graph_features": {
                    "group_id": group_id,
                    "model": "pyramid",
                    "width": list(width),
                    "conv_count": 20 + index,
                },
            }
        )
    groups.append(
        {
            "group_id": "codriving|16x32x64",
            "model": "codriving",
            "width": [16, 32, 64],
            "source_status": "ready",
            "source_evidence_sha256": "f" * 64,
            "source_contract": {"onnx_path": "/frozen/codriving.onnx"},
            "graph_features": {
                "group_id": "codriving|16x32x64",
                "model": "codriving",
                "width": [16, 32, 64],
                "conv_count": 19,
            },
        }
    )
    return {"schema_version": "stage5_candidate_source_registry_v1", "groups": groups}


def _predictions(rows: list[dict]) -> list[dict]:
    return [
        {
            **row,
            "predictions": {
                "latency_ms": 1.0 + index,
                "energy_j": 0.1 + index,
                "ap70": 0.8 - 0.01 * index,
            },
            "prediction_intervals": {
                target: {"lower": 0.0, "median": 0.5, "upper": 1.0}
                for target in ("latency_ms", "energy_j", "ap70")
            },
        }
        for index, row in enumerate(rows)
    ]


def _training_context() -> tuple[list[dict], list[dict], list[dict], dict]:
    profiles = [_profile()]
    profiles.append(
        build_capability_profile(
            capability_profile_id="h800-trt-formal-v3",
            hardware_target="h800",
            compiler_fingerprint=hashlib.sha256(b"trt").hexdigest(),
            dispatch_key="trt_engine",
            features={
                "s1p_fp16_build_success_coverage": 1.0,
                "s1q_fp16_build_success_coverage": 1.0,
                "s1p_int8_build_success_coverage": 1.0,
                "s1q_int8_build_success_coverage": 1.0,
                "operator_coverage": 0.5,
            },
        )
    )
    rows = []
    graphs = []
    for group_index, width in enumerate(([16, 32, 64], [24, 48, 96])):
        group_id = "pyramid|" + "x".join(map(str, width))
        graphs.append(
            {
                "group_id": group_id,
                "model": "pyramid",
                "width": list(width),
                "conv_count": 20 + group_index,
            }
        )
        for profile in profiles:
            for q_index, q_mode in enumerate(("fp16", "int8")):
                row_id = (
                    f"{group_id}|q={q_mode}|profile="
                    f"{profile['capability_profile_id']}"
                )
                rows.append(
                    {
                        "manifest_job_id": row_id,
                        "row_id": row_id,
                        "group_id": group_id,
                        "model": "pyramid",
                        "width": list(width),
                        "q_mode": q_mode,
                        "capability_profile_id": profile[
                            "capability_profile_id"
                        ],
                        "dispatch_key": profile["dispatch_key"],
                        "terminal_status": "measured_success_gold",
                        "training_source": "initial_coldstart",
                        "latency_ms": 2.0 + group_index + q_index,
                        "energy_j": 0.2 + 0.1 * group_index + 0.05 * q_index,
                        "ap70": 0.7 + 0.01 * group_index - 0.02 * q_index,
                    }
                )
    closure = {
        "schema_version": "stage4_p1_p3_closure_audit_v1",
        "stage4_closed": True,
        "stage5_search_ready": True,
        "canonical_value_heads": {
            "latency_ms": "extra_trees_log",
            "energy_j": "extra_trees_log",
            "ap70": "lgbm_huber_residual",
        },
        "uncertainty_policy": "lgbm_quantile_plus_group_conformal",
        "selected_acquisition_policy": "predicted_frontier_diversity",
        "training_source_rows": {"initial_coldstart": len(rows)},
    }
    return rows, graphs, profiles, closure


class Stage7SearchPolicyV1Tests(unittest.TestCase):
    def test_a2_anchor_uses_the_same_complete_group_view_as_stage5_bundle(
        self,
    ) -> None:
        rows, _, _, _ = _training_context()
        rows[0]["terminal_status"] = "feasibility_failure"

        anchor = policy.derive_a2_anchor(rows)

        self.assertEqual(anchor, {"pyramid": 0.7})

    def test_a2_anchor_rejects_partial_four_arm_groups_like_stage5_bundle(
        self,
    ) -> None:
        rows, _, _, _ = _training_context()

        with self.assertRaisesRegex(ValueError, "incomplete four-arm group"):
            policy.derive_a2_anchor(rows[:-1])

    def test_a2_frozen_contract_recomputes_training_graph_anchor_and_artifact_views(
        self,
    ) -> None:
        rows, graphs, _, _ = _training_context()
        candidates = policy.prepare_frozen_candidate_pools(
            _registry(),
            raw_profile=_profile(),
            measured_row_ids={row["row_id"] for row in rows},
        )["variant_pools"]["without_measured_feedback"]
        predictions = _predictions(candidates)
        anchor = policy.derive_a2_anchor(rows)
        payload = policy._a2_frozen_payload(
            {"bundle_config_sha256": "a" * 64},
            predictions,
            rows,
            graphs,
            anchor,
        )

        validated = policy.validate_a2_frozen_contract(
            payload,
            initial_rows=rows,
            initial_graph_features=graphs,
            expected_anchor=anchor,
            artifact_predictions=predictions,
        )

        self.assertEqual(validated, payload)
        mutations = (
            (
                "training",
                "A2 training view deterministic drift",
                lambda value: value.update(
                    {
                        "training_view_sha256": policy.canonical_sha256(
                            [{"rewritten": True}]
                        )
                    }
                ),
            ),
            (
                "graph",
                "A2 graph view deterministic drift",
                lambda value: value.update(
                    {
                        "graph_feature_view_sha256": policy.canonical_sha256(
                            [{"rewritten": True}]
                        )
                    }
                ),
            ),
            (
                "anchor",
                "A2 anchor deterministic drift",
                lambda value: (
                    value["anchor"].update(
                        {"pyramid": value["anchor"]["pyramid"] + 1.0}
                    ),
                    value.update(
                        {
                            "anchor_sha256": policy.canonical_sha256(
                                value["anchor"]
                            )
                        }
                    ),
                ),
            ),
            (
                "prediction",
                "A2 prediction artifact drift",
                lambda value: (
                    value["candidate_predictions"][0]["predictions"].update(
                        {"latency_ms": 999.0}
                    ),
                    value.update(
                        {
                            "prediction_view_sha256": policy.canonical_sha256(
                                value["candidate_predictions"]
                            )
                        }
                    ),
                ),
            ),
        )
        for label, error, mutate in mutations:
            with self.subTest(label=label):
                tampered = copy.deepcopy(payload)
                mutate(tampered)
                tampered.pop("frozen_payload_sha256")
                tampered["frozen_payload_sha256"] = (
                    policy.canonical_sha256(tampered)
                )
                with self.assertRaisesRegex(ValueError, error):
                    policy.validate_a2_frozen_contract(
                        tampered,
                        initial_rows=rows,
                        initial_graph_features=graphs,
                        expected_anchor=anchor,
                        artifact_predictions=predictions,
                    )

    def test_task_and_pool_freeze_scan_all_pyramid_genomes_before_gold_exclusion(self) -> None:
        task = policy.build_stage7_task(_profile())
        first_row = (
            "pyramid|16x32x64|q=fp16|profile=h800-tvm-auto-formal-v3"
        )

        frozen = policy.prepare_frozen_candidate_pools(
            _registry(),
            raw_profile=_profile(),
            measured_row_ids={first_row},
        )

        self.assertEqual(task.task_id, "S7-PYR-TVM")
        self.assertEqual(task.target_model, "pyramid")
        self.assertEqual(task.hardware_id, "h800")
        self.assertEqual(frozen["pre_scan_count"], 8)
        self.assertEqual(frozen["scanner_audit"]["pre_scan_count"], 8)
        self.assertEqual(frozen["selectable_pre_scan_count"], 7)
        self.assertEqual(
            frozen["pre_scan_count"] - frozen["selectable_pre_scan_count"], 1
        )
        self.assertIn(
            first_row,
            {row["row_id"] for row in frozen["pre_scan_candidates"]},
        )
        self.assertNotIn(
            first_row,
            {row["row_id"] for row in frozen["variant_pools"]["full"]},
        )
        self.assertEqual(
            frozen["scanner_profile"]["static_shape_contract"],
            {
                "width_alignment": [8, 16, 32],
                "width_min": [16, 32, 64],
                "width_max": [64, 128, 256],
            },
        )

    def test_a4_uses_pre_scan_while_full_uses_only_scan_pass(self) -> None:
        registry = _registry()
        registry["groups"][-2]["width"] = [42, 80, 160]
        registry["groups"][-2]["group_id"] = "pyramid|42x80x160"
        registry["groups"][-2]["graph_features"]["group_id"] = "pyramid|42x80x160"
        registry["groups"][-2]["graph_features"]["width"] = [42, 80, 160]

        frozen = policy.prepare_frozen_candidate_pools(
            registry,
            raw_profile=_profile(),
            measured_row_ids=set(),
        )

        full_ids = {row["row_id"] for row in frozen["variant_pools"]["full"]}
        a4_ids = {
            row["row_id"]
            for row in frozen["variant_pools"]["without_capability_scan"]
        }
        self.assertEqual(len(a4_ids - full_ids), 2)
        self.assertTrue(all("42x80x160" in row_id for row_id in a4_ids - full_ids))

    def test_request_binding_detects_variant_seed_path_and_request_drift(self) -> None:
        trajectory = policy.build_trajectory_contract(
            variant="full",
            seed=20260718,
            result_root="/tmp/stage7",
            frozen_input_sha256="a" * 64,
            candidate_pool_sha256="b" * 64,
            pre_scan_sha256="d" * 64,
            scan_pass_sha256="e" * 64,
            scanner_rule_sha256="f" * 64,
            scanner_decision_sha256="1" * 64,
        )
        request = {
            "measurement_request_sha256": "c" * 64,
            "round_index": 0,
            "rows": [{"row_id": f"row-{index}"} for index in range(4)],
        }

        binding = policy.build_request_binding(
            trajectory,
            request,
            round_index=0,
            trajectory_dir="/tmp/stage7/variants/full/seed_20260718",
        )

        self.assertEqual(binding["selected_row_ids"], [f"row-{index}" for index in range(4)])
        self.assertEqual(
            binding["prior_chain_head_sha256"],
            trajectory["trajectory_contract_sha256"],
        )
        self.assertIsNone(binding["previous_released_feedback_sha256"])
        self.assertIsNone(binding["previous_request_binding_sha256"])
        self.assertRegex(
            policy.build_round_chain_head(trajectory, binding),
            r"^[0-9a-f]{64}$",
        )
        self.assertEqual(trajectory["pre_scan_sha256"], "d" * 64)
        self.assertEqual(trajectory["scan_pass_sha256"], "e" * 64)
        self.assertEqual(trajectory["scanner_rule_sha256"], "f" * 64)
        self.assertEqual(trajectory["scanner_decision_sha256"], "1" * 64)
        with self.assertRaisesRegex(ValueError, "trajectory directory"):
            policy.build_request_binding(
                trajectory,
                request,
                round_index=0,
                trajectory_dir="/tmp/stage7/variants/full/seed_20260719",
            )
        with self.assertRaisesRegex(ValueError, "measurement request SHA"):
            policy.validate_request_binding(
                binding,
                {**request, "measurement_request_sha256": "d" * 64},
                trajectory,
            )
        drifted_binding = {
            **binding,
            "variant": "without_surrogate",
        }
        drifted_binding.pop("request_binding_sha256")
        drifted_binding["request_binding_sha256"] = policy.canonical_sha256(
            drifted_binding
        )
        with self.assertRaisesRegex(ValueError, "request binding identity"):
            policy.validate_request_binding(
                drifted_binding,
                request,
                trajectory,
            )

    def test_full_calls_stage5_acquisition_and_keeps_exact_policy_name(self) -> None:
        frozen = policy.prepare_frozen_candidate_pools(
            _registry(), raw_profile=_profile(), measured_row_ids=set()
        )
        task = policy.build_stage7_task(_profile())
        dummy_bundle = mock.Mock(
            manifest={"bundle_config_sha256": "a" * 64, "feature_names": []}
        )
        with (
            mock.patch.object(policy, "fit_production_bundle", return_value=dummy_bundle),
            mock.patch.object(
                policy,
                "predict_candidate_rows",
                side_effect=lambda _bundle, rows, _profiles: _predictions(list(rows)),
            ),
            mock.patch.object(
                policy,
                "select_task_batch",
                wraps=stage5.select_task_batch,
            ) as select,
        ):
            result = policy.select_stage7_round(
                variant="full",
                seed=20260718,
                round_index=0,
                task=task,
                candidate_pool=frozen["variant_pools"]["full"],
                initial_rows=[],
                initial_graph_features=[],
                capability_profiles=[_profile()],
                closure={},
            )

        select.assert_called_once()
        self.assertEqual(result["acquisition"]["policy"], "predicted_frontier_diversity")
        self.assertEqual(result["measurement_request"]["batch_size"], 4)
        self.assertNotIn("cache", result)

    def test_a1_never_calls_fit_predict_or_stage5_acquisition(self) -> None:
        frozen = policy.prepare_frozen_candidate_pools(
            _registry(), raw_profile=_profile(), measured_row_ids=set()
        )
        with (
            mock.patch.object(
                policy, "fit_production_bundle", side_effect=AssertionError("fit called")
            ),
            mock.patch.object(
                policy, "fit_online_bundle", side_effect=AssertionError("fit called")
            ),
            mock.patch.object(
                policy, "predict_candidate_rows", side_effect=AssertionError("predict called")
            ),
            mock.patch.object(
                policy, "select_task_batch", side_effect=AssertionError("acquisition called")
            ),
        ):
            result = policy.select_stage7_round(
                variant="without_surrogate",
                seed=20260718,
                round_index=0,
                task=policy.build_stage7_task(_profile()),
                candidate_pool=frozen["variant_pools"]["without_surrogate"],
                initial_rows=[],
                initial_graph_features=[],
                capability_profiles=[_profile()],
                closure={},
            )

        self.assertEqual(result["acquisition"]["policy"], "uniform_random_without_replacement")
        self.assertNotIn("predicted_candidates", result)
        self.assertNotIn("model_bundle_manifest", result)

    def test_a2_later_round_reuses_frozen_bundle_predictions_and_diversity_view(self) -> None:
        frozen = policy.prepare_frozen_candidate_pools(
            _registry(), raw_profile=_profile(), measured_row_ids=set()
        )
        dummy_bundle = mock.Mock(
            manifest={"bundle_config_sha256": "a" * 64, "feature_names": ["width:axis0"]},
            model_anchors={},
        )
        task = policy.build_stage7_task(_profile())
        with (
            mock.patch.object(policy, "fit_production_bundle", return_value=dummy_bundle),
            mock.patch.object(
                policy,
                "predict_candidate_rows",
                side_effect=lambda _bundle, rows, _profiles: _predictions(list(rows)),
            ),
        ):
            round_zero = policy.select_stage7_round(
                variant="without_measured_feedback",
                seed=20260718,
                round_index=0,
                task=task,
                candidate_pool=frozen["variant_pools"]["without_measured_feedback"],
                initial_rows=[],
                initial_graph_features=[],
                capability_profiles=[_profile()],
                closure={},
            )
        selected = set(round_zero["acquisition"]["selected_row_ids"])
        with (
            mock.patch.object(
                policy, "fit_production_bundle", side_effect=AssertionError("refit called")
            ),
            mock.patch.object(
                policy, "fit_online_bundle", side_effect=AssertionError("online fit called")
            ),
            mock.patch.object(
                policy, "predict_candidate_rows", side_effect=AssertionError("predict called")
            ),
        ):
            round_one = policy.select_stage7_round(
                variant="without_measured_feedback",
                seed=20260718,
                round_index=1,
                task=task,
                candidate_pool=frozen["variant_pools"]["without_measured_feedback"],
                initial_rows=[],
                initial_graph_features=[],
                capability_profiles=[_profile()],
                closure={},
                selected_ids=selected,
                feedback_rows=[{"row_id": row_id, "latency_ms": 999} for row_id in selected],
                a2_frozen=round_zero["a2_frozen"],
            )

        self.assertEqual(
            round_one["model_bundle_manifest"]["bundle_config_sha256"],
            round_zero["model_bundle_manifest"]["bundle_config_sha256"],
        )
        self.assertEqual(
            round_one["a2_frozen"]["training_view_sha256"],
            round_zero["a2_frozen"]["training_view_sha256"],
        )
        self.assertTrue(
            selected.isdisjoint(round_one["acquisition"]["selected_row_ids"])
        )

    def test_backend_blind_fit_and_predict_remove_capability_columns_on_both_sides(self) -> None:
        rows, graphs, profiles, closure = _training_context()
        bundle = policy.fit_backend_blind_bundle(
            rows,
            graphs,
            profiles,
            closure=closure,
            seed=20260718,
            training_view_policy="initial_coldstart_only",
        )
        candidates = policy.prepare_frozen_candidate_pools(
            _registry(), raw_profile=_profile(), measured_row_ids=set()
        )["variant_pools"]["backend_blind"]

        predicted = policy.predict_backend_blind_rows(
            bundle, candidates, profiles
        )

        self.assertTrue(any(name.startswith("cap:") for name in bundle.full_feature_names))
        self.assertFalse(
            any(
                name.startswith(("cap:", "cap_x_q:"))
                for name in bundle.feature_names
            )
        )
        self.assertEqual(len(predicted), len(candidates))
        self.assertEqual(
            {tuple(row["predictions"]) for row in predicted},
            {("latency_ms", "energy_j", "ap70")},
        )
        self.assertEqual(bundle.manifest["leakage_verdict"], "no_forbidden_features")
        self.assertNotEqual(
            bundle.manifest["full_training_matrix_sha256"],
            bundle.manifest["blind_training_matrix_sha256"],
        )

    def test_backend_blind_private_stage5_api_drift_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "model API drift"):
            policy.audit_stage5_model_api(expected_source_sha256="0" * 64)

    def test_backend_blind_round_emits_full_comparison_and_selects_from_blind_rows(self) -> None:
        rows, graphs, profiles, closure = _training_context()
        candidates = policy.prepare_frozen_candidate_pools(
            _registry(), raw_profile=_profile(), measured_row_ids=set()
        )["variant_pools"]["backend_blind"]

        result = policy.select_stage7_round(
            variant="backend_blind",
            seed=20260718,
            round_index=0,
            task=policy.build_stage7_task(_profile()),
            candidate_pool=candidates,
            initial_rows=rows,
            initial_graph_features=graphs,
            capability_profiles=profiles,
            closure=closure,
        )

        audit = result["backend_blind_audit"]
        self.assertEqual(audit["leakage_verdict"], "no_forbidden_features")
        self.assertEqual(len(audit["prediction_deltas"]), len(candidates))
        self.assertEqual(len(audit["full_candidate_matrix_sha256"]), 64)
        self.assertEqual(len(audit["blind_candidate_matrix_sha256"]), 64)
        self.assertEqual(
            audit["selected_id_overlap"]["blind_selected_row_ids"],
            result["acquisition"]["selected_row_ids"],
        )
        self.assertLessEqual(audit["selected_id_overlap"]["overlap_count"], 4)


if __name__ == "__main__":
    unittest.main()
