from __future__ import annotations

import inspect
import hashlib
import unittest

from framework.stage2.canonical_search_v3 import build_capability_profile
from framework.stage7 import online_component_ablation_v1 as stage7


class Stage7OnlineComponentAblationV1Tests(unittest.TestCase):
    @staticmethod
    def _profile(*, support: dict[str, bool] | None = None, shape_contract: bool = True) -> dict:
        raw = build_capability_profile(
            capability_profile_id="h800-tvm-auto-v1", hardware_target="h800",
            compiler_fingerprint=hashlib.sha256(b"tvm_auto").hexdigest(), dispatch_key="tvm_auto",
            features={
                "s1p_fp16_build_success_coverage": 1.0,
                "s1q_fp16_build_success_coverage": 1.0,
                "s1p_int8_build_success_coverage": 1.0 if (support or {}).get("int8", True) else 0.0,
                "s1q_int8_build_success_coverage": 1.0 if (support or {}).get("int8", True) else 0.0,
            },
        )
        rules = None if not shape_contract else {
            "width_alignment": [8, 8, 8], "width_min": [8, 8, 8], "width_max": [128, 128, 128],
        }
        return stage7.derive_scanner_profile(raw, rules)

    @staticmethod
    def _candidate(row_id: str = "candidate-1", **overrides: object) -> dict:
        candidate = {
            "row_id": row_id, "task_id": "S7-PYR-TVM", "model": "pyramid", "hardware_id": "h800",
            "dispatch_key": "tvm_auto", "capability_profile_id": "h800-tvm-auto-v1", "width": [16, 32, 64],
            "width_schema": ["w0", "w1", "w2"], "q_mode": "fp16", "source_status": "ready",
            "source_evidence_sha256": "a" * 64,
            "source_contract": {"checkpoint_path": "/frozen/checkpoint.pth", "onnx_path": "/frozen/model.onnx"},
            "graph_features": {"flops": 12.0, "operator_count": 4, "shape": 1},
        }
        return {**candidate, **overrides}

    def test_frozen_variants_have_single_variable_isolation(self) -> None:
        full, variants = stage7.frozen_experiment_contracts()

        self.assertEqual(full["task_id"], "S7-PYR-TVM")
        self.assertEqual(full["policy_name"], "predicted_frontier_diversity")
        self.assertEqual(set(variants), {
            "without_surrogate",
            "without_measured_feedback",
            "backend_blind",
            "without_capability_scan",
        })
        for variant in variants.values():
            self.assertEqual(stage7.audit_single_variable_isolation(full, variant)["verdict"], "pass")

    def test_a1_selects_four_unique_ids_deterministically_without_surrogate_inputs(self) -> None:
        candidates = [{"row_id": f"candidate-{index}"} for index in range(9)]

        first = stage7.a1_select_unselected(candidates, {"candidate-0"}, seed=20260718)
        second = stage7.a1_select_unselected(candidates, {"candidate-0"}, seed=20260718)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)
        self.assertEqual(len(set(first)), 4)
        self.assertNotIn("candidate-0", first)
        self.assertNotIn("surrogate", stage7.a1_select_unselected.__annotations__)

    def test_a1_selection_api_cannot_receive_a_surrogate_callback_or_labels(self) -> None:
        parameters = inspect.signature(stage7.a1_select_unselected).parameters

        self.assertFalse({"surrogate", "acquisition", "uncertainty", "cache", "labels"} & set(parameters))

    def test_a2_feedback_projection_records_deduplicated_identities_without_training_feedback(self) -> None:
        initial_rows = [{"row_id": "gold-1", "latency_ms": 1.0}]
        graph_view = {"gold-1": {"flops": 12.0}}
        result = stage7.project_a2_feedback(
            initial_rows,
            graph_view,
            anchor={"round": 0},
            bundle_sha256="a" * 64,
            selected_results=[
                {"row_id": "new-1", "latency_ms": 2.0},
                {"row_id": "new-1", "latency_ms": 3.0},
            ],
        )

        self.assertEqual(result["training_view"], initial_rows)
        self.assertEqual(result["graph_feature_view"], graph_view)
        self.assertEqual(result["anchor"], {"round": 0})
        self.assertEqual(result["bundle_sha256"], "a" * 64)
        self.assertEqual([row["row_id"] for row in result["recorded_results"]], ["new-1"])
        self.assertNotIn("new-1", result["graph_feature_view"])

    def test_a2_freezes_bundle_sha_and_hides_feedback_labels_from_training_view(self) -> None:
        training = [{"row_id": "gold", "latency_ms": 1.0}]
        feedback = [{"row_id": "new", "latency_ms": 99.0, "ap70": 0.9}]

        result = stage7.project_a2_feedback(
            training, {}, anchor={"anchor": "fixed"}, bundle_sha256="d" * 64, selected_results=feedback
        )

        self.assertEqual(result["bundle_sha256"], "d" * 64)
        self.assertEqual(result["training_view"], training)
        self.assertNotIn("new", {row["row_id"] for row in result["training_view"]})

    def test_a3_removes_capability_backend_and_profile_features_but_preserves_graph_features(self) -> None:
        rows = [{
            "row_id": "candidate-1",
            "width": [16, 32, 64],
            "q_mode": "int8",
            "graph_features": {"flops": 12.0, "operator_count": 4},
            "model_features": {"capability:sm_count": 132, "backend:relay": 1, "hidden": 2},
            "features": {"hidden": 3},
            "feature_provenance": {
                "capability:sm_count": "capability-derived", "backend:relay": "backend-derived",
                "hidden": "profile-derived",
            },
        }]

        audit = stage7.project_backend_blind_features(rows)

        self.assertIn("graph:flops", audit["blind_schema"])
        self.assertIn("width:w0", audit["blind_schema"])
        self.assertIn("q_mode", audit["blind_schema"])
        self.assertNotIn("model:capability:sm_count", audit["blind_schema"])
        self.assertIn("model:capability:sm_count", audit["removed_names"])
        self.assertIn("model:hidden", audit["removed_names"])
        self.assertIn("feature:hidden", audit["removed_names"])
        self.assertEqual(audit["leakage_verdict"], "no_forbidden_features")
        self.assertNotEqual(audit["full_matrix_sha256"], audit["blind_matrix_sha256"])

    def test_a3_fails_closed_when_direct_profile_identity_remains(self) -> None:
        with self.assertRaisesRegex(ValueError, "direct identity"):
            stage7.project_backend_blind_features([{
                "row_id": "candidate-1", "width": [16, 32, 64], "q_mode": "fp16",
                "capability_profile_id": "h800-profile", "graph_features": {"flops": 1.0},
            }])

    def test_a3_fails_closed_on_neutral_feature_with_unknown_provenance(self) -> None:
        with self.assertRaisesRegex(ValueError, "provenance"):
            stage7.project_backend_blind_features([{
                "row_id": "candidate-1", "width": [16, 32, 64], "q_mode": "fp16",
                "graph_features": {"flops": 1.0}, "model_features": {"neutral_proxy": 3.0},
            }])

    def test_static_scanner_is_deterministic_and_never_uses_performance_labels(self) -> None:
        profile = self._profile()
        candidate = self._candidate(latency_ms=0.001)

        first = stage7.scan_candidate_pool([candidate], capability_profile=profile)
        second = stage7.scan_candidate_pool([candidate], capability_profile=profile)

        self.assertEqual(first, second)
        self.assertEqual(first["scan_pass_count"], 0)
        self.assertEqual(first["decisions"][0]["decision"], "reject")
        self.assertIn("forbidden_input:latency_ms", first["decisions"][0]["reasons"])
        self.assertEqual(first["scan_pass_candidates"], [])
        self.assertIn("latency", first["rule_manifest"]["forbidden_inputs"])
        self.assertEqual(first["scanner_note"], "capability_scan_discriminative_on_frozen_pool")

    def test_a4_uses_pre_scan_pool_when_scanner_rejects_a_candidate(self) -> None:
        profile = self._profile(support={"fp16": True, "int8": False})
        good = self._candidate("good")
        rejected = {**good, "row_id": "reject", "q_mode": "int8"}

        scan = stage7.scan_candidate_pool([good, rejected], capability_profile=profile)
        pools = stage7.variant_candidate_pools([good, rejected], scan)

        self.assertEqual([row["row_id"] for row in pools["full"]], ["good"])
        self.assertEqual([row["row_id"] for row in pools["without_capability_scan"]], ["good", "reject"])

    def test_scanner_requires_static_shape_contract_and_rejects_alignment_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "static shape"):
            self._profile(shape_contract=False)
        misaligned = stage7.scan_candidate_pool([self._candidate(width=[17, 32, 64])], capability_profile=self._profile())

        self.assertIn("width_alignment_mismatch", misaligned["decisions"][0]["reasons"])

    def test_scanner_allows_shape_and_profile_identity_without_ap_substring_false_positive(self) -> None:
        scan = stage7.scan_candidate_pool([self._candidate()], capability_profile=self._profile())

        self.assertEqual(scan["scan_pass_count"], 1)
        selected = scan["scan_pass_candidates"][0]
        self.assertEqual(selected["capability_profile_id"], "h800-tvm-auto-v1")
        self.assertIn("shape", selected["graph_features"])

    def test_scanner_rejects_nested_performance_labels_and_never_projects_them_to_selection(self) -> None:
        candidate = self._candidate(source_contract={
            "checkpoint_path": "/frozen/checkpoint.pth", "onnx_path": "/frozen/model.onnx",
            "nested": {"frontier_membership": True},
        })

        scan = stage7.scan_candidate_pool([candidate], capability_profile=self._profile())

        self.assertEqual(scan["scan_pass_count"], 0)
        self.assertIn(
            "forbidden_input:source_contract.nested.frontier_membership", scan["decisions"][0]["reasons"]
        )
        self.assertEqual(scan["scan_pass_candidates"], [])

    def test_scanner_accepts_real_stage5_materializable_paths_without_post_selection_onnx_sha(self) -> None:
        scan = stage7.scan_candidate_pool(
            [self._candidate(source_status="materializable")], capability_profile=self._profile()
        )

        self.assertEqual(scan["scan_pass_count"], 1)
        self.assertEqual(scan["decisions"][0]["decision"], "pass")

    def test_derive_scanner_profile_keeps_formal_profile_immutable_and_binds_rules_to_manifest(self) -> None:
        raw = build_capability_profile(
            capability_profile_id="h800-formal", hardware_target="h800",
            compiler_fingerprint=hashlib.sha256(b"formal").hexdigest(), dispatch_key="tvm_auto",
            features={
                "s1p_fp16_build_success_coverage": 1.0, "s1q_fp16_build_success_coverage": 1.0,
                "s1p_int8_build_success_coverage": 0.0, "s1q_int8_build_success_coverage": 0.5,
            },
        )
        original = dict(raw)
        rules = {"width_alignment": [8, 8, 8], "width_min": [8, 8, 8], "width_max": [128, 128, 128]}
        derived = stage7.derive_scanner_profile(raw, rules)
        changed_rules = stage7.derive_scanner_profile(raw, {**rules, "width_max": [256, 128, 128]})

        self.assertEqual(raw, original)
        self.assertFalse(derived["q_mode_support"]["int8"])
        self.assertEqual(derived["raw_capability_digest"], raw["capability_digest"])
        self.assertNotEqual(derived["rule_manifest"]["rule_sha256"], changed_rules["rule_manifest"]["rule_sha256"])
        self.assertIn("candidate_allowed_inputs", derived["rule_manifest"])
        self.assertIn("profile_allowed_inputs", derived["rule_manifest"])

    def test_a3_adapter_marks_raw_profile_features_as_capability_derived_and_removes_them(self) -> None:
        raw = build_capability_profile(
            capability_profile_id="h800-formal", hardware_target="h800",
            compiler_fingerprint=hashlib.sha256(b"formal").hexdigest(), dispatch_key="tvm_auto",
            features={"s1p_fp16_build_success_coverage": 1.0},
        )
        audit = stage7.project_stage5_backend_blind_features(
            [{"row_id": "x", "width": [16, 32, 64], "q_mode": "fp16", "graph_features": {"flops": 7.0}}], raw
        )

        self.assertIn("graph:flops", audit["blind_schema"])
        self.assertTrue(all("s1p_fp16_build_success_coverage" not in name for name in audit["blind_schema"]))

    def test_metric_field_detects_intermediate_ap_segment_without_shape_false_positives(self) -> None:
        self.assertTrue(stage7._metric_field("full_ap_eval_report"))
        self.assertFalse(stage7._metric_field("shape"))
        self.assertFalse(stage7._metric_field("capability_profile_id"))

    def test_cache_key_requires_exact_dimensions_and_hides_labels_until_selection(self) -> None:
        dimensions = {
            "model": "pyramid", "capability_profile_id": "h800-tvm-auto-v1", "hardware_id": "h800",
            "measurement_scope": "e2e", "input_protocol_sha256": "a" * 64, "batch_size": 1,
            "genome": [16, 32, 64], "q_mode": "fp16", "source_checkpoint_sha256": "b" * 64,
            "onnx_sha256": "c" * 64, "build_protocol_sha256": "d" * 64, "tuning_protocol_sha256": "e" * 64,
            "measurement_protocol_sha256": "f" * 64, "ap_protocol_sha256": "0" * 64,
        }
        key = stage7.build_measurement_cache_key(dimensions)
        near_key = stage7.build_measurement_cache_key({**dimensions, "batch_size": 2})
        cache = {key: {"latency_ms": 1.0, "energy_j": 2.0, "ap70": 0.3}}

        view = stage7.selection_candidate_view([{"row_id": "candidate-1", **dimensions, "cache_key": key, "cache_hit": True}])

        self.assertNotEqual(key, near_key)
        self.assertNotIn("cache_key", view[0])
        self.assertNotIn("cache_hit", view[0])
        with self.assertRaisesRegex(ValueError, "selected"):
            stage7.reveal_measurement_cache("candidate-1", set(), dimensions, cache)
        self.assertEqual(
            stage7.reveal_measurement_cache("candidate-1", {"candidate-1"}, dimensions, cache)["latency_ms"], 1.0
        )

    def test_cache_selection_invariance_detects_cache_dependent_drift(self) -> None:
        candidates = [{"row_id": "a"}, {"row_id": "b"}]
        audit = stage7.audit_cache_selection_invariance(
            candidates, cache={"irrelevant": {"latency_ms": 1.0}}, selector=lambda rows: [row["row_id"] for row in rows]
        )
        self.assertEqual(audit["verdict"], "pass")

    def test_cache_invariance_constructs_candidate_level_exact_hit_view_while_production_view_hides_it(self) -> None:
        dimensions = {
            "model": "pyramid", "capability_profile_id": "h800-tvm-auto-v1", "hardware_id": "h800",
            "measurement_scope": "e2e", "input_protocol_sha256": "a" * 64, "batch_size": 1,
            "genome": [16, 32, 64], "q_mode": "fp16", "source_checkpoint_sha256": "b" * 64,
            "onnx_sha256": "c" * 64, "build_protocol_sha256": "d" * 64, "tuning_protocol_sha256": "e" * 64,
            "measurement_protocol_sha256": "f" * 64, "ap_protocol_sha256": "0" * 64,
        }
        key = stage7.build_measurement_cache_key(dimensions)
        candidates = [{"row_id": "hit", "measurement_cache_dimensions": dimensions}, {"row_id": "miss", "measurement_cache_dimensions": {**dimensions, "batch_size": 2}}]

        with self.assertRaisesRegex(ValueError, "drift"):
            stage7.audit_cache_selection_invariance(
                candidates, cache={key: {"latency_ms": 1.0}},
                selector=lambda rows: ["hit"] if rows[0].get("cache_membership") else ["miss"],
            )
        self.assertNotIn("measurement_cache_dimensions", stage7.selection_candidate_view(candidates)[0])

    def test_failure_budget_and_result_row_validator_preserve_measured_evidence_only(self) -> None:
        self.assertTrue(stage7.classify_failure("candidate_runtime_capability_failure")["consumes_selected_event_budget"])
        self.assertFalse(stage7.classify_failure("gpu_unavailable")["consumes_selected_event_budget"])
        self.assertTrue(stage7.classify_failure("gpu_unavailable")["same_request_retry"])
        success = {
            "terminal_status": "measured_success_gold", "latency_ms": 1.0, "energy_j": 2.0, "ap70": 0.3,
            "latency_artifact_sha256": "a" * 64, "energy_artifact_sha256": "b" * 64,
            "ap_artifact_sha256": "c" * 64,
        }
        self.assertEqual(stage7.validate_result_row(success)["verdict"], "pass")
        with self.assertRaisesRegex(ValueError, "surrogate"):
            stage7.validate_result_row({**success, "latency_source": "surrogate_prediction"})

    def test_result_rows_require_metric_artifacts_for_success(self) -> None:
        with self.assertRaisesRegex(ValueError, "latency_artifact"):
            stage7.validate_result_row({
                "terminal_status": "measured_success_gold", "latency_ms": 1.0, "energy_j": 2.0, "ap70": 0.3,
            })

    def test_failure_rows_reject_metrics_and_artifacts(self) -> None:
        with self.assertRaisesRegex(ValueError, "failure"):
            stage7.validate_result_row({
                "terminal_status": "build_failure", "latency_ms": 1.0, "latency_artifact_sha256": "a" * 64,
            })
