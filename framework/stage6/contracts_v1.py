"""Immutable, machine-checkable contracts for the Stage6 six-arm ablation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


ARM_IDS = (
    "original_default",
    "compression_only",
    "schedule_only",
    "compress_then_tune",
    "tune_then_compress",
    "joint_shcosearch",
)
BACKEND_LABELS = {
    "latency_ms",
    "energy_j",
    "latency_prediction",
    "energy_prediction",
    "capability_profile",
    "dispatch_key",
    "backend",
}


def build_stage6_manifest() -> dict[str, Any]:
    """Return a fresh Stage6 contract; callers may serialize but not mutate shared state."""
    common = {
        "target_model": "pyramid",
        "hardware_id": "h800",
        "scope": "pyramid_multiscale_backbone",
        "input_shape": [2, 64, 128, 256],
        "base_width": [64, 128, 256],
        "genome": ["w0", "w1", "w2", "q_mode"],
        "q_modes": ["fp16", "int8"],
        "theoretical_width_grid_size": 343,
        "theoretical_candidate_grid_size": 686,
        "registered_width_count": 60,
        "registered_candidate_count": 120,
        "effective_candidate_pool_size": 100,
        "effective_pool_policy": "registered_materializable_minus_frozen_gold176_rows",
        "search_batch_size": 4,
        "inference_batch_size": 2,
        "search_arm_outer_budget": 16,
        "formal_measurement_repeats": 3,
        "full_ap_required": True,
        "actual_feedback_required": True,
        "legacy_lut_allowed": False,
        "historical_proxy_allowed_as_final_result": False,
        "backend_is_outer_gene": False,
        "precision_dispatch_contract": {
            "fp16": "stage5_route_b_fp16_auto",
            "int8": "stage5_route_b_int8_auto_decomp",
            "legacy_native_dp4a_allowed": False,
            "hand_rewrite_allowed": False,
        },
    }
    arms = [
        {
            "arm_id": "original_default",
            "action": "fixed_native_reference",
            "width": [64, 128, 256],
            "q_modes": ["fp32"],
            "outer_budget": 1,
            "backend": "pytorch_eager",
            "backend_tuning_allowed": False,
            "reuse_policy": "reuse_only_if_exact_contract_else_repeat_fixed_point",
        },
        {
            "arm_id": "compression_only",
            "action": "hardware_blind_pq_search",
            "q_modes": ["fp16", "int8"],
            "outer_budget": 16,
            "acquisition_features": [
                "widths",
                "q_mode_one_hot",
                "parameter_count",
                "flops",
                "ap_surrogate",
                "static_graph_features",
            ],
            "target_backend_labels_allowed_in_acquisition": False,
            "selected_candidates_measured_on_target_profile": True,
            "inner_effort": "backend_standard_build_no_extra_tuning",
        },
        {
            "arm_id": "schedule_only",
            "action": "fixed_base_backend_tuning",
            "width": [64, 128, 256],
            "q_modes": ["fp32"],
            "outer_budget": 1,
            "backend_contracts": {
                "tvm": {
                    "max_trials_global": 32,
                    "seed": 0,
                    "database_required": True,
                    "wallclock_timeout_s_per_candidate": 7200,
                    "gpu_hour_cap_per_candidate": 2.0,
                },
                "trt": {
                    "precision": "fp32",
                    "builder_optimization_level": "runtime_default_frozen",
                    "timing_cache_policy": "fresh_empty",
                    "wallclock_timeout_s_per_candidate": 1800,
                    "gpu_hour_cap_per_candidate": 0.5,
                },
            },
            "trial_and_gpu_hour_accounting_required": True,
        },
        {
            "arm_id": "compress_then_tune",
            "action": "serial_compress_then_backend_tune",
            "q_modes": ["fp16", "int8"],
            "outer_budget": {"screen": 12, "locked": 4},
            "lock_before_tuning": True,
            "equal_inner_effort_per_locked_candidate": True,
            "runtime_budget_mutation_allowed": False,
            "backend_contract_ref": "schedule_only.backend_contracts",
        },
        {
            "arm_id": "tune_then_compress",
            "action": "serial_backend_tune_then_compress",
            "q_modes": ["fp16", "int8"],
            "outer_budget": {"base_tune": 1, "compressed_attempts": 16},
            "base_policy_frozen_before_compression": True,
            "compressed_shape_retune_allowed": False,
            "fallback_allowed": False,
            "incompatibility_is_feasibility_failure": True,
            "failure_rate_required": True,
        },
        {
            "arm_id": "joint_shcosearch",
            "action": "outer_pq_inner_backend_feedback",
            "q_modes": ["fp16", "int8"],
            "outer_budget": 16,
            "search_batch_size": 4,
            "feedback_contract": "actual_feedback_v3",
            "joint_evidence": {
                "tvm": "results/stage5_pyramid_actual_v3_20260720/closure_pyramid_v1/S5-PYR-TVM_closure.json",
                "trt": "results/stage5_pyramid_actual_v3_20260720/closure_pyramid_v1/S5-PYR-TRT_closure.json",
            },
            "joint_summary_evidence": "results/stage5_pyramid_actual_v3_20260720/closure_pyramid_v1/pyramid_actual_v3_closure_summary.json",
            "joint_online_rows_evidence": "results/stage5_pyramid_actual_v3_20260720/closure_pyramid_v1/pyramid_actual_v3_online_rows.csv",
            "inner_effort_accounting_required": True,
        },
    ]
    return deepcopy(
        {
            "schema_version": "stage6_pyramid_arm_manifest_v2",
            "experiment_id": "stage6-pyramid-h800-six-arm-v1",
            "independent_backends": ["tvm", "trt"],
            "paper_eligible": False,
            "launch_allowed": False,
            "common_contract": common,
            "arms": arms,
        }
    )


def validate_stage6_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    arms = list(manifest.get("arms") or [])
    by_id = {str(arm.get("arm_id")): arm for arm in arms}
    if tuple(arm.get("arm_id") for arm in arms) != ARM_IDS:
        failures.append("six_arm_identity_or_order_mismatch")
    common = manifest.get("common_contract") or {}
    if common.get("genome") != ["w0", "w1", "w2", "q_mode"]:
        failures.append("outer_genome_contract_mismatch")
    if common.get("backend_is_outer_gene") is not False:
        failures.append("backend_must_not_be_outer_gene")
    dispatch = common.get("precision_dispatch_contract") or {}
    if (
        dispatch.get("int8") != "stage5_route_b_int8_auto_decomp"
        or dispatch.get("legacy_native_dp4a_allowed") is not False
        or dispatch.get("hand_rewrite_allowed") is not False
    ):
        failures.append("formal_int8_dispatch_contract_mismatch")
    if (
        common.get("theoretical_width_grid_size") != 343
        or common.get("theoretical_candidate_grid_size") != 686
        or common.get("registered_width_count") != 60
        or common.get("registered_candidate_count") != 120
        or common.get("effective_candidate_pool_size") != 100
        or common.get("search_batch_size") != 4
        or common.get("inference_batch_size") != 2
    ):
        failures.append("effective_candidate_pool_contract_mismatch")

    compression = by_id.get("compression_only", {})
    features = set(compression.get("acquisition_features") or [])
    if features & BACKEND_LABELS or compression.get("target_backend_labels_allowed_in_acquisition") is not False:
        failures.append("compression_only_backend_label_leakage")
    if compression.get("q_modes") != ["fp16", "int8"]:
        failures.append("compression_only_q_mode_mismatch")

    forward = by_id.get("compress_then_tune", {})
    if forward.get("outer_budget") != {"screen": 12, "locked": 4}:
        failures.append("compress_then_tune_budget_mismatch")
    if not forward.get("lock_before_tuning") or forward.get("runtime_budget_mutation_allowed") is not False:
        failures.append("compress_then_tune_lock_violation")

    reverse = by_id.get("tune_then_compress", {})
    reverse_ok = all(
        (
            reverse.get("base_policy_frozen_before_compression") is True,
            reverse.get("compressed_shape_retune_allowed") is False,
            reverse.get("fallback_allowed") is False,
            reverse.get("incompatibility_is_feasibility_failure") is True,
            reverse.get("failure_rate_required") is True,
        )
    )
    if not reverse_ok:
        failures.append("tune_then_compress_contract_violation")

    schedule = by_id.get("schedule_only", {})
    backend_contracts = schedule.get("backend_contracts") or {}
    if set(backend_contracts) != {"tvm", "trt"}:
        failures.append("schedule_only_backend_contract_missing")
    elif any(
        float(contract.get("gpu_hour_cap_per_candidate") or 0) <= 0
        or int(contract.get("wallclock_timeout_s_per_candidate") or 0) <= 0
        for contract in backend_contracts.values()
    ):
        failures.append("schedule_only_resource_cap_missing")
    joint = by_id.get("joint_shcosearch", {})
    if set((joint.get("joint_evidence") or {}).keys()) != {"tvm", "trt"}:
        failures.append("joint_evidence_binding_missing")
    if not joint.get("joint_summary_evidence") or not joint.get("joint_online_rows_evidence"):
        failures.append("joint_actual_feedback_evidence_missing")

    return {
        "schema_version": "stage6_arm_manifest_audit_v1",
        "passed": not failures,
        "arm_count": len(arms),
        "failures": failures,
    }
