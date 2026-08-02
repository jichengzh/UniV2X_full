from __future__ import annotations

from collections import Counter
import math
import re
from typing import Any, Mapping, Sequence


TARGETS = ("latency_ms", "energy_j", "ap70")
LABEL_FIELDS = {"latency_ms", "energy_j", "ap30", "ap50", "ap70", "terminal_status"}
EXPECTED_ARMS = {
    ("tvm_auto", "fp16"),
    ("tvm_auto", "int8"),
    ("trt_engine", "fp16"),
    ("trt_engine", "int8"),
}
REQUIRED_REPRO_ARTIFACTS = {
    "gold176_baseline_completion",
    "feedback_cost_model",
    "feedback_update_eval",
    "feedback_completion_ranking_uncertainty_pareto_replay",
}
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _canonical_heads(report: Mapping[str, Any]) -> dict[str, str]:
    output = {}
    for target in TARGETS:
        counts = report.get("targets", {}).get(target, {}).get("selected_candidate_counts")
        if not isinstance(counts, Mapping) or not counts:
            raise ValueError(f"missing candidate selection evidence for {target}")
        maximum = max(int(value) for value in counts.values())
        output[target] = min(str(name) for name, value in counts.items() if int(value) == maximum)
    return output


def _summary(completion: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = completion.get("summary")
    if not isinstance(payload, Mapping):
        raise ValueError("completion report lacks summary")
    return payload


def _eligible_acquisitions(
    baseline: Mapping[str, Any], feedback: Mapping[str, Any]
) -> tuple[list[str], dict[str, Any]]:
    baseline_policies = baseline.get("replay", {}).get("policies", {})
    feedback_policies = feedback.get("replay", {}).get("policies", {})
    random_baseline = baseline_policies.get("random", {})
    random_feedback = feedback_policies.get("random", {})
    random_medians = (
        random_baseline.get("groups_to_95pct_oracle_HV_median"),
        random_feedback.get("groups_to_95pct_oracle_HV_median"),
    )
    if any(value is None for value in random_medians):
        raise ValueError("random replay did not reach the 95% HV threshold")
    comparison = {}
    eligible = []
    for policy in sorted((set(baseline_policies) & set(feedback_policies)) - {"random"}):
        before = baseline_policies[policy]
        after = feedback_policies[policy]
        medians = (
            before.get("groups_to_95pct_oracle_HV_median"),
            after.get("groups_to_95pct_oracle_HV_median"),
        )
        passes = (
            all(value is not None for value in medians)
            and float(medians[0]) <= float(random_medians[0])
            and float(medians[1]) <= float(random_medians[1])
            and float(before.get("success_rate", 0.0)) >= float(random_baseline.get("success_rate", 0.0))
            and float(after.get("success_rate", 0.0)) >= float(random_feedback.get("success_rate", 0.0))
        )
        comparison[policy] = {
            "gold176_median_groups": medians[0],
            "gold176_random_median_groups": random_medians[0],
            "feedback_view_median_groups": medians[1],
            "feedback_view_random_median_groups": random_medians[1],
            "passes": bool(passes),
        }
        if passes:
            eligible.append(policy)
    return eligible, comparison


def _holdout_gate(
    holdout: Mapping[str, Any], merged_rows: Sequence[Mapping[str, Any]]
) -> tuple[bool, dict[str, Any]]:
    groups = holdout.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("independent holdout has no groups")
    for group in groups:
        leaked = sorted(LABEL_FIELDS & set(group))
        if leaked:
            raise ValueError(f"independent holdout contains label fields: {leaked}")
    holdout_id_list = [str(group.get("group_id") or "") for group in groups]
    holdout_ids = set(holdout_id_list)
    used_ids = {str(row.get("group_id") or "") for row in merged_rows}
    overlap = sorted(holdout_ids & used_ids)
    models = sorted({str(group.get("model") or "") for group in groups})
    width_keys = {"x".join(map(str, group.get("width") or [])) for group in groups}
    freeze_audit = holdout.get("freeze_audit", {})
    prior_references = freeze_audit.get(
        "pre_freeze_results_or_script_reference_count_by_width", {}
    )
    passes = (
        holdout.get("schema_version") == "stage4_independent_holdout_v1"
        and
        holdout.get("status") == "frozen_before_source_materialization_and_labels"
        and holdout.get("group_count") == 4
        and len(groups) == 4
        and len(holdout_ids) == 4
        and holdout.get("row_count_after_four_arm_measurement") == 16
        and {
            (str(item[0]), str(item[1]))
            for item in holdout.get("required_arm_product", [])
            if isinstance(item, list) and len(item) == 2
        } == EXPECTED_ARMS
        and not overlap
        and models == ["codriving", "pyramid"]
        and all(holdout_ids)
        and all(group.get("label_state_at_freeze") == "unavailable" for group in groups)
        and holdout.get("freeze_audit", {}).get(
            "overlap_with_gold176_or_feedback16_groups"
        ) == []
        and isinstance(prior_references, Mapping)
        and set(prior_references) == width_keys
        and all(int(value) == 0 for value in prior_references.values())
        and LABEL_FIELDS <= set(freeze_audit.get("forbidden_until_measurement", []))
    )
    return passes, {
        "group_count": len(groups),
        "models": models,
        "overlap_with_training_or_feedback": overlap,
        "status": holdout.get("status"),
        "contract_valid": bool(passes),
    }


def _feedback_protocol_valid(
    feedback_update: Mapping[str, Any],
    canonical_heads: Mapping[str, str],
    merged_rows: Sequence[Mapping[str, Any]],
) -> bool:
    folds = feedback_update.get("folds")
    if not isinstance(folds, list) or len(folds) != 4:
        return False
    heldouts = [str(fold.get("heldout_feedback_group") or "") for fold in folds]
    feedback_groups = set(heldouts)
    if len(feedback_groups) != 4 or not all(feedback_groups):
        return False
    baseline_sets = {tuple(sorted(map(str, fold.get("baseline_fit_groups", [])))) for fold in folds}
    calibration_sets = {tuple(sorted(map(str, fold.get("calibration_groups", [])))) for fold in folds}
    if len(baseline_sets) != 1 or len(calibration_sets) != 1:
        return False
    baseline = set(next(iter(baseline_sets)))
    calibration = set(next(iter(calibration_sets)))
    if not baseline or not calibration or baseline & calibration:
        return False
    for fold, heldout in zip(folds, heldouts):
        added = set(map(str, fold.get("added_feedback_groups", [])))
        updated = set(map(str, fold.get("updated_fit_groups", [])))
        if (
            added != feedback_groups - {heldout}
            or updated != baseline | added
            or heldout in updated
            or updated & calibration
        ):
            return False

    rows_by_group: dict[str, list[Mapping[str, Any]]] = {}
    for row in merged_rows:
        rows_by_group.setdefault(str(row.get("group_id") or ""), []).append(row)
    valid_groups = {
        group_id
        for group_id, group_rows in rows_by_group.items()
        if len(group_rows) == 4
        and all(
            all(
                isinstance(row.get(target), (int, float))
                and not isinstance(row.get(target), bool)
                and math.isfinite(float(row[target]))
                for target in TARGETS
            )
            for row in group_rows
        )
    }
    expected_baseline = {
        group_id
        for group_id in valid_groups
        if {
            (str(row.get("training_source") or ""), str(row.get("split") or ""))
            for row in rows_by_group[group_id]
        } == {("initial_coldstart", "train")}
    }
    expected_calibration = {
        group_id
        for group_id in valid_groups
        if {
            (str(row.get("training_source") or ""), str(row.get("split") or ""))
            for row in rows_by_group[group_id]
        } == {("initial_coldstart", "locked_holdout")}
    }
    expected_feedback = {
        group_id
        for group_id in valid_groups
        if {
            (str(row.get("training_source") or ""), str(row.get("split") or ""))
            for row in rows_by_group[group_id]
        } == {("online_feedback", "online_feedback")}
    }
    if (
        baseline != expected_baseline
        or calibration != expected_calibration
        or feedback_groups != expected_feedback
    ):
        return False

    before = feedback_update.get("before_feedback", {})
    after = feedback_update.get("after_feedback", {})
    row_count = before.get("row_count")
    if row_count != 16 or after.get("row_count") != 16:
        return False
    expected_feedback_ids = {
        str(row.get("manifest_job_id") or "")
        for group_id in expected_feedback
        for row in rows_by_group[group_id]
    }
    for target in TARGETS:
        before_records = before.get("records", {}).get(target)
        after_records = after.get("records", {}).get(target)
        if not isinstance(before_records, list) or not isinstance(after_records, list):
            return False
        before_ids = [str(item.get("manifest_job_id") or "") for item in before_records]
        after_ids = [str(item.get("manifest_job_id") or "") for item in after_records]
        if (
            len(before_ids) != 16
            or len(set(before_ids)) != 16
            or set(before_ids) != expected_feedback_ids
            or set(before_ids) != set(after_ids)
            or {str(item.get("group_id") or "") for item in before_records} != feedback_groups
        ):
            return False
    return (
        feedback_update.get("schema_version") == "stage4_feedback_update_eval_v1"
        and feedback_update.get("canonical_value_heads") == canonical_heads
    )


def _reproducibility_valid(audit: Mapping[str, Any]) -> bool:
    artifacts = audit.get("artifacts")
    if (
        audit.get("schema_version") != "stage4_p1_p3_reproducibility_audit_v1"
        or audit.get("all_exact_match") is not True
        or not isinstance(artifacts, Mapping)
        or not REQUIRED_REPRO_ARTIFACTS <= set(artifacts)
    ):
        return False
    return all(
        isinstance(artifacts[name], Mapping)
        and artifacts[name].get("byte_exact") is True
        and isinstance(artifacts[name].get("original_sha256"), str)
        and SHA256.fullmatch(artifacts[name]["original_sha256"]) is not None
        and artifacts[name].get("original_sha256") == artifacts[name].get("rerun_sha256")
        for name in REQUIRED_REPRO_ARTIFACTS
    )


def _target_scoped_feedback_gain(feedback_update: Mapping[str, Any]) -> list[str]:
    improved = []
    before = feedback_update.get("before_feedback", {}).get("targets", {})
    after = feedback_update.get("after_feedback", {}).get("targets", {})
    for target in TARGETS:
        old = before.get(target, {})
        new = after.get(target, {})
        try:
            if (
                float(new["point_metrics"]["mae"]) < float(old["point_metrics"]["mae"])
                and float(new["interval"]["mean_width"]) <= float(old["interval"]["mean_width"])
                and float(new["interval"]["row_coverage"]) >= float(old["interval"]["row_coverage"])
            ):
                improved.append(target)
        except (KeyError, TypeError, ValueError):
            continue
    return improved


def build_stage4_closure_audit(
    cold_cost_report: Mapping[str, Any],
    baseline_completion: Mapping[str, Any],
    feedback_completion: Mapping[str, Any],
    feedback_update: Mapping[str, Any],
    independent_holdout: Mapping[str, Any],
    *,
    reproducibility_audit: Mapping[str, Any],
    merged_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    canonical_heads = _canonical_heads(cold_cost_report)
    baseline_summary = _summary(baseline_completion)
    feedback_summary = _summary(feedback_completion)
    eligible, acquisition_comparison = _eligible_acquisitions(
        baseline_completion, feedback_completion
    )
    selected_policy = None
    if eligible:
        selected_policy = min(
            eligible,
            key=lambda policy: (
                float(acquisition_comparison[policy]["gold176_median_groups"])
                / float(acquisition_comparison[policy]["gold176_random_median_groups"])
                + float(acquisition_comparison[policy]["feedback_view_median_groups"])
                / float(acquisition_comparison[policy]["feedback_view_random_median_groups"]),
                policy,
            ),
        )

    feedback_protocol_valid = _feedback_protocol_valid(
        feedback_update, canonical_heads, merged_rows
    )
    reproducibility_valid = _reproducibility_valid(reproducibility_audit)
    improved_feedback_targets = _target_scoped_feedback_gain(feedback_update)
    sources = Counter(str(row.get("training_source") or "") for row in merged_rows)
    source_roles_valid = (
        set(sources) == {"initial_coldstart", "online_feedback"}
        and all(count > 0 for count in sources.values())
        and all(
            (row.get("training_source"), row.get("split"))
            in {
                ("initial_coldstart", "train"),
                ("initial_coldstart", "locked_holdout"),
                ("online_feedback", "online_feedback"),
            }
            for row in merged_rows
        )
    )
    holdout_valid, holdout_audit = _holdout_gate(independent_holdout, merged_rows)
    calibrator = baseline_summary.get("selected_uncertainty_method")
    gates = {
        "canonical_value_heads_frozen": feedback_update.get("canonical_value_heads") == canonical_heads,
        "ranker_decision_evidenced": (
            baseline_summary.get("retain_ranker") is False
            and feedback_summary.get("retain_ranker") is False
        ),
        "acquisition_not_worse_than_random": bool(eligible),
        "feedback_update_reproducible": bool(
            feedback_protocol_valid and reproducibility_valid
        ),
        "feedback_has_target_scoped_gain": bool(improved_feedback_targets),
        "training_source_roles_explicit": bool(source_roles_valid),
        "independent_holdout_frozen_before_labels": bool(holdout_valid),
        "uncertainty_calibrator_frozen": (
            calibrator == "lgbm_quantile"
            and feedback_summary.get("selected_uncertainty_method") == calibrator
            and feedback_update.get("uncertainty_method")
            == "lgbm_quantile_plus_group_conformal"
        ),
    }
    closed = all(gates.values())
    return {
        "schema_version": "stage4_p1_p3_closure_audit_v1",
        "stage4_closed": closed,
        "stage5_search_ready": closed,
        "gates": gates,
        "canonical_value_heads": canonical_heads,
        "ranker_policy": "rejected_use_value_heads_only",
        "uncertainty_policy": "lgbm_quantile_plus_group_conformal",
        "selected_acquisition_policy": selected_policy,
        "eligible_acquisition_policies": eligible,
        "acquisition_comparison": acquisition_comparison,
        "feedback_update_scope": (
            "target_specific_update_evidence_not_a_claim_of_uniform_all_target_improvement"
        ),
        "reproducibility_exact_match": reproducibility_valid,
        "feedback_improved_targets": improved_feedback_targets,
        "training_source_rows": dict(sorted(sources.items())),
        "independent_holdout": holdout_audit,
    }


__all__ = ["build_stage4_closure_audit"]
