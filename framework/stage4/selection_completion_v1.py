from __future__ import annotations

import copy
import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from framework.stage4.ranking_pareto_v1 import (
    build_stage4_ranking_pareto_report,
    validate_value_oof_provenance,
)
from framework.stage4.uncertainty_replay_v1 import (
    run_grouped_oof_uncertainty,
    run_offline_closed_loop_replay,
)


SCHEMA = "stage4_selection_completion_v1"
TARGETS = ("latency_ms", "energy_j", "ap70")
EXPECTED_ARMS = {
    ("tvm_auto", "fp16"),
    ("tvm_auto", "int8"),
    ("trt_engine", "fp16"),
    ("trt_engine", "int8"),
}


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _row_id(row: Mapping[str, Any]) -> str:
    return str(row.get("manifest_job_id") or row.get("row_id") or "")


def select_complete_four_arm_groups(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source in rows:
        row = dict(source)
        groups[str(row["group_id"])].append(row)

    selected: list[dict[str, Any]] = []
    excluded = []
    for group_id in sorted(groups):
        group_rows = groups[group_id]
        reasons = []
        arms = {
            (str(row.get("dispatch_key")), str(row.get("q_mode")))
            for row in group_rows
        }
        if len(group_rows) != 4:
            reasons.append(f"row_count_{len(group_rows)}")
        if arms != EXPECTED_ARMS:
            reasons.append("incomplete_four_arm_contract")
        for target in TARGETS:
            if any(not _finite(row.get(target)) for row in group_rows):
                reasons.append(f"non_finite_{target}")
        row_ids = [_row_id(row) for row in group_rows]
        if any(not row_id for row_id in row_ids) or len(set(row_ids)) != len(row_ids):
            reasons.append("empty_or_duplicate_row_id")
        if reasons:
            excluded.append(
                {
                    "group_id": group_id,
                    "reasons": sorted(set(reasons)),
                    "terminal_statuses": sorted(
                        {str(row.get("terminal_status") or "") for row in group_rows}
                    ),
                }
            )
        else:
            selected.extend(group_rows)

    selected.sort(key=lambda row: _row_id(row))
    audit = {
        "input_row_count": len(rows),
        "input_group_count": len(groups),
        "selected_row_count": len(selected),
        "selected_group_count": len({str(row["group_id"]) for row in selected}),
        "excluded_group_count": len(excluded),
        "excluded_groups": excluded,
    }
    return selected, audit


def filter_cost_model_oof_report(
    report: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    selected_ids = {_row_id(row) for row in rows}
    validate_value_oof_provenance(rows, report)
    targets = report.get("targets")
    if not isinstance(targets, Mapping):
        raise ValueError("cost model report lacks targets")
    filtered_targets = {}
    for target in TARGETS:
        payload = targets.get(target)
        predictions = payload.get("oof_predictions") if isinstance(payload, Mapping) else None
        if not isinstance(predictions, list):
            raise ValueError(f"cost model report lacks OOF predictions for {target}")
        selected = [copy.deepcopy(item) for item in predictions if str(item.get("row_id")) in selected_ids]
        observed_ids = {str(item.get("row_id")) for item in selected}
        if observed_ids != selected_ids or len(selected) != len(selected_ids):
            raise ValueError(f"filtered OOF predictions do not cover selected rows for {target}")
        filtered_targets[target] = {
            "outer_folds": copy.deepcopy(payload["outer_folds"]),
            "oof_predictions": selected,
        }
    return {
        "schema_version": "stage4_cost_model_oof_complete_groups_v1",
        "source_schema_version": report.get("schema_version"),
        "split_protocol": report.get("split_protocol"),
        "row_count": len(rows),
        "group_count": len({str(row["group_id"]) for row in rows}),
        "targets": filtered_targets,
    }


def _select_uncertainty_method(
    rows: Sequence[Mapping[str, Any]], report: Mapping[str, Any]
) -> dict[str, Any]:
    target_spreads = {
        target: max(float(row[target]) for row in rows)
        - min(float(row[target]) for row in rows)
        for target in TARGETS
    }
    candidates = {}
    for method, payload in report["methods"].items():
        metrics = payload["targets"]
        target_coverages = [
            float(metrics[target]["fully_covered_group_coverage"])
            for target in TARGETS
        ]
        coverage = float(np.mean(target_coverages))
        relative_width = float(
            np.mean(
                [
                    metrics[target]["mean_width"] / max(target_spreads[target], 1e-12)
                    for target in TARGETS
                ]
            )
        )
        candidates[method] = {
            "mean_fully_covered_group_coverage": coverage,
            "min_target_fully_covered_group_coverage": min(target_coverages),
            "simultaneous_all_targets_group_coverage": float(
                payload["simultaneous_all_targets_group_coverage"]
            ),
            "mean_relative_width": relative_width,
        }
    eligible = [
        method
        for method, metrics in candidates.items()
        if metrics["min_target_fully_covered_group_coverage"] >= 0.80
    ]
    if eligible:
        selected = min(eligible, key=lambda method: candidates[method]["mean_relative_width"])
    else:
        selected = max(
            candidates,
            key=lambda method: (
                candidates[method]["min_target_fully_covered_group_coverage"],
                -candidates[method]["mean_relative_width"],
            ),
        )
    return {
        "targetwise_minimum_coverage_threshold": 0.80,
        "selection_basis": "all_targetwise_group_coverages_then_relative_width",
        "simultaneous_coverage_is_diagnostic_not_nominal_guarantee": True,
        "selected_method": selected,
        "candidates": candidates,
    }


def run_stage4_completion(
    rows: Sequence[Mapping[str, Any]],
    graph_features: Sequence[Mapping[str, Any]],
    capability_profiles: Sequence[Mapping[str, Any]],
    cost_model_selection_report: Mapping[str, Any],
    *,
    outer_splits: int = 5,
    uncertainty_splits: int = 5,
    replay_initial_groups: int = 6,
    replay_budget_groups: int | None = None,
    replay_seeds: Sequence[int] = (0, 1, 2, 3, 4),
    seed: int = 20260716,
) -> dict[str, Any]:
    selected_rows, audit = select_complete_four_arm_groups(rows)
    if not selected_rows:
        raise ValueError("no complete finite four-arm groups available")
    group_count = audit["selected_group_count"]
    budget_groups = group_count if replay_budget_groups is None else replay_budget_groups
    filtered_cost = filter_cost_model_oof_report(cost_model_selection_report, selected_rows)
    ranking_pareto = build_stage4_ranking_pareto_report(
        selected_rows,
        cost_model_selection_report=filtered_cost,
        graph_features=graph_features,
        capability_profiles=capability_profiles,
        outer_splits=outer_splits,
        seed=seed,
    )
    uncertainty = run_grouped_oof_uncertainty(
        selected_rows,
        graph_features,
        capability_profiles,
        n_splits=uncertainty_splits,
        seed=seed,
    )
    uncertainty["row_count"] = len(selected_rows)
    uncertainty["group_count"] = group_count
    uncertainty["selection"] = _select_uncertainty_method(selected_rows, uncertainty)
    replay = run_offline_closed_loop_replay(
        selected_rows,
        graph_features,
        capability_profiles,
        initial_group_count=replay_initial_groups,
        budget_groups=budget_groups,
        seeds=replay_seeds,
    )
    return {
        "schema_version": SCHEMA,
        "evaluation_audit": audit,
        "filtered_cost_model_report": filtered_cost,
        "ranking_pareto": ranking_pareto,
        "uncertainty": uncertainty,
        "replay": replay,
        "summary": {
            "evaluation_row_count": len(selected_rows),
            "evaluation_group_count": group_count,
            "evaluation_row_ids": [_row_id(row) for row in selected_rows],
            "retain_ranker": ranking_pareto["ranker_value_comparison"]["retain_ranker"],
            "selected_uncertainty_method": uncertainty["selection"]["selected_method"],
            "pareto": ranking_pareto["pareto"]["summary"],
            "replay_is_offline_proxy": True,
            "generalization_scope": "within_model_grouped_interpolation_not_model_holdout",
        },
    }


__all__ = [
    "filter_cost_model_oof_report",
    "run_stage4_completion",
    "select_complete_four_arm_groups",
]
