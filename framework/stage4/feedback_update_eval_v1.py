from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from framework.stage4.cost_model_selection_v1 import (
    TARGETS,
    _fit_predict_candidate,
    _metrics,
    encode_rows,
)
from framework.stage4.selection_completion_v1 import select_complete_four_arm_groups
from framework.stage4.uncertainty_replay_v1 import (
    _apply_corrections,
    _group_conformal_corrections,
    _predict_lgbm_interval,
)


def _canonical_heads(report: Mapping[str, Any]) -> dict[str, str]:
    heads = {}
    for target in TARGETS:
        payload = report.get("targets", {}).get(target, {})
        counts = payload.get("selected_candidate_counts")
        if not isinstance(counts, Mapping) or not counts:
            raise ValueError(f"cost report lacks selected_candidate_counts for {target}")
        maximum = max(int(value) for value in counts.values())
        if maximum <= 0:
            raise ValueError(f"cost report did not select a value head for {target}")
        heads[target] = min(
            str(name) for name, value in counts.items() if int(value) == maximum
        )
    return heads


def _indices(rows: Sequence[Mapping[str, Any]], groups: set[str]) -> list[int]:
    return [index for index, row in enumerate(rows) if str(row["group_id"]) in groups]


def _interval_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    covered = [
        float(record["lower"]) <= float(record["truth"]) <= float(record["upper"])
        for record in records
    ]
    by_group: dict[str, list[bool]] = defaultdict(list)
    for record, is_covered in zip(records, covered):
        by_group[str(record["group_id"])].append(is_covered)
    widths = [float(record["upper"]) - float(record["lower"]) for record in records]
    truths = np.asarray([float(record["truth"]) for record in records], dtype=float)
    spread = max(float(np.quantile(truths, 0.9) - np.quantile(truths, 0.1)), 1e-12)
    return {
        "row_coverage": float(np.mean(covered)),
        "fully_covered_group_coverage": float(np.mean([all(flags) for flags in by_group.values()])),
        "mean_width": float(np.mean(widths)),
        "relative_mean_width": float(np.mean(widths) / spread),
    }


def _mode_summary(
    records: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    groups = sorted({str(record["group_id"]) for items in records.values() for record in items})
    simultaneous = []
    for group_id in groups:
        simultaneous.append(all(
            float(record["lower"]) <= float(record["truth"]) <= float(record["upper"])
            for target in TARGETS
            for record in records[target]
            if str(record["group_id"]) == group_id
        ))
    targets = {}
    for target in TARGETS:
        items = list(records[target])
        truth = np.asarray([float(item["truth"]) for item in items], dtype=float)
        prediction = np.asarray([float(item["prediction"]) for item in items], dtype=float)
        targets[target] = {
            "point_metrics": _metrics(truth, prediction),
            "interval": _interval_summary(items),
        }
    return {
        "row_count": len(records[TARGETS[0]]),
        "group_count": len(groups),
        "simultaneous_group_coverage": float(np.mean(simultaneous)),
        "targets": targets,
        "records": {target: list(records[target]) for target in TARGETS},
    }


def run_feedback_update_evaluation(
    rows: Sequence[Mapping[str, Any]],
    graph_features: Sequence[Mapping[str, Any]],
    capability_profiles: Sequence[Mapping[str, Any]],
    cold_cost_report: Mapping[str, Any],
    *,
    seed: int = 20260716,
) -> dict[str, Any]:
    source_rows = [dict(row) for row in rows]
    if any(row.get("training_source") not in {"initial_coldstart", "online_feedback"} for row in source_rows):
        raise ValueError("every row must declare training_source")
    selected_rows, selection_audit = select_complete_four_arm_groups(source_rows)
    cold_fit_groups = {
        str(row["group_id"])
        for row in selected_rows
        if row["training_source"] == "initial_coldstart" and row["split"] == "train"
    }
    calibration_groups = {
        str(row["group_id"])
        for row in selected_rows
        if row["training_source"] == "initial_coldstart" and row["split"] == "locked_holdout"
    }
    feedback_groups = {
        str(row["group_id"])
        for row in selected_rows
        if row["training_source"] == "online_feedback" and row["split"] == "online_feedback"
    }
    if not cold_fit_groups:
        raise ValueError("no initial_coldstart train groups")
    if not calibration_groups:
        raise ValueError("no initial_coldstart locked_holdout calibration groups")
    if len(feedback_groups) != 4:
        raise ValueError("feedback update evaluation requires exactly four complete feedback groups")
    if (cold_fit_groups | calibration_groups) & feedback_groups:
        raise ValueError("feedback groups overlap cold-start roles")

    encoded = encode_rows(selected_rows, graph_features, capability_profiles)
    canonical_heads = _canonical_heads(cold_cost_report)
    records: dict[str, dict[str, list[dict[str, Any]]]] = {
        mode: {target: [] for target in TARGETS}
        for mode in ("before_feedback", "after_feedback")
    }
    folds = []
    calibration_indices = _indices(selected_rows, calibration_groups)
    for fold_index, heldout_group in enumerate(sorted(feedback_groups)):
        added_feedback = feedback_groups - {heldout_group}
        fit_groups_by_mode = {
            "before_feedback": cold_fit_groups,
            "after_feedback": cold_fit_groups | added_feedback,
        }
        test_indices = _indices(selected_rows, {heldout_group})
        folds.append({
            "fold": fold_index,
            "heldout_feedback_group": heldout_group,
            "baseline_fit_groups": sorted(cold_fit_groups),
            "updated_fit_groups": sorted(cold_fit_groups | added_feedback),
            "added_feedback_groups": sorted(added_feedback),
            "calibration_groups": sorted(calibration_groups),
        })
        for mode, fit_groups in fit_groups_by_mode.items():
            fit_indices = _indices(selected_rows, set(fit_groups))
            for target_index, target in enumerate(TARGETS):
                fit_y = np.asarray([float(selected_rows[index][target]) for index in fit_indices])
                test_y = np.asarray([float(selected_rows[index][target]) for index in test_indices])
                point_prediction = _fit_predict_candidate(
                    canonical_heads[target],
                    encoded.matrix[fit_indices],
                    fit_y,
                    encoded.matrix[test_indices],
                    [selected_rows[index] for index in fit_indices],
                    [selected_rows[index] for index in test_indices],
                    seed=seed + 100 * fold_index + target_index,
                )
                predict_matrix = np.vstack([
                    encoded.matrix[calibration_indices], encoded.matrix[test_indices]
                ])
                lower, _median, upper = _predict_lgbm_interval(
                    encoded.matrix[fit_indices],
                    fit_y,
                    predict_matrix,
                    seed=seed + 1000 + 100 * fold_index + target_index,
                )
                split_at = len(calibration_indices)
                corrections = _group_conformal_corrections(
                    selected_rows,
                    calibration_indices,
                    lower[:split_at],
                    upper[:split_at],
                    target,
                )
                test_lower, test_upper = _apply_corrections(
                    selected_rows,
                    test_indices,
                    lower[split_at:],
                    upper[split_at:],
                    corrections,
                )
                for local_index, row_index in enumerate(test_indices):
                    row = selected_rows[row_index]
                    records[mode][target].append({
                        "manifest_job_id": str(row["manifest_job_id"]),
                        "group_id": str(row["group_id"]),
                        "model": str(row["model"]),
                        "truth": float(test_y[local_index]),
                        "prediction": float(point_prediction[local_index]),
                        "lower": float(test_lower[local_index]),
                        "upper": float(test_upper[local_index]),
                    })
    return {
        "schema_version": "stage4_feedback_update_eval_v1",
        "protocol": "leave_one_feedback_group_out_with_frozen_cold_heads_and_locked_calibration",
        "canonical_value_heads": canonical_heads,
        "uncertainty_method": "lgbm_quantile_plus_group_conformal",
        "selection_audit": selection_audit,
        "folds": folds,
        "before_feedback": _mode_summary(records["before_feedback"]),
        "after_feedback": _mode_summary(records["after_feedback"]),
    }


__all__ = ["run_feedback_update_evaluation"]
