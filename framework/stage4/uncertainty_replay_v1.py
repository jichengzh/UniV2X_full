from __future__ import annotations

import math
import warnings
from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor

from framework.stage4.cost_model_selection_v1 import encode_rows, grouped_folds

try:  # pragma: no cover - exercised when lightgbm is available in the runtime.
    from lightgbm import LGBMRegressor
except Exception:  # pragma: no cover - deterministic sklearn fallback for lean CI.
    LGBMRegressor = None  # type: ignore[assignment]


TARGETS = ("latency_ms", "energy_j", "ap70")
ARMS = {
    ("tvm_auto", "fp16"),
    ("tvm_auto", "int8"),
    ("trt_engine", "fp16"),
    ("trt_engine", "int8"),
}
REPLAY_POLICIES = (
    "pareto_uncertainty",
    "uncertainty_only",
    "expected_hv_improvement",
    "predicted_frontier_diversity",
    "ehvi_uncertainty_diversity",
    "random",
)


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _row_id(row: Mapping[str, Any]) -> str:
    return str(row.get("manifest_job_id") or row.get("id") or "")


def _group_id(row: Mapping[str, Any]) -> str:
    return str(row["group_id"])


def _complete_groups(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[_group_id(row)].append(index)
    for group_id, indices in groups.items():
        arms = {
            (str(rows[index]["dispatch_key"]), str(rows[index]["q_mode"]))
            for index in indices
        }
        if len(indices) != 4 or arms != ARMS:
            raise ValueError(f"group is not a complete four-arm group: {group_id}")
    return dict(sorted(groups.items()))


def _target_values(rows: Sequence[Mapping[str, Any]], indices: Sequence[int], target: str) -> np.ndarray:
    values = [float(rows[index][target]) for index in indices]
    if not all(_finite(value) for value in values):
        raise ValueError(f"non-finite target value for {target}")
    return np.asarray(values, dtype=float)


def _split_fit_calibration_groups(
    train_groups: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    by_model: dict[str, list[str]] = defaultdict(list)
    model_by_group = {
        _group_id(row): str(row["model"])
        for row in rows
    }
    for group_id in sorted(train_groups):
        by_model[model_by_group[group_id]].append(group_id)
    rng = np.random.default_rng(seed)
    calibration: list[str] = []
    for model in sorted(by_model):
        candidates = list(by_model[model])
        rng.shuffle(candidates)
        take = (
            min(len(candidates) - 1, max(1, math.ceil(0.20 * len(candidates))))
            if len(candidates) >= 2
            else 0
        )
        calibration.extend(sorted(candidates[:take]))
    fit = sorted(set(train_groups) - set(calibration))
    if not fit or not calibration:
        raise ValueError("outer train split must leave both fit and calibration groups")
    return tuple(fit), tuple(sorted(calibration))


def _indices_for_groups(rows: Sequence[Mapping[str, Any]], groups: set[str]) -> list[int]:
    return [index for index, row in enumerate(rows) if _group_id(row) in groups]


def _build_lgbm_quantile(alpha: float, seed: int):
    if LGBMRegressor is not None:
        return LGBMRegressor(
            objective="quantile",
            alpha=alpha,
            n_estimators=8,
            learning_rate=0.05,
            num_leaves=7,
            max_depth=4,
            min_child_samples=2,
            subsample=1.0,
            colsample_bytree=1.0,
            random_state=seed,
            n_jobs=1,
            verbosity=-1,
        )
    return GradientBoostingRegressor(
        loss="quantile",
        alpha=alpha,
        n_estimators=8,
        learning_rate=0.05,
        max_depth=2,
        random_state=seed,
    )


def _predict_lgbm_interval(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_predict: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predictions = []
    for alpha in (0.05, 0.50, 0.95):
        model = _build_lgbm_quantile(alpha, seed)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names.*",
                category=UserWarning,
            )
            model.fit(x_fit, y_fit)
            predictions.append(np.asarray(model.predict(x_predict), dtype=float))
    lower = np.minimum(predictions[0], predictions[2])
    upper = np.maximum(predictions[0], predictions[2])
    median = predictions[1]
    return lower, median, upper


def _predict_extra_trees_interval(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_predict: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model = ExtraTreesRegressor(
        n_estimators=16,
        max_depth=8,
        min_samples_leaf=1,
        max_features=0.8,
        bootstrap=True,
        random_state=seed,
        n_jobs=1,
    )
    model.fit(x_fit, y_fit)
    members = np.vstack([tree.predict(x_predict) for tree in model.estimators_])
    lower, median, upper = np.quantile(members, [0.05, 0.50, 0.95], axis=0)
    return lower, median, upper


def _group_conformal_corrections(
    rows: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
    lower: np.ndarray,
    upper: np.ndarray,
    target: str,
) -> dict[str, float]:
    by_model_group: dict[tuple[str, str], list[float]] = defaultdict(list)
    for local_index, row_index in enumerate(indices):
        row = rows[row_index]
        truth = float(row[target])
        score = max(float(lower[local_index]) - truth, truth - float(upper[local_index]), 0.0)
        by_model_group[(str(row["model"]), _group_id(row))].append(score)
    by_model: dict[str, list[float]] = defaultdict(list)
    for (model, _group), scores in by_model_group.items():
        by_model[model].append(max(scores))
    corrections = {}
    for model, scores in by_model.items():
        ordered = sorted(scores)
        rank = min(len(ordered), max(1, math.ceil((len(ordered) + 1) * 0.90)))
        corrections[model] = float(ordered[rank - 1])
    return corrections


def _apply_corrections(
    rows: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
    lower: np.ndarray,
    upper: np.ndarray,
    corrections: Mapping[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    adjusted_lower = np.asarray(lower, dtype=float).copy()
    adjusted_upper = np.asarray(upper, dtype=float).copy()
    for local_index, row_index in enumerate(indices):
        correction = float(corrections.get(str(rows[row_index]["model"]), 0.0))
        adjusted_lower[local_index] -= correction
        adjusted_upper[local_index] += correction
    return adjusted_lower, adjusted_upper


def _interval_metrics(
    rows: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    target: str,
    calibration_group_total: int,
) -> dict[str, Any]:
    covered = []
    widths = []
    by_group: dict[str, list[bool]] = defaultdict(list)
    for record in records:
        truth = float(record["truth"])
        lower = float(record["lower"])
        upper = float(record["upper"])
        is_covered = lower <= truth <= upper
        covered.append(is_covered)
        widths.append(upper - lower)
        by_group[str(record["group_id"])].append(is_covered)
    return {
        "target": target,
        "row_coverage": float(np.mean(covered)) if covered else 0.0,
        "fully_covered_group_coverage": (
            float(np.mean([all(items) for items in by_group.values()])) if by_group else 0.0
        ),
        "mean_width": float(np.mean(widths)) if widths else 0.0,
        "row_count": len(records),
        "group_count": len(by_group),
        "calibration_group_count": int(calibration_group_total),
    }


def _simultaneous_group_coverage(
    by_target: Mapping[str, Sequence[Mapping[str, Any]]],
) -> float:
    covered_by_group: dict[str, list[bool]] = defaultdict(list)
    for target in TARGETS:
        for record in by_target[target]:
            covered_by_group[str(record["group_id"])].append(
                float(record["lower"]) <= float(record["truth"]) <= float(record["upper"])
            )
    return (
        float(np.mean([all(flags) for flags in covered_by_group.values()]))
        if covered_by_group
        else 0.0
    )


def run_grouped_oof_uncertainty(
    rows: Sequence[Mapping[str, Any]],
    graph_features: Sequence[Mapping[str, Any]],
    capability_profiles: Sequence[Mapping[str, Any]],
    *,
    n_splits: int = 5,
    seed: int = 0,
) -> dict[str, Any]:
    rows = [dict(row) for row in rows]
    _complete_groups(rows)
    encoded = encode_rows(rows, graph_features, capability_profiles)
    folds = grouped_folds(rows, n_splits=n_splits, seed=seed)
    methods = {
        "lgbm_quantile": _predict_lgbm_interval,
        "extra_trees_ensemble": _predict_extra_trees_interval,
    }
    records: dict[str, dict[str, list[dict[str, Any]]]] = {
        method: {target: [] for target in TARGETS}
        for method in methods
    }
    fold_reports = []
    for fold_index, fold in enumerate(folds):
        fit_groups, calibration_groups = _split_fit_calibration_groups(
            fold.train_groups, rows, seed=seed + fold_index
        )
        fit_indices = _indices_for_groups(rows, set(fit_groups))
        calibration_indices = _indices_for_groups(rows, set(calibration_groups))
        test_indices = list(fold.test_indices)
        fold_reports.append(
            {
                "fold": fold_index,
                "train_groups": list(fold.train_groups),
                "fit_groups": list(fit_groups),
                "calibration_groups": list(calibration_groups),
                "test_groups": list(fold.test_groups),
            }
        )
        for target in TARGETS:
            y_fit = _target_values(rows, fit_indices, target)
            y_calibration = _target_values(rows, calibration_indices, target)
            y_test = _target_values(rows, test_indices, target)
            x_fit = encoded.matrix[fit_indices]
            x_calibration = encoded.matrix[calibration_indices]
            x_test = encoded.matrix[test_indices]
            x_predict = np.vstack([x_calibration, x_test])
            split_at = len(calibration_indices)
            for method_name, predictor in methods.items():
                predicted_lower, predicted_median, predicted_upper = predictor(
                    x_fit, y_fit, x_predict, seed=seed + fold_index
                )
                cal_lower = predicted_lower[:split_at]
                cal_upper = predicted_upper[:split_at]
                corrections = _group_conformal_corrections(
                    rows, calibration_indices, cal_lower, cal_upper, target
                )
                lower = predicted_lower[split_at:]
                median = predicted_median[split_at:]
                upper = predicted_upper[split_at:]
                lower, upper = _apply_corrections(rows, test_indices, lower, upper, corrections)
                for local_index, row_index in enumerate(test_indices):
                    records[method_name][target].append(
                        {
                            "manifest_job_id": _row_id(rows[row_index]),
                            "group_id": _group_id(rows[row_index]),
                            "model": str(rows[row_index]["model"]),
                            "truth": float(y_test[local_index]),
                            "lower": float(lower[local_index]),
                            "median": float(median[local_index]),
                            "upper": float(upper[local_index]),
                        }
                    )
    method_reports = {}
    for method_name, by_target in records.items():
        all_ids = sorted({record["manifest_job_id"] for items in by_target.values() for record in items})
        method_reports[method_name] = {
            "oof_row_count": len(all_ids),
            "oof_manifest_job_ids": all_ids,
            "targets": {
                target: _interval_metrics(
                    rows,
                    by_target[target],
                    target,
                    sum(len(fold["calibration_groups"]) for fold in fold_reports),
                )
                for target in TARGETS
            },
            "simultaneous_all_targets_group_coverage": _simultaneous_group_coverage(
                by_target
            ),
            "intervals": by_target,
        }
    expected_ids = {_row_id(row) for row in rows}
    for method_name, report in method_reports.items():
        if set(report["oof_manifest_job_ids"]) != expected_ids:
            raise AssertionError(f"{method_name} did not produce exactly one OOF interval per row")
    return {
        "schema_version": "stage4_uncertainty_oof_v1",
        "mode": "grouped_outer_oof_uncertainty",
        "targets": list(TARGETS),
        "feature_names": list(encoded.feature_names),
        "folds": fold_reports,
        "methods": method_reports,
    }


def _normalized_points(rows: Sequence[Mapping[str, Any]], selected_groups: set[str]) -> list[tuple[float, float, float]]:
    selected = [row for row in rows if _group_id(row) in selected_groups]
    if not selected:
        return []
    all_values = {
        target: np.asarray([float(row[target]) for row in rows], dtype=float)
        for target in TARGETS
    }
    bounds = {
        target: (float(np.min(values)), float(np.max(values)))
        for target, values in all_values.items()
    }
    points = []
    for row in selected:
        ap_min, ap_max = bounds["ap70"]
        lat_min, lat_max = bounds["latency_ms"]
        energy_min, energy_max = bounds["energy_j"]
        points.append(
            (
                (ap_max - float(row["ap70"])) / max(ap_max - ap_min, 1e-12),
                (float(row["latency_ms"]) - lat_min) / max(lat_max - lat_min, 1e-12),
                (float(row["energy_j"]) - energy_min) / max(energy_max - energy_min, 1e-12),
            )
        )
    return points


def _hypervolume_2d(points: Sequence[tuple[float, float]], ref: tuple[float, float]) -> float:
    eligible = sorted({point for point in points if point[0] < ref[0] and point[1] < ref[1]})
    if not eligible:
        return 0.0
    xs = sorted({point[0] for point in eligible} | {ref[0]})
    area = 0.0
    for left, right in zip(xs, xs[1:]):
        active_y = [y for x, y in eligible if x <= left]
        area += (right - left) * max(0.0, ref[1] - min(active_y))
    return area


def _hypervolume(points: Sequence[tuple[float, float, float]]) -> float:
    reference = (1.05, 1.05, 1.05)
    eligible = sorted({point for point in points if all(point[i] < reference[i] for i in range(3))})
    eligible = [
        point
        for index, point in enumerate(eligible)
        if not any(
            all(other[axis] <= point[axis] for axis in range(3))
            and any(other[axis] < point[axis] for axis in range(3))
            for other_index, other in enumerate(eligible)
            if other_index != index
        )
    ]
    if not eligible:
        return 0.0
    xs = sorted({point[0] for point in eligible} | {reference[0]})
    volume = 0.0
    for left, right in zip(xs, xs[1:]):
        active = [(y, z) for x, y, z in eligible if x <= left]
        volume += (right - left) * _hypervolume_2d(active, (reference[1], reference[2]))
    return float(volume)


def _model_internal_hv(
    rows: Sequence[Mapping[str, Any]],
    selected_groups: set[str],
) -> tuple[float, dict[str, float], dict[str, float]]:
    selected_hv = {}
    oracle_hv = {}
    ratios = []
    for model_name in sorted({str(row["model"]) for row in rows}):
        model_rows = [row for row in rows if str(row["model"]) == model_name]
        model_groups = {_group_id(row) for row in model_rows}
        oracle = _hypervolume(_normalized_points(model_rows, model_groups))
        observed = _hypervolume(_normalized_points(model_rows, selected_groups & model_groups))
        oracle_hv[model_name] = oracle
        selected_hv[model_name] = observed
        ratios.append(observed / max(oracle, 1e-12))
    return float(np.mean(ratios)), selected_hv, oracle_hv


def _predicted_frontier_indices(
    rows: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
    predictions: Mapping[str, np.ndarray],
) -> set[int]:
    frontier = set()
    by_model: dict[str, list[int]] = defaultdict(list)
    for local_index, row_index in enumerate(indices):
        by_model[str(rows[row_index]["model"])].append(local_index)
    for local_indices in by_model.values():
        for index in local_indices:
            dominated = any(
                predictions["ap70"][other] >= predictions["ap70"][index]
                and predictions["latency_ms"][other] <= predictions["latency_ms"][index]
                and predictions["energy_j"][other] <= predictions["energy_j"][index]
                and (
                    predictions["ap70"][other] > predictions["ap70"][index]
                    or predictions["latency_ms"][other] < predictions["latency_ms"][index]
                    or predictions["energy_j"][other] < predictions["energy_j"][index]
                )
                for other in local_indices
                if other != index
            )
            if not dominated:
                frontier.add(index)
    return frontier


def _pareto_uncertainty_choice(
    rows: Sequence[Mapping[str, Any]],
    matrix: np.ndarray,
    measured_groups: set[str],
    candidates: Sequence[str],
) -> tuple[str, dict[str, Any]]:
    measured_indices = [
        index for index, row in enumerate(rows) if _group_id(row) in measured_groups
    ]
    candidate_indices = [
        index for index, row in enumerate(rows) if _group_id(row) in set(candidates)
    ]
    predictions: dict[str, np.ndarray] = {}
    normalized_uncertainty = np.zeros(len(candidate_indices), dtype=float)
    for target_index, target in enumerate(TARGETS):
        train_y = _target_values(rows, measured_indices, target)
        model = ExtraTreesRegressor(
            n_estimators=24,
            max_depth=6,
            min_samples_leaf=2,
            max_features=0.8,
            random_state=1701 + 31 * len(measured_groups) + target_index,
            n_jobs=1,
        )
        model.fit(matrix[measured_indices], train_y)
        tree_predictions = np.vstack(
            [tree.predict(matrix[candidate_indices]) for tree in model.estimators_]
        )
        predictions[target] = np.mean(tree_predictions, axis=0)
        spread = max(float(np.max(train_y) - np.min(train_y)), 1e-12)
        normalized_uncertainty += np.std(tree_predictions, axis=0) / spread
    normalized_uncertainty /= len(TARGETS)
    frontier = _predicted_frontier_indices(rows, candidate_indices, predictions)
    scored = []
    for group_id in candidates:
        local_indices = [
            local_index
            for local_index, row_index in enumerate(candidate_indices)
            if _group_id(rows[row_index]) == group_id
        ]
        frontier_hits = sum(index in frontier for index in local_indices)
        group_uncertainty = float(np.max(normalized_uncertainty[local_indices]))
        scored.append((frontier_hits > 0, group_uncertainty, frontier_hits, group_id))
    winner = max(scored)
    return winner[-1], {
        "predicted_frontier_candidate_rows": len(frontier),
        "selected_group_frontier_hits": int(winner[2]),
        "selected_group_uncertainty": float(winner[1]),
        "acquisition_model": "extra_trees_ensemble_on_measured_rows",
        "acquisition_fit_scope": "measured_complete_groups_only",
    }


def _candidate_acquisition_context(
    rows: Sequence[Mapping[str, Any]],
    matrix: np.ndarray,
    measured_groups: set[str],
    candidates: Sequence[str],
) -> dict[str, Any]:
    measured_indices = [
        index for index, row in enumerate(rows) if _group_id(row) in measured_groups
    ]
    candidate_set = set(candidates)
    candidate_indices = [
        index for index, row in enumerate(rows) if _group_id(row) in candidate_set
    ]
    means: dict[str, np.ndarray] = {}
    members: dict[str, np.ndarray] = {}
    uncertainty = np.zeros(len(candidate_indices), dtype=float)
    for target_index, target in enumerate(TARGETS):
        train_y = _target_values(rows, measured_indices, target)
        model = ExtraTreesRegressor(
            n_estimators=24,
            max_depth=6,
            min_samples_leaf=2,
            max_features=0.8,
            random_state=1701 + 31 * len(measured_groups) + target_index,
            n_jobs=1,
        )
        model.fit(matrix[measured_indices], train_y)
        target_members = np.vstack(
            [tree.predict(matrix[candidate_indices]) for tree in model.estimators_]
        )
        members[target] = target_members
        means[target] = np.mean(target_members, axis=0)
        spread = max(float(np.max(train_y) - np.min(train_y)), 1e-12)
        uncertainty += np.std(target_members, axis=0) / spread
    uncertainty /= len(TARGETS)

    frontier = _predicted_frontier_indices(rows, candidate_indices, means)
    local_by_group = {
        group_id: [
            local_index
            for local_index, row_index in enumerate(candidate_indices)
            if _group_id(rows[row_index]) == group_id
        ]
        for group_id in candidates
    }
    measured_by_group = {
        group_id: [
            index for index in measured_indices if _group_id(rows[index]) == group_id
        ]
        for group_id in sorted(measured_groups)
    }
    feature_scale = np.std(matrix[measured_indices], axis=0)
    feature_scale = np.where(feature_scale > 1e-12, feature_scale, 1.0)
    measured_centroids = np.vstack(
        [np.mean(matrix[indices], axis=0) for indices in measured_by_group.values()]
    )
    feature_diversity = {}
    for group_id, local_indices in local_by_group.items():
        row_indices = [candidate_indices[index] for index in local_indices]
        centroid = np.mean(matrix[row_indices], axis=0)
        distances = np.linalg.norm(
            (measured_centroids - centroid) / feature_scale,
            axis=1,
        )
        feature_diversity[group_id] = float(np.min(distances))
    return {
        "measured_indices": measured_indices,
        "candidate_indices": candidate_indices,
        "local_by_group": local_by_group,
        "means": means,
        "members": members,
        "uncertainty": uncertainty,
        "frontier": frontier,
        "feature_diversity": feature_diversity,
    }


def _acquisition_bounds(
    rows: Sequence[Mapping[str, Any]], context: Mapping[str, Any], model: str
) -> dict[str, tuple[float, float]]:
    measured_indices = [
        index for index in context["measured_indices"] if str(rows[index]["model"]) == model
    ]
    candidate_local = [
        local_index
        for local_index, row_index in enumerate(context["candidate_indices"])
        if str(rows[row_index]["model"]) == model
    ]
    bounds = {}
    for target in TARGETS:
        values = [float(rows[index][target]) for index in measured_indices]
        values.extend(float(context["means"][target][index]) for index in candidate_local)
        bounds[target] = (float(np.min(values)), float(np.max(values)))
    return bounds


def _normalize_acquisition_point(
    values: Mapping[str, float], bounds: Mapping[str, tuple[float, float]]
) -> tuple[float, float, float]:
    normalized = {}
    for target in TARGETS:
        lower, upper = bounds[target]
        scale = max(upper - lower, 1e-6 * max(abs(lower), abs(upper), 1.0))
        if target == "ap70":
            value = (upper - float(values[target])) / scale
        else:
            value = (float(values[target]) - lower) / scale
        normalized[target] = float(np.clip(value, -0.05, 1.05))
    return normalized["ap70"], normalized["latency_ms"], normalized["energy_j"]


def _expected_hv_scores(
    rows: Sequence[Mapping[str, Any]], context: Mapping[str, Any], candidates: Sequence[str]
) -> dict[str, float]:
    scores = {}
    for group_id in candidates:
        local_indices = context["local_by_group"][group_id]
        model = str(rows[context["candidate_indices"][local_indices[0]]]["model"])
        bounds = _acquisition_bounds(rows, context, model)
        measured_rows = [
            rows[index]
            for index in context["measured_indices"]
            if str(rows[index]["model"]) == model
        ]
        base_points = [
            _normalize_acquisition_point(
                {target: float(row[target]) for target in TARGETS}, bounds
            )
            for row in measured_rows
        ]
        base_hv = _hypervolume(base_points)
        member_count = min(
            8,
            *(context["members"][target].shape[0] for target in TARGETS),
        )
        improvements = []
        for member_index in range(member_count):
            candidate_points = []
            for local_index in local_indices:
                candidate_points.append(
                    _normalize_acquisition_point(
                        {
                            target: float(context["members"][target][member_index, local_index])
                            for target in TARGETS
                        },
                        bounds,
                    )
                )
            improvements.append(max(0.0, _hypervolume([*base_points, *candidate_points]) - base_hv))
        scores[group_id] = float(np.mean(improvements))
    return scores


def _normalized_component(scores: Mapping[str, float]) -> dict[str, float]:
    unique = sorted(set(float(value) for value in scores.values()))
    if len(unique) == 1:
        return {group_id: 0.0 for group_id in scores}
    rank = {value: index / (len(unique) - 1) for index, value in enumerate(unique)}
    return {group_id: float(rank[float(value)]) for group_id, value in scores.items()}


def _deterministic_max(candidates: Sequence[str], *scores: Mapping[str, float]) -> str:
    return min(
        candidates,
        key=lambda group_id: tuple(-float(score[group_id]) for score in scores) + (group_id,),
    )


def _new_policy_choice(
    rows: Sequence[Mapping[str, Any]],
    matrix: np.ndarray,
    measured_groups: set[str],
    candidates: Sequence[str],
    policy: str,
) -> tuple[str, dict[str, Any]]:
    context = _candidate_acquisition_context(rows, matrix, measured_groups, candidates)
    uncertainty_scores = {
        group_id: float(np.max(context["uncertainty"][local_indices]))
        for group_id, local_indices in context["local_by_group"].items()
    }
    frontier_hits = {
        group_id: float(sum(index in context["frontier"] for index in local_indices))
        for group_id, local_indices in context["local_by_group"].items()
    }
    diversity_scores = dict(context["feature_diversity"])
    common = {
        "acquisition_model": "extra_trees_ensemble_on_measured_rows",
        "acquisition_fit_scope": "measured_complete_groups_only",
        "predicted_frontier_candidate_rows": len(context["frontier"]),
    }
    if policy == "uncertainty_only":
        choice = _deterministic_max(candidates, uncertainty_scores)
        return choice, {**common, "selected_group_uncertainty": uncertainty_scores[choice]}
    if policy == "predicted_frontier_diversity":
        choice = _deterministic_max(
            candidates, frontier_hits, diversity_scores, uncertainty_scores
        )
        return choice, {
            **common,
            "selected_group_frontier_hits": int(frontier_hits[choice]),
            "selected_group_feature_diversity": diversity_scores[choice],
            "selected_group_uncertainty": uncertainty_scores[choice],
        }
    ehvi_scores = _expected_hv_scores(rows, context, candidates)
    if policy == "expected_hv_improvement":
        choice = _deterministic_max(candidates, ehvi_scores, uncertainty_scores)
        return choice, {
            **common,
            "selected_group_expected_hv_improvement": ehvi_scores[choice],
            "selected_group_uncertainty": uncertainty_scores[choice],
        }
    if policy == "ehvi_uncertainty_diversity":
        components = [
            _normalized_component(ehvi_scores),
            _normalized_component(uncertainty_scores),
            _normalized_component(diversity_scores),
        ]
        combined = {
            group_id: float(np.mean([component[group_id] for component in components]))
            for group_id in candidates
        }
        choice = _deterministic_max(candidates, combined, ehvi_scores)
        return choice, {
            **common,
            "combination_weights": {
                "expected_hv_improvement": 1 / 3,
                "uncertainty": 1 / 3,
                "feature_diversity": 1 / 3,
            },
            "selected_group_combined_score": combined[choice],
            "selected_group_expected_hv_improvement": ehvi_scores[choice],
            "selected_group_uncertainty": uncertainty_scores[choice],
            "selected_group_feature_diversity": diversity_scores[choice],
        }
    raise ValueError(f"unknown offline replay policy: {policy}")


def _initial_groups(
    rows: Sequence[Mapping[str, Any]], group_count: int, rng: np.random.Generator
) -> set[str]:
    by_model: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        group_id = _group_id(row)
        if group_id not in by_model[str(row["model"])]:
            by_model[str(row["model"])].append(group_id)
    if group_count < len(by_model):
        raise ValueError("initial_group_count must cover every model")
    selected = []
    for model_name in sorted(by_model):
        candidates = sorted(by_model[model_name])
        selected.append(str(rng.choice(candidates)))
    remaining = sorted({ _group_id(row) for row in rows } - set(selected))
    rng.shuffle(remaining)
    selected.extend(remaining[: group_count - len(selected)])
    return set(selected)


def _trace_for_policy(
    rows: Sequence[Mapping[str, Any]],
    matrix: np.ndarray,
    policy: str,
    *,
    initial_group_count: int,
    budget_groups: int,
    seed: int,
    threshold_hv: float,
) -> dict[str, Any]:
    groups = sorted({ _group_id(row) for row in rows })
    rng = np.random.default_rng(seed)
    measured = _initial_groups(rows, initial_group_count, rng)
    initial_groups = sorted(measured)
    steps = []
    groups_to_threshold = None
    while len(measured) <= min(budget_groups, len(groups)):
        hv, hv_by_model, _ = _model_internal_hv(rows, measured)
        if groups_to_threshold is None and hv >= threshold_hv:
            groups_to_threshold = len(measured)
        if len(measured) == min(budget_groups, len(groups)):
            break
        candidates = sorted(set(groups) - measured)
        if policy == "random":
            choice = str(rng.choice(candidates))
            diagnostics = {"acquisition_model": "uniform_random_group"}
        elif policy == "pareto_uncertainty":
            choice, diagnostics = _pareto_uncertainty_choice(
                rows, matrix, measured, candidates
            )
        elif policy in {
            "uncertainty_only",
            "expected_hv_improvement",
            "predicted_frontier_diversity",
            "ehvi_uncertainty_diversity",
        }:
            choice, diagnostics = _new_policy_choice(
                rows, matrix, measured, candidates, policy
            )
        else:
            raise ValueError(f"unknown offline replay policy: {policy}")
        group_rows = [row for row in rows if _group_id(row) == choice]
        steps.append(
            {
                "step": len(steps),
                "policy": policy,
                "selected_group": choice,
                "candidate_group_count": len(candidates),
                "candidate_features_visible": True,
                "candidate_labels_visible_before_measurement": False,
                "initial_groups": initial_groups,
                "measured_manifest_job_ids": sorted(_row_id(row) for row in group_rows),
                "model_internal_hv_fraction_before_measurement": hv,
                "model_internal_hv_by_model_before_measurement": hv_by_model,
                **diagnostics,
            }
        )
        measured.add(choice)
    final_hv, final_hv_by_model, _ = _model_internal_hv(rows, measured)
    if groups_to_threshold is None and final_hv >= threshold_hv:
        groups_to_threshold = len(measured)
    return {
        "seed": seed,
        "initial_groups": initial_groups,
        "sampled_groups": sorted(measured),
        "steps": steps,
        "final_hv": final_hv,
        "final_hv_by_model": final_hv_by_model,
        "groups_to_95pct_oracle_HV": groups_to_threshold,
        "samples_to_95pct_oracle_HV": (
            groups_to_threshold * len(ARMS) if groups_to_threshold is not None else None
        ),
    }


def run_offline_closed_loop_replay(
    rows: Sequence[Mapping[str, Any]],
    graph_features: Sequence[Mapping[str, Any]],
    capability_profiles: Sequence[Mapping[str, Any]],
    *,
    initial_group_count: int = 4,
    budget_groups: int = 16,
    seeds: Sequence[int] = (0, 1, 2),
) -> dict[str, Any]:
    rows = [dict(row) for row in rows]
    groups = _complete_groups(rows)
    encoded = encode_rows(rows, graph_features, capability_profiles)
    if initial_group_count <= 0 or budget_groups < initial_group_count:
        raise ValueError("budget_groups must be at least initial_group_count")
    if budget_groups > len(groups):
        raise ValueError("budget_groups cannot exceed available complete groups")
    oracle_hv, _, oracle_hv_by_model = _model_internal_hv(rows, set(groups))
    threshold_hv = 0.95 * oracle_hv
    reports = {}
    for policy in REPLAY_POLICIES:
        traces = [
            _trace_for_policy(
                rows,
                encoded.matrix,
                policy,
                initial_group_count=initial_group_count,
                budget_groups=budget_groups,
                seed=int(seed),
                threshold_hv=threshold_hv,
            )
            for seed in seeds
        ]
        reached_groups = [
            trace["groups_to_95pct_oracle_HV"]
            for trace in traces
            if trace["groups_to_95pct_oracle_HV"] is not None
        ]
        median_groups = (
            float(np.median(reached_groups)) if reached_groups else None
        )
        median_samples = median_groups * len(ARMS) if median_groups is not None else None
        reports[policy] = {
            "groups_to_95pct_oracle_HV_median": median_groups,
            "samples_to_95pct_oracle_HV_median": median_samples,
            "samples_to_95pct_oracle_HV": median_samples,
            "success_rate": len(reached_groups) / len(traces) if traces else 0.0,
            "traces": traces,
        }
    return {
        "schema_version": "stage4_offline_closed_loop_replay_v2",
        "mode": "offline_replay",
        "hv_aggregation": "mean_model_internal_oracle_fraction",
        "oracle_hv": oracle_hv,
        "oracle_hv_by_model": oracle_hv_by_model,
        "threshold_hv": threshold_hv,
        "initial_group_count": int(initial_group_count),
        "budget_groups": int(budget_groups),
        "policies": reports,
    }
