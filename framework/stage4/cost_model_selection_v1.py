from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold


TARGETS = ("latency_ms", "energy_j", "ap70")
DEFAULT_CANDIDATES = (
    "extra_trees_raw",
    "extra_trees_log",
    "lgbm_l1_raw",
    "lgbm_l1_log",
    "lgbm_huber_raw",
    "lgbm_huber_log",
    "lgbm_quantile_raw",
    "lgbm_quantile_log",
    "extra_trees_residual",
    "lgbm_l1_residual",
    "lgbm_huber_residual",
    "lgbm_quantile_residual",
)
LABEL_LIKE_CONTEXT_TOKENS = ("latency", "energy", "ap30", "ap50", "ap70")
LABEL_LIKE_CONTEXT_FIELDS = {
    "terminal_status",
    "failure_reason",
    "measurement_status",
    "performance_status",
}


@dataclass(frozen=True)
class GroupFold:
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    train_groups: tuple[str, ...]
    test_groups: tuple[str, ...]


@dataclass(frozen=True)
class EncodedRows:
    matrix: np.ndarray
    feature_names: tuple[str, ...]


@dataclass(frozen=True)
class QuantileInterval:
    lower: np.ndarray
    median: np.ndarray
    upper: np.ndarray


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def grouped_folds(
    rows: Sequence[Mapping[str, Any]], *, n_splits: int, seed: int
) -> tuple[GroupFold, ...]:
    groups = np.asarray([str(row["group_id"]) for row in rows], dtype=object)
    unique_groups = set(groups.tolist())
    if n_splits < 2 or n_splits > len(unique_groups):
        raise ValueError("n_splits must be between 2 and the number of groups")
    splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    dummy = np.zeros(len(rows), dtype=float)
    folds = []
    for train, test in splitter.split(dummy, groups=groups):
        train_groups = tuple(sorted(set(groups[train].tolist())))
        test_groups = tuple(sorted(set(groups[test].tolist())))
        if set(train_groups) & set(test_groups):
            raise AssertionError("group leakage in grouped fold")
        folds.append(
            GroupFold(
                train_indices=tuple(int(index) for index in train),
                test_indices=tuple(int(index) for index in test),
                train_groups=train_groups,
                test_groups=test_groups,
            )
        )
    return tuple(folds)


def _numeric_feature_names(records: Sequence[Mapping[str, Any]], excluded: set[str]) -> list[str]:
    return sorted(
        {
            str(name)
            for record in records
            for name, value in record.items()
            if name not in excluded and _finite(value)
        }
    )


def _reject_label_like_context(records: Sequence[Mapping[str, Any]], context: str) -> None:
    fields = {
        str(name).lower()
        for record in records
        for name in record
    }
    rejected = sorted(
        name
        for name in fields
        if any(token in name for token in LABEL_LIKE_CONTEXT_TOKENS)
        or name in LABEL_LIKE_CONTEXT_FIELDS
        or name.startswith(("target_", "observed_", "measured_"))
    )
    if rejected:
        raise ValueError(f"label-like fields are forbidden in {context} context: {rejected}")


def encode_rows(
    rows: Sequence[Mapping[str, Any]],
    graph_features: Sequence[Mapping[str, Any]],
    capability_profiles: Sequence[Mapping[str, Any]],
) -> EncodedRows:
    graph_by_group = {str(item["group_id"]): item for item in graph_features}
    profile_by_id = {
        str(item["capability_profile_id"]): item for item in capability_profiles
    }
    missing_graph = {str(row["group_id"]) for row in rows} - set(graph_by_group)
    missing_profile = {
        str(row["capability_profile_id"]) for row in rows
    } - set(profile_by_id)
    if missing_graph or missing_profile:
        raise ValueError(
            f"missing feature context: graph={sorted(missing_graph)}, profile={sorted(missing_profile)}"
        )

    graph_names = _numeric_feature_names(
        list(graph_by_group.values()),
        {"width", "input_dims"},
    )
    capability_records = [item.get("features") or {} for item in profile_by_id.values()]
    _reject_label_like_context(list(graph_by_group.values()), "graph feature")
    _reject_label_like_context(capability_records, "capability profile")
    capability_names = _numeric_feature_names(capability_records, set())
    feature_names = (
        "width:axis0",
        "width:axis1",
        "width:axis2",
        "width:axis3",
        "width:axis4",
        "width:log_product",
        "q:int8",
        "model:codriving",
        "model:fcooper",
        *(f"graph:{name}" for name in graph_names),
        *(f"cap:{name}" for name in capability_names),
        *(f"cap_x_q:{name}" for name in capability_names),
    )

    encoded = []
    for row in rows:
        width = [float(value) for value in row["width"]]
        if not 1 <= len(width) <= 5:
            raise ValueError("structure width vector must contain one to five axes")
        padded_width = [*width, *([0.0] * (5 - len(width)))]
        q_int8 = float(str(row["q_mode"]).lower() == "int8")
        graph = graph_by_group[str(row["group_id"])]
        capability = profile_by_id[str(row["capability_profile_id"])].get("features") or {}
        vector = [
            *padded_width,
            float(np.log1p(np.prod(width))),
            q_int8,
            float(str(row["model"]).lower() == "codriving"),
            float(str(row["model"]).lower() == "fcooper"),
            *(float(graph.get(name, 0.0)) if _finite(graph.get(name)) else 0.0 for name in graph_names),
            *(
                float(capability.get(name, 0.0))
                if _finite(capability.get(name))
                else 0.0
                for name in capability_names
            ),
            *(
                q_int8 * float(capability.get(name, 0.0))
                if _finite(capability.get(name))
                else 0.0
                for name in capability_names
            ),
        ]
        encoded.append(vector)
    return EncodedRows(
        matrix=np.asarray(encoded, dtype=float),
        feature_names=tuple(feature_names),
    )


def _candidate_names(target: str, requested: Sequence[str]) -> tuple[str, ...]:
    names = []
    for name in requested:
        if target == "ap70" and name.endswith("_log"):
            continue
        if target != "ap70" and name.endswith("_residual"):
            continue
        names.append(name)
    if not names:
        raise ValueError(f"no candidates apply to {target}")
    return tuple(names)


def _build_regressor(name: str, seed: int, *, quantile: float | None = None):
    if name.startswith("extra_trees"):
        if quantile is not None:
            raise ValueError("ExtraTrees does not provide a quantile objective")
        return ExtraTreesRegressor(
            n_estimators=160,
            max_depth=8,
            min_samples_leaf=2,
            max_features=0.8,
            random_state=seed,
            n_jobs=1,
        )
    if name.startswith("lgbm"):
        if quantile is not None or name.startswith("lgbm_quantile"):
            objective = "quantile"
            kwargs = {"alpha": 0.5 if quantile is None else quantile}
        elif name.startswith("lgbm_huber"):
            objective = "huber"
            kwargs = {}
        else:
            objective = "mae"
            kwargs = {}
        return LGBMRegressor(
            objective=objective,
            n_estimators=120,
            learning_rate=0.04,
            num_leaves=7,
            max_depth=4,
            min_child_samples=4,
            reg_alpha=0.1,
            reg_lambda=0.1,
            verbosity=-1,
            random_state=seed,
            n_jobs=1,
            **kwargs,
        )
    raise ValueError(f"unknown candidate: {name}")


def _candidate_spec(name: str) -> dict[str, str]:
    if name.startswith("extra_trees"):
        objective = "squared_error"
        family = "extra_trees"
    elif name.startswith("lgbm_l1"):
        objective = "mae"
        family = "lightgbm"
    elif name.startswith("lgbm_huber"):
        objective = "huber"
        family = "lightgbm"
    elif name.startswith("lgbm_quantile"):
        objective = "quantile_p50"
        family = "lightgbm"
    else:
        raise ValueError(f"unknown candidate: {name}")
    encoding = (
        "model_anchor_residual"
        if name.endswith("_residual")
        else _transform_for(name)
    )
    return {"family": family, "objective": objective, "target_encoding": encoding}


def _transform_for(name: str) -> str:
    return "log1p" if name.endswith("_log") else "raw"


def _transform_target(values: np.ndarray, transform: str) -> np.ndarray:
    if transform == "raw":
        return values
    if transform == "log1p":
        if np.any(values < 0):
            raise ValueError("log1p targets must be non-negative")
        return np.log1p(values)
    raise ValueError(transform)


def _inverse_target(values: np.ndarray, transform: str) -> np.ndarray:
    if transform == "raw":
        return values
    if transform == "log1p":
        return np.maximum(0.0, np.expm1(values))
    raise ValueError(transform)


def _model_anchors(rows: Sequence[Mapping[str, Any]], y: np.ndarray) -> dict[str, float]:
    models = sorted({str(row["model"]) for row in rows})
    return {
        model: float(np.median([value for row, value in zip(rows, y) if row["model"] == model]))
        for model in models
    }


def _fit_predict_candidate(
    name: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    train_rows: Sequence[Mapping[str, Any]],
    test_rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
) -> np.ndarray:
    residual = name.endswith("_residual")
    anchors = _model_anchors(train_rows, train_y) if residual else {}
    fit_y = np.asarray(
        [value - anchors[str(row["model"])] for row, value in zip(train_rows, train_y)],
        dtype=float,
    ) if residual else train_y
    transform = _transform_for(name)
    model = _build_regressor(name, seed)
    columns = [f"f{index}" for index in range(train_x.shape[1])]
    model.fit(pd.DataFrame(train_x, columns=columns), _transform_target(fit_y, transform))
    prediction = _inverse_target(
        np.asarray(model.predict(pd.DataFrame(test_x, columns=columns)), dtype=float),
        transform,
    )
    if residual:
        fallback = float(np.median(train_y))
        prediction = prediction + np.asarray(
            [anchors.get(str(row["model"]), fallback) for row in test_rows], dtype=float
        )
    return prediction


def _spearman(truth: np.ndarray, prediction: np.ndarray) -> float:
    if len(truth) < 2 or np.allclose(truth, truth[0]) or np.allclose(prediction, prediction[0]):
        return 0.0
    value = float(spearmanr(truth, prediction).statistic)
    return value if math.isfinite(value) else 0.0


def _metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = np.abs(truth - prediction)
    denominator = np.maximum(np.abs(truth), 1e-9)
    return {
        "mae": float(np.mean(error)),
        "mape": float(np.mean(error / denominator)),
        "spearman": _spearman(truth, prediction),
    }


def _selection_score(truth: np.ndarray, prediction: np.ndarray) -> float:
    spread = max(float(np.quantile(truth, 0.9) - np.quantile(truth, 0.1)), 1e-9)
    metrics = _metrics(truth, prediction)
    return float(metrics["mae"] / spread + 0.25 * (1.0 - metrics["spearman"]))


def _subset_indices(indices: Sequence[int], rows: Sequence[Mapping[str, Any]], target: str) -> list[int]:
    return [index for index in indices if _finite(rows[index].get(target))]


def run_nested_selection(
    rows: Sequence[Mapping[str, Any]],
    graph_features: Sequence[Mapping[str, Any]],
    capability_profiles: Sequence[Mapping[str, Any]],
    *,
    outer_splits: int = 5,
    inner_splits: int = 3,
    seed: int = 20260716,
    candidates: Sequence[str] = DEFAULT_CANDIDATES,
) -> dict[str, Any]:
    source_rows = [dict(row) for row in rows]
    encoded = encode_rows(source_rows, graph_features, capability_profiles)
    outer_folds = grouped_folds(source_rows, n_splits=outer_splits, seed=seed)
    target_reports: dict[str, Any] = {}

    for target in TARGETS:
        candidate_names = _candidate_names(target, candidates)
        fold_reports = []
        all_truth: list[float] = []
        all_prediction: list[float] = []
        oof_predictions: list[dict[str, Any]] = []
        for outer_index, outer in enumerate(outer_folds):
            train_indices = _subset_indices(outer.train_indices, source_rows, target)
            test_indices = _subset_indices(outer.test_indices, source_rows, target)
            if not train_indices or not test_indices:
                raise ValueError(f"empty measured fold for {target}")
            inner_rows = [source_rows[index] for index in train_indices]
            inner = grouped_folds(
                inner_rows,
                n_splits=min(inner_splits, len({row["group_id"] for row in inner_rows})),
                seed=seed + outer_index + 1,
            )
            candidate_scores: dict[str, float] = {}
            for candidate_index, name in enumerate(candidate_names):
                scores = []
                for inner_index, fold in enumerate(inner):
                    fit_indices = [train_indices[index] for index in fold.train_indices]
                    validation_indices = [train_indices[index] for index in fold.test_indices]
                    fit_rows = [source_rows[index] for index in fit_indices]
                    validation_rows = [source_rows[index] for index in validation_indices]
                    fit_y = np.asarray([source_rows[index][target] for index in fit_indices], dtype=float)
                    validation_y = np.asarray(
                        [source_rows[index][target] for index in validation_indices], dtype=float
                    )
                    prediction = _fit_predict_candidate(
                        name,
                        encoded.matrix[fit_indices],
                        fit_y,
                        encoded.matrix[validation_indices],
                        fit_rows,
                        validation_rows,
                        seed=seed + 100 * outer_index + 10 * candidate_index + inner_index,
                    )
                    scores.append(_selection_score(validation_y, prediction))
                candidate_scores[name] = float(np.mean(scores))
            selected = min(candidate_scores, key=candidate_scores.get)
            train_y = np.asarray([source_rows[index][target] for index in train_indices], dtype=float)
            test_y = np.asarray([source_rows[index][target] for index in test_indices], dtype=float)
            prediction = _fit_predict_candidate(
                selected,
                encoded.matrix[train_indices],
                train_y,
                encoded.matrix[test_indices],
                [source_rows[index] for index in train_indices],
                [source_rows[index] for index in test_indices],
                seed=seed + 1000 + outer_index,
            )
            all_truth.extend(test_y.tolist())
            all_prediction.extend(prediction.tolist())
            oof_predictions.extend(
                {
                    "row_id": str(
                        source_rows[index].get("manifest_job_id")
                        or source_rows[index].get("row_id")
                        or index
                    ),
                    "group_id": str(source_rows[index]["group_id"]),
                    "model": str(source_rows[index]["model"]),
                    "outer_fold": outer_index,
                    "selected_candidate": selected,
                    "truth": float(source_rows[index][target]),
                    "prediction": float(value),
                }
                for index, value in zip(test_indices, prediction)
            )
            fold_reports.append(
                {
                    "outer_fold": outer_index,
                    "train_groups": list(outer.train_groups),
                    "test_groups": list(outer.test_groups),
                    "train_rows": len(train_indices),
                    "test_rows": len(test_indices),
                    "selected_candidate": selected,
                    "inner_candidate_scores": candidate_scores,
                    "metrics": _metrics(test_y, prediction),
                }
            )
        truth = np.asarray(all_truth, dtype=float)
        prediction = np.asarray(all_prediction, dtype=float)
        target_reports[target] = {
            "outer_folds": fold_reports,
            "oof_predictions": sorted(oof_predictions, key=lambda row: row["row_id"]),
            "summary": _metrics(truth, prediction),
            "selected_candidate_counts": {
                name: sum(fold["selected_candidate"] == name for fold in fold_reports)
                for name in candidate_names
            },
        }
    return {
        "schema_version": "stage4_cost_model_selection_v1",
        "split_protocol": "nested_grouped_cv_by_model_width",
        "outer_splits": outer_splits,
        "inner_splits": inner_splits,
        "seed": seed,
        "row_count": len(source_rows),
        "group_count": len({str(row["group_id"]) for row in source_rows}),
        "feature_names": list(encoded.feature_names),
        "candidate_specs": {name: _candidate_spec(name) for name in candidates},
        "targets": target_reports,
    }


def fit_predict_quantile_interval(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    *,
    transform: str,
    seed: int,
) -> QuantileInterval:
    predictions = []
    transformed = _transform_target(np.asarray(train_y, dtype=float), transform)
    columns = [f"f{index}" for index in range(train_x.shape[1])]
    for quantile in (0.1, 0.5, 0.9):
        model = _build_regressor("lgbm_quantile", seed, quantile=quantile)
        model.fit(pd.DataFrame(train_x, columns=columns), transformed)
        predictions.append(
            _inverse_target(
                np.asarray(
                    model.predict(pd.DataFrame(test_x, columns=columns)), dtype=float
                ),
                transform,
            )
        )
    stacked = np.vstack(predictions)
    ordered = np.sort(stacked, axis=0)
    return QuantileInterval(lower=ordered[0], median=ordered[1], upper=ordered[2])


__all__ = [
    "EncodedRows",
    "GroupFold",
    "QuantileInterval",
    "encode_rows",
    "fit_predict_quantile_interval",
    "grouped_folds",
    "run_nested_selection",
]
