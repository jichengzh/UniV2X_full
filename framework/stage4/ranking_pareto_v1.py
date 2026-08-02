from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker
from scipy.stats import spearmanr

from framework.stage4.cost_model_selection_v1 import EncodedRows, encode_rows, grouped_folds


SCHEMA = "stage4_ranking_pareto_v1"
RANKER_SCHEMA = "stage4_ranking_pareto_ranker_v1"
TARGETS = ("latency_ms", "energy_j", "ap70")
RANKER_RETAIN_MIN_DELTA = 0.05


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _row_id(row: Mapping[str, Any], index: int) -> str:
    return str(row.get("manifest_job_id") or row.get("row_id") or index)


def _require_unique_row_ids(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    ids = [_row_id(row, index) for index, row in enumerate(rows)]
    if len(set(ids)) != len(ids):
        raise ValueError("rows must have unique row IDs")
    return ids


def _target_bounds(rows: Sequence[Mapping[str, Any]], target: str) -> tuple[float, float]:
    values = [float(row[target]) for row in rows if _finite(row.get(target))]
    if not values:
        raise ValueError(f"no finite rows for {target}")
    return min(values), max(values)


def _normalized_goodness(
    rows: Sequence[Mapping[str, Any]], reference_rows: Sequence[Mapping[str, Any]]
) -> np.ndarray:
    bounds = {target: _target_bounds(reference_rows, target) for target in TARGETS}
    values = []
    for row in rows:
        parts = []
        for target in TARGETS:
            low, high = bounds[target]
            spread = max(high - low, 1e-12)
            value = (float(row[target]) - low) / spread
            if target in {"latency_ms", "energy_j"}:
                value = 1.0 - value
            parts.append(float(np.clip(value, 0.0, 1.0)))
        values.append(float(np.mean(parts)))
    return np.asarray(values, dtype=float)


def _relevance_by_query(rows: Sequence[Mapping[str, Any]], goodness: np.ndarray) -> np.ndarray:
    labels = np.zeros(len(rows), dtype=int)
    by_model: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_model[str(row["model"])].append(index)
    for indices in by_model.values():
        ordered = sorted(indices, key=lambda item: goodness[item])
        denominator = max(len(ordered) - 1, 1)
        for rank, index in enumerate(ordered):
            labels[index] = int(round(rank / denominator * 30))
    return labels


def _sort_for_queries(indices: Sequence[int], rows: Sequence[Mapping[str, Any]]) -> tuple[list[int], list[int]]:
    by_model: dict[str, list[int]] = defaultdict(list)
    for index in indices:
        by_model[str(rows[index]["model"])].append(int(index))
    ordered_indices = []
    query = []
    for model in sorted(by_model):
        items = by_model[model]
        ordered_indices.extend(items)
        query.append(len(items))
    return ordered_indices, query


def _spearman(truth: Sequence[float], prediction: Sequence[float]) -> float:
    truth_array = np.asarray(truth, dtype=float)
    prediction_array = np.asarray(prediction, dtype=float)
    if (
        len(truth_array) < 2
        or np.allclose(truth_array, truth_array[0])
        or np.allclose(prediction_array, prediction_array[0])
    ):
        return 0.0
    value = float(spearmanr(truth_array, prediction_array).statistic)
    return value if math.isfinite(value) else 0.0


def _top_fraction_recall(
    truth: np.ndarray, prediction: np.ndarray, *, fraction: float = 0.10
) -> float:
    count = max(1, int(math.ceil(len(truth) * fraction)))
    truth_top = set(np.argsort(-truth)[:count].tolist())
    predicted_top = set(np.argsort(-prediction)[:count].tolist())
    return len(truth_top & predicted_top) / count


def _ranker(seed: int) -> LGBMRanker:
    return LGBMRanker(
        objective="lambdarank",
        n_estimators=60,
        learning_rate=0.05,
        num_leaves=7,
        max_depth=4,
        min_child_samples=1,
        reg_alpha=0.05,
        reg_lambda=0.05,
        verbosity=-1,
        random_state=seed,
        n_jobs=1,
    )


def _encoded(
    rows: Sequence[Mapping[str, Any]],
    *,
    encoded_rows: EncodedRows | None,
    graph_features: Sequence[Mapping[str, Any]] | None,
    capability_profiles: Sequence[Mapping[str, Any]] | None,
) -> EncodedRows:
    if encoded_rows is not None:
        if encoded_rows.matrix.shape[0] != len(rows):
            raise ValueError("encoded row count does not match rows")
        return encoded_rows
    if graph_features is None or capability_profiles is None:
        raise ValueError("provide encoded_rows or graph_features plus capability_profiles")
    return encode_rows(rows, graph_features, capability_profiles)


def run_ranker_oof(
    rows: Sequence[Mapping[str, Any]],
    *,
    encoded_rows: EncodedRows | None = None,
    graph_features: Sequence[Mapping[str, Any]] | None = None,
    capability_profiles: Sequence[Mapping[str, Any]] | None = None,
    outer_splits: int = 5,
    seed: int = 20260716,
) -> dict[str, Any]:
    source_rows = [dict(row) for row in rows]
    row_ids = _require_unique_row_ids(source_rows)
    encoded = _encoded(
        source_rows,
        encoded_rows=encoded_rows,
        graph_features=graph_features,
        capability_profiles=capability_profiles,
    )
    folds = grouped_folds(source_rows, n_splits=outer_splits, seed=seed)
    oof_scores: list[dict[str, Any]] = []
    score_by_index: dict[int, float] = {}
    fold_reports = []

    for fold_index, fold in enumerate(folds):
        train_indices = list(fold.train_indices)
        test_indices = list(fold.test_indices)
        train_rows = [source_rows[index] for index in train_indices]
        train_goodness = _normalized_goodness(train_rows, train_rows)
        train_labels = _relevance_by_query(train_rows, train_goodness)
        ordered_train, query = _sort_for_queries(range(len(train_rows)), train_rows)
        absolute_ordered_train = [train_indices[index] for index in ordered_train]
        model = _ranker(seed + fold_index)
        columns = [f"f{index}" for index in range(encoded.matrix.shape[1])]
        model.fit(
            pd.DataFrame(encoded.matrix[absolute_ordered_train], columns=columns),
            train_labels[ordered_train],
            group=query,
        )
        scores = np.asarray(
            model.predict(pd.DataFrame(encoded.matrix[test_indices], columns=columns)),
            dtype=float,
        )
        for row_index, score in zip(test_indices, scores):
            score_by_index[row_index] = float(score)
            oof_scores.append(
                {
                    "row_id": row_ids[row_index],
                    "group_id": str(source_rows[row_index]["group_id"]),
                    "model": str(source_rows[row_index]["model"]),
                    "outer_fold": fold_index,
                    "score": float(score),
                }
            )
        fold_reports.append(
            {
                "outer_fold": fold_index,
                "train_groups": list(fold.train_groups),
                "test_groups": list(fold.test_groups),
                "train_rows": len(train_indices),
                "test_rows": len(test_indices),
                "query_by_model": {
                    model_name: sum(str(train_rows[index]["model"]) == model_name for index in range(len(train_rows)))
                    for model_name in sorted({str(row["model"]) for row in train_rows})
                },
            }
        )

    if set(score_by_index) != set(range(len(source_rows))):
        raise AssertionError("OOF ranker did not score each row exactly once")
    truth_goodness = _normalized_goodness(source_rows, source_rows)
    ordered_scores = np.asarray([score_by_index[index] for index in range(len(source_rows))], dtype=float)
    by_model = {}
    for model_name in sorted({str(row["model"]) for row in source_rows}):
        indices = [index for index, row in enumerate(source_rows) if str(row["model"]) == model_name]
        by_model[model_name] = _top_fraction_recall(
            truth_goodness[indices],
            ordered_scores[indices],
        )
    return {
        "schema_version": RANKER_SCHEMA,
        "row_count": len(source_rows),
        "group_count": len({str(row["group_id"]) for row in source_rows}),
        "outer_splits": outer_splits,
        "seed": seed,
        "outer_folds": fold_reports,
        "oof_scores": sorted(oof_scores, key=lambda row: row["row_id"]),
        "spearman": _spearman(truth_goodness, ordered_scores),
        "top10_recall_by_model": by_model,
    }


def pareto_ids(rows: Iterable[Mapping[str, Any]]) -> list[str]:
    items = list(rows)
    frontier = []
    for row in items:
        dominated = any(
            float(other["ap70"]) >= float(row["ap70"])
            and float(other["latency_ms"]) <= float(row["latency_ms"])
            and float(other["energy_j"]) <= float(row["energy_j"])
            and (
                float(other["ap70"]) > float(row["ap70"])
                or float(other["latency_ms"]) < float(row["latency_ms"])
                or float(other["energy_j"]) < float(row["energy_j"])
            )
            for other in items
            if str(other["id"]) != str(row["id"])
        )
        if not dominated:
            frontier.append(str(row["id"]))
    return sorted(frontier)


def _hv2(points: Iterable[tuple[float, float]], ref: tuple[float, float]) -> float:
    eligible = sorted({(float(x), float(y)) for x, y in points if x < ref[0] and y < ref[1]})
    if not eligible:
        return 0.0
    xs = sorted({point[0] for point in eligible} | {ref[0]})
    area = 0.0
    for left, right in zip(xs, xs[1:]):
        active_y = [y for x, y in eligible if x <= left]
        area += (right - left) * max(0.0, ref[1] - min(active_y))
    return area


def _hv3(points: Iterable[tuple[float, float, float]], ref: tuple[float, float, float]) -> float:
    eligible = sorted({point for point in points if all(point[index] < ref[index] for index in range(3))})
    if not eligible:
        return 0.0
    xs = sorted({point[0] for point in eligible} | {ref[0]})
    volume = 0.0
    for left, right in zip(xs, xs[1:]):
        active = [(y, z) for x, y, z in eligible if x <= left]
        volume += (right - left) * _hv2(active, (ref[1], ref[2]))
    return volume


def _normalized_truth(rows: Sequence[Mapping[str, Any]]) -> dict[str, tuple[float, float, float]]:
    bounds = {target: _target_bounds(rows, target) for target in TARGETS}
    result = {}
    for row in rows:
        ap_low, ap_high = bounds["ap70"]
        lat_low, lat_high = bounds["latency_ms"]
        energy_low, energy_high = bounds["energy_j"]
        result[str(row["id"])] = (
            (ap_high - float(row["ap70"])) / max(ap_high - ap_low, 1e-12),
            (float(row["latency_ms"]) - lat_low) / max(lat_high - lat_low, 1e-12),
            (float(row["energy_j"]) - energy_low) / max(energy_high - energy_low, 1e-12),
        )
    return result


def _value_oof_by_row(cost_model_selection_report: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    targets = cost_model_selection_report.get("targets")
    if not isinstance(targets, Mapping):
        raise ValueError("cost model selection report lacks targets")
    by_row: dict[str, dict[str, float]] = defaultdict(dict)
    for target in TARGETS:
        payload = targets.get(target)
        if not isinstance(payload, Mapping) or not isinstance(payload.get("oof_predictions"), list):
            raise ValueError(f"cost model selection report lacks OOF predictions for {target}")
        seen: set[str] = set()
        for item in payload["oof_predictions"]:
            row_id = str(item.get("row_id") or "")
            if not row_id or row_id in seen:
                raise ValueError(f"duplicate or empty OOF row ID for {target}")
            seen.add(row_id)
            prediction = item.get("prediction")
            if not _finite(prediction):
                raise ValueError(f"non-finite OOF prediction for {target}/{row_id}")
            by_row[row_id][target] = float(prediction)
    return by_row


def validate_value_oof_provenance(
    rows: Sequence[Mapping[str, Any]],
    cost_model_selection_report: Mapping[str, Any],
) -> None:
    if cost_model_selection_report.get("split_protocol") != "nested_grouped_cv_by_model_width":
        raise ValueError("OOF provenance requires nested grouped CV split protocol")
    source_rows = [dict(row) for row in rows]
    row_ids = _require_unique_row_ids(source_rows)
    row_by_id = dict(zip(row_ids, source_rows))
    expected_groups = {str(row["group_id"]) for row in source_rows}
    targets = cost_model_selection_report.get("targets")
    if not isinstance(targets, Mapping):
        raise ValueError("OOF provenance requires target reports")
    reference_group_folds: dict[str, int] | None = None

    for target in TARGETS:
        payload = targets.get(target)
        if not isinstance(payload, Mapping):
            raise ValueError(f"OOF provenance missing target report for {target}")
        folds = payload.get("outer_folds")
        predictions = payload.get("oof_predictions")
        if not isinstance(folds, list) or len(folds) < 2 or not isinstance(predictions, list):
            raise ValueError(f"OOF provenance missing folds or predictions for {target}")
        fold_contract: dict[int, tuple[set[str], set[str]]] = {}
        test_occurrences = {group_id: 0 for group_id in expected_groups}
        for fold in folds:
            fold_index = int(fold.get("outer_fold"))
            if fold_index in fold_contract:
                raise ValueError(f"OOF provenance has duplicate fold {fold_index} for {target}")
            train_groups = {str(group) for group in fold.get("train_groups", [])}
            test_groups = {str(group) for group in fold.get("test_groups", [])}
            if not train_groups or not test_groups or train_groups & test_groups:
                raise ValueError(f"OOF provenance has invalid group split for {target}/{fold_index}")
            fold_contract[fold_index] = (train_groups, test_groups)
            for group_id in expected_groups & test_groups:
                test_occurrences[group_id] += 1
        if any(count != 1 for count in test_occurrences.values()):
            raise ValueError(f"OOF provenance does not test each selected group once for {target}")

        selected_predictions = [
            item for item in predictions if str(item.get("row_id") or "") in row_by_id
        ]
        if len(selected_predictions) != len(row_by_id):
            raise ValueError(f"OOF provenance does not predict each selected row once for {target}")
        group_folds: dict[str, int] = {}
        seen_rows: set[str] = set()
        for item in selected_predictions:
            row_id = str(item.get("row_id") or "")
            if row_id in seen_rows:
                raise ValueError(f"OOF provenance duplicates row {row_id} for {target}")
            seen_rows.add(row_id)
            row = row_by_id[row_id]
            group_id = str(item.get("group_id") or "")
            fold_index = int(item.get("outer_fold"))
            if group_id != str(row["group_id"]) or fold_index not in fold_contract:
                raise ValueError(f"OOF provenance row identity mismatch for {target}/{row_id}")
            train_groups, test_groups = fold_contract[fold_index]
            if group_id not in test_groups or group_id in train_groups:
                raise ValueError(f"OOF provenance row is not held out for {target}/{row_id}")
            if not _finite(item.get("truth")) or not math.isclose(
                float(item["truth"]), float(row[target]), rel_tol=1e-9, abs_tol=1e-12
            ):
                raise ValueError(f"OOF provenance truth mismatch for {target}/{row_id}")
            previous = group_folds.setdefault(group_id, fold_index)
            if previous != fold_index:
                raise ValueError(f"OOF provenance splits group rows for {target}/{group_id}")
        if reference_group_folds is None:
            reference_group_folds = group_folds
        elif group_folds != reference_group_folds:
            raise ValueError("OOF provenance uses inconsistent folds across targets")


def _pareto_metrics(rows: Sequence[Mapping[str, Any]], predictions: dict[str, dict[str, float]]) -> dict[str, Any]:
    truth_rows = [
        {
            "id": _row_id(row, index),
            "latency_ms": float(row["latency_ms"]),
            "energy_j": float(row["energy_j"]),
            "ap70": float(row["ap70"]),
        }
        for index, row in enumerate(rows)
    ]
    predicted_rows = [
        {"id": truth["id"], **predictions[truth["id"]]}
        for truth in truth_rows
    ]
    true_frontier = pareto_ids(truth_rows)
    predicted_frontier = pareto_ids(predicted_rows)
    overlap = set(true_frontier) & set(predicted_frontier)
    normalized = _normalized_truth(truth_rows)
    reference = (1.05, 1.05, 1.05)
    oracle_hv = _hv3(normalized.values(), reference)
    selected_hv = _hv3((normalized[row_id] for row_id in predicted_frontier), reference)
    return {
        "rows": len(truth_rows),
        "true_frontier": true_frontier,
        "predicted_frontier": predicted_frontier,
        "pareto_recall": len(overlap) / len(true_frontier) if true_frontier else 1.0,
        "pareto_precision": len(overlap) / len(predicted_frontier) if predicted_frontier else 1.0,
        "oracle_hv": oracle_hv,
        "selected_true_hv": selected_hv,
        "hv_regret": max(0.0, oracle_hv - selected_hv) / max(oracle_hv, 1e-12),
    }


def evaluate_pareto_from_value_oof(
    rows: Sequence[Mapping[str, Any]],
    cost_model_selection_report: Mapping[str, Any],
    *,
    ranker_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    source_rows = [dict(row) for row in rows]
    row_ids = _require_unique_row_ids(source_rows)
    validate_value_oof_provenance(source_rows, cost_model_selection_report)
    predictions = _value_oof_by_row(cost_model_selection_report)
    missing = set(row_ids) - set(predictions)
    extra = set(predictions) - set(row_ids)
    incomplete = {row_id for row_id in row_ids if set(predictions[row_id]) != set(TARGETS)}
    if missing or extra or incomplete:
        raise ValueError(
            "value OOF predictions must match rows for all targets: "
            f"missing={sorted(missing)} extra={sorted(extra)} incomplete={sorted(incomplete)}"
        )
    by_model = {}
    for model_name in sorted({str(row["model"]) for row in source_rows}):
        selected = [row for row in source_rows if str(row["model"]) == model_name]
        by_model[model_name] = _pareto_metrics(selected, predictions)
    summary = {
        "pareto_recall_mean": float(np.mean([item["pareto_recall"] for item in by_model.values()])),
        "pareto_precision_mean": float(np.mean([item["pareto_precision"] for item in by_model.values()])),
        "hv_regret_mean": float(np.mean([item["hv_regret"] for item in by_model.values()])),
        "hv_regret_max": float(np.max([item["hv_regret"] for item in by_model.values()])),
    }
    return {
        "schema_version": "stage4_ranking_pareto_value_oof_v1",
        "row_count": len(source_rows),
        "ranker_score_note": "ranker_report is accepted only for traceability; value OOF coordinates define predicted Pareto/HV",
        "ranker_rows": len(ranker_report.get("oof_scores", [])) if ranker_report else None,
        "by_model": by_model,
        "summary": summary,
    }


def build_stage4_ranking_pareto_report(
    rows: Sequence[Mapping[str, Any]],
    *,
    cost_model_selection_report: Mapping[str, Any],
    encoded_rows: EncodedRows | None = None,
    graph_features: Sequence[Mapping[str, Any]] | None = None,
    capability_profiles: Sequence[Mapping[str, Any]] | None = None,
    outer_splits: int = 5,
    seed: int = 20260716,
) -> dict[str, Any]:
    ranker = run_ranker_oof(
        rows,
        encoded_rows=encoded_rows,
        graph_features=graph_features,
        capability_profiles=capability_profiles,
        outer_splits=outer_splits,
        seed=seed,
    )
    pareto = evaluate_pareto_from_value_oof(
        rows,
        cost_model_selection_report,
        ranker_report=ranker,
    )
    source_rows = [dict(row) for row in rows]
    predictions = _value_oof_by_row(cost_model_selection_report)
    predicted_rows = [
        {**row, **predictions[_row_id(row, index)]}
        for index, row in enumerate(source_rows)
    ]
    truth_goodness = _normalized_goodness(source_rows, source_rows)
    value_goodness = _normalized_goodness(predicted_rows, source_rows)
    value_recall_by_model = {}
    for model_name in sorted({str(row["model"]) for row in source_rows}):
        indices = [
            index
            for index, row in enumerate(source_rows)
            if str(row["model"]) == model_name
        ]
        value_recall_by_model[model_name] = _top_fraction_recall(
            truth_goodness[indices],
            value_goodness[indices],
        )
    ranker_delta_by_model = {
        model_name: float(ranker["top10_recall_by_model"][model_name])
        - float(value_recall_by_model[model_name])
        for model_name in value_recall_by_model
    }
    mean_delta = float(np.mean(list(ranker_delta_by_model.values())))
    ranker_value_comparison = {
        "value_top10_recall_by_model": value_recall_by_model,
        "ranker_top10_recall_by_model": dict(ranker["top10_recall_by_model"]),
        "ranker_delta_by_model": ranker_delta_by_model,
        "mean_ranker_delta": mean_delta,
        "retain_threshold": RANKER_RETAIN_MIN_DELTA,
        "retain_ranker": mean_delta >= RANKER_RETAIN_MIN_DELTA,
    }
    return {
        "schema_version": SCHEMA,
        "row_count": len(rows),
        "group_count": len({str(row["group_id"]) for row in rows}),
        "split_protocol": "outer_group_folds_no_group_leakage_query_by_model",
        "generalization_scope": "within_model_grouped_interpolation_not_model_holdout",
        "ranker": ranker,
        "pareto": pareto,
        "ranker_value_comparison": ranker_value_comparison,
    }


__all__ = [
    "build_stage4_ranking_pareto_report",
    "evaluate_pareto_from_value_oof",
    "pareto_ids",
    "run_ranker_oof",
    "validate_value_oof_provenance",
]
