#!/usr/bin/env python3
"""Evaluate Gold96 grouped sufficiency and cross-model K-shot calibration."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


TARGETS = ("ap70", "latency_ms", "energy_j")
MODEL_KINDS = ("ridge", "extra_trees")
SCHEMA = "stage35_gold96_sufficiency_transfer_locked_v2"
GRAPH_FEATURE_NAMES = (
    "conv_count", "group_conv_count", "stride2_conv_count", "kernel1_conv_count",
    "kernel3_conv_count", "group_conv_ratio", "stride2_conv_ratio", "kernel1_conv_ratio",
    "kernel3_conv_ratio", "conv_node_ratio", "log_conv_macs", "log_parameter_elements",
    "log_conv_output_elements", "log_input_elements", "log_arithmetic_intensity",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def manifest_group_split(
    train_pools: dict[str, list[str]],
    *,
    locked_groups: list[str],
    reference_groups: dict[str, str],
    train_count: int,
    seed: int,
) -> tuple[list[str], list[str]]:
    rng = np.random.default_rng(seed)
    models = sorted(train_pools)
    if train_count < len(models) or train_count > sum(len(items) for items in train_pools.values()):
        raise ValueError("train_count exceeds manifest training pool")
    allocations = {model: train_count // len(models) for model in models}
    for model in models[: train_count % len(models)]:
        allocations[model] += 1
    selected: list[str] = []
    for model in models:
        reference = reference_groups[model]
        if reference not in train_pools[model]:
            raise ValueError(f"reference group {reference} is outside the training pool")
        candidates = [group for group in train_pools[model] if group != reference]
        shuffled = list(candidates)
        rng.shuffle(shuffled)
        selected.extend([reference, *shuffled[: allocations[model] - 1]])
    if set(selected) & set(locked_groups):
        raise ValueError("locked holdout leaked into training groups")
    return selected, list(locked_groups)


def fit_affine_calibrator(prediction: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    design = np.column_stack([np.ones(len(prediction)), prediction])
    intercept, scale = np.linalg.lstsq(design, truth, rcond=None)[0]
    return float(intercept), float(scale)


def encode_targets(values: np.ndarray, models: np.ndarray, *, target: str, centers: dict[str, float]) -> np.ndarray:
    base = np.log(values) if target in {"latency_ms", "energy_j"} else values
    return np.asarray([float(value) - float(centers[str(model)]) for value, model in zip(base, models)], dtype=float)


def decode_targets(values: np.ndarray, models: np.ndarray, *, target: str, centers: dict[str, float]) -> np.ndarray:
    restored = np.asarray([float(value) + float(centers[str(model)]) for value, model in zip(values, models)], dtype=float)
    return np.exp(restored) if target in {"latency_ms", "energy_j"} else restored


def target_centers(rows: list[dict[str, Any]], target: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if finite(row.get(target)):
            value = float(row[target])
            grouped[str(row["model"])].append(math.log(value) if target != "ap70" else value)
    return {model: float(np.median(values)) for model, values in grouped.items()}


def pareto_ids(rows: Iterable[dict[str, Any]]) -> list[str]:
    items = list(rows)
    result = []
    for row in items:
        dominated = any(
            other["ap70"] >= row["ap70"]
            and other["latency_ms"] <= row["latency_ms"]
            and other["energy_j"] <= row["energy_j"]
            and (other["ap70"] > row["ap70"] or other["latency_ms"] < row["latency_ms"] or other["energy_j"] < row["energy_j"])
            for other in items if other["id"] != row["id"]
        )
        if not dominated:
            result.append(str(row["id"]))
    return sorted(result)


def pareto_precision_recall(truth: list[dict[str, Any]], predictions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    true_frontier = pareto_ids(truth)
    predicted_rows = [{"id": row["id"], **predictions[str(row["id"])]} for row in truth]
    predicted_frontier = pareto_ids(predicted_rows)
    overlap = set(true_frontier) & set(predicted_frontier)
    return {
        "true_frontier": true_frontier, "predicted_frontier": predicted_frontier,
        "recall": len(overlap) / len(true_frontier) if true_frontier else 1.0,
        "precision": len(overlap) / len(predicted_frontier) if predicted_frontier else 1.0,
    }


def load_profiles(path: Path) -> dict[str, dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["capability_profile_id"]): row for row in rows}


def load_graph_features(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["group_id"]): row for row in json.loads(path.read_text(encoding="utf-8"))}


def graph_feature_values(graph: dict[str, Any]) -> dict[str, float]:
    conv_count = max(1.0, float(graph["conv_count"]))
    node_count = max(1.0, float(graph["node_count"]))
    return {
        "conv_count": float(graph["conv_count"]),
        "group_conv_count": float(graph["group_conv_count"]),
        "stride2_conv_count": float(graph["stride2_conv_count"]),
        "kernel1_conv_count": float(graph["kernel1_conv_count"]),
        "kernel3_conv_count": float(graph["kernel3_conv_count"]),
        "group_conv_ratio": float(graph["group_conv_count"]) / conv_count,
        "stride2_conv_ratio": float(graph["stride2_conv_count"]) / conv_count,
        "kernel1_conv_ratio": float(graph["kernel1_conv_count"]) / conv_count,
        "kernel3_conv_ratio": float(graph["kernel3_conv_count"]) / conv_count,
        "conv_node_ratio": float(graph["conv_count"]) / node_count,
        "log_conv_macs": math.log1p(float(graph["conv_macs"])),
        "log_parameter_elements": math.log1p(float(graph["parameter_elements"])),
        "log_conv_output_elements": math.log1p(float(graph["conv_output_elements"])),
        "log_input_elements": math.log1p(float(graph["input_elements"])),
        "log_arithmetic_intensity": math.log1p(float(graph["arithmetic_intensity_proxy"])),
    }


def load_manifest_context(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    jobs = payload["jobs"]
    groups: dict[str, dict[str, Any]] = {}
    for job in jobs:
        groups.setdefault(
            str(job["group_id"]),
            {
                "model": str(job["model"]),
                "split": str(job["split"]),
                "width_stratum": str(job["width_stratum"]),
            },
        )
    train_pools: dict[str, list[str]] = defaultdict(list)
    locked_groups: list[str] = []
    for group_id, metadata in groups.items():
        if metadata["split"] == "locked_holdout":
            locked_groups.append(group_id)
        else:
            train_pools[metadata["model"]].append(group_id)
    pilot_groups = [str(group) for group in payload["pilot_group_ids"]]
    reference_groups = {
        model: next(group for group in pilot_groups if groups[group]["model"] == model)
        for model in sorted(train_pools)
    }
    return {
        "groups": groups,
        "train_pools": {model: sorted(values) for model, values in train_pools.items()},
        "locked_groups": sorted(locked_groups),
        "reference_groups": reference_groups,
    }


def reference_centers(
    rows: list[dict[str, Any]], reference_groups: dict[str, str], target: str
) -> dict[str, float]:
    centers: dict[str, float] = {}
    for model, group_id in reference_groups.items():
        matches = [
            row for row in rows
            if row["group_id"] == group_id
            and row["q_mode"] == "fp16"
            and row["dispatch_key"] == "tvm_auto"
            and finite(row.get(target))
        ]
        if len(matches) != 1:
            raise ValueError(f"expected one TVM-FP16 reference row for {model}/{target}, got {len(matches)}")
        value = float(matches[0][target])
        centers[model] = math.log(value) if target in {"latency_ms", "energy_j"} else value
    return centers


def feature_names(
    rows: list[dict[str, Any]], profiles: dict[str, dict[str, Any]], *, include_graph: bool
) -> list[str]:
    capabilities = sorted({name for profile in profiles.values() for name in profile.get("features", {})})
    base = ["w0", "w1", "w2", "log_product", "sum", "ratio10", "ratio21", "q_int8"]
    base += [f"align{alignment}_{stage}" for alignment in (8, 16, 32) for stage in range(3)]
    base += [f"cap:{name}" for name in capabilities]
    base += [f"cap_obs:{name}" for name in capabilities]
    base += [f"qcap:{name}" for name in capabilities]
    if include_graph:
        base += [f"graph:{name}" for name in GRAPH_FEATURE_NAMES]
        base += [f"qgraph:{name}" for name in GRAPH_FEATURE_NAMES]
    return base


def row_features(
    row: dict[str, Any], profiles: dict[str, dict[str, Any]], names: list[str],
    graph_features: dict[str, dict[str, Any]] | None,
) -> np.ndarray:
    width = [float(value) for value in row["width"]]
    profile = profiles[str(row["capability_profile_id"])]
    capability = profile.get("features", {})
    q = float(row["q_mode"] == "int8")
    values: dict[str, float] = {
        "w0": width[0], "w1": width[1], "w2": width[2],
        "log_product": math.log(max(1.0, math.prod(width))), "sum": sum(width),
        "ratio10": width[1] / width[0], "ratio21": width[2] / width[1],
        "q_int8": q,
    }
    for alignment in (8, 16, 32):
        for stage, value in enumerate(width):
            values[f"align{alignment}_{stage}"] = float(int(value) % alignment == 0)
    for name, value in capability.items():
        observed = finite(value)
        values[f"cap:{name}"] = float(value) if observed else 0.0
        values[f"cap_obs:{name}"] = float(observed)
        values[f"qcap:{name}"] = q * (float(value) if observed else 0.0)
    if graph_features is not None:
        graph = graph_feature_values(graph_features[str(row["group_id"])])
        for name, value in graph.items():
            values[f"graph:{name}"] = value
            values[f"qgraph:{name}"] = q * value
    return np.asarray([values.get(name, 0.0) for name in names], dtype=float)


def make_model(kind: str, seed: int):
    if kind == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    if kind == "extra_trees":
        return ExtraTreesRegressor(n_estimators=80, max_depth=4, min_samples_leaf=2,
                                   max_features=0.8, random_state=seed, n_jobs=1)
    raise ValueError(kind)


def metric_record(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    error = np.abs(y - pred)
    denominator = np.maximum(np.abs(y), 1e-9)
    correlation = spearmanr(y, pred).correlation if len(y) >= 3 else float("nan")
    return {"mae": float(np.mean(error)), "mape": float(np.mean(error / denominator)),
            "spearman": float(correlation) if np.isfinite(correlation) else 0.0}


def topk_recall(y: np.ndarray, pred: np.ndarray, target: str) -> float:
    count = max(1, math.ceil(len(y) * 0.25))
    direction = -1.0 if target == "ap70" else 1.0
    truth = set(np.argsort(direction * y)[:count].tolist())
    chosen = set(np.argsort(direction * pred)[:count].tolist())
    return len(truth & chosen) / count


def valid_rows(rows: list[dict[str, Any]], groups: set[str], target: str) -> list[dict[str, Any]]:
    return [row for row in rows if row["group_id"] in groups and finite(row.get(target))]


def sample_calibration_groups(
    candidates: list[str], group_metadata: dict[str, dict[str, Any]], *, count: int, seed: int
) -> set[str]:
    rng = np.random.default_rng(seed)
    buckets: dict[str, list[str]] = defaultdict(list)
    for group in candidates:
        buckets[group_metadata[group]["width_stratum"]].append(group)
    for values in buckets.values():
        rng.shuffle(values)
    ordered: list[str] = []
    while any(buckets.values()):
        for stratum in sorted(buckets):
            if buckets[stratum]:
                ordered.append(buckets[stratum].pop())
    return set(ordered[:count])


def transformed_targets(rows: list[dict[str, Any]], target: str) -> np.ndarray:
    values = np.asarray([row[target] for row in rows], dtype=float)
    return np.log(values) if target in {"latency_ms", "energy_j"} else values


def restore_transformed(values: np.ndarray, target: str) -> np.ndarray:
    return np.exp(values) if target in {"latency_ms", "energy_j"} else values


def fit_predict(train: list[dict[str, Any]], test: list[dict[str, Any]], *, target: str,
                encoding: str, kind: str, profiles: dict[str, dict[str, Any]], names: list[str], seed: int,
                graph_features: dict[str, dict[str, Any]] | None,
                centers: dict[str, float] | None = None) -> np.ndarray:
    x_train = np.vstack([row_features(row, profiles, names, graph_features) for row in train])
    x_test = np.vstack([row_features(row, profiles, names, graph_features) for row in test])
    y = np.asarray([row[target] for row in train], dtype=float)
    if encoding == "relative":
        if centers is None:
            raise ValueError("relative encoding requires fixed reference centers")
        y = encode_targets(y, np.asarray([row["model"] for row in train]), target=target, centers=centers)
    model = make_model(kind, seed)
    model.fit(x_train, y)
    pred = np.asarray(model.predict(x_test), dtype=float)
    if encoding == "relative":
        assert centers is not None
        pred = decode_targets(pred, np.asarray([row["model"] for row in test]), target=target, centers=centers)
    return pred


def grouped_learning_curve(
    rows: list[dict[str, Any]],
    profiles: dict[str, dict[str, Any]],
    names: list[str],
    context: dict[str, Any],
    graph_features: dict[str, dict[str, Any]] | None,
    repeats: int,
) -> list[dict[str, Any]]:
    output = []
    for repeat in range(repeats):
        for train_count in (6, 12, 18, 20):
            train_groups, test_groups = manifest_group_split(
                context["train_pools"],
                locked_groups=context["locked_groups"],
                reference_groups=context["reference_groups"],
                train_count=train_count,
                seed=1000 + repeat,
            )
            for target in TARGETS:
                train = valid_rows(rows, set(train_groups), target)
                test = valid_rows(rows, set(test_groups), target)
                centers = reference_centers(rows, context["reference_groups"], target)
                for encoding in ("raw", "relative"):
                    for kind in MODEL_KINDS:
                        pred = fit_predict(train, test, target=target, encoding=encoding, kind=kind,
                                           profiles=profiles, names=names, seed=repeat,
                                           graph_features=graph_features,
                                           centers=centers if encoding == "relative" else None)
                        metrics = metric_record(np.asarray([row[target] for row in test]), pred)
                        output.append({"repeat": repeat, "train_groups": train_count, "target": target,
                                       "encoding": encoding, "model_kind": kind, "test_rows": len(test),
                                       "test_groups": len(set(test_groups)),
                                       "topk_recall": topk_recall(np.asarray([row[target] for row in test]), pred, target),
                                       **metrics})
    return output


def transfer_experiment(
    rows: list[dict[str, Any]],
    profiles: dict[str, dict[str, Any]],
    names: list[str],
    context: dict[str, Any],
    graph_features: dict[str, dict[str, Any]] | None,
    repeats: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    pareto_records: list[dict[str, Any]] = []
    models = sorted({row["model"] for row in rows})
    for source, target_model in ((models[0], models[1]), (models[1], models[0])):
        source_groups = set(context["train_pools"][source])
        target_groups = list(context["train_pools"][target_model])
        target_test_groups = {
            group for group in context["locked_groups"]
            if context["groups"][group]["model"] == target_model
        }
        for repeat in range(repeats):
            for k in (2, 4, 6, 8):
                calibration_groups = sample_calibration_groups(
                    target_groups, context["groups"], count=k, seed=5000 + repeat
                )
                predictions: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
                truth_by_id: dict[str, dict[str, Any]] = {}
                for target in TARGETS:
                    source_rows = valid_rows(rows, source_groups, target)
                    calibration = valid_rows(rows, calibration_groups, target)
                    test = valid_rows(rows, target_test_groups, target)
                    y_test = np.asarray([row[target] for row in test], dtype=float)
                    for kind in MODEL_KINDS:
                        global_pred = fit_predict(source_rows, test, target=target, encoding="raw", kind=kind,
                                                  profiles=profiles, names=names, seed=repeat,
                                                  graph_features=graph_features, centers=None)
                        local_pred = fit_predict(calibration, test, target=target, encoding="raw", kind=kind,
                                                 profiles=profiles, names=names, seed=repeat,
                                                 graph_features=graph_features, centers=None)
                        source_centers = reference_centers(rows, context["reference_groups"], target)
                        x_source = np.vstack([row_features(row, profiles, names, graph_features) for row in source_rows])
                        y_source = encode_targets(np.asarray([row[target] for row in source_rows]),
                                                  np.asarray([source] * len(source_rows)), target=target,
                                                  centers=source_centers)
                        relative_model = make_model(kind, repeat)
                        relative_model.fit(x_source, y_source)
                        cal_relative = relative_model.predict(np.vstack([row_features(row, profiles, names, graph_features) for row in calibration]))
                        test_relative = relative_model.predict(
                            np.vstack([row_features(row, profiles, names, graph_features) for row in test])
                        )
                        calibration_truth = transformed_targets(calibration, target)
                        offset = float(np.median(calibration_truth - cal_relative))
                        offset_pred = restore_transformed(offset + test_relative, target)
                        intercept, scale = fit_affine_calibrator(cal_relative, calibration_truth)
                        affine_pred = restore_transformed(intercept + scale * test_relative, target)
                        methods = (
                            ("global_only", global_pred),
                            ("local_only", local_pred),
                            ("global_plus_k_offset", offset_pred),
                            ("global_plus_k_affine", affine_pred),
                        )
                        for method, pred in methods:
                            metrics = metric_record(y_test, pred)
                            records.append({"source_model": source, "target_model": target_model, "repeat": repeat,
                                            "k_groups": k, "target": target, "model_kind": kind, "method": method,
                                            "test_candidate_rows": len(test), "test_groups": len(target_test_groups),
                                            "topk_recall": topk_recall(y_test, pred, target), **metrics})
                            for row, value in zip(test, pred):
                                predictions[(kind, method)].setdefault(row["manifest_job_id"], {})[target] = float(value)
                                truth_by_id.setdefault(row["manifest_job_id"], {"id": row["manifest_job_id"]})[target] = float(row[target])
                for (kind, method), predicted in predictions.items():
                    complete_ids = sorted(set(predicted) & {key for key, value in truth_by_id.items() if all(name in value for name in TARGETS)})
                    complete_ids = [key for key in complete_ids if all(name in predicted[key] for name in TARGETS)]
                    if complete_ids:
                        result = pareto_precision_recall([truth_by_id[key] for key in complete_ids], {key: predicted[key] for key in complete_ids})
                        pareto_records.append({"source_model": source, "target_model": target_model, "repeat": repeat,
                                               "k_groups": k, "model_kind": kind, "method": method,
                                               "test_candidate_rows": len(complete_ids),
                                               "test_groups": len(target_test_groups),
                                               "pareto_recall": result["recall"],
                                               "pareto_precision": result["precision"]})
    return records, pareto_records


def summarize(records: list[dict[str, Any]], keys: list[str], metrics: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[tuple(row[key] for key in keys)].append(row)
    output = []
    for identity, rows in sorted(grouped.items(), key=lambda item: tuple(str(v) for v in item[0])):
        result = dict(zip(keys, identity)); result["n"] = len(rows)
        for metric in metrics:
            values = np.asarray([row[metric] for row in rows], dtype=float)
            result[f"{metric}_mean"] = float(np.mean(values))
            result[f"{metric}_p10"] = float(np.percentile(values, 10))
            result[f"{metric}_p90"] = float(np.percentile(values, 90))
        output.append(result)
    return output


def decision_evidence(
    learning_summary: list[dict[str, Any]], transfer_summary: list[dict[str, Any]],
    pareto_summary: list[dict[str, Any]],
) -> dict[str, Any]:
    learning: dict[str, dict[str, float]] = {}
    for target in TARGETS:
        selected = {
            int(row["train_groups"]): row
            for row in learning_summary
            if row["target"] == target and row["encoding"] == "relative"
            and row["model_kind"] == "extra_trees" and row["train_groups"] in {18, 20}
        }
        learning[target] = {
            "mae_18_groups": selected[18]["mae_mean"],
            "mae_20_groups": selected[20]["mae_mean"],
            "mae_improvement_18_to_20": (
                selected[18]["mae_mean"] - selected[20]["mae_mean"]
            ) / selected[18]["mae_mean"],
            "spearman_20_groups": selected[20]["spearman_mean"],
            "topk_recall_20_groups": selected[20]["topk_recall_mean"],
        }

    comparisons = []
    directions = sorted({(row["source_model"], row["target_model"]) for row in transfer_summary})
    for source, target_model in directions:
        for target in TARGETS:
            candidates = [
                row for row in transfer_summary
                if row["source_model"] == source and row["target_model"] == target_model
                and row["target"] == target and row["k_groups"] == 4
            ]
            best_by_family = {}
            for family, methods in {
                "global_only": {"global_only"},
                "local_only": {"local_only"},
                "global_plus_k": {"global_plus_k_offset", "global_plus_k_affine"},
            }.items():
                best_by_family[family] = min(
                    (row for row in candidates if row["method"] in methods),
                    key=lambda row: row["mae_mean"],
                )
            calibrated = best_by_family["global_plus_k"]
            comparisons.append({
                "source_model": source,
                "target_model": target_model,
                "target": target,
                "calibrated_method": calibrated["method"],
                "calibrated_mae": calibrated["mae_mean"],
                "global_only_mae": best_by_family["global_only"]["mae_mean"],
                "local_only_mae": best_by_family["local_only"]["mae_mean"],
                "beats_global_only": calibrated["mae_mean"] < best_by_family["global_only"]["mae_mean"],
                "beats_local_only": calibrated["mae_mean"] < best_by_family["local_only"]["mae_mean"],
            })
    calibrated_pareto = [
        row for row in pareto_summary
        if row["k_groups"] == 4 and row["method"] in {"global_plus_k_offset", "global_plus_k_affine"}
    ]
    return {
        "learning_curve": learning,
        "k4_transfer_comparisons": comparisons,
        "comparison_selection_note": "optimistic best MAE across two low-capacity regressors and two calibrated variants",
        "k4_calibrated_beats_global_only_count": sum(row["beats_global_only"] for row in comparisons),
        "k4_calibrated_beats_local_only_count": sum(row["beats_local_only"] for row in comparisons),
        "k4_transfer_task_count": len(comparisons),
        "k4_calibrated_pareto_recall_range": [
            min(row["pareto_recall_mean"] for row in calibrated_pareto),
            max(row["pareto_recall_mean"] for row in calibrated_pareto),
        ],
        "decision": "targeted_supplement_required_before_final_stage4_lock",
        "decision_reasons": [
            "only_two_models_cannot_establish_leave_one_model_out_generalization",
            "only_two_shared_widths_confound_model_and_shape_effects",
            "global_plus_k_does_not_consistently_outperform_local_only",
            "latency_learning_curve_still_improves_from_18_to_20_groups",
            "only_two_locked_widths_per_model_make_pareto_recall_high_variance",
        ],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def build_report(dataset: list[dict[str, Any]], learning_summary: list[dict[str, Any]], transfer_summary: list[dict[str, Any]], pareto_summary: list[dict[str, Any]]) -> str:
    models = sorted({row["model"] for row in dataset})
    overlap = set(tuple(row["width"]) for row in dataset if row["model"] == models[0]) & set(tuple(row["width"]) for row in dataset if row["model"] == models[1])
    primary_transfer = sorted(
        (row for row in transfer_summary
         if row["method"] == "global_plus_k_affine" and row["model_kind"] == "extra_trees"
         and row["k_groups"] == 4),
        key=lambda row: (row["source_model"], row["target_model"], row["target"]),
    )
    lines = ["# Stage 3.5 Gold96 sufficiency and transfer report", "", "## Dataset audit", "",
             f"- rows: {len(dataset)}; independent groups: {len({row['group_id'] for row in dataset})}",
             f"- models: {', '.join(models)}; shared widths: {len(overlap)} ({sorted(overlap)})",
             "- evaluation: 20 manifest training groups; four manifest locked-holdout groups are never used for fitting or K-shot calibration.",
             "- each transfer direction has two independent holdout width groups and eight arm candidates; candidate metrics are not eight independent samples.",
             "- conclusion: Gold96 is a seed Gold pool, not yet proof of broad cross-model sufficiency.", "",
             "## K=4 calibrated transfer snapshots", "",
             "|source|target|target metric|calibrator|model|MAPE|Spearman|top-k recall|", "|---|---|---|---|---|---:|---:|---:|"]
    for row in primary_transfer:
        lines.append(f"|{row['source_model']}|{row['target_model']}|{row['target']}|{row['method']}|{row['model_kind']}|{row['mape_mean']:.3f}|{row['spearman_mean']:.3f}|{row['topk_recall_mean']:.3f}|")
    p4 = [row for row in pareto_summary if row["method"] in {"global_plus_k_offset", "global_plus_k_affine"} and row["k_groups"] == 4]
    lines += ["", "## Decision", "", "Gold96 should not be locked for the final Stage4 model yet.", "",
              "Reasons: only 24 independent groups, only two shared widths, two-model transfer cannot establish broad generalization, and feasibility has only two failures.",
              f"K=4 calibrated Pareto recall range: {min((r['pareto_recall_mean'] for r in p4), default=0):.3f}-{max((r['pareto_recall_mean'] for r in p4), default=0):.3f}.",
              "Next data should be targeted: common-width cross-model anchors, sparse/high-disagreement regions, repeated noise anchors, and shape-failure neighborhoods.", "",
              "The raw experiment tables and split/seed bands are in the sibling CSV/JSON files."]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-json", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--capability-profiles-json", type=Path, required=True)
    parser.add_argument("--graph-features-json", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=12)
    args = parser.parse_args()
    rows = json.loads(args.gold_json.read_text(encoding="utf-8"))
    profiles = load_profiles(args.capability_profiles_json)
    graph_features = load_graph_features(args.graph_features_json) if args.graph_features_json else None
    context = load_manifest_context(args.manifest_json)
    names = feature_names(rows, profiles, include_graph=graph_features is not None)
    learning = grouped_learning_curve(rows, profiles, names, context, graph_features, args.repeats)
    transfer, pareto = transfer_experiment(rows, profiles, names, context, graph_features, args.repeats)
    learning_summary = summarize(learning, ["train_groups", "target", "encoding", "model_kind"], ["mae", "mape", "spearman", "topk_recall"])
    transfer_summary = summarize(transfer, ["source_model", "target_model", "k_groups", "target", "model_kind", "method"], ["mae", "mape", "spearman", "topk_recall"])
    pareto_summary = summarize(pareto, ["source_model", "target_model", "k_groups", "model_kind", "method"], ["pareto_recall", "pareto_precision"])
    evidence = decision_evidence(learning_summary, transfer_summary, pareto_summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "learning_curve_raw.csv", learning)
    write_csv(args.output_dir / "learning_curve_summary.csv", learning_summary)
    write_csv(args.output_dir / "transfer_raw.csv", transfer)
    write_csv(args.output_dir / "transfer_summary.csv", transfer_summary)
    write_csv(args.output_dir / "pareto_transfer_raw.csv", pareto)
    write_csv(args.output_dir / "pareto_transfer_summary.csv", pareto_summary)
    result = {"schema": SCHEMA, "gold_json": str(args.gold_json.resolve()), "gold_sha256": sha256_file(args.gold_json),
              "manifest_json": str(args.manifest_json.resolve()), "manifest_sha256": sha256_file(args.manifest_json),
              "capability_profiles_json": str(args.capability_profiles_json.resolve()),
              "capability_profiles_sha256": sha256_file(args.capability_profiles_json),
              "graph_features_json": str(args.graph_features_json.resolve()) if args.graph_features_json else None,
              "graph_features_sha256": sha256_file(args.graph_features_json) if args.graph_features_json else None,
              "runner_script_sha256": sha256_file(Path(__file__)),
              "rows": len(rows), "groups": len({row["group_id"] for row in rows}), "feature_names": names,
              "locked_holdout_groups": context["locked_groups"],
              "reference_groups": context["reference_groups"],
              "repeats": args.repeats, "learning_curve_summary": learning_summary,
              "transfer_summary": transfer_summary, "pareto_transfer_summary": pareto_summary,
              "decision_evidence": evidence,
              "decision": evidence["decision"]}
    (args.output_dir / "stage35_report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "stage35_report.md").write_text(build_report(rows, learning_summary, transfer_summary, pareto_summary), encoding="utf-8")
    print(json.dumps({"status": "success", "rows": len(rows), "groups": result["groups"], "decision": result["decision"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
