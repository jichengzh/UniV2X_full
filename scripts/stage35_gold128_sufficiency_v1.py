#!/usr/bin/env python3
"""Evaluate Gold128 grouped sufficiency, transfer, Pareto quality, and uncertainty."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import stage35_gold96_sufficiency_transfer_v1 as base


SCHEMA = "stage35_gold128_sufficiency_v1"
LEARNING_SIZES = (6, 12, 18, 24, 26)
K_GROUPS = (2, 4, 6, 8)
TARGETS = base.TARGETS
MODEL_KINDS = base.MODEL_KINDS
PRIMARY_MODEL_KIND = "extra_trees"
PRIMARY_ENCODING = "relative"
PRIMARY_TRANSFER_METHOD = "global_plus_k_residual"
PRIMARY_K = 4
EXPECTED_ARMS = {
    ("tvm_auto", "fp16"),
    ("tvm_auto", "int8"),
    ("trt_engine", "fp16"),
    ("trt_engine", "int8"),
}


def hypervolume_2d(points: Iterable[tuple[float, float]], ref: tuple[float, float]) -> float:
    eligible = sorted({(float(x), float(y)) for x, y in points if x < ref[0] and y < ref[1]})
    if not eligible:
        return 0.0
    xs = sorted({point[0] for point in eligible} | {ref[0]})
    area = 0.0
    for left, right in zip(xs, xs[1:]):
        active_y = [y for x, y in eligible if x <= left]
        area += (right - left) * max(0.0, ref[1] - min(active_y))
    return area


def hypervolume_3d(points: Iterable[tuple[float, float, float]], ref: tuple[float, float, float]) -> float:
    eligible = sorted({tuple(float(value) for value in point) for point in points if all(point[i] < ref[i] for i in range(3))})
    if not eligible:
        return 0.0
    xs = sorted({point[0] for point in eligible} | {ref[0]})
    volume = 0.0
    for left, right in zip(xs, xs[1:]):
        active = [(y, z) for x, y, z in eligible if x <= left]
        volume += (right - left) * hypervolume_2d(active, (ref[1], ref[2]))
    return volume


def _normalized_truth(rows: list[dict[str, Any]]) -> dict[str, tuple[float, float, float]]:
    values = {
        target: np.asarray([float(row[target]) for row in rows], dtype=float)
        for target in TARGETS
    }
    bounds = {
        target: (float(np.min(items)), float(np.max(items)))
        for target, items in values.items()
    }
    output = {}
    for row in rows:
        ap_min, ap_max = bounds["ap70"]
        lat_min, lat_max = bounds["latency_ms"]
        energy_min, energy_max = bounds["energy_j"]
        output[str(row["id"])] = (
            (ap_max - float(row["ap70"])) / max(ap_max - ap_min, 1e-12),
            (float(row["latency_ms"]) - lat_min) / max(lat_max - lat_min, 1e-12),
            (float(row["energy_j"]) - energy_min) / max(energy_max - energy_min, 1e-12),
        )
    return output


def selection_quality(truth: list[dict[str, Any]], predictions: dict[str, dict[str, float]]) -> dict[str, float]:
    pareto = base.pareto_precision_recall(truth, predictions)
    normalized = _normalized_truth(truth)
    reference = (1.05, 1.05, 1.05)
    oracle_hv = hypervolume_3d(normalized.values(), reference)
    selected_ids = pareto["predicted_frontier"]
    selected_hv = hypervolume_3d((normalized[item] for item in selected_ids), reference)
    return {
        "pareto_recall": float(pareto["recall"]),
        "pareto_precision": float(pareto["precision"]),
        "oracle_hv": oracle_hv,
        "selected_true_hv": selected_hv,
        "hv_regret": max(0.0, oracle_hv - selected_hv) / max(oracle_hv, 1e-12),
    }


def interval_coverage(records: list[dict[str, Any]]) -> dict[str, float]:
    if not records:
        raise ValueError("prediction interval records are empty")
    covered_by_group: dict[str, list[bool]] = defaultdict(list)
    widths = []
    for row in records:
        covered = float(row["lower"]) <= float(row["truth"]) <= float(row["upper"])
        covered_by_group[str(row["group_id"])].append(covered)
        widths.append(float(row["upper"]) - float(row["lower"]))
    covered_count = sum(sum(items) for items in covered_by_group.values())
    total = sum(len(items) for items in covered_by_group.values())
    return {
        "coverage_90": covered_count / total,
        "fully_covered_group_rate_90": sum(all(items) for items in covered_by_group.values()) / len(covered_by_group),
        "mean_interval_width": float(np.mean(widths)),
        "rows": total,
        "groups": len(covered_by_group),
    }


def conformalize_intervals(
    calibration: Sequence[Mapping[str, Any]],
    test: Sequence[Mapping[str, Any]],
    *,
    coverage: float,
) -> list[dict[str, Any]]:
    if not 0.0 < coverage < 1.0:
        raise ValueError("coverage must be between zero and one")
    scores_by_group: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in calibration:
        truth = float(row["truth"])
        score = max(float(row["lower"]) - truth, truth - float(row["upper"]), 0.0)
        group_id = str(row.get("group_id") or "")
        if not group_id:
            raise ValueError("conformal calibration rows require group_id")
        scores_by_group[(str(row["model"]), group_id)].append(score)
    scores_by_model: dict[str, list[float]] = defaultdict(list)
    for (model, _), scores in scores_by_group.items():
        scores_by_model[model].append(max(scores))
    corrections: dict[str, float] = {}
    for model, scores in scores_by_model.items():
        ordered = sorted(scores)
        rank = min(len(ordered), math.ceil((len(ordered) + 1) * coverage))
        corrections[model] = float(ordered[rank - 1])
    output = []
    for source in test:
        model = str(source["model"])
        if model not in corrections:
            raise ValueError(f"missing conformal calibration rows for {model}")
        correction = corrections[model]
        output.append({
            **dict(source),
            "lower": float(source["lower"]) - correction,
            "upper": float(source["upper"]) + correction,
            "conformal_correction": correction,
        })
    return output


def residual_calibration_prediction(
    calibration_features: np.ndarray,
    calibration_residual: np.ndarray,
    test_features: np.ndarray,
    base_prediction: np.ndarray,
    *,
    seed: int = 0,
) -> np.ndarray:
    model = base.make_model("ridge", seed)
    model.fit(calibration_features, calibration_residual)
    predicted_residual = np.asarray(model.predict(test_features), dtype=float)
    observed = np.asarray(calibration_residual, dtype=float)
    bounded_residual = np.clip(predicted_residual, np.min(observed), np.max(observed))
    return np.asarray(base_prediction, dtype=float) + bounded_residual


def validate_dataset_contract(
    rows: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]
) -> None:
    jobs = manifest.get("jobs")
    if len(rows) != 128 or not isinstance(jobs, list) or len(jobs) != 128:
        raise ValueError("Gold128 requires exactly 128 result rows and manifest jobs")
    row_ids = [str(row.get("manifest_job_id")) for row in rows]
    job_ids = [str(job.get("job_id")) for job in jobs]
    if len(set(row_ids)) != 128 or len(set(job_ids)) != 128:
        raise ValueError("Gold128 result and manifest job IDs must be unique")
    if set(row_ids) != set(job_ids):
        raise ValueError("Gold128 result and manifest job IDs must match")
    jobs_by_id = {str(job["job_id"]): job for job in jobs}
    binding_fields = (
        "group_id", "model", "width", "q_mode", "dispatch_key", "capability_profile_id"
    )
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        job = jobs_by_id[str(row["manifest_job_id"])]
        mismatches = [field for field in binding_fields if row.get(field) != job.get(field)]
        if row.get("split") != job.get("split"):
            mismatches.append("split")
        if mismatches:
            raise ValueError(
                f"Gold128 row/manifest binding mismatch for {row['manifest_job_id']}: {mismatches}"
            )
        status = row.get("terminal_status")
        if status == "measured_success_gold":
            if not all(base.finite(row.get(target)) for target in TARGETS):
                raise ValueError(f"measured Gold row lacks targets: {row['manifest_job_id']}")
        elif status != "feasibility_failure":
            raise ValueError(f"Gold128 row is not terminal: {row['manifest_job_id']}")
        if job.get("split") == "locked_holdout" and status != "measured_success_gold":
            raise ValueError(f"locked holdout must be measured: {row['manifest_job_id']}")
        grouped[str(row["group_id"])].append(row)
    if len(grouped) != 32:
        raise ValueError("Gold128 requires exactly 32 groups")
    for group_id, group_rows in grouped.items():
        arms = {(str(row["dispatch_key"]), str(row["q_mode"])) for row in group_rows}
        if len(group_rows) != 4 or arms != EXPECTED_ARMS:
            raise ValueError(f"Gold128 group is not a complete four-arm group: {group_id}")


def _same_metric(first: Any, second: Any) -> bool:
    return base.finite(first) and base.finite(second) and math.isclose(
        float(first), float(second), rel_tol=0.0, abs_tol=1e-12
    )


def validate_repeat_audit(
    audit: Mapping[str, Any],
    gold_rows: Sequence[Mapping[str, Any]],
    *,
    source_gold_sha256: str,
) -> bool:
    if audit.get("schema_version") != "stage35_gold128_repeat_audit_v1":
        raise ValueError("repeat audit schema mismatch")
    if audit.get("source_gold_sha256") != source_gold_sha256:
        raise ValueError("repeat audit is not bound to the current Gold128 SHA256")
    performance_checks = audit.get("performance_checks")
    performance_rows = audit.get("rows")
    if (
        audit.get("qualified") is not True
        or audit.get("performance_repeat_qualified") is not True
        or audit.get("terminal_rows") != 32
        or not isinstance(performance_checks, Mapping)
        or not performance_checks
        or not all(value is True for value in performance_checks.values())
        or not isinstance(performance_rows, list)
        or len(performance_rows) != 32
    ):
        raise ValueError("repeat audit performance contract is incomplete")
    gold_by_id = {str(row.get("manifest_job_id")): row for row in gold_rows}
    repeated_ids: set[str] = set()
    categories: set[str] = set()
    for row in performance_rows:
        manifest_id = str(row.get("manifest_job_id") or "")
        if manifest_id in repeated_ids or manifest_id not in gold_by_id:
            raise ValueError("repeat audit performance IDs must be unique Gold rows")
        repeated_ids.add(manifest_id)
        categories.add(str(row.get("repeat_category") or ""))
        baseline = gold_by_id[manifest_id]
        if not _same_metric(row.get("baseline_latency_ms"), baseline.get("latency_ms")):
            raise ValueError(f"repeat latency baseline mismatch: {manifest_id}")
        if not _same_metric(row.get("baseline_energy_j"), baseline.get("energy_j")):
            raise ValueError(f"repeat energy baseline mismatch: {manifest_id}")
        result_path = Path(str(row.get("repeat_result_json") or ""))
        if (
            not result_path.is_file()
            or base.sha256_file(result_path) != row.get("repeat_result_sha256")
        ):
            raise ValueError(f"repeat performance artifact mismatch: {manifest_id}")
    if len(categories) < 4:
        raise ValueError("repeat audit requires at least four repeat categories")

    ap_audit = audit.get("ap_repeat_audit")
    if not isinstance(ap_audit, Mapping) or ap_audit.get("schema_version") != "stage35_gold128_ap_repeat_audit_v1":
        raise ValueError("AP repeat audit schema mismatch")
    ap_checks = ap_audit.get("checks")
    ap_rows = ap_audit.get("rows")
    if (
        ap_audit.get("qualified") is not True
        or ap_audit.get("terminal_rows") != 4
        or not isinstance(ap_checks, Mapping)
        or not ap_checks
        or not all(value is True for value in ap_checks.values())
        or not isinstance(ap_rows, list)
        or len(ap_rows) != 4
    ):
        raise ValueError("AP repeat audit contract is incomplete")
    ap_ids: set[str] = set()
    ap_categories: set[str] = set()
    for row in ap_rows:
        manifest_id = str(row.get("source_manifest_job_id") or "")
        if manifest_id in ap_ids or manifest_id not in gold_by_id:
            raise ValueError("AP repeat IDs must be unique Gold rows")
        ap_ids.add(manifest_id)
        ap_categories.add(str(row.get("repeat_category") or ""))
        baseline = gold_by_id[manifest_id]
        for key in ("ap30", "ap50", "ap70"):
            if not _same_metric(row.get(f"baseline_{key}"), baseline.get(key)):
                raise ValueError(f"AP repeat baseline mismatch for {manifest_id}/{key}")
        if row.get("baseline_ap_report_sha256") != baseline.get("ap_report_sha256"):
            raise ValueError(f"AP repeat source report mismatch: {manifest_id}")
        report_path = Path(str(row.get("repeat_report_path") or ""))
        if (
            not report_path.is_file()
            or base.sha256_file(report_path) != row.get("repeat_report_sha256")
        ):
            raise ValueError(f"AP repeat artifact mismatch: {manifest_id}")
    if len(ap_categories) < 4:
        raise ValueError("AP repeat audit requires four repeat categories")
    return True


def validate_context(context: dict[str, Any]) -> None:
    train_counts = {model: len(groups) for model, groups in context["train_pools"].items()}
    locked_counts = defaultdict(int)
    for group_id in context["locked_groups"]:
        locked_counts[context["groups"][group_id]["model"]] += 1
    if train_counts != {"codriving": 13, "pyramid": 13}:
        raise ValueError(f"Gold128 requires 13 train groups per model, got {train_counts}")
    if dict(locked_counts) != {"codriving": 3, "pyramid": 3}:
        raise ValueError(f"Gold128 requires 3 locked groups per model, got {dict(locked_counts)}")


def grouped_learning_curve(
    rows: list[dict[str, Any]], profiles: dict[str, dict[str, Any]], names: list[str],
    context: dict[str, Any], graph_features: dict[str, dict[str, Any]], repeats: int,
) -> list[dict[str, Any]]:
    output = []
    for repeat in range(repeats):
        for train_count in LEARNING_SIZES:
            train_groups, test_groups = base.manifest_group_split(
                context["train_pools"], locked_groups=context["locked_groups"],
                reference_groups=context["reference_groups"], train_count=train_count, seed=1000 + repeat,
            )
            for target in TARGETS:
                train = base.valid_rows(rows, set(train_groups), target)
                test = base.valid_rows(rows, set(test_groups), target)
                centers = base.reference_centers(rows, context["reference_groups"], target)
                for encoding in ("raw", "relative"):
                    for kind in MODEL_KINDS:
                        pred = base.fit_predict(
                            train, test, target=target, encoding=encoding, kind=kind,
                            profiles=profiles, names=names, seed=repeat, graph_features=graph_features,
                            centers=centers if encoding == "relative" else None,
                        )
                        truth = np.asarray([row[target] for row in test], dtype=float)
                        output.append({
                            "repeat": repeat, "train_groups": train_count, "target": target,
                            "encoding": encoding, "model_kind": kind, "test_rows": len(test),
                            "test_groups": len(set(test_groups)),
                            "topk_recall": base.topk_recall(truth, pred, target),
                            **base.metric_record(truth, pred),
                        })
    return output


def transfer_experiment(
    rows: list[dict[str, Any]], profiles: dict[str, dict[str, Any]], names: list[str],
    context: dict[str, Any], graph_features: dict[str, dict[str, Any]], repeats: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    selection_records: list[dict[str, Any]] = []
    models = sorted({row["model"] for row in rows})
    for source, target_model in ((models[0], models[1]), (models[1], models[0])):
        source_groups = set(context["train_pools"][source])
        target_groups = list(context["train_pools"][target_model])
        target_test_groups = {
            group for group in context["locked_groups"]
            if context["groups"][group]["model"] == target_model
        }
        for repeat in range(repeats):
            for k in K_GROUPS:
                calibration_groups = base.sample_calibration_groups(
                    target_groups, context["groups"], count=k, seed=5000 + repeat,
                )
                predictions: dict[tuple[str, str], dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
                truth_by_id: dict[str, dict[str, Any]] = {}
                for target in TARGETS:
                    source_rows = base.valid_rows(rows, source_groups, target)
                    calibration = base.valid_rows(rows, calibration_groups, target)
                    test = base.valid_rows(rows, target_test_groups, target)
                    truth = np.asarray([row[target] for row in test], dtype=float)
                    for kind in MODEL_KINDS:
                        global_pred = base.fit_predict(
                            source_rows, test, target=target, encoding="raw", kind=kind,
                            profiles=profiles, names=names, seed=repeat, graph_features=graph_features,
                        )
                        local_pred = base.fit_predict(
                            calibration, test, target=target, encoding="raw", kind=kind,
                            profiles=profiles, names=names, seed=repeat, graph_features=graph_features,
                        )
                        centers = base.reference_centers(rows, context["reference_groups"], target)
                        relative_model = base.make_model(kind, repeat)
                        x_source = np.vstack([base.row_features(row, profiles, names, graph_features) for row in source_rows])
                        y_source = base.encode_targets(
                            np.asarray([row[target] for row in source_rows]),
                            np.asarray([source] * len(source_rows)), target=target, centers=centers,
                        )
                        relative_model.fit(x_source, y_source)
                        cal_relative = relative_model.predict(
                            np.vstack([base.row_features(row, profiles, names, graph_features) for row in calibration])
                        )
                        test_relative = relative_model.predict(
                            np.vstack([base.row_features(row, profiles, names, graph_features) for row in test])
                        )
                        calibration_truth = base.transformed_targets(calibration, target)
                        offset = float(np.median(calibration_truth - cal_relative))
                        offset_pred = base.restore_transformed(offset + test_relative, target)
                        intercept, scale = base.fit_affine_calibrator(cal_relative, calibration_truth)
                        affine_pred = base.restore_transformed(intercept + scale * test_relative, target)
                        x_calibration = np.vstack([
                            base.row_features(row, profiles, names, graph_features)
                            for row in calibration
                        ])
                        x_test = np.vstack([
                            base.row_features(row, profiles, names, graph_features)
                            for row in test
                        ])
                        residual_transformed = residual_calibration_prediction(
                            x_calibration,
                            calibration_truth - cal_relative,
                            x_test,
                            test_relative,
                            seed=repeat,
                        )
                        residual_pred = base.restore_transformed(residual_transformed, target)
                        for method, encoding, pred in (
                            ("global_only", "raw_absolute", global_pred),
                            ("local_only", "raw_absolute", local_pred),
                            ("global_plus_k_offset", "relative_residual", offset_pred),
                            ("global_plus_k_affine", "relative_residual", affine_pred),
                            ("global_plus_k_residual", "relative_residual", residual_pred),
                        ):
                            records.append({
                                "source_model": source, "target_model": target_model, "repeat": repeat,
                                "k_groups": k, "target": target, "model_kind": kind, "method": method,
                                "encoding": encoding, "test_candidate_rows": len(test),
                                "test_groups": len(target_test_groups),
                                "topk_recall": base.topk_recall(truth, pred, target),
                                **base.metric_record(truth, pred),
                            })
                            for row, value in zip(test, pred):
                                predictions[(kind, method)][str(row["manifest_job_id"])][target] = float(value)
                                truth_by_id.setdefault(str(row["manifest_job_id"]), {
                                    "id": str(row["manifest_job_id"]),
                                })[target] = float(row[target])
                for (kind, method), predicted in predictions.items():
                    complete_ids = [
                        item for item in sorted(predicted)
                        if all(target in predicted[item] and target in truth_by_id[item] for target in TARGETS)
                    ]
                    truth_rows = [truth_by_id[item] for item in complete_ids]
                    predicted_rows = {item: predicted[item] for item in complete_ids}
                    selection_records.append({
                        "source_model": source, "target_model": target_model, "repeat": repeat,
                        "k_groups": k, "model_kind": kind, "method": method,
                        "encoding": "relative_residual" if method.startswith("global_plus") else "raw_absolute",
                        "test_candidate_rows": len(complete_ids), "test_groups": len(target_test_groups),
                        **selection_quality(truth_rows, predicted_rows),
                    })
    return records, selection_records


def _bootstrap_train_rows(rows: list[dict[str, Any]], groups: list[str], rng: np.random.Generator) -> list[dict[str, Any]]:
    sampled = rng.choice(groups, size=len(groups), replace=True)
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[str(row["group_id"])].append(row)
    return [row for group in sampled for row in by_group[str(group)]]


def prediction_interval_experiment(
    rows: list[dict[str, Any]], profiles: dict[str, dict[str, Any]], names: list[str],
    context: dict[str, Any], graph_features: dict[str, dict[str, Any]], bootstrap_repeats: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_groups = sorted(group for groups in context["train_pools"].values() for group in groups)
    calibration_groups: set[str] = set()
    for model, candidates in context["train_pools"].items():
        reference = context["reference_groups"][model]
        calibration_groups.update(base.sample_calibration_groups(
            [group for group in candidates if group != reference],
            context["groups"], count=3, seed=8100,
        ))
    fit_groups = [group for group in train_groups if group not in calibration_groups]
    test_groups = set(context["locked_groups"])
    records: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for target in TARGETS:
        train = base.valid_rows(rows, set(fit_groups), target)
        calibration = base.valid_rows(rows, calibration_groups, target)
        test = base.valid_rows(rows, test_groups, target)
        centers = base.reference_centers(rows, context["reference_groups"], target)
        for encoding in ("raw", "relative"):
            for kind in MODEL_KINDS:
                test_predictions = []
                calibration_predictions = []
                for bootstrap in range(bootstrap_repeats):
                    rng = np.random.default_rng(9000 + bootstrap)
                    sampled = _bootstrap_train_rows(train, fit_groups, rng)
                    test_predictions.append(base.fit_predict(
                        sampled, test, target=target, encoding=encoding, kind=kind,
                        profiles=profiles, names=names, seed=bootstrap, graph_features=graph_features,
                        centers=centers if encoding == "relative" else None,
                    ))
                    calibration_predictions.append(base.fit_predict(
                        sampled, calibration, target=target, encoding=encoding, kind=kind,
                        profiles=profiles, names=names, seed=bootstrap, graph_features=graph_features,
                        centers=centers if encoding == "relative" else None,
                    ))
                test_lower, test_upper = np.quantile(np.vstack(test_predictions), [0.05, 0.95], axis=0)
                cal_lower, cal_upper = np.quantile(
                    np.vstack(calibration_predictions), [0.05, 0.95], axis=0
                )
                calibration_records = [
                    {
                        "model": row["model"], "truth": float(row[target]),
                        "group_id": row["group_id"],
                        "lower": float(low), "upper": float(high),
                    }
                    for row, low, high in zip(calibration, cal_lower, cal_upper)
                ]
                uncalibrated_test = [
                    {
                        "target": target, "encoding": encoding, "model_kind": kind,
                        "manifest_job_id": row["manifest_job_id"], "group_id": row["group_id"],
                        "model": row["model"],
                        "truth": float(row[target]), "lower": float(low), "upper": float(high),
                    }
                    for row, low, high in zip(test, test_lower, test_upper)
                ]
                current = conformalize_intervals(
                    calibration_records, uncalibrated_test, coverage=0.90
                )
                records.extend(current)
                for model in sorted({str(row["model"]) for row in current}):
                    selected = [row for row in current if row["model"] == model]
                    summaries.append({
                        "target": target, "encoding": encoding, "model_kind": kind,
                        "model": model, **interval_coverage(selected),
                    })
    return records, summaries


def _summary_index(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[tuple[Any, ...], dict[str, Any]]:
    return {tuple(row[key] for key in keys): row for row in rows}


def decision_evidence(
    learning_summary: list[dict[str, Any]], transfer_summary: list[dict[str, Any]],
    selection_summary: list[dict[str, Any]], interval_summary: list[dict[str, Any]],
) -> dict[str, Any]:
    learning_index = _summary_index(learning_summary, ("train_groups", "target", "encoding", "model_kind"))
    learning = {}
    for target in TARGETS:
        at24 = learning_index[(24, target, PRIMARY_ENCODING, PRIMARY_MODEL_KIND)]
        at26 = learning_index[(26, target, PRIMARY_ENCODING, PRIMARY_MODEL_KIND)]
        learning[target] = {
            "mae_24": at24["mae_mean"], "mae_26": at26["mae_mean"],
            "mae_improvement_24_to_26": (at24["mae_mean"] - at26["mae_mean"]) / max(at24["mae_mean"], 1e-12),
            "spearman_26": at26["spearman_mean"], "topk_recall_26": at26["topk_recall_mean"],
        }

    transfer_index = _summary_index(
        transfer_summary, ("source_model", "target_model", "k_groups", "target", "model_kind", "method")
    )
    directions = sorted({(row["source_model"], row["target_model"]) for row in transfer_summary})
    comparisons = []
    for k_groups in K_GROUPS:
        for source, target_model in directions:
            for target in TARGETS:
                common = (source, target_model, k_groups, target, PRIMARY_MODEL_KIND)
                calibrated = transfer_index[(*common, PRIMARY_TRANSFER_METHOD)]
                global_only = transfer_index[(*common, "global_only")]
                local_only = transfer_index[(*common, "local_only")]
                comparisons.append({
                    "source_model": source, "target_model": target_model, "k_groups": k_groups,
                    "target": target, "calibrated_mae": calibrated["mae_mean"],
                    "global_only_mae": global_only["mae_mean"],
                    "local_only_mae": local_only["mae_mean"],
                    "beats_global_only": calibrated["mae_mean"] < global_only["mae_mean"],
                    "beats_local_only": calibrated["mae_mean"] < local_only["mae_mean"],
                })
    primary_selection = [
        row for row in selection_summary
        if row["k_groups"] in K_GROUPS and row["model_kind"] == PRIMARY_MODEL_KIND
        and row["method"] == PRIMARY_TRANSFER_METHOD
    ]
    primary_intervals = [
        row for row in interval_summary
        if row["encoding"] == PRIMARY_ENCODING and row["model_kind"] == PRIMARY_MODEL_KIND
    ]
    return {
        "learning": learning,
        "transfer_comparisons": comparisons,
        "calibrated_beats_global_only_count": sum(row["beats_global_only"] for row in comparisons),
        "calibrated_beats_local_only_count": sum(row["beats_local_only"] for row in comparisons),
        "transfer_task_count": len(comparisons),
        "pareto_recall_mean": float(np.mean([row["pareto_recall_mean"] for row in primary_selection])),
        "pareto_recall_p10": float(np.min([row["pareto_recall_p10"] for row in primary_selection])),
        "hv_regret_mean": float(np.mean([row["hv_regret_mean"] for row in primary_selection])),
        "hv_regret_p90": float(np.max([row["hv_regret_p90"] for row in primary_selection])),
        "coverage_90_min": float(min(row["coverage_90"] for row in primary_intervals)),
        "fully_covered_group_rate_90_min": float(min(
            row["fully_covered_group_rate_90"] for row in primary_intervals
        )),
        "primary_contract": {
            "model_kind": PRIMARY_MODEL_KIND, "encoding": PRIMARY_ENCODING,
            "transfer_method": PRIMARY_TRANSFER_METHOD, "k_groups": list(K_GROUPS),
        },
    }


def lock_decision(
    evidence: dict[str, Any], repeat_audit_qualified: bool | None
) -> dict[str, Any]:
    checks = {
        "learning_plateau": all(row["mae_improvement_24_to_26"] <= 0.05 for row in evidence["learning"].values()),
        "ranking": all(row["spearman_26"] >= 0.90 and row["topk_recall_26"] >= 0.75 for row in evidence["learning"].values()),
        "beats_global_only": evidence["calibrated_beats_global_only_count"] == evidence["transfer_task_count"],
        "beats_local_only": evidence["calibrated_beats_local_only_count"] >= 16,
        "pareto_recall": evidence["pareto_recall_p10"] >= 0.80,
        "hv_regret": evidence["hv_regret_p90"] <= 0.10,
        "prediction_interval_coverage": (
            evidence["coverage_90_min"] >= 0.80
            and evidence["fully_covered_group_rate_90_min"] >= 0.80
        ),
    }
    if not all(checks.values()):
        decision = "targeted_supplement_required"
    elif repeat_audit_qualified is None:
        decision = "repeat_audit_required"
    elif not repeat_audit_qualified:
        decision = "targeted_supplement_required"
    else:
        decision = "lock_gold128_for_stage4"
    return {"decision": decision, "checks": checks}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-json", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--capability-profiles-json", type=Path, required=True)
    parser.add_argument("--graph-features-json", type=Path, required=True)
    parser.add_argument("--repeat-audit-json", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--bootstrap-repeats", type=int, default=48)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = json.loads(args.gold_json.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest_json.read_text(encoding="utf-8"))
    validate_dataset_contract(rows, manifest)
    profiles = base.load_profiles(args.capability_profiles_json)
    graph_features = base.load_graph_features(args.graph_features_json)
    context = base.load_manifest_context(args.manifest_json)
    validate_context(context)
    names = base.feature_names(rows, profiles, include_graph=True)

    learning = grouped_learning_curve(rows, profiles, names, context, graph_features, args.repeats)
    transfer, selection = transfer_experiment(rows, profiles, names, context, graph_features, args.repeats)
    intervals, interval_summary = prediction_interval_experiment(
        rows, profiles, names, context, graph_features, args.bootstrap_repeats,
    )
    learning_summary = base.summarize(
        learning, ["train_groups", "target", "encoding", "model_kind"],
        ["mae", "mape", "spearman", "topk_recall"],
    )
    transfer_summary = base.summarize(
        transfer, ["source_model", "target_model", "k_groups", "target", "model_kind", "method", "encoding"],
        ["mae", "mape", "spearman", "topk_recall"],
    )
    selection_summary = base.summarize(
        selection, ["source_model", "target_model", "k_groups", "model_kind", "method", "encoding"],
        ["pareto_recall", "pareto_precision", "oracle_hv", "selected_true_hv", "hv_regret"],
    )
    evidence = decision_evidence(learning_summary, transfer_summary, selection_summary, interval_summary)
    repeat_audit = json.loads(args.repeat_audit_json.read_text(encoding="utf-8")) if args.repeat_audit_json else None
    repeat_audit_qualified = (
        validate_repeat_audit(
            repeat_audit, rows, source_gold_sha256=base.sha256_file(args.gold_json)
        )
        if repeat_audit is not None else None
    )
    decision = lock_decision(evidence, repeat_audit_qualified)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, payload in (
        ("learning_curve_raw.csv", learning), ("learning_curve_summary.csv", learning_summary),
        ("transfer_raw.csv", transfer), ("transfer_summary.csv", transfer_summary),
        ("selection_raw.csv", selection), ("selection_summary.csv", selection_summary),
        ("prediction_interval_raw.csv", intervals), ("prediction_interval_summary.csv", interval_summary),
    ):
        write_csv(args.output_dir / filename, payload)
    report = {
        "schema_version": SCHEMA, "rows": len(rows), "groups": len(context["groups"]),
        "train_groups": sum(len(items) for items in context["train_pools"].values()),
        "locked_holdout_groups": len(context["locked_groups"]),
        "learning_sizes": list(LEARNING_SIZES), "k_groups": list(K_GROUPS),
        "feature_names": names, "decision_evidence": evidence, **decision,
        "repeat_audit": repeat_audit,
        "source_sha256": {
            "gold": base.sha256_file(args.gold_json), "manifest": base.sha256_file(args.manifest_json),
            "capability_profiles": base.sha256_file(args.capability_profiles_json),
            "graph_features": base.sha256_file(args.graph_features_json),
            "runner": base.sha256_file(Path(__file__)),
        },
    }
    (args.output_dir / "gold128_sufficiency_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "success", "decision": report["decision"], "checks": report["checks"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
