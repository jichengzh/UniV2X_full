#!/usr/bin/env python3
"""Evaluate Gold176 sufficiency against the frozen Gold144 v4 protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import stage35_gold128_sufficiency_v1 as gold128
from scripts import stage35_gold144_sufficiency_v1 as gold144
from scripts import stage35_gold96_sufficiency_transfer_v1 as base


SCHEMA = "stage35_gold176_sufficiency_v1"
MANIFEST_SCHEMA = "stage35_gold176_manifest_v1"
ROW_SCHEMA = "stage35_gold176_final_v1"
LEARNING_SIZES = (6, 12, 18, 24, 28, 30, 32, 34, 38)
K_GROUPS = gold128.K_GROUPS
TARGETS = gold128.TARGETS
MODEL_KINDS = gold128.MODEL_KINDS
EXPECTED_ARMS = gold128.EXPECTED_ARMS

manifest_group_split = gold144.manifest_group_split
transfer_experiment = gold128.transfer_experiment
prediction_interval_experiment = gold128.prediction_interval_experiment


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_sha256(path: Path, expected_sha256: str, label: str) -> str:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"{label} SHA256 mismatch: {actual} != {expected_sha256}")
    return actual


def validate_dataset_contract(
    rows: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]
) -> None:
    jobs = manifest.get("jobs")
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError(f"Gold176 manifest must use {MANIFEST_SCHEMA}")
    if len(rows) != 176 or not isinstance(jobs, list) or len(jobs) != 176:
        raise ValueError("Gold176 requires exactly 176 result rows and manifest jobs")
    if any(not isinstance(row, Mapping) for row in rows) or any(
        not isinstance(job, Mapping) for job in jobs
    ):
        raise ValueError("Gold176 rows and manifest jobs must be objects")
    if any(row.get("schema_version") != ROW_SCHEMA for row in rows):
        raise ValueError(f"Gold176 rows must use {ROW_SCHEMA}")

    row_ids = [str(row.get("manifest_job_id") or "") for row in rows]
    job_ids = [str(job.get("job_id") or "") for job in jobs]
    if any(not item for item in row_ids + job_ids):
        raise ValueError("Gold176 result and manifest IDs must be non-empty")
    if len(set(row_ids)) != 176 or len(set(job_ids)) != 176:
        raise ValueError("Gold176 result and manifest job IDs must be unique")
    if set(row_ids) != set(job_ids):
        raise ValueError("Gold176 result and manifest job IDs must match")

    jobs_by_id = {str(job["job_id"]): job for job in jobs}
    binding_fields = (
        "group_id",
        "model",
        "width",
        "q_mode",
        "dispatch_key",
        "capability_profile_id",
        "split",
        "source_pool",
        "width_stratum",
    )
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    group_metadata: dict[str, tuple[str, str]] = {}
    for row in rows:
        manifest_id = str(row["manifest_job_id"])
        job = jobs_by_id[manifest_id]
        mismatches = [field for field in binding_fields if row.get(field) != job.get(field)]
        if mismatches:
            raise ValueError(
                f"Gold176 row/manifest binding mismatch for {manifest_id}: {mismatches}"
            )
        status = str(row.get("terminal_status") or "")
        if status == "measured_success_gold":
            if not all(base.finite(row.get(target)) for target in TARGETS):
                raise ValueError(f"measured Gold176 row lacks targets: {manifest_id}")
        elif status not in {"feasibility_failure", "numerical_feasibility_failure"}:
            raise ValueError(f"Gold176 row is not terminal: {manifest_id}")
        if job.get("split") == "locked_holdout" and status != "measured_success_gold":
            raise ValueError(f"Gold176 locked holdout must be measured: {manifest_id}")

        group_id = str(row.get("group_id") or "")
        metadata = (str(row.get("model") or ""), str(row.get("split") or ""))
        if not group_id or metadata[1] not in {"train", "locked_holdout"}:
            raise ValueError(f"Gold176 row has invalid group or split: {manifest_id}")
        if group_id in group_metadata and group_metadata[group_id] != metadata:
            raise ValueError(f"Gold176 group crosses model or split boundaries: {group_id}")
        group_metadata[group_id] = metadata
        grouped[group_id].append(row)

    if len(grouped) != 44:
        raise ValueError("Gold176 requires exactly 44 groups")
    for group_id, group_rows in grouped.items():
        arms = {
            (str(row.get("dispatch_key") or ""), str(row.get("q_mode") or ""))
            for row in group_rows
        }
        if len(group_rows) != 4 or arms != EXPECTED_ARMS:
            raise ValueError(f"Gold176 group is not a complete four-arm group: {group_id}")

    split_counts: dict[str, int] = defaultdict(int)
    for _, split in group_metadata.values():
        split_counts[split] += 1
    if dict(split_counts) != {"train": 38, "locked_holdout": 6}:
        raise ValueError(f"Gold176 requires 38 train and 6 locked groups, got {dict(split_counts)}")


def manifest_context(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return gold144.manifest_context(manifest)


def validate_context(context: Mapping[str, Any]) -> None:
    train_counts = {
        str(model): len(groups) for model, groups in context["train_pools"].items()
    }
    locked_counts: dict[str, int] = defaultdict(int)
    for group_id in context["locked_groups"]:
        locked_counts[str(context["groups"][group_id]["model"])] += 1
    if train_counts != {"codriving": 17, "pyramid": 21}:
        raise ValueError(f"Gold176 requires 17/21 train groups, got {train_counts}")
    if dict(locked_counts) != {"codriving": 3, "pyramid": 3}:
        raise ValueError(f"Gold176 requires 3 locked groups per model, got {dict(locked_counts)}")


def validate_gold144_source(
    rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    *,
    source_gold144_rows: Sequence[Mapping[str, Any]],
    source_gold144_sha256: str,
) -> bool:
    source_manifests = manifest.get("source_manifests")
    if (
        not isinstance(source_manifests, Mapping)
        or str(source_manifests.get("gold144_sha256") or "") != source_gold144_sha256
    ):
        raise ValueError("Gold176 provenance does not bind the source Gold144 SHA256")
    if len(source_gold144_rows) != 144:
        raise ValueError("source Gold144 must contain exactly 144 rows")

    source_by_id = {
        str(row.get("manifest_job_id") or ""): row for row in source_gold144_rows
    }
    current_rows = [row for row in rows if row.get("source_pool") == "gold144"]
    current_by_id = {str(row.get("manifest_job_id") or ""): row for row in current_rows}
    if "" in source_by_id or len(source_by_id) != 144:
        raise ValueError("source Gold144 IDs must be unique")
    if len(current_by_id) != 144 or set(current_by_id) != set(source_by_id):
        raise ValueError("Gold176 source_pool=gold144 IDs must match source Gold144")
    for manifest_id, source in source_by_id.items():
        if not gold144._evidence_matches(source, current_by_id[manifest_id]):
            raise ValueError(f"Gold144 frozen evidence changed in Gold176: {manifest_id}")
    return True


def validate_graph_features(
    graph_rows: Sequence[Mapping[str, Any]],
    *,
    context: Mapping[str, Any],
    source_gold_sha256: str,
) -> None:
    group_ids = [str(row.get("group_id") or "") for row in graph_rows]
    if len(group_ids) != 44 or len(set(group_ids)) != 44:
        raise ValueError("Gold176 graph features must contain 44 unique groups")
    if set(group_ids) != set(context["groups"]):
        raise ValueError("Gold176 graph features do not match manifest groups")
    if any(str(row.get("source_gold_sha256") or "") != source_gold_sha256 for row in graph_rows):
        raise ValueError("Gold176 graph features are not bound to the Gold176 source SHA256")


def grouped_learning_curve(
    rows: list[dict[str, Any]],
    profiles: dict[str, dict[str, Any]],
    names: list[str],
    context: dict[str, Any],
    graph_features: dict[str, dict[str, Any]],
    repeats: int,
) -> list[dict[str, Any]]:
    output = []
    for repeat in range(repeats):
        for train_count in LEARNING_SIZES:
            train_groups, test_groups = manifest_group_split(
                context["train_pools"],
                locked_groups=context["locked_groups"],
                reference_groups=context["reference_groups"],
                train_count=train_count,
                seed=1000 + repeat,
            )
            for target in TARGETS:
                train = base.valid_rows(rows, set(train_groups), target)
                test = base.valid_rows(rows, set(test_groups), target)
                centers = base.reference_centers(rows, context["reference_groups"], target)
                for encoding in ("raw", "relative"):
                    for kind in MODEL_KINDS:
                        prediction = base.fit_predict(
                            train,
                            test,
                            target=target,
                            encoding=encoding,
                            kind=kind,
                            profiles=profiles,
                            names=names,
                            seed=repeat,
                            graph_features=graph_features,
                            centers=centers if encoding == "relative" else None,
                        )
                        truth = np.asarray([row[target] for row in test], dtype=float)
                        output.append({
                            "repeat": repeat,
                            "train_groups": train_count,
                            "target": target,
                            "encoding": encoding,
                            "model_kind": kind,
                            "test_rows": len(test),
                            "test_groups": len(set(test_groups)),
                            "topk_recall": base.topk_recall(truth, prediction, target),
                            **base.metric_record(truth, prediction),
                        })
    return output


def decision_evidence(
    learning_summary: list[dict[str, Any]],
    transfer_summary: list[dict[str, Any]],
    selection_summary: list[dict[str, Any]],
    interval_summary: list[dict[str, Any]],
) -> dict[str, Any]:
    learning_index = gold128._summary_index(
        learning_summary, ("train_groups", "target", "encoding", "model_kind")
    )
    learning = {}
    for target in TARGETS:
        at32 = learning_index[(32, target, gold128.PRIMARY_ENCODING, gold128.PRIMARY_MODEL_KIND)]
        at34 = learning_index[(34, target, gold128.PRIMARY_ENCODING, gold128.PRIMARY_MODEL_KIND)]
        at38 = learning_index[(38, target, gold128.PRIMARY_ENCODING, gold128.PRIMARY_MODEL_KIND)]
        learning[target] = {
            "mae_32": at32["mae_mean"],
            "mae_34": at34["mae_mean"],
            "mae_38": at38["mae_mean"],
            "mae_improvement_32_to_34": (
                at32["mae_mean"] - at34["mae_mean"]
            ) / max(at32["mae_mean"], 1e-12),
            "mae_improvement_34_to_38": (
                at34["mae_mean"] - at38["mae_mean"]
            ) / max(at34["mae_mean"], 1e-12),
            "spearman_38": at38["spearman_mean"],
            "topk_recall_38": at38["topk_recall_mean"],
        }

    transfer_index = gold128._summary_index(
        transfer_summary,
        ("source_model", "target_model", "k_groups", "target", "model_kind", "method"),
    )
    directions = sorted(
        {(row["source_model"], row["target_model"]) for row in transfer_summary}
    )
    comparisons = []
    for k_groups in K_GROUPS:
        for source, target_model in directions:
            for target in TARGETS:
                common = (
                    source,
                    target_model,
                    k_groups,
                    target,
                    gold128.PRIMARY_MODEL_KIND,
                )
                calibrated = transfer_index[(*common, gold128.PRIMARY_TRANSFER_METHOD)]
                global_only = transfer_index[(*common, "global_only")]
                local_only = transfer_index[(*common, "local_only")]
                comparisons.append({
                    "source_model": source,
                    "target_model": target_model,
                    "k_groups": k_groups,
                    "target": target,
                    "calibrated_mae": calibrated["mae_mean"],
                    "global_only_mae": global_only["mae_mean"],
                    "local_only_mae": local_only["mae_mean"],
                    "beats_global_only": calibrated["mae_mean"] < global_only["mae_mean"],
                    "beats_local_only": calibrated["mae_mean"] < local_only["mae_mean"],
                })
    primary_selection = [
        row for row in selection_summary
        if row["k_groups"] in K_GROUPS
        and row["model_kind"] == gold128.PRIMARY_MODEL_KIND
        and row["method"] == gold128.PRIMARY_TRANSFER_METHOD
    ]
    primary_intervals = [
        row for row in interval_summary
        if row["encoding"] == gold128.PRIMARY_ENCODING
        and row["model_kind"] == gold128.PRIMARY_MODEL_KIND
    ]
    return {
        "learning": learning,
        "transfer_comparisons": comparisons,
        "calibrated_beats_global_only_count": sum(
            row["beats_global_only"] for row in comparisons
        ),
        "calibrated_beats_local_only_count": sum(
            row["beats_local_only"] for row in comparisons
        ),
        "transfer_task_count": len(comparisons),
        "pareto_recall_mean": float(np.mean([
            row["pareto_recall_mean"] for row in primary_selection
        ])),
        "pareto_recall_p10": float(np.min([
            row["pareto_recall_p10"] for row in primary_selection
        ])),
        "hv_regret_mean": float(np.mean([
            row["hv_regret_mean"] for row in primary_selection
        ])),
        "hv_regret_p90": float(np.max([
            row["hv_regret_p90"] for row in primary_selection
        ])),
        "coverage_90_min": float(min(
            row["coverage_90"] for row in primary_intervals
        )),
        "fully_covered_group_rate_90_min": float(min(
            row["fully_covered_group_rate_90"] for row in primary_intervals
        )),
        "primary_contract": {
            "model_kind": gold128.PRIMARY_MODEL_KIND,
            "encoding": gold128.PRIMARY_ENCODING,
            "transfer_method": gold128.PRIMARY_TRANSFER_METHOD,
            "k_groups": list(K_GROUPS),
        },
    }


def lock_decision(
    evidence: Mapping[str, Any],
    repeat_audit_qualified: bool | None,
    holdout_independent: bool,
) -> dict[str, Any]:
    checks = {
        "learning_plateau": all(
            row["mae_improvement_32_to_34"] <= 0.05
            and row["mae_improvement_34_to_38"] <= 0.05
            for row in evidence["learning"].values()
        ),
        "ranking": all(
            row["spearman_38"] >= 0.90 and row["topk_recall_38"] >= 0.75
            for row in evidence["learning"].values()
        ),
        "beats_global_only": (
            evidence["calibrated_beats_global_only_count"] == evidence["transfer_task_count"]
        ),
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
    elif not holdout_independent:
        decision = "independent_holdout_required"
    elif repeat_audit_qualified is None:
        decision = "repeat_audit_required"
    elif not repeat_audit_qualified:
        decision = "targeted_supplement_required"
    else:
        decision = "lock_gold176_for_stage4"
    return {"decision": decision, "checks": checks}


def assess_holdout_independence(candidate_plan: Mapping[str, Any]) -> dict[str, Any]:
    used_prior_report = bool(candidate_plan.get("source_sufficiency_report_sha256"))
    return {
        "holdout_independent": not used_prior_report,
        "reason": (
            "candidate_selection_used_prior_holdout_report"
            if used_prior_report
            else "candidate_selection_did_not_bind_prior_holdout_report"
        ),
    }


def compare_gate_reports(
    baseline_report: Mapping[str, Any], current_report: Mapping[str, Any]
) -> dict[str, Any]:
    baseline_checks = dict(baseline_report.get("checks") or {})
    current_checks = dict(current_report.get("checks") or {})
    if set(baseline_checks) != set(current_checks):
        raise ValueError("Gold144 and Gold176 reports must expose identical gate names")
    non_comparable = ["learning_plateau"]
    comparable = set(baseline_checks) - set(non_comparable)
    previous_failures = sorted(
        name for name, passed in baseline_checks.items()
        if name in comparable and not passed
    )
    resolved = sorted(name for name in previous_failures if current_checks[name])
    unresolved = sorted(name for name in previous_failures if not current_checks[name])
    regressions = sorted(
        name for name, passed in baseline_checks.items()
        if name in comparable and passed and not current_checks[name]
    )
    metric_deltas: dict[str, Any] = {}
    baseline_evidence = baseline_report.get("decision_evidence")
    current_evidence = current_report.get("decision_evidence")
    if isinstance(baseline_evidence, Mapping) and isinstance(current_evidence, Mapping):
        scalar_keys = (
            "calibrated_beats_global_only_count",
            "calibrated_beats_local_only_count",
            "pareto_recall_mean",
            "pareto_recall_p10",
            "hv_regret_mean",
            "hv_regret_p90",
            "coverage_90_min",
            "fully_covered_group_rate_90_min",
        )
        metric_deltas["scalars"] = {
            key: {
                "gold144": baseline_evidence[key],
                "gold176": current_evidence[key],
                "delta": current_evidence[key] - baseline_evidence[key],
            }
            for key in scalar_keys
        }
        metric_deltas["ranking"] = {
            target: {
                "spearman_gold144": baseline_evidence["learning"][target]["spearman_30"],
                "spearman_gold176": current_evidence["learning"][target]["spearman_38"],
                "spearman_delta": (
                    current_evidence["learning"][target]["spearman_38"]
                    - baseline_evidence["learning"][target]["spearman_30"]
                ),
                "topk_gold144": baseline_evidence["learning"][target]["topk_recall_30"],
                "topk_gold176": current_evidence["learning"][target]["topk_recall_38"],
                "topk_delta": (
                    current_evidence["learning"][target]["topk_recall_38"]
                    - baseline_evidence["learning"][target]["topk_recall_30"]
                ),
            }
            for target in TARGETS
        }
    return {
        "schema_version": "stage35_gold144_vs_gold176_gate_comparison_v1",
        "baseline_decision": baseline_report.get("decision"),
        "gold176_decision": current_report.get("decision"),
        "baseline_checks": baseline_checks,
        "gold176_checks": current_checks,
        "previous_failures": previous_failures,
        "resolved_failures": resolved,
        "unresolved_failures": unresolved,
        "new_regressions": regressions,
        "non_comparable_gates": non_comparable,
        "non_comparable_reason": {
            "learning_plateau": (
                "Gold144 26->30 and Gold176 34->38 add only Pyramid groups; "
                "Gold176 now uses balanced 32->34 plus a separately labelled Pyramid extension"
            ),
        },
        "all_previous_failures_resolved": not unresolved and not regressions,
        "metric_deltas": metric_deltas,
    }


def write_report(output_dir: Path, report: Mapping[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "gold176_sufficiency_report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_gate_comparison(output_dir: Path, payload: Mapping[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "gold144_vs_gold176_gate_comparison.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-json", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--source-gold144-json", type=Path, required=True)
    parser.add_argument("--source-gold144-manifest-json", type=Path, required=True)
    parser.add_argument("--expected-source-gold144-manifest-sha256", required=True)
    parser.add_argument("--baseline-report-json", type=Path, required=True)
    parser.add_argument("--targeted-candidate-plan-json", type=Path, required=True)
    parser.add_argument("--capability-profiles-json", type=Path, required=True)
    parser.add_argument("--graph-features-json", type=Path, required=True)
    parser.add_argument("--repeat-audit-json", type=Path, required=True)
    parser.add_argument("--repeat-audit-source-gold128-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--bootstrap-repeats", type=int, default=48)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.repeats <= 0 or args.bootstrap_repeats <= 0:
        raise ValueError("repeats and bootstrap-repeats must be positive")
    manifest_sha256 = validate_sha256(
        args.manifest_json, args.expected_manifest_sha256, "Gold176 manifest"
    )
    source_manifest_sha256 = validate_sha256(
        args.source_gold144_manifest_json,
        args.expected_source_gold144_manifest_sha256,
        "source Gold144 manifest",
    )
    gold_sha256 = sha256_file(args.gold_json)
    source_gold144_sha256 = sha256_file(args.source_gold144_json)

    rows = json.loads(args.gold_json.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest_json.read_text(encoding="utf-8"))
    source_gold144_rows = json.loads(args.source_gold144_json.read_text(encoding="utf-8"))
    source_gold144_manifest = json.loads(
        args.source_gold144_manifest_json.read_text(encoding="utf-8")
    )
    validate_dataset_contract(rows, manifest)
    validate_gold144_source(
        rows,
        manifest,
        source_gold144_rows=source_gold144_rows,
        source_gold144_sha256=source_gold144_sha256,
    )
    context = manifest_context(manifest)
    validate_context(context)

    graph_rows = json.loads(args.graph_features_json.read_text(encoding="utf-8"))
    validate_graph_features(
        graph_rows, context=context, source_gold_sha256=gold_sha256
    )
    graph_features = {str(row["group_id"]): row for row in graph_rows}
    profiles = base.load_profiles(args.capability_profiles_json)
    names = base.feature_names(rows, profiles, include_graph=True)

    learning = grouped_learning_curve(
        rows, profiles, names, context, graph_features, args.repeats
    )
    transfer, selection = transfer_experiment(
        rows, profiles, names, context, graph_features, args.repeats
    )
    intervals, interval_summary = prediction_interval_experiment(
        rows, profiles, names, context, graph_features, args.bootstrap_repeats
    )
    learning_summary = base.summarize(
        learning,
        ["train_groups", "target", "encoding", "model_kind"],
        ["mae", "mape", "spearman", "topk_recall"],
    )
    transfer_summary = base.summarize(
        transfer,
        [
            "source_model",
            "target_model",
            "k_groups",
            "target",
            "model_kind",
            "method",
            "encoding",
        ],
        ["mae", "mape", "spearman", "topk_recall"],
    )
    selection_summary = base.summarize(
        selection,
        [
            "source_model",
            "target_model",
            "k_groups",
            "model_kind",
            "method",
            "encoding",
        ],
        ["pareto_recall", "pareto_precision", "oracle_hv", "selected_true_hv", "hv_regret"],
    )
    evidence = decision_evidence(
        learning_summary, transfer_summary, selection_summary, interval_summary
    )

    repeat_audit = json.loads(args.repeat_audit_json.read_text(encoding="utf-8"))
    source_gold128_rows = json.loads(
        args.repeat_audit_source_gold128_json.read_text(encoding="utf-8")
    )
    inherited_repeat_audit_qualified = gold144.validate_repeat_audit(
        repeat_audit,
        source_gold144_rows,
        gold144_manifest=source_gold144_manifest,
        source_gold_rows=source_gold128_rows,
        source_gold_sha256=sha256_file(args.repeat_audit_source_gold128_json),
    )
    candidate_plan = json.loads(
        args.targeted_candidate_plan_json.read_text(encoding="utf-8")
    )
    holdout_assessment = assess_holdout_independence(candidate_plan)
    supplement_repeat_audit_qualified = None
    decision = lock_decision(
        evidence,
        supplement_repeat_audit_qualified,
        holdout_independent=holdout_assessment["holdout_independent"],
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, payload in (
        ("learning_curve_raw.csv", learning),
        ("learning_curve_summary.csv", learning_summary),
        ("transfer_raw.csv", transfer),
        ("transfer_summary.csv", transfer_summary),
        ("selection_raw.csv", selection),
        ("selection_summary.csv", selection_summary),
        ("prediction_interval_raw.csv", intervals),
        ("prediction_interval_summary.csv", interval_summary),
    ):
        gold128.write_csv(args.output_dir / filename, payload)
    report = {
        "schema_version": SCHEMA,
        "rows": len(rows),
        "groups": len(context["groups"]),
        "train_groups": sum(len(items) for items in context["train_pools"].values()),
        "train_groups_by_model": {
            model: len(items) for model, items in context["train_pools"].items()
        },
        "locked_holdout_groups": len(context["locked_groups"]),
        "learning_sizes": list(LEARNING_SIZES),
        "k_groups": list(K_GROUPS),
        "feature_names": names,
        "decision_evidence": evidence,
        **decision,
        "holdout_assessment": holdout_assessment,
        "inherited_repeat_audit_qualified": inherited_repeat_audit_qualified,
        "gold176_supplement_repeat_audit_qualified": supplement_repeat_audit_qualified,
        "source_sha256": {
            "gold": gold_sha256,
            "manifest": manifest_sha256,
            "source_gold144": source_gold144_sha256,
            "source_gold144_manifest": source_manifest_sha256,
            "capability_profiles": sha256_file(args.capability_profiles_json),
            "graph_features": sha256_file(args.graph_features_json),
            "repeat_audit": sha256_file(args.repeat_audit_json),
            "repeat_audit_source_gold128": sha256_file(
                args.repeat_audit_source_gold128_json
            ),
            "baseline_report": sha256_file(args.baseline_report_json),
            "targeted_candidate_plan": sha256_file(
                args.targeted_candidate_plan_json
            ),
            "runner": sha256_file(Path(__file__)),
        },
    }
    report_path = write_report(args.output_dir, report)
    baseline_report = json.loads(args.baseline_report_json.read_text(encoding="utf-8"))
    comparison_path = write_gate_comparison(
        args.output_dir, compare_gate_reports(baseline_report, report)
    )
    print(json.dumps({
        "status": "success",
        "decision": report["decision"],
        "report": str(report_path),
        "comparison": str(comparison_path),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
