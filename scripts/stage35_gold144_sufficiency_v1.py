#!/usr/bin/env python3
"""Evaluate strict Gold144 sufficiency while preserving the Gold128 v6 protocol."""

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
from scripts import stage35_gold96_sufficiency_transfer_v1 as base


SCHEMA = "stage35_gold144_sufficiency_v1"
MANIFEST_SCHEMA = "stage35_gold144_manifest_v1"
ROW_SCHEMA = "stage35_gold144_final_v1"
LEARNING_SIZES = (6, 12, 18, 24, 26, 30)
K_GROUPS = gold128.K_GROUPS
TARGETS = gold128.TARGETS
MODEL_KINDS = gold128.MODEL_KINDS
EXPECTED_ARMS = gold128.EXPECTED_ARMS
LOCKED_EVIDENCE_FIELDS = (
    "manifest_job_id",
    "group_id",
    "model",
    "width",
    "q_mode",
    "dispatch_key",
    "capability_profile_id",
    "split",
    "width_stratum",
    "terminal_status",
    "failure_reason",
    "latency_ms",
    "energy_j",
    "ap30",
    "ap50",
    "ap70",
    "performance_result_sha256",
    "ap_report_sha256",
)

# The expensive v6 experiments remain owned by Gold128.
transfer_experiment = gold128.transfer_experiment
prediction_interval_experiment = gold128.prediction_interval_experiment
selection_quality = gold128.selection_quality
interval_coverage = gold128.interval_coverage
conformalize_intervals = gold128.conformalize_intervals
residual_calibration_prediction = gold128.residual_calibration_prediction
hypervolume_3d = gold128.hypervolume_3d


def validate_dataset_contract(
    rows: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]
) -> None:
    jobs = manifest.get("jobs")
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError(f"Gold144 manifest must use {MANIFEST_SCHEMA}")
    if len(rows) != 144 or not isinstance(jobs, list) or len(jobs) != 144:
        raise ValueError("Gold144 requires exactly 144 result rows and manifest jobs")
    if any(not isinstance(row, Mapping) for row in rows) or any(
        not isinstance(job, Mapping) for job in jobs
    ):
        raise ValueError("Gold144 rows and manifest jobs must be objects")
    if any(row.get("schema_version") != ROW_SCHEMA for row in rows):
        raise ValueError(f"Gold144 rows must use {ROW_SCHEMA}")

    row_ids = [str(row.get("manifest_job_id") or "") for row in rows]
    job_ids = [str(job.get("job_id") or "") for job in jobs]
    if any(not item for item in row_ids + job_ids):
        raise ValueError("Gold144 result and manifest IDs must be non-empty")
    if len(set(row_ids)) != 144 or len(set(job_ids)) != 144:
        raise ValueError("Gold144 result and manifest job IDs must be unique")
    if set(row_ids) != set(job_ids):
        raise ValueError("Gold144 result and manifest job IDs must match")

    jobs_by_id = {str(job["job_id"]): job for job in jobs}
    binding_fields = (
        "group_id",
        "model",
        "width",
        "q_mode",
        "dispatch_key",
        "capability_profile_id",
        "split",
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
                f"Gold144 row/manifest binding mismatch for {manifest_id}: {mismatches}"
            )
        status = str(row.get("terminal_status") or "")
        if status == "measured_success_gold":
            if not all(base.finite(row.get(target)) for target in TARGETS):
                raise ValueError(f"measured Gold144 row lacks targets: {manifest_id}")
        elif status not in {"feasibility_failure", "numerical_feasibility_failure"}:
            raise ValueError(f"Gold144 row is not terminal: {manifest_id}")
        if job.get("split") == "locked_holdout" and status != "measured_success_gold":
            raise ValueError(f"Gold144 locked holdout must be measured: {manifest_id}")

        group_id = str(row.get("group_id") or "")
        metadata = (str(row.get("model") or ""), str(row.get("split") or ""))
        if not group_id or metadata[1] not in {"train", "locked_holdout"}:
            raise ValueError(f"Gold144 row has invalid group or split: {manifest_id}")
        if group_id in group_metadata and group_metadata[group_id] != metadata:
            raise ValueError(f"Gold144 group crosses model or split boundaries: {group_id}")
        group_metadata[group_id] = metadata
        grouped[group_id].append(row)

    if len(grouped) != 36:
        raise ValueError("Gold144 requires exactly 36 groups")
    for group_id, group_rows in grouped.items():
        arms = {
            (str(row.get("dispatch_key") or ""), str(row.get("q_mode") or ""))
            for row in group_rows
        }
        if len(group_rows) != 4 or arms != EXPECTED_ARMS:
            raise ValueError(f"Gold144 group is not a complete four-arm group: {group_id}")

    split_counts: dict[str, int] = defaultdict(int)
    for _, split in group_metadata.values():
        split_counts[split] += 1
    if dict(split_counts) != {"train": 30, "locked_holdout": 6}:
        raise ValueError(f"Gold144 requires 30 train and 6 locked groups, got {dict(split_counts)}")


def manifest_context(manifest: Mapping[str, Any]) -> dict[str, Any]:
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list):
        raise ValueError("Gold144 manifest jobs must be a list")
    groups: dict[str, dict[str, str]] = {}
    for job in jobs:
        group_id = str(job["group_id"])
        metadata = {
            "model": str(job["model"]),
            "split": str(job["split"]),
            "width_stratum": str(job.get("width_stratum") or ""),
        }
        if group_id in groups and groups[group_id] != metadata:
            raise ValueError(f"Gold144 manifest group metadata changed within {group_id}")
        groups[group_id] = metadata

    train_pools: dict[str, list[str]] = defaultdict(list)
    locked_groups = []
    for group_id, metadata in groups.items():
        if metadata["split"] == "locked_holdout":
            locked_groups.append(group_id)
        elif metadata["split"] == "train":
            train_pools[metadata["model"]].append(group_id)
        else:
            raise ValueError(f"Gold144 manifest has invalid split for {group_id}")

    pilot_groups = [str(group) for group in manifest.get("pilot_group_ids", [])]
    if len(pilot_groups) != 2 or not set(pilot_groups) <= set(groups):
        raise ValueError("Gold144 requires two valid pilot/reference groups")
    reference_groups: dict[str, str] = {}
    for model in sorted(train_pools):
        matches = [group for group in pilot_groups if groups[group]["model"] == model]
        if len(matches) != 1 or matches[0] not in train_pools[model]:
            raise ValueError(f"Gold144 requires one train reference group for {model}")
        reference_groups[model] = matches[0]
    return {
        "groups": groups,
        "train_pools": {model: sorted(values) for model, values in train_pools.items()},
        "locked_groups": sorted(locked_groups),
        "reference_groups": reference_groups,
    }


def validate_context(context: Mapping[str, Any]) -> None:
    train_counts = {
        str(model): len(groups) for model, groups in context["train_pools"].items()
    }
    locked_counts: dict[str, int] = defaultdict(int)
    for group_id in context["locked_groups"]:
        locked_counts[str(context["groups"][group_id]["model"])] += 1
    if set(train_counts) != {"codriving", "pyramid"} or sum(train_counts.values()) != 30:
        raise ValueError(f"Gold144 requires 30 train groups across both models, got {train_counts}")
    if any(count < 13 for count in train_counts.values()):
        raise ValueError(f"Gold144 requires at least 13 train groups per model, got {train_counts}")
    if dict(locked_counts) != {"codriving": 3, "pyramid": 3}:
        raise ValueError(
            f"Gold144 requires 3 locked groups per model, got {dict(locked_counts)}"
        )


def manifest_group_split(
    train_pools: Mapping[str, Sequence[str]],
    *,
    locked_groups: Sequence[str],
    reference_groups: Mapping[str, str],
    train_count: int,
    seed: int,
) -> tuple[list[str], list[str]]:
    models = sorted(train_pools)
    capacity = {model: len(train_pools[model]) for model in models}
    if train_count < len(models) or train_count > sum(capacity.values()):
        raise ValueError("train_count exceeds Gold144 manifest training pool")

    allocations = {
        model: min(train_count // len(models), capacity[model]) for model in models
    }
    remaining = train_count - sum(allocations.values())
    while remaining:
        progressed = False
        for model in models:
            if allocations[model] < capacity[model]:
                allocations[model] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            raise ValueError("cannot allocate requested Gold144 training groups")

    rng = np.random.default_rng(seed)
    selected: list[str] = []
    for model in models:
        reference = str(reference_groups[model])
        if reference not in train_pools[model]:
            raise ValueError(f"reference group {reference} is outside the training pool")
        candidates = [str(group) for group in train_pools[model] if group != reference]
        rng.shuffle(candidates)
        selected.extend([reference, *candidates[: allocations[model] - 1]])
    if len(selected) != train_count:
        raise ValueError("Gold144 grouped split did not select the requested training count")
    if set(selected) & set(locked_groups):
        raise ValueError("locked holdout leaked into Gold144 training groups")
    return selected, list(locked_groups)


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
    compatibility_learning = []
    for row in learning_summary:
        train_groups = int(row["train_groups"])
        if train_groups in {26, 30}:
            compatibility_learning.append({
                **row,
                "train_groups": 24 if train_groups == 26 else 26,
            })
    evidence = gold128.decision_evidence(
        compatibility_learning,
        transfer_summary,
        selection_summary,
        interval_summary,
    )
    learning = {}
    for target, row in evidence["learning"].items():
        learning[target] = {
            "mae_26": row["mae_24"],
            "mae_30": row["mae_26"],
            "mae_improvement_26_to_30": row["mae_improvement_24_to_26"],
            "spearman_30": row["spearman_26"],
            "topk_recall_30": row["topk_recall_26"],
        }
    return {**evidence, "learning": learning}


def lock_decision(
    evidence: Mapping[str, Any], repeat_audit_qualified: bool | None
) -> dict[str, Any]:
    compatibility_evidence = {
        **evidence,
        "learning": {
            target: {
                "mae_improvement_24_to_26": row["mae_improvement_26_to_30"],
                "spearman_26": row["spearman_30"],
                "topk_recall_26": row["topk_recall_30"],
            }
            for target, row in evidence["learning"].items()
        },
    }
    result = gold128.lock_decision(compatibility_evidence, repeat_audit_qualified)
    decision = (
        "lock_gold144_for_stage4"
        if result["decision"] == "lock_gold128_for_stage4"
        else "targeted_supplement_required"
    )
    return {**result, "decision": decision}


def _evidence_matches(source: Mapping[str, Any], current: Mapping[str, Any]) -> bool:
    for field in LOCKED_EVIDENCE_FIELDS:
        first = source.get(field)
        second = current.get(field)
        if field in {"latency_ms", "energy_j", "ap30", "ap50", "ap70"}:
            if first is None and second is None:
                continue
            if not (
                base.finite(first)
                and base.finite(second)
                and math.isclose(float(first), float(second), rel_tol=0.0, abs_tol=1e-12)
            ):
                return False
        elif first != second:
            return False
    return True


def validate_repeat_audit(
    audit: Mapping[str, Any],
    gold_rows: Sequence[Mapping[str, Any]],
    *,
    gold144_manifest: Mapping[str, Any],
    source_gold_rows: Sequence[Mapping[str, Any]],
    source_gold_sha256: str,
) -> bool:
    if len(source_gold_rows) != 128:
        raise ValueError("repeat-audit source Gold128 must contain exactly 128 rows")
    gold128.validate_repeat_audit(
        audit,
        source_gold_rows,
        source_gold_sha256=source_gold_sha256,
    )
    source_manifests = gold144_manifest.get("source_manifests")
    if (
        not isinstance(source_manifests, Mapping)
        or str(source_manifests.get("gold128_sha256") or "") != source_gold_sha256
    ):
        raise ValueError("Gold144 manifest provenance does not bind the source Gold128 SHA256")

    source_by_id = {
        str(row.get("manifest_job_id") or ""): row for row in source_gold_rows
    }
    if "" in source_by_id or len(source_by_id) != 128:
        raise ValueError("repeat-audit source Gold128 manifest IDs must be unique")
    current_gold128_rows = [
        row for row in gold_rows if str(row.get("source_pool") or "") == "gold128"
    ]
    current_by_id = {
        str(row.get("manifest_job_id") or ""): row for row in current_gold128_rows
    }
    if len(current_gold128_rows) != 128 or set(current_by_id) != set(source_by_id):
        raise ValueError("Gold144 source_pool=gold128 row IDs must match all source Gold128 rows")

    baseline_ids = {
        str(row.get("manifest_job_id") or "") for row in audit["rows"]
    }
    baseline_ids.update(
        str(row.get("source_manifest_job_id") or "")
        for row in audit["ap_repeat_audit"]["rows"]
    )
    for manifest_id in baseline_ids:
        if manifest_id not in source_by_id or manifest_id not in current_by_id:
            raise ValueError(f"repeat audit baseline is absent from Gold144: {manifest_id}")
        if not _evidence_matches(source_by_id[manifest_id], current_by_id[manifest_id]):
            raise ValueError(f"repeat audit baseline metrics or hashes changed: {manifest_id}")

    locked_source = [
        row for row in source_gold_rows if row.get("split") == "locked_holdout"
    ]
    if len(locked_source) != 24:
        raise ValueError("repeat-audit source Gold128 must contain six complete locked groups")
    for source in locked_source:
        manifest_id = str(source["manifest_job_id"])
        current = current_by_id.get(manifest_id)
        if current is None or not _evidence_matches(source, current):
            raise ValueError(f"original Gold128 locked holdout changed: {manifest_id}")
    for manifest_id, source in source_by_id.items():
        if not _evidence_matches(source, current_by_id[manifest_id]):
            raise ValueError(f"original Gold128 frozen evidence changed: {manifest_id}")
    return True


def write_report(output_dir: Path, report: Mapping[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "gold144_sufficiency_report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def validate_manifest_sha256(path: Path, expected_sha256: str) -> str:
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected_sha256:
        raise ValueError(
            f"Gold144 manifest SHA256 does not match frozen value: {actual} != {expected_sha256}"
        )
    return actual


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-json", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--capability-profiles-json", type=Path, required=True)
    parser.add_argument("--graph-features-json", type=Path, required=True)
    parser.add_argument("--repeat-audit-json", type=Path, required=True)
    parser.add_argument(
        "--repeat-audit-source-gold",
        "--repeat-audit-source-gold-json",
        dest="repeat_audit_source_gold",
        type=Path,
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--bootstrap-repeats", type=int, default=48)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.repeats <= 0 or args.bootstrap_repeats <= 0:
        raise ValueError("repeats and bootstrap-repeats must be positive")
    manifest_sha256 = validate_manifest_sha256(
        args.manifest_json, args.expected_manifest_sha256
    )
    rows = json.loads(args.gold_json.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest_json.read_text(encoding="utf-8"))
    validate_dataset_contract(rows, manifest)
    context = manifest_context(manifest)
    validate_context(context)
    profiles = base.load_profiles(args.capability_profiles_json)
    graph_features = base.load_graph_features(args.graph_features_json)
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
    source_gold_rows = json.loads(
        args.repeat_audit_source_gold.read_text(encoding="utf-8")
    )
    repeat_audit_qualified = validate_repeat_audit(
        repeat_audit,
        rows,
        gold144_manifest=manifest,
        source_gold_rows=source_gold_rows,
        source_gold_sha256=base.sha256_file(args.repeat_audit_source_gold),
    )
    decision = lock_decision(evidence, repeat_audit_qualified)

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
        "repeat_audit": repeat_audit,
        "source_sha256": {
            "gold": base.sha256_file(args.gold_json),
            "manifest": manifest_sha256,
            "capability_profiles": base.sha256_file(args.capability_profiles_json),
            "graph_features": base.sha256_file(args.graph_features_json),
            "repeat_audit": base.sha256_file(args.repeat_audit_json),
            "repeat_audit_source_gold": base.sha256_file(args.repeat_audit_source_gold),
            "runner": base.sha256_file(Path(__file__)),
        },
    }
    write_report(args.output_dir, report)
    print(json.dumps({
        "status": "success",
        "decision": report["decision"],
        "checks": report["checks"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
