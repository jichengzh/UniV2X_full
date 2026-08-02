"""Validated descriptive statistics for the Stage7 v2 core ablation."""

from __future__ import annotations

import copy
import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from framework.stage7 import core_ablation_v2 as contracts
from framework.stage7.formal_evidence_v2 import (
    CORE_VARIANTS,
    EVIDENCE_KINDS,
    FORMAL_EVIDENCE_SCHEMA,
    REQUIRED_AUDIT_FLAGS,
    ROUND_COUNT,
    BATCH_SIZE,
    TOTAL_EVENTS,
    TOTAL_TRAJECTORIES,
    JSON,
    fail,
    is_sha,
)
from framework.stage7.online_component_ablation_v1 import SEEDS
from scripts import stage7_finalize_ablation_v1 as metric_helpers


def validate_ready_evidence(evidence: Mapping[str, Any]) -> list[JSON]:
    if evidence.get("schema_version") != FORMAL_EVIDENCE_SCHEMA:
        fail("formal evidence schema drift")
    events = evidence.get("events")
    if (
        not isinstance(events, list)
        or len(events) != TOTAL_EVENTS
        or evidence.get("selected_event_count") != TOTAL_EVENTS
    ):
        fail("formal finalization requires exactly 192 selected events")
    identities: set[tuple[str, int, int, int]] = set()
    matrix: defaultdict[tuple[str, int, int], int] = defaultdict(int)
    misses = 0
    for row in events:
        if not isinstance(row, Mapping):
            fail("formal event row is invalid")
        try:
            identity = (
                str(row.get("variant") or ""),
                int(row.get("seed", -1)),
                int(row.get("round_index", -1)),
                int(row.get("event_index", -1)),
            )
        except (TypeError, ValueError):
            fail("formal event identity is invalid")
        if (
            identity in identities
            or identity[0] not in CORE_VARIANTS
            or identity[1] not in SEEDS
            or identity[2] not in range(ROUND_COUNT)
            or identity[3] not in range(16)
        ):
            fail("formal 12-trajectory event matrix drift")
        identities.add(identity)
        matrix[identity[:3]] += 1
        kind = row.get("terminal_evidence_kind")
        disposition = row.get("cache_disposition")
        if kind not in EVIDENCE_KINDS:
            fail("formal event lacks actual-v3 terminal evidence")
        if disposition == "miss":
            misses += 1
            if kind not in {
                "actual_v3_success",
                "actual_v3_candidate_failure",
            }:
                fail("formal miss lacks actual-v3 terminal evidence")
        elif disposition != "hit" or kind != "cross_request_exact_hit":
            fail("formal exact-hit evidence kind drift")
        if kind == "actual_v3_candidate_failure" and any(
            row.get(field) is not None
            for field in ("latency_ms", "energy_j", "ap30", "ap50", "ap70")
        ):
            fail("candidate failure objectives must be null")
    if (
        len(identities) != TOTAL_EVENTS
        or len(matrix) != TOTAL_TRAJECTORIES * ROUND_COUNT
        or any(count != BATCH_SIZE for count in matrix.values())
        or evidence.get("trajectory_count") != TOTAL_TRAJECTORIES
    ):
        fail("formal finalization requires exactly 12 trajectories")
    audits = evidence.get("audits")
    if (
        not isinstance(audits, Mapping)
        or set(audits) != set(REQUIRED_AUDIT_FLAGS)
        or any(value is not True for value in audits.values())
    ):
        fail("formal global audit bundle is incomplete")
    if (
        evidence.get("miss_count") != misses
        or evidence.get("actual_v3_miss_evidence_count") != misses
        or evidence.get("formal_v2_gpu_jobs_launched") != misses
    ):
        fail("formal actual-v3 miss/GPU evidence count drift")
    if evidence.get("silent_surrogate_fallback_count") != 0:
        fail("silent surrogate fallback count must be zero")
    if evidence.get(
        "ordered_pre_scan_sha256"
    ) != contracts.EXPECTED_ORDERED_PRE_SCAN_SHA256 or not is_sha(
        evidence.get("contract_sha256")
    ):
        fail("formal frozen contract identity drift")
    initial = evidence.get("initial_gold176_rows")
    reference = evidence.get("hv_reference")
    if (
        not isinstance(initial, list)
        or not initial
        or not isinstance(reference, list)
        or len(reference) != 3
        or not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            for value in reference
        )
    ):
        fail("formal Gold176/HV metric context is missing")
    return [copy.deepcopy(dict(row)) for row in events]


def trajectory_summaries(
    events: Sequence[JSON],
    *,
    initial_rows: Sequence[JSON],
    reference: tuple[float, float, float],
) -> list[JSON]:
    baseline = metric_helpers._hypervolume_3d(initial_rows, reference)
    summaries: list[JSON] = []
    for variant in CORE_VARIANTS:
        for seed in SEEDS:
            rows = sorted(
                (
                    row
                    for row in events
                    if row["variant"] == variant and row["seed"] == seed
                ),
                key=lambda row: int(row["event_index"]),
            )
            successful: list[JSON] = []
            curve = [0.0]
            for round_index in range(ROUND_COUNT):
                successful.extend(
                    row
                    for row in rows
                    if row["round_index"] == round_index
                    and row["terminal_evidence_kind"]
                    in {"actual_v3_success", "cross_request_exact_hit"}
                )
                curve.append(
                    max(
                        0.0,
                        metric_helpers._hypervolume_3d(
                            [*initial_rows, *successful], reference
                        )
                        - baseline,
                    )
                )
            auc = sum(2.0 * (left + right) for left, right in zip(curve, curve[1:]))
            summaries.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "delta_hv_auc": auc,
                    "delta_hv_at_16": curve[-1],
                    "invalid_rate": sum(
                        row["terminal_evidence_kind"] == "actual_v3_candidate_failure"
                        for row in rows
                    )
                    / 16,
                    "valid_yield": len(successful) / 16,
                    "fp16_count": sum(row.get("q_mode") == "fp16" for row in rows),
                    "int8_count": sum(row.get("q_mode") == "int8" for row in rows),
                    "exact_cache_hits": sum(
                        row.get("cache_disposition") == "hit" for row in rows
                    ),
                    "exact_cache_misses": sum(
                        row.get("cache_disposition") == "miss" for row in rows
                    ),
                    "best_latency_ms": min(
                        (float(row["latency_ms"]) for row in successful),
                        default=None,
                    ),
                    "best_energy_j": min(
                        (float(row["energy_j"]) for row in successful),
                        default=None,
                    ),
                    "selected_candidate_ids": [
                        str(row["candidate_id"]) for row in rows
                    ],
                }
            )
    return summaries


def describe(values: Sequence[float]) -> JSON:
    return {
        "values": list(values),
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "median": statistics.median(values),
        "range": [min(values), max(values)],
    }


def paired_statistics(trajectories: Sequence[JSON]) -> JSON:
    by_key = {(str(row["variant"]), int(row["seed"])): row for row in trajectories}
    metrics = (
        "delta_hv_auc",
        "delta_hv_at_16",
        "invalid_rate",
        "valid_yield",
        "exact_cache_hits",
        "exact_cache_misses",
    )
    variants: dict[str, JSON] = {}
    for variant in CORE_VARIANTS:
        metric_rows: dict[str, JSON] = {}
        for metric in metrics:
            values = [float(by_key[(variant, seed)][metric]) for seed in SEEDS]
            deltas = [
                value - float(by_key[("full", seed)][metric])
                for value, seed in zip(values, SEEDS)
            ]
            metric_rows[metric] = {
                **describe(values),
                "per_seed": {
                    str(seed): values[index] for index, seed in enumerate(SEEDS)
                },
                "paired_delta_vs_full": {
                    **describe(deltas),
                    "per_seed": {
                        str(seed): deltas[index] for index, seed in enumerate(SEEDS)
                    },
                    "direction_consistency": {
                        "positive": sum(value > 0 for value in deltas),
                        "equal": sum(value == 0 for value in deltas),
                        "negative": sum(value < 0 for value in deltas),
                    },
                },
            }
        variants[variant] = metric_rows
    return {
        "schema_version": "stage7_descriptive_paired_statistics_v2",
        "statistics_policy": "descriptive_paired_only",
        "seed_count": len(SEEDS),
        "significance_tests_performed": False,
        "variants": variants,
    }


def paper_rows(stats: Mapping[str, Any]) -> list[JSON]:
    rows: list[JSON] = []
    for variant in CORE_VARIANTS:
        metrics = stats["variants"][variant]

        def compact(metric: str) -> str:
            value = metrics[metric]
            return (
                f"{value['mean']:.6g}±{value['sample_std']:.3g}; "
                f"median {value['median']:.6g} "
                f"[{value['range'][0]:.6g},{value['range'][1]:.6g}]"
            )

        delta = metrics["delta_hv_auc"]["paired_delta_vs_full"]
        rows.append(
            {
                "Variant": variant,
                "DeltaHV-AUC ↑": compact("delta_hv_auc"),
                "DeltaHV@16 ↑": compact("delta_hv_at_16"),
                "Invalid rate ↓": compact("invalid_rate"),
                "Valid yield ↑": compact("valid_yield"),
                "Paired ΔHV-AUC vs Full": (
                    f"{delta['mean']:.6g}; "
                    f"+/0/-={delta['direction_consistency']['positive']}/"
                    f"{delta['direction_consistency']['equal']}/"
                    f"{delta['direction_consistency']['negative']}"
                ),
                "Status": "complete",
            }
        )
    return rows
