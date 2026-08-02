"""Pareto/uncertainty acquisition and measurement feedback for canonical Stage2 v3."""

from __future__ import annotations

import copy
import math
from typing import Any, Mapping, Sequence

from framework.stage2.canonical_search_v3 import validate_capability_profile
from framework.stage2.cost_model_bundle_v3 import update_model_bundle


OBJECTIVES = ("latency_ms", "energy_j", "ap70")


def _objective_vector(
    row: Mapping[str, Any], objectives: Sequence[str] = OBJECTIVES
) -> tuple[float, ...]:
    predictions = row.get("predictions") or {}
    missing = [name for name in objectives if predictions.get(name) is None]
    if missing:
        raise ValueError(f"candidate missing predicted objectives: {missing}")
    return tuple(
        -float(predictions[name]) if name == "ap70" else float(predictions[name])
        for name in objectives
    )


def dominates(left: Sequence[float], right: Sequence[float]) -> bool:
    return all(a <= b for a, b in zip(left, right)) and any(a < b for a, b in zip(left, right))


def nondominated_ranks(vectors: Sequence[Sequence[float]]) -> list[int]:
    domination_sets = [[] for _ in vectors]
    dominated_counts = [0 for _ in vectors]
    fronts: list[list[int]] = [[]]
    ranks = [-1 for _ in vectors]
    for left_index, left in enumerate(vectors):
        for right_index, right in enumerate(vectors):
            if left_index == right_index:
                continue
            if dominates(left, right):
                domination_sets[left_index].append(right_index)
            elif dominates(right, left):
                dominated_counts[left_index] += 1
        if dominated_counts[left_index] == 0:
            ranks[left_index] = 0
            fronts[0].append(left_index)
    front_index = 0
    while front_index < len(fronts) and fronts[front_index]:
        next_front = []
        for left_index in fronts[front_index]:
            for right_index in domination_sets[left_index]:
                dominated_counts[right_index] -= 1
                if dominated_counts[right_index] == 0:
                    ranks[right_index] = front_index + 1
                    next_front.append(right_index)
        if next_front:
            fronts.append(next_front)
        front_index += 1
    return ranks


def crowding_distances(vectors: Sequence[Sequence[float]], ranks: Sequence[int]) -> list[float]:
    distances = [0.0 for _ in vectors]
    for rank in sorted(set(ranks)):
        indices = [index for index, value in enumerate(ranks) if value == rank]
        if len(indices) <= 2:
            for index in indices:
                distances[index] = math.inf
            continue
        for objective in range(len(vectors[0])):
            ordered = sorted(indices, key=lambda index: vectors[index][objective])
            distances[ordered[0]] = math.inf
            distances[ordered[-1]] = math.inf
            low = vectors[ordered[0]][objective]
            high = vectors[ordered[-1]][objective]
            span = high - low or 1.0
            for position in range(1, len(ordered) - 1):
                index = ordered[position]
                if not math.isinf(distances[index]):
                    distances[index] += (
                        vectors[ordered[position + 1]][objective]
                        - vectors[ordered[position - 1]][objective]
                    ) / span
    return distances


def _uncertainty_score(row: Mapping[str, Any]) -> float:
    uncertainty = row.get("uncertainty_p90") or {}
    values = [float(uncertainty[name]) for name in OBJECTIVES if uncertainty.get(name) is not None]
    return sum(values)


def select_candidates(
    predicted_rows: Sequence[Mapping[str, Any]],
    *,
    measured_row_ids: set[str],
    budget: int,
    objectives: Sequence[str] = OBJECTIVES,
) -> list[dict[str, Any]]:
    if budget <= 0:
        raise ValueError("budget must be positive")
    candidates = [
        copy.deepcopy(dict(row))
        for row in predicted_rows
        if str(row.get("row_id")) not in measured_row_ids
        and float((row.get("predictions") or {}).get("feasibility", 1.0)) >= 0.5
    ]
    if not objectives or any(name not in OBJECTIVES for name in objectives):
        raise ValueError("objectives must be a non-empty subset of canonical objectives")
    vectors = [_objective_vector(row, objectives) for row in candidates]
    ranks = nondominated_ranks(vectors)
    crowding = crowding_distances(vectors, ranks)
    enriched = [
        {
            **row,
            "pareto_rank": ranks[index],
            "crowding_distance": crowding[index],
            "acquisition_uncertainty": sum(
                float((row.get("uncertainty_p90") or {}).get(name) or 0.0)
                for name in objectives
            ),
        }
        for index, row in enumerate(candidates)
    ]
    enriched.sort(
        key=lambda row: (
            row["pareto_rank"],
            -row["acquisition_uncertainty"],
            -row["crowding_distance"],
            str(row.get("row_id")),
        )
    )
    return enriched[:budget]


def select_candidate_groups(
    predicted_rows: Sequence[Mapping[str, Any]],
    *,
    measured_group_ids: set[str],
    group_budget: int,
    objectives: Sequence[str] = OBJECTIVES,
    expected_group_size: int = 4,
) -> list[dict[str, Any]]:
    """Select complete paired groups while retaining row-level Pareto evidence."""

    if group_budget <= 0:
        raise ValueError("group_budget must be positive")
    candidates = [
        copy.deepcopy(dict(row))
        for row in predicted_rows
        if str(row.get("group_id") or "") not in measured_group_ids
    ]
    if any(not str(row.get("group_id") or "") for row in candidates):
        raise ValueError("group acquisition requires group_id on every candidate")
    candidate_group_sizes: dict[str, int] = {}
    for row in candidates:
        group_id = str(row["group_id"])
        candidate_group_sizes[group_id] = candidate_group_sizes.get(group_id, 0) + 1
    if any(size != expected_group_size for size in candidate_group_sizes.values()):
        raise ValueError(f"group acquisition requires complete groups of {expected_group_size} rows")
    enriched = select_candidates(
        candidates,
        measured_row_ids=set(),
        budget=len(candidates),
        objectives=objectives,
    )
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in enriched:
        grouped.setdefault(str(row["group_id"]), []).append(row)
    ranked_groups = sorted(
        grouped,
        key=lambda group_id: (
            min(row["pareto_rank"] for row in grouped[group_id]),
            -sum(row["acquisition_uncertainty"] for row in grouped[group_id]),
            group_id,
        ),
    )
    selected_ids = set(ranked_groups[:group_budget])
    return [row for row in enriched if str(row["group_id"]) in selected_ids]


def build_measurement_request(
    candidate: Mapping[str, Any], capability_profiles: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    profiles = {
        profile["capability_profile_id"]: validate_capability_profile(profile)
        for profile in capability_profiles
    }
    profile_id = str(candidate["capability_profile_id"])
    if profile_id not in profiles:
        raise ValueError(f"unknown capability profile: {profile_id}")
    q_mode = str(candidate["q_mode"])
    if q_mode not in {"fp16", "int8"}:
        raise ValueError("measurement request q_mode must be fp16 or int8")
    return {
        "candidate_id": str(candidate.get("row_id") or candidate.get("job_id")),
        "logical_genome": [*candidate["width"], q_mode],
        "strategy_id": f"q={q_mode}",
        "q_mode": q_mode,
        "capability_profile_id": profile_id,
        "capability_digest": profiles[profile_id]["capability_digest"],
        "dispatch_key": profiles[profile_id]["dispatch_key"],
    }


def apply_measurement_feedback(
    bundle: Mapping[str, Any],
    measured_rows: Sequence[Mapping[str, Any]],
    capability_profiles: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return update_model_bundle(bundle, measured_rows, capability_profiles)


def best_q_by_profile(
    predicted_rows: Sequence[Mapping[str, Any]], *, objective: str
) -> dict[str, dict[str, Any]]:
    maximize = objective == "ap70"
    decisions: dict[str, dict[str, Any]] = {}
    for source in predicted_rows:
        row = copy.deepcopy(dict(source))
        value = float(row["predictions"][objective])
        profile_id = str(row["capability_profile_id"])
        current = decisions.get(profile_id)
        is_better = (
            value > current["predicted_value"]
            if maximize and current is not None
            else current is None or value < current["predicted_value"]
        )
        if is_better:
            decisions[profile_id] = {
                "q_mode": row["q_mode"],
                "predicted_value": value,
                "row_id": row.get("row_id"),
            }
    return decisions


def unconditional_int8_regret(
    predicted_rows: Sequence[Mapping[str, Any]], *, objective: str
) -> dict[str, dict[str, float]]:
    maximize = objective == "ap70"
    by_profile: dict[str, list[Mapping[str, Any]]] = {}
    for row in predicted_rows:
        profile_id = str(row["capability_profile_id"])
        by_profile[profile_id] = [*by_profile.get(profile_id, []), row]
    report = {}
    for profile_id, rows in by_profile.items():
        selector = max if maximize else min
        best = selector(float(row["predictions"][objective]) for row in rows)
        int8 = selector(
            float(row["predictions"][objective]) for row in rows if row["q_mode"] == "int8"
        )
        relative_regret = (best - int8) / best if maximize else int8 / best - 1.0
        report[profile_id] = {
            "best_value": best,
            "unconditional_int8_value": int8,
            "relative_regret": relative_regret,
        }
    return report


__all__ = [
    "apply_measurement_feedback",
    "best_q_by_profile",
    "build_measurement_request",
    "crowding_distances",
    "dominates",
    "nondominated_ranks",
    "select_candidates",
    "select_candidate_groups",
    "unconditional_int8_regret",
]
