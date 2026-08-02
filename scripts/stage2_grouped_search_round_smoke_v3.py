#!/usr/bin/env python3
"""Replay one complete-group canonical acquisition and feedback round."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from framework.stage2.canonical_search_v3 import build_capability_profile
from framework.stage2.cost_model_bundle_v3 import fit_model_bundle, predict_rows
from framework.stage2.search_loop_v3 import (
    apply_measurement_feedback,
    nondominated_ranks,
    select_candidate_groups,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _profiles(s1_report: dict[str, Any]) -> list[dict[str, Any]]:
    common = {
        "supports_fp16_tensorcore": 1.0,
        "supports_int8_tensorcore": 1.0,
        "int8_channel_alignment": 16.0,
        "fp16_channel_alignment": 8.0,
    }
    specs = (
        ("tvm-profile", "tvm_auto", s1_report["s1_features"]["tvm-profile"]),
        ("trt-profile", "trt_engine", s1_report["s1_features"]["trt-profile"]),
    )
    return [
        build_capability_profile(
            capability_profile_id=profile_id,
            hardware_target="h800",
            compiler_fingerprint=_sha(profile_id),
            dispatch_key=dispatch_key,
            features={**common, **features},
        )
        for profile_id, dispatch_key, features in specs
    ]


def _rows(table: dict[str, Any]) -> list[dict[str, Any]]:
    profile_map = {
        "h800_tvm_routeb_20260708": "tvm-profile",
        "h800_trt_20260708": "trt-profile",
    }
    rows = []
    for source in table["rows"]:
        profile_id = profile_map.get(source.get("compiler_profile_id"))
        if profile_id is None or source.get("q_mode") not in {"fp16", "int8"}:
            continue
        if source.get("mixed_policy_id") not in {"none", "all_eligible_conv"}:
            continue
        if not all(isinstance(source.get(name), (int, float)) for name in ("latency_ms", "energy_j")):
            continue
        width_text = str(source["width"])
        rows.append(
            {
                "row_id": f"{profile_id}|{width_text}|{source['q_mode']}",
                "group_id": f"codriving|{width_text}",
                "width": [int(value) for value in width_text.split("x")],
                "q_mode": source["q_mode"],
                "capability_profile_id": profile_id,
                "graph_features": {},
                "build_status": "success",
                "numerical_status": "pass",
                "latency_ms": float(source["latency_ms"]),
                "energy_j": float(source["energy_j"]),
            }
        )
    groups = {row["group_id"] for row in rows}
    if len(rows) != 48 or len(groups) != 12:
        raise ValueError("grouped smoke requires 12 complete four-row groups")
    if any(sum(row["group_id"] == group for row in rows) != 4 for group in groups):
        raise ValueError("every group must contain TVM/TRT x FP16/INT8")
    return rows


def _mape(predicted: list[dict[str, Any]], objective: str) -> float:
    return sum(
        abs(float(row["predictions"][objective]) - float(row[objective]))
        / float(row[objective])
        for row in predicted
    ) / len(predicted)


def _choice_accuracy(predicted: list[dict[str, Any]]) -> float:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in predicted:
        key = (row["group_id"], row["capability_profile_id"])
        grouped.setdefault(key, []).append(row)
    correct = 0
    for pair in grouped.values():
        actual = min(pair, key=lambda row: row["latency_ms"])["q_mode"]
        predicted_q = min(pair, key=lambda row: row["predictions"]["latency_ms"])["q_mode"]
        correct += actual == predicted_q
    return correct / len(grouped)


def _actual_pareto(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranks = nondominated_ranks([(row["latency_ms"], row["energy_j"]) for row in rows])
    return [{**row, "pareto_rank": ranks[index]} for index, row in enumerate(rows)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--s1-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = _rows(json.loads(args.table.read_text(encoding="utf-8")))
    profiles = _profiles(json.loads(args.s1_report.read_text(encoding="utf-8")))
    groups = sorted({row["group_id"] for row in rows}, key=_sha)
    initial_groups = set(groups[:3])
    acquisition_pool_groups = set(groups[3:8])
    holdout_groups = set(groups[8:])
    initial_rows = [row for row in rows if row["group_id"] in initial_groups]
    pool_rows = [row for row in rows if row["group_id"] in acquisition_pool_groups]
    holdout_rows = [row for row in rows if row["group_id"] in holdout_groups]

    initial_bundle = fit_model_bundle(initial_rows, profiles, ridge=1e-4)
    pool_predictions = predict_rows(initial_bundle, pool_rows, profiles)
    holdout_before = predict_rows(initial_bundle, holdout_rows, profiles)
    acquired_predictions = select_candidate_groups(
        pool_predictions,
        measured_group_ids=initial_groups,
        group_budget=2,
        objectives=("latency_ms", "energy_j"),
    )
    acquired_groups = {row["group_id"] for row in acquired_predictions}
    acquired_rows = [row for row in rows if row["group_id"] in acquired_groups]
    updated_bundle = apply_measurement_feedback(initial_bundle, acquired_rows, profiles)
    holdout_after = predict_rows(updated_bundle, holdout_rows, profiles)
    pareto_before = _actual_pareto(initial_rows)
    pareto_after = _actual_pareto([*initial_rows, *acquired_rows])
    acquired_frontier_rows = [
        row for row in pareto_after if row["group_id"] in acquired_groups and row["pareto_rank"] == 0
    ]
    gates = {
        "initial_has_three_complete_groups": len(initial_rows) == 12,
        "acquisition_selected_two_complete_groups": len(acquired_rows) == 8
        and len(acquired_groups) == 2,
        "feedback_updated_in_one_batch": updated_bundle["training_row_count"] == 20,
        "holdout_groups_never_acquired": not (holdout_groups & acquired_groups),
        "pareto_recomputed_after_feedback": len(pareto_after) == 20,
    }
    report = {
        "schema_version": "stage2_grouped_search_round_smoke_v3",
        "mode": "real_measurement_replay_not_causal_dispatch",
        "stage2_grouped_search_behavior_closed": all(gates.values()),
        "stage2_method_validated": False,
        "groups": {
            "initial": sorted(initial_groups),
            "acquisition_pool": sorted(acquisition_pool_groups),
            "acquired": sorted(acquired_groups),
            "locked_holdout": sorted(holdout_groups),
        },
        "row_counts": {
            "initial": len(initial_rows),
            "acquired_batch": len(acquired_rows),
            "updated_training": updated_bundle["training_row_count"],
            "holdout": len(holdout_rows),
        },
        "holdout_metrics": {
            "before": {
                "latency_mape": _mape(holdout_before, "latency_ms"),
                "energy_mape": _mape(holdout_before, "energy_j"),
                "precision_choice_accuracy": _choice_accuracy(holdout_before),
            },
            "after": {
                "latency_mape": _mape(holdout_after, "latency_ms"),
                "energy_mape": _mape(holdout_after, "energy_j"),
                "precision_choice_accuracy": _choice_accuracy(holdout_after),
            },
        },
        "pareto": {
            "frontier_rows_before": sum(row["pareto_rank"] == 0 for row in pareto_before),
            "frontier_rows_after": sum(row["pareto_rank"] == 0 for row in pareto_after),
            "acquired_rows_entering_frontier": len(acquired_frontier_rows),
        },
        "closure_gates": gates,
    }
    if not report["stage2_grouped_search_behavior_closed"]:
        raise RuntimeError("grouped search round closure gate failed")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
