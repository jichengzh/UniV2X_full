#!/usr/bin/env python3
"""Gate Stage5 v2 with hidden-label, single-genome acquisition replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from framework.stage5.production_search_v1 import (  # noqa: E402
    _frontier_indices,
    _hypervolume_3d,
    predict_candidate_rows,
)
from framework.stage5.single_target_search_v2 import (  # noqa: E402
    SearchTask,
    fit_online_bundle,
    freeze_initial_coldstart,
    select_task_batch,
    validate_search_task,
    verify_frozen_coldstart_artifacts,
)


DEFAULT_COLDSTART_ROOT = (
    REPO_ROOT
    / "results/stage35_gold144_targeted_supplement_v2_20260714/final_gold176_v1"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "results/stage5_single_target_search_v2_gold176_20260718/compatibility_replay.json"
)


def _rows(payload: Any, field: str) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(item) for item in payload]
    if isinstance(payload, Mapping) and isinstance(payload.get(field), list):
        return [dict(item) for item in payload[field]]
    raise ValueError(f"expected list or object containing {field}")


def _row_id(row: Mapping[str, Any]) -> str:
    return str(row.get("manifest_job_id") or row.get("row_id") or "")


def _score(selected: list[dict[str, Any]], pool: list[dict[str, Any]]) -> dict[str, float]:
    pool_raw = np.asarray(
        [
            (float(row["latency_ms"]), float(row["energy_j"]), -float(row["ap70"]))
            for row in pool
        ],
        dtype=float,
    )
    low = np.min(pool_raw, axis=0)
    span = np.maximum(np.max(pool_raw, axis=0) - low, 1e-12)
    selected_raw = np.asarray(
        [
            (float(row["latency_ms"]), float(row["energy_j"]), -float(row["ap70"]))
            for row in selected
        ],
        dtype=float,
    )
    normalized = (selected_raw - low) / span
    hv = _hypervolume_3d(
        [tuple(map(float, point)) for point in normalized], (1.1, 1.1, 1.1)
    )
    pool_frontier = {
        _row_id(pool[index])
        for index in _frontier_indices([tuple(map(float, row)) for row in pool_raw])
    }
    recall = len(pool_frontier & {_row_id(row) for row in selected}) / max(
        len(pool_frontier), 1
    )
    return {"hypervolume": float(hv), "pareto_recall": float(recall)}


def _candidate(row: Mapping[str, Any], task: SearchTask, graph: Mapping[str, Any]) -> dict:
    contract = validate_search_task(task)
    row_id = _row_id(row)
    return {
        "schema_version": "stage5_candidate_row_v2",
        "task_id": task.task_id,
        "task_sha256": contract["task_sha256"],
        "row_id": row_id,
        "manifest_job_id": row_id,
        "group_id": row["group_id"],
        "model": row["model"],
        "width": list(row["width"]),
        "genome": [*row["width"], row["q_mode"]],
        "strategy_id": f"q={row['q_mode']}",
        "q_mode": row["q_mode"],
        "hardware_id": task.hardware_id,
        "capability_profile_id": row["capability_profile_id"],
        "dispatch_key": row["dispatch_key"],
        "graph_features": dict(graph),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coldstart-rows-json",
        type=Path,
        default=DEFAULT_COLDSTART_ROOT / "gold176_final.json",
    )
    parser.add_argument(
        "--coldstart-graph-features-json",
        type=Path,
        default=DEFAULT_COLDSTART_ROOT / "graph_features.json",
    )
    parser.add_argument(
        "--profiles-json",
        type=Path,
        default=REPO_ROOT / "results/s1_profile_final_v3_20260711/capability_profiles_v3.json",
    )
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--random-repeats", type=int, default=128)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    coldstart_artifact_audit = verify_frozen_coldstart_artifacts(
        args.coldstart_rows_json, args.coldstart_graph_features_json
    )
    rows = freeze_initial_coldstart(
        _rows(json.loads(args.coldstart_rows_json.read_text()), "rows")
    )
    graphs = _rows(
        json.loads(args.coldstart_graph_features_json.read_text()),
        "graph_features",
    )
    profiles = _rows(json.loads(args.profiles_json.read_text()), "capability_profiles")
    graph_by_group = {str(graph["group_id"]): graph for graph in graphs}
    profile_by_dispatch = {str(profile["dispatch_key"]): profile for profile in profiles}
    specs = (
        ("S5-PYR-TVM", "pyramid", "tvm_auto"),
        ("S5-PYR-TRT", "pyramid", "trt_engine"),
        ("S5-COD-TVM", "codriving", "tvm_auto"),
        ("S5-COD-TRT", "codriving", "trt_engine"),
    )
    reports = []
    for task_index, (task_id, model, dispatch) in enumerate(specs):
        task = SearchTask(task_id, model, "h800", profile_by_dispatch[dispatch])
        target_rows = [
            dict(row)
            for row in rows
            if row.get("model") == model
            and row.get("dispatch_key") == dispatch
            and row.get("terminal_status") == "measured_success_gold"
        ]
        ordered_target = sorted(
            target_rows,
            key=lambda row: hashlib.sha256(
                f"{args.seed}|{task_id}|{_row_id(row)}".encode()
            ).hexdigest(),
        )
        pool = ordered_target[:24]
        heldout_genomes = {
            (str(row["model"]), tuple(row["width"]), str(row["q_mode"]))
            for row in pool
        }
        training = [
            dict(row)
            for row in rows
            if (str(row["model"]), tuple(row["width"]), str(row["q_mode"]))
            not in heldout_genomes
        ]
        available = {
            _row_id(row): _candidate(row, task, graph_by_group[str(row["group_id"])])
            for row in pool
        }
        selected: list[dict[str, Any]] = []
        q_prediction_delta_count = 0
        rounds = []
        for round_index in range(4):
            bundle = fit_online_bundle(
                training, graphs, profiles, seed=args.seed + task_index * 10 + round_index
            )
            predicted = predict_candidate_rows(bundle, list(available.values()), profiles)
            by_group: dict[str, dict[str, dict[str, float]]] = {}
            for row in predicted:
                by_group.setdefault(str(row["group_id"]), {})[str(row["q_mode"])] = row[
                    "predictions"
                ]
            q_prediction_delta_count += sum(
                any(abs(pair["fp16"][target] - pair["int8"][target]) > 1e-12 for target in pair["fp16"])
                for pair in by_group.values()
                if set(pair) == {"fp16", "int8"}
            )
            acquisition = select_task_batch(
                predicted, training, graphs, task=task
            )
            chosen_ids = acquisition["selected_row_ids"]
            truth = [next(row for row in pool if _row_id(row) == row_id) for row_id in chosen_ids]
            selected.extend(truth)
            training.extend({**row, "training_source": "online_feedback"} for row in truth)
            for row_id in chosen_ids:
                available.pop(row_id)
            rounds.append(
                {
                    "round_index": round_index,
                    "selected_row_ids": chosen_ids,
                    "selected_q_modes": [row["q_mode"] for row in truth],
                }
            )
        acquisition_score = _score(selected, pool)
        random_scores = []
        for repeat in range(args.random_repeats):
            rng = np.random.default_rng(args.seed + task_index * 1000 + repeat)
            indices = rng.choice(len(pool), size=16, replace=False)
            random_scores.append(_score([pool[int(index)] for index in indices], pool))
        random_hv = [score["hypervolume"] for score in random_scores]
        random_recall = [score["pareto_recall"] for score in random_scores]
        reports.append(
            {
                "task_id": task_id,
                "hidden_label_pool_rows": len(pool),
                "selected_rows": len(selected),
                "selected_q_modes": {
                    q: sum(row["q_mode"] == q for row in selected) for q in ("fp16", "int8")
                },
                "q_mode_prediction_delta_pair_count": q_prediction_delta_count,
                "acquisition": acquisition_score,
                "random_median": {
                    "hypervolume": float(np.median(random_hv)),
                    "pareto_recall": float(np.median(random_recall)),
                },
                "random_p25_p75": {
                    "hypervolume": [
                        float(np.quantile(random_hv, 0.25)),
                        float(np.quantile(random_hv, 0.75)),
                    ],
                    "pareto_recall": [
                        float(np.quantile(random_recall, 0.25)),
                        float(np.quantile(random_recall, 0.75)),
                    ],
                },
                "hypervolume_vs_random_median_ratio": acquisition_score["hypervolume"]
                / max(float(np.median(random_hv)), 1e-12),
                "gate_hv_ge_random_median_and_recall_ge_random_p25": (
                    acquisition_score["hypervolume"] >= float(np.median(random_hv))
                    and acquisition_score["pareto_recall"]
                    >= float(np.quantile(random_recall, 0.25))
                ),
                "rounds": rounds,
            }
        )
    payload = {
        "schema_version": "stage5_single_genome_compatibility_replay_v2",
        "protocol": "blocked_24_genome_holdout_excluding_same_genome_all_profiles_4x4_budget",
        "training_source_contract": "standalone_gold176_initial_coldstart_only",
        "input_row_count": len(rows),
        "stage5_online_budget_consumed": 0,
        "coldstart_artifact_audit": coldstart_artifact_audit,
        "random_repeats": args.random_repeats,
        "tasks": reports,
        "q_mode_participates": all(
            report["q_mode_prediction_delta_pair_count"] > 0 for report in reports
        ),
        "acquisition_gate_passed": all(
            report["gate_hv_ge_random_median_and_recall_ge_random_p25"]
            for report in reports
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0 if payload["q_mode_participates"] and payload["acquisition_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
