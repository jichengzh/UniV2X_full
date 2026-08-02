#!/usr/bin/env python3
"""Append an atomic feedback batch and emit the next independent task batch."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from framework.stage5.production_search_v1 import predict_candidate_rows
from framework.stage5.single_target_search_v2 import (
    SearchTask,
    build_measurement_request,
    build_task_candidate_manifest,
    fit_online_bundle,
    freeze_initial_coldstart,
    select_task_batch,
    validate_task_feedback_history,
    verify_frozen_coldstart_artifacts,
)
from framework.stage5.source_registry_v1 import validate_full_source_registry


DEFAULT_COLDSTART_ROOT = (
    REPO_ROOT
    / "results/stage35_gold144_targeted_supplement_v2_20260714/final_gold176_v1"
)
TASK_SPECS = {
    "S5-PYR-TVM": ("pyramid", "tvm_auto"),
    "S5-PYR-TRT": ("pyramid", "trt_engine"),
    "S5-COD-TVM": ("codriving", "tvm_auto"),
    "S5-COD-TRT": ("codriving", "trt_engine"),
}


def _rows(payload: Any, field: str) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(item) for item in payload]
    if isinstance(payload, Mapping) and isinstance(payload.get(field), list):
        return [dict(item) for item in payload[field]]
    raise ValueError(f"expected list or object containing {field}")


def _write(path: Path, payload: Any) -> None:
    content = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.is_file() and path.read_text(encoding="utf-8") != content:
        raise ValueError(f"refusing to overwrite drifted round checkpoint: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--feedback-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--round-index", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument(
        "--stage4-dir",
        type=Path,
        default=REPO_ROOT / "results/stage4_p1_p3_closure_v1_20260716",
    )
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
    parser.add_argument(
        "--source-registry-json",
        type=Path,
        default=REPO_ROOT / "results/stage5_single_target_search_v2_gold176_20260718/candidate_source_registry_full.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.round_index not in {1, 2, 3}:
        raise ValueError("next round index must be 1, 2, or 3")
    if args.task_id not in TASK_SPECS:
        raise ValueError("task-id is not one of the four frozen Stage5 tasks")
    verify_frozen_coldstart_artifacts(
        args.coldstart_rows_json, args.coldstart_graph_features_json
    )
    initial = freeze_initial_coldstart(
        _rows(json.loads(args.coldstart_rows_json.read_text()), "rows")
    )
    base_graphs = _rows(
        json.loads(args.coldstart_graph_features_json.read_text()),
        "graph_features",
    )
    feedback = _rows(json.loads(args.feedback_json.read_text()), "rows")
    profiles = _rows(json.loads(args.profiles_json.read_text()), "capability_profiles")
    model, dispatch = TASK_SPECS[args.task_id]
    profile = next(item for item in profiles if item["dispatch_key"] == dispatch)
    task = SearchTask(
        args.task_id,
        model,
        "h800",
        profile,
    )
    validate_task_feedback_history(
        feedback, task=task, completed_rounds=args.round_index
    )
    training = [*initial, *feedback]
    graph_by_group = {str(row["group_id"]): dict(row) for row in base_graphs}
    for row in feedback:
        graph_by_group[str(row["group_id"])] = dict(row["graph_features"])
    graphs = list(graph_by_group.values())
    bundle = fit_online_bundle(
        training, graphs, profiles, seed=args.seed + args.round_index
    )
    measured_ids = {
        str(row.get("manifest_job_id") or row.get("row_id")) for row in training
    }
    registry = json.loads(args.source_registry_json.read_text())
    validate_full_source_registry(registry)
    manifest = build_task_candidate_manifest(
        registry,
        task=task,
        measured_row_ids=measured_ids,
    )
    predicted = predict_candidate_rows(bundle, manifest["rows"], profiles)
    selection = select_task_batch(predicted, training, graphs, task=task)
    request = build_measurement_request(
        task=task, selected_rows=selection["selected_rows"], round_index=args.round_index
    )
    round_dir = args.output_dir / args.task_id / f"round_{args.round_index:02d}"
    _write(round_dir / "candidate_manifest.json", manifest)
    _write(round_dir / "predicted_candidates.json", {"rows": predicted})
    _write(round_dir / "acquisition.json", selection)
    _write(round_dir / "measurement_request.json", request)
    state = {
        "schema_version": "stage5_task_round_state_v2",
        "task_id": args.task_id,
        "round_index": args.round_index,
        "completed_feedback_rows": len(feedback),
        "budget_consumed": len(feedback),
        "budget_remaining_after_next_batch": 16 - len(feedback) - 4,
        "status": "awaiting_real_measurement",
        "selected_row_ids": selection["selected_row_ids"],
        "measurement_request_sha256": request["measurement_request_sha256"],
    }
    _write(round_dir / "round_state.json", state)
    print(json.dumps(state, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
