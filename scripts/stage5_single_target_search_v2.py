#!/usr/bin/env python3
"""Initialize four independent Stage5 v2 tasks and emit round-0 requests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from framework.stage5.production_search_v1 import (  # noqa: E402
    fit_production_bundle,
    predict_candidate_rows,
)
from framework.stage5.single_target_search_v2 import (  # noqa: E402
    SearchTask,
    build_measurement_request,
    build_task_candidate_manifest,
    freeze_initial_coldstart,
    select_task_batch,
    validate_search_task,
    verify_frozen_coldstart_artifacts,
)
from framework.stage5.source_registry_v1 import validate_full_source_registry  # noqa: E402


DEFAULT_STAGE4 = REPO_ROOT / "results/stage4_p1_p3_closure_v1_20260716"
DEFAULT_OUTPUT = REPO_ROOT / "results/stage5_single_target_search_v2_gold176_20260718"
DEFAULT_REGISTRY = DEFAULT_OUTPUT / "candidate_source_registry_full.json"
DEFAULT_COLDSTART_ROOT = (
    REPO_ROOT
    / "results/stage35_gold144_targeted_supplement_v2_20260714/final_gold176_v1"
)


def _rows(payload: Any, field: str) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(item) for item in payload]
    if isinstance(payload, Mapping) and isinstance(payload.get(field), list):
        return [dict(item) for item in payload[field]]
    raise ValueError(f"expected list or object containing {field}")


def _write_json(path: Path, payload: Any) -> None:
    content = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    if path.is_file() and path.read_bytes() != content:
        raise ValueError(f"refusing to overwrite drifted checkpoint: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(content)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage4-dir", type=Path, default=DEFAULT_STAGE4)
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
    parser.add_argument("--source-registry-json", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument(
        "--task-id",
        choices=("S5-PYR-TVM", "S5-PYR-TRT", "S5-COD-TVM", "S5-COD-TRT"),
        help="Emit one task only; default emits all four independent tasks.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    coldstart_artifact_audit = verify_frozen_coldstart_artifacts(
        args.coldstart_rows_json, args.coldstart_graph_features_json
    )
    training_rows = freeze_initial_coldstart(
        _rows(json.loads(args.coldstart_rows_json.read_text()), "rows")
    )
    graph_features = _rows(
        json.loads(args.coldstart_graph_features_json.read_text()),
        "graph_features",
    )
    closure = json.loads((args.stage4_dir / "stage4_p1_p3_closure_audit.json").read_text())
    profiles = _rows(json.loads(args.profiles_json.read_text()), "capability_profiles")
    registry = json.loads(args.source_registry_json.read_text())
    registry_audit = validate_full_source_registry(registry)
    profile_by_dispatch = {str(profile["dispatch_key"]): profile for profile in profiles}
    task_specs = (
        ("S5-PYR-TVM", "pyramid", "tvm_auto"),
        ("S5-PYR-TRT", "pyramid", "trt_engine"),
        ("S5-COD-TVM", "codriving", "tvm_auto"),
        ("S5-COD-TRT", "codriving", "trt_engine"),
    )
    tasks = [
        SearchTask(task_id, model, "h800", profile_by_dispatch[dispatch])
        for task_id, model, dispatch in task_specs
        if args.task_id is None or task_id == args.task_id
    ]
    bundle = fit_production_bundle(
        training_rows,
        graph_features,
        profiles,
        closure,
        seed=args.seed,
        training_view_policy="initial_coldstart_only",
    )
    measured_ids = {
        str(row.get("manifest_job_id") or row.get("row_id")) for row in training_rows
    }
    summary = {
        "schema_version": "stage5_four_independent_task_dryrun_gold176_v2",
        "frozen_training_rows": len(training_rows),
        "frozen_initial_coldstart_rows": sum(
            row.get("training_source") == "initial_coldstart" for row in training_rows
        ),
        "frozen_online_feedback_rows": 0,
        "coldstart_artifact_audit": coldstart_artifact_audit,
        "excluded_pre_stage5_smoke_rows": 8,
        "full_search_universe_per_task": 686,
        "source_registry_audit": registry_audit,
        "tasks": [],
    }
    for task in tasks:
        contract = validate_search_task(task)
        manifest = build_task_candidate_manifest(
            registry,
            task=task,
            measured_row_ids=measured_ids,
        )
        predicted = predict_candidate_rows(bundle, manifest["rows"], profiles)
        selection = select_task_batch(
            predicted, training_rows, graph_features, task=task
        )
        request = build_measurement_request(
            task=task, selected_rows=selection["selected_rows"], round_index=0
        )
        task_dir = args.output_dir / task.task_id / "round_00"
        _write_json(args.output_dir / task.task_id / "task_contract.json", contract)
        _write_json(args.output_dir / task.task_id / "candidate_manifest.json", manifest)
        _write_json(task_dir / "predicted_candidates.json", {"rows": predicted})
        _write_json(task_dir / "acquisition.json", selection)
        _write_json(task_dir / "measurement_request.json", request)
        summary["tasks"].append(
            {
                "task_id": task.task_id,
                "eligible_row_count": manifest["eligible_row_count"],
                "selected_row_ids": selection["selected_row_ids"],
                "selected_q_modes": [row["q_mode"] for row in selection["selected_rows"]],
                "measurement_request_sha256": request["measurement_request_sha256"],
            }
        )
    _write_json(args.output_dir / "dryrun_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
