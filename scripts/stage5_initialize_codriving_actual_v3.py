#!/usr/bin/env python3
"""Initialize fresh CoDriving TVM/TRT actual-feedback v3 searches."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


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
    freeze_initial_coldstart as _freeze_initial_coldstart,
    select_task_batch,
    validate_search_task,
    verify_frozen_coldstart_artifacts,
)
from framework.stage5.source_registry_v1 import validate_full_source_registry  # noqa: E402


TASK_SPECS = (
    ("S5-COD-TVM", "codriving", "tvm_auto"),
    ("S5-COD-TRT", "codriving", "trt_engine"),
)
TRAINING_VIEW_POLICY = "initial_coldstart_only"
DEFAULT_OUTPUT = REPO_ROOT / "results/stage5_codriving_actual_v3_20260721"
DEFAULT_COLDSTART_ROOT = (
    REPO_ROOT
    / "results/stage35_gold144_targeted_supplement_v2_20260714/final_gold176_v1"
)
DEFAULT_REGISTRY = (
    REPO_ROOT
    / "results/stage5_single_target_search_v2_gold176_20260718"
    / "candidate_source_registry_full.json"
)


def freeze_initial_coldstart(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Expose the reviewed cold-start gate for entry-level contract tests."""
    return _freeze_initial_coldstart(rows)


def _rows(payload: Any, field: str) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(item) for item in payload]
    if isinstance(payload, Mapping) and isinstance(payload.get(field), list):
        return [dict(item) for item in payload[field]]
    raise ValueError(f"expected list or object containing {field}")


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    content = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.is_file() and path.read_text(encoding="utf-8") != content:
        raise ValueError(f"refusing to overwrite drifted checkpoint: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--source-registry-json", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260721)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    coldstart_audit = verify_frozen_coldstart_artifacts(
        args.coldstart_rows_json, args.coldstart_graph_features_json
    )
    training_rows = freeze_initial_coldstart(
        _rows(json.loads(args.coldstart_rows_json.read_text()), "rows")
    )
    graph_features = _rows(
        json.loads(args.coldstart_graph_features_json.read_text()), "graph_features"
    )
    closure = json.loads(
        (args.stage4_dir / "stage4_p1_p3_closure_audit.json").read_text()
    )
    profiles = _rows(json.loads(args.profiles_json.read_text()), "capability_profiles")
    registry = json.loads(args.source_registry_json.read_text())
    registry_audit = validate_full_source_registry(registry)
    profiles_by_dispatch = {str(row["dispatch_key"]): row for row in profiles}
    bundle = fit_production_bundle(
        training_rows,
        graph_features,
        profiles,
        closure,
        seed=args.seed,
        training_view_policy=TRAINING_VIEW_POLICY,
    )
    measured_ids = {
        str(row.get("manifest_job_id") or row.get("row_id")) for row in training_rows
    }
    summary: dict[str, Any] = {
        "schema_version": "stage5_codriving_actual_v3_initialization_summary_v1",
        "training_view_policy": TRAINING_VIEW_POLICY,
        "frozen_training_rows": len(training_rows),
        "frozen_online_feedback_rows": 0,
        "v2_online_labels_loaded": False,
        "coldstart_artifact_audit": coldstart_audit,
        "source_registry_sha256": _sha_file(args.source_registry_json),
        "source_registry_audit": registry_audit,
        "full_search_universe_per_task": 686,
        "batch_size": 4,
        "sample_budget": 16,
        "round_count": 4,
        "tasks": [],
    }
    for task_id, model, dispatch in TASK_SPECS:
        task = SearchTask(task_id, model, "h800", profiles_by_dispatch[dispatch])
        contract = validate_search_task(task)
        manifest = build_task_candidate_manifest(
            registry, task=task, measured_row_ids=measured_ids
        )
        predicted = predict_candidate_rows(bundle, manifest["rows"], profiles)
        selection = select_task_batch(
            predicted, training_rows, graph_features, task=task
        )
        request = build_measurement_request(
            task=task, selected_rows=selection["selected_rows"], round_index=0
        )
        task_root = args.output_dir / task_id
        round_root = task_root / "round_00"
        _write_json(task_root / "task_contract.json", contract)
        _write_json(task_root / "candidate_manifest.json", manifest)
        _write_json(round_root / "predicted_candidates.json", {"rows": predicted})
        _write_json(round_root / "acquisition.json", selection)
        _write_json(round_root / "measurement_request.json", request)
        summary["tasks"].append(
            {
                "task_id": task_id,
                "target_model": model,
                "dispatch_key": dispatch,
                "eligible_row_count": manifest["eligible_row_count"],
                "selected_row_ids": selection["selected_row_ids"],
                "measurement_request_sha256": request["measurement_request_sha256"],
            }
        )
    _write_json(
        args.output_dir / "codriving_actual_v3_initialization_summary.json", summary
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
