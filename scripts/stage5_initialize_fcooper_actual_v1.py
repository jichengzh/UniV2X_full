#!/usr/bin/env python3
"""Initialize the F-Cooper TRT/H800 B=4,T=16 actual-feedback search."""

from __future__ import annotations

import argparse
import hashlib
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


def rows(payload: Any, field: str) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(item) for item in payload]
    if isinstance(payload, Mapping) and isinstance(payload.get(field), list):
        return [dict(item) for item in payload[field]]
    raise ValueError(f"expected list or object containing {field}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage4-dir",
        type=Path,
        default=REPO_ROOT / "results/stage4_p1_p3_closure_v1_20260716",
    )
    parser.add_argument(
        "--coldstart-root",
        type=Path,
        default=(
            REPO_ROOT
            / "results/stage35_gold144_targeted_supplement_v2_20260714/final_gold176_v1"
        ),
    )
    parser.add_argument(
        "--profiles-json",
        type=Path,
        default=REPO_ROOT
        / "results/s1_profile_final_v3_20260711/capability_profiles_v3.json",
    )
    parser.add_argument("--source-registry-json", type=Path, required=True)
    parser.add_argument("--frozen-contract-json", type=Path, required=True)
    parser.add_argument("--probe-audit-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260723)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cold_rows_path = args.coldstart_root / "gold176_final.json"
    cold_graph_path = args.coldstart_root / "graph_features.json"
    cold_audit = verify_frozen_coldstart_artifacts(cold_rows_path, cold_graph_path)
    training_rows = freeze_initial_coldstart(
        rows(json.loads(cold_rows_path.read_text()), "rows")
    )
    graph_features = rows(
        json.loads(cold_graph_path.read_text()), "graph_features"
    )
    profiles = rows(
        json.loads(args.profiles_json.read_text()), "capability_profiles"
    )
    profile = next(row for row in profiles if row["dispatch_key"] == "trt_engine")
    closure = json.loads(
        (args.stage4_dir / "stage4_p1_p3_closure_audit.json").read_text()
    )
    frozen_contract = json.loads(args.frozen_contract_json.read_text())
    probe_audit = json.loads(args.probe_audit_json.read_text())
    if probe_audit.get("all_probes_terminal") is not True:
        raise ValueError("F-Cooper search cannot start before all probes are terminal")
    registry = json.loads(args.source_registry_json.read_text())
    bundle = fit_production_bundle(
        training_rows,
        graph_features,
        profiles,
        closure,
        seed=args.seed,
        training_view_policy="initial_coldstart_only",
    )
    bundle.model_anchors["fcooper"] = float(frozen_contract["ap70_ref"])
    task = SearchTask("S5-FCO-TRT", "fcooper", "h800", profile)
    task_contract = validate_search_task(task)
    probe_row_ids = {
        str(row["row_id"])
        for row in probe_audit.get("rows", [])
        if row.get("row_id")
    }
    manifest = build_task_candidate_manifest(
        registry, task=task, measured_row_ids=probe_row_ids
    )
    predicted = predict_candidate_rows(bundle, manifest["rows"], profiles)
    selection = select_task_batch(
        predicted, training_rows, graph_features, task=task
    )
    request = build_measurement_request(
        task=task, selected_rows=selection["selected_rows"], round_index=0
    )
    root = args.output_dir
    write_json(root / "task_contract.json", task_contract)
    write_json(root / "candidate_manifest.json", manifest)
    write_json(root / "round_00/predicted_candidates.json", {"rows": predicted})
    write_json(root / "round_00/acquisition.json", selection)
    write_json(root / "round_00/measurement_request.json", request)
    summary = {
        "schema_version": "stage5_fcooper_actual_v1_initialization_summary",
        "task_id": task.task_id,
        "training_view_policy": "initial_coldstart_only",
        "coldstart_rows": len(training_rows),
        "coldstart_audit": cold_audit,
        "cross_model_online_labels_loaded": False,
        "probe_rows_excluded_from_online_budget": len(probe_row_ids),
        "source_registry_sha256": sha256_file(args.source_registry_json),
        "eligible_genomes": manifest["eligible_row_count"],
        "selected_row_ids": selection["selected_row_ids"],
        "measurement_request_sha256": request["measurement_request_sha256"],
    }
    write_json(root / "initialization_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
