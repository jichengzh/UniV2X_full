#!/usr/bin/env python3
"""Advance one atomic F-Cooper TRT/H800 actual-feedback search round."""

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

from framework.stage5.fcooper_space_v1 import WIDTH_SCHEMA  # noqa: E402
from framework.stage5.production_search_v1 import predict_candidate_rows  # noqa: E402
from framework.stage5.single_target_search_v2 import (  # noqa: E402
    SearchTask,
    build_measurement_request,
    build_task_candidate_manifest,
    fit_online_bundle,
    freeze_initial_coldstart,
    select_task_batch,
    validate_task_feedback_history,
    verify_frozen_coldstart_artifacts,
)


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


def _sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def _write_frozen(path: Path, payload: Any) -> None:
    content = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.is_file() and path.read_text(encoding="utf-8") != content:
        raise ValueError(f"refusing to overwrite drifted round checkpoint: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(content, encoding="utf-8")


def validate_fcooper_registry(registry: Mapping[str, Any]) -> dict[str, Any]:
    schema = list(registry.get("width_schema") or [])
    if registry.get("model") != "fcooper":
        raise ValueError("registry model must be fcooper")
    if schema != list(WIDTH_SCHEMA):
        raise ValueError("registry must use the scanner-derived five-width schema")
    groups = [dict(row) for row in registry.get("groups") or []]
    if not groups:
        raise ValueError("F-Cooper registry cannot be empty")
    group_ids = [str(row.get("group_id")) for row in groups]
    if len(set(group_ids)) != len(group_ids):
        raise ValueError("duplicate F-Cooper registry group_id")
    for row in groups:
        if row.get("model") != "fcooper":
            raise ValueError("registry contains a non-F-Cooper group")
        if list(row.get("width_schema") or []) != schema:
            raise ValueError("registry group width schema drift")
        if len(row.get("width") or []) != len(schema):
            raise ValueError("registry group width arity drift")
    return {
        "schema_version": "stage5_fcooper_source_registry_audit_v1",
        "group_count": len(groups),
        "genome_count": 2 * len(groups),
        "width_schema": schema,
        "registry_sha256": _sha(registry),
    }


def merge_actual_graph_features(
    base_graphs: Sequence[Mapping[str, Any]],
    feedback: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_group = {str(row["group_id"]): dict(row) for row in base_graphs}
    for row in feedback:
        if str(row.get("model")) != "fcooper":
            raise ValueError("cross-model online feedback is forbidden")
        graph = row.get("graph_features")
        if not isinstance(graph, Mapping):
            raise ValueError("F-Cooper feedback is missing actual graph features")
        if str(graph.get("group_id")) != str(row.get("group_id")):
            raise ValueError("feedback graph group_id mismatch")
        if graph.get("graph_feature_provenance") not in {
            "materialized_onnx_extracted_v1",
            "materialized_tvm_prepared_onnx_extracted_v1",
        }:
            raise ValueError("feedback graph features are not actual materialized features")
        by_group[str(row["group_id"])] = dict(graph)
    return [by_group[key] for key in sorted(by_group)]


def validate_fcooper_feedback_evidence(
    feedback: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    required_sha_fields = (
        "performance_result_sha256",
        "ap_report_sha256",
        "materialized_source_evidence_sha256",
    )
    for source in feedback:
        row = dict(source)
        graph = row.get("graph_features")
        if not isinstance(graph, Mapping):
            raise ValueError("F-Cooper feedback is missing actual graph features")
        if _sha(graph) != row.get("materialized_graph_features_sha256"):
            raise ValueError("F-Cooper feedback graph feature SHA drift")
        recorded = row.pop("actual_feedback_row_sha256", None)
        if recorded != _sha(row):
            raise ValueError("F-Cooper actual feedback row SHA drift")
        if row.get("terminal_status") == "measured_success_gold":
            for field in required_sha_fields:
                value = row.get(field)
                if not isinstance(value, str) or len(value) != 64:
                    raise ValueError(f"F-Cooper feedback missing evidence SHA: {field}")
            artifact_field = (
                "tvm_artifact_sha256"
                if row.get("dispatch_key") == "tvm_auto"
                else "engine_sha256"
            )
            artifact_sha = row.get(artifact_field)
            if not isinstance(artifact_sha, str) or len(artifact_sha) != 64:
                raise ValueError(
                    f"F-Cooper feedback missing evidence SHA: {artifact_field}"
                )
    return {
        "schema_version": "stage5_fcooper_feedback_evidence_audit_v1",
        "feedback_rows": len(feedback),
        "verified_actual_feedback_rows": len(feedback),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feedback-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--round-index", type=int, required=True)
    parser.add_argument("--source-registry-json", type=Path, required=True)
    parser.add_argument("--probe-audit-json", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260723)
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
        default=REPO_ROOT
        / "results/s1_profile_final_v3_20260711/capability_profiles_v3.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.round_index not in {1, 2, 3}:
        raise ValueError("next round index must be 1, 2, or 3")
    verify_frozen_coldstart_artifacts(
        args.coldstart_rows_json, args.coldstart_graph_features_json
    )
    cold_rows = freeze_initial_coldstart(
        _rows(json.loads(args.coldstart_rows_json.read_text()), "rows")
    )
    base_graphs = _rows(
        json.loads(args.coldstart_graph_features_json.read_text()),
        "graph_features",
    )
    feedback = _rows(json.loads(args.feedback_json.read_text()), "rows")
    profiles = _rows(json.loads(args.profiles_json.read_text()), "capability_profiles")
    profile = next(row for row in profiles if row["dispatch_key"] == "trt_engine")
    task = SearchTask("S5-FCO-TRT", "fcooper", "h800", profile)
    validate_task_feedback_history(
        feedback, task=task, completed_rounds=args.round_index
    )
    feedback_evidence_audit = validate_fcooper_feedback_evidence(feedback)
    graphs = merge_actual_graph_features(base_graphs, feedback)
    training = [*cold_rows, *feedback]
    bundle = fit_online_bundle(
        training, graphs, profiles, seed=args.seed + args.round_index
    )
    registry = json.loads(args.source_registry_json.read_text())
    registry_audit = validate_fcooper_registry(registry)
    probe_audit = json.loads(args.probe_audit_json.read_text())
    excluded_ids = {
        str(row["row_id"])
        for row in probe_audit.get("rows", [])
        if row.get("row_id")
    }
    excluded_ids.update(
        str(row.get("manifest_job_id") or row.get("row_id")) for row in feedback
    )
    manifest = build_task_candidate_manifest(
        registry, task=task, measured_row_ids=excluded_ids
    )
    predicted = predict_candidate_rows(bundle, manifest["rows"], profiles)
    selection = select_task_batch(predicted, training, graphs, task=task)
    request = build_measurement_request(
        task=task,
        selected_rows=selection["selected_rows"],
        round_index=args.round_index,
    )
    round_dir = args.output_dir / f"round_{args.round_index:02d}"
    _write_frozen(round_dir / "candidate_manifest.json", manifest)
    _write_frozen(round_dir / "predicted_candidates.json", {"rows": predicted})
    _write_frozen(round_dir / "acquisition.json", selection)
    _write_frozen(round_dir / "measurement_request.json", request)
    state = {
        "schema_version": "stage5_fcooper_round_state_v1",
        "task_id": task.task_id,
        "round_index": args.round_index,
        "completed_feedback_rows": len(feedback),
        "budget_consumed": len(feedback),
        "budget_remaining_after_next_batch": 16 - len(feedback) - 4,
        "cross_model_online_labels_loaded": False,
        "actual_graph_feedback_rows": len(feedback),
        "feedback_evidence_audit": feedback_evidence_audit,
        "registry_audit": registry_audit,
        "status": "awaiting_real_measurement",
        "selected_row_ids": selection["selected_row_ids"],
        "measurement_request_sha256": request["measurement_request_sha256"],
    }
    _write_frozen(round_dir / "round_state.json", state)
    print(json.dumps(state, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
