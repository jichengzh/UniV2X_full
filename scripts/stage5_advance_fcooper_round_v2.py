#!/usr/bin/env python3
"""Advance one formal atomic F-Cooper TRT/H800 recovered-source round."""

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
from scripts.stage5_advance_fcooper_round_v1 import (  # noqa: E402
    _rows,
    _write_frozen,
    merge_actual_graph_features,
    validate_fcooper_feedback_evidence,
    validate_fcooper_registry,
)
from scripts.fcooper_execute_measurement_row_v2 import (  # noqa: E402
    validate_recovery_training_evidence,
)
from scripts.stage5_initialize_fcooper_actual_v2 import (  # noqa: E402
    resolve_fcooper_task,
)


BASE_WIDTH = [64, 128, 256, 128, 256]
PILOT_FRAGMENT = "fcooper_workpackage_a_20260723"


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_formal_feedback_evidence(
    feedback: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    base = validate_fcooper_feedback_evidence(feedback)
    recovered_rows = 0
    for source in feedback:
        row = dict(source)
        if row.get("terminal_status") != "measured_success_gold":
            continue
        checkpoint_sha = row.get("checkpoint_sha256")
        if not isinstance(checkpoint_sha, str) or len(checkpoint_sha) != 64:
            raise ValueError("formal F-Cooper feedback is missing checkpoint SHA")
        if list(row.get("width") or []) != BASE_WIDTH:
            recovery_sha = row.get("recovery_training_report_sha256")
            if not isinstance(recovery_sha, str) or len(recovery_sha) != 64:
                raise ValueError(
                    "pruned formal feedback is missing recovery-training SHA"
                )
            recovery_path = Path(str(row.get("recovery_training_report_path") or ""))
            if (
                not recovery_path.is_file()
                or PILOT_FRAGMENT in str(recovery_path)
                or _sha_file(recovery_path) != recovery_sha
            ):
                raise ValueError("formal recovery-training report file or SHA drift")
            report = json.loads(recovery_path.read_text())
            audit = validate_recovery_training_evidence(
                report_path=recovery_path,
                recovery_contract_path=Path(
                    str(report.get("recovery_contract_path") or "")
                ),
                config_path=Path(str(report.get("config_path") or "")),
                initial_checkpoint_path=Path(
                    str(report.get("initial_checkpoint_path") or "")
                ),
                recovered_checkpoint_path=Path(
                    str(report.get("recovered_checkpoint_path") or "")
                ),
            )
            if audit["recovered_checkpoint_sha256"] != checkpoint_sha:
                raise ValueError("formal recovery-training checkpoint drift")
            recovered_rows += 1
    return {
        **base,
        "schema_version": "stage5_fcooper_formal_feedback_evidence_audit_v2",
        "recovered_pruned_rows": recovered_rows,
        "prefix_only_measurement_rows": 0,
    }


def validate_atomic_release(
    audit: Mapping[str, Any],
    feedback: Sequence[Mapping[str, Any]],
    *,
    completed_rounds: int,
) -> dict[str, Any]:
    if (
        audit.get("schema_version") != "stage5_atomic_batch_audit_v2"
        or audit.get("feedback_released") is not True
        or audit.get("batch_quarantined") is not False
        or int(audit.get("budget_consumed") or 0) != 4
    ):
        raise ValueError("latest formal atomic batch was not released")
    latest_round = completed_rounds - 1
    latest = {
        str(row.get("manifest_job_id") or row.get("row_id") or "")
        for row in feedback
        if int(row.get("round_index", -1)) == latest_round
    }
    released = {
        str(row.get("manifest_job_id") or row.get("row_id") or "")
        for row in audit.get("released_feedback_rows") or []
    }
    if len(latest) != 4 or released != latest:
        raise ValueError("atomic release rows do not match the latest formal batch")
    return {
        "schema_version": "stage5_fcooper_atomic_release_validation_v2",
        "round_index": latest_round,
        "released_row_ids": sorted(released),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feedback-json", type=Path, required=True)
    parser.add_argument("--atomic-audit-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--round-index", type=int, required=True)
    parser.add_argument("--source-registry-json", type=Path, required=True)
    parser.add_argument("--probe-isolation-audit-json", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--coldstart-rows-json", type=Path, required=True)
    parser.add_argument("--coldstart-graph-features-json", type=Path, required=True)
    parser.add_argument("--profiles-json", type=Path, required=True)
    parser.add_argument("--task-id", default="S5-FCO-TRT-V2")
    parser.add_argument("--dispatch-key", default="trt_engine")
    parser.add_argument("--capability-profile-id")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for path in (
        args.feedback_json,
        args.atomic_audit_json,
        args.output_dir,
        args.source_registry_json,
        args.probe_isolation_audit_json,
    ):
        if PILOT_FRAGMENT in str(path):
            raise ValueError(f"formal round cannot use pilot path: {path}")
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
    task = resolve_fcooper_task(
        profiles,
        task_id=args.task_id,
        dispatch_key=args.dispatch_key,
        capability_profile_id=args.capability_profile_id,
    )
    validate_task_feedback_history(
        feedback, task=task, completed_rounds=args.round_index
    )
    atomic_release_audit = validate_atomic_release(
        json.loads(args.atomic_audit_json.read_text()),
        feedback,
        completed_rounds=args.round_index,
    )
    feedback_evidence_audit = validate_formal_feedback_evidence(feedback)
    graphs = merge_actual_graph_features(base_graphs, feedback)
    training = [*cold_rows, *feedback]
    bundle = fit_online_bundle(
        training, graphs, profiles, seed=args.seed + args.round_index
    )
    registry = json.loads(args.source_registry_json.read_text())
    registry_audit = validate_fcooper_registry(registry)
    isolation = json.loads(args.probe_isolation_audit_json.read_text())
    if (
        isolation.get("status") != "passed"
        or isolation.get("probe_metrics_allowed_as_cost_model_labels") is not False
        or isolation.get("probe_rows_allowed_in_t16_budget") is not False
        or isolation.get("probe_rows_allowed_as_winner") is not False
        or not isolation.get("probe_row_ids")
    ):
        raise ValueError("probe isolation audit is not passed")
    excluded_ids = set(isolation.get("probe_row_ids") or [])
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
        "schema_version": "stage5_fcooper_formal_round_state_v2",
        "task_id": task.task_id,
        "round_index": args.round_index,
        "completed_feedback_rows": len(feedback),
        "budget_consumed": len(feedback),
        "budget_remaining_after_next_batch": 16 - len(feedback) - 4,
        "cross_model_online_labels_loaded": False,
        "pilot_online_labels_loaded": False,
        "probe_metrics_loaded_as_labels": False,
        "actual_graph_feedback_rows": len(feedback),
        "feedback_evidence_audit": feedback_evidence_audit,
        "atomic_release_audit": atomic_release_audit,
        "registry_audit": registry_audit,
        "status": "awaiting_recovered_source_measurement",
        "selected_row_ids": selection["selected_row_ids"],
        "measurement_request_sha256": request["measurement_request_sha256"],
    }
    _write_frozen(round_dir / "round_state.json", state)
    print(json.dumps(state, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
