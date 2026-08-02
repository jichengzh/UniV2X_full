#!/usr/bin/env python3
"""Initialize the formal F-Cooper TRT/H800 recovered-source T16 search."""

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
from scripts.stage5_advance_fcooper_round_v1 import (  # noqa: E402
    validate_fcooper_registry,
)


PILOT_FRAGMENT = "fcooper_workpackage_a_20260723"


def rows(payload: Any, field: str) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(item) for item in payload]
    if isinstance(payload, Mapping) and isinstance(payload.get(field), list):
        return [dict(item) for item in payload[field]]
    raise ValueError(f"expected list or object containing {field}")


def write_frozen(path: Path, payload: Any) -> None:
    content = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != content:
        raise ValueError(f"refusing to overwrite drifted formal artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(content, encoding="utf-8")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_formal_registry_provenance(
    registry: Mapping[str, Any],
) -> dict[str, Any]:
    serialized = json.dumps(registry, ensure_ascii=False, sort_keys=True)
    if PILOT_FRAGMENT in serialized:
        raise ValueError("formal source registry contains pilot provenance")
    return {
        "schema_version": "stage5_fcooper_registry_provenance_audit_v2",
        "pilot_references": 0,
    }


def resolve_fcooper_task(
    profiles: list[dict[str, Any]],
    *,
    task_id: str,
    dispatch_key: str,
    capability_profile_id: str | None = None,
) -> SearchTask:
    matches = [
        profile for profile in profiles
        if str(profile.get("dispatch_key")) == str(dispatch_key)
        and (
            capability_profile_id is None
            or str(profile.get("capability_profile_id"))
            == str(capability_profile_id)
        )
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one {dispatch_key} capability profile, got {len(matches)}"
        )
    return SearchTask(str(task_id), "fcooper", "h800", matches[0])


def relabel_dispatch_profile(
    source_rows: list[dict[str, Any]],
    target_profile: Mapping[str, Any],
) -> list[dict[str, Any]]:
    dispatch_key = str(target_profile["dispatch_key"])
    return [
        {
            **row,
            "capability_profile_id": target_profile["capability_profile_id"],
            "capability_digest": target_profile["capability_digest"],
        }
        if str(row.get("dispatch_key")) == dispatch_key
        else dict(row)
        for row in source_rows
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage4-dir", type=Path, required=True)
    parser.add_argument("--coldstart-root", type=Path, required=True)
    parser.add_argument("--profiles-json", type=Path, required=True)
    parser.add_argument("--source-registry-json", type=Path, required=True)
    parser.add_argument("--formal-contract-json", type=Path, required=True)
    parser.add_argument("--probe-audit-json", type=Path, required=True)
    parser.add_argument("--probe-isolation-audit-json", type=Path, required=True)
    parser.add_argument("--numeric-gate-summary-json", type=Path, required=True)
    parser.add_argument("--scanner-execution-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--task-id", default="S5-FCO-TRT-V2")
    parser.add_argument("--dispatch-key", default="trt_engine")
    parser.add_argument("--capability-profile-id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (
        args.source_registry_json,
        args.formal_contract_json,
        args.probe_audit_json,
        args.probe_isolation_audit_json,
        args.numeric_gate_summary_json,
        args.scanner_execution_json,
    ):
        if PILOT_FRAGMENT in str(path):
            raise ValueError(f"formal initialization cannot read pilot path: {path}")

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
    task = resolve_fcooper_task(
        profiles,
        task_id=args.task_id,
        dispatch_key=args.dispatch_key,
        capability_profile_id=args.capability_profile_id,
    )
    training_rows = relabel_dispatch_profile(
        training_rows, task.capability_profile
    )
    closure = json.loads(
        (args.stage4_dir / "stage4_p1_p3_closure_audit.json").read_text()
    )
    contract = json.loads(args.formal_contract_json.read_text())
    scanner = json.loads(args.scanner_execution_json.read_text())
    registry = json.loads(args.source_registry_json.read_text())
    registry_audit = validate_fcooper_registry(registry)
    registry_provenance_audit = validate_formal_registry_provenance(registry)
    if (
        scanner.get("status") != "success"
        or scanner.get("hardware_schema_validated") is not True
        or scanner.get("manual_override_used") is not False
        or scanner.get("partition_sha256") != contract.get("partition_sha256")
        or registry_audit["registry_sha256"]
        != hashlib.sha256(
            json.dumps(
                registry, ensure_ascii=True, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
    ):
        raise ValueError("fresh scanner or source registry audit failed")

    gate = json.loads(args.numeric_gate_summary_json.read_text())
    if gate.get("status") != "passed" or gate.get("t16_search_allowed") is not True:
        raise ValueError("formal T16 cannot start before recovery numeric gate")
    probe_audit = json.loads(args.probe_audit_json.read_text())
    isolation = json.loads(args.probe_isolation_audit_json.read_text())
    if (
        probe_audit.get("all_probes_terminal") is not True
        or isolation.get("status") != "passed"
        or isolation.get("probe_metrics_allowed_as_cost_model_labels") is not False
        or isolation.get("probe_rows_allowed_in_t16_budget") is not False
        or isolation.get("probe_rows_allowed_as_winner") is not False
        or not isolation.get("probe_row_ids")
    ):
        raise ValueError("capability probes are not terminal and isolated")

    bundle = fit_production_bundle(
        training_rows,
        graph_features,
        profiles,
        closure,
        seed=args.seed,
        training_view_policy="initial_coldstart_only",
    )
    bundle.model_anchors["fcooper"] = float(contract["ap70_ref"])
    task_contract = validate_search_task(task)
    probe_row_ids = set(isolation.get("probe_row_ids") or [])
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
    write_frozen(root / "task_contract.json", task_contract)
    write_frozen(root / "candidate_manifest.json", manifest)
    write_frozen(root / "round_00/predicted_candidates.json", {"rows": predicted})
    write_frozen(root / "round_00/acquisition.json", selection)
    write_frozen(root / "round_00/measurement_request.json", request)
    summary = {
        "schema_version": "stage5_fcooper_formal_v2_initialization_summary",
        "task_id": task.task_id,
        "training_view_policy": "initial_coldstart_only",
        "coldstart_rows": len(training_rows),
        "coldstart_audit": cold_audit,
        "cross_model_online_labels_loaded": False,
        "pilot_online_labels_loaded": False,
        "probe_metrics_loaded_as_labels": False,
        "probe_rows_excluded_from_online_budget": len(probe_row_ids),
        "source_registry_sha256": sha256_file(args.source_registry_json),
        "source_registry_provenance_audit": registry_provenance_audit,
        "fresh_partition_sha256": scanner["partition_sha256"],
        "recovery_gate_sha256": sha256_file(args.numeric_gate_summary_json),
        "eligible_genomes": manifest["eligible_row_count"],
        "selected_row_ids": selection["selected_row_ids"],
        "measurement_request_sha256": request["measurement_request_sha256"],
    }
    write_frozen(root / "initialization_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
