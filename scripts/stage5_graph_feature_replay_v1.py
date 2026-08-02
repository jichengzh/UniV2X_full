#!/usr/bin/env python3
"""Teacher-forced one-step replay with actual features for measured Stage5 rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from framework.stage5.production_search_v1 import predict_candidate_rows  # noqa: E402
from framework.stage5.single_target_search_v2 import (  # noqa: E402
    SearchTask,
    build_task_candidate_manifest,
    fit_online_bundle,
    freeze_initial_coldstart,
    select_task_batch,
    validate_task_feedback_history,
    verify_frozen_coldstart_artifacts,
)
from framework.stage5.source_registry_v1 import validate_full_source_registry  # noqa: E402


TASK_SPECS = {
    "S5-PYR-TVM": ("pyramid", "tvm_auto"),
    "S5-PYR-TRT": ("pyramid", "trt_engine"),
    "S5-COD-TVM": ("codriving", "tvm_auto"),
    "S5-COD-TRT": ("codriving", "trt_engine"),
}
TARGETS = ("latency_ms", "energy_j", "ap70")


def _sha_payload(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _read(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"missing required artifact: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON artifact: {path}: {error.msg}") from error


def _rows(payload: Any, field: str = "rows") -> list[dict[str, Any]]:
    source = payload.get(field) if isinstance(payload, Mapping) else payload
    if not isinstance(source, list) or not all(isinstance(row, Mapping) for row in source):
        raise ValueError(f"expected list or object containing {field}")
    return [dict(row) for row in source]


def _row_id(row: Mapping[str, Any]) -> str:
    return str(row.get("manifest_job_id") or row.get("row_id") or "")


def objective_projection_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    projection = []
    for row in sorted(rows, key=_row_id):
        projection.append(
            {
                "manifest_job_id": _row_id(row),
                "task_id": row.get("task_id"),
                "group_id": row.get("group_id"),
                "model": row.get("model"),
                "width": row.get("width"),
                "q_mode": row.get("q_mode"),
                "dispatch_key": row.get("dispatch_key"),
                "capability_profile_id": row.get("capability_profile_id"),
                "terminal_status": row.get("terminal_status"),
                **{
                    target: (
                        float(row[target])
                        if not isinstance(row.get(target), bool)
                        and row.get(target) is not None
                        and math.isfinite(float(row[target]))
                        else None
                    )
                    for target in ("latency_ms", "energy_j", "ap30", "ap50", "ap70")
                },
            }
        )
    return _sha_payload(projection)


def selection_comparison(
    historical_ids: Sequence[str], actual_replay_ids: Sequence[str]
) -> dict[str, Any]:
    historical = set(historical_ids)
    replay = set(actual_replay_ids)
    if (
        len(historical) != len(historical_ids)
        or len(replay) != len(actual_replay_ids)
        or len(historical) != 4
        or len(replay) != 4
    ):
        raise ValueError("selection comparison requires two unique four-genome batches")
    overlap = historical & replay
    union = historical | replay
    return {
        "exact_match": historical == replay,
        "overlap_count": len(overlap),
        "jaccard": float(len(overlap) / len(union)),
        "historical_only": sorted(historical - replay),
        "actual_replay_only": sorted(replay - historical),
    }


def prediction_delta(
    historical: Mapping[str, Any], actual: Mapping[str, Any]
) -> dict[str, dict[str, float]]:
    before = historical.get("predictions")
    after = actual.get("predictions")
    if not isinstance(before, Mapping) or not isinstance(after, Mapping):
        raise ValueError("prediction rows must contain predictions")
    result = {}
    for target in TARGETS:
        old = float(before[target])
        new = float(after[target])
        result[target] = {
            "historical_surrogate_feedback": old,
            "actual_feedback_replay": new,
            "absolute_change": new - old,
            "relative_change": (new - old) / max(abs(old), 1e-12),
        }
    return result


def _validate_actual_view(
    rows: Sequence[Mapping[str, Any]],
    historical_rows: Sequence[Mapping[str, Any]],
    historical_round_by_id: Mapping[str, tuple[str, int]],
) -> None:
    if len(rows) != 64:
        raise ValueError("actual graph feedback view must contain exactly 64 rows")
    historical_by_id = {_row_id(row): row for row in historical_rows}
    if len(historical_by_id) != 64 or set(historical_by_id) != {_row_id(row) for row in rows}:
        raise ValueError("actual view identities do not match historical Stage5 feedback")
    for row in rows:
        row_id = _row_id(row)
        historical = historical_by_id[row_id]
        visible = {key: value for key, value in row.items() if key != "derived_view_row_sha256"}
        graph = row.get("graph_features")
        expected_task, expected_round = historical_round_by_id.get(row_id, ("", -1))
        identity_drift = sorted(
            key
            for key, value in historical.items()
            if key != "graph_features" and row.get(key) != value
        )
        if (
            identity_drift
            or row.get("candidate_graph_features") != historical.get("graph_features")
            or row.get("task_id") != expected_task
            or row.get("graph_feature_backfill_round_index") != expected_round
            or row.get("derived_view_only") is not True
            or row.get("historical_request_identity_matches_visible_row") is not False
            or row.get("derived_view_row_sha256") != _sha_payload(visible)
            or row.get("historical_feedback_row_sha256")
            != _sha_payload(historical_by_id[row_id])
            or not isinstance(graph, Mapping)
            or graph.get("graph_feature_provenance") != "materialized_onnx_extracted_v1"
        ):
            raise ValueError(f"invalid actual graph feedback view row: {row_id}")


def _dominates(left: Sequence[float], right: Sequence[float]) -> bool:
    return all(a <= b for a, b in zip(left, right)) and any(
        a < b for a, b in zip(left, right)
    )


def online_pareto_ids_by_task(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for task_id in TASK_SPECS:
        selected = [
            row
            for row in rows
            if row.get("task_id") == task_id
            and row.get("terminal_status") == "measured_success_gold"
            and all(row.get(target) is not None for target in TARGETS)
        ]
        vectors = [
            (
                float(row["latency_ms"]),
                float(row["energy_j"]),
                -float(row["ap70"]),
            )
            for row in selected
        ]
        result[task_id] = sorted(
            _row_id(row)
            for index, row in enumerate(selected)
            if not any(
                other_index != index and _dominates(other, vectors[index])
                for other_index, other in enumerate(vectors)
            )
        )
    return result


def run_replay(
    *,
    formal_root: Path,
    coldstart_rows_path: Path,
    coldstart_graphs_path: Path,
    profiles_path: Path,
    registry_path: Path,
    actual_feedback_view_path: Path,
    seed: int = 20260718,
) -> dict[str, Any]:
    coldstart_sha = verify_frozen_coldstart_artifacts(
        coldstart_rows_path, coldstart_graphs_path
    )
    initial = freeze_initial_coldstart(_rows(_read(coldstart_rows_path)))
    cold_graphs = _rows(_read(coldstart_graphs_path), "graph_features")
    profiles = _rows(_read(profiles_path), "capability_profiles")
    registry = _read(registry_path)
    registry_audit = validate_full_source_registry(registry)
    actual_payload = _read(actual_feedback_view_path)
    actual_rows = _rows(actual_payload)

    historical_rows = []
    historical_by_task_round: dict[tuple[str, int], list[dict[str, Any]]] = {}
    historical_round_by_id: dict[str, tuple[str, int]] = {}
    for task_id in TASK_SPECS:
        for round_index in range(4):
            batch = _rows(
                _read(
                    formal_root
                    / task_id
                    / f"round_{round_index:02d}/final/stage5_feedback_v2_final.json"
                )
            )
            historical_by_task_round[(task_id, round_index)] = batch
            for row in batch:
                row_id = _row_id(row)
                if not row_id or row_id in historical_round_by_id:
                    raise ValueError(f"historical feedback identity drift: {row_id}")
                historical_round_by_id[row_id] = (task_id, round_index)
            historical_rows.extend(batch)
    _validate_actual_view(actual_rows, historical_rows, historical_round_by_id)
    historical_objectives_sha = objective_projection_sha256(historical_rows)
    actual_objectives_sha = objective_projection_sha256(actual_rows)
    if historical_objectives_sha != actual_objectives_sha:
        raise ValueError("actual graph view changed measured objective evidence")
    historical_pareto = online_pareto_ids_by_task(historical_rows)
    actual_pareto = online_pareto_ids_by_task(actual_rows)
    if historical_pareto != actual_pareto:
        raise ValueError("actual graph view changed online measured Pareto identities")

    actual_by_task_round: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in actual_rows:
        key = historical_round_by_id[_row_id(row)]
        actual_by_task_round.setdefault(key, []).append(row)
    if any(len(actual_by_task_round.get((task_id, index), [])) != 4 for task_id in TASK_SPECS for index in range(4)):
        raise ValueError("actual graph view must preserve four rows per task round")

    profile_by_dispatch = {str(row["dispatch_key"]): row for row in profiles}
    task_reports = []
    all_rounds = []
    for task_id, (model, dispatch) in TASK_SPECS.items():
        task = SearchTask(task_id, model, "h800", profile_by_dispatch[dispatch])
        rounds = [
            {
                "round_index": 0,
                "replay_kind": "unchanged_by_construction_no_online_feedback",
                "comparison": {
                    "exact_match": True,
                    "overlap_count": 4,
                    "jaccard": 1.0,
                    "historical_only": [],
                    "actual_replay_only": [],
                },
                "prediction_deltas_for_historical_candidate_batch_under_actual_prefix": [],
            }
        ]
        for round_index in (1, 2, 3):
            feedback = [
                row
                for completed in range(round_index)
                for row in actual_by_task_round[(task_id, completed)]
            ]
            validate_task_feedback_history(
                feedback, task=task, completed_rounds=round_index
            )
            training = [*initial, *feedback]
            graph_by_group = {
                str(row["group_id"]): dict(row) for row in cold_graphs
            }
            for row in feedback:
                graph_by_group[str(row["group_id"])] = dict(row["graph_features"])
            graphs = list(graph_by_group.values())
            bundle = fit_online_bundle(
                training, graphs, profiles, seed=seed + round_index
            )
            measured_ids = {_row_id(row) for row in training}
            manifest = build_task_candidate_manifest(
                registry, task=task, measured_row_ids=measured_ids
            )
            replay_predicted = predict_candidate_rows(bundle, manifest["rows"], profiles)
            replay_selection = select_task_batch(
                replay_predicted, training, graphs, task=task
            )
            historical_request = _read(
                formal_root
                / task_id
                / f"round_{round_index:02d}/measurement_request.json"
            )
            historical_ids = [_row_id(row) for row in _rows(historical_request)]
            replay_ids = list(replay_selection["selected_row_ids"])
            comparison = selection_comparison(historical_ids, replay_ids)
            historical_predicted = _rows(
                _read(
                    formal_root
                    / task_id
                    / f"round_{round_index:02d}/predicted_candidates.json"
                )
            )
            historical_map = {_row_id(row): row for row in historical_predicted}
            replay_map = {_row_id(row): row for row in replay_predicted}
            deltas = []
            for row_id in historical_ids:
                if row_id not in historical_map or row_id not in replay_map:
                    raise ValueError(f"historical candidate missing from replay: {row_id}")
                deltas.append(
                    {
                        "manifest_job_id": row_id,
                        "targets": prediction_delta(
                            historical_map[row_id], replay_map[row_id]
                        ),
                    }
                )
            rounds.append(
                {
                    "round_index": round_index,
                    "replay_kind": "teacher_forced_one_step_actual_measured_history",
                    "history_row_count": len(feedback),
                    "comparison": comparison,
                    "prediction_deltas_for_historical_candidate_batch_under_actual_prefix": deltas,
                }
            )
        task_report = {"task_id": task_id, "rounds": rounds}
        task_reports.append(task_report)
        all_rounds.extend(rounds)

    replay_rounds = [
        row for row in all_rounds if row["round_index"] in {1, 2, 3}
    ]
    comparisons = [row["comparison"] for row in replay_rounds]
    relative_changes_by_target = {
        target: [
            abs(float(item["targets"][target]["relative_change"]))
            for task in task_reports
            for round_row in task["rounds"]
            for item in round_row[
                "prediction_deltas_for_historical_candidate_batch_under_actual_prefix"
            ]
        ]
        for target in TARGETS
    }
    relative_changes = [
        value
        for values in relative_changes_by_target.values()
        for value in values
    ]
    return {
        "schema_version": "stage5_graph_feature_teacher_forced_replay_v1",
        "replay_scope": "one_step_only_after_each_historical_feedback_prefix",
        "candidate_graph_feature_policy": (
            "actual features for completed measured history; surrogate features for "
            "unmaterialized candidates"
        ),
        "counterfactual_limitation": (
            "After the first selection divergence, labels for replay-only genomes are unavailable; "
            "this artifact is not a complete counterfactual search trajectory."
        ),
        "coldstart_artifact_audit": coldstart_sha,
        "source_registry_audit": registry_audit,
        "historical_objectives_sha256": historical_objectives_sha,
        "actual_view_objectives_sha256": actual_objectives_sha,
        "online_measured_objective_projection_invariant": True,
        "online_measured_pareto_ids_invariant": True,
        "historical_online_pareto_ids_by_task": historical_pareto,
        "summary": {
            "task_count": 4,
            "historical_round_count": 16,
            "round0_unchanged_by_construction_count": 4,
            "teacher_forced_replay_round_count": 12,
            "exact_match_round_count": sum(
                bool(row["exact_match"]) for row in comparisons
            ),
            "selection_divergence_round_count": sum(
                not bool(row["exact_match"]) for row in comparisons
            ),
            "mean_batch_overlap": float(
                sum(int(row["overlap_count"]) for row in comparisons)
                / len(comparisons)
            ),
            "mean_absolute_prediction_relative_change": float(
                sum(relative_changes) / len(relative_changes)
            )
            if relative_changes
            else 0.0,
            "max_absolute_prediction_relative_change": float(
                max(relative_changes, default=0.0)
            ),
            "prediction_relative_change_by_target": {
                target: {
                    "mean_absolute": float(sum(values) / len(values)) if values else 0.0,
                    "max_absolute": float(max(values, default=0.0)),
                    "count": len(values),
                }
                for target, values in relative_changes_by_target.items()
            },
        },
        "tasks": task_reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--coldstart-rows-json", type=Path, required=True)
    parser.add_argument("--coldstart-graph-features-json", type=Path, required=True)
    parser.add_argument("--profiles-json", type=Path, required=True)
    parser.add_argument("--source-registry-json", type=Path, required=True)
    parser.add_argument("--actual-feedback-view-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260718)
    args = parser.parse_args()
    report = run_replay(
        formal_root=args.formal_root,
        coldstart_rows_path=args.coldstart_rows_json,
        coldstart_graphs_path=args.coldstart_graph_features_json,
        profiles_path=args.profiles_json,
        registry_path=args.source_registry_json,
        actual_feedback_view_path=args.actual_feedback_view_json,
        seed=args.seed,
    )
    content = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_json.with_name(f".{args.output_json.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(args.output_json)
    print(json.dumps(report["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
