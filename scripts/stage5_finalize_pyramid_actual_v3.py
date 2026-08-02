#!/usr/bin/env python3
"""Close the two Pyramid actual-feedback v3 search tasks."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from framework.stage5.closure_v3 import build_stage5_closure_audit
from framework.stage5.single_target_search_v2 import finalize_atomic_batch


TASK_IDS = ("S5-PYR-TVM", "S5-PYR-TRT")
ACTUAL_PROVENANCE = "materialized_onnx_extracted_v1"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_actual_feedback_rows(rows: Sequence[Mapping[str, Any]]) -> int:
    for row in rows:
        row_id = str(row.get("manifest_job_id") or "")
        if row.get("feedback_feature_contract") != "actual_feedback_v3":
            raise ValueError(f"actual feedback contract missing: {row_id}")
        features = row.get("graph_features")
        if (
            not isinstance(features, Mapping)
            or features.get("graph_feature_provenance") != ACTUAL_PROVENANCE
        ):
            raise ValueError(f"actual graph feature provenance missing: {row_id}")
        digest = row.get("materialized_graph_features_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"materialized graph feature SHA missing: {row_id}")
    return len(rows)


def _task_closure(root: Path, gold176: Sequence[Mapping[str, Any]], task_id: str) -> tuple[dict[str, Any], list[Mapping[str, Any]]]:
    task_root = root / task_id
    requests = []
    feedback_batches = []
    atomic_audits = []
    online_rows: list[Mapping[str, Any]] = []
    for round_index in range(4):
        round_root = task_root / f"round_{round_index:02d}"
        request = _read_json(round_root / "measurement_request.json")
        feedback = _read_json(
            round_root / "actual_feedback" / "stage5_feedback_v3_actual.json"
        )
        requests.append(request)
        feedback_batches.append(feedback)
        online_rows.extend(feedback)
        if round_index == 0:
            import_audit = _read_json(round_root / "round0_import_audit_v3.json")
            if (
                import_audit.get("schema_version")
                != "stage5_round0_actual_feedback_import_audit_v3"
                or import_audit.get("row_count") != 4
                or import_audit.get("silent_surrogate_fallback_count") != 0
                or import_audit.get("imported_measurements_reexecuted") is not False
            ):
                raise ValueError(f"{task_id} round-0 import audit is incomplete")
        else:
            source_audit = _read_json(round_root / "final" / "atomic_batch_audit.json")
            source_ids = {
                str(row.get("manifest_job_id") or row.get("row_id") or "")
                for row in source_audit.get("released_feedback_rows") or []
            }
            request_ids = {str(row["manifest_job_id"]) for row in request["rows"]}
            if (
                source_audit.get("schema_version") != "stage5_atomic_batch_audit_v2"
                or source_audit.get("feedback_released") is not True
                or source_audit.get("batch_quarantined") is not False
                or source_audit.get("budget_consumed") != 4
                or source_ids != request_ids
            ):
                raise ValueError(f"{task_id} round-{round_index} source atomic audit is incomplete")
        atomic_audits.append(finalize_atomic_batch(request, feedback))

    if len(online_rows) != 16 or len({row["manifest_job_id"] for row in online_rows}) != 16:
        raise ValueError(f"{task_id} must contain exactly 16 unique online rows")
    validate_actual_feedback_rows(online_rows)
    closure = build_stage5_closure_audit(
        gold176,
        _read_json(task_root / "candidate_manifest.json"),
        requests,
        feedback_batches,
        atomic_audits,
    )
    return closure, online_rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, ensure_ascii=False, sort_keys=True)
                if isinstance(value, (list, dict))
                else value
                for key, value in row.items()
            })


def finalize(root: Path, gold176_path: Path, output_dir: Path) -> dict[str, Any]:
    gold176 = _read_json(gold176_path)
    scheduler_terminal = _read_json(
        root / "controller" / "pyramid_actual_v3_scheduler_terminal.json"
    )
    if (
        scheduler_terminal.get("status") != "budget_exhausted"
        or scheduler_terminal.get("feedback_contract") != "actual_v3"
        or scheduler_terminal.get("total_online_rows") != 32
    ):
        raise ValueError("Pyramid actual-v3 scheduler terminal is incomplete")

    output_dir.mkdir(parents=True, exist_ok=True)
    closures: dict[str, dict[str, Any]] = {}
    all_online: list[Mapping[str, Any]] = []
    pareto_rows = []
    for task_id in TASK_IDS:
        closure, online_rows = _task_closure(root, gold176, task_id)
        closures[task_id] = closure
        all_online.extend(online_rows)
        (output_dir / f"{task_id}_closure.json").write_text(
            json.dumps(closure, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        for point in closure["frontier_points"]:
            pareto_rows.append({
                "task_id": task_id,
                "capability_profile_id": closure["capability_profile_id"],
                "manifest_job_id": point["manifest_job_id"],
                "round_index": point["round_index"],
                "q_mode": point["q_mode"],
                **point["objectives"],
            })

    summary = {
        "schema_version": "stage5_pyramid_actual_v3_closure_v1",
        "status": "pyramid_search_closed",
        "task_count": 2,
        "online_row_count": len(all_online),
        "actual_graph_feature_count": validate_actual_feedback_rows(all_online),
        "silent_surrogate_fallback_count": 0,
        "terminal_status_counts": dict(
            Counter(str(row["terminal_status"]) for row in all_online)
        ),
        "q_mode_counts": dict(Counter(str(row["q_mode"]) for row in all_online)),
        "tasks": {
            task_id: {
                "terminal_counts": closure["terminal_counts"],
                "frontier_count": len(closure["frontier_ids"]),
                "frontier_ids": closure["frontier_ids"],
                "hv_curve": closure["hv_curve"],
                "independent_validation_ids": closure["independent_validation_ids"],
            }
            for task_id, closure in closures.items()
        },
        "stage6_pyramid_preliminary_protocol_ready": True,
        "stage5_full_four_task_closure": False,
    }
    (output_dir / "pyramid_actual_v3_closure_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(output_dir / "pyramid_actual_v3_online_rows.csv", all_online)
    _write_csv(output_dir / "pyramid_actual_v3_pareto.csv", pareto_rows)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--gold176-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = finalize(args.root, args.gold176_json, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
