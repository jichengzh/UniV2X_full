#!/usr/bin/env python3
"""Audit and summarize the completed CoDriving actual-feedback v3 searches."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


TASKS = {
    "S5-COD-TVM": "tvm_auto",
    "S5-COD-TRT": "trt_engine",
}
SUCCESS = "measured_success_gold"
FAILURES = {"feasibility_failure", "numerical_feasibility_failure"}


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha_payload(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _finite(value: Any) -> bool:
    try:
        return not isinstance(value, bool) and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _verify_file(row: Mapping[str, Any], path_key: str, sha_key: str) -> None:
    path = Path(str(row.get(path_key) or ""))
    expected = str(row.get(sha_key) or "")
    if not path.is_file() or len(expected) != 64 or _sha_file(path) != expected:
        raise ValueError(f"artifact SHA audit failed: {path_key}={path}")


def _validate_task(root: Path, task_id: str, dispatch: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    task_root = root / task_id
    terminal = _read(task_root / "task_budget_terminal.json")
    expected_terminal = {
        "status": "budget_exhausted",
        "budget_consumed": 16,
        "round_count": 4,
        "feedback_contract": "actual_v3",
    }
    if any(terminal.get(key) != value for key, value in expected_terminal.items()):
        raise ValueError(f"task terminal drift: {task_id}")
    rows: list[dict[str, Any]] = []
    request_file_shas: list[str] = []
    artifact_sha_audits = 0
    for round_index in range(4):
        round_root = task_root / f"round_{round_index:02d}"
        request_path = round_root / "measurement_request.json"
        actual_path = round_root / "actual_feedback/stage5_feedback_v3_actual.json"
        audit_path = round_root / "actual_feedback/actual_feedback_batch_audit_v3.json"
        historical_path = round_root / "final/stage5_feedback_v2_final.json"
        request = _read(request_path)
        actual = _read(actual_path)
        audit = _read(audit_path)
        historical = _read(historical_path)
        request_sha = _sha_file(request_path)
        request_file_shas.append(request_sha)
        if request.get("task_id") != task_id or request.get("round_index") != round_index:
            raise ValueError(f"request identity drift: {task_id} round {round_index}")
        request_payload = dict(request)
        request_digest = request_payload.pop("measurement_request_sha256", None)
        if request_digest != _sha_payload(request_payload):
            raise ValueError(f"request payload SHA drift: {task_id} round {round_index}")
        if not isinstance(actual, list) or len(actual) != 4:
            raise ValueError(f"actual feedback batch size drift: {task_id} round {round_index}")
        expected_audit = {
            "task_id": task_id,
            "round_index": round_index,
            "promoted_row_count": 4,
            "silent_surrogate_fallback_count": 0,
            "measurement_request_file_sha256": request_sha,
            "historical_feedback_file_sha256": _sha_file(historical_path),
        }
        if any(audit.get(key) != value for key, value in expected_audit.items()):
            raise ValueError(f"actual batch audit drift: {task_id} round {round_index}")
        requested = {row["manifest_job_id"]: row for row in request["rows"]}
        promoted = {row["manifest_job_id"]: row for row in actual}
        historical_by_id = {row["manifest_job_id"]: row for row in historical}
        audit_by_id = {row["manifest_job_id"]: row for row in audit["rows"]}
        if not (set(requested) == set(promoted) == set(historical_by_id) == set(audit_by_id)):
            raise ValueError(f"batch identity mismatch: {task_id} round {round_index}")
        for row_id, row in promoted.items():
            requested_row = requested[row_id]
            request_row_sha = request["row_sha256"][row_id]
            if request_row_sha != _sha_payload(requested_row):
                raise ValueError(f"request row SHA drift: {row_id}")
            checks = {
                "task_id": task_id,
                "model": "codriving",
                "dispatch_key": dispatch,
                "training_source": "online_feedback",
                "feedback_feature_contract": "actual_feedback_v3",
                "measurement_request_row_sha256": request_row_sha,
            }
            if any(row.get(key) != value for key, value in checks.items()):
                raise ValueError(f"feedback contract drift: {row_id}")
            if row.get("graph_features", {}).get("graph_feature_provenance") != "materialized_onnx_extracted_v1":
                raise ValueError(f"actual graph provenance drift: {row_id}")
            if row.get("candidate_graph_features_sha256") != _sha_payload(row["candidate_graph_features"]):
                raise ValueError(f"candidate graph SHA drift: {row_id}")
            if row.get("materialized_graph_features_sha256") != _sha_payload(row["graph_features"]):
                raise ValueError(f"actual graph SHA drift: {row_id}")
            actual_digest = row.get("actual_feedback_row_sha256")
            unhashed = dict(row)
            unhashed.pop("actual_feedback_row_sha256", None)
            if actual_digest != _sha_payload(unhashed):
                raise ValueError(f"actual feedback row SHA drift: {row_id}")
            if row.get("historical_feedback_row_sha256") != _sha_payload(historical_by_id[row_id]):
                raise ValueError(f"historical feedback row SHA drift: {row_id}")
            audit_row = audit_by_id[row_id]
            for key in (
                "candidate_graph_features_sha256",
                "materialized_graph_features_sha256",
                "actual_feedback_row_sha256",
            ):
                if audit_row.get(key) != row.get(key):
                    raise ValueError(f"row audit SHA drift: {row_id} {key}")
            status = row.get("terminal_status")
            if status == SUCCESS:
                if not all(_finite(row.get(metric)) for metric in ("latency_ms", "energy_j", "ap30", "ap50", "ap70")):
                    raise ValueError(f"success metric missing: {row_id}")
                for path_key, sha_key in (
                    ("performance_result_json", "performance_result_sha256"),
                    ("ap_report_path", "ap_report_sha256"),
                    ("materialized_source_evidence_path", "materialized_source_evidence_sha256"),
                ):
                    _verify_file(row, path_key, sha_key)
                    artifact_sha_audits += 1
                onnx_path = Path(str(row["graph_features"].get("onnx_path") or ""))
                if not onnx_path.is_file() or _sha_file(onnx_path) != row["graph_features"].get("onnx_sha256"):
                    raise ValueError(f"ONNX SHA audit failed: {row_id}")
                artifact_sha_audits += 1
            elif status in FAILURES:
                if not row.get("failure_reason"):
                    raise ValueError(f"feasibility failure reason missing: {row_id}")
            else:
                raise ValueError(f"non-terminal feedback row: {row_id}")
            rows.append(dict(row))
    identities = [row["manifest_job_id"] for row in rows]
    if len(rows) != 16 or len(set(identities)) != 16:
        raise ValueError(f"task feedback identity/count drift: {task_id}")
    return rows, {
        "task_id": task_id,
        "dispatch_key": dispatch,
        "row_count": len(rows),
        "success_count": sum(row["terminal_status"] == SUCCESS for row in rows),
        "feasibility_failure_count": sum(row["terminal_status"] in FAILURES for row in rows),
        "round_count": 4,
        "budget_consumed": 16,
        "silent_surrogate_fallback_count": 0,
        "artifact_sha_audit_count": artifact_sha_audits,
        "measurement_request_file_sha256": request_file_shas,
        "task_budget_terminal_sha256": _sha_file(task_root / "task_budget_terminal.json"),
    }


def _write_outputs(output_dir: Path, rows: list[dict[str, Any]], audit: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "codriving_actual_v3_closure_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    fields = (
        "task_id", "manifest_job_id", "width", "q_mode", "terminal_status",
        "latency_ms", "energy_j", "ap30", "ap50", "ap70", "failure_reason",
        "actual_feedback_row_sha256",
    )
    with (output_dir / "codriving_actual_v3_rows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "width": "x".join(map(str, row["width"]))})
    lines = [
        "# CoDriving actual-feedback v3 closure",
        "",
        "| Task | Rows | Success | Feasibility failure | Artifact SHA audits |",
        "|---|---:|---:|---:|---:|",
    ]
    for task in audit["tasks"]:
        lines.append(
            f"| {task['task_id']} | {task['row_count']} | {task['success_count']} | "
            f"{task['feasibility_failure_count']} | {task['artifact_sha_audit_count']} |"
        )
    lines.extend(["", f"Overall status: `{audit['status']}`; total rows: `{audit['total_row_count']}`."])
    (output_dir / "codriving_actual_v3_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    task_rows = []
    task_audits = []
    for task_id, dispatch in TASKS.items():
        rows, audit = _validate_task(args.root, task_id, dispatch)
        task_rows.extend(rows)
        task_audits.append(audit)
    identities = [row["manifest_job_id"] for row in task_rows]
    if len(task_rows) != 32 or len(set(identities)) != 32:
        raise ValueError("combined online feedback must contain 32 unique rows")
    audit = {
        "schema_version": "stage5_codriving_actual_v3_closure_audit_v1",
        "status": "complete",
        "passed": True,
        "root": str(args.root),
        "total_row_count": 32,
        "task_count": 2,
        "silent_surrogate_fallback_count": 0,
        "tasks": task_audits,
    }
    output_dir = args.output_dir or args.root / "closure"
    _write_outputs(output_dir, task_rows, audit)
    print(json.dumps(audit, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
