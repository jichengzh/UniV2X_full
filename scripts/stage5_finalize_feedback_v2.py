#!/usr/bin/env python3
"""Finalize one Stage5 v2 atomic batch without four-arm group assumptions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from framework.stage5.single_target_search_v2 import finalize_atomic_batch
from scripts.stage3_finalize_gold96_v3 import (
    FAILURE_STATUSES,
    _finalize_row,
    read_jsonl,
    sha256_file,
    write_dataset_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--measurement-request-json", type=Path, required=True)
    parser.add_argument("--ap-plan-jsonl", type=Path, required=True)
    parser.add_argument("--performance-state-jsonl", type=Path, required=True)
    parser.add_argument("--ap-state-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest_json.read_text(encoding="utf-8"))
    request = json.loads(args.measurement_request_json.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "stage5_performance_manifest_v2":
        raise ValueError("unexpected Stage5 v2 performance manifest")
    jobs = [dict(row) for row in manifest.get("jobs") or []]
    request_rows = [dict(row) for row in request.get("rows") or []]
    if len(jobs) != 4 or len(request_rows) != 4:
        raise ValueError("Stage5 v2 finalization requires one four-genome atomic batch")
    if manifest.get("source_request_sha256") != request.get("measurement_request_sha256"):
        raise ValueError("performance manifest/request SHA mismatch")
    request_by_id = {
        str(row["manifest_job_id"]): row for row in request_rows
    }
    job_by_id = {str(row["manifest_job_id"]): row for row in jobs}
    if set(request_by_id) != set(job_by_id):
        raise ValueError("performance manifest rows do not match the atomic request")
    plan_rows = read_jsonl(args.ap_plan_jsonl)
    plan_by_id = {str(row["manifest_job_id"]): row for row in plan_rows}
    if set(plan_by_id) != set(job_by_id):
        raise ValueError("AP plan rows do not match the atomic request")
    performance_by_id = {}
    for row in read_jsonl(args.performance_state_jsonl):
        performance_by_id.setdefault(
            str(row.get("manifest_job_id") or row.get("job_id") or ""), []
        ).append(row)
    ap_by_id = {}
    for row in read_jsonl(args.ap_state_jsonl):
        ap_by_id.setdefault(str(row.get("manifest_job_id") or row.get("job_id") or ""), []).append(row)

    finalized_rows = []
    for job_id, source in job_by_id.items():
        performance_id = str(plan_by_id[job_id].get("performance_job_id") or job_id)
        finalized = _finalize_row(
            source,
            performance_by_id.get(performance_id, ()),
            ap_by_id.get(job_id, ()),
            output_schema="stage5_feedback_row_v2",
        )
        requested = request_by_id[job_id]
        evidence_path = Path(str(source.get("source_evidence_path") or ""))
        evidence_sha = str(source.get("source_evidence_sha256") or "")
        if not evidence_path.is_file() or sha256_file(evidence_path) != evidence_sha:
            raise ValueError(f"materialized source evidence mismatch: {job_id}")
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        if evidence.get("source_plan_sha256") != requested["source_evidence_sha256"]:
            raise ValueError(f"materialized source plan drift: {job_id}")
        row = {
            **requested,
            **finalized,
            "row_id": job_id,
            "manifest_job_id": job_id,
            "training_source": "online_feedback",
            "measurement_request_row_sha256": request["row_sha256"][job_id],
            "materialized_source_evidence_path": str(evidence_path),
            "materialized_source_evidence_sha256": evidence_sha,
        }
        if row["terminal_status"] in FAILURE_STATUSES:
            failure_path = Path(str(row.get("ap_report_path") or args.performance_state_jsonl))
            row = {
                **row,
                "failure_evidence_path": str(failure_path),
                "failure_evidence_sha256": sha256_file(failure_path),
            }
        finalized_rows.append(row)
    pending = [
        row
        for row in finalized_rows
        if row["terminal_status"] not in FAILURE_STATUSES | {"measured_success_gold"}
    ]
    if pending:
        raise ValueError(f"atomic batch still has {len(pending)} non-terminal rows")
    atomic_audit = finalize_atomic_batch(request, finalized_rows)
    result = {
        "rows": finalized_rows,
        "group_audit": [
            {
                "group_id": row["group_id"],
                "row_count": 1,
                "manifest_job_ids": [row["manifest_job_id"]],
                "terminal_statuses": [row["terminal_status"]],
                "all_terminal": True,
            }
            for row in finalized_rows
        ],
        "summary": {
            "measured": sum(row["terminal_status"] == "measured_success_gold" for row in finalized_rows),
            "failure": sum(row["terminal_status"] in FAILURE_STATUSES for row in finalized_rows),
            "pending": 0,
            "total": 4,
        },
    }
    write_dataset_outputs(
        result,
        args.output_dir,
        file_prefix="stage5_feedback_v2",
        output_schema="stage5_feedback_batch_v2",
    )
    (args.output_dir / "atomic_batch_audit.json").write_text(
        json.dumps(atomic_audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
