#!/usr/bin/env python3
"""Finalize a Stage5 feedback round into SHA-bound Gold rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.stage3_finalize_gold96_v3 import (
    FAILURE_STATUSES,
    finalize_gold_dataset,
    read_jsonl,
    sha256_file,
    write_dataset_outputs,
)


MANIFEST_SCHEMA = "stage5_performance_manifest_v1"
OUTPUT_SCHEMA = "stage5_feedback_rows_v1"
REQUEST_IDENTITY_FIELDS = (
    "genome",
    "strategy_id",
    "capability_digest",
    "source_status",
    "source_contract",
    "source_evidence_sha256",
)
MANIFEST_STABLE_FIELDS = (
    "manifest_job_id",
    "group_id",
    "model",
    "width",
    "q_mode",
    "capability_profile_id",
    "dispatch_key",
    "genome",
    "strategy_id",
    "capability_digest",
)


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_manifest_request_binding(
    manifest: dict, request: dict
) -> tuple[dict[str, dict], dict[str, dict]]:
    jobs = [dict(row) for row in manifest.get("jobs") or []]
    job_by_id = {str(row["manifest_job_id"]): row for row in jobs}
    request_rows = [dict(row) for row in request.get("rows") or []]
    requested_subset = [
        row for row in request_rows if str(row.get("manifest_job_id") or "") in job_by_id
    ]
    request_by_id = {str(row["manifest_job_id"]): row for row in requested_subset}
    if not jobs or set(job_by_id) != set(request_by_id):
        raise ValueError("performance manifest rows do not exactly match request subset")
    group_ids = {str(row["group_id"]) for row in requested_subset}
    reduced_subset = {
        "schema_version": request.get("schema_version"),
        "group_count": len(group_ids),
        "row_count": len(requested_subset),
        "rows": requested_subset,
    }
    derived_subset = {
        **request,
        "group_count": len(group_ids),
        "row_count": len(requested_subset),
        "rows": requested_subset,
    }
    accepted_request_hashes = {
        canonical_sha256(reduced_subset),
        canonical_sha256(derived_subset),
    }
    if len(requested_subset) == len(request_rows):
        accepted_request_hashes.add(canonical_sha256(request))
    if manifest.get("source_request_sha256") not in accepted_request_hashes:
        raise ValueError("performance manifest source request SHA mismatch")
    if int(manifest.get("group_count", -1)) != len(group_ids) or int(
        manifest.get("row_count", -1)
    ) != len(requested_subset):
        raise ValueError("performance manifest request cardinality drift")
    for row_id, requested in request_by_id.items():
        job = job_by_id[row_id]
        if any(job.get(field) != requested.get(field) for field in MANIFEST_STABLE_FIELDS):
            raise ValueError(f"performance manifest job identity drift for {row_id}")
        planned_contract = requested.get("source_contract") or {}
        materialized_contract = job.get("source_contract") or {}
        if not isinstance(planned_contract, dict) or not isinstance(materialized_contract, dict):
            raise ValueError(f"performance manifest source contract invalid for {row_id}")
        if any(materialized_contract.get(key) != value for key, value in planned_contract.items()):
            raise ValueError(f"performance manifest source contract drift for {row_id}")
        if job.get("source_status") != "ready":
            raise ValueError(f"performance manifest source is not ready for {row_id}")
    return request_by_id, job_by_id


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
    groups = int(manifest.get("group_count", -1))
    rows = int(manifest.get("row_count", -1))
    if groups <= 0 or rows != groups * 4:
        raise ValueError("Stage5 finalization requires complete four-arm groups")
    request = json.loads(args.measurement_request_json.read_text(encoding="utf-8"))
    request_by_id, job_by_id = validate_manifest_request_binding(manifest, request)
    finalized = finalize_gold_dataset(
        manifest,
        ap_plan_rows=read_jsonl(args.ap_plan_jsonl),
        performance_state_rows=read_jsonl(args.performance_state_jsonl),
        ap_state_rows=read_jsonl(args.ap_state_jsonl),
        manifest_schema=MANIFEST_SCHEMA,
        output_schema=OUTPUT_SCHEMA,
        expected_rows=rows,
        expected_groups=groups,
    )
    state_path = args.performance_state_jsonl.resolve()
    enriched_rows = []
    for source in finalized["rows"]:
        row = dict(source)
        row_id = str(row["manifest_job_id"])
        requested = request_by_id.get(row_id)
        job = job_by_id.get(row_id)
        if requested is None or job is None:
            raise ValueError(f"feedback source identity missing for {row_id}")
        if any(field not in requested for field in REQUEST_IDENTITY_FIELDS):
            raise ValueError(f"measurement request source contract incomplete for {row_id}")
        source_evidence_path = Path(str(job.get("source_evidence_path") or ""))
        source_evidence_sha = str(job.get("source_evidence_sha256") or "")
        if (
            not source_evidence_path.is_file()
            or len(source_evidence_sha) != 64
            or sha256_file(source_evidence_path) != source_evidence_sha
        ):
            raise ValueError(f"materialized source evidence mismatch for {row_id}")
        source_evidence = json.loads(source_evidence_path.read_text(encoding="utf-8"))
        if source_evidence.get("source_plan_sha256") != requested["source_evidence_sha256"]:
            raise ValueError(f"materialized source plan drift for {row_id}")
        row = {
            **row,
            **{field: requested[field] for field in REQUEST_IDENTITY_FIELDS},
            "measurement_request_row_sha256": canonical_sha256(requested),
            "materialized_source_evidence_path": str(source_evidence_path),
            "materialized_source_evidence_sha256": source_evidence_sha,
        }
        failure_path = None
        failure_sha = None
        if row["terminal_status"] in FAILURE_STATUSES:
            ap_report = Path(str(row.get("ap_report_path") or ""))
            if ap_report.is_file() and len(str(row.get("ap_report_sha256") or "")) == 64:
                failure_path = str(ap_report)
                failure_sha = str(row["ap_report_sha256"])
                if row["terminal_status"] == "numerical_feasibility_failure":
                    report = json.loads(ap_report.read_text(encoding="utf-8"))
                    blockers = report.get("feasibility_blockers")
                    if isinstance(blockers, list) and blockers:
                        row = {**row, "failure_reason": ";".join(map(str, blockers))}
            else:
                failure_path = str(state_path)
                failure_sha = sha256_file(state_path)
        enriched_rows.append(
            {
                **row,
                "failure_evidence_path": failure_path,
                "failure_evidence_sha256": failure_sha,
            }
        )
    result = {**finalized, "rows": enriched_rows}
    write_dataset_outputs(
        result,
        args.output_dir,
        file_prefix="stage5_feedback",
        output_schema=OUTPUT_SCHEMA,
    )
    summary = result["summary"]
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if int(summary.get("pending", -1)) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
