#!/usr/bin/env python3
"""Finalize the 32-row Gold176 targeted supplement from performance/AP evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.stage3_finalize_gold96_v3 import (
    finalize_gold_dataset,
    read_jsonl,
    write_dataset_outputs,
)


MANIFEST_SCHEMA = "stage35_gold176_targeted_supplement_manifest_v1"
OUTPUT_SCHEMA = "stage35_targeted32_final_v1"
ARMS = {
    ("tvm_auto", "fp16"),
    ("tvm_auto", "int8"),
    ("trt_engine", "fp16"),
    ("trt_engine", "int8"),
}
REPAIR_SOURCE = "stage3_tvm_int8_repair_v3"


def _validate_manifest_contract(manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    jobs = manifest.get("jobs")
    if manifest.get("schema_version") != MANIFEST_SCHEMA or not isinstance(jobs, list) or len(jobs) != 32:
        raise ValueError(f"expected {MANIFEST_SCHEMA} with exactly 32 jobs")
    ids = [str(job.get("job_id") or "") for job in jobs]
    if any(not value for value in ids) or len(set(ids)) != 32:
        raise ValueError("targeted32 job identities must be unique")
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for job in jobs:
        if job.get("split") != "train":
            raise ValueError("targeted32 jobs must use split=train")
        grouped.setdefault(str(job.get("group_id") or ""), []).append(job)
    if len(grouped) != 8:
        raise ValueError("targeted32 must contain eight groups")
    for group_id, group_jobs in grouped.items():
        arms = {(str(job.get("dispatch_key")), str(job.get("q_mode"))) for job in group_jobs}
        if len(group_jobs) != 4 or arms != ARMS:
            raise ValueError(f"targeted32 group is not a complete four-arm group: {group_id}")
    return jobs


def select_repair_overlays(
    ap_plan_rows: Sequence[Mapping[str, Any]],
    performance_state_rows: Sequence[Mapping[str, Any]],
    ap_state_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    repair_plan = [
        row for row in ap_plan_rows
        if row.get("q") == "int8" and "h800-tvm" in str(row.get("profile") or "")
    ]
    if not repair_plan or (not performance_state_rows and not ap_state_rows):
        return [dict(row) for row in performance_state_rows], [dict(row) for row in ap_state_rows]
    performance_ids = {str(row["performance_job_id"]) for row in repair_plan}
    manifest_ids = {str(row["manifest_job_id"]) for row in repair_plan}
    repair_performance = [
        dict(row) for row in performance_state_rows
        if str(row.get("manifest_job_id") or row.get("job_id") or "") in performance_ids
        and row.get("source") == REPAIR_SOURCE
        and row.get("status") == "success"
    ]
    repair_ap = [
        dict(row) for row in ap_state_rows
        if str(row.get("manifest_job_id") or row.get("job_id") or "") in manifest_ids
        and row.get("source") == REPAIR_SOURCE
        and row.get("stage") == "full"
        and row.get("status") == "success"
    ]
    if {str(row.get("manifest_job_id") or row.get("job_id")) for row in repair_performance} != performance_ids:
        raise ValueError("each TVM INT8 row requires one explicit repaired performance overlay")
    if {str(row.get("manifest_job_id") or row.get("job_id")) for row in repair_ap} != manifest_ids:
        raise ValueError("each TVM INT8 row requires one explicit repaired AP overlay")
    if len(repair_performance) != len(performance_ids) or len(repair_ap) != len(manifest_ids):
        raise ValueError("duplicate TVM INT8 repair overlays are forbidden")
    normal_performance = [
        dict(row) for row in performance_state_rows
        if str(row.get("manifest_job_id") or row.get("job_id") or "") not in performance_ids
    ]
    normal_ap = [
        dict(row) for row in ap_state_rows
        if str(row.get("manifest_job_id") or row.get("job_id") or "") not in manifest_ids
    ]
    return [*normal_performance, *repair_performance], [*normal_ap, *repair_ap]


def finalize_targeted32(
    manifest: Mapping[str, Any],
    *,
    ap_plan_rows: Sequence[Mapping[str, Any]],
    performance_state_rows: Sequence[Mapping[str, Any]],
    ap_state_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    jobs = _validate_manifest_contract(manifest)
    jobs_by_id = {str(job["job_id"]): job for job in jobs}
    enriched_plan = [
        {
            **dict(row),
            "q": row.get("q") or jobs_by_id[str(row["manifest_job_id"])].get("q_mode"),
            "profile": row.get("profile") or jobs_by_id[str(row["manifest_job_id"])].get("capability_profile_id"),
        }
        for row in ap_plan_rows
    ]
    selected_performance, selected_ap = select_repair_overlays(
        enriched_plan, performance_state_rows, ap_state_rows
    )
    result = finalize_gold_dataset(
        manifest,
        ap_plan_rows=enriched_plan,
        performance_state_rows=selected_performance,
        ap_state_rows=selected_ap,
        manifest_schema=MANIFEST_SCHEMA,
        output_schema=OUTPUT_SCHEMA,
        expected_rows=32,
        expected_groups=8,
    )
    rows = [
        {
            **dict(row),
            "split": str(jobs_by_id[str(row["manifest_job_id"])].get("split") or "train"),
            "width_stratum": str(
                jobs_by_id[str(row["manifest_job_id"])].get("width_stratum")
                or "targeted_failure_boundary"
            ),
        }
        for row in result["rows"]
    ]
    return {**result, "rows": rows, "manifest": manifest}


def write_targeted32_outputs(result: Mapping[str, Any], output_dir: str | Path) -> None:
    write_dataset_outputs(
        result,
        output_dir,
        file_prefix="targeted32",
        output_schema=OUTPUT_SCHEMA,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--ap-plan-jsonl", type=Path, required=True)
    parser.add_argument("--performance-state-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--ap-state-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = finalize_targeted32(
        json.loads(args.manifest_json.read_text(encoding="utf-8")),
        ap_plan_rows=read_jsonl(args.ap_plan_jsonl),
        performance_state_rows=[
            row for path in args.performance_state_jsonl for row in read_jsonl(path)
        ],
        ap_state_rows=[row for path in args.ap_state_jsonl for row in read_jsonl(path)],
    )
    write_targeted32_outputs(result, args.output_dir)
    print(json.dumps(result["summary"], sort_keys=True))
    return 0 if result["summary"] == {"measured": 32, "failure": 0, "pending": 0, "total": 32} else 1


if __name__ == "__main__":
    raise SystemExit(main())
