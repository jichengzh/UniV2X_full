#!/usr/bin/env python3
"""Build a strict AP plan for Gold128 targeted four-arm supplements."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import stage3_gold96_ap_plan_v3 as gold96


SOURCE_SCHEMA = "stage35_gold128_targeted_supplement_manifest_v1"
SCHEMA_VERSION = "stage35_gold128_targeted_ap_plan_v1"
EXPECTED_ARMS = {
    ("tvm_auto", "fp16"),
    ("tvm_auto", "int8"),
    ("trt_engine", "fp16"),
    ("trt_engine", "int8"),
}


def validate_targeted_manifest(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    jobs = manifest.get("jobs")
    if manifest.get("schema_version") != SOURCE_SCHEMA or not isinstance(jobs, list) or not jobs:
        raise ValueError(f"expected non-empty {SOURCE_SCHEMA}")

    rows = [dict(row) for row in jobs]
    if any(str(row.get("split")) != "train" for row in rows):
        raise ValueError("targeted supplement rows must all be train")

    manifest_ids = [str(row.get("job_id") or "") for row in rows]
    if any(not value for value in manifest_ids) or len(set(manifest_ids)) != len(manifest_ids):
        raise ValueError("targeted supplement manifest job_id values must be non-empty and unique")

    grouped: dict[str, set[tuple[str, str]]] = {}
    for row in rows:
        group_id = str(row.get("group_id") or "")
        if not group_id:
            raise ValueError("targeted supplement row missing group_id")
        arm = (str(row.get("dispatch_key") or ""), str(row.get("q_mode") or row.get("q") or ""))
        grouped.setdefault(group_id, set()).add(arm)
    if any(arms != EXPECTED_ARMS for arms in grouped.values()) or len(rows) != 4 * len(grouped):
        raise ValueError("targeted supplement must contain complete four-arm groups")
    return rows


def _validate_performance_job_coverage(
    manifest_rows: Sequence[Mapping[str, Any]],
    performance_jobs: Sequence[Mapping[str, Any]],
) -> None:
    expected = {str(row["job_id"]) for row in manifest_rows}
    counts = {manifest_id: 0 for manifest_id in expected}
    for job in performance_jobs:
        manifest_id = str(job.get("manifest_job_id") or "")
        if manifest_id in counts:
            counts[manifest_id] += 1
    invalid = {manifest_id: count for manifest_id, count in counts.items() if count != 1}
    if invalid:
        raise ValueError(f"expected exactly one performance job per targeted manifest row: {invalid}")


def build_targeted_ap_plan(
    manifest: Mapping[str, Any],
    *,
    performance_jobs: Sequence[Mapping[str, Any]],
    performance_state_rows: Sequence[Mapping[str, Any]],
    output_root: str | Path,
) -> list[dict[str, Any]]:
    manifest_rows = validate_targeted_manifest(manifest)
    _validate_performance_job_coverage(manifest_rows, performance_jobs)
    rows = gold96.build_ap_plan(
        manifest,
        performance_jobs=performance_jobs,
        performance_state_rows=performance_state_rows,
        pilot_root=output_root,
        manifest_schema=SOURCE_SCHEMA,
        expected_row_count=len(manifest_rows),
    )
    not_ready = {
        str(row["manifest_job_id"]): str(row.get("ap_terminal"))
        for row in rows
        if row.get("ap_terminal") != "ready"
    }
    if not_ready:
        raise ValueError(
            f"all {len(manifest_rows)} targeted AP rows must be ready; non-ready rows: {not_ready}"
        )
    return [{**row, "schema_version": SCHEMA_VERSION} for row in rows]


def write_outputs(
    rows: Sequence[Mapping[str, Any]], output_json: str | Path, output_jsonl: str | Path
) -> tuple[Path, Path]:
    json_path, jsonl_path = Path(output_json), Path(output_jsonl)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": SCHEMA_VERSION, "row_count": len(rows), "jobs": list(rows)}
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    jsonl_path.write_text(
        "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return json_path, jsonl_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--performance-jobs-jsonl", type=Path, nargs="+", required=True)
    parser.add_argument("--performance-state-jsonl", type=Path, nargs="+", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = build_targeted_ap_plan(
        json.loads(args.manifest_json.read_text(encoding="utf-8")),
        performance_jobs=[
            row for path in args.performance_jobs_jsonl for row in gold96.read_jsonl(path)
        ],
        performance_state_rows=[
            row for path in args.performance_state_jsonl for row in gold96.read_jsonl(path)
        ],
        output_root=args.output_root,
    )
    output_json, output_jsonl = write_outputs(rows, args.output_json, args.output_jsonl)
    print(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "row_count": len(rows),
                "output_json": str(output_json),
                "output_jsonl": str(output_jsonl),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
