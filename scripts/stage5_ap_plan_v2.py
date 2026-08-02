#!/usr/bin/env python3
"""Build AP jobs for one four-genome Stage5 v2 atomic batch."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import stage3_gold96_ap_plan_v3 as gold96


def expected_row_count(manifest: dict) -> int:
    count = int(manifest.get("row_count", -1))
    jobs = manifest.get("jobs")
    if count not in {1, 2, 3, 4} or (
        isinstance(jobs, list) and len(jobs) != count
    ):
        raise ValueError("Stage5 AP manifest row_count must be between one and four")
    return count


def write_stage5_outputs(rows: list[dict], output_json: Path, output_jsonl: Path) -> None:
    normalized = [{**row, "schema_version": "stage5_ap_plan_v2"} for row in rows]
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(
            {
                "schema_version": "stage5_ap_plan_v2",
                "row_count": len(normalized),
                "jobs": normalized,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    output_jsonl.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in normalized
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--performance-jobs-jsonl", type=Path, required=True)
    parser.add_argument("--performance-state-jsonl", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest_json.read_text(encoding="utf-8"))
    row_count = expected_row_count(manifest)
    rows = gold96.build_ap_plan(
        manifest,
        performance_jobs=gold96.read_jsonl(args.performance_jobs_jsonl),
        performance_state_rows=gold96.read_jsonl(args.performance_state_jsonl),
        pilot_root=args.output_root,
        manifest_schema="stage5_performance_manifest_v2",
        expected_row_count=row_count,
    )
    rows = [{**row, "schema_version": "stage5_ap_plan_v2"} for row in rows]
    write_stage5_outputs(rows, args.output_json, args.output_jsonl)
    counts = {}
    for row in rows:
        terminal = str(row.get("ap_terminal"))
        counts[terminal] = counts.get(terminal, 0) + 1
    print(json.dumps({"row_count": len(rows), "terminal_counts": counts}, sort_keys=True))
    return 0 if counts == {"ready": len(rows)} else 1


if __name__ == "__main__":
    raise SystemExit(main())
