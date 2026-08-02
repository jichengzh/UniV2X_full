#!/usr/bin/env python3
"""Build the Gold176 supplement numerical-sanity and full-AP plan."""

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


SOURCE_SCHEMA = "stage35_gold176_targeted_supplement_manifest_v1"
SCHEMA_VERSION = "stage35_gold176_ap_plan_v1"


def build_gold176_ap_plan(
    manifest: Mapping[str, Any],
    *,
    performance_jobs: Sequence[Mapping[str, Any]],
    performance_state_rows: Sequence[Mapping[str, Any]],
    output_root: str | Path,
) -> list[dict[str, Any]]:
    rows = gold96.build_ap_plan(
        manifest,
        performance_jobs=performance_jobs,
        performance_state_rows=performance_state_rows,
        pilot_root=output_root,
        manifest_schema=SOURCE_SCHEMA,
        expected_row_count=32,
    )
    return [{**row, "schema_version": SCHEMA_VERSION} for row in rows]


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
    rows = build_gold176_ap_plan(
        json.loads(args.manifest_json.read_text(encoding="utf-8")),
        performance_jobs=[
            row for path in args.performance_jobs_jsonl for row in gold96.read_jsonl(path)
        ],
        performance_state_rows=[
            row for path in args.performance_state_jsonl for row in gold96.read_jsonl(path)
        ],
        output_root=args.output_root,
    )
    gold96.write_outputs(rows, args.output_json, args.output_jsonl)
    counts: dict[str, int] = {}
    for row in rows:
        terminal = str(row.get("ap_terminal"))
        counts[terminal] = counts.get(terminal, 0) + 1
    print(
        json.dumps(
            {"schema_version": SCHEMA_VERSION, "row_count": len(rows), "terminal_counts": counts},
            sort_keys=True,
        )
    )
    return 0 if counts == {"ready": 32} else 1


if __name__ == "__main__":
    raise SystemExit(main())
