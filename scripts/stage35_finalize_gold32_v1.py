#!/usr/bin/env python3
"""Finalize the 32-row Stage3.5 backend/AP supplement with Gold evidence gates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.stage3_finalize_gold96_v3 import finalize_gold_dataset, read_jsonl, write_dataset_outputs


MANIFEST_SCHEMA = "stage35_gold32_supplement_manifest_v1"
OUTPUT_SCHEMA = "stage35_gold32_final_v1"


def finalize_gold32(
    manifest: Mapping[str, Any],
    *,
    ap_plan_rows: Sequence[Mapping[str, Any]],
    performance_state_rows: Sequence[Mapping[str, Any]],
    ap_state_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return finalize_gold_dataset(
        manifest,
        ap_plan_rows=ap_plan_rows,
        performance_state_rows=performance_state_rows,
        ap_state_rows=ap_state_rows,
        manifest_schema=MANIFEST_SCHEMA,
        output_schema=OUTPUT_SCHEMA,
        expected_rows=32,
        expected_groups=8,
    )


def write_gold32_outputs(result: Mapping[str, Any], output_dir: str | Path) -> None:
    write_dataset_outputs(result, output_dir, file_prefix="gold32", output_schema=OUTPUT_SCHEMA)


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
    result = finalize_gold32(
        json.loads(args.manifest_json.read_text(encoding="utf-8")),
        ap_plan_rows=read_jsonl(args.ap_plan_jsonl),
        performance_state_rows=[row for path in args.performance_state_jsonl for row in read_jsonl(path)],
        ap_state_rows=[row for path in args.ap_state_jsonl for row in read_jsonl(path)],
    )
    write_gold32_outputs(result, args.output_dir)
    print(json.dumps(result["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
