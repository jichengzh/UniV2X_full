#!/usr/bin/env python3
"""Build artifact_registry_v1.jsonl from canonical Stage2 LUT rows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.stage2.artifact_registry import (  # noqa: E402
    build_artifact_registry_rows,
    summarize_artifact_registry,
    validate_artifact_registry_row,
)
from framework.stage2.lut_productization import read_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latency-rows", action="append", default=[])
    parser.add_argument("--ap-rows", action="append", default=[])
    parser.add_argument("--energy-rows", action="append", default=[])
    parser.add_argument("--quarantine-rows", action="append", default=[])
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--summary-json")
    return parser.parse_args()


def _read_many(paths: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in paths:
        rows.extend(read_jsonl(path))
    return rows


def _write_registry(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            validate_artifact_registry_row(row)
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()
    rows = build_artifact_registry_rows(
        latency_rows=_read_many(args.latency_rows),
        ap_rows=_read_many(args.ap_rows),
        energy_rows=_read_many(args.energy_rows),
        quarantine_rows=_read_many(args.quarantine_rows),
    )
    out_jsonl = Path(args.out_jsonl)
    _write_registry(out_jsonl, rows)
    summary = summarize_artifact_registry(rows)
    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "schema": "stage2_artifact_registry_build_summary_v1",
                "out": str(out_jsonl),
                **summary,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
