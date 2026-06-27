#!/usr/bin/env python3
"""Validate Stage2 artifact_registry_v1 JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.stage2.artifact_registry import (  # noqa: E402
    summarize_artifact_registry,
    validate_artifact_registry_row,
)
from framework.stage2.lut_productization import read_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--summary-json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_jsonl(args.registry)
    for row in rows:
        validate_artifact_registry_row(row)
    summary = summarize_artifact_registry(rows)
    if args.summary_json:
        path = Path(args.summary_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
