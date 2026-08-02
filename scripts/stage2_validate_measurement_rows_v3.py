#!/usr/bin/env python3
"""Validate paper-grade Stage2 measurement rows and emit final-frontier gold."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.stage2.measurement_contract_v3 import (  # noqa: E402
    build_artifact_registry,
    filter_final_frontier_rows,
    partition_measurement_rows,
    summarize_measurement_registry,
    validate_measurement_row,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--frontier-jsonl", type=Path, required=True)
    parser.add_argument("--artifact-registry-json", type=Path)
    parser.add_argument("--evidence-views-json", type=Path)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        try:
            rows.append(validate_measurement_row(payload))
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    rows = read_rows(args.rows_jsonl)
    summary = summarize_measurement_registry(rows)
    frontier_rows = filter_final_frontier_rows(rows)
    artifact_registry = build_artifact_registry(rows)
    evidence_views = partition_measurement_rows(rows)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_jsonl(args.frontier_jsonl, frontier_rows)
    if args.artifact_registry_json is not None:
        args.artifact_registry_json.parent.mkdir(parents=True, exist_ok=True)
        args.artifact_registry_json.write_text(
            json.dumps(artifact_registry, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.evidence_views_json is not None:
        args.evidence_views_json.parent.mkdir(parents=True, exist_ok=True)
        args.evidence_views_json.write_text(
            json.dumps(evidence_views, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
