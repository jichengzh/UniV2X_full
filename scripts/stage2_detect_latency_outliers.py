#!/usr/bin/env python3
"""Detect latency repeat and multi-run outliers in Stage2 LUT rows."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.stage2.lut_productization import read_jsonl  # noqa: E402
from framework.stage2.outlier_policy import detect_latency_outliers  # noqa: E402


CSV_FIELDS = [
    "row_id",
    "run_id",
    "config_id",
    "model",
    "candidate_id",
    "schedule_policy",
    "latency_p50_ms",
    "latency_min_ms",
    "latency_max_ms",
    "quality_flag",
    "claim_status",
    "reasons",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latency-rows", action="append", required=True)
    parser.add_argument("--grade", choices=("calibration", "paper"), default="calibration")
    parser.add_argument("--historical-anchors-json")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-csv")
    parser.add_argument("--fail-on-unstable-paper", action="store_true")
    return parser.parse_args()


def _read_many(paths: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in paths:
        rows.extend(read_jsonl(path))
    return rows


def _anchors(path: str | None) -> dict[str, float]:
    if not path:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {str(key): float(value) for key, value in data.items()}


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["reasons"] = ";".join(str(item) for item in row.get("reasons", []))
            writer.writerow({field: out.get(field) for field in CSV_FIELDS})


def main() -> int:
    args = parse_args()
    report = detect_latency_outliers(
        _read_many(args.latency_rows),
        grade=args.grade,
        historical_anchors_ms=_anchors(args.historical_anchors_json),
    )
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.out_csv:
        _write_csv(Path(args.out_csv), list(report["rows"]))
    print(
        json.dumps(
            {
                "schema": "stage2_latency_outlier_cli_summary_v1",
                "out_json": str(out_json),
                "unstable_rows": report["unstable_rows"],
                "claimable_rows": report["claimable_rows"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    if args.fail_on_unstable_paper and args.grade == "paper" and report["unstable_rows"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
