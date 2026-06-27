#!/usr/bin/env python3
"""Summarize Stage2 LUT coverage from latency/AP/energy JSONL rows."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.stage2.lut_productization import read_jsonl  # noqa: E402


SUMMARY_SCHEMA = "stage2_lut_coverage_summary_v1"
REPEAT_HEAVY_THRESHOLD = 0.30
BLOCKED_STATUSES = {"preflight_blocked", "skipped"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latency-rows", action="append", default=[])
    parser.add_argument("--ap-rows", action="append", default=[])
    parser.add_argument("--energy-rows", action="append", default=[])
    parser.add_argument("--quarantine-rows", action="append", default=[])
    parser.add_argument("--missing-artifact-rows", action="append", default=[])
    parser.add_argument("--job-state", action="append", default=[])
    parser.add_argument("--out-json", required=True)
    return parser.parse_args()


def read_many_jsonl(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(read_jsonl(path))
    return rows


def _kind_from_schema(row: dict[str, Any]) -> str:
    schema = str(row.get("schema") or "")
    if schema.startswith("latency_"):
        return "latency"
    if schema.startswith("ap_"):
        return "ap"
    if schema.startswith("energy_"):
        return "energy"
    return str(row.get("lut_kind") or "unknown")


def _width_key(row: dict[str, Any]) -> str:
    width = row.get("width")
    if isinstance(width, list):
        return "x".join(str(int(item)) for item in width)
    if width is None:
        return ""
    return str(width)


def _config_key(row: dict[str, Any]) -> tuple[str, ...]:
    return (
        str(row.get("model") or ""),
        str(row.get("dense_stage") or ""),
        str(row.get("optimized_scope") or ""),
        _width_key(row),
        str(row.get("quant_policy") or ""),
        str(row.get("schedule_policy") or ""),
        str(row.get("backend") or ""),
    )


def _axis_cell_key(kind: str, row: dict[str, Any]) -> tuple[str, ...]:
    base = _config_key(row)
    if kind == "ap":
        return (
            *base,
            str(row.get("backend") or ""),
            str(row.get("metric") or ""),
            str(row.get("dataset") or ""),
            str(row.get("eval_split") or ""),
        )
    return (
        *base,
        str(row.get("schedule_policy") or ""),
        str(row.get("backend") or ""),
    )


def latest_status_by_job(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        job_id = row.get("job_id")
        if job_id is None:
            continue
        latest[str(job_id)] = row
    return latest


def _repeat_cell_preview(
    *,
    latency_rows: list[dict[str, Any]],
    ap_rows: list[dict[str, Any]],
    energy_rows: list[dict[str, Any]],
    limit: int = 10,
) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, ...]] = Counter()
    examples: dict[tuple[str, ...], dict[str, Any]] = {}
    for rows in (latency_rows, ap_rows, energy_rows):
        for row in rows:
            if row.get("measurement_status") != "measured":
                continue
            kind = _kind_from_schema(row)
            key = (kind, *_axis_cell_key(kind, row))
            counts[key] += 1
            examples.setdefault(key, row)

    preview: list[dict[str, Any]] = []
    for key, count in counts.most_common(limit):
        if count <= 1:
            continue
        row = examples[key]
        preview.append(
            {
                "kind": key[0],
                "count": count,
                "extra_rows": count - 1,
                "model": row.get("model"),
                "dense_stage": row.get("dense_stage"),
                "width": row.get("width"),
                "quant_policy": row.get("quant_policy"),
                "schedule_policy": row.get("schedule_policy"),
                "backend": row.get("backend"),
                "config_id": row.get("config_id"),
            }
        )
    return preview


def build_coverage_summary(
    *,
    latency_rows: list[dict[str, Any]],
    ap_rows: list[dict[str, Any]],
    energy_rows: list[dict[str, Any]],
    quarantine_rows: list[dict[str, Any]] | None = None,
    missing_artifact_rows: list[dict[str, Any]] | None = None,
    job_state_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    quarantine_rows = [] if quarantine_rows is None else quarantine_rows
    missing_artifact_rows = [] if missing_artifact_rows is None else missing_artifact_rows
    job_state_rows = [] if job_state_rows is None else job_state_rows
    rows_by_kind = {
        "latency": latency_rows,
        "ap": ap_rows,
        "energy": energy_rows,
    }
    measured_by_kind = {
        kind: [row for row in rows if row.get("measurement_status") == "measured"]
        for kind, rows in rows_by_kind.items()
    }

    unique_cells = {
        kind: {_axis_cell_key(kind, row) for row in rows}
        for kind, rows in measured_by_kind.items()
    }
    measured_rows = [row for rows in measured_by_kind.values() for row in rows]
    raw_row_count = sum(len(rows) for rows in rows_by_kind.values())
    measured_row_count = len(measured_rows)
    unique_axis_cell_count = sum(len(cells) for cells in unique_cells.values())
    repeat_rows = max(0, measured_row_count - unique_axis_cell_count)
    repeat_ratio = (repeat_rows / raw_row_count) if raw_row_count else 0.0

    active_quarantine = [
        row for row in quarantine_rows if str(row.get("status") or "active") == "active"
    ]
    missing_artifact_configs = {
        str(row.get("config_id") or row.get("candidate_id"))
        for row in missing_artifact_rows
        if row.get("config_id") or row.get("candidate_id")
    }
    quarantined_configs = {
        str(row.get("config_id"))
        for row in active_quarantine
        if row.get("config_id")
    }
    latest_jobs = latest_status_by_job(job_state_rows)
    blocked_jobs = [
        row
        for row in latest_jobs.values()
        if str(row.get("status") or "") in BLOCKED_STATUSES
    ]
    latest_status_counts = Counter(
        str(row.get("status") or "unknown") for row in latest_jobs.values()
    )

    summary = {
        "schema": SUMMARY_SCHEMA,
        "raw_row_count": raw_row_count,
        "measured_row_count": measured_row_count,
        "latency_row_count": len(latency_rows),
        "energy_row_count": len(energy_rows),
        "ap_row_count": len(ap_rows),
        "unique_config_count": len({_config_key(row) for row in measured_rows}),
        "unique_latency_cells": len(unique_cells["latency"]),
        "unique_energy_cells": len(unique_cells["energy"]),
        "unique_ap_cells": len(unique_cells["ap"]),
        "unique_axis_cell_count": unique_axis_cell_count,
        "repeat_rows": repeat_rows,
        "repeat_ratio": repeat_ratio,
        "repeat_policy_label": (
            "repeat-heavy"
            if repeat_ratio > REPEAT_HEAVY_THRESHOLD
            else "coverage-first"
        ),
        "quarantine_row_count": len(quarantine_rows),
        "active_quarantine_row_count": len(active_quarantine),
        "quarantined_config_count": len(quarantined_configs),
        "quarantined_job_count": len(
            {str(row.get("job_id")) for row in active_quarantine if row.get("job_id")}
        ),
        "missing_artifact_row_count": len(missing_artifact_rows),
        "missing_artifact_config_count": len(missing_artifact_configs),
        "blocked_config_count": len(quarantined_configs | missing_artifact_configs),
        "job_state_row_count": len(job_state_rows),
        "latest_job_count": len(latest_jobs),
        "blocked_job_count": len(blocked_jobs),
        "latest_job_status_counts": dict(sorted(latest_status_counts.items())),
        "top_repeated_cells": _repeat_cell_preview(
            latency_rows=latency_rows,
            ap_rows=ap_rows,
            energy_rows=energy_rows,
        ),
    }
    return summary


def write_summary(path: str | Path, summary: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    summary = build_coverage_summary(
        latency_rows=read_many_jsonl(args.latency_rows),
        ap_rows=read_many_jsonl(args.ap_rows),
        energy_rows=read_many_jsonl(args.energy_rows),
        quarantine_rows=read_many_jsonl(args.quarantine_rows),
        missing_artifact_rows=read_many_jsonl(args.missing_artifact_rows),
        job_state_rows=read_many_jsonl(args.job_state),
    )
    write_summary(args.out_json, summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
