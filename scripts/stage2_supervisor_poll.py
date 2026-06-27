#!/usr/bin/env python3
"""Build the Stage2 LUT supervisor polling dashboard and readiness gate."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.stage2_summarize_lut_coverage import (  # noqa: E402
    build_coverage_summary,
    latest_status_by_job,
    read_many_jsonl,
)


GATE_SCHEMA = "stage2_supervisor_readiness_gate_v1"
READY_STATUSES = {"queued", "stale_lock_requeued"}
RUNNING_STATUSES = {"running", "payload_ready", "salvaged"}
TERMINAL_STATUSES = {"succeeded", "failed", "skipped", "preflight_blocked"}
LOW_WATER_LATENCY_READY_JOBS = 10
AXIS_LAG_RATIO = 0.40
REPEAT_RATIO_THRESHOLD = 0.30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latency-rows", action="append", default=[])
    parser.add_argument("--ap-rows", action="append", default=[])
    parser.add_argument("--energy-rows", action="append", default=[])
    parser.add_argument("--job-plan", action="append", default=[])
    parser.add_argument("--job-state", action="append", default=[])
    parser.add_argument("--axis-gap-report", action="append", default=[])
    parser.add_argument("--quarantine-rows", action="append", default=[])
    parser.add_argument("--missing-artifact-rows", action="append", default=[])
    parser.add_argument("--previous-quarantine-rows", action="append", default=[])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--min-latency-ready-jobs",
        type=int,
        default=LOW_WATER_LATENCY_READY_JOBS,
    )
    parser.add_argument("--axis-lag-ratio", type=float, default=AXIS_LAG_RATIO)
    parser.add_argument("--repeat-ratio-threshold", type=float, default=REPEAT_RATIO_THRESHOLD)
    return parser.parse_args()


def _job_kind(row: dict[str, Any]) -> str:
    lut_kind = row.get("lut_kind")
    if lut_kind:
        return str(lut_kind)
    job_id = str(row.get("job_id") or "")
    if ":" in job_id:
        return job_id.split(":", 1)[0]
    return "unknown"


def _active_quarantine_configs(rows: Iterable[dict[str, Any]]) -> set[str]:
    configs: set[str] = set()
    for row in rows:
        if str(row.get("status") or "active") != "active":
            continue
        config_id = row.get("config_id")
        if config_id:
            configs.add(str(config_id))
    return configs


def _status_for_plan_job(
    row: dict[str, Any],
    latest_jobs: dict[str, dict[str, Any]],
) -> str:
    job_id = row.get("job_id")
    if job_id is None:
        return "queued"
    return str(latest_jobs.get(str(job_id), {}).get("status") or "queued")


def build_queue_summary(
    *,
    job_plan_rows: list[dict[str, Any]],
    latest_jobs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    by_kind: dict[str, Counter[str]] = {}
    for row in job_plan_rows:
        kind = _job_kind(row)
        by_kind.setdefault(kind, Counter())
        status = _status_for_plan_job(row, latest_jobs)
        if status in READY_STATUSES:
            by_kind[kind]["ready_jobs"] += 1
        elif status in RUNNING_STATUSES:
            by_kind[kind]["running_jobs"] += 1
        elif status in TERMINAL_STATUSES:
            by_kind[kind][f"{status}_jobs"] += 1
        else:
            by_kind[kind][f"{status}_jobs"] += 1
        by_kind[kind]["planned_jobs"] += 1

    summary: dict[str, Any] = {"planned_jobs": len(job_plan_rows)}
    for kind in ("latency", "energy", "ap", "artifact", "unknown"):
        counts = by_kind.get(kind, Counter())
        for metric in (
            "planned_jobs",
            "ready_jobs",
            "running_jobs",
            "succeeded_jobs",
            "failed_jobs",
            "skipped_jobs",
            "preflight_blocked_jobs",
        ):
            summary[f"{kind}_{metric}"] = int(counts.get(metric, 0))
    return summary


def build_recommendations(
    *,
    coverage_summary: dict[str, Any],
    queue: dict[str, Any],
    quarantine_growth: int,
    min_latency_ready_jobs: int,
    axis_lag_ratio: float,
    repeat_ratio_threshold: float,
) -> list[dict[str, Any]]:
    recommendations: list[dict[str, Any]] = []
    repeat_ratio = float(coverage_summary.get("repeat_ratio") or 0.0)
    latency_cells = int(coverage_summary.get("unique_latency_cells") or 0)
    energy_cells = int(coverage_summary.get("unique_energy_cells") or 0)
    ap_cells = int(coverage_summary.get("unique_ap_cells") or 0)
    latency_ready_jobs = int(queue.get("latency_ready_jobs") or 0)

    if repeat_ratio > repeat_ratio_threshold:
        recommendations.append(
            {
                "code": "repeat_ratio_high",
                "severity": "stop",
                "condition": f"repeat_ratio={repeat_ratio:.3f} > {repeat_ratio_threshold:.3f}",
                "action": "stop repeat, generate new candidates",
            }
        )
    if latency_ready_jobs < min_latency_ready_jobs:
        recommendations.append(
            {
                "code": "latency_queue_low",
                "severity": "warn",
                "condition": (
                    f"latency_ready_jobs={latency_ready_jobs} < "
                    f"{min_latency_ready_jobs}"
                ),
                "action": "ask artifact-agent for next batch",
            }
        )
    if latency_cells and energy_cells < axis_lag_ratio * latency_cells:
        recommendations.append(
            {
                "code": "energy_axis_lag",
                "severity": "warn",
                "condition": (
                    f"energy_cells={energy_cells} < "
                    f"{axis_lag_ratio:.2f} * latency_cells={latency_cells}"
                ),
                "action": "prioritize energy subset",
            }
        )
    if latency_cells and ap_cells < axis_lag_ratio * latency_cells:
        recommendations.append(
            {
                "code": "ap_axis_lag",
                "severity": "warn",
                "condition": (
                    f"ap_cells={ap_cells} < "
                    f"{axis_lag_ratio:.2f} * latency_cells={latency_cells}"
                ),
                "action": "prioritize AP source/eval",
            }
        )
    if quarantine_growth > 0:
        recommendations.append(
            {
                "code": "quarantine_growth",
                "severity": "stop",
                "condition": f"active quarantine configs grew by {quarantine_growth}",
                "action": "pause related workdir/template",
            }
        )
    return recommendations


def read_axis_gap_reports(paths: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        item = Path(path)
        if not item.exists():
            continue
        if item.suffix.lower() == ".csv":
            with item.open(encoding="utf-8", newline="") as handle:
                rows.extend(dict(row) for row in csv.DictReader(handle))
            continue
        payload = json.loads(item.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            rows.extend(row for row in payload if isinstance(row, dict))
        elif isinstance(payload, dict):
            inner = payload.get("rows")
            if isinstance(inner, list):
                rows.extend(row for row in inner if isinstance(row, dict))
    return rows


def build_top_missing_axes(rows: list[dict[str, Any]], *, limit: int = 10) -> list[dict[str, Any]]:
    healthy = {
        "",
        "queued",
        "queued_repeat",
        "measured",
        "measured_exists",
        "not_required",
        "not_applicable",
        "required",
        "unknown",
        "not_evaluated_by_energy_generator",
    }
    counts: Counter[tuple[str, str]] = Counter()
    examples: dict[tuple[str, str], list[str]] = {}
    for row in rows:
        candidate_id = str(row.get("candidate_id") or row.get("config_id") or row.get("label") or "unknown")
        for axis in ("latency", "energy", "ap"):
            status = str(row.get(f"{axis}_status") or "").strip()
            if status in healthy:
                continue
            key = (axis, status)
            counts[key] += 1
            examples.setdefault(key, [])
            if len(examples[key]) < 5 and candidate_id not in examples[key]:
                examples[key].append(candidate_id)
    output: list[dict[str, Any]] = []
    for (axis, status), count in sorted(counts.items(), key=lambda item: (-item[1], item[0][0], item[0][1]))[:limit]:
        output.append(
            {
                "axis": axis,
                "status": status,
                "count": int(count),
                "examples": examples.get((axis, status), []),
            }
        )
    return output


def _decision(recommendations: list[dict[str, Any]]) -> str:
    if any(item.get("severity") == "stop" for item in recommendations):
        return "NO_GO"
    if recommendations:
        return "CONDITIONAL_GO"
    return "GO"


def _dashboard_rows(gate: dict[str, Any]) -> list[dict[str, str]]:
    coverage = gate["coverage_summary"]
    queue = gate["queue"]
    quarantine = gate["quarantine"]
    metric_items: list[tuple[str, Any, str]] = [
        ("raw_row_count", coverage.get("raw_row_count"), "all LUT rows read"),
        ("unique_config_count", coverage.get("unique_config_count"), "axis-independent configs"),
        ("unique_latency_cells", coverage.get("unique_latency_cells"), "measured latency cells"),
        ("unique_energy_cells", coverage.get("unique_energy_cells"), "measured energy cells"),
        ("unique_ap_cells", coverage.get("unique_ap_cells"), "measured AP cells"),
        ("repeat_rows", coverage.get("repeat_rows"), "extra measured rows beyond unique cells"),
        ("repeat_ratio", coverage.get("repeat_ratio"), coverage.get("repeat_policy_label")),
        ("blocked_job_count", coverage.get("blocked_job_count"), "latest state only"),
        ("blocked_config_count", coverage.get("blocked_config_count"), "quarantine or missing artifact"),
        ("missing_artifact_config_count", coverage.get("missing_artifact_config_count"), "artifact gap"),
        ("quarantined_config_count", quarantine.get("active_config_count"), "active quarantine"),
        ("quarantine_growth", quarantine.get("growth"), "current minus previous active configs"),
        ("latency_ready_jobs", queue.get("latency_ready_jobs"), "low-water gate"),
        ("energy_ready_jobs", queue.get("energy_ready_jobs"), "queue ready"),
        ("ap_ready_jobs", queue.get("ap_ready_jobs"), "queue ready"),
        ("recommendation_count", len(gate["recommendations"]), "supervisor actions"),
        ("top_missing_axes_count", len(gate.get("top_missing_axes", [])), "axis gap groups"),
    ]
    rows: list[dict[str, str]] = []
    for metric, value, detail in metric_items:
        rows.append(
            {
                "metric": metric,
                "value": f"{value:.6f}" if isinstance(value, float) else str(value),
                "detail": "" if detail is None else str(detail),
            }
        )
    return rows


def write_dashboard_csv(path: str | Path, gate: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value", "detail"])
        writer.writeheader()
        writer.writerows(_dashboard_rows(gate))


def write_report(path: str | Path, gate: dict[str, Any]) -> None:
    coverage = gate["coverage_summary"]
    queue = gate["queue"]
    quarantine = gate["quarantine"]
    lines = [
        "# Stage2 LUT Supervisor Report",
        "",
        f"- decision: {gate['decision']}",
        f"- repeat_ratio: {coverage['repeat_ratio']:.3f}",
        f"- unique configs: {coverage['unique_config_count']}",
        f"- latency/AP/energy cells: {coverage['unique_latency_cells']} / {coverage['unique_ap_cells']} / {coverage['unique_energy_cells']}",
        f"- latency ready jobs: {queue['latency_ready_jobs']}",
        f"- blocked jobs: {coverage['blocked_job_count']}",
        f"- active quarantined configs: {quarantine['active_config_count']}",
        "",
        "## Recommendations",
    ]
    if gate["recommendations"]:
        for item in gate["recommendations"]:
            lines.append(
                f"- {item['code']}: {item['action']} ({item['condition']})"
            )
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Latest Job Status Counts",
        ]
    )
    for status, count in sorted(gate["latest_job_status_counts"].items()):
        lines.append(f"- {status}: {count}")
    lines.extend(["", "## Top Missing Axes"])
    if gate.get("top_missing_axes"):
        for item in gate["top_missing_axes"]:
            examples = ", ".join(item.get("examples", []))
            lines.append(
                f"- {item['axis']} {item['status']}: {item['count']} ({examples})"
            )
    else:
        lines.append("- none")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_gate_json(path: str | Path, gate: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(gate, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_supervisor_gate(
    *,
    latency_rows: list[dict[str, Any]],
    ap_rows: list[dict[str, Any]],
    energy_rows: list[dict[str, Any]],
    job_plan_rows: list[dict[str, Any]],
    job_state_rows: list[dict[str, Any]],
    axis_gap_rows: list[dict[str, Any]],
    quarantine_rows: list[dict[str, Any]],
    missing_artifact_rows: list[dict[str, Any]],
    previous_quarantine_rows: list[dict[str, Any]] | None,
    min_latency_ready_jobs: int = LOW_WATER_LATENCY_READY_JOBS,
    axis_lag_ratio: float = AXIS_LAG_RATIO,
    repeat_ratio_threshold: float = REPEAT_RATIO_THRESHOLD,
) -> dict[str, Any]:
    coverage = build_coverage_summary(
        latency_rows=latency_rows,
        ap_rows=ap_rows,
        energy_rows=energy_rows,
        quarantine_rows=quarantine_rows,
        missing_artifact_rows=missing_artifact_rows,
        job_state_rows=job_state_rows,
    )
    latest_jobs = latest_status_by_job(job_state_rows)
    latest_status_counts = Counter(
        str(row.get("status") or "unknown") for row in latest_jobs.values()
    )
    queue = build_queue_summary(job_plan_rows=job_plan_rows, latest_jobs=latest_jobs)
    active_configs = _active_quarantine_configs(quarantine_rows)
    previous_available = previous_quarantine_rows is not None
    previous_active_configs = (
        _active_quarantine_configs(previous_quarantine_rows)
        if previous_available
        else set(active_configs)
    )
    quarantine = {
        "active_config_count": len(active_configs),
        "previous_active_config_count": len(previous_active_configs),
        "previous_available": previous_available,
        "growth": max(0, len(active_configs) - len(previous_active_configs)),
        "active_configs": sorted(active_configs),
    }
    recommendations = build_recommendations(
        coverage_summary=coverage,
        queue=queue,
        quarantine_growth=int(quarantine["growth"]),
        min_latency_ready_jobs=min_latency_ready_jobs,
        axis_lag_ratio=axis_lag_ratio,
        repeat_ratio_threshold=repeat_ratio_threshold,
    )
    top_missing_axes = build_top_missing_axes(axis_gap_rows)
    gate = {
        "schema": GATE_SCHEMA,
        "decision": _decision(recommendations),
        "coverage_summary": coverage,
        "queue": queue,
        "quarantine": quarantine,
        "latest_job_status": dict(sorted(latest_jobs.items())),
        "latest_job_status_counts": dict(sorted(latest_status_counts.items())),
        "top_missing_axes": top_missing_axes,
        "recommendations": recommendations,
    }
    return gate


def main() -> int:
    args = parse_args()
    gate = build_supervisor_gate(
        latency_rows=read_many_jsonl(args.latency_rows),
        ap_rows=read_many_jsonl(args.ap_rows),
        energy_rows=read_many_jsonl(args.energy_rows),
        job_plan_rows=read_many_jsonl(args.job_plan),
        job_state_rows=read_many_jsonl(args.job_state),
        axis_gap_rows=read_axis_gap_reports(args.axis_gap_report),
        quarantine_rows=read_many_jsonl(args.quarantine_rows),
        missing_artifact_rows=read_many_jsonl(args.missing_artifact_rows),
        previous_quarantine_rows=(
            read_many_jsonl(args.previous_quarantine_rows)
            if args.previous_quarantine_rows
            else None
        ),
        min_latency_ready_jobs=args.min_latency_ready_jobs,
        axis_lag_ratio=args.axis_lag_ratio,
        repeat_ratio_threshold=args.repeat_ratio_threshold,
    )
    out_dir = Path(args.out_dir)
    write_dashboard_csv(out_dir / "coverage_dashboard_latest.csv", gate)
    write_report(out_dir / "supervisor_report_latest.md", gate)
    write_gate_json(out_dir / "readiness_gate_latest.json", gate)
    print(json.dumps(gate, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
