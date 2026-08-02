#!/usr/bin/env python3
"""Incrementally aggregate the Pyramid Stage6 six-arm experiment.

The aggregator is intentionally evidence-conservative: it creates every slot
defined by the manifest, but reads metric values only from the native baseline
or an explicitly supplied terminal summary. Historical evidence and predicted,
surrogate, or proxy values are never used to fill an unmeasured cell.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


FIELDS = (
    "arm",
    "backend",
    "status",
    "config",
    "AP70",
    "latency",
    "energy",
    "HV",
    "frontier_count",
    "outer_genomes",
    "tuning_trials",
    "gpu_hours",
    "wallclock",
    "failures",
    "evidence",
)

FINAL_SUCCESS_STATUSES = {
    "complete",
    "completed",
    "closed",
    "measured_success_gold",
    "paper_ready",
    "ready",
    "success",
}
FINAL_FAILURE_STATUSES = {
    "failed",
    "feasibility_failure",
    "numerical_feasibility_failure",
    "terminal_failure",
}
SEARCH_ARMS = {
    "compression_only",
    "compress_then_tune",
    "tune_then_compress",
    "joint_shcosearch",
}


def _first(mapping: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return None


def _measured_objectives(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("objectives", "measured_objectives", "measured_metrics"):
        value = summary.get(key)
        if isinstance(value, Mapping):
            return value
    return {}


def _metric(summary: Mapping[str, Any], names: Sequence[str]) -> Any:
    value = _first(summary, names)
    if value is not None:
        return value
    return _first(_measured_objectives(summary), names)


def _failure_count(value: Any) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        return len(value)
    if isinstance(value, Mapping):
        count = _first(value, ("count", "total", "failure_count"))
        return count if isinstance(count, (int, float)) else None
    return value if isinstance(value, (int, float)) else None


def _evidence_list(value: Any, source: str | None) -> list[str]:
    evidence: list[str] = []
    if isinstance(value, str) and value:
        evidence.append(value)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        evidence.extend(str(item) for item in value if item)
    elif isinstance(value, Mapping):
        evidence.extend(str(item) for item in value.values() if isinstance(item, str) and item)
    if source:
        evidence.append(source)
    return list(dict.fromkeys(evidence))


def _empty_row(arm: str, backend: str) -> dict[str, Any]:
    return {
        "arm": arm,
        "backend": backend,
        "status": "provisional",
        "config": None,
        "AP70": None,
        "latency": None,
        "energy": None,
        "HV": None,
        "frontier_count": None,
        "outer_genomes": None,
        "tuning_trials": None,
        "gpu_hours": None,
        "wallclock": None,
        "failures": None,
        "evidence": [],
    }


def _expected_slots(manifest: Mapping[str, Any]) -> list[tuple[str, str]]:
    backends = [str(value) for value in manifest.get("independent_backends") or []]
    slots: list[tuple[str, str]] = []
    for arm in manifest.get("arms") or []:
        arm_id = str(arm.get("arm_id") or "")
        if not arm_id:
            raise ValueError("manifest arm is missing arm_id")
        if arm_id == "original_default":
            backend = str(arm.get("backend") or "pytorch_eager")
            slots.append((arm_id, backend))
        else:
            slots.extend((arm_id, backend) for backend in backends)
    if not slots:
        raise ValueError("manifest defines no Stage6 result slots")
    if len(set(slots)) != len(slots):
        raise ValueError("manifest defines duplicate Stage6 result slots")
    return slots


def _baseline_row(
    baseline: Mapping[str, Any],
    arm: Mapping[str, Any],
    evidence: str | None,
) -> dict[str, Any]:
    backend = str(baseline.get("backend") or arm.get("backend") or "pytorch_eager")
    row = _empty_row("original_default", backend)
    width = baseline.get("width") or arm.get("width")
    precision = baseline.get("precision") or "fp32"
    row.update(
        {
            "config": [*width, precision] if isinstance(width, Sequence) else None,
            "AP70": _metric(baseline, ("AP70", "ap70")),
            "latency": _metric(baseline, ("latency", "latency_ms", "latency_p50_ms")),
            "energy": _metric(baseline, ("energy", "energy_j")),
            "outer_genomes": 1,
            "tuning_trials": 0 if baseline.get("backend_tuning") is False else None,
            "failures": 0,
            "evidence": _evidence_list(baseline.get("evidence"), evidence),
        }
    )
    required = ("config", "AP70", "latency", "energy")
    if all(row[field] is not None for field in required) and row["evidence"]:
        row["status"] = "complete"
    return row


def _summary_row(summary: Mapping[str, Any], source: str | None) -> dict[str, Any]:
    arm = str(summary.get("arm_id") or summary.get("arm") or "")
    backend = str(summary.get("backend") or "")
    if not arm or not backend:
        raise ValueError(f"terminal summary lacks arm/backend identity: {source or '<memory>'}")

    row = _empty_row(arm, backend)
    frontier_count = _first(summary, ("frontier_count", "pareto_count"))
    frontier_ids = summary.get("frontier_ids")
    if (
        frontier_count is None
        and isinstance(frontier_ids, Sequence)
        and not isinstance(frontier_ids, (str, bytes))
    ):
        frontier_count = len(frontier_ids)
    row.update(
        {
            "config": _first(
                summary,
                ("config", "representative_config", "best_config", "genome"),
            ),
            "AP70": _metric(summary, ("AP70", "ap70")),
            "latency": _metric(summary, ("latency", "latency_ms", "latency_p50_ms")),
            "energy": _metric(summary, ("energy", "energy_j")),
            "HV": _first(summary, ("HV", "hv", "hypervolume")),
            "frontier_count": frontier_count,
            "outer_genomes": _first(
                summary,
                ("outer_genomes", "outer_measured_genomes", "online_count"),
            ),
            "tuning_trials": _first(
                summary,
                ("tuning_trials", "backend_tuning_trials", "trial_count"),
            ),
            "gpu_hours": _first(summary, ("gpu_hours", "gpu_hour")),
            "wallclock": _first(
                summary,
                ("wallclock", "wallclock_s", "wallclock_seconds"),
            ),
            "failures": _failure_count(
                _first(summary, ("failures", "failure_count", "terminal_failures"))
            ),
            "evidence": _evidence_list(summary.get("evidence"), source),
        }
    )

    raw_status = str(summary.get("status") or summary.get("terminal_status") or "").lower()
    measurement_required = ["config", "AP70", "latency", "energy"]
    accounting_required = [
        "outer_genomes",
        "tuning_trials",
        "gpu_hours",
        "wallclock",
        "failures",
    ]
    if arm in SEARCH_ARMS:
        measurement_required.extend(("HV", "frontier_count"))
    if raw_status in FINAL_SUCCESS_STATUSES and all(
        row[field] is not None for field in measurement_required + accounting_required
    ) and row["evidence"]:
        row["status"] = "complete"
    elif raw_status in FINAL_FAILURE_STATUSES and all(
        row[field] is not None for field in accounting_required
    ) and row["failures"] and row["evidence"]:
        row["status"] = "complete_failure"
    return row


def aggregate_six_arm_results(
    manifest: Mapping[str, Any],
    baseline: Mapping[str, Any],
    terminal_summaries: Iterable[tuple[Mapping[str, Any], str | None]],
    *,
    manifest_evidence: str | None = None,
    baseline_evidence: str | None = None,
) -> dict[str, Any]:
    """Return a fixed-slot incremental table and its readiness audit."""
    slots = _expected_slots(manifest)
    rows = {slot: _empty_row(*slot) for slot in slots}
    arms = {str(arm.get("arm_id")): arm for arm in manifest.get("arms") or []}

    original_arm = arms.get("original_default") or {}
    baseline_row = _baseline_row(baseline, original_arm, baseline_evidence)
    baseline_key = (baseline_row["arm"], baseline_row["backend"])
    if baseline_key not in rows:
        raise ValueError(f"baseline does not match manifest slot: {baseline_key}")
    rows[baseline_key] = baseline_row

    seen: set[tuple[str, str]] = set()
    for summary, source in terminal_summaries:
        row = _summary_row(summary, source)
        key = (row["arm"], row["backend"])
        if key not in rows:
            raise ValueError(f"terminal summary does not match manifest slot: {key}")
        if key in seen:
            raise ValueError(f"duplicate terminal summary for slot: {key}")
        seen.add(key)
        rows[key] = row

    ordered_rows = [rows[slot] for slot in slots]
    complete_rows = sum(row["status"] != "provisional" for row in ordered_rows)
    provisional = [
        {"arm": row["arm"], "backend": row["backend"]}
        for row in ordered_rows
        if row["status"] == "provisional"
    ]
    readiness = {
        "ready": not provisional,
        "expected_rows": len(ordered_rows),
        "complete_rows": complete_rows,
        "provisional_rows": len(provisional),
        "provisional_slots": provisional,
        "policy": "measured terminal summaries only; no surrogate or historical proxy backfill",
    }
    return {
        "schema_version": "stage6_pyramid_six_arm_aggregate_v1",
        "experiment_id": manifest.get("experiment_id"),
        "manifest_evidence": manifest_evidence,
        "rows": ordered_rows,
        "readiness": readiness,
    }


def _tabular_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=True, separators=(",", ":"))
    return str(value)


def _markdown_table(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# Stage6 Pyramid Six-Arm Incremental Results",
        "",
        "Unmeasured or incomplete cells are blank and the row is marked `provisional`.",
        "No surrogate or historical proxy values are used.",
        "",
        "| " + " | ".join(FIELDS) + " |",
        "| " + " | ".join("---" for _ in FIELDS) + " |",
    ]
    for row in rows:
        values = [_tabular_value(row.get(field)).replace("|", "\\|") for field in FIELDS]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def write_aggregate_outputs(
    aggregate: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "stage6_pyramid_six_arm_incremental_v1"
    paths = {
        "csv": output_dir / f"{stem}.csv",
        "markdown": output_dir / f"{stem}.md",
        "json": output_dir / f"{stem}.json",
    }
    with paths["csv"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(
            {field: _tabular_value(row.get(field)) for field in FIELDS}
            for row in aggregate.get("rows") or []
        )
    paths["markdown"].write_text(
        _markdown_table(aggregate.get("rows") or []),
        encoding="utf-8",
    )
    paths["json"].write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return paths


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _load_terminal_summaries(paths: Sequence[Path]) -> list[tuple[Mapping[str, Any], str]]:
    summaries: list[tuple[Mapping[str, Any], str]] = []
    for path in paths:
        payload = _load_json(path)
        if isinstance(payload, Mapping):
            if "rows" in payload:
                items = payload["rows"]
            elif "summaries" in payload:
                items = payload["summaries"]
            else:
                items = [payload]
            if not isinstance(items, list):
                raise ValueError(f"terminal summary container must hold a list: {path}")
        elif isinstance(payload, list):
            items = payload
        else:
            raise ValueError(f"terminal summary must be an object or list: {path}")
        for item in items:
            if not isinstance(item, Mapping):
                raise ValueError(f"terminal summary row must be an object: {path}")
            summaries.append((item, str(path)))
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument(
        "--terminal-summary",
        type=Path,
        action="append",
        default=[],
        help="Repeat for each available backend/arm terminal summary JSON.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    aggregate = aggregate_six_arm_results(
        _load_json(args.manifest),
        _load_json(args.baseline),
        _load_terminal_summaries(args.terminal_summary),
        manifest_evidence=str(args.manifest),
        baseline_evidence=str(args.baseline),
    )
    paths = write_aggregate_outputs(aggregate, args.output_dir)
    print(json.dumps({key: str(path) for key, path in paths.items()}, indent=2))


if __name__ == "__main__":
    main()
