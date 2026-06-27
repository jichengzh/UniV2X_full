#!/usr/bin/env python3
"""Export the latest Original60 AP/latency/energy review table."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "multi_agent/data/stage2_lut_generation_v1/generated"
DEFAULT_COVERAGE = DEFAULT_BASE / "coverage_pipeline_v1"
DEFAULT_AP = DEFAULT_BASE / "ap_stability_20260626"

CSV_FIELDS = [
    "label",
    "candidate_id",
    "width",
    "latency_default_ms",
    "latency_tuned_ms",
    "latency_speedup_default_over_tuned",
    "energy_j_per_inference",
    "ap30",
    "ap50",
    "ap70",
    "latency_status",
    "energy_status",
    "ap_status",
    "ap_source_kind",
    "ap_quality_gate_status",
    "ap_next_action",
    "latency_default_run_id",
    "latency_tuned_run_id",
    "energy_run_id",
    "ap_source_path",
]

MD_FIELDS = [
    "label",
    "width",
    "latency_default_ms",
    "latency_tuned_ms",
    "energy_j_per_inference",
    "ap30",
    "ap50",
    "ap70",
    "latency_status",
    "energy_status",
    "ap_status",
    "ap_quality_gate_status",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-registry",
        default=str(DEFAULT_COVERAGE / "artifacts/artifact_registry_original60_v1.jsonl"),
    )
    parser.add_argument(
        "--latency-rows",
        action="append",
        default=[str(DEFAULT_COVERAGE / "rows/latency_lut_rows_original60_v1.jsonl")],
    )
    parser.add_argument(
        "--energy-rows",
        action="append",
        default=[str(DEFAULT_COVERAGE / "rows/energy_lut_rows_original60_v1.jsonl")],
    )
    parser.add_argument(
        "--ap-source-map",
        default=str(DEFAULT_AP / "exports/original60_ap_source_map_v1.jsonl"),
    )
    parser.add_argument(
        "--ap-summary-csv",
        default=str(DEFAULT_AP / "exports/ap_stability_summary.csv"),
    )
    parser.add_argument(
        "--ap-quality-report",
        default=str(DEFAULT_AP / "exports/ap_quality_gate_report.json"),
    )
    parser.add_argument(
        "--fallback-summary-csv",
        default=str(DEFAULT_COVERAGE / "exports/original60_measured_lut_summary.csv"),
    )
    parser.add_argument(
        "--out-csv",
        default=str(DEFAULT_COVERAGE / "exports/original60_three_metric_summary_latest.csv"),
    )
    parser.add_argument(
        "--out-json",
        default=str(DEFAULT_COVERAGE / "exports/original60_three_metric_summary_latest.json"),
    )
    parser.add_argument(
        "--out-md",
        default=str(DEFAULT_COVERAGE / "exports/original60_three_metric_summary_latest.md"),
    )
    return parser.parse_args()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    jsonl_path = Path(path)
    if not jsonl_path.exists():
        return []
    return [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_csv(path: str | Path) -> list[dict[str, str]]:
    csv_path = Path(path)
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def width_tuple(value: Any) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return ()
        if "x" in stripped and not stripped.startswith("["):
            return tuple(int(item) for item in stripped.split("x") if item)
        value = ast.literal_eval(stripped)
    return tuple(int(item) for item in value)


def width_text(width: Any) -> str:
    return "x".join(str(item) for item in width_tuple(width))


def fmt_float(value: Any, digits: int = 6) -> str:
    if value is None or value == "":
        return ""
    number = float(value)
    text = f"{number:.{digits}f}"
    return text.rstrip("0").rstrip(".")


def rows_from_many(paths: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(read_jsonl(path))
    return rows


def latest_key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("created_at") or ""), str(row.get("run_id") or ""))


def index_latency(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        candidate_id = str(row.get("candidate_id") or "")
        schedule = str(row.get("schedule_policy") or "")
        if not candidate_id or not schedule:
            continue
        key = (candidate_id, schedule)
        if key not in index or latest_key(row) >= latest_key(index[key]):
            index[key] = row
    return index


def index_energy(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for row in rows:
        candidate_id = str(row.get("candidate_id") or "")
        if not candidate_id:
            continue
        if candidate_id not in index or latest_key(row) >= latest_key(index[candidate_id]):
            index[candidate_id] = row
    return index


def index_ap_summary(rows: list[dict[str, str]]) -> tuple[dict[str, dict[str, str]], dict[tuple[int, ...], dict[str, str]]]:
    by_label: dict[str, dict[str, str]] = {}
    by_width: dict[tuple[int, ...], dict[str, str]] = {}
    for row in rows:
        label = row.get("label") or ""
        if label:
            by_label[label] = row
        width = width_tuple(row.get("width"))
        if width:
            by_width[width] = row
    return by_label, by_width


def index_ap_map(rows: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[tuple[int, ...], dict[str, Any]]]:
    by_label: dict[str, dict[str, Any]] = {}
    by_width: dict[tuple[int, ...], dict[str, Any]] = {}
    for row in rows:
        label = str(row.get("label") or "")
        if label:
            by_label[label] = row
        width = width_tuple(row.get("width"))
        if width:
            by_width[width] = row
    return by_label, by_width


def index_ap_quality(path: str | Path) -> dict[str, dict[str, Any]]:
    quality_path = Path(path)
    if not quality_path.exists():
        return {}
    report = json.loads(quality_path.read_text(encoding="utf-8"))
    return {
        str(row.get("label")): row
        for row in report.get("rows", [])
        if row.get("label")
    }


def status_from_latency(default_row: dict[str, Any] | None, tuned_row: dict[str, Any] | None, fallback: dict[str, str] | None) -> str:
    if default_row is not None or tuned_row is not None:
        return "measured"
    if fallback:
        return fallback.get("latency_status") or "missing_latency_rows"
    return "missing_latency_rows"


def status_from_energy(row: dict[str, Any] | None, fallback: dict[str, str] | None) -> str:
    if row is not None and row.get("joule_per_inference") is not None:
        return "measured"
    if fallback:
        return fallback.get("energy_status") or "missing_energy_rows"
    return "missing_energy_rows"


def latency_ms(row: dict[str, Any] | None) -> float | None:
    if row is None or row.get("latency_p50_us") is None:
        return None
    return float(row["latency_p50_us"]) / 1000.0


def speedup(default_ms: float | None, tuned_ms: float | None) -> float | None:
    if default_ms is None or tuned_ms in (None, 0):
        return None
    return default_ms / tuned_ms


def build_rows(
    *,
    artifact_rows: list[dict[str, Any]],
    latency_index: dict[tuple[str, str], dict[str, Any]],
    energy_index: dict[str, dict[str, Any]],
    ap_summary_by_label: dict[str, dict[str, str]],
    ap_summary_by_width: dict[tuple[int, ...], dict[str, str]],
    ap_map_by_label: dict[str, dict[str, Any]],
    ap_map_by_width: dict[tuple[int, ...], dict[str, Any]],
    quality_by_label: dict[str, dict[str, Any]],
    fallback_by_label: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for artifact in artifact_rows:
        label = str(artifact.get("label") or "")
        candidate_id = str(artifact.get("candidate_id") or artifact.get("config_id") or "")
        width = width_tuple(artifact.get("width"))
        fallback = fallback_by_label.get(label)

        default_row = latency_index.get((candidate_id, "default"))
        tuned_row = latency_index.get((candidate_id, "metaschedule_tuned"))
        energy_row = energy_index.get(candidate_id)

        ap_summary = ap_summary_by_label.get(label) or ap_summary_by_width.get(width)
        ap_map = ap_map_by_label.get(label) or ap_map_by_width.get(width)
        quality = quality_by_label.get(label)

        default_ms = latency_ms(default_row)
        tuned_ms = latency_ms(tuned_row)
        ap70 = None
        if ap_summary and ap_summary.get("ap70"):
            ap70 = float(ap_summary["ap70"])
        elif ap_map and ap_map.get("metric_value") is not None:
            ap70 = float(ap_map["metric_value"])

        out.append(
            {
                "label": label,
                "candidate_id": candidate_id,
                "width": "x".join(str(item) for item in width),
                "latency_default_ms": default_ms,
                "latency_tuned_ms": tuned_ms,
                "latency_speedup_default_over_tuned": speedup(default_ms, tuned_ms),
                "energy_j_per_inference": (
                    float(energy_row["joule_per_inference"])
                    if energy_row is not None and energy_row.get("joule_per_inference") is not None
                    else None
                ),
                "ap30": float(ap_summary["ap30"]) if ap_summary and ap_summary.get("ap30") else None,
                "ap50": float(ap_summary["ap50"]) if ap_summary and ap_summary.get("ap50") else None,
                "ap70": ap70,
                "latency_status": status_from_latency(default_row, tuned_row, fallback),
                "energy_status": status_from_energy(energy_row, fallback),
                "ap_status": (
                    str(ap_map.get("claim_status"))
                    if ap_map and ap_map.get("claim_status")
                    else "no_claim_missing_source"
                ),
                "ap_source_kind": (
                    str(ap_map.get("source_kind"))
                    if ap_map and ap_map.get("source_kind")
                    else "no_claim"
                ),
                "ap_quality_gate_status": (
                    str(quality.get("quality_gate_status"))
                    if quality and quality.get("quality_gate_status")
                    else ""
                ),
                "ap_next_action": (
                    str(ap_map.get("next_action"))
                    if ap_map and ap_map.get("next_action")
                    else ""
                ),
                "latency_default_run_id": default_row.get("run_id") if default_row else "",
                "latency_tuned_run_id": tuned_row.get("run_id") if tuned_row else "",
                "energy_run_id": energy_row.get("run_id") if energy_row else "",
                "ap_source_path": (
                    str(ap_map.get("source_path"))
                    if ap_map and ap_map.get("source_path")
                    else ""
                ),
            }
        )
    return out


def csv_value(row: dict[str, Any], field: str) -> str:
    value = row.get(field)
    if field.endswith("_ms") or field in {"latency_speedup_default_over_tuned", "energy_j_per_inference"}:
        return fmt_float(value, 6)
    if field in {"ap30", "ap50", "ap70"}:
        return fmt_float(value, 9)
    return "" if value is None else str(value)


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row, field) for field in CSV_FIELDS})


def summary_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def count_if(predicate: Any) -> int:
        return sum(1 for row in rows if predicate(row))

    return {
        "total_rows": len(rows),
        "latency_measured_rows": count_if(lambda row: row["latency_status"] == "measured"),
        "energy_measured_rows": count_if(lambda row: row["energy_status"] == "measured"),
        "ap_with_ap70_rows": count_if(lambda row: row.get("ap70") is not None),
        "ap_true_eval_rows": count_if(lambda row: row.get("ap_source_kind") == "true_eval"),
        "ap_transfer_rows": count_if(lambda row: row.get("ap_source_kind") == "weight_identity_transfer"),
        "ap_no_claim_rows": count_if(lambda row: row.get("ap_source_kind") == "no_claim"),
        "ap_quality_pending_repeat_rows": count_if(
            lambda row: bool(row.get("ap_quality_gate_status"))
            and "pending_repeat" in str(row.get("ap_quality_gate_status"))
        ),
        "quarantined_rows": count_if(
            lambda row: "quarantine" in str(row.get("latency_status"))
            or "quarantine" in str(row.get("energy_status"))
            or "quarantine" in str(row.get("ap_quality_gate_status"))
            or "quarantine" in str(row.get("ap_next_action"))
        ),
    }


def write_json(path: str | Path, rows: list[dict[str, Any]], sources: dict[str, str]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "stage2_original60_three_metric_summary_v1",
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "scope": "original60_pyramid_lidar_h800_tvm_backbone_only",
        "sources": sources,
        "summary": summary_counts(rows),
        "rows": rows,
    }
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_md(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    counts = summary_counts(rows)
    lines = [
        "# Original60 AP/Latency/Energy 三指标总表",
        "",
        f"生成时间: {datetime.now(timezone.utc).replace(microsecond=0).isoformat()}",
        "",
        "## 汇总",
        "",
        "| item | count |",
        "|---|---:|",
    ]
    for key, value in counts.items():
        lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            "说明: latency 单位为 ms; energy 单位为 joule/inference; AP 为空表示当前没有合法 true source 或 transfer source。",
            "",
            "## 明细",
            "",
            "| " + " | ".join(MD_FIELDS) + " |",
            "| " + " | ".join("---" for _ in MD_FIELDS) + " |",
        ]
    )
    for row in rows:
        lines.append("| " + " | ".join(csv_value(row, field) for field in MD_FIELDS) + " |")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    artifact_rows = read_jsonl(args.artifact_registry)
    if not artifact_rows:
        print(f"No artifact rows found: {args.artifact_registry}", file=sys.stderr)
        return 2

    ap_summary_by_label, ap_summary_by_width = index_ap_summary(read_csv(args.ap_summary_csv))
    ap_map_by_label, ap_map_by_width = index_ap_map(read_jsonl(args.ap_source_map))
    fallback_by_label = {
        row.get("label", ""): row
        for row in read_csv(args.fallback_summary_csv)
        if row.get("label")
    }
    rows = build_rows(
        artifact_rows=artifact_rows,
        latency_index=index_latency(rows_from_many(args.latency_rows)),
        energy_index=index_energy(rows_from_many(args.energy_rows)),
        ap_summary_by_label=ap_summary_by_label,
        ap_summary_by_width=ap_summary_by_width,
        ap_map_by_label=ap_map_by_label,
        ap_map_by_width=ap_map_by_width,
        quality_by_label=index_ap_quality(args.ap_quality_report),
        fallback_by_label=fallback_by_label,
    )

    write_csv(args.out_csv, rows)
    sources = {
        "artifact_registry": args.artifact_registry,
        "latency_rows": ";".join(args.latency_rows),
        "energy_rows": ";".join(args.energy_rows),
        "ap_source_map": args.ap_source_map,
        "ap_summary_csv": args.ap_summary_csv,
        "ap_quality_report": args.ap_quality_report,
        "fallback_summary_csv": args.fallback_summary_csv,
    }
    write_json(args.out_json, rows, sources)
    write_md(args.out_md, rows)
    print(
        json.dumps(
            {
                "schema": "stage2_original60_three_metric_summary_export_v1",
                "out_csv": args.out_csv,
                "out_json": args.out_json,
                "out_md": args.out_md,
                "summary": summary_counts(rows),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
