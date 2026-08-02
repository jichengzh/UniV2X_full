#!/usr/bin/env python3
"""Generate production/quarantine queues for native INT8 original60 AP bulk runs."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = (
    ROOT / "multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627"
)
DEFAULT_CKPT_ROOT = Path("/home/jichengzhi/heal_research/checkpoints/stage1")
DEFAULT_QUARANTINE_LABELS = {
    "frontier_16": "export_failure_or_hang",
    "frontier_18": "export_failure_or_hang",
    "frontier_25": "numeric_sanity_failure_and_range_capture_hang",
    "frontier_26": "export_hang_observed_in_recovery",
    "frontier_27": "export_failure",
    "frontier_01": "ap70_zero_suspicious",
    "s1_112": "export_hang_observed_in_recovery",
    "s2_096": "export_hang_observed_in_recovery",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--ckpt-root", default=str(DEFAULT_CKPT_ROOT))
    parser.add_argument("--quarantine-label", action="append", default=[])
    parser.add_argument("--include-default-quarantine", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--production-out", default=None)
    parser.add_argument("--quarantine-out", default=None)
    parser.add_argument("--review-json", default=None)
    parser.add_argument("--review-md", default=None)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def measured_labels(output_root: Path) -> set[str]:
    rows = read_jsonl(output_root / "rows/native_int8_original60_ap_rows_v1.jsonl")
    labels: set[str] = set()
    for row in rows:
        if row.get("metric_value") is not None or row.get("measurement_status") == "measured":
            label = str(row.get("label") or "")
            if label:
                labels.add(label)
    return labels


def completion_widths(output_root: Path) -> dict[str, list[int]]:
    widths: dict[str, list[int]] = {}
    for row in read_jsonl(output_root / "jobs/fp16_int8_original60_completion_queue_v1.jsonl"):
        if row.get("precision") != "int8":
            continue
        label = str(row.get("label") or "")
        width = row.get("width")
        if label and isinstance(width, list):
            widths[label] = [int(item) for item in width]
    return widths


def resolve_ckpt_dir(ckpt_root: Path, label: str) -> Path | None:
    candidates = sorted(ckpt_root.glob(f"Pyramid_DAIR_m1_stage2_ap_{label}_2026_06_*"))
    usable = [p for p in candidates if (p / "config.yaml").exists() and list(p.glob("net_epoch*.pth"))]
    return usable[-1] if usable else None


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    ckpt_root = Path(args.ckpt_root)
    queue_dir = output_root / "jobs"
    exports_dir = output_root / "exports"
    production_out = Path(args.production_out) if args.production_out else queue_dir / "native_int8_ap_production_queue_v1.jsonl"
    quarantine_out = Path(args.quarantine_out) if args.quarantine_out else queue_dir / "native_int8_ap_quarantine_queue_v1.jsonl"
    review_json = Path(args.review_json) if args.review_json else exports_dir / "native_int8_ap_bulk_queue_review_latest.json"
    review_md = Path(args.review_md) if args.review_md else exports_dir / "native_int8_ap_bulk_queue_review_latest.md"

    quarantine_reasons: dict[str, str] = {}
    if args.include_default_quarantine:
        quarantine_reasons.update(DEFAULT_QUARANTINE_LABELS)
    for item in args.quarantine_label:
        label, _, reason = item.partition(":")
        if label.strip():
            quarantine_reasons[label.strip()] = reason.strip() or "manual_quarantine"

    widths = completion_widths(output_root)
    measured = measured_labels(output_root)
    production_rows: list[dict[str, Any]] = []
    quarantine_rows: list[dict[str, Any]] = []
    missing_checkpoint_rows: list[dict[str, Any]] = []
    created_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    for label in sorted(widths):
        width = widths[label]
        ckpt_dir = resolve_ckpt_dir(ckpt_root, label)
        base = {
            "schema": "stage2_native_int8_ap_bulk_queue_row_v1",
            "label": label,
            "width": width,
            "precision": "int8",
            "backend": "h800_tvm",
            "quant_scope": "backbone_subnet_native_int8",
            "full_network_claim": False,
            "created_at": created_at,
            "ckpt_dir": str(ckpt_dir) if ckpt_dir else None,
        }
        if label in measured:
            continue
        if ckpt_dir is None:
            missing_checkpoint_rows.append({**base, "queue_status": "blocked", "reason": "missing_checkpoint"})
            continue
        if label in quarantine_reasons:
            quarantine_rows.append(
                {
                    **base,
                    "queue_status": "quarantined",
                    "quarantine_status": "active",
                    "reason": quarantine_reasons[label],
                }
            )
            continue
        production_rows.append({**base, "queue_status": "ready"})

    write_jsonl(production_out, production_rows)
    write_jsonl(quarantine_out, quarantine_rows)
    summary = {
        "schema": "stage2_native_int8_ap_bulk_queue_review_v1",
        "created_at": created_at,
        "output_root": str(output_root),
        "production_queue": str(production_out),
        "quarantine_queue": str(quarantine_out),
        "production_count": len(production_rows),
        "quarantine_count": len(quarantine_rows),
        "missing_checkpoint_count": len(missing_checkpoint_rows),
        "measured_count": len(measured),
        "measured_labels": sorted(measured),
        "production_labels": [row["label"] for row in production_rows],
        "quarantine_labels": [row["label"] for row in quarantine_rows],
        "missing_checkpoint_labels": [row["label"] for row in missing_checkpoint_rows],
    }
    write_json(review_json, summary)
    review_lines = [
        "# Native INT8 AP bulk queue review",
        "",
        f"- created_at: `{created_at}`",
        f"- measured_count: {len(measured)}",
        f"- production_count: {len(production_rows)}",
        f"- quarantine_count: {len(quarantine_rows)}",
        f"- missing_checkpoint_count: {len(missing_checkpoint_rows)}",
        "",
        "## Production labels",
        "",
        ", ".join(row["label"] for row in production_rows) or "(empty)",
        "",
        "## Quarantine labels",
        "",
        ", ".join(f"{row['label']}({row['reason']})" for row in quarantine_rows) or "(empty)",
        "",
        "## Missing checkpoint labels",
        "",
        ", ".join(row["label"] for row in missing_checkpoint_rows) or "(empty)",
        "",
    ]
    review_md.write_text("\n".join(review_lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
