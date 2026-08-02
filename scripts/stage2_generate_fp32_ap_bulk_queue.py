#!/usr/bin/env python3
"""Generate smoke and full-val queues for Original60 FP32 AP backfill."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = (
    ROOT / "multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627"
)
DEFAULT_CKPT_ROOT = Path("/home/jichengzhi/heal_research/checkpoints/stage1")
DEFAULT_SMOKE_LABELS = ("s0_024", "s1_048", "lhc_07", "frontier_02")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--ckpt-root", default=str(DEFAULT_CKPT_ROOT))
    parser.add_argument("--summary-csv", default=None)
    parser.add_argument("--existing-fp32-ap-rows", default=None)
    parser.add_argument("--smoke-label", action="append", default=[])
    parser.add_argument("--smoke-count", type=int, default=4)
    parser.add_argument("--smoke-samples", type=int, default=32)
    parser.add_argument("--full-samples", type=int, default=1789)
    parser.add_argument("--smoke-out", default=None)
    parser.add_argument("--full-out", default=None)
    parser.add_argument("--review-json", default=None)
    parser.add_argument("--review-md", default=None)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def width_list(value: str) -> list[int]:
    return [int(item) for item in str(value).replace(",", "x").split("x") if item]


def resolve_ckpt_dir(ckpt_root: Path, label: str) -> Path | None:
    candidates = sorted(ckpt_root.glob(f"Pyramid_DAIR_m1_stage2_ap_{label}_2026_06_*"))
    usable = [path for path in candidates if (path / "config.yaml").exists() and list(path.glob("net_epoch*.pth"))]
    return usable[-1] if usable else None


def measured_fp32_labels(rows_path: Path, summary_rows: list[dict[str, str]]) -> set[str]:
    labels = {
        str(row.get("label") or "")
        for row in read_jsonl(rows_path)
        if str(row.get("precision") or row.get("quant_policy") or "") == "fp32"
        and str(row.get("measurement_status") or "") == "measured"
        and row.get("metric_value") is not None
    }
    labels.update(
        str(row.get("label") or "")
        for row in summary_rows
        if str(row.get("precision") or "") == "fp32"
        and str(row.get("ap_status") or "") == "measured"
        and str(row.get("ap_measurement_source") or "") == "true_eval"
        and "true_fp32" in json.dumps(row, ensure_ascii=False).lower()
    )
    return {label for label in labels if label}


def make_job(
    *,
    output_root: Path,
    label: str,
    width: list[int],
    ckpt_dir: Path,
    queue_kind: str,
    num_samples: int,
    created_at: str,
) -> dict[str, Any]:
    raw_dir = output_root / "raw/fp32_ap_backfill_20260701" / queue_kind / label
    return {
        "schema": "stage2_fp32_ap_backfill_queue_row_v1",
        "label": label,
        "width": width,
        "width_csv": ",".join(str(item) for item in width),
        "precision": "fp32",
        "backend": "model_eval",
        "queue_kind": queue_kind,
        "num_samples": num_samples,
        "ckpt_dir": str(ckpt_dir),
        "raw_dir": str(raw_dir),
        "rows_out": str(output_root / "rows/fp32_true_original60_ap_rows_v1.jsonl"),
        "created_at": created_at,
        "full_network_claim": False,
    }


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    ckpt_root = Path(args.ckpt_root)
    summary_csv = Path(args.summary_csv) if args.summary_csv else output_root / "exports/original60_quant_three_metric_summary_latest.csv"
    existing_rows = (
        Path(args.existing_fp32_ap_rows)
        if args.existing_fp32_ap_rows
        else output_root / "rows/fp32_true_original60_ap_rows_v1.jsonl"
    )
    smoke_out = Path(args.smoke_out) if args.smoke_out else output_root / "jobs/fp32_ap_smoke_queue_v1.jsonl"
    full_out = Path(args.full_out) if args.full_out else output_root / "jobs/fp32_ap_full_queue_v1.jsonl"
    review_json = Path(args.review_json) if args.review_json else output_root / "exports/fp32_ap_backfill_queue_review_latest.json"
    review_md = Path(args.review_md) if args.review_md else output_root / "exports/fp32_ap_backfill_queue_review_latest.md"

    summary_rows = read_csv(summary_csv)
    measured = measured_fp32_labels(existing_rows, summary_rows)
    created_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    missing_summary = [
        row
        for row in summary_rows
        if str(row.get("precision") or "") == "fp32"
        and str(row.get("label") or "") not in measured
    ]
    full_jobs: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    for row in missing_summary:
        label = str(row.get("label") or "")
        if not label:
            continue
        ckpt_dir = resolve_ckpt_dir(ckpt_root, label)
        base = {
            "label": label,
            "width": width_list(str(row.get("width") or "")),
            "precision": "fp32",
            "created_at": created_at,
        }
        if ckpt_dir is None:
            blocked.append({**base, "queue_status": "blocked", "reason": "missing_checkpoint"})
            continue
        full_jobs.append(
            make_job(
                output_root=output_root,
                label=label,
                width=base["width"],
                ckpt_dir=ckpt_dir,
                queue_kind="full",
                num_samples=args.full_samples,
                created_at=created_at,
            )
        )

    preferred = list(args.smoke_label or DEFAULT_SMOKE_LABELS)
    full_by_label = {str(job["label"]): job for job in full_jobs}
    smoke_labels = [label for label in preferred if label in full_by_label]
    if len(smoke_labels) < args.smoke_count:
        for job in full_jobs:
            label = str(job["label"])
            if label not in smoke_labels:
                smoke_labels.append(label)
            if len(smoke_labels) >= args.smoke_count:
                break
    smoke_jobs = [
        {
            **full_by_label[label],
            "queue_kind": "smoke",
            "num_samples": args.smoke_samples,
            "raw_dir": str(output_root / "raw/fp32_ap_backfill_20260701/smoke" / label),
        }
        for label in smoke_labels
    ]

    write_jsonl(smoke_out, smoke_jobs)
    write_jsonl(full_out, full_jobs)
    review = {
        "schema": "stage2_fp32_ap_backfill_queue_review_v1",
        "created_at": created_at,
        "summary_csv": str(summary_csv),
        "existing_fp32_ap_rows": str(existing_rows),
        "smoke_queue": str(smoke_out),
        "full_queue": str(full_out),
        "measured_count": len(measured),
        "full_queue_count": len(full_jobs),
        "smoke_queue_count": len(smoke_jobs),
        "blocked_count": len(blocked),
        "smoke_labels": [row["label"] for row in smoke_jobs],
        "full_labels": [row["label"] for row in full_jobs],
        "blocked": blocked,
        "note": "Smoke queue is report-only and must not append partial AP rows.",
    }
    write_json(review_json, review)
    review_md.write_text(
        "\n".join(
            [
                "# FP32 AP backfill queue review",
                "",
                f"- created_at: `{created_at}`",
                f"- measured_count: {len(measured)}",
                f"- full_queue_count: {len(full_jobs)}",
                f"- smoke_queue_count: {len(smoke_jobs)}",
                f"- blocked_count: {len(blocked)}",
                "",
                "## Smoke labels",
                "",
                ", ".join(row["label"] for row in smoke_jobs) or "(empty)",
                "",
                "## Blocked",
                "",
                ", ".join(f"{row['label']}({row['reason']})" for row in blocked) or "(empty)",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(review, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
