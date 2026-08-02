#!/usr/bin/env python3
"""Generate original60 FP16/INT8 AP true-eval queue and source audit artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.stage2.lut_productization import utc_timestamp  # noqa: E402
from scripts.stage2_generate_original60_quant_state_coverage import (  # noqa: E402
    is_compliant_original60_ap_row,
    smoke_label,
)


DEFAULT_OUTPUT_ROOT = (
    ROOT / "multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627"
)
DEFAULT_SOURCE_ROWS = (
    ROOT
    / "multi_agent/data/stage2_lut_generation_v1/generated/ap_stability_20260626/"
    "rows/ap_anchor_rows_v1.jsonl",
    ROOT
    / "multi_agent/data/stage2_lut_generation_v1/generated/ap_stability_20260626/"
    "registry/ap_source_registry_v1.jsonl",
    ROOT
    / "multi_agent/data/stage2_lut_generation_v1/generated/smoke/h800_tvm_v1/"
    "ap/ap_anchor_rows_v1.jsonl",
    ROOT
    / "multi_agent/data/stage2_lut_generation_v1/generated/overnight_6h_20260626/"
    "merged/ap_anchor_rows_merged_v1.jsonl",
)
TARGET_PRECISIONS = ("fp16", "int8")
SCHEMA = "original60_fp16_int8_ap_true_eval_queue_v1"
JOB_SCHEMA = "original60_fp16_int8_ap_true_eval_job_v1"
BLOCKER_SCHEMA = "original60_fp16_int8_ap_true_eval_blocker_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--completion-queue",
        default=None,
        help=(
            "Completion queue JSONL. Defaults to "
            "<output-root>/jobs/fp16_int8_original60_completion_queue_v1.jsonl."
        ),
    )
    parser.add_argument("--fp16-ap-source-rows", action="append", default=[])
    parser.add_argument("--int8-ap-source-rows", action="append", default=[])
    parser.add_argument(
        "--no-default-source-rows",
        action="store_true",
        help="Only inspect source rows explicitly passed on the CLI.",
    )
    parser.add_argument("--created-at", default=None)
    return parser.parse_args()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    item = Path(path)
    if not item.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in item.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def row_text(row: dict[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=False, sort_keys=True).lower()


def source_label(row: dict[str, Any]) -> str:
    label = smoke_label(row)
    if label:
        return label
    for key in ("source_label", "original60_label"):
        value = str(row.get(key) or "")
        if value:
            return value
    return ""


def source_precision(row: dict[str, Any]) -> str:
    precision = str(row.get("precision") or row.get("quant_policy") or "").lower()
    if precision in TARGET_PRECISIONS:
        return precision
    text = row_text(row)
    if "int8" in text:
        return "int8"
    if "true_fp16" in text or "float16" in text or "fp16" in text:
        return "fp16"
    return ""


def normalize_width(value: Any) -> list[int]:
    if isinstance(value, list):
        return [int(item) for item in value]
    if isinstance(value, tuple):
        return [int(item) for item in value]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        sep = "x" if "x" in text else ","
        return [int(item) for item in text.split(sep) if item.strip()]
    return []


def source_matches(row: dict[str, Any], label: str, precision: str) -> bool:
    row_label = source_label(row)
    if row_label != label and label not in row_text(row):
        return False
    return source_precision(row) == precision


def _has_unknown_required_text(row: dict[str, Any], fields: tuple[str, ...]) -> list[str]:
    return [
        field
        for field in fields
        if str(row.get(field) or "").strip() in ("", "unknown")
    ]


def ap_reject_reasons(
    row: dict[str, Any],
    precision: str,
    expected_width: list[int] | None = None,
) -> list[str]:
    reasons: list[str] = []
    text = row_text(row)
    if expected_width is not None and normalize_width(row.get("width")) != expected_width:
        reasons.append("width_mismatch_against_completion_queue")
    if str(row.get("measurement_status") or "") != "measured":
        reasons.append("source_not_measured")
    if str(row.get("backend") or "") != "model_eval":
        reasons.append("backend_not_model_eval")
    if bool(row.get("full_network_claim")):
        reasons.append("full_network_claim_not_allowed")
    if str(row.get("metric") or "") != "AP70":
        reasons.append("metric_not_ap70")
    if row.get("metric_value") is None:
        reasons.append("metric_value_missing")
    if str(row.get("measurement_source") or "") != "true_eval":
        reasons.append("measurement_source_not_true_eval")
    banned = ("predicted", "model_fit", "model-fit", "interpolated", "trt", "simulated")
    for token in banned:
        if token in text:
            reasons.append(f"banned_source_token:{token}")
    required_missing = _has_unknown_required_text(
        row,
        ("dataset", "eval_split", "ckpt_path", "ckpt_digest", "eval_command", "raw_artifact"),
    )
    for field in required_missing:
        reasons.append(f"missing_{field}")
    if not row.get("source_files"):
        reasons.append("source_files_missing")
    secondary = row.get("secondary_metrics")
    if not isinstance(secondary, dict) or "AP30" not in secondary or "AP50" not in secondary:
        reasons.append("secondary_metrics_missing_ap30_ap50")
    if precision == "fp16":
        if str(row.get("quant_method") or "") != "h800_tvm_true_fp16_onnx_relax":
            reasons.append("fp16_quant_method_not_true_fp16_onnx_relax")
        marker_text = " ".join(
            str(row.get(key) or "")
            for key in (
                "quality_gate_status",
                "layer_precision_summary",
                "source_files",
                "notes",
                "eval_command",
            )
        ).lower()
        if "true_fp16" not in marker_text and "float16" not in marker_text:
            reasons.append("true_fp16_marker_missing")
    if precision == "int8":
        if str(row.get("quant_method") or "") != "h800_tvm_native_int8_backbone_subnet":
            reasons.append("int8_quant_method_not_native_int8")
        if str(row.get("quant_scope") or "") != "backbone_subnet_native_int8":
            reasons.append("int8_quant_scope_not_native_backbone_subnet")
    return sorted(set(reasons))


def load_sources(paths: list[str], precision: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for index, row in enumerate(read_jsonl(path)):
            row = dict(row)
            row["_source_row_path"] = str(path)
            row["_source_row_index"] = index
            if source_precision(row) == precision:
                rows.append(row)
    return rows


def source_paths(args: argparse.Namespace) -> dict[str, list[str]]:
    defaults = [] if args.no_default_source_rows else [str(path) for path in DEFAULT_SOURCE_ROWS]
    return {
        "fp16": [*defaults, *args.fp16_ap_source_rows],
        "int8": [*defaults, *args.int8_ap_source_rows],
    }


def best_compliant_source(
    *,
    label: str,
    precision: str,
    expected_width: list[int],
    rows: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    candidates = [row for row in rows if source_matches(row, label, precision)]
    compliant = [
        row
        for row in candidates
        if normalize_width(row.get("width")) == expected_width
        and is_compliant_original60_ap_row(row, precision)
    ]

    def score(row: dict[str, Any]) -> tuple[str, str]:
        return str(row.get("created_at") or ""), str(row.get("run_id") or "")

    compliant_row = max(compliant, key=score) if compliant else None
    rejections = [
        {
            "source_row_path": row.get("_source_row_path"),
            "source_row_index": row.get("_source_row_index"),
            "metric_value": row.get("metric_value"),
            "raw_artifact": row.get("raw_artifact"),
            "reject_reasons": ap_reject_reasons(row, precision, expected_width),
        }
        for row in candidates
        if normalize_width(row.get("width")) != expected_width
        or not is_compliant_original60_ap_row(row, precision)
    ]
    return compliant_row, rejections


def blocker_for_precision(precision: str) -> str:
    if precision == "fp16":
        return "no_compliant_true_fp16_model_eval_source"
    return "no_compliant_native_int8_model_eval_source"


def next_action_for_precision(precision: str) -> str:
    if precision == "fp16":
        return "build_true_fp16_model_eval_backend_then_run_eval"
    return "build_or_connect_native_int8_eval_backend_then_run_eval"


def build_queue_row(
    *,
    completion_job: dict[str, Any],
    source_row: dict[str, Any] | None,
    source_rejections: list[dict[str, Any]],
    output_root: Path,
    created_at: str,
) -> dict[str, Any]:
    label = str(completion_job.get("label") or "")
    precision = str(completion_job.get("precision") or "")
    raw_artifact = f"raw/ap_eval_original60/{precision}_{label}"
    row = {
        "schema": JOB_SCHEMA,
        "job_id": f"original60_ap_true_eval:{label}:{precision}",
        "completion_job_id": completion_job.get("job_id"),
        "label": label,
        "precision": precision,
        "width": completion_job.get("width") or [],
        "onnx_backbone_path": completion_job.get("onnx_backbone_path"),
        "workdir": completion_job.get("workdir"),
        "latency_status": completion_job.get("latency_status"),
        "energy_status": completion_job.get("energy_status"),
        "ap_status": completion_job.get("ap_status"),
        "full_network_claim": False,
        "expected_raw_artifact": raw_artifact,
        "blocker_artifact": str(
            output_root / "quarantine" / f"ap_true_eval_{precision}_{label}_blocker.json"
        ),
        "created_at": created_at,
        "source_rejections": source_rejections[:8],
    }
    if source_row is not None:
        row.update(
            {
                "ap_eval_status": "ready_for_import",
                "next_action": "import_compliant_ap_row",
                "blocker": None,
                "metric": source_row.get("metric") or "AP70",
                "metric_value": source_row.get("metric_value"),
                "secondary_metrics": source_row.get("secondary_metrics") or {},
                "source_row_path": source_row.get("_source_row_path"),
                "source_row_index": source_row.get("_source_row_index"),
                "source_raw_artifact": source_row.get("raw_artifact"),
                "source_files": source_row.get("source_files") or [],
                "eval_command": source_row.get("eval_command"),
                "ckpt_path": source_row.get("ckpt_path"),
                "ckpt_digest": source_row.get("ckpt_digest"),
                "dataset": source_row.get("dataset"),
                "eval_split": source_row.get("eval_split"),
            }
        )
    else:
        row.update(
            {
                "ap_eval_status": "blocked",
                "next_action": next_action_for_precision(precision),
                "blocker": blocker_for_precision(precision),
                "metric": "AP70",
                "metric_value": None,
            }
        )
    return row


def build_blocker_rows(queue_rows: list[dict[str, Any]], created_at: str) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    for row in queue_rows:
        if row.get("ap_eval_status") != "blocked":
            continue
        blockers.append(
            {
                "schema": BLOCKER_SCHEMA,
                "job_id": row["job_id"],
                "completion_job_id": row.get("completion_job_id"),
                "label": row["label"],
                "precision": row["precision"],
                "failure_stage": "ap_source_compliance_audit",
                "failure_reason": row.get("blocker"),
                "next_action": row.get("next_action"),
                "onnx_backbone_path": row.get("onnx_backbone_path"),
                "workdir": row.get("workdir"),
                "expected_raw_artifact": row.get("expected_raw_artifact"),
                "blocker_artifact": row.get("blocker_artifact"),
                "source_rejections": row.get("source_rejections") or [],
                "created_at": created_at,
            }
        )
    return blockers


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# FP16/INT8 Original60 AP True-Eval Source Audit",
        "",
        f"- created_at: `{payload['created_at']}`",
        f"- total_jobs: `{payload['summary']['total_jobs']}`",
        f"- ready_for_import: `{payload['summary'].get('ready_for_import', 0)}`",
        f"- blocked: `{payload['summary'].get('blocked', 0)}`",
        "",
        "| label | precision | AP eval status | next action | blocker |",
        "|---|---|---|---|---|",
    ]
    for row in payload["jobs"]:
        lines.append(
            "| {label} | {precision} | {status} | {action} | {blocker} |".format(
                label=row["label"],
                precision=row["precision"],
                status=row["ap_eval_status"],
                action=row["next_action"],
                blocker=row.get("blocker") or "",
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    completion_queue = (
        Path(args.completion_queue)
        if args.completion_queue
        else output_root / "jobs/fp16_int8_original60_completion_queue_v1.jsonl"
    )
    created_at = args.created_at or utc_timestamp()
    source_path_map = source_paths(args)
    sources = {
        precision: load_sources(paths, precision)
        for precision, paths in source_path_map.items()
    }
    completion_jobs = [
        row
        for row in read_jsonl(completion_queue)
        if str(row.get("precision") or "") in TARGET_PRECISIONS
        and str(row.get("ap_status") or "") != "measured"
    ]
    queue_rows: list[dict[str, Any]] = []
    for job in completion_jobs:
        precision = str(job["precision"])
        label = str(job["label"])
        source_row, source_rejections = best_compliant_source(
            label=label,
            precision=precision,
            expected_width=normalize_width(job.get("width")),
            rows=sources[precision],
        )
        queue_rows.append(
            build_queue_row(
                completion_job=job,
                source_row=source_row,
                source_rejections=source_rejections,
                output_root=output_root,
                created_at=created_at,
            )
        )

    blockers = build_blocker_rows(queue_rows, created_at)
    status_counts = Counter(str(row.get("ap_eval_status")) for row in queue_rows)
    summary = {
        "schema": SCHEMA,
        "total_jobs": len(queue_rows),
        "source_rows": {precision: len(rows) for precision, rows in sources.items()},
        **dict(sorted(status_counts.items())),
    }
    audit = {
        "schema": "original60_fp16_int8_ap_true_eval_source_audit_v1",
        "created_at": created_at,
        "completion_queue": str(completion_queue),
        "source_paths": source_path_map,
        "summary": summary,
        "jobs": queue_rows,
        "blockers": blockers,
    }

    queue_path = output_root / "jobs/fp16_int8_original60_ap_true_eval_queue_v1.jsonl"
    blocker_path = output_root / "quarantine/fp16_int8_original60_ap_true_eval_blockers_v1.jsonl"
    audit_json = output_root / "exports/fp16_int8_original60_ap_true_eval_source_audit_latest.json"
    audit_md = output_root / "exports/fp16_int8_original60_ap_true_eval_source_audit_latest.md"
    write_jsonl(queue_path, queue_rows)
    write_jsonl(blocker_path, blockers)
    write_json(audit_json, audit)
    write_markdown(audit_md, audit)
    print(
        json.dumps(
            {
                "status": "ok",
                "summary": summary,
                "outputs": {
                    "queue": str(queue_path),
                    "blockers": str(blocker_path),
                    "audit_json": str(audit_json),
                    "audit_md": str(audit_md),
                },
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
