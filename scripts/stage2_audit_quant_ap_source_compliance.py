#!/usr/bin/env python3
"""Audit AP sources for the Stage2 true-FP16/INT8 quant smoke targets."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ORIGINAL60_ROOT = (
    ROOT
    / "multi_agent/data/stage2_lut_generation_v1/generated/"
    "original60_quant_20260627"
)
DEFAULT_QUANT_ROOT = (
    ROOT / "multi_agent/data/stage2_lut_generation_v1/generated/quant_smoke_20260627"
)
DEFAULT_AP_SOURCE_PATHS = [
    ROOT
    / "multi_agent/data/stage2_lut_generation_v1/generated/ap_stability_20260626/"
    "registry/ap_source_registry_v1.jsonl",
    ROOT
    / "multi_agent/data/stage2_lut_generation_v1/generated/ap_stability_20260626/"
    "rows/ap_anchor_rows_v1.jsonl",
    ROOT
    / "multi_agent/data/stage2_lut_generation_v1/generated/smoke/h800_tvm_v1/"
    "ap/ap_anchor_rows_v1.jsonl",
    ROOT
    / "multi_agent/data/stage2_lut_generation_v1/generated/overnight_6h_20260626/"
    "merged/ap_anchor_rows_merged_v1.jsonl",
]
TARGET_LABELS = ("base", "s0_024", "s1_048")
TARGET_PRECISIONS = ("fp16", "int8")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def jsonish_text(row: dict[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=False, sort_keys=True).lower()


def label_matches(row: dict[str, Any], label: str) -> bool:
    values = {
        str(row.get("label", "")),
        str(row.get("source_label", "")),
        str(row.get("candidate_id", "")),
        str(row.get("config_id", "")),
        str(row.get("software_point_id", "")),
    }
    return any(label in value for value in values)


def precision_matches(row: dict[str, Any], precision: str) -> bool:
    quant_policy = str(row.get("quant_policy", row.get("precision", ""))).lower()
    precision_value = str(row.get("precision", "")).lower()
    text = jsonish_text(row)
    if precision == "fp16":
        return "fp16" in {quant_policy, precision_value} or ":fp16" in text
    if precision == "int8":
        return "int8" in {quant_policy, precision_value} or "int8" in text
    return False


def source_kind(row: dict[str, Any]) -> str:
    return str(row.get("source_kind") or row.get("ap_source_kind") or "unknown")


def reject_reasons(row: dict[str, Any], precision: str) -> list[str]:
    text = jsonish_text(row)
    reasons: list[str] = []
    if str(row.get("measurement_status", "measured")) not in {"measured", ""}:
        reasons.append("source_not_measured")
    if row.get("metric") not in {None, "AP70"}:
        reasons.append("not_ap70")
    if precision == "fp16":
        if "trt" in text or ".engine" in text:
            reasons.append("trt_based_fp16_eval_not_tvm")
        if "tvm" not in text:
            reasons.append("no_tvm_eval_marker")
        if "true_fp16" not in text and "fp16" not in text:
            reasons.append("no_true_fp16_marker")
    elif precision == "int8":
        if "tvm" not in text:
            reasons.append("no_tvm_int8_eval_marker")
        if "int8" not in text:
            reasons.append("no_int8_marker")
        if "trt" in text:
            reasons.append("trt_int8_source_not_allowed")
        if "sim" in text or "predicted" in text or "model-fit" in text:
            reasons.append("simulated_or_predicted_source_not_allowed")
    return sorted(set(reasons))


def classify_target(
    *,
    label: str,
    precision: str,
    source_rows: list[dict[str, Any]],
    ap_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    candidates = [
        row
        for row in source_rows
        if label_matches(row, label) and precision_matches(row, precision)
    ]
    compliant = [
        row
        for row in candidates
        if not reject_reasons(row, precision)
        and row.get("metric_value") is not None
        and source_kind(row) in {"true_eval", "true_import"}
    ]
    existing_ap_rows = [
        row
        for row in ap_rows
        if str(row.get("precision", "")).lower() == precision
        and (row.get("label") == label or row.get("candidate_id") == label)
    ]
    if compliant:
        status = "compliant_source_available"
        blocker = None
    elif precision == "fp16":
        status = "blocked"
        blocker = "no_non_trt_true_fp16_tvm_ap_eval_source"
    else:
        status = "blocked"
        blocker = "no_real_tvm_int8_ap_eval_source"
    return {
        "label": label,
        "precision": precision,
        "status": status,
        "blocker": blocker,
        "candidate_source_count": len(candidates),
        "candidate_source_rejections": [
            {
                "source_kind": source_kind(row),
                "metric_value": row.get("metric_value"),
                "path": row.get("source_path") or row.get("raw_artifact"),
                "reject_reasons": reject_reasons(row, precision),
            }
            for row in candidates[:8]
        ],
        "current_ap_rows": [
            {
                "measurement_status": row.get("measurement_status"),
                "claim_status": row.get("claim_status"),
                "failure_reason": row.get("failure_reason"),
                "metric_value": row.get("metric_value"),
                "raw_artifact": row.get("raw_artifact"),
            }
            for row in existing_ap_rows
        ],
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Stage2 Quant AP Source Compliance Audit",
        "",
        f"Created at: `{report['created_at']}`",
        "",
        "Scope: `base/s0_024/s1_048` x `fp16/int8` AP source compliance.",
        "",
        "Acceptance rule: FP16 AP requires a non-TRT true-FP16/TVM eval source; "
        "INT8 AP requires a real TVM INT8 eval source. TRT, simulated, predicted, "
        "model-fit, and historical ambiguous sources are blocker evidence only.",
        "",
        "## Target Status",
        "",
        "| label | precision | status | blocker | candidate sources | current AP row |",
        "|---|---|---|---|---:|---|",
    ]
    for row in report["targets"]:
        current = row["current_ap_rows"][0] if row["current_ap_rows"] else {}
        current_status = current.get("measurement_status", "missing")
        current_reason = current.get("failure_reason") or ""
        lines.append(
            "| `{label}` | `{precision}` | `{status}` | `{blocker}` | {count} | {current} |".format(
                label=row["label"],
                precision=row["precision"],
                status=row["status"],
                blocker=row["blocker"] or "",
                count=row["candidate_source_count"],
                current=(
                    f"`{current_status}`"
                    + (f" / `{current_reason}`" if current_reason else "")
                ),
            )
        )
    lines.extend(
        [
            "",
            "## Rejection Summary",
            "",
            "| reason | count |",
            "|---|---:|",
        ]
    )
    for reason, count in sorted(report["rejection_reason_counts"].items()):
        lines.append(f"| `{reason}` | {count} |")
    lines.extend(
        [
            "",
            "## Evidence Paths",
            "",
        ]
    )
    for path_item in report["source_paths"]:
        lines.append(f"- `{path_item}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original60-root", type=Path, default=DEFAULT_ORIGINAL60_ROOT)
    parser.add_argument("--quant-root", type=Path, default=DEFAULT_QUANT_ROOT)
    parser.add_argument("--created-at", default="2026-06-27T11:45:00Z")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_rows: list[dict[str, Any]] = []
    for path in DEFAULT_AP_SOURCE_PATHS:
        source_rows.extend(load_jsonl(path))
    ap_rows = load_jsonl(args.original60_root / "rows/ap_original60_quant_rows_v1.jsonl")
    ap_rows.extend(load_jsonl(args.quant_root / "rows/ap_quant_rows_v1.jsonl"))
    targets = [
        classify_target(
            label=label,
            precision=precision,
            source_rows=source_rows,
            ap_rows=ap_rows,
        )
        for label in TARGET_LABELS
        for precision in TARGET_PRECISIONS
    ]
    rejection_counts: Counter[str] = Counter()
    for row in targets:
        for candidate in row["candidate_source_rejections"]:
            rejection_counts.update(candidate["reject_reasons"])
    report = {
        "schema": "stage2_quant_ap_source_compliance_audit_v1",
        "created_at": args.created_at,
        "target_labels": list(TARGET_LABELS),
        "target_precisions": list(TARGET_PRECISIONS),
        "acceptance_rule": {
            "fp16": "non_trt_true_fp16_tvm_eval_source_required",
            "int8": "real_tvm_int8_eval_source_required",
        },
        "source_paths": [
            str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)
            for path in DEFAULT_AP_SOURCE_PATHS
        ],
        "source_row_count": len(source_rows),
        "targets": targets,
        "status_counts": dict(Counter(row["status"] for row in targets)),
        "rejection_reason_counts": dict(rejection_counts),
    }
    exports = args.original60_root / "exports"
    exports.mkdir(parents=True, exist_ok=True)
    json_path = exports / "quant_ap_source_compliance_audit_latest.json"
    md_path = exports / "quant_ap_source_compliance_audit_latest.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown(md_path, report)
    print(json.dumps({"json": str(json_path), "md": str(md_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
