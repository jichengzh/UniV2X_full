#!/usr/bin/env python3
"""Audit suspect original60 FP16-tagged latency rows for FP32 remap."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.stage2.lut_productization import (  # noqa: E402
    latency_lut_row,
    stable_config_id,
    utc_timestamp,
    write_jsonl,
)


DEFAULT_INPUT_ROWS = (
    ROOT
    / "multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1/rows/"
    "latency_lut_rows_original60_v1.jsonl"
)
DEFAULT_OUTPUT_ROOT = (
    ROOT / "multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627"
)
MODEL = "pyramid_lidar"
MANIFEST_DIGEST = "original60_fp32_remap_audit_20260627"
OPTIMIZED_SCOPE = "backbone_only"
CLASSIFICATIONS = ("remap_to_fp32_candidate", "needs_remeasure", "quarantine")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-rows", default=str(DEFAULT_INPUT_ROWS))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--created-at")
    return parser.parse_args()


def read_jsonl_rows(path: str | Path) -> list[dict[str, Any]]:
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


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def width_text(width: list[int]) -> str:
    return "x".join(str(item) for item in width)


def label_from_row(row: dict[str, Any]) -> str:
    explicit = str(row.get("label") or "")
    if explicit:
        return explicit
    software_point_id = str(row.get("software_point_id") or "")
    if software_point_id.startswith("original60:"):
        parts = software_point_id.split(":")
        if len(parts) > 1 and parts[1]:
            return parts[1]
    for source in as_list(row.get("source_files")):
        basename = Path(str(source)).name
        if basename.endswith("_backbone.onnx"):
            return basename[: -len("_backbone.onnx")]
    raw_artifact = str(row.get("raw_artifact") or "")
    match = re.search(r"original60_([^/]+?)_gpu", raw_artifact)
    if match:
        return match.group(1)
    return ""


def width_from_row(row: dict[str, Any]) -> list[int]:
    width = row.get("width")
    if isinstance(width, list) and width:
        return [int(item) for item in width]
    for key in ("software_point_id", "candidate_id"):
        match = re.search(r"(\d+)x(\d+)x(\d+)", str(row.get(key) or ""))
        if match:
            return [int(match.group(1)), int(match.group(2)), int(match.group(3))]
    raise ValueError(f"cannot infer width for row {row.get('run_id') or row.get('candidate_id')}")


def source_files(row: dict[str, Any]) -> list[str]:
    return [str(item) for item in as_list(row.get("source_files")) if str(item)]


def has_plain_fp32_backbone_source(row: dict[str, Any]) -> bool:
    for source in source_files(row):
        text = source.lower()
        basename = Path(source).name.lower()
        if any(token in text for token in ("true_fp16", "_fp16", "fp16.", "half")):
            continue
        if basename.endswith("_backbone.onnx"):
            return True
    return False


def has_fp16_source_marker(row: dict[str, Any]) -> bool:
    text = " ".join(source_files(row)).lower()
    return any(token in text for token in ("true_fp16", "_fp16", "fp16.", "half"))


def canonical_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def score(row: dict[str, Any]) -> tuple[int, str]:
        schedule = str(row.get("schedule_policy") or row.get("schedule_profile") or "")
        tuned_score = 1 if schedule == "metaschedule_tuned" else 0
        return tuned_score, str(row.get("created_at") or row.get("run_id") or "")

    return max(rows, key=score)


def classify(row: dict[str, Any]) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if not label_from_row(row):
        return "quarantine", ["missing_label"]
    if str(row.get("measurement_status")) != "measured":
        return "quarantine", ["measurement_status_not_measured"]
    if not str(row.get("backend") or "").startswith("h800_tvm"):
        return "quarantine", ["backend_not_h800_tvm"]
    if bool(row.get("full_network_claim")):
        return "quarantine", ["full_network_claim_true"]
    if row.get("latency_p50_us") is None:
        return "quarantine", ["missing_latency_p50_us"]
    quant_text = " ".join(str(row.get(key) or "") for key in ("precision", "quant_policy", "candidate_id", "software_point_id"))
    if "fp16" not in quant_text.lower():
        reasons.append("not_fp16_tagged")
    if has_fp16_source_marker(row):
        return "needs_remeasure", reasons + ["source_contains_fp16_marker"]
    if not has_plain_fp32_backbone_source(row):
        return "needs_remeasure", reasons + ["missing_plain_backbone_onnx_source"]
    return "remap_to_fp32_candidate", reasons + ["source_files_point_to_plain_backbone_onnx"]


def config_id(label: str, schedule_policy: str) -> str:
    return stable_config_id(
        model=MODEL,
        candidate_id=label,
        software_point_id=f"original60:{label}:fp32:{OPTIMIZED_SCOPE}:latency_remapped",
        quant_policy="fp32",
        schedule_policy=schedule_policy,
    )


def fp32_candidate_id(row: dict[str, Any], label: str, width: list[int]) -> str:
    candidate_id = str(row.get("candidate_id") or "")
    parts = candidate_id.split(":")
    if parts and parts[-1] in {"fp16", "fp32", "int8"}:
        return ":".join([*parts[:-1], "fp32"])
    if candidate_id and candidate_id != label:
        return f"{candidate_id}:fp32"
    return f"coverage:pyramid_lidar:w{width_text(width)}:fp32"


def remapped_row(row: dict[str, Any], *, created_at: str) -> dict[str, Any]:
    label = label_from_row(row)
    width = width_from_row(row)
    schedule_policy = str(row.get("schedule_policy") or row.get("schedule_profile") or "metaschedule_tuned")
    return latency_lut_row(
        config_id=config_id(label, schedule_policy),
        model=MODEL,
        manifest_digest=MANIFEST_DIGEST,
        candidate_id=fp32_candidate_id(row, label, width),
        software_point_id=f"original60:{label}:{width_text(width)}:fp32:latency_remapped",
        dense_stage="backbone",
        optimized_scope=OPTIMIZED_SCOPE,
        width=width,
        quant_policy="fp32",
        schedule_policy=schedule_policy,
        backend="h800_tvm",
        measurement_status="measured",
        latency_p50_us=float(row["latency_p50_us"]),
        latency_p90_us=row.get("latency_p90_us"),
        latency_mean_us=row.get("latency_mean_us"),
        latency_std_us=row.get("latency_std_us"),
        latency_min_us=row.get("latency_min_us"),
        latency_max_us=row.get("latency_max_us"),
        warmup_iters=int(row.get("warmup_iters") or 50),
        measure_iters=int(row.get("measure_iters") or 200),
        repeat=int(row.get("repeat") or 5),
        batch_size=int(row.get("batch_size") or 1),
        tvm_target=str(row.get("tvm_target") or "cuda"),
        tvm_strategy=str(row.get("tvm_strategy") or "relax_metaschedule_reuse_existing_ms_db"),
        build_status=str(row.get("build_status") or "success"),
        provenance="FP32 remap audit from suspect original60 FP16-tagged H800 TVM row",
        run_id=str(row.get("run_id") or f"fp32_remap_{label}_{schedule_policy}"),
        created_at=created_at,
        source_files=source_files(row),
        raw_artifact=row.get("raw_artifact"),
        failure_reason=None,
        notes="historical FP16-tagged row reclassified only after source_files pointed to plain FP32 backbone ONNX",
        precision="fp32",
        quant_scheme="none",
        quant_method="h800_tvm_relax_fp32",
        quant_scope=OPTIMIZED_SCOPE,
        calibration_source="none",
        calibration_digest="none",
        calibrator="none",
        calibration_inputs=[],
        fallback_policy="none",
        layer_precision_summary="not_applicable_fp32_remapped_from_suspect_fp16_tagged_row",
        full_network_claim=False,
        engine_kind="tvm_vm",
        engine_digest=str(row.get("engine_digest") or "unknown"),
        measurement_source="historical_true_measurement_reclassified",
        claim_status="claimable_true_measurement_remapped",
        quality_gate_status="fp32_reclassified_from_suspect_fp16_tagged_row",
        schedule_profile=schedule_policy,
        tune_budget=str(row.get("tune_budget") or "historical_original60_measurement"),
        label=label,
        original_precision=str(row.get("precision") or "fp16"),
        original_quant_policy=str(row.get("quant_policy") or "fp16"),
        original_candidate_id=str(row.get("candidate_id") or ""),
        original_software_point_id=str(row.get("software_point_id") or ""),
        original_run_id=str(row.get("run_id") or ""),
    )


def audit_rows(rows: list[dict[str, Any]], *, created_at: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    orphan_rows: list[dict[str, Any]] = []
    for row in rows:
        label = label_from_row(row)
        if label:
            grouped[label].append(row)
        else:
            orphan_rows.append(row)

    audit: list[dict[str, Any]] = []
    remapped: list[dict[str, Any]] = []
    for label in sorted(grouped):
        chosen = canonical_row(grouped[label])
        classification, reasons = classify(chosen)
        audit.append(
            {
                "label": label,
                "classification": classification,
                "reasons": reasons,
                "run_id": chosen.get("run_id"),
                "schedule_policy": chosen.get("schedule_policy"),
                "latency_ms": None
                if chosen.get("latency_p50_us") is None
                else float(chosen["latency_p50_us"]) / 1000.0,
                "source_files": source_files(chosen),
                "raw_artifact": chosen.get("raw_artifact"),
                "row_count": len(grouped[label]),
            }
        )
        if classification == "remap_to_fp32_candidate":
            remapped.append(remapped_row(chosen, created_at=created_at))

    for row in orphan_rows:
        audit.append(
            {
                "label": "",
                "classification": "quarantine",
                "reasons": ["missing_label"],
                "run_id": row.get("run_id"),
                "schedule_policy": row.get("schedule_policy"),
                "latency_ms": None,
                "source_files": source_files(row),
                "raw_artifact": row.get("raw_artifact"),
                "row_count": 1,
            }
        )
    return audit, remapped


def write_audit(output_root: Path, audit: list[dict[str, Any]], remapped: list[dict[str, Any]], *, created_at: str) -> None:
    exports = output_root / "exports"
    exports.mkdir(parents=True, exist_ok=True)
    counts = Counter(row["classification"] for row in audit)
    classification_counts = {key: int(counts.get(key, 0)) for key in CLASSIFICATIONS}
    payload = {
        "schema": "fp32_original60_remap_audit_v1",
        "created_at": created_at,
        "label_count": len({row["label"] for row in audit if row["label"]}),
        "classification_counts": classification_counts,
        "remapped_row_count": len(remapped),
        "rows": audit,
    }
    (exports / "fp32_original60_remap_audit_latest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# FP32 Original60 Remap Audit",
        "",
        "说明: 历史 FP16-tagged row 只有在 source_files 指向 plain `*_backbone.onnx` 且没有 FP16 marker 时, 才生成 FP32 remapped row。",
        "",
        "| classification | count |",
        "|---|---:|",
    ]
    for key in CLASSIFICATIONS:
        lines.append(f"| {key} | {classification_counts[key]} |")
    lines.extend(
        [
            "",
            "| label | classification | latency ms | schedule | reasons | source files |",
            "|---|---|---:|---|---|---|",
        ]
    )
    for row in audit:
        latency = "" if row["latency_ms"] is None else f"{row['latency_ms']:.6f}"
        lines.append(
            "| {label} | {classification} | {latency} | {schedule} | {reasons} | {sources} |".format(
                label=row["label"],
                classification=row["classification"],
                latency=latency,
                schedule=row.get("schedule_policy") or "",
                reasons=";".join(row.get("reasons") or []),
                sources="<br>".join(row.get("source_files") or []),
            )
        )
    (exports / "fp32_original60_remap_audit_latest.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    created_at = args.created_at or utc_timestamp()
    output_root = Path(args.output_root)
    rows = read_jsonl_rows(args.input_rows)
    audit, remapped = audit_rows(rows, created_at=created_at)
    write_jsonl(output_root / "rows/fp32_latency_original60_remapped_rows_v1.jsonl", remapped)
    write_audit(output_root, audit, remapped, created_at=created_at)
    print(
        json.dumps(
            {
                "schema": "fp32_original60_remap_audit_result_v1",
                "output_root": str(output_root),
                "labels": len({row["label"] for row in audit if row["label"]}),
                "remapped_rows": len(remapped),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
