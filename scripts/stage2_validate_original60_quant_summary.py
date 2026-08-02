#!/usr/bin/env python3
"""Validate the original60 quant three-metric summary before large GPU runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = (
    ROOT / "multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627"
)
EXPECTED_LABELS = {"s0_024", "lhc_07", "lhc_20", "s0_056", "frontier_25"}
EXPECTED_PRECISIONS = ("fp32", "fp16", "int8")
SUMMARY_SCHEMA = "original60_quant_three_metric_summary_v1"
SOURCE_AUDIT_SCHEMA = "original60_quant_measurement_source_audit_v3"
FP32_ENERGY_REVIEW_SCHEMA = "fp32_threaded_window_energy_60label_review_v1"
FP16_TIR_AUDIT_SCHEMA = "fp16_tir_lowering_5label_audit_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--write-freeze",
        action="store_true",
        help="Write exports/original60_quant_trusted_freeze_latest.{json,md} after PASS.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def dynamic_watt(row: dict[str, Any]) -> float | None:
    watt = row.get("watt_avg")
    idle = row.get("idle_watt_avg")
    if watt is None or idle is None:
        return None
    return float(watt) - float(idle)


def row_label(row: dict[str, Any]) -> str:
    explicit = row.get("label")
    if explicit not in (None, ""):
        return str(explicit)
    software_point_id = str(row.get("software_point_id") or "")
    if software_point_id.startswith("original60:"):
        parts = software_point_id.split(":")
        if len(parts) >= 2 and parts[1]:
            return parts[1]
    run_id = str(row.get("run_id") or "")
    prefix = "fp32_energy_threaded60_"
    suffix = "_20260629_"
    if run_id.startswith(prefix) and suffix in run_id:
        return run_id[len(prefix) : run_id.index(suffix)]
    return ""


def validate_summary(summary: dict[str, Any], errors: list[str]) -> dict[str, Any]:
    rows = summary.get("rows")
    require(summary.get("schema") == SUMMARY_SCHEMA, f"summary schema must be {SUMMARY_SCHEMA}", errors)
    require(isinstance(rows, list), "summary rows must be a list", errors)
    rows = rows if isinstance(rows, list) else []
    require(len(rows) == 180, f"summary row count must be 180, got {len(rows)}", errors)
    labels = {str(row.get("label")) for row in rows}
    require(len(labels) == 60, f"summary unique label count must be 60, got {len(labels)}", errors)

    precision_counts = Counter(str(row.get("precision")) for row in rows)
    for precision in EXPECTED_PRECISIONS:
        require(
            precision_counts.get(precision, 0) == 60,
            f"summary precision {precision} must have 60 rows, got {precision_counts.get(precision, 0)}",
            errors,
        )

    required_fields = (
        "label",
        "precision",
        "latency_ms",
        "energy_j_per_inference",
        "ap70",
        "latency_measurement_source",
        "energy_measurement_source",
        "ap_measurement_source",
        "latency_schedule_policy",
        "energy_schedule_policy",
        "quality_gate_status",
    )
    for index, row in enumerate(rows):
        missing = [field for field in required_fields if field not in row]
        require(not missing, f"summary row {index} missing fields: {','.join(missing)}", errors)
        if row.get("latency_status") == "measured":
            require(row.get("latency_ms") is not None, f"{row.get('label')} {row.get('precision')} measured latency missing ms", errors)
        if row.get("energy_status") == "measured":
            require(
                row.get("energy_j_per_inference") is not None,
                f"{row.get('label')} {row.get('precision')} measured energy missing joule/inference",
                errors,
            )

    by_precision = {precision: [row for row in rows if row.get("precision") == precision] for precision in EXPECTED_PRECISIONS}
    fp32 = by_precision["fp32"]
    require(sum(row.get("energy_status") == "measured" for row in fp32) == 60, "FP32 energy must be measured 60/60", errors)
    require(
        all(row.get("energy_measurement_source") == "true_measurement" for row in fp32),
        "FP32 energy source must be true_measurement for all 60 rows",
        errors,
    )
    require(
        all("fp32_energy_threaded_window_h800" in str(row.get("quality_gate_status") or "") for row in fp32),
        "FP32 summary rows must include fp32_energy_threaded_window_h800 quality gate",
        errors,
    )
    fp32_energy_schedule = Counter(str(row.get("energy_schedule_policy") or "") for row in fp32)
    require(
        fp32_energy_schedule == Counter({"metaschedule_tuned": 56, "default": 4}),
        f"FP32 energy schedule distribution must be metaschedule_tuned=56/default=4, got {dict(fp32_energy_schedule)}",
        errors,
    )

    int8 = by_precision["int8"]
    int8_measured_ap = [row for row in int8 if row.get("ap_status") == "measured"]
    for row in int8_measured_ap:
        quality = str(row.get("quality_gate_status") or "")
        require("smoke" not in quality.lower(), f"INT8 AP measured row {row.get('label')} must not come from smoke gate", errors)
        require(
            "native_int8_full_ap_eval" in quality or "native_int8_full_onnx_original60_ap_eval" in quality,
            f"INT8 AP measured row {row.get('label')} must use full AP eval quality gate",
            errors,
        )

    fp16 = by_precision["fp16"]
    require(sum(row.get("latency_status") == "measured" for row in fp16) == 60, "FP16 latency must remain measured 60/60", errors)
    require(sum(row.get("energy_status") == "measured" for row in fp16) == 60, "FP16 energy must remain measured 60/60", errors)

    return {
        "precision_counts": dict(sorted(precision_counts.items())),
        "fp32_energy_schedule": dict(sorted(fp32_energy_schedule.items())),
        "fp32_energy_measured": sum(row.get("energy_status") == "measured" for row in fp32),
        "fp16_latency_measured": sum(row.get("latency_status") == "measured" for row in fp16),
        "fp16_energy_measured": sum(row.get("energy_status") == "measured" for row in fp16),
        "int8_ap_measured": len(int8_measured_ap),
    }


def validate_fp32_energy_rows(output_root: Path, errors: list[str]) -> dict[str, Any]:
    row_path = output_root / "rows/fp32_original60_energy_threaded60_rows_v1.jsonl"
    rows = read_jsonl(row_path)
    require(len(rows) == 60, f"FP32 threaded energy row file must contain 60 rows, got {len(rows)}", errors)
    labels = {row_label(row) for row in rows if row_label(row)}
    require(len(labels) == 60, f"FP32 threaded energy rows must contain 60 unique labels, got {len(labels)}", errors)
    bad_status = [row.get("label") for row in rows if row.get("measurement_status") != "measured"]
    require(not bad_status, f"FP32 threaded energy rows must all be measured, bad={bad_status[:5]}", errors)
    bad_quality = [
        row_label(row)
        for row in rows
        if row.get("quality_gate_status") != "fp32_energy_threaded_window_h800"
    ]
    require(not bad_quality, f"FP32 threaded energy rows must use fp32_energy_threaded_window_h800, bad={bad_quality[:5]}", errors)
    missing_trace = [
        row_label(row)
        for row in rows
        if not row.get("raw_artifact") or not row.get("run_id") or not row.get("source_files")
    ]
    require(not missing_trace, f"FP32 threaded energy rows missing raw/run/source trace, bad={missing_trace[:5]}", errors)
    low_dynamic = [
        row_label(row)
        for row in rows
        if dynamic_watt(row) is None or float(dynamic_watt(row) or 0.0) < 50.0
    ]
    require(not low_dynamic, f"FP32 threaded energy dynamic watt must be >=50W for all rows, bad={low_dynamic[:5]}", errors)
    schedule = Counter(str(row.get("schedule_policy") or "") for row in rows)
    return {
        "row_path": str(row_path),
        "row_count": len(rows),
        "unique_labels": len(labels),
        "dynamic_watt_lt_50_count": len(low_dynamic),
        "schedule_counts": dict(sorted(schedule.items())),
    }


def validate_fp32_energy_review(review: dict[str, Any], errors: list[str]) -> dict[str, Any]:
    require(
        review.get("schema") == FP32_ENERGY_REVIEW_SCHEMA,
        f"FP32 energy review schema must be {FP32_ENERGY_REVIEW_SCHEMA}",
        errors,
    )
    require(review.get("row_count") == 60, f"FP32 energy review row_count must be 60, got {review.get('row_count')}", errors)
    require(
        review.get("unique_label_count") == 60,
        f"FP32 energy review unique_label_count must be 60, got {review.get('unique_label_count')}",
        errors,
    )
    require(
        review.get("low_new_dynamic_w_lt_50_count") == 0,
        f"FP32 energy review low_new_dynamic_w_lt_50_count must be 0, got {review.get('low_new_dynamic_w_lt_50_count')}",
        errors,
    )
    return {
        "row_count": review.get("row_count"),
        "unique_label_count": review.get("unique_label_count"),
        "low_new_dynamic_w_lt_50_count": review.get("low_new_dynamic_w_lt_50_count"),
        "schedule_counts": review.get("schedule_counts") or {},
    }


def validate_source_audit(source_audit: dict[str, Any], errors: list[str]) -> dict[str, Any]:
    require(
        source_audit.get("schema") == SOURCE_AUDIT_SCHEMA,
        f"source audit schema must be {SOURCE_AUDIT_SCHEMA}",
        errors,
    )
    conclusion = source_audit.get("conclusion")
    require(isinstance(conclusion, dict), "source audit conclusion must be an object", errors)
    conclusion = conclusion if isinstance(conclusion, dict) else {}
    fp16_note = str(conclusion.get("fp16_latency_energy") or "")
    require(
        "no tensorcore/wmma" in fp16_note or "no tensorcore" in fp16_note,
        "source audit must preserve FP16 tensorcore/wmma limitation",
        errors,
    )
    fp32_energy = source_audit.get("fp32_energy_threaded60") or {}
    require(
        fp32_energy.get("canonical_fp32_energy_rows") == 60,
        "source audit must record 60 canonical FP32 threaded energy rows",
        errors,
    )
    require(
        fp32_energy.get("dynamic_watt_lt_50_count") == 0,
        "source audit must record FP32 dynamic_watt_lt_50_count=0",
        errors,
    )
    return {
        "all_cells_are_direct_same_level_measurements": conclusion.get(
            "all_cells_are_direct_same_level_measurements"
        ),
        "fp16_limitation_recorded": True,
        "canonical_fp32_energy_rows": fp32_energy.get("canonical_fp32_energy_rows"),
        "dynamic_watt_lt_50_count": fp32_energy.get("dynamic_watt_lt_50_count"),
    }


def validate_fp16_tir_audit(tir_audit: dict[str, Any], errors: list[str]) -> dict[str, Any]:
    require(
        tir_audit.get("schema") == FP16_TIR_AUDIT_SCHEMA,
        f"FP16 TIR audit schema must be {FP16_TIR_AUDIT_SCHEMA}",
        errors,
    )
    rows = tir_audit.get("rows")
    require(isinstance(rows, list), "FP16 TIR audit rows must be a list", errors)
    rows = rows if isinstance(rows, list) else []
    labels = {str(row.get("label")) for row in rows}
    require(labels == EXPECTED_LABELS, f"FP16 TIR audit labels must be {sorted(EXPECTED_LABELS)}, got {sorted(labels)}", errors)
    classifications = Counter(str(row.get("classification") or "") for row in rows)
    require(
        classifications == Counter({"fp16_dtype_present_no_tensorcore_evidence": 5}),
        f"FP16 TIR audit must classify all 5 labels as fp16_dtype_present_no_tensorcore_evidence, got {dict(classifications)}",
        errors,
    )
    for row in rows:
        input_dtypes = row.get("input_dtypes") or {}
        init_counts = row.get("initializer_dtype_counts") or {}
        keyword_counts = row.get("keyword_file_counts") or {}
        require("float16" in set(str(value) for value in input_dtypes.values()), f"{row.get('label')} FP16 input dtype evidence missing", errors)
        require(int(init_counts.get("float16") or 0) > 0, f"{row.get('label')} FP16 initializer dtype evidence missing", errors)
        require(int(keyword_counts.get("wmma") or 0) == 0, f"{row.get('label')} unexpectedly has wmma evidence", errors)
        require(int(keyword_counts.get("tensorcore") or 0) == 0, f"{row.get('label')} unexpectedly has tensorcore evidence", errors)
        require(int(keyword_counts.get("tensor_core") or 0) == 0, f"{row.get('label')} unexpectedly has tensor_core evidence", errors)
        require(int(keyword_counts.get("float32") or 0) > 0, f"{row.get('label')} should retain float32-heavy workdir evidence", errors)
    return {
        "labels": sorted(labels),
        "classifications": dict(sorted(classifications.items())),
        "closure_conclusion": (
            "A. 当前 FP16 route 不是 tensor-core optimized; "
            "证据显示 FP16 ONNX/input/initializer 成立, 但 sampled MetaSchedule workdir 无 wmma/tensorcore/tensor_core 命中, "
            "后续需独立 FP16 workdir + 正确 tensorization target 重新 tune/build 后再重测 latency/energy。"
        ),
    }


def build_freeze_payload(
    output_root: Path,
    paths: dict[str, Path],
    validation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "original60_quant_trusted_freeze_v1",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "output_root": str(output_root),
        "status": validation["status"],
        "source_files": {name: str(path) for name, path in paths.items()},
        "source_sha256": {name: sha256_file(path) for name, path in paths.items()},
        "validation": validation,
    }


def write_freeze(output_root: Path, payload: dict[str, Any]) -> None:
    exports = output_root / "exports"
    json_path = exports / "original60_quant_trusted_freeze_latest.json"
    md_path = exports / "original60_quant_trusted_freeze_latest.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    validation = payload["validation"]
    md = [
        "# Original60 Quant Trusted Freeze",
        "",
        f"- status: `{payload['status']}`",
        f"- created_at: `{payload['created_at']}`",
        f"- output_root: `{payload['output_root']}`",
        "",
        "## Gate Summary",
        "",
        f"- summary rows: `{validation['summary']['precision_counts']}`",
        f"- FP32 energy schedule: `{validation['summary']['fp32_energy_schedule']}`",
        f"- FP32 dynamic watt < 50W: `{validation['fp32_energy_rows']['dynamic_watt_lt_50_count']}`",
        f"- INT8 measured AP rows: `{validation['summary']['int8_ap_measured']}`",
        f"- FP16 TIR closure: {validation['fp16_tir_audit']['closure_conclusion']}",
        "",
        "## Source SHA256",
        "",
    ]
    for name, digest in payload["source_sha256"].items():
        md.append(f"- `{name}`: `{digest}`")
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    exports = output_root / "exports"
    paths = {
        "summary_json": exports / "original60_quant_three_metric_summary_latest.json",
        "summary_csv": exports / "original60_quant_three_metric_summary_latest.csv",
        "summary_md": exports / "original60_quant_three_metric_summary_latest.md",
        "source_audit_json": exports / "original60_quant_measurement_source_audit_latest.json",
        "source_audit_md": exports / "original60_quant_measurement_source_audit_latest.md",
        "fp32_energy_review_json": exports / "fp32_threaded_window_energy_60label_review_latest.json",
        "fp16_tir_audit_json": exports / "fp16_tir_lowering_5label_audit_latest.json",
        "fp16_tir_audit_md": exports / "fp16_tir_lowering_5label_audit_latest.md",
        "fp32_threaded_energy_rows": output_root / "rows/fp32_original60_energy_threaded60_rows_v1.jsonl",
    }
    errors: list[str] = []
    for name, path in paths.items():
        require(path.exists(), f"required file missing: {name} -> {path}", errors)
    if errors:
        print(json.dumps({"schema": "original60_quant_summary_validation_v1", "status": "FAIL", "errors": errors}, ensure_ascii=False, indent=2))
        return 1

    validation = {
        "schema": "original60_quant_summary_validation_v1",
        "status": "PASS",
        "summary": validate_summary(load_json(paths["summary_json"]), errors),
        "fp32_energy_rows": validate_fp32_energy_rows(output_root, errors),
        "fp32_energy_review": validate_fp32_energy_review(load_json(paths["fp32_energy_review_json"]), errors),
        "source_audit": validate_source_audit(load_json(paths["source_audit_json"]), errors),
        "fp16_tir_audit": validate_fp16_tir_audit(load_json(paths["fp16_tir_audit_json"]), errors),
    }
    if errors:
        validation["status"] = "FAIL"
        validation["errors"] = errors
        print(json.dumps(validation, ensure_ascii=False, indent=2, sort_keys=True))
        return 1

    if args.write_freeze:
        write_freeze(output_root, build_freeze_payload(output_root, paths, validation))
    print(json.dumps(validation, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
