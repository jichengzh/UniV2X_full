#!/usr/bin/env python3
"""Generate Stage2 quant anchor smoke state rows and summary reports."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.stage2.lut_productization import (  # noqa: E402
    ap_anchor_row,
    energy_lut_row,
    job_plan_row,
    latency_lut_row,
    quarantine_row,
    stable_config_id,
    utc_timestamp,
    write_jsonl,
)


DEFAULT_OUTPUT_ROOT = (
    ROOT / "multi_agent/data/stage2_lut_generation_v1/generated/quant_smoke_20260627"
)
DEFAULT_HISTORICAL_LATENCY = (
    ROOT / "multi_agent/data/stage2_lut_generation_v1/existing_h800_tvm/"
    "pyramid_h800_tvm_gap1_corrected_ms.csv"
)
DEFAULT_ORIGINAL60_SUMMARY = (
    ROOT / "multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1/"
    "exports/original60_three_metric_summary_latest.csv"
)
DEFAULT_AP_SUMMARY = (
    ROOT / "multi_agent/data/stage2_lut_generation_v1/generated/ap_stability_20260626/"
    "exports/ap_stability_summary.csv"
)

ANCHORS: list[dict[str, Any]] = [
    {"label": "base", "width": [64, 128, 256], "source": "historical_h800_tvm"},
    {"label": "p50", "width": [32, 64, 128], "source": "historical_h800_tvm"},
    {"label": "p75", "width": [16, 32, 64], "source": "historical_h800_tvm"},
    {"label": "trap25", "width": [48, 96, 192], "source": "historical_h800_tvm"},
    {"label": "s0_024", "width": [24, 128, 256], "source": "original60_phase_c"},
    {"label": "s0_040", "width": [40, 128, 256], "source": "original60_phase_c"},
    {"label": "s0_056", "width": [56, 128, 256], "source": "original60_phase_c"},
    {"label": "s1_048", "width": [64, 48, 256], "source": "original60_phase_c"},
]
PRECISIONS = ("fp32", "fp16", "int8")
MANIFEST_DIGEST = "quant_smoke_20260627_existing_h800_tvm_seed"
MODEL = "pyramid_lidar"
OPTIMIZED_SCOPE = "backbone_only"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--historical-latency-csv", default=str(DEFAULT_HISTORICAL_LATENCY))
    parser.add_argument("--original60-summary-csv", default=str(DEFAULT_ORIGINAL60_SUMMARY))
    parser.add_argument("--ap-summary-csv", default=str(DEFAULT_AP_SUMMARY))
    parser.add_argument(
        "--fp32-latency-smoke-rows",
        action="append",
        default=[],
        help="Optional JSONL latency rows from the narrow base/s0_024/s1_048 FP32 H800 TVM smoke.",
    )
    parser.add_argument(
        "--fp16-latency-smoke-rows",
        action="append",
        default=[],
        help="Optional JSONL latency rows from the true-FP16 ONNX base/s0_024/s1_048 H800 TVM smoke.",
    )
    parser.add_argument(
        "--fp16-energy-smoke-rows",
        action="append",
        default=[],
        help="Optional JSONL energy rows from the true-FP16 ONNX base/s0_024/s1_048 H800 power telemetry smoke.",
    )
    parser.add_argument(
        "--int8-latency-smoke-rows",
        action="append",
        default=[],
        help="Optional JSONL latency rows from the INT8 QDQ TVM VM base/s0_024/s1_048 H800 TVM smoke.",
    )
    parser.add_argument(
        "--int8-energy-smoke-rows",
        action="append",
        default=[],
        help="Optional JSONL energy rows from the INT8 QDQ TVM VM base/s0_024/s1_048 H800 power telemetry smoke.",
    )
    parser.add_argument("--created-at", default=None)
    return parser.parse_args()


def read_csv(path: str | Path) -> list[dict[str, str]]:
    item = Path(path)
    if not item.exists():
        return []
    with item.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl_rows(path: str | Path) -> list[dict[str, Any]]:
    item = Path(path)
    if not item.exists():
        return []
    rows = []
    for line in item.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def smoke_label(row: dict[str, Any]) -> str:
    explicit = str(row.get("label") or "")
    if explicit:
        return explicit
    software_point_id = str(row.get("software_point_id") or "")
    if software_point_id.startswith("original60:"):
        parts = software_point_id.split(":")
        if len(parts) > 1 and parts[1]:
            return parts[1]
    candidate_id = str(row.get("candidate_id") or "")
    if candidate_id and ":" not in candidate_id:
        return candidate_id
    return ""


def measured_fp32_smoke_index(paths: list[str]) -> dict[str, dict[str, Any]]:
    measured = []
    for path in paths:
        for row in read_jsonl_rows(path):
            if str(row.get("precision")) != "fp32":
                continue
            if str(row.get("measurement_status")) != "measured":
                continue
            if not str(row.get("backend", "")).startswith("h800_tvm"):
                continue
            if bool(row.get("full_network_claim")):
                continue
            if row.get("latency_p50_us") is None:
                continue
            measured.append(row)

    def score(row: dict[str, Any]) -> tuple[int, str]:
        schedule = str(row.get("schedule_policy") or row.get("schedule_profile") or "")
        tuned_score = 1 if schedule == "metaschedule_tuned" else 0
        return tuned_score, str(row.get("created_at") or row.get("run_id") or "")

    out: dict[str, dict[str, Any]] = {}
    for row in measured:
        label = smoke_label(row)
        if not label:
            continue
        if label not in out or score(row) > score(out[label]):
            out[label] = row
    return out


def measured_fp16_smoke_index(paths: list[str]) -> dict[str, dict[str, Any]]:
    measured = []
    for path in paths:
        for row in read_jsonl_rows(path):
            if str(row.get("precision")) != "fp16":
                continue
            if str(row.get("measurement_status")) != "measured":
                continue
            if not str(row.get("backend", "")).startswith("h800_tvm"):
                continue
            if bool(row.get("full_network_claim")):
                continue
            if row.get("latency_p50_us") is None:
                continue
            measured.append(row)

    def score(row: dict[str, Any]) -> tuple[int, str]:
        schedule = str(row.get("schedule_policy") or row.get("schedule_profile") or "")
        tuned_score = 1 if schedule == "metaschedule_tuned" else 0
        return tuned_score, str(row.get("created_at") or row.get("run_id") or "")

    out: dict[str, dict[str, Any]] = {}
    for row in measured:
        label = smoke_label(row)
        if not label:
            continue
        if label not in out or score(row) > score(out[label]):
            out[label] = row
    return out


def measured_int8_smoke_index(paths: list[str]) -> dict[str, dict[str, Any]]:
    measured = []
    for path in paths:
        for row in read_jsonl_rows(path):
            if str(row.get("precision")) != "int8":
                continue
            if str(row.get("measurement_status")) != "measured":
                continue
            if not str(row.get("backend", "")).startswith("h800_tvm"):
                continue
            if bool(row.get("full_network_claim")):
                continue
            if row.get("latency_p50_us") is None:
                continue
            measured.append(row)

    def score(row: dict[str, Any]) -> tuple[int, str]:
        schedule = str(row.get("schedule_policy") or row.get("schedule_profile") or "")
        direct_score = 1 if schedule == "int8_qdq_direct_tvm_vm" else 0
        return direct_score, str(row.get("created_at") or row.get("run_id") or "")

    out: dict[str, dict[str, Any]] = {}
    for row in measured:
        label = smoke_label(row)
        if not label:
            continue
        if label not in out or score(row) > score(out[label]):
            out[label] = row
    return out


def measured_energy_index(paths: list[str], precision: str) -> dict[str, dict[str, Any]]:
    measured = []
    for path in paths:
        for row in read_jsonl_rows(path):
            if str(row.get("precision") or row.get("quant_policy")) != precision:
                continue
            if str(row.get("measurement_status")) != "measured":
                continue
            if not str(row.get("backend", "")).startswith("h800_tvm_power_telemetry"):
                continue
            if bool(row.get("full_network_claim")):
                continue
            if row.get("joule_per_inference") is None:
                continue
            source_files = " ".join(str(item) for item in row.get("source_files") or [])
            if "idle_power_samples.csv" not in source_files or "active_power_samples.csv" not in source_files:
                continue
            measured.append(row)

    def score(row: dict[str, Any]) -> tuple[str, str]:
        return str(row.get("created_at") or ""), str(row.get("run_id") or "")

    out: dict[str, dict[str, Any]] = {}
    for row in measured:
        label = smoke_label(row)
        if not label:
            continue
        if label not in out or score(row) > score(out[label]):
            out[label] = row
    return out


def width_text(width: list[int]) -> str:
    return "x".join(str(item) for item in width)


def parse_width(value: Any) -> list[int]:
    if isinstance(value, list):
        return [int(item) for item in value]
    text = str(value or "").strip()
    if not text:
        return []
    if "x" in text and not text.startswith("["):
        return [int(item) for item in text.split("x") if item]
    parsed = ast.literal_eval(text)
    return [int(item) for item in parsed]


def float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def evidence_scope(row: dict[str, Any]) -> str:
    if str(row.get("measurement_status") or "") != "measured":
        return "no_claim"
    tokens = " ".join(
        str(row.get(key) or "")
        for key in ("measurement_source", "claim_status", "quality_gate_status")
    ).lower()
    if "smoke" in tokens:
        return "smoke_only"
    if "reclassified" in tokens or "remapped" in tokens:
        return "remapped_reference"
    if str(row.get("measurement_source") or "") in ("true_measurement", "true_eval"):
        return "measured"
    return "measured_unspecified"


def quant_contract(precision: str, *, status: str) -> dict[str, Any]:
    if precision == "int8":
        return {
            "precision": "int8",
            "quant_scheme": "tvm_int8_experimental",
            "quant_method": "h800_tvm_int8_backbone_subnet_experimental",
            "quant_scope": "backbone_only_requested",
            "calibration_source": "missing_tvm_int8_backbone_subnet_calibration_manifest",
            "calibration_digest": "unknown",
            "calibrator": "missing_tvm_quant_calibration_manifest",
            "calibration_inputs": ["spatial_features"],
            "fallback_policy": "unknown_pending_tvm_inventory",
            "layer_precision_summary": "unknown_pending_tvm_inventory",
            "full_network_claim": False,
            "engine_kind": "tvm_vm",
            "engine_digest": "unknown",
            "measurement_source": "no_claim",
            "claim_status": "no_claim",
            "quality_gate_status": (
                "tvm_int8_backbone_subnet_not_ready"
                if status != "measured"
                else "unexpected_int8_measured"
            ),
            "schedule_profile": "metaschedule_tuned",
            "tune_budget": "unknown",
        }
    if precision == "fp32":
        return {
            "precision": "fp32",
            "quant_scheme": "none",
            "quant_method": "h800_tvm_relax_fp32",
            "quant_scope": OPTIMIZED_SCOPE,
            "calibration_source": "none",
            "calibration_digest": "none",
            "calibrator": "none",
            "calibration_inputs": [],
            "fallback_policy": "none",
            "layer_precision_summary": "not_applicable_fp32",
            "full_network_claim": False,
            "engine_kind": "tvm_vm",
            "engine_digest": "unknown",
            "measurement_source": "no_claim" if status != "measured" else "true_measurement_smoke",
            "claim_status": "no_claim" if status != "measured" else "claimable_true_measurement_smoke",
            "quality_gate_status": "tvm_fp32_backbone_measurement_not_run",
            "schedule_profile": "default",
            "tune_budget": "not_applicable",
        }
    return {
        "precision": "fp16",
        "quant_scheme": "fp16_cast",
        "quant_method": "h800_tvm_relax_metaschedule_fp16",
        "quant_scope": OPTIMIZED_SCOPE,
        "calibration_source": "none",
        "calibration_digest": "none",
        "calibrator": "none",
        "calibration_inputs": [],
        "fallback_policy": "none",
        "layer_precision_summary": "not_applicable_fp16",
        "full_network_claim": False,
        "engine_kind": "tvm_vm",
        "engine_digest": "unknown",
        "measurement_source": "true_measurement_smoke" if status == "measured" else "no_claim",
        "claim_status": "claimable_true_measurement_smoke" if status == "measured" else "no_claim",
        "quality_gate_status": "existing_h800_tvm_fp16_seed" if status == "measured" else "",
        "schedule_profile": "metaschedule_tuned",
        "tune_budget": "historical_or_original60",
    }


def ap_quant_contract(precision: str, *, status: str, source_kind: str = "") -> dict[str, Any]:
    contract = quant_contract(precision, status=status)
    contract["engine_kind"] = "model_eval" if status == "measured" else "tvm_vm"
    if status == "measured":
        contract["measurement_source"] = source_kind or "true_eval"
        contract["claim_status"] = "claimable_true_eval"
        contract["quality_gate_status"] = "fp16_true_ap_available"
    return contract


def config_id(label: str, precision: str, axis: str, schedule_policy: str | None = None) -> str:
    return stable_config_id(
        model=MODEL,
        candidate_id=label,
        software_point_id=f"{label}:{precision}:{OPTIMIZED_SCOPE}:{axis}",
        quant_policy=precision,
        schedule_policy=schedule_policy or ("metaschedule_tuned" if precision != "fp32" else "default"),
    )


def index_historical_latency(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {str(row.get("label")): row for row in rows if row.get("label")}


def index_original60(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {str(row.get("label")): row for row in rows if row.get("label")}


def index_ap(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {str(row.get("label")): row for row in rows if row.get("label")}


def make_latency_row(
    *,
    anchor: dict[str, Any],
    precision: str,
    historical_latency: dict[str, dict[str, str]],
    original60: dict[str, dict[str, str]],
    fp32_smoke_rows: dict[str, dict[str, Any]],
    fp16_smoke_rows: dict[str, dict[str, Any]],
    int8_smoke_rows: dict[str, dict[str, Any]],
    created_at: str,
) -> dict[str, Any]:
    label = str(anchor["label"])
    width = list(anchor["width"])
    source_files: list[str] = []
    latency_ms: float | None = None
    failure_reason: str | None = None
    notes = ""

    smoke_row: dict[str, Any] | None = None
    if precision == "fp16":
        smoke_row = fp16_smoke_rows.get(label)
        if smoke_row:
            latency_ms = float(smoke_row["latency_p50_us"]) / 1000.0
            source_files = smoke_row.get("source_files") or []
            notes = "imported from true-FP16 ONNX H800 TVM latency smoke"
        if latency_ms is None:
            failure_reason = "historical_fp16_tagged_latency_not_true_fp16_evidence"
            notes = "historical FP16-tagged latency is not accepted as true-FP16 evidence"
    elif precision == "fp32":
        failure_reason = "tvm_fp32_backbone_measurement_not_run"
    else:
        smoke_row = int8_smoke_rows.get(label)
        if smoke_row:
            latency_ms = float(smoke_row["latency_p50_us"]) / 1000.0
            source_files = smoke_row.get("source_files") or []
            notes = "imported from INT8 QDQ TVM VM H800 latency smoke"
        else:
            failure_reason = "tvm_int8_backbone_subnet_not_ready"

    smoke_row = fp32_smoke_rows.get(label) if precision == "fp32" else smoke_row
    if smoke_row:
        latency_ms = float(smoke_row["latency_p50_us"]) / 1000.0
        failure_reason = None
        source_files = smoke_row.get("source_files") or []
        if precision == "fp32":
            notes = "imported from narrow FP32 H800 TVM latency smoke"
        elif precision == "fp16":
            notes = "imported from true-FP16 ONNX H800 TVM latency smoke"
        elif precision == "int8":
            notes = "imported from INT8 QDQ TVM VM H800 latency smoke"
    status = "measured" if latency_ms is not None and failure_reason is None else "no_claim"
    contract = quant_contract(precision, status=status)
    if smoke_row:
        for key in (
            "quant_scheme",
            "quant_method",
            "quant_scope",
            "calibration_source",
            "calibration_digest",
            "calibration_manifest_digest",
            "quant_recipe_digest",
            "quantized_onnx_digest",
            "calibrator",
            "fallback_policy",
            "layer_precision_summary",
            "layer_precision_summary_digest",
            "engine_kind",
            "engine_digest",
            "measurement_source",
            "claim_status",
            "quality_gate_status",
            "schedule_profile",
            "tune_budget",
        ):
            value = smoke_row.get(key)
            if value not in (None, ""):
                contract[key] = value
        contract["full_network_claim"] = False
        if smoke_row.get("calibration_inputs") is not None:
            contract["calibration_inputs"] = smoke_row.get("calibration_inputs") or []
        contract["measurement_source"] = "true_measurement_smoke"
        contract["claim_status"] = "claimable_true_measurement_smoke"
    schedule_policy = str(
        (smoke_row or {}).get("schedule_policy")
        or (smoke_row or {}).get("schedule_profile")
        or ("metaschedule_tuned" if precision != "fp32" else "default")
    )
    return latency_lut_row(
        config_id=config_id(label, precision, "latency", schedule_policy),
        model=MODEL,
        manifest_digest=MANIFEST_DIGEST,
        candidate_id=label,
        software_point_id=f"{label}:{width_text(width)}:{precision}:latency",
        dense_stage="backbone",
        optimized_scope=OPTIMIZED_SCOPE,
        width=width,
        quant_policy=precision,
        schedule_policy=schedule_policy,
        backend="h800_tvm",
        measurement_status=status,
        latency_p50_us=None if latency_ms is None else latency_ms * 1000.0,
        warmup_iters=int((smoke_row or {}).get("warmup_iters") or 50),
        measure_iters=int((smoke_row or {}).get("measure_iters") or 200),
        repeat=int((smoke_row or {}).get("repeat") or 5),
        batch_size=1,
        tvm_target=str((smoke_row or {}).get("tvm_target") or "cuda"),
        tvm_strategy=str((smoke_row or {}).get("tvm_strategy") or "relax_metaschedule"),
        build_status="not_run" if status != "measured" else "success",
        provenance="quant anchor smoke state row",
        run_id=str((smoke_row or {}).get("run_id") or f"quant_smoke_20260627:{label}:{precision}:latency"),
        created_at=created_at,
        source_files=source_files,
        raw_artifact=(smoke_row or {}).get("raw_artifact"),
        failure_reason=failure_reason,
        notes=notes or "state row only; no new H800 measurement performed",
        **contract,
    )


def make_energy_row(
    *,
    anchor: dict[str, Any],
    precision: str,
    original60: dict[str, dict[str, str]],
    fp16_energy_rows: dict[str, dict[str, Any]],
    int8_energy_rows: dict[str, dict[str, Any]],
    created_at: str,
) -> dict[str, Any]:
    label = str(anchor["label"])
    width = list(anchor["width"])
    source_files: list[str] = []
    energy_j: float | None = None
    failure_reason: str | None = None
    notes = "state row only; no new H800 energy telemetry performed"
    smoke_row = None
    if precision == "fp16":
        smoke_row = fp16_energy_rows.get(label)
    elif precision == "int8":
        smoke_row = int8_energy_rows.get(label)

    if smoke_row:
        energy_j = float(smoke_row["joule_per_inference"])
        source_files = list(smoke_row.get("source_files") or [])
        notes = (
            "imported from true-FP16 H800 power telemetry smoke"
            if precision == "fp16"
            else "imported from INT8 QDQ TVM VM H800 power telemetry smoke"
        )
    elif precision == "fp16":
        failure_reason = "fp16_energy_requires_true_fp16_h800_power_telemetry"
    elif precision == "fp32":
        failure_reason = "energy_fp32_latency_measurement_not_run"
    elif precision == "int8":
        failure_reason = "tvm_int8_energy_backend_missing"

    status = "measured" if energy_j is not None and failure_reason is None else "no_claim"
    contract = quant_contract(precision, status=status)
    if smoke_row:
        for key in (
            "quant_method",
            "quant_scope",
            "quant_scheme",
            "calibration_source",
            "calibration_digest",
            "calibration_manifest_digest",
            "quant_recipe_digest",
            "quantized_onnx_digest",
            "calibrator",
            "fallback_policy",
            "layer_precision_summary",
            "layer_precision_summary_digest",
            "engine_kind",
            "engine_digest",
            "measurement_source",
            "claim_status",
            "quality_gate_status",
            "schedule_profile",
            "tune_budget",
        ):
            value = smoke_row.get(key)
            if value not in (None, ""):
                contract[key] = value
        contract["full_network_claim"] = False
        if smoke_row.get("calibration_inputs") is not None:
            contract["calibration_inputs"] = smoke_row.get("calibration_inputs") or []
        contract["measurement_source"] = "true_measurement_smoke"
        contract["claim_status"] = "claimable_true_measurement_smoke"
    schedule_policy = str(
        (smoke_row or {}).get("schedule_policy")
        or ("metaschedule_tuned" if precision != "fp32" else "default")
    )
    return energy_lut_row(
        config_id=config_id(label, precision, "energy"),
        model=MODEL,
        manifest_digest=MANIFEST_DIGEST,
        candidate_id=label,
        software_point_id=f"{label}:{width_text(width)}:{precision}:energy",
        dense_stage="backbone",
        optimized_scope=OPTIMIZED_SCOPE,
        width=width,
        quant_policy=precision,
        schedule_policy=schedule_policy,
        backend="h800_tvm_power_telemetry",
        measurement_status=status,
        joule_per_inference=energy_j,
        watt_avg=(smoke_row or {}).get("watt_avg"),
        watt_p50=(smoke_row or {}).get("watt_p50"),
        watt_p90=(smoke_row or {}).get("watt_p90"),
        idle_watt_avg=(smoke_row or {}).get("idle_watt_avg"),
        telemetry_source=(smoke_row or {}).get("telemetry_source") if status == "measured" else None,
        idle_baseline_policy=(smoke_row or {}).get("idle_baseline_policy") if status == "measured" else None,
        sample_window_ms=(smoke_row or {}).get("sample_window_ms"),
        latency_run_id=(smoke_row or {}).get("latency_run_id") or f"quant_smoke_20260627:{label}:{precision}:latency",
        measurement_run_id=(smoke_row or {}).get("measurement_run_id") or f"quant_smoke_20260627:{label}:{precision}:energy",
        row_source="quant_anchor_smoke_import_or_no_claim",
        provenance="quant anchor smoke state row",
        run_id=str((smoke_row or {}).get("run_id") or f"quant_smoke_20260627:{label}:{precision}:energy"),
        created_at=created_at,
        source_files=source_files,
        raw_artifact=(smoke_row or {}).get("raw_artifact"),
        failure_reason=failure_reason,
        notes=notes,
        **contract,
    )


def make_ap_row(
    *,
    anchor: dict[str, Any],
    precision: str,
    ap_index: dict[str, dict[str, str]],
    created_at: str,
) -> dict[str, Any]:
    label = str(anchor["label"])
    width = list(anchor["width"])
    source_files: list[str] = []
    ap70: float | None = None
    ap30: float | None = None
    ap50: float | None = None
    source_kind = ""
    ckpt_digest = "unknown"
    raw_artifact = None
    failure_reason: str | None = None

    if precision == "fp16":
        failure_reason = "fp16_ap_requires_true_fp16_eval_source_revalidation"
    elif precision == "fp32":
        failure_reason = "ap_fp32_true_eval_not_available"
    elif precision == "int8":
        failure_reason = "tvm_int8_ap_eval_backend_missing"

    status = "measured" if ap70 is not None and failure_reason is None else "no_claim"
    return ap_anchor_row(
        config_id=config_id(label, precision, "ap"),
        model=MODEL,
        manifest_digest=MANIFEST_DIGEST,
        candidate_id=label,
        software_point_id=f"{label}:{width_text(width)}:{precision}:ap",
        dense_stage="model",
        optimized_scope=OPTIMIZED_SCOPE,
        width=width,
        quant_policy=precision,
        schedule_policy="not_applicable",
        backend="model_eval",
        measurement_status=status,
        metric="AP70",
        metric_value=ap70,
        secondary_metrics={"AP30": ap30, "AP50": ap50} if status == "measured" else {},
        dataset="DAIR-V2X",
        eval_split="val",
        ckpt_path=raw_artifact or "unknown",
        ckpt_digest=ckpt_digest,
        finetune_protocol="true_import_or_phase_c_eval" if status == "measured" else "none",
        training_budget="existing_source" if status == "measured" else "not_run",
        eval_command="imported_ap_summary" if status == "measured" else "",
        provenance="quant anchor smoke AP state row",
        run_id=f"quant_smoke_20260627:{label}:{precision}:ap",
        created_at=created_at,
        source_files=source_files,
        raw_artifact=raw_artifact,
        failure_reason=failure_reason,
        notes=(
            "true AP source imported; AP is not predicted"
            if status == "measured"
            else "state row only; no AP eval backend claim"
        ),
        **ap_quant_contract(precision, status=status, source_kind=source_kind),
    )


def make_job_plan(output_root: Path, created_at: str) -> dict[str, Any]:
    jobs: list[dict[str, Any]] = []
    for anchor in ANCHORS:
        label = str(anchor["label"])
        for precision in PRECISIONS:
            for axis in ("latency", "energy", "ap"):
                jobs.append(
                    {
                        "job_id": f"quant_anchor:{axis}:{label}:{precision}",
                        "label": label,
                        "width": anchor["width"],
                        "precision": precision,
                        "lut_kind": axis,
                        "backend": "h800_tvm" if axis == "latency" else (
                            "h800_tvm_power_telemetry" if axis == "energy" else "model_eval"
                        ),
                        "state_policy": "measured_if_existing_source_else_no_claim",
                    }
                )
    return {
        "schema": "quant_anchor_job_plan_v1",
        "created_at": created_at,
        "anchors": ANCHORS,
        "precisions": list(PRECISIONS),
        "output_root": str(output_root),
        "jobs": jobs,
    }


def make_job_states(rows: list[dict[str, Any]], axis: str) -> list[dict[str, Any]]:
    states = []
    for row in rows:
        status = "succeeded" if row.get("measurement_status") == "measured" else "skipped"
        states.append(
            {
                "schema": "lut_job_state_row_v1",
                "job_id": f"quant_anchor:{axis}:{row['candidate_id']}:{row['precision']}",
                "status": status,
                "attempt": 1,
                "config_id": row["config_id"],
                "failure_reason": row.get("failure_reason"),
                "created_at": row["created_at"],
            }
        )
    return states


def make_quarantine_rows(
    latency_rows: list[dict[str, Any]],
    energy_rows: list[dict[str, Any]],
    ap_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for axis, rows in (
        ("latency", latency_rows),
        ("energy", energy_rows),
        ("ap", ap_rows),
    ):
        for row in rows:
            reason = row.get("failure_reason")
            if not reason:
                continue
            out.append(
                quarantine_row(
                    job_id=f"quant_anchor:{axis}:{row['candidate_id']}:{row['precision']}",
                    config_id=str(row["config_id"]),
                    model=str(row["model"]),
                    lut_kind=axis,
                    job_type=f"generate_{axis}_lut",
                    failure_reason=str(reason),
                    status="active",
                    created_at=str(row["created_at"]),
                )
            )
    return out


def make_summary_rows(
    latency_rows: list[dict[str, Any]],
    energy_rows: list[dict[str, Any]],
    ap_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    latency = {(row["candidate_id"], row["precision"]): row for row in latency_rows}
    energy = {(row["candidate_id"], row["precision"]): row for row in energy_rows}
    ap = {(row["candidate_id"], row["precision"]): row for row in ap_rows}
    out: list[dict[str, Any]] = []
    for anchor in ANCHORS:
        label = str(anchor["label"])
        for precision in PRECISIONS:
            key = (label, precision)
            latency_row = latency[key]
            energy_row = energy[key]
            ap_row = ap[key]
            failures = [
                str(row.get("failure_reason"))
                for row in (latency_row, energy_row, ap_row)
                if row.get("failure_reason")
            ]
            out.append(
                {
                    "label": label,
                    "width": width_text(list(anchor["width"])),
                    "precision": precision,
                    "quant_method": latency_row.get("quant_method"),
                    "quant_scope": latency_row.get("quant_scope"),
                    "full_network_claim": False,
                    "latency_ms": (
                        None
                        if latency_row.get("latency_p50_us") is None
                        else round(float(latency_row["latency_p50_us"]) / 1000.0, 6)
                    ),
                    "latency_status": latency_row.get("measurement_status"),
                    "latency_measurement_source": latency_row.get("measurement_source"),
                    "latency_claim_status": latency_row.get("claim_status"),
                    "latency_evidence_scope": evidence_scope(latency_row),
                    "latency_repeat": latency_row.get("repeat"),
                    "energy_j_per_inference": energy_row.get("joule_per_inference"),
                    "energy_status": energy_row.get("measurement_status"),
                    "energy_measurement_source": energy_row.get("measurement_source"),
                    "energy_claim_status": energy_row.get("claim_status"),
                    "energy_evidence_scope": evidence_scope(energy_row),
                    "energy_repeat": energy_row.get("repeat"),
                    "ap70": ap_row.get("metric_value"),
                    "ap_status": ap_row.get("measurement_status"),
                    "ap_source_kind": ap_row.get("measurement_source"),
                    "ap_measurement_source": ap_row.get("measurement_source"),
                    "ap_claim_status": ap_row.get("claim_status"),
                    "ap_evidence_scope": evidence_scope(ap_row),
                    "quality_gate_status": ";".join(
                        item
                        for item in (
                            str(latency_row.get("quality_gate_status") or ""),
                            str(energy_row.get("quality_gate_status") or ""),
                            str(ap_row.get("quality_gate_status") or ""),
                        )
                        if item
                    ),
                    "failure_reasons": ";".join(failures),
                }
            )
    return out


def write_summary(output_root: Path, summary_rows: list[dict[str, Any]]) -> None:
    exports = output_root / "exports"
    exports.mkdir(parents=True, exist_ok=True)
    csv_path = exports / "quant_three_metric_summary_latest.csv"
    fields = [
        "label",
        "width",
        "precision",
        "quant_method",
        "quant_scope",
        "full_network_claim",
        "latency_ms",
        "latency_status",
        "energy_j_per_inference",
        "energy_status",
        "ap70",
        "ap_status",
        "ap_source_kind",
        "quality_gate_status",
        "failure_reasons",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary_rows)

    precision_counts = Counter(str(row["precision"]) for row in summary_rows)
    latency_status_counts = Counter(str(row["latency_status"]) for row in summary_rows)
    energy_status_counts = Counter(str(row["energy_status"]) for row in summary_rows)
    ap_status_counts = Counter(str(row["ap_status"]) for row in summary_rows)
    payload = {
        "schema": "quant_three_metric_summary_v1",
        "total_cells": len(summary_rows),
        "precision_counts": dict(sorted(precision_counts.items())),
        "latency_status_counts": dict(sorted(latency_status_counts.items())),
        "energy_status_counts": dict(sorted(energy_status_counts.items())),
        "ap_status_counts": dict(sorted(ap_status_counts.items())),
        "rows": summary_rows,
    }
    (exports / "quant_three_metric_summary_latest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    md_lines = [
        "# Stage2 Quant Anchor 三指标状态表",
        "",
        "说明: 本表是状态覆盖表, 不是全部 measured claim。FP32/INT8 当前不可测项写 no-claim/gap。",
        "",
        "| label | width | precision | latency ms | latency | energy J | energy | AP70 | AP | failure reasons |",
        "|---|---|---|---:|---|---:|---|---:|---|---|",
    ]
    for row in summary_rows:
        md_lines.append(
            "| {label} | {width} | {precision} | {latency_ms} | {latency_status} | "
            "{energy_j_per_inference} | {energy_status} | {ap70} | {ap_status} | "
            "{failure_reasons} |".format(
                label=row["label"],
                width=row["width"],
                precision=row["precision"],
                latency_ms="" if row["latency_ms"] is None else f"{row['latency_ms']:.6f}",
                latency_status=row["latency_status"],
                energy_j_per_inference=(
                    ""
                    if row["energy_j_per_inference"] is None
                    else f"{float(row['energy_j_per_inference']):.6f}"
                ),
                energy_status=row["energy_status"],
                ap70="" if row["ap70"] is None else f"{float(row['ap70']):.9f}",
                ap_status=row["ap_status"],
                failure_reasons=row["failure_reasons"],
            )
        )
    (exports / "quant_three_metric_summary_latest.md").write_text(
        "\n".join(md_lines) + "\n",
        encoding="utf-8",
    )


def write_gap_report(
    output_root: Path,
    latency_rows: list[dict[str, Any]],
    energy_rows: list[dict[str, Any]],
    ap_rows: list[dict[str, Any]],
) -> None:
    reason_counts: Counter[str] = Counter()
    axis_counts: dict[str, Counter[str]] = defaultdict(Counter)
    rows: list[dict[str, Any]] = []
    for axis, axis_rows in (
        ("latency", latency_rows),
        ("energy", energy_rows),
        ("ap", ap_rows),
    ):
        for row in axis_rows:
            reason = row.get("failure_reason")
            if not reason:
                continue
            reason_text = str(reason)
            reason_counts[reason_text] += 1
            axis_counts[axis][reason_text] += 1
            rows.append(
                {
                    "axis": axis,
                    "label": row.get("candidate_id"),
                    "precision": row.get("precision"),
                    "failure_reason": reason_text,
                    "measurement_status": row.get("measurement_status"),
                    "quant_method": row.get("quant_method"),
                    "quant_scope": row.get("quant_scope"),
                }
            )
    payload = {
        "schema": "quant_gap_report_v1",
        "failure_reason_counts": dict(sorted(reason_counts.items())),
        "axis_failure_reason_counts": {
            axis: dict(sorted(counter.items())) for axis, counter in sorted(axis_counts.items())
        },
        "rows": rows,
    }
    exports = output_root / "exports"
    exports.mkdir(parents=True, exist_ok=True)
    (exports / "quant_gap_report_latest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    created_at = args.created_at or utc_timestamp()

    historical_latency = index_historical_latency(read_csv(args.historical_latency_csv))
    original60 = index_original60(read_csv(args.original60_summary_csv))
    ap_summary = index_ap(read_csv(args.ap_summary_csv))
    fp32_smoke_rows = measured_fp32_smoke_index(args.fp32_latency_smoke_rows)
    fp16_smoke_rows = measured_fp16_smoke_index(args.fp16_latency_smoke_rows)
    fp16_energy_rows = measured_energy_index(args.fp16_energy_smoke_rows, "fp16")
    int8_energy_rows = measured_energy_index(args.int8_energy_smoke_rows, "int8")
    int8_smoke_rows = measured_int8_smoke_index(args.int8_latency_smoke_rows)

    latency_rows: list[dict[str, Any]] = []
    energy_rows: list[dict[str, Any]] = []
    ap_rows: list[dict[str, Any]] = []
    for anchor in ANCHORS:
        for precision in PRECISIONS:
            latency_rows.append(
                make_latency_row(
                    anchor=anchor,
                    precision=precision,
                    historical_latency=historical_latency,
                    original60=original60,
                    fp32_smoke_rows=fp32_smoke_rows,
                    fp16_smoke_rows=fp16_smoke_rows,
                    int8_smoke_rows=int8_smoke_rows,
                    created_at=created_at,
                )
            )
            energy_rows.append(
                make_energy_row(
                    anchor=anchor,
                    precision=precision,
                    original60=original60,
                    fp16_energy_rows=fp16_energy_rows,
                    int8_energy_rows=int8_energy_rows,
                    created_at=created_at,
                )
            )
            ap_rows.append(
                make_ap_row(
                    anchor=anchor,
                    precision=precision,
                    ap_index=ap_summary,
                    created_at=created_at,
                )
            )

    (output_root / "plans").mkdir(parents=True, exist_ok=True)
    (output_root / "plans/quant_anchor_job_plan_v1.json").write_text(
        json.dumps(make_job_plan(output_root, created_at), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    write_jsonl(output_root / "rows/latency_quant_rows_v1.jsonl", latency_rows)
    write_jsonl(output_root / "rows/energy_quant_rows_v1.jsonl", energy_rows)
    write_jsonl(output_root / "rows/ap_quant_rows_v1.jsonl", ap_rows)
    write_jsonl(output_root / "jobs/latency_quant_job_state_v1.jsonl", make_job_states(latency_rows, "latency"))
    write_jsonl(output_root / "jobs/energy_quant_job_state_v1.jsonl", make_job_states(energy_rows, "energy"))
    write_jsonl(output_root / "jobs/ap_quant_job_state_v1.jsonl", make_job_states(ap_rows, "ap"))
    write_jsonl(
        output_root / "quarantine/quant_unclaimable_v1.jsonl",
        make_quarantine_rows(latency_rows, energy_rows, ap_rows),
    )
    summary_rows = make_summary_rows(latency_rows, energy_rows, ap_rows)
    write_summary(output_root, summary_rows)
    write_gap_report(output_root, latency_rows, energy_rows, ap_rows)

    print(
        json.dumps(
            {
                "schema": "quant_anchor_smoke_generation_result_v1",
                "output_root": str(output_root),
                "latency_rows": len(latency_rows),
                "energy_rows": len(energy_rows),
                "ap_rows": len(ap_rows),
                "summary_rows": len(summary_rows),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
