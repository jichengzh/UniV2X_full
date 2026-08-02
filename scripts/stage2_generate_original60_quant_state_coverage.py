#!/usr/bin/env python3
"""Generate original60 x precision Stage2 quant state coverage rows."""

from __future__ import annotations

import argparse
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
    latency_lut_row,
    quarantine_row,
    stable_config_id,
    utc_timestamp,
    write_jsonl,
)
from scripts.stage2_generate_quant_anchor_smoke import (  # noqa: E402
    float_or_none,
    quant_contract,
    width_text,
)


DEFAULT_OUTPUT_ROOT = (
    ROOT / "multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627"
)
DEFAULT_ORIGINAL60_SUMMARY = (
    ROOT / "multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1/"
    "exports/original60_three_metric_summary_latest.csv"
)
MODEL = "pyramid_lidar"
MANIFEST_DIGEST = "original60_quant_20260627_state_coverage"
PRECISIONS = ("fp32", "fp16", "int8")
OPTIMIZED_SCOPE = "backbone_only"
NATIVE_INT8_QUANT_METHOD = "h800_tvm_native_int8_backbone_subnet"
NATIVE_INT8_QUANT_SCOPE = "backbone_subnet_native_int8"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--original60-summary-csv", default=str(DEFAULT_ORIGINAL60_SUMMARY))
    parser.add_argument(
        "--fp32-latency-smoke-rows",
        action="append",
        default=[],
        help="Optional JSONL latency rows from the narrow base/s0_024/s1_048 FP32 H800 TVM smoke.",
    )
    parser.add_argument(
        "--fp32-latency-remap-rows",
        action="append",
        default=[],
        help="Optional JSONL latency rows reclassified from suspect original60 FP16-tagged H800 TVM rows.",
    )
    parser.add_argument(
        "--fp16-latency-smoke-rows",
        "--fp16-latency-rows",
        action="append",
        default=[],
        help=(
            "Optional JSONL latency rows from FP16 H800 TVM runs; defaults prefer "
            "rewritten TensorCore full60 rows before historical true-FP16 smoke rows."
        ),
    )
    parser.add_argument(
        "--fp16-energy-smoke-rows",
        "--fp16-energy-rows",
        action="append",
        default=[],
        help=(
            "Optional JSONL energy rows from FP16 H800 power telemetry runs; defaults prefer "
            "rewritten TensorCore full60 rows before historical true-FP16 smoke rows."
        ),
    )
    parser.add_argument(
        "--fp32-energy-remeasure-rows",
        action="append",
        default=[],
        help="Optional JSONL energy rows from direct FP32 original60 H800 power telemetry remeasurement runs.",
    )
    parser.add_argument(
        "--int8-latency-smoke-rows",
        action="append",
        default=[],
        help="Optional JSONL latency rows from the INT8 QDQ TVM VM H800 smoke.",
    )
    parser.add_argument(
        "--int8-energy-smoke-rows",
        action="append",
        default=[],
        help="Optional JSONL energy rows from the INT8 QDQ TVM VM H800 power telemetry smoke.",
    )
    parser.add_argument(
        "--native-int8-full-onnx-latency-rows",
        action="append",
        default=[],
        help="Optional JSONL latency rows from native INT8 full-ONNX topology H800 TVM runs.",
    )
    parser.add_argument(
        "--native-int8-full-onnx-energy-rows",
        action="append",
        default=[],
        help="Optional JSONL energy rows from native INT8 full-ONNX topology H800 telemetry runs.",
    )
    parser.add_argument(
        "--fp16-ap-rows",
        action="append",
        default=[],
        help="Optional JSONL AP rows from compliant true-FP16 original60 eval/import runs.",
    )
    parser.add_argument(
        "--fp32-ap-rows",
        action="append",
        default=[],
        help="Optional JSONL AP rows from compliant true-FP32 original60 eval/import runs.",
    )
    parser.add_argument(
        "--int8-ap-rows",
        action="append",
        default=[],
        help="Optional JSONL AP rows from compliant native INT8 original60 eval/import runs.",
    )
    parser.add_argument("--created-at")
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


def default_existing_row_paths(output_root: Path, paths: list[str], *filenames: str) -> list[str]:
    if paths:
        return paths
    rows_dir = output_root / "rows"
    return [str(rows_dir / filename) for filename in filenames if (rows_dir / filename).exists()]


def measured_fp32_smoke_index(paths: list[str]) -> dict[str, dict[str, Any]]:
    rows = []
    for path in paths:
        rows.extend(read_jsonl_rows(path))

    measured = []
    for row in rows:
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

    def score(row: dict[str, Any]) -> tuple[int, int, str]:
        quality = str(row.get("quality_gate_status") or "")
        measurement_source = str(row.get("measurement_source") or "")
        if quality == "fp32_latency_smoke_only":
            source_score = 3
        elif measurement_source == "true_measurement":
            source_score = 2
        elif "reclassified" in measurement_source or "reclassified" in quality:
            source_score = 1
        else:
            source_score = 0
        schedule = str(row.get("schedule_policy") or row.get("schedule_profile") or "")
        tuned_score = 1 if schedule == "metaschedule_tuned" else 0
        return source_score, tuned_score, str(row.get("created_at") or row.get("run_id") or "")

    out: dict[str, dict[str, Any]] = {}
    for row in measured:
        label = smoke_label(row)
        if not label:
            continue
        if label not in out or score(row) > score(out[label]):
            out[label] = row
    return out


def measured_int8_smoke_index(paths: list[str]) -> dict[str, dict[str, Any]]:
    rows = []
    for path in paths:
        rows.extend(read_jsonl_rows(path))

    measured = []
    for row in rows:
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


def is_native_int8_full_onnx_row(row: dict[str, Any]) -> bool:
    return (
        str(row.get("precision")) == "int8"
        and str(row.get("measurement_status")) == "measured"
        and str(row.get("quant_method") or "") == NATIVE_INT8_QUANT_METHOD
        and str(row.get("quant_scope") or "") == NATIVE_INT8_QUANT_SCOPE
        and str(row.get("engine_kind") or "") == "tvm_graph_executor"
        and not bool(row.get("full_network_claim"))
    )


def measured_native_int8_full_onnx_latency_index(paths: list[str]) -> dict[str, dict[str, Any]]:
    rows = []
    for path in paths:
        rows.extend(read_jsonl_rows(path))

    measured = []
    for row in rows:
        if not is_native_int8_full_onnx_row(row):
            continue
        if not str(row.get("backend", "")).startswith("h800_tvm"):
            continue
        if row.get("latency_p50_us") is None:
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


def measured_native_int8_full_onnx_energy_index(paths: list[str]) -> dict[str, dict[str, Any]]:
    rows = []
    for path in paths:
        rows.extend(read_jsonl_rows(path))

    measured = []
    for row in rows:
        if not is_native_int8_full_onnx_row(row):
            continue
        if not str(row.get("backend", "")).startswith("h800_tvm_power_telemetry"):
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


def measured_fp16_smoke_index(paths: list[str]) -> dict[str, dict[str, Any]]:
    rows = []
    for path in paths:
        rows.extend(read_jsonl_rows(path))

    measured = []
    for row in rows:
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
        quality = str(row.get("quality_gate_status") or "")
        source_files = " ".join(str(item) for item in row.get("source_files") or [])
        layer_summary = str(row.get("layer_precision_summary") or "")
        fp16_marker_text = " ".join([quality, source_files, layer_summary]).lower()
        if (
            "true_fp16" not in fp16_marker_text
            and "float16" not in fp16_marker_text
            and "fp16_rewritten_tensorcore" not in fp16_marker_text
        ):
            continue
        measured.append(row)

    def score(row: dict[str, Any]) -> tuple[int, int, str]:
        schedule = str(row.get("schedule_policy") or row.get("schedule_profile") or "")
        marker_text = " ".join(
            str(row.get(key) or "")
            for key in (
                "quality_gate_status",
                "quant_method",
                "schedule_policy",
                "schedule_profile",
                "tvm_strategy",
                "source_files",
            )
        ).lower()
        rewritten_score = 1 if "fp16_rewritten_tensorcore" in marker_text else 0
        tuned_score = 1 if schedule == "metaschedule_tuned" else 0
        return rewritten_score, tuned_score, str(row.get("created_at") or row.get("run_id") or "")

    out: dict[str, dict[str, Any]] = {}
    for row in measured:
        label = smoke_label(row)
        if not label:
            continue
        if label not in out or score(row) > score(out[label]):
            out[label] = row
    return out


def measured_energy_index(paths: list[str], precision: str) -> dict[str, dict[str, Any]]:
    rows = []
    for path in paths:
        rows.extend(read_jsonl_rows(path))

    measured = []
    for row in rows:
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
        has_legacy_power_files = (
            "idle_power_samples.csv" in source_files
            and "active_power_samples.csv" in source_files
        )
        has_threaded_window_evidence = (
            str(row.get("energy_sampling_mode") or "") == "threaded_window"
            and "nvidia-smi" in str(row.get("telemetry_source") or "")
            and str(row.get("idle_baseline_policy") or "") == "subtract_idle_avg_5s_pre_window"
            and row.get("idle_watt_avg") is not None
            and row.get("watt_avg") is not None
            and row.get("sample_window_ms") is not None
        )
        if not has_legacy_power_files and not has_threaded_window_evidence:
            continue
        measured.append(row)

    def score(row: dict[str, Any]) -> tuple[int, str, str]:
        marker_text = " ".join(
            str(row.get(key) or "")
            for key in (
                "quality_gate_status",
                "quant_method",
                "energy_sampling_mode",
                "schedule_policy",
                "schedule_profile",
                "source_files",
            )
        ).lower()
        rewritten_score = 1 if "fp16_rewritten_tensorcore" in marker_text else 0
        return rewritten_score, str(row.get("created_at") or ""), str(row.get("run_id") or "")

    out: dict[str, dict[str, Any]] = {}
    for row in measured:
        label = smoke_label(row)
        if not label:
            continue
        if label not in out or score(row) > score(out[label]):
            out[label] = row
    return out


def _has_real_ap_eval_evidence(row: dict[str, Any]) -> bool:
    secondary = row.get("secondary_metrics")
    source_files = row.get("source_files") or []
    required_text_fields = (
        "dataset",
        "eval_split",
        "ckpt_path",
        "ckpt_digest",
        "eval_command",
        "raw_artifact",
    )
    if any(str(row.get(field) or "").strip() in ("", "unknown") for field in required_text_fields):
        return False
    if not isinstance(source_files, list) or not source_files:
        return False
    if not isinstance(secondary, dict) or "AP30" not in secondary or "AP50" not in secondary:
        return False
    return True


def _has_banned_ap_source(row: dict[str, Any]) -> bool:
    text = " ".join(
        str(row.get(key) or "").lower()
        for key in (
            "backend",
            "engine_kind",
            "measurement_source",
            "claim_status",
            "quality_gate_status",
            "quant_method",
            "provenance",
            "notes",
            "eval_command",
            "raw_artifact",
        )
    )
    banned = ("predicted", "model_fit", "model-fit", "interpolated", "trt", "simulated")
    return any(token in text for token in banned)


def is_compliant_original60_ap_row(
    row: dict[str, Any],
    precision: str,
    canonical_widths: dict[str, list[int]] | None = None,
) -> bool:
    label = smoke_label(row)
    if canonical_widths and label in canonical_widths:
        if normalize_width(row.get("width")) != canonical_widths[label]:
            return False
    row_precision = str(row.get("precision") or row.get("quant_policy") or "")
    if row_precision != precision:
        return False
    if str(row.get("measurement_status") or "") != "measured":
        return False
    if str(row.get("backend") or "") != "model_eval":
        return False
    if bool(row.get("full_network_claim")):
        return False
    if str(row.get("metric") or "") != "AP70":
        return False
    if row.get("metric_value") is None:
        return False
    if str(row.get("measurement_source") or "") != "true_eval":
        return False
    if _has_banned_ap_source(row):
        return False
    if not _has_real_ap_eval_evidence(row):
        return False

    if precision == "fp16":
        if str(row.get("quant_method") or "") != "h800_tvm_true_fp16_onnx_relax":
            return False
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
        return "true_fp16" in marker_text or "float16" in marker_text

    if precision == "fp32":
        marker_text = " ".join(
            str(row.get(key) or "")
            for key in (
                "quality_gate_status",
                "layer_precision_summary",
                "source_files",
                "notes",
                "eval_command",
                "provenance",
            )
        ).lower()
        return (
            str(row.get("quant_method") or "") == "h800_tvm_true_fp32_model_eval"
            and str(row.get("quant_scope") or "") == "full_model_ap_eval_true_fp32"
            and ("true_fp32" in marker_text or "float32" in marker_text)
        )

    if precision == "int8":
        return (
            str(row.get("quant_method") or "") == NATIVE_INT8_QUANT_METHOD
            and str(row.get("quant_scope") or "") == NATIVE_INT8_QUANT_SCOPE
        )

    return False


def measured_ap_index(
    paths: list[str],
    precision: str,
    canonical_widths: dict[str, list[int]] | None = None,
) -> dict[str, dict[str, Any]]:
    rows = []
    for path in paths:
        rows.extend(read_jsonl_rows(path))

    measured = [
        row
        for row in rows
        if is_compliant_original60_ap_row(row, precision, canonical_widths)
    ]

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


def parse_width_text(value: str) -> list[int]:
    return [int(item) for item in str(value).split("x") if item]


def normalize_width(value: Any) -> list[int]:
    if isinstance(value, list):
        return [int(item) for item in value]
    if isinstance(value, tuple):
        return [int(item) for item in value]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if "x" in text:
            return parse_width_text(text)
        return [int(item) for item in text.split(",") if item.strip()]
    return []


def label_from_summary(row: dict[str, str]) -> str:
    return str(row.get("label") or row.get("candidate_id") or "unknown")


def original_candidate_id(row: dict[str, str]) -> str:
    return str(row.get("candidate_id") or label_from_summary(row))


def precision_candidate_id(row: dict[str, str], precision: str) -> str:
    original = original_candidate_id(row)
    parts = original.split(":")
    if parts and parts[-1] in PRECISIONS:
        return ":".join([*parts[:-1], precision])
    if original == label_from_summary(row):
        return f"{original}:{precision}"
    return f"{original}:q_{precision}"


def config_id(label: str, precision: str, axis: str, schedule_policy: str | None = None) -> str:
    return stable_config_id(
        model=MODEL,
        candidate_id=label,
        software_point_id=f"original60:{label}:{precision}:{OPTIMIZED_SCOPE}:{axis}",
        quant_policy=precision,
        schedule_policy=schedule_policy or ("metaschedule_tuned" if precision != "fp32" else "default"),
    )


def source_file() -> str:
    return str(DEFAULT_ORIGINAL60_SUMMARY.relative_to(ROOT))


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


def latency_status(row: dict[str, str], precision: str) -> tuple[str, float | None, str | None]:
    if precision == "fp16":
        return "no_claim", None, "historical_fp16_tagged_latency_not_true_fp16_evidence"
    if precision == "fp32":
        return "no_claim", None, "tvm_fp32_backbone_measurement_not_run"
    return "no_claim", None, "tvm_int8_backbone_subnet_not_ready"


def energy_status(row: dict[str, str], precision: str) -> tuple[str, float | None, str | None]:
    if precision == "fp16":
        return "no_claim", None, "fp16_energy_requires_true_fp16_h800_power_telemetry"
    if precision == "fp32":
        energy_j = float_or_none(row.get("energy_j_per_inference"))
        if str(row.get("energy_status") or "") == "measured" and energy_j is not None:
            return "measured", energy_j, None
        return "no_claim", None, "fp32_energy_historical_remap_not_available"
    return "no_claim", None, "tvm_int8_energy_backend_missing"


def ap_status(row: dict[str, str], precision: str) -> tuple[str, float | None, str, str | None]:
    if precision == "fp16":
        return "no_claim", None, "no_claim", "fp16_ap_requires_true_fp16_eval_source_revalidation"
    if precision == "fp32":
        ap70 = float_or_none(row.get("ap70"))
        claim_status = str(row.get("ap_status") or "")
        if ap70 is not None and not claim_status.startswith("no_claim"):
            return "measured", ap70, str(row.get("ap_source_kind") or "reference"), None
        return "no_claim", None, "no_claim", "ap_fp32_true_eval_not_available"
    return "no_claim", None, "no_claim", "tvm_int8_ap_eval_backend_missing"


def make_latency_row(
    summary_row: dict[str, str],
    precision: str,
    created_at: str,
    fp32_smoke_rows: dict[str, dict[str, Any]] | None = None,
    fp16_smoke_rows: dict[str, dict[str, Any]] | None = None,
    int8_smoke_rows: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    label = label_from_summary(summary_row)
    width = parse_width_text(str(summary_row["width"]))
    status, latency_ms, failure_reason = latency_status(summary_row, precision)
    smoke_row = None
    if precision == "fp32":
        smoke_row = (fp32_smoke_rows or {}).get(label)
    elif precision == "fp16":
        smoke_row = (fp16_smoke_rows or {}).get(label)
    elif precision == "int8":
        smoke_row = (int8_smoke_rows or {}).get(label)
    if smoke_row:
        status = "measured"
        latency_ms = float(smoke_row["latency_p50_us"]) / 1000.0
        failure_reason = None
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
        if precision == "fp16":
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
        candidate_id=precision_candidate_id(summary_row, precision),
        software_point_id=f"original60:{label}:{width_text(width)}:{precision}:latency",
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
        batch_size=int((smoke_row or {}).get("batch_size") or 1),
        tvm_target=str((smoke_row or {}).get("tvm_target") or "cuda"),
        tvm_strategy=str(
            (smoke_row or {}).get("tvm_strategy")
            or ("relax_metaschedule" if precision != "fp32" else "relax_default")
        ),
        build_status="success" if status == "measured" else "not_run",
        provenance="original60 quant state coverage",
        run_id=str((smoke_row or {}).get("run_id") or f"original60_quant_20260627:{label}:{precision}:latency"),
        created_at=created_at,
        source_files=(smoke_row or {}).get("source_files") or ([source_file()] if status == "measured" else []),
        raw_artifact=(smoke_row or {}).get("raw_artifact"),
        failure_reason=failure_reason,
        notes=(
            (
                (
                    "imported from native INT8 full-ONNX H800 TVM latency run"
                    if str(smoke_row.get("quant_method") or "") == NATIVE_INT8_QUANT_METHOD
                    else "imported from INT8 QDQ TVM VM H800 latency smoke"
                )
                if precision == "int8"
                else (
                    "imported from true-FP16 ONNX H800 TVM latency smoke"
                    if precision == "fp16"
                    else (
                        "imported from FP32 H800 TVM latency smoke"
                        if str(smoke_row.get("measurement_source") or "")
                        != "historical_true_measurement_reclassified"
                        else "imported from FP32 remap audit of suspect FP16-tagged original60 row"
                    )
                )
            )
            if smoke_row
            else "normalized original60 state coverage row; not a new H800 measurement"
        ),
        label=label,
        original60_candidate_id=original_candidate_id(summary_row),
        latency_default_run_id=summary_row.get("latency_default_run_id") or "",
        latency_tuned_run_id=(smoke_row or {}).get("run_id") or summary_row.get("latency_tuned_run_id") or "",
        **contract,
    )


def make_energy_row(
    summary_row: dict[str, str],
    precision: str,
    created_at: str,
    fp32_energy_rows: dict[str, dict[str, Any]] | None = None,
    fp16_energy_rows: dict[str, dict[str, Any]] | None = None,
    int8_energy_rows: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    label = label_from_summary(summary_row)
    width = parse_width_text(str(summary_row["width"]))
    status, energy_j, failure_reason = energy_status(summary_row, precision)
    smoke_row = None
    if precision == "fp32":
        smoke_row = (fp32_energy_rows or {}).get(label)
    elif precision == "fp16":
        smoke_row = (fp16_energy_rows or {}).get(label)
    elif precision == "int8":
        smoke_row = (int8_energy_rows or {}).get(label)
    if smoke_row:
        status = "measured"
        energy_j = float(smoke_row["joule_per_inference"])
        failure_reason = None
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
        if precision == "fp32":
            contract["measurement_source"] = str(
                smoke_row.get("measurement_source") or "true_measurement"
            )
            contract["claim_status"] = str(
                smoke_row.get("claim_status") or "claimable_true_measurement"
            )
            contract["quality_gate_status"] = str(
                smoke_row.get("quality_gate_status") or "fp32_energy_remeasured_h800"
            )
        else:
            contract["measurement_source"] = "true_measurement_smoke"
            contract["claim_status"] = "claimable_true_measurement_smoke"
    elif precision == "fp32" and status == "measured":
        contract["measurement_source"] = "historical_true_measurement_reclassified"
        contract["claim_status"] = "claimable_true_measurement_remapped"
        contract["quality_gate_status"] = "fp32_energy_reclassified_from_historical_plain_backbone_onnx"
    schedule_policy = str(
        (smoke_row or {}).get("schedule_policy")
        or ("metaschedule_tuned" if precision != "fp32" else "default")
    )
    run_id = str(
        (smoke_row or {}).get("run_id")
        or (
            summary_row.get("energy_run_id")
            if precision == "fp32" and status == "measured"
            else None
        )
        or f"original60_quant_20260627:{label}:{precision}:energy"
    )
    return energy_lut_row(
        config_id=config_id(label, precision, "energy", schedule_policy),
        model=MODEL,
        manifest_digest=MANIFEST_DIGEST,
        candidate_id=precision_candidate_id(summary_row, precision),
        software_point_id=f"original60:{label}:{width_text(width)}:{precision}:energy",
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
        telemetry_source=(
            (smoke_row or {}).get("telemetry_source")
            or (
                "nvidia-smi power.draw polling 50ms"
                if precision == "fp32" and status == "measured"
                else None
            )
        )
        if status == "measured"
        else None,
        idle_baseline_policy=(
            (smoke_row or {}).get("idle_baseline_policy")
            or (
                "subtract_idle_avg_5s_pre_window"
                if precision == "fp32" and status == "measured"
                else None
            )
        )
        if status == "measured"
        else None,
        sample_window_ms=(smoke_row or {}).get("sample_window_ms"),
        latency_run_id=(smoke_row or {}).get("latency_run_id") or summary_row.get("latency_tuned_run_id") or "same_config_pending",
        measurement_run_id=(smoke_row or {}).get("measurement_run_id") or run_id,
        row_source="original60_quant_state_coverage",
        provenance="original60 quant state coverage",
        run_id=run_id,
        created_at=created_at,
        source_files=(smoke_row or {}).get("source_files") or ([source_file()] if status == "measured" else []),
        raw_artifact=(smoke_row or {}).get("raw_artifact")
        or (
            f"{source_file()}#{summary_row.get('energy_run_id') or label}"
            if precision == "fp32" and status == "measured"
            else None
        ),
        failure_reason=failure_reason,
        notes=(
            "imported from true-FP16 H800 power telemetry smoke"
            if smoke_row and precision == "fp16"
            else "imported from FP32 original60 H800 power telemetry remeasurement"
            if smoke_row and precision == "fp32"
            else "imported from native INT8 full-ONNX H800 power telemetry run"
            if smoke_row
            and precision == "int8"
            and str(smoke_row.get("quant_method") or "") == NATIVE_INT8_QUANT_METHOD
            else "imported from INT8 QDQ TVM VM H800 power telemetry smoke"
            if smoke_row and precision == "int8"
            else "imported from FP32 remap of historical plain-backbone H800 power telemetry"
            if precision == "fp32" and status == "measured"
            else "no measured claim without true-FP16 H800 power telemetry evidence"
            if precision == "fp16"
            else "normalized original60 state coverage row; not a new H800 energy measurement"
        ),
        label=label,
        original60_candidate_id=original_candidate_id(summary_row),
        energy_run_id=summary_row.get("energy_run_id") or "",
        **contract,
    )


def make_ap_row(
    summary_row: dict[str, str],
    precision: str,
    created_at: str,
    fp16_ap_rows: dict[str, dict[str, Any]] | None = None,
    fp32_ap_rows: dict[str, dict[str, Any]] | None = None,
    int8_ap_rows: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    label = label_from_summary(summary_row)
    width = parse_width_text(str(summary_row["width"]))
    status, ap70, source_kind, failure_reason = ap_status(summary_row, precision)
    measured_row = None
    if precision == "fp16":
        measured_row = (fp16_ap_rows or {}).get(label)
    elif precision == "fp32":
        measured_row = (fp32_ap_rows or {}).get(label)
    elif precision == "int8":
        measured_row = (int8_ap_rows or {}).get(label)
    if measured_row:
        status = "measured"
        ap70 = float(measured_row["metric_value"])
        source_kind = str(measured_row.get("measurement_source") or "true_eval")
        failure_reason = None
    contract = quant_contract(precision, status=status)
    contract["engine_kind"] = "model_eval" if status == "measured" else "tvm_vm"
    if measured_row:
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
            value = measured_row.get(key)
            if value not in (None, ""):
                contract[key] = value
        contract["full_network_claim"] = False
        if measured_row.get("calibration_inputs") is not None:
            contract["calibration_inputs"] = measured_row.get("calibration_inputs") or []
    elif status == "measured":
        contract["measurement_source"] = (
            "stage2_original60_ap_source_map_reference"
            if precision == "fp32"
            else source_kind
        )
        contract["claim_status"] = str(summary_row.get("ap_status") or "claimable_true_eval")
        contract["quality_gate_status"] = str(summary_row.get("ap_quality_gate_status") or "")
    raw = str((measured_row or {}).get("raw_artifact") or summary_row.get("ap_source_path") or "")
    secondary_metrics = (
        dict(measured_row.get("secondary_metrics") or {})
        if measured_row
        else {
            "AP30": float_or_none(summary_row.get("ap30")),
            "AP50": float_or_none(summary_row.get("ap50")),
        }
        if status == "measured"
        else {}
    )
    notes = (
        str((measured_row or {}).get("notes") or "AP imported from compliant true eval source")
        if status == "measured"
        and (measured_row or precision != "fp32")
        else "Reference AP imported from Stage2 original60 AP source map; precision-specific TVM FP32 AP was not separately executed."
        if status == "measured"
        else "AP no-claim pending compliant true eval source"
    )
    return ap_anchor_row(
        config_id=config_id(label, precision, "ap"),
        model=MODEL,
        manifest_digest=MANIFEST_DIGEST,
        candidate_id=precision_candidate_id(summary_row, precision),
        software_point_id=f"original60:{label}:{width_text(width)}:{precision}:ap",
        dense_stage="model",
        optimized_scope=OPTIMIZED_SCOPE,
        width=width,
        quant_policy=precision,
        schedule_policy="not_applicable",
        backend="model_eval",
        measurement_status=status,
        metric=str((measured_row or {}).get("metric") or "AP70"),
        metric_value=ap70,
        secondary_metrics=secondary_metrics,
        dataset=str((measured_row or {}).get("dataset") or "DAIR-V2X"),
        eval_split=str((measured_row or {}).get("eval_split") or "val"),
        ckpt_path=str((measured_row or {}).get("ckpt_path") or raw or "unknown"),
        ckpt_digest=str((measured_row or {}).get("ckpt_digest") or "unknown"),
        finetune_protocol=str((measured_row or {}).get("finetune_protocol") or source_kind if status == "measured" else "none"),
        training_budget=str((measured_row or {}).get("training_budget") or "existing_original60_source" if status == "measured" else "not_run"),
        eval_command=str((measured_row or {}).get("eval_command") or "imported_original60_summary" if status == "measured" else ""),
        provenance=str((measured_row or {}).get("provenance") or "original60 quant state coverage AP row"),
        run_id=str((measured_row or {}).get("run_id") or f"original60_quant_20260627:{label}:{precision}:ap"),
        created_at=created_at,
        source_files=(measured_row or {}).get("source_files") or ([item for item in raw.split(";") if item] if status == "measured" else []),
        raw_artifact=raw or None,
        failure_reason=failure_reason,
        notes=notes,
        label=label,
        original60_candidate_id=original_candidate_id(summary_row),
        ap_source_kind=source_kind,
        ap_quality_gate_status=summary_row.get("ap_quality_gate_status") or "",
        **contract,
    )


def make_plan(summary_rows: list[dict[str, str]], output_root: Path, created_at: str) -> dict[str, Any]:
    jobs = []
    for row in summary_rows:
        label = label_from_summary(row)
        for precision in PRECISIONS:
            for axis in ("latency", "energy", "ap"):
                jobs.append(
                    {
                        "job_id": f"original60_quant:{axis}:{label}:{precision}",
                        "label": label,
                        "candidate_id": precision_candidate_id(row, precision),
                        "original60_candidate_id": original_candidate_id(row),
                        "width": row.get("width"),
                        "precision": precision,
                        "axis": axis,
                        "state_policy": "fp16_import_existing_else_no_claim",
                    }
                )
    return {
        "schema": "original60_quant_state_plan_v1",
        "created_at": created_at,
        "candidate_count": len(summary_rows),
        "precisions": list(PRECISIONS),
        "jobs": jobs,
        "output_root": str(output_root),
    }


def make_quarantine_rows(axis_rows: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    out = []
    for axis, rows in axis_rows.items():
        for row in rows:
            reason = row.get("failure_reason")
            if not reason:
                continue
            out.append(
                quarantine_row(
                    job_id=f"original60_quant:{axis}:{row['label']}:{row['precision']}",
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
    latency = {(row["label"], row["precision"]): row for row in latency_rows}
    energy = {(row["label"], row["precision"]): row for row in energy_rows}
    ap = {(row["label"], row["precision"]): row for row in ap_rows}
    keys = sorted(latency)
    out = []
    for key in keys:
        label, precision = key
        lrow = latency[key]
        erow = energy[key]
        arow = ap[key]
        failures = [
            str(row.get("failure_reason"))
            for row in (lrow, erow, arow)
            if row.get("failure_reason")
        ]
        out.append(
            {
                "label": label,
                "candidate_id": lrow.get("candidate_id"),
                "original60_candidate_id": lrow.get("original60_candidate_id"),
                "width": width_text(lrow["width"]),
                "precision": precision,
                "quant_method": lrow.get("quant_method"),
                "quant_scope": lrow.get("quant_scope"),
                "full_network_claim": False,
                "latency_ms": None
                if lrow.get("latency_p50_us") is None
                else round(float(lrow["latency_p50_us"]) / 1000.0, 6),
                "latency_status": lrow.get("measurement_status"),
                "latency_schedule_policy": lrow.get("schedule_policy"),
                "latency_tvm_strategy": lrow.get("tvm_strategy"),
                "latency_run_id": lrow.get("run_id"),
                "latency_measurement_source": lrow.get("measurement_source"),
                "latency_claim_status": lrow.get("claim_status"),
                "latency_evidence_scope": evidence_scope(lrow),
                "latency_repeat": lrow.get("repeat"),
                "energy_j_per_inference": erow.get("joule_per_inference"),
                "energy_status": erow.get("measurement_status"),
                "energy_schedule_policy": erow.get("schedule_policy"),
                "energy_measurement_source": erow.get("measurement_source"),
                "energy_claim_status": erow.get("claim_status"),
                "energy_evidence_scope": evidence_scope(erow),
                "energy_repeat": erow.get("repeat"),
                "ap70": arow.get("metric_value"),
                "ap_status": arow.get("measurement_status"),
                "ap_schedule_policy": arow.get("schedule_policy"),
                "ap_source_kind": arow.get("ap_source_kind") or arow.get("measurement_source"),
                "ap_measurement_source": arow.get("measurement_source"),
                "ap_claim_status": arow.get("claim_status"),
                "ap_evidence_scope": evidence_scope(arow),
                "quality_gate_status": ";".join(
                    item
                    for item in (
                        str(lrow.get("quality_gate_status") or ""),
                        str(erow.get("quality_gate_status") or ""),
                        str(arow.get("quality_gate_status") or ""),
                    )
                    if item
                ),
                "failure_reasons": ";".join(failures),
            }
        )
    return out


def write_summary(output_root: Path, rows: list[dict[str, Any]]) -> None:
    exports = output_root / "exports"
    exports.mkdir(parents=True, exist_ok=True)
    fields = [
        "label",
        "candidate_id",
        "original60_candidate_id",
        "width",
        "precision",
        "quant_method",
        "quant_scope",
        "full_network_claim",
        "latency_ms",
        "latency_status",
        "latency_schedule_policy",
        "latency_tvm_strategy",
        "energy_j_per_inference",
        "energy_status",
        "energy_schedule_policy",
        "ap70",
        "ap_status",
        "ap_schedule_policy",
        "ap_source_kind",
        "quality_gate_status",
        "failure_reasons",
    ]
    csv_path = exports / "original60_quant_three_metric_summary_latest.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "schema": "original60_quant_three_metric_summary_v1",
        "candidate_count": len({row["label"] for row in rows}),
        "total_cells": len(rows),
        "precision_counts": dict(sorted(Counter(row["precision"] for row in rows).items())),
        "latency_status_counts": dict(sorted(Counter(row["latency_status"] for row in rows).items())),
        "energy_status_counts": dict(sorted(Counter(row["energy_status"] for row in rows).items())),
        "ap_status_counts": dict(sorted(Counter(row["ap_status"] for row in rows).items())),
        "update_note": (
            "FP32 latency should prefer rows/fp32_latency_strict_direct_rows_v1.jsonl when supplied, "
            "with historical remap/smoke rows retained only as fallback audit evidence. FP32 energy "
            "prefers rows/fp32_original60_energy_threaded60_rows_v1.jsonl H800 threaded_window telemetry. "
            "FP32 AP prefers rows/fp32_true_original60_ap_rows_v1.jsonl when available; otherwise it "
            "falls back to Stage2 original60 full-model AP reference rows. FP16 latency/energy now "
            "prefer rows/fp16_rewritten_tensorcore_full60_latency_rows_v1.jsonl and "
            "rows/fp16_rewritten_tensorcore_full60_energy_rows_v1.jsonl; legacy true-FP16 smoke rows "
            "are retained only as fallback audit evidence."
        ),
        "rows": rows,
    }
    (exports / "original60_quant_three_metric_summary_latest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    def measured_count(precision: str, axis: str) -> int:
        return sum(
            1
            for row in rows
            if row["precision"] == precision and row.get(f"{axis}_status") == "measured"
        )

    def precision_count(precision: str) -> int:
        return sum(1 for row in rows if row["precision"] == precision)

    md = [
        "# Original60 Quant 三指标状态表",
        "",
        "说明: 这是 60 x 3 precision 状态覆盖表; latency 单位统一为 ms; energy 单位为 joule/inference; 是否可声明 measured 以每行 status/claim gate 为准。",
        "",
        "FP32 补充说明: latency 当前优先使用 `rows/fp32_latency_strict_direct_rows_v1.jsonl` 的 H800 TVM strict direct 全量重测数据；旧 smoke/remap row 仅作为历史审计或 fallback。energy 优先使用 `rows/fp32_original60_energy_threaded60_rows_v1.jsonl` 的 H800 TVM `threaded_window` power telemetry 全量重测数据。具体 tuned/default 路线见 `latency schedule` 和 `energy schedule` 列。AP 优先使用 `rows/fp32_true_original60_ap_rows_v1.jsonl` 的 true-FP32 full-val 实测；缺失时才 fallback 到 Stage2 original60 AP source map reference。",
        "",
        "FP16 补充说明: latency/energy 当前优先使用 `rows/fp16_rewritten_tensorcore_full60_latency_rows_v1.jsonl` 与 `rows/fp16_rewritten_tensorcore_full60_energy_rows_v1.jsonl` 的 H800 TVM rewritten TensorCore full60 实测；旧 true-FP16 smoke row 仅作为 fallback 审计证据。该路径仍声明 `backbone_only`，不声明 full network。",
        "",
        "| item | fp32 | fp16 | int8 |",
        "|---|---:|---:|---:|",
        *[
            "| {axis}_measured | {fp32}/{fp32_total} | {fp16}/{fp16_total} | {int8}/{int8_total} |".format(
                axis=axis,
                fp32=measured_count("fp32", axis),
                fp32_total=precision_count("fp32"),
                fp16=measured_count("fp16", axis),
                fp16_total=precision_count("fp16"),
                int8=measured_count("int8", axis),
                int8_total=precision_count("int8"),
            )
            for axis in ("latency", "energy", "ap")
        ],
        "",
        "| label | width | precision | latency ms | latency | latency schedule | latency tvm strategy | energy J | energy schedule | energy | AP70 | AP | failure reasons |",
        "|---|---|---|---:|---|---|---|---:|---|---|---:|---|---|",
    ]
    for row in rows:
        md.append(
            "| {label} | {width} | {precision} | {latency_ms} | {latency_status} | "
            "{latency_schedule_policy} | {latency_tvm_strategy} | "
            "{energy_j_per_inference} | {energy_schedule_policy} | {energy_status} | {ap70} | {ap_status} | "
            "{failure_reasons} |".format(
                label=row["label"],
                width=row["width"],
                precision=row["precision"],
                latency_ms="" if row["latency_ms"] is None else f"{row['latency_ms']:.6f}",
                latency_status=row["latency_status"],
                latency_schedule_policy=row.get("latency_schedule_policy") or "",
                latency_tvm_strategy=row.get("latency_tvm_strategy") or "",
                energy_j_per_inference=""
                if row["energy_j_per_inference"] is None
                else f"{float(row['energy_j_per_inference']):.6f}",
                energy_schedule_policy=row.get("energy_schedule_policy") or "",
                energy_status=row["energy_status"],
                ap70="" if row["ap70"] is None else f"{float(row['ap70']):.9f}",
                ap_status=row["ap_status"],
                failure_reasons=row["failure_reasons"],
            )
        )
    (exports / "original60_quant_three_metric_summary_latest.md").write_text(
        "\n".join(md) + "\n",
        encoding="utf-8",
    )


def write_gap_report(output_root: Path, axis_rows: dict[str, list[dict[str, Any]]]) -> None:
    reason_counts: Counter[str] = Counter()
    axis_counts: dict[str, Counter[str]] = defaultdict(Counter)
    rows = []
    for axis, data in axis_rows.items():
        for row in data:
            reason = row.get("failure_reason")
            if not reason:
                continue
            reason_text = str(reason)
            reason_counts[reason_text] += 1
            axis_counts[axis][reason_text] += 1
            rows.append(
                {
                    "axis": axis,
                    "label": row.get("label"),
                    "candidate_id": row.get("original60_candidate_id"),
                    "target_candidate_id": row.get("candidate_id"),
                    "precision": row.get("precision"),
                    "failure_reason": reason_text,
                    "measurement_status": row.get("measurement_status"),
                    "quant_method": row.get("quant_method"),
                    "quant_scope": row.get("quant_scope"),
                }
            )
    payload = {
        "schema": "original60_quant_gap_report_v1",
        "failure_reason_counts": dict(sorted(reason_counts.items())),
        "axis_failure_reason_counts": {
            axis: dict(sorted(counter.items())) for axis, counter in sorted(axis_counts.items())
        },
        "rows": rows,
    }
    exports = output_root / "exports"
    exports.mkdir(parents=True, exist_ok=True)
    (exports / "original60_quant_gap_report_latest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    created_at = args.created_at or utc_timestamp()
    summary_rows = read_csv(args.original60_summary_csv)
    fp32_smoke_paths = default_existing_row_paths(
        output_root,
        args.fp32_latency_remap_rows + args.fp32_latency_smoke_rows,
        "fp32_latency_original60_remapped_rows_v1.jsonl",
        "fp32_latency_smoke_rows_v1.jsonl",
        # gap2: lhc_17/s2_096 default-only salvage rows (tuned path CUDA illegal
        # memory access, deterministic). Additive source, only these 2 labels.
        "fp32_latency_gap2_default_salvage_rows_20260702.jsonl",
    )
    fp16_latency_paths = default_existing_row_paths(
        output_root,
        args.fp16_latency_smoke_rows,
        "fp16_rewritten_tensorcore_full60_latency_rows_v1.jsonl",
        "fp16_true_original60_latency_rows_v1.jsonl",
        "fp16_true_latency_smoke_rows_v1.jsonl",
    )
    fp16_energy_paths = default_existing_row_paths(
        output_root,
        args.fp16_energy_smoke_rows,
        "fp16_rewritten_tensorcore_full60_energy_rows_v1.jsonl",
        "fp16_true_original60_energy_rows_v1.jsonl",
        "fp16_true_ap_energy_smoke_rows_v1.jsonl",
    )
    fp32_energy_paths = default_existing_row_paths(
        output_root,
        args.fp32_energy_remeasure_rows,
        "fp32_original60_energy_threaded60_rows_v1.jsonl",
        "fp32_original60_energy_remeasured_rows_v1.jsonl",
    )
    int8_latency_paths = default_existing_row_paths(
        output_root,
        args.int8_latency_smoke_rows,
        "int8_latency_smoke_rows_v1.jsonl",
    )
    int8_energy_paths = default_existing_row_paths(
        output_root,
        args.int8_energy_smoke_rows,
        "int8_ap_energy_smoke_rows_v1.jsonl",
    )
    native_int8_latency_paths = default_existing_row_paths(
        output_root,
        args.native_int8_full_onnx_latency_rows,
        "native_int8_full_onnx_original60_latency_rows_v1.jsonl",
        "native_int8_full_onnx_latency_rows_v1.jsonl",
    )
    native_int8_energy_paths = default_existing_row_paths(
        output_root,
        args.native_int8_full_onnx_energy_rows,
        "native_int8_full_onnx_original60_energy_rows_v1.jsonl",
        "native_int8_full_onnx_energy_rows_v1.jsonl",
    )
    fp16_ap_paths = default_existing_row_paths(
        output_root,
        args.fp16_ap_rows,
        "fp16_true_original60_ap_rows_v1.jsonl",
    )
    fp32_ap_paths = default_existing_row_paths(
        output_root,
        args.fp32_ap_rows,
        "fp32_true_original60_ap_rows_v1.jsonl",
    )
    int8_ap_paths = default_existing_row_paths(
        output_root,
        args.int8_ap_rows,
        "native_int8_original60_ap_rows_v1.jsonl",
    )

    fp32_smoke_rows = measured_fp32_smoke_index(fp32_smoke_paths)
    fp16_smoke_rows = measured_fp16_smoke_index(fp16_latency_paths)
    fp16_energy_rows = measured_energy_index(fp16_energy_paths, "fp16")
    fp32_energy_rows = measured_energy_index(fp32_energy_paths, "fp32")
    int8_energy_rows = measured_energy_index(int8_energy_paths, "int8")
    int8_energy_rows.update(
        measured_native_int8_full_onnx_energy_index(native_int8_energy_paths)
    )
    int8_smoke_rows = measured_int8_smoke_index(int8_latency_paths)
    int8_smoke_rows.update(
        measured_native_int8_full_onnx_latency_index(native_int8_latency_paths)
    )
    canonical_widths = {
        label_from_summary(row): parse_width_text(str(row["width"]))
        for row in summary_rows
    }
    fp16_ap_rows = measured_ap_index(fp16_ap_paths, "fp16", canonical_widths)
    fp32_ap_rows = measured_ap_index(fp32_ap_paths, "fp32", canonical_widths)
    int8_ap_rows = measured_ap_index(int8_ap_paths, "int8", canonical_widths)
    latency_rows = []
    energy_rows = []
    ap_rows = []
    for summary_row in summary_rows:
        for precision in PRECISIONS:
            latency_rows.append(
                make_latency_row(
                    summary_row,
                    precision,
                    created_at,
                    fp32_smoke_rows,
                    fp16_smoke_rows,
                    int8_smoke_rows,
                )
            )
            energy_rows.append(
                make_energy_row(
                    summary_row,
                    precision,
                    created_at,
                    fp32_energy_rows,
                    fp16_energy_rows,
                    int8_energy_rows,
                )
            )
            ap_rows.append(
                make_ap_row(
                    summary_row,
                    precision,
                    created_at,
                    fp16_ap_rows,
                    fp32_ap_rows,
                    int8_ap_rows,
                )
            )

    (output_root / "plans").mkdir(parents=True, exist_ok=True)
    (output_root / "plans/original60_quant_state_plan_v1.json").write_text(
        json.dumps(make_plan(summary_rows, output_root, created_at), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    write_jsonl(output_root / "rows/latency_original60_quant_rows_v1.jsonl", latency_rows)
    write_jsonl(output_root / "rows/energy_original60_quant_rows_v1.jsonl", energy_rows)
    write_jsonl(output_root / "rows/ap_original60_quant_rows_v1.jsonl", ap_rows)
    axis_rows = {"latency": latency_rows, "energy": energy_rows, "ap": ap_rows}
    write_jsonl(
        output_root / "quarantine/original60_quant_unclaimable_v1.jsonl",
        make_quarantine_rows(axis_rows),
    )
    combined = make_summary_rows(latency_rows, energy_rows, ap_rows)
    write_summary(output_root, combined)
    write_gap_report(output_root, axis_rows)
    print(
        json.dumps(
            {
                "schema": "original60_quant_state_generation_result_v1",
                "output_root": str(output_root),
                "candidate_count": len(summary_rows),
                "latency_rows": len(latency_rows),
                "energy_rows": len(energy_rows),
                "ap_rows": len(ap_rows),
                "summary_rows": len(combined),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
