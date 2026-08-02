#!/usr/bin/env python3
"""Run one energy telemetry command and append a Stage2 energy LUT row."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.stage2.lut_productization import (  # noqa: E402
    LutProductizationError,
    append_jsonl,
    energy_lut_row,
    parse_width_csv,
    run_json_command,
    utc_timestamp,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id")
    parser.add_argument("--model", required=True)
    parser.add_argument("--config-id", required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--software-point-id", required=True)
    parser.add_argument("--dense-stage", required=True)
    parser.add_argument("--width", required=True)
    parser.add_argument("--quant-policy", required=True)
    parser.add_argument("--precision")
    parser.add_argument("--quant-scheme")
    parser.add_argument("--quant-method")
    parser.add_argument("--quant-scope")
    parser.add_argument("--calibration-source")
    parser.add_argument("--calibration-digest")
    parser.add_argument("--calibration-manifest-digest")
    parser.add_argument("--quant-recipe-digest")
    parser.add_argument("--quantized-onnx-digest")
    parser.add_argument("--calibrator")
    parser.add_argument("--calibration-inputs", default="")
    parser.add_argument("--fallback-policy")
    parser.add_argument("--layer-precision-summary")
    parser.add_argument("--layer-precision-summary-digest")
    parser.add_argument("--full-network-claim", choices=("true", "false"))
    parser.add_argument("--engine-kind")
    parser.add_argument("--engine-digest")
    parser.add_argument("--measurement-source")
    parser.add_argument("--claim-status")
    parser.add_argument("--quality-gate-status")
    parser.add_argument("--schedule-profile")
    parser.add_argument("--tune-budget")
    parser.add_argument("--schedule-policy", required=True)
    parser.add_argument("--optimized-scope", default="rsu_dense_core")
    parser.add_argument("--backend", default="h800_tvm_power_telemetry")
    parser.add_argument("--manifest-digest", default="unknown")
    parser.add_argument("--latency-run-id")
    parser.add_argument("--run-id")
    parser.add_argument("--created-at")
    parser.add_argument("--warmup-iters", type=int, default=50)
    parser.add_argument("--measure-iters", type=int, default=200)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--telemetry-command-json", required=True)
    parser.add_argument("--out-jsonl", required=True)
    return parser.parse_args()


def _optional_float(payload: dict[str, Any], key: str) -> float | None:
    value = payload.get(key)
    return None if value is None else float(value)


def _optional_int(payload: dict[str, Any], key: str) -> int | None:
    value = payload.get(key)
    return None if value is None else int(value)


def _quant_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for arg_name, row_name in (
        ("precision", "precision"),
        ("quant_scheme", "quant_scheme"),
        ("quant_method", "quant_method"),
        ("quant_scope", "quant_scope"),
        ("calibration_source", "calibration_source"),
        ("calibration_digest", "calibration_digest"),
        ("calibration_manifest_digest", "calibration_manifest_digest"),
        ("quant_recipe_digest", "quant_recipe_digest"),
        ("quantized_onnx_digest", "quantized_onnx_digest"),
        ("calibrator", "calibrator"),
        ("fallback_policy", "fallback_policy"),
        ("layer_precision_summary", "layer_precision_summary"),
        ("layer_precision_summary_digest", "layer_precision_summary_digest"),
        ("engine_kind", "engine_kind"),
        ("engine_digest", "engine_digest"),
        ("measurement_source", "measurement_source"),
        ("claim_status", "claim_status"),
        ("quality_gate_status", "quality_gate_status"),
        ("schedule_profile", "schedule_profile"),
        ("tune_budget", "tune_budget"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            out[row_name] = value
    if args.calibration_inputs:
        out["calibration_inputs"] = [
            item.strip() for item in args.calibration_inputs.split(",") if item.strip()
        ]
    if args.full_network_claim is not None:
        out["full_network_claim"] = args.full_network_claim == "true"
    return out


def main() -> int:
    args = parse_args()
    payload = run_json_command(args.telemetry_command_json)
    for key in ("joule_per_inference", "telemetry_source", "raw_artifact"):
        if payload.get(key) is None:
            raise LutProductizationError(f"energy telemetry command must emit {key}")

    run_id = args.run_id or str(
        payload.get("run_id") or args.job_id or f"energy_{utc_timestamp()}"
    )
    row = energy_lut_row(
        config_id=args.config_id,
        model=args.model,
        manifest_digest=args.manifest_digest,
        candidate_id=args.candidate_id,
        software_point_id=args.software_point_id,
        dense_stage=args.dense_stage,
        optimized_scope=args.optimized_scope,
        width=parse_width_csv(args.width),
        quant_policy=args.quant_policy,
        schedule_policy=args.schedule_policy,
        backend=args.backend,
        measurement_status="measured",
        joule_per_inference=float(payload["joule_per_inference"]),
        watt_avg=_optional_float(payload, "watt_avg"),
        watt_p50=_optional_float(payload, "watt_p50"),
        watt_p90=_optional_float(payload, "watt_p90"),
        idle_watt_avg=_optional_float(payload, "idle_watt_avg"),
        idle_baseline_policy=payload.get("idle_baseline_policy", "subtract_idle_avg"),
        sample_window_ms=_optional_int(payload, "sample_window_ms"),
        telemetry_source=payload["telemetry_source"],
        power_cap_watt=_optional_float(payload, "power_cap_watt"),
        clock_policy=payload.get("clock_policy", "default"),
        latency_config_id=payload.get("latency_config_id", args.config_id),
        latency_run_id=payload.get(
            "latency_run_id",
            args.latency_run_id or "same_as_config_id",
        ),
        warmup_iters=int(payload.get("warmup_iters", args.warmup_iters)),
        measure_iters=int(payload.get("measure_iters", args.measure_iters)),
        repeat=int(payload.get("repeat", args.repeat)),
        provenance=payload.get("provenance", "H800 power telemetry aligned with latency run"),
        run_id=run_id,
        measurement_run_id=run_id,
        row_source="direct_generation",
        created_at=args.created_at or str(payload.get("created_at") or utc_timestamp()),
        source_files=list(payload.get("source_files", [])),
        raw_artifact=payload["raw_artifact"],
        notes=payload.get("notes", "same config_id as latency row; dense-core only"),
        **_quant_kwargs(args),
    )
    append_jsonl(args.out_jsonl, row)
    print(json.dumps({"schema": "lut_generation_result_v1", "row_id": row["row_id"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
