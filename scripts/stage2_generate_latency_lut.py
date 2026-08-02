#!/usr/bin/env python3
"""Run one latency measurement command and append a Stage2 latency LUT row."""

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
    latency_lut_row,
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
    parser.add_argument("--backend", default="h800_tvm")
    parser.add_argument("--manifest-digest", default="unknown")
    parser.add_argument("--run-id")
    parser.add_argument("--created-at")
    parser.add_argument("--warmup-iters", type=int, default=50)
    parser.add_argument("--measure-iters", type=int, default=200)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--measurement-command-json", required=True)
    parser.add_argument("--out-jsonl", required=True)
    return parser.parse_args()


def _optional_float(payload: dict[str, Any], key: str) -> float | None:
    value = payload.get(key)
    return None if value is None else float(value)


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
    payload = run_json_command(args.measurement_command_json)
    if payload.get("latency_p50_us") is None:
        raise LutProductizationError("latency command must emit latency_p50_us")

    run_id = args.run_id or str(
        payload.get("run_id") or args.job_id or f"latency_{utc_timestamp()}"
    )
    row = latency_lut_row(
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
        latency_p50_us=float(payload["latency_p50_us"]),
        latency_p90_us=_optional_float(payload, "latency_p90_us"),
        latency_mean_us=_optional_float(payload, "latency_mean_us"),
        latency_std_us=_optional_float(payload, "latency_std_us"),
        latency_min_us=_optional_float(payload, "latency_min_us"),
        latency_max_us=_optional_float(payload, "latency_max_us"),
        warmup_iters=int(payload.get("warmup_iters", args.warmup_iters)),
        measure_iters=int(payload.get("measure_iters", args.measure_iters)),
        repeat=int(payload.get("repeat", args.repeat)),
        batch_size=int(payload.get("batch_size", args.batch_size)),
        input_shape=payload.get("input_shape", {}),
        tvm_target=payload.get("tvm_target", "cuda"),
        tvm_strategy=payload.get("tvm_strategy", "relax_metaschedule"),
        build_status=payload.get("build_status", "success"),
        build_time_s=_optional_float(payload, "build_time_s"),
        provenance=payload.get(
            "provenance",
            "H800 TVM Relax/MetaSchedule measured dense-core LUT",
        ),
        run_id=run_id,
        created_at=args.created_at or str(payload.get("created_at") or utc_timestamp()),
        source_files=list(payload.get("source_files", [])),
        raw_artifact=payload.get("raw_artifact"),
        notes=payload.get("notes", "dense-core only; not full-model latency"),
        **_quant_kwargs(args),
    )
    append_jsonl(args.out_jsonl, row)
    print(json.dumps({"schema": "lut_generation_result_v1", "row_id": row["row_id"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
