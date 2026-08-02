#!/usr/bin/env python3
"""Scale-aware paired backbone performance measurement for CoDriving TVM RouteB."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stage2_codriving_int8_provenance import (
    INT8_QUANTIZATION_SEMANTICS,
    report_has_valid_int8_calibration,
    validate_int8_calibration_manifest,
)
from stage2_v2_gold_coldstart96_codriving_tvm_resnet_ap_eval import (
    TvmRouteBResnetRuntime,
    routeb_mode_spec,
    sha256_file,
)


SCHEMA = "v2_gold_coldstart_96_codriving_tvm_resnet_perf_eval_v1"
PIPELINE_SCOPE = "tvm_routeb_resnet_backbone_only_vm"
MEASUREMENT_SEMANTICS = "scale_aware_paired_backbone_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_calibration_manifest_from_ap_report(
    path: Path,
    *,
    expected_mode: str,
    expected_width: str,
    expected_onnx: Path,
    expected_onnx_sha256: str,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_precision = routeb_mode_spec(expected_mode)["precision"]
    if payload.get("mode") != expected_mode:
        raise ValueError(
            f"AP report mode {payload.get('mode')!r} does not match requested mode {expected_mode!r}"
        )
    if payload.get("precision") != expected_precision:
        raise ValueError("AP report precision does not match requested mode")
    if payload.get("width") != expected_width:
        raise ValueError("AP report width does not match requested width")
    if Path(str(payload.get("onnx") or "")).resolve() != expected_onnx.resolve():
        raise ValueError("AP report ONNX path does not match performance ONNX")
    if payload.get("onnx_sha256") != expected_onnx_sha256:
        raise ValueError("AP report ONNX SHA256 does not match performance ONNX")
    if payload.get("pipeline_scope") != "tvm_routeb_resnet_in_full_pytorch_eval":
        raise ValueError("AP report does not use the full PyTorch evaluation pipeline")
    if not report_has_valid_int8_calibration(payload):
        raise ValueError("AP report does not contain a valid calibrated INT8 manifest")
    validate_compile_summary_for_mode(payload["compile_summary"], expected_mode)
    manifest = payload["compile_summary"]["int8_calibration"]
    return validate_int8_calibration_manifest(manifest)


def validate_compile_summary_for_mode(summary: Any, mode: str) -> dict[str, Any]:
    if not isinstance(summary, Mapping):
        raise ValueError("compile_summary must be a mapping")
    plan = summary.get("conv_precision_plan")
    signatures = summary.get("int8_signature_plan")
    if not isinstance(plan, Mapping) or not plan:
        raise ValueError("compile_summary lacks conv_precision_plan")
    if not isinstance(signatures, Mapping):
        raise ValueError("compile_summary lacks int8_signature_plan")
    invalid_values = set(plan.values()) - {"fp16", "int8"}
    if invalid_values:
        raise ValueError(f"invalid Conv precision values: {sorted(invalid_values)}")
    int8_names = {str(name) for name, precision in plan.items() if precision == "int8"}
    signature_names = {str(name) for name in signatures}
    spec = routeb_mode_spec(mode)
    if spec["precision"] == "fp16":
        if int8_names or signatures:
            raise ValueError("FP16 mode must not contain INT8 Conv entries")
        return dict(summary)
    if summary.get("quantization_semantics") != INT8_QUANTIZATION_SEMANTICS:
        raise ValueError("INT8/mixed compile_summary is not calibrated")
    if mode == "int8_all" and len(int8_names) != len(plan):
        raise ValueError("int8_all requires all Conv entries to use INT8")
    if mode in {"mixed_top25_flops", "mixed_top50_flops"}:
        divisor = 4 if mode == "mixed_top25_flops" else 2
        expected_int8 = (len(plan) + divisor - 1) // divisor
        if len(int8_names) != expected_int8 or len(int8_names) == len(plan):
            raise ValueError(
                f"{mode} requires {expected_int8}/{len(plan)} INT8 Conv entries, got {len(int8_names)}"
            )
    if signature_names != int8_names:
        raise ValueError("int8_signature_plan keys must exactly match INT8 Conv entries")
    manifest = validate_int8_calibration_manifest(
        summary.get("int8_calibration"),
        required_signatures={str(value) for value in signatures.values()},
    )
    return {**dict(summary), "int8_calibration": manifest}


def validate_gpu_mapping(
    *,
    tvm_gpu: int,
    physical_gpu: int,
    cuda_visible_devices: str | None,
) -> dict[str, int | str]:
    visible_text = str(cuda_visible_devices or "").strip()
    if visible_text:
        entries = [entry.strip() for entry in visible_text.split(",") if entry.strip()]
        if any(not entry.isdigit() for entry in entries):
            raise ValueError("CUDA_VISIBLE_DEVICES must use numeric physical GPU indices")
        if tvm_gpu < 0 or tvm_gpu >= len(entries):
            raise ValueError("TVM logical GPU is outside CUDA_VISIBLE_DEVICES")
        mapped_physical = int(entries[tvm_gpu])
    else:
        mapped_physical = tvm_gpu
    if mapped_physical != physical_gpu:
        raise ValueError(
            f"TVM logical GPU {tvm_gpu} maps to physical GPU {mapped_physical}, not {physical_gpu}"
        )
    return {
        "tvm_logical_gpu": tvm_gpu,
        "physical_gpu": physical_gpu,
        "cuda_visible_devices": visible_text,
    }


def query_gpu_identity(physical_gpu: int) -> dict[str, str]:
    import pynvml  # type: ignore

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(int(physical_gpu))

    def normalize(value: Any) -> str:
        return value.decode("utf-8") if isinstance(value, bytes) else str(value)

    return {
        "gpu_uuid": normalize(pynvml.nvmlDeviceGetUUID(handle)),
        "gpu_name": normalize(pynvml.nvmlDeviceGetName(handle)),
        "driver_version": normalize(pynvml.nvmlSystemGetDriverVersion()),
    }


def validate_runtime_plan(expected: Any, observed: Any) -> dict[str, str]:
    if not isinstance(expected, Mapping) or not isinstance(observed, Mapping):
        raise ValueError("AP/runtime precision plan must be mappings")
    expected_plan = {str(name): str(value) for name, value in expected.items()}
    observed_plan = {str(name): str(value) for name, value in observed.items()}
    if observed_plan != expected_plan:
        raise ValueError("runtime Conv precision plan does not exactly match AP report precision plan")
    return observed_plan


def validate_energy_result(energy: Any) -> dict[str, Any]:
    if not isinstance(energy, Mapping) or energy.get("status") != "success":
        raise ValueError(f"energy sampling failed: {energy!r}")
    joule = float(energy.get("joule_per_inference", math.nan))
    if not math.isfinite(joule) or joule < 0.0:
        raise ValueError(f"energy sampling returned invalid joule_per_inference: {joule!r}")
    idle_samples = int(energy.get("idle_sample_count") or 0)
    active_samples = int(energy.get("active_sample_count") or 0)
    completed = int(energy.get("completed_measure_iters") or 0)
    requested = int(energy.get("requested_measure_iters") or 0)
    elapsed = float(energy.get("elapsed_s") or 0.0)
    minimum = float(energy.get("min_active_s") or 0.0)
    if idle_samples <= 0 or active_samples <= 0:
        raise ValueError("energy sampling requires positive idle/active sample counts")
    if requested <= 0 or completed < requested:
        raise ValueError("energy sampling did not complete the requested positive iteration count")
    if minimum <= 0.0 or elapsed < minimum:
        raise ValueError("energy sampling did not satisfy positive min_active_s")
    return dict(energy)


def _linear_percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("latency samples must be nonempty")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize_latencies(values: list[float]) -> dict[str, float | int]:
    normalized = [float(value) for value in values]
    if not normalized or any(not math.isfinite(value) or value <= 0.0 for value in normalized):
        raise ValueError("latency samples must be finite positive values")
    return {
        "samples": len(normalized),
        "min_ms": min(normalized),
        "p50_ms": _linear_percentile(normalized, 0.5),
        "p90_ms": _linear_percentile(normalized, 0.9),
        "max_ms": max(normalized),
        "mean_ms": statistics.fmean(normalized),
        "stdev_ms": statistics.stdev(normalized) if len(normalized) > 1 else 0.0,
    }


def _pipeline_fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_report(
    *,
    width: str,
    mode: str,
    graph_io_dtype: str,
    onnx: Path,
    onnx_sha256: str,
    input_sha256: str,
    independent_run_id: int,
    latency: dict[str, Any],
    energy: dict[str, Any] | None,
    compile_summary: dict[str, Any],
    measurement_config: dict[str, Any],
) -> dict[str, Any]:
    mode_spec = routeb_mode_spec(mode)
    compile_summary = validate_compile_summary_for_mode(compile_summary, mode)
    if energy is not None:
        energy = validate_energy_result(energy)
    calibration = compile_summary.get("int8_calibration") or {}
    fingerprint_fields = {
        "schema": SCHEMA,
        "pipeline_scope": PIPELINE_SCOPE,
        "measurement_semantics": MEASUREMENT_SEMANTICS,
        "width": width,
        "mode": mode,
        "precision": mode_spec["precision"],
        "mixed_policy": mode_spec["mixed_policy"],
        "graph_io_dtype": graph_io_dtype,
        "onnx_sha256": onnx_sha256,
        "input_sha256": input_sha256,
        "quantization_semantics": compile_summary.get("quantization_semantics"),
        "calibration_source_sha256": calibration.get("calibration_source_sha256"),
        "calibration_summary_sha256": calibration.get("calibration_summary_sha256"),
        "conv_precision_plan": compile_summary.get("conv_precision_plan"),
        "measurement_config": measurement_config,
    }
    return {
        "schema": SCHEMA,
        "created_at_utc": utc_now(),
        "status": "success",
        "pipeline_scope": PIPELINE_SCOPE,
        "measurement_semantics": MEASUREMENT_SEMANTICS,
        "width": width,
        "mode": mode,
        "precision": mode_spec["precision"],
        "mixed_policy": mode_spec["mixed_policy"],
        "graph_io_dtype": graph_io_dtype,
        "onnx": str(onnx),
        "onnx_sha256": onnx_sha256,
        "input_sha256": input_sha256,
        "independent_run_id": independent_run_id,
        "pipeline_fingerprint": _pipeline_fingerprint(fingerprint_fields),
        "measurement_config": measurement_config,
        "latency": latency,
        "energy": energy,
        "compile_summary": compile_summary,
    }


def _load_vm_argument(runtime: TvmRouteBResnetRuntime, input_npz: Path) -> Any:
    import numpy as np

    input_specs = runtime.compile_summary.get("input_specs") or []
    if len(input_specs) != 1:
        raise ValueError(f"expected exactly one RouteB backbone input, got {len(input_specs)}")
    spec = input_specs[0]
    expected_shape = tuple(int(value) for value in spec["shape"])
    expected_dtype = str(spec["dtype"])
    with np.load(input_npz, allow_pickle=False) as payload:
        if "spatial_features" not in payload:
            raise ValueError("input NPZ lacks spatial_features")
        spatial = payload["spatial_features"]
        if spatial.ndim != len(expected_shape) + 1 or tuple(spatial.shape[1:]) != expected_shape:
            raise ValueError(
                f"input NPZ spatial_features shape {spatial.shape} does not match (*,{expected_shape})"
            )
        value = np.asarray(spatial[0], dtype=expected_dtype)
    return runtime.tvm.runtime.tensor(value, device=runtime.dev)


def run_measurement(args: argparse.Namespace) -> dict[str, Any]:
    mode_spec = routeb_mode_spec(args.mode)
    gpu_mapping = validate_gpu_mapping(
        tvm_gpu=args.tvm_gpu,
        physical_gpu=args.physical_gpu,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
    )
    int8_calibration = None
    expected_ap_plan = None
    onnx_sha256 = sha256_file(args.onnx)
    if mode_spec["precision"] != "fp16":
        if args.ap_report is None:
            raise ValueError("--ap-report is required for INT8/mixed performance measurement")
        int8_calibration = load_calibration_manifest_from_ap_report(
            args.ap_report,
            expected_mode=args.mode,
            expected_width=args.width,
            expected_onnx=args.onnx,
            expected_onnx_sha256=onnx_sha256,
        )
        ap_payload = json.loads(args.ap_report.read_text(encoding="utf-8"))
        expected_ap_plan = ap_payload["compile_summary"]["conv_precision_plan"]
    input_sha256 = sha256_file(args.input_npz)
    if int8_calibration is not None and input_sha256 != int8_calibration["calibration_source_sha256"]:
        raise ValueError("performance input NPZ must match the calibrated AP artifact SHA256")

    runtime = TvmRouteBResnetRuntime(
        args.onnx,
        mode=args.mode,
        gpu=args.tvm_gpu,
        graph_io_dtype=args.graph_io_dtype,
        int8_calibration=int8_calibration,
    )
    if expected_ap_plan is not None:
        validate_runtime_plan(expected_ap_plan, runtime.compile_summary["conv_precision_plan"])
    vm_arg = _load_vm_argument(runtime, args.input_npz)
    runtime.vm["main"](vm_arg)
    runtime.dev.sync()
    for _ in range(args.warmup):
        runtime.vm["main"](vm_arg)
    runtime.dev.sync()

    latency_samples: list[float] = []
    for _ in range(args.reps):
        start = time.perf_counter()
        runtime.vm["main"](vm_arg)
        runtime.dev.sync()
        latency_samples.append((time.perf_counter() - start) * 1000.0)
    latency = summarize_latencies(latency_samples)
    latency["raw_ms"] = latency_samples

    energy = None
    if args.measure_energy:
        from stage2_codriving_whole_engine_tc_v1 import _measure_energy_loop

        energy = validate_energy_result(_measure_energy_loop(
            runtime.vm,
            [vm_arg],
            runtime.dev,
            args.physical_gpu,
            energy_iters=args.energy_iters,
            min_active_s=args.energy_min_active_s,
        ))
    measurement_config = {
        **gpu_mapping,
        **query_gpu_identity(args.physical_gpu),
        "warmup": args.warmup,
        "reps": args.reps,
        "measure_energy": bool(args.measure_energy),
        "energy_iters": args.energy_iters if args.measure_energy else None,
        "energy_min_active_s": args.energy_min_active_s if args.measure_energy else None,
        "timing_method": "host_perf_counter_vm_call_plus_device_sync",
        "timing_includes_vm_launch_and_sync": True,
        "tvm_version": str(getattr(runtime.tvm, "__version__", "unknown")),
        "ap_report": None if args.ap_report is None else str(args.ap_report),
        "ap_report_sha256": None if args.ap_report is None else sha256_file(args.ap_report),
    }
    report = build_report(
        width=args.width,
        mode=args.mode,
        graph_io_dtype=args.graph_io_dtype,
        onnx=args.onnx,
        onnx_sha256=onnx_sha256,
        input_sha256=input_sha256,
        independent_run_id=args.independent_run_id,
        latency=latency,
        energy=energy,
        compile_summary=runtime.compile_summary,
        measurement_config=measurement_config,
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("/exdata/jichengzhi/V2Xverse_pyramid"))
    parser.add_argument("--width", required=True)
    parser.add_argument(
        "--mode",
        choices=["fp16", "int8_all", "mixed_top25_flops", "mixed_top50_flops"],
        required=True,
    )
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--ap-report", type=Path, default=None)
    parser.add_argument("--graph-io-dtype", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--tvm-gpu", type=int, default=0)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--measure-energy", action="store_true")
    parser.add_argument("--energy-iters", type=int, default=300)
    parser.add_argument("--energy-min-active-s", type=float, default=5.0)
    parser.add_argument("--independent-run-id", type=int, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()
    if args.warmup < 1 or args.reps < 1 or args.independent_run_id < 1:
        parser.error("--warmup, --reps, and --independent-run-id must be positive")
    if args.measure_energy and (args.energy_iters < 1 or args.energy_min_active_s <= 0.0):
        parser.error("--energy-iters and --energy-min-active-s must be positive when measuring energy")
    if not args.onnx.is_file():
        parser.error(f"ONNX file not found: {args.onnx}")
    if not args.input_npz.is_file():
        parser.error(f"input NPZ not found: {args.input_npz}")
    return args


def main() -> int:
    args = parse_args()
    if str(args.repo_root) not in sys.path:
        sys.path.insert(0, str(args.repo_root))
    os.chdir(args.repo_root)
    report = run_measurement(args)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out_json.with_suffix(args.out_json.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(args.out_json)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
