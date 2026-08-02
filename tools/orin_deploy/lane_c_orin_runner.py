#!/usr/bin/env python3
"""Build, audit, and measure Lane C TensorRT body engines on Jetson Orin."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

try:
    from .lane_c_contract import parse_tegrastats
except ImportError:
    from lane_c_contract import parse_tegrastats


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify_inspector_precisions(inspector: dict[str, Any]) -> dict[str, int]:
    """Classify each detailed inspector layer by its most specific datatype."""
    counts = {"int8": 0, "fp16": 0, "fp32": 0, "other": 0}
    for layer in inspector.get("Layers", []):
        text = json.dumps(layer, sort_keys=True).lower()
        if "int8" in text:
            counts["int8"] += 1
        elif "half" in text or "fp16" in text:
            counts["fp16"] += 1
        elif "float" in text or "fp32" in text:
            counts["fp32"] += 1
        else:
            counts["other"] += 1
    return counts


def validate_repeat_contract(
    *,
    warmup_count: int,
    warmup_seconds: float,
    inference_count: int,
    measurement_seconds: float,
    power_sample_count: int,
) -> None:
    requirements = {
        "warmup_count": (warmup_count, 200),
        "warmup_seconds": (warmup_seconds, 10.0),
        "inference_count": (inference_count, 500),
        "measurement_seconds": (measurement_seconds, 10.0),
        "power_sample_count": (power_sample_count, 1),
    }
    failures = [
        f"{name}={actual} < {minimum}"
        for name, (actual, minimum) in requirements.items()
        if actual < minimum
    ]
    if failures:
        raise ValueError("repeat contract failed: " + "; ".join(failures))


def _runtime_modules():
    import tensorrt as trt
    import torch

    return trt, torch


def _make_calibrator(trt, torch, calibration_dir: Path, cache_path: Path):
    class FullBodyEntropyCalibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self):
            super().__init__()
            self.files = sorted(Path(calibration_dir).glob("batch2_*.npy"))
            if not self.files:
                raise ValueError(f"no batch2 calibration tensors in {calibration_dir}")
            self.index = 0
            self.device_tensors: dict[str, Any] = {}

        def get_batch_size(self):
            return 2

        def get_batch(self, names):
            if self.index >= len(self.files):
                return None
            spatial = np.load(self.files[self.index], allow_pickle=False).astype(
                np.float32, copy=False
            )
            identity = np.eye(2, 3, dtype=np.float32)
            t_ego = np.repeat(identity[None, ...], 2, axis=0)
            arrays = {
                "spatial_features": spatial,
                "t_ego": t_ego,
            }
            pointers = []
            self.device_tensors = {}
            for name in names:
                if name not in arrays:
                    raise KeyError(f"unsupported calibration input: {name}")
                tensor = torch.from_numpy(np.ascontiguousarray(arrays[name])).cuda()
                self.device_tensors[name] = tensor
                pointers.append(int(tensor.data_ptr()))
            self.index += 1
            return pointers

        def read_calibration_cache(self):
            return cache_path.read_bytes() if cache_path.exists() else None

        def write_calibration_cache(self, cache):
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(cache)

    return FullBodyEntropyCalibrator()


def _load_engine(engine_path: Path):
    trt, _ = _runtime_modules()
    logger = trt.Logger(trt.Logger.ERROR)
    engine = trt.Runtime(logger).deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise RuntimeError(f"could not deserialize engine: {engine_path}")
    return trt, engine


def _inspect_engine(trt, engine) -> tuple[dict[str, Any], dict[str, int]]:
    raw = engine.create_engine_inspector().get_engine_information(
        trt.LayerInformationFormat.JSON
    )
    parsed = json.loads(raw)
    if isinstance(parsed, list):
        parsed = {"Layers": parsed}
    if not isinstance(parsed, dict):
        raise ValueError("unexpected TensorRT inspector JSON")
    return parsed, classify_inspector_precisions(parsed)


def build_engine(args: argparse.Namespace) -> int:
    trt, torch = _runtime_modules()
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(args.onnx.read_bytes()):
        errors = [str(parser.get_error(index)) for index in range(parser.num_errors)]
        raise RuntimeError("ONNX parse failed: " + " | ".join(errors))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    if hasattr(trt, "ProfilingVerbosity"):
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    flags: list[str] = []
    calibrator = None
    if args.precision == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16)
        flags.extend(["int8", "fp16"])
        calibrator = _make_calibrator(
            trt, torch, args.calibration_dir, args.calibration_cache
        )
        config.int8_calibrator = calibrator
    started = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT build returned no serialized engine")
    args.engine.parent.mkdir(parents=True, exist_ok=True)
    args.engine.write_bytes(bytes(serialized))
    build_seconds = time.time() - started
    _, engine = _load_engine(args.engine)
    inspector, precision_counts = _inspect_engine(trt, engine)
    args.inspector_json.write_text(
        json.dumps(inspector, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = {
        "schema_version": "lane_c_orin_engine_build_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "onnx_path": str(args.onnx),
        "onnx_sha256": sha256_file(args.onnx),
        "engine_path": str(args.engine),
        "engine_sha256": sha256_file(args.engine),
        "engine_bytes": args.engine.stat().st_size,
        "precision": args.precision,
        "builder_flags": flags,
        "build_seconds": build_seconds,
        "tensorrt_version": trt.__version__,
        "calibration_cache_path": (
            str(args.calibration_cache) if args.precision == "int8" else None
        ),
        "calibration_cache_sha256": (
            sha256_file(args.calibration_cache)
            if args.precision == "int8" and args.calibration_cache.exists()
            else None
        ),
        "inspector_json_path": str(args.inspector_json),
        "inspector_json_sha256": sha256_file(args.inspector_json),
        "layer_precision_counts": precision_counts,
    }
    args.artifact_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _prepare_execution(engine, spatial: np.ndarray, t_ego: np.ndarray):
    _, torch = _runtime_modules()
    context = engine.create_execution_context()
    names = [engine.get_tensor_name(index) for index in range(engine.num_io_tensors)]
    tensors: dict[str, Any] = {}
    output_names: list[str] = []
    arrays = {
        "spatial_features": spatial.astype(np.float32, copy=False),
        "t_ego": t_ego.astype(np.float32, copy=False),
    }
    for name in names:
        is_input = engine.get_tensor_mode(name).name == "INPUT"
        if is_input:
            if name not in arrays:
                raise KeyError(f"unsupported engine input: {name}")
            array = np.ascontiguousarray(arrays[name])
            context.set_input_shape(name, array.shape)
            tensor = torch.from_numpy(array).cuda()
        else:
            output_names.append(name)
            shape = tuple(context.get_tensor_shape(name))
            tensor = torch.empty(shape, dtype=torch.float32, device="cuda")
        tensors[name] = tensor
        context.set_tensor_address(name, int(tensor.data_ptr()))
    stream = torch.cuda.Stream()

    def execute_once(timed: bool) -> float | None:
        start = torch.cuda.Event(enable_timing=True) if timed else None
        end = torch.cuda.Event(enable_timing=True) if timed else None
        with torch.cuda.stream(stream):
            if start is not None:
                start.record(stream)
            if not context.execute_async_v3(stream.cuda_stream):
                raise RuntimeError("TensorRT execute_async_v3 returned false")
            if end is not None:
                end.record(stream)
        stream.synchronize()
        return float(start.elapsed_time(end)) if start is not None else None

    return execute_once, tensors, output_names


def run_numerical(args: argparse.Namespace) -> int:
    _, engine = _load_engine(args.engine)
    spatial_batches = np.load(args.inputs_npy, allow_pickle=False)
    t_ego_batches = np.load(args.tego_npy, allow_pickle=False)
    if spatial_batches.shape[0] != t_ego_batches.shape[0]:
        raise ValueError("numerical input batch counts differ")
    outputs: dict[str, list[np.ndarray]] = {}
    for index in range(spatial_batches.shape[0]):
        execute, tensors, output_names = _prepare_execution(
            engine, spatial_batches[index], t_ego_batches[index]
        )
        execute(False)
        for name in output_names:
            outputs.setdefault(name, []).append(tensors[name].detach().cpu().numpy())
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output_npz, **{name: np.stack(values) for name, values in outputs.items()})
    report = {
        "schema_version": "lane_c_orin_numerical_output_v1",
        "engine_sha256": sha256_file(args.engine),
        "inputs_sha256": sha256_file(args.inputs_npy),
        "tego_sha256": sha256_file(args.tego_npy),
        "output_npz": str(args.output_npz),
        "output_sha256": sha256_file(args.output_npz),
        "output_shapes": {
            name: list(np.stack(values).shape) for name, values in outputs.items()
        },
        "all_finite": all(np.isfinite(np.stack(values)).all() for values in outputs.values()),
        "execution_provider": "tensorrt_engine",
        "external_fallback_count": 0,
    }
    args.report_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _scrubbed_child_env() -> dict[str, str]:
    child_env = dict(os.environ)
    child_env.pop("ORIN_SUDO_PW", None)
    return child_env


def _start_tegrastats(log_path: Path):
    password = os.environ.get("ORIN_SUDO_PW", "")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        ["sudo", "-S", "-p", "", "tegrastats", "--interval", "50"],
        stdin=subprocess.PIPE,
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
        env=_scrubbed_child_env(),
    )
    if password and process.stdin is not None:
        process.stdin.write(password + "\n")
        process.stdin.flush()
    return process, handle


def _stop_tegrastats(process, handle) -> None:
    cleanup_error = None
    try:
        password = os.environ.get("ORIN_SUDO_PW", "")
        sudo_kwargs = {
            "input": password + "\n",
            "text": True,
            "check": True,
            "timeout": 5,
            "env": _scrubbed_child_env(),
        }
        try:
            subprocess.run(
                ["sudo", "-S", "-p", "", "tegrastats", "--stop"],
                **sudo_kwargs,
            )
        except (OSError, subprocess.SubprocessError):
            try:
                subprocess.run(
                    ["sudo", "-S", "-p", "", "pkill", "-x", "tegrastats"],
                    **sudo_kwargs,
                )
            except (OSError, subprocess.SubprocessError) as error:
                cleanup_error = error
    finally:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                process.kill()
            except (OSError, PermissionError):
                pass
        handle.close()
    if cleanup_error is not None:
        raise RuntimeError("failed to stop tegrastats with stop and pkill") from cleanup_error


def measure_repeat(args: argparse.Namespace) -> int:
    _, engine = _load_engine(args.engine)
    spatial = np.load(args.inputs_npy, allow_pickle=False)[0]
    t_ego = np.load(args.tego_npy, allow_pickle=False)[0]
    execute, _, _ = _prepare_execution(engine, spatial, t_ego)
    warmup_started = time.perf_counter()
    warmup_count = 0
    while warmup_count < 200 or time.perf_counter() - warmup_started < 10.0:
        execute(False)
        warmup_count += 1
    warmup_seconds = time.perf_counter() - warmup_started

    tegrastats_process, tegrastats_handle = _start_tegrastats(args.tegrastats_log)
    time.sleep(0.25)
    latencies_ms: list[float] = []
    measurement_started = time.perf_counter()
    try:
        while len(latencies_ms) < 500 or time.perf_counter() - measurement_started < 10.0:
            latency = execute(True)
            if latency is None:
                raise RuntimeError("timed execution did not return a latency")
            latencies_ms.append(latency)
    finally:
        measurement_seconds = time.perf_counter() - measurement_started
        _stop_tegrastats(tegrastats_process, tegrastats_handle)
    raw_tegrastats = args.tegrastats_log.read_text(encoding="utf-8")
    power = parse_tegrastats(raw_tegrastats)
    validate_repeat_contract(
        warmup_count=warmup_count,
        warmup_seconds=warmup_seconds,
        inference_count=len(latencies_ms),
        measurement_seconds=measurement_seconds,
        power_sample_count=power["sample_count"],
    )
    throughput = len(latencies_ms) / measurement_seconds
    report = {
        "schema_version": "lane_c_orin_measurement_repeat_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "process_id": os.getpid(),
        "repeat_index": args.repeat_index,
        "engine_sha256": sha256_file(args.engine),
        "scope": "body_dense_engine_compute_no_data_transfers",
        "agent_batch": 2,
        "power_mode": "MODE_30W",
        "jetson_clocks": True,
        "warmup_count": warmup_count,
        "warmup_seconds": warmup_seconds,
        "inference_count": len(latencies_ms),
        "measurement_seconds": measurement_seconds,
        "median_ms": float(np.median(latencies_ms)),
        "p90_ms": float(np.percentile(latencies_ms, 90)),
        "p99_ms": float(np.percentile(latencies_ms, 99)),
        "mean_ms": float(np.mean(latencies_ms)),
        "throughput_qps": throughput,
        "latency_sample_cv_percent": float(
            np.std(latencies_ms) / np.mean(latencies_ms) * 100.0
        ),
        "power_sample_count": power["sample_count"],
        "vin_sys_5v0_mean_w": power["vin_sys_5v0_mean_w"],
        "vdd_gpu_soc_mean_w": power["vdd_gpu_soc_mean_w"],
        "energy_per_frame_j": power["vin_sys_5v0_mean_w"] / throughput,
        "tegrastats_log": str(args.tegrastats_log),
        "tegrastats_sha256": sha256_file(args.tegrastats_log),
    }
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def validate_output_paths(args: argparse.Namespace) -> None:
    """Reject writes outside the operator-declared Lane C artifact root."""
    output_fields = {
        "build": ("engine", "inspector_json", "artifact_json", "calibration_cache"),
        "numerical": ("output_npz", "report_json"),
        "measure-repeat": ("tegrastats_log", "output_json"),
    }
    root = args.artifact_root.resolve()
    for field in output_fields[args.command]:
        path = getattr(args, field, None)
        if path is None:
            continue
        try:
            path.resolve().relative_to(root)
        except ValueError as error:
            raise ValueError(f"{field} is outside artifact root {root}: {path}") from error


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build")
    build.add_argument("--artifact-root", type=_path, required=True)
    build.add_argument("--onnx", type=_path, required=True)
    build.add_argument("--engine", type=_path, required=True)
    build.add_argument("--precision", choices=("fp32", "int8"), required=True)
    build.add_argument("--calibration-dir", type=_path)
    build.add_argument("--calibration-cache", type=_path)
    build.add_argument("--inspector-json", type=_path, required=True)
    build.add_argument("--artifact-json", type=_path, required=True)
    build.set_defaults(handler=build_engine)

    numerical = subparsers.add_parser("numerical")
    numerical.add_argument("--artifact-root", type=_path, required=True)
    numerical.add_argument("--engine", type=_path, required=True)
    numerical.add_argument("--inputs-npy", type=_path, required=True)
    numerical.add_argument("--tego-npy", type=_path, required=True)
    numerical.add_argument("--output-npz", type=_path, required=True)
    numerical.add_argument("--report-json", type=_path, required=True)
    numerical.set_defaults(handler=run_numerical)

    measure = subparsers.add_parser("measure-repeat")
    measure.add_argument("--artifact-root", type=_path, required=True)
    measure.add_argument("--engine", type=_path, required=True)
    measure.add_argument("--inputs-npy", type=_path, required=True)
    measure.add_argument("--tego-npy", type=_path, required=True)
    measure.add_argument("--repeat-index", type=int, required=True)
    measure.add_argument("--tegrastats-log", type=_path, required=True)
    measure.add_argument("--output-json", type=_path, required=True)
    measure.set_defaults(handler=measure_repeat)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_output_paths(args)
    if args.command == "build" and args.precision == "int8":
        if args.calibration_dir is None or args.calibration_cache is None:
            raise ValueError("INT8 build requires calibration directory and cache")
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
