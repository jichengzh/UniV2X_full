#!/usr/bin/env python3
"""Try a compressed TRT build using only a frozen base-graph timing cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import tensorrt as trt


LOGGER = trt.Logger(trt.Logger.WARNING)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def parse_network(builder: trt.Builder, onnx_path: Path):
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, LOGGER)
    if not parser.parse(onnx_path.read_bytes()):
        errors = [str(parser.get_error(index)) for index in range(parser.num_errors)]
        raise RuntimeError(f"ONNX parse failed for {onnx_path}: {errors}")
    return network


def base_timing_cache(onnx_path: Path, workspace_gb: int) -> tuple[bytes, float]:
    builder = trt.Builder(LOGGER)
    network = parse_network(builder, onnx_path)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    config.set_timing_cache(config.create_timing_cache(b""), ignore_mismatch=False)
    started = time.monotonic()
    serialized = builder.build_serialized_network(network, config)
    elapsed = time.monotonic() - started
    if serialized is None:
        raise RuntimeError("base engine build failed")
    return bytes(config.get_timing_cache().serialize()), elapsed


def compressed_build_from_frozen_cache(
    onnx_path: Path, cache_bytes: bytes, workspace_gb: int
) -> tuple[bool, float, str | None]:
    builder = trt.Builder(LOGGER)
    network = parse_network(builder, onnx_path)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    config.set_timing_cache(config.create_timing_cache(cache_bytes), ignore_mismatch=False)
    config.set_flag(trt.BuilderFlag.ERROR_ON_TIMING_CACHE_MISS)
    started = time.monotonic()
    try:
        serialized = builder.build_serialized_network(network, config)
        elapsed = time.monotonic() - started
        if serialized is None:
            return False, elapsed, "compressed_engine_build_returned_none_on_timing_cache_miss"
        return True, elapsed, None
    except Exception as exc:
        return False, time.monotonic() - started, f"{type(exc).__name__}:{exc}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-onnx", type=Path, required=True)
    parser.add_argument("--compressed-onnx", type=Path, required=True)
    parser.add_argument("--base-width", default="64,128,256")
    parser.add_argument("--compressed-width", default="48,96,192")
    parser.add_argument("--intended-q-mode", choices=["fp16", "int8"], required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--workspace-gb", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache-out", type=Path)
    parser.add_argument("--cache-in", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import pycuda.driver as cuda

    cuda.init()
    context = cuda.Device(args.gpu).make_context()
    try:
        if args.cache_in is not None:
            if not args.cache_in.is_file():
                raise FileNotFoundError(f"missing frozen timing cache: {args.cache_in}")
            cache = args.cache_in.read_bytes()
            base_build_s = 0.0
            cache_path = args.cache_in
        else:
            if args.cache_out is None:
                raise ValueError("--cache-out is required when --cache-in is omitted")
            cache, base_build_s = base_timing_cache(args.base_onnx, args.workspace_gb)
            args.cache_out.parent.mkdir(parents=True, exist_ok=True)
            args.cache_out.write_bytes(cache)
            cache_path = args.cache_out
        transferred, compressed_build_s, failure = compressed_build_from_frozen_cache(
            args.compressed_onnx, cache, args.workspace_gb
        )
    finally:
        context.pop()
    payload = {
        "schema_version": "stage6_trt_base_timing_cache_transfer_probe_v1",
        "trt_version": trt.__version__,
        "base_width": [int(value) for value in args.base_width.split(",")],
        "compressed_width": [int(value) for value in args.compressed_width.split(",")],
        "intended_q_mode": args.intended_q_mode,
        "q_dispatch_reached": False,
        "blocked_stage": "frozen_base_policy_transfer",
        "base_onnx": str(args.base_onnx),
        "compressed_onnx": str(args.compressed_onnx),
        "base_build_s": base_build_s,
        "compressed_build_s": compressed_build_s,
        "timing_cache_path": str(cache_path),
        "timing_cache_sha256": sha256_bytes(cache),
        "base_cache_reused": args.cache_in is not None,
        "error_on_timing_cache_miss": True,
        "full_transfer": transferred,
        "compressed_shape_retuned": False,
        "fallback_used": False,
        "terminal_status": "transferred_success" if transferred else "feasibility_failure",
        "failure_reason": failure,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
