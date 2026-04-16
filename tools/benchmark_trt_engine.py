"""
benchmark_trt_engine.py — 测量 TRT engine 的纯推理时间 (CUDA Event)

用法:
    python tools/benchmark_trt_engine.py \\
        --engine trt_engines/univ2x_ego_bev_pruned_60_50_fp32.trt \\
        --plugin plugins/build/libuniv2x_plugins.so \\
        --n-warmup 20 --n-runs 50

仅测 engine.execute_v2/execute_async_v3, 不含数据加载、不含预/后处理。
输出: mean / std / p50 / p90 延迟 (ms), 以及所有输入/输出 shape+dtype.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import statistics
import sys
import time
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark TRT engine latency")
    p.add_argument("--engine", required=True, help="TRT engine file path")
    p.add_argument("--plugin", default="plugins/build/libuniv2x_plugins.so")
    p.add_argument("--n-warmup", type=int, default=20)
    p.add_argument("--n-runs", type=int, default=50)
    p.add_argument("--output", default=None,
                   help="Optional: save JSON report")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    import numpy as np
    import torch
    import tensorrt as trt

    # Load plugin
    if os.path.exists(args.plugin):
        ctypes.CDLL(os.path.abspath(args.plugin), mode=ctypes.RTLD_GLOBAL)
        print(f"[plugin] loaded {args.plugin}")
    else:
        print(f"[plugin] WARN: {args.plugin} missing — custom ops may fail")

    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, "")

    # Load engine
    with open(args.engine, "rb") as f:
        engine_bytes = f.read()
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    if engine is None:
        print(f"[engine] FAILED to deserialize {args.engine}")
        return 1

    context = engine.create_execution_context()
    print(f"[engine] loaded {args.engine} ({os.path.getsize(args.engine)/1e6:.2f} MB)")

    # Inspect I/O
    input_specs: List[Dict] = []
    output_specs: List[Dict] = []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = list(engine.get_tensor_shape(name))
        dtype = engine.get_tensor_dtype(name)
        mode = engine.get_tensor_mode(name)
        info = {"name": name, "shape": shape, "dtype": str(dtype), "mode": str(mode)}
        if mode == trt.TensorIOMode.INPUT:
            input_specs.append(info)
        else:
            output_specs.append(info)

    print(f"[engine] {len(input_specs)} inputs, {len(output_specs)} outputs")
    for info in input_specs:
        print(f"  [IN]  {info['name']:<20} {info['shape']} {info['dtype']}")
    for info in output_specs:
        print(f"  [OUT] {info['name']:<20} {info['shape']} {info['dtype']}")

    # Allocate GPU tensors (with explicit dtype/shape)
    def _trt_to_torch_dtype(tdt):
        mapping = {
            trt.DataType.FLOAT: torch.float32,
            trt.DataType.HALF: torch.float16,
            trt.DataType.INT8: torch.int8,
            trt.DataType.INT32: torch.int32,
            trt.DataType.INT64: torch.int64,
            trt.DataType.BOOL: torch.bool,
        }
        return mapping.get(tdt, torch.float32)

    tensors: Dict[str, torch.Tensor] = {}
    for info in input_specs + output_specs:
        dtype = _trt_to_torch_dtype(engine.get_tensor_dtype(info["name"]))
        shape = tuple(info["shape"])
        # Replace -1 with 1 (dynamic dim → batch 1)
        shape = tuple(1 if d < 0 else d for d in shape)
        t = torch.zeros(shape, dtype=dtype, device="cuda")
        tensors[info["name"]] = t
        context.set_tensor_address(info["name"], t.data_ptr())

    # CUDA stream
    stream = torch.cuda.Stream()

    def _run_once() -> float:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        ok = context.execute_async_v3(stream_handle=stream.cuda_stream)
        end.record(stream)
        stream.synchronize()
        if not ok:
            raise RuntimeError("TRT execute_async_v3 failed")
        return start.elapsed_time(end)  # ms

    # Warmup
    print(f"[bench] warmup x{args.n_warmup} ...")
    for _ in range(args.n_warmup):
        _run_once()

    # Timed runs
    print(f"[bench] measuring x{args.n_runs} ...")
    samples = []
    for i in range(args.n_runs):
        t = _run_once()
        samples.append(t)
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{args.n_runs}] last={t:.3f} ms")

    # Stats
    s_sorted = sorted(samples)
    mean = statistics.mean(samples)
    std = statistics.stdev(samples) if len(samples) > 1 else 0.0
    p50 = s_sorted[len(s_sorted) // 2]
    p90 = s_sorted[int(len(s_sorted) * 0.9)] if len(s_sorted) > 10 else s_sorted[-1]

    print(f"\n{'='*50}")
    print(f"{'mean':<10}: {mean:.3f} ms")
    print(f"{'std':<10}: {std:.3f} ms")
    print(f"{'p50':<10}: {p50:.3f} ms")
    print(f"{'p90':<10}: {p90:.3f} ms")
    print(f"{'min':<10}: {s_sorted[0]:.3f} ms")
    print(f"{'max':<10}: {s_sorted[-1]:.3f} ms")
    print(f"{'n':<10}: {len(samples)}")
    print(f"{'='*50}")

    # Save
    if args.output:
        report = {
            "engine": args.engine,
            "engine_size_mb": round(os.path.getsize(args.engine) / 1e6, 3),
            "inputs": input_specs,
            "outputs": output_specs,
            "n_warmup": args.n_warmup,
            "n_runs": args.n_runs,
            "latency_ms": {
                "mean": round(mean, 3),
                "std": round(std, 3),
                "p50": round(p50, 3),
                "p90": round(p90, 3),
                "min": round(s_sorted[0], 3),
                "max": round(s_sorted[-1], 3),
            },
            "samples": [round(x, 3) for x in samples],
        }
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n[out] report saved -> {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
