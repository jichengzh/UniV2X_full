"""Part B-4090.1 — CUDA Graph on/off bench on 4090.

Reuse already-built engines from p0_random_cache, bench each engine with
CUDA Graph capture + replay vs vanilla execute_async_v3 + event timing.

Triplets (same as Part A):
    T_baseline (64,128,256) — engine_064_128_256_{fp16,int8}.engine
    T_prune50  (32,64,136)  — engine_032_064_136_{fp16,int8}.engine
    T_prune75  (16,32,64)   — engine_016_032_064_{fp16,int8}.engine

For each (triplet, precision) pair, run two bench passes:
    CG_OFF: standard execute_async_v3 loop (same as m4_8_trt_build_bench)
    CG_ON:  capture once, replay n_measure times

Output: data/cudagraph_bench.parquet
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = REPO_ROOT / "models/p0_random_cache"

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

TRIPLETS = [
    ("T_baseline", 64, 128, 256),
    ("T_prune50",  32,  64, 136),
    ("T_prune75",  16,  32,  64),
]


def alloc_io(engine, context, input_shape=(2, 64, 128, 256),
              extra_shapes=None):
    """Return dict of name -> torch tensor, set tensor addresses."""
    extra_shapes = extra_shapes or {"t_ego": (2, 2, 3)}
    n_io = engine.num_io_tensors
    names = [engine.get_tensor_name(i) for i in range(n_io)]
    input_names = [n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
    bufs = {}
    for name in input_names:
        shape = tuple(extra_shapes.get(name, input_shape))
        context.set_input_shape(name, shape)

    for name in names:
        dtype = engine.get_tensor_dtype(name)
        shape = tuple(context.get_tensor_shape(name))
        torch_dtype = {trt.float32: torch.float32, trt.float16: torch.float16,
                        trt.int32: torch.int32, trt.int8: torch.int8,
                        trt.int64: torch.int64}.get(dtype, torch.float32)
        bufs[name] = torch.empty(shape, dtype=torch_dtype, device="cuda")
        context.set_tensor_address(name, int(bufs[name].data_ptr()))

    torch.manual_seed(42)
    for nm in input_names:
        bufs[nm].copy_(torch.randn_like(bufs[nm].float()).to(bufs[nm].dtype))
    return bufs


def bench_normal(context, stream, n_warmup=100, n_measure=200):
    """Standard execute_async_v3 + cuda event timing (same as m4_8)."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = np.empty(n_measure, dtype=np.float64)
    with torch.cuda.stream(stream):
        for _ in range(n_warmup):
            context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        for k in range(n_measure):
            start.record(stream)
            context.execute_async_v3(stream.cuda_stream)
            end.record(stream)
            end.synchronize()
            times[k] = start.elapsed_time(end)
    return times


def bench_cuda_graph(context, stream, n_warmup=100, n_measure=200):
    """CUDA Graph capture + replay using torch.cuda.graphs.

    Capture one inference (after warmup so kernels are JITed),
    then replay n_measure times measuring per-replay time.
    """
    with torch.cuda.stream(stream):
        for _ in range(n_warmup):
            context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=stream):
        context.execute_async_v3(stream.cuda_stream)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = np.empty(n_measure, dtype=np.float64)
    with torch.cuda.stream(stream):
        # warm graph replay
        for _ in range(50):
            g.replay()
        stream.synchronize()
        for k in range(n_measure):
            start.record(stream)
            g.replay()
            end.record(stream)
            end.synchronize()
            times[k] = start.elapsed_time(end)
    return times


def bench_engine(engine_path: Path, mode: str, n_warmup=100, n_measure=200):
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()
    bufs = alloc_io(engine, context)
    stream = torch.cuda.Stream()
    if mode == "off":
        times = bench_normal(context, stream, n_warmup, n_measure)
    elif mode == "on":
        times = bench_cuda_graph(context, stream, n_warmup, n_measure)
    else:
        raise ValueError(mode)
    return {
        "n_warmup": n_warmup, "n_measure": n_measure,
        "mean_ms": float(times.mean()), "std_ms": float(times.std()),
        "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
        "p99_ms": float(np.percentile(times, 99)),
        "min_ms": float(times.min()), "max_ms": float(times.max()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="data/cudagraph_bench.parquet")
    args = ap.parse_args()

    rows = []
    for tri_label, s0, s1, s2 in TRIPLETS:
        sig = f"{s0:03d}_{s1:03d}_{s2:03d}"
        for prec in ("fp16", "int8"):
            engine_path = CACHE_ROOT / f"engine_{sig}_{prec}.engine"
            if not engine_path.exists():
                print(f"MISSING: {engine_path.name}")
                continue
            for mode in ("off", "on"):
                print(f"[{tri_label} {sig} {prec} CG_{mode}] benching ...")
                try:
                    t0 = time.time()
                    stats = bench_engine(engine_path, mode)
                    elapsed = time.time() - t0
                except Exception as e:
                    print(f"  FAILED: {e}")
                    continue
                print(f"  mean={stats['mean_ms']:.3f} p50={stats['p50_ms']:.3f}"
                      f" p99={stats['p99_ms']:.3f} ({elapsed:.0f}s)")
                rows.append({
                    "triplet": tri_label, "triplet_sig": sig,
                    "stage0_planes": s0, "stage1_planes": s1, "stage2_planes": s2,
                    "precision": prec, "cuda_graph": mode,
                    "lat_p50_ms": stats["p50_ms"], "lat_p99_ms": stats["p99_ms"],
                    "lat_mean_ms": stats["mean_ms"], "lat_std_ms": stats["std_ms"],
                })

    import pandas as pd
    df = pd.DataFrame(rows)
    out = REPO_ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    df.to_csv(out.with_suffix(".csv"), index=False)
    print(f"\nwrote {len(df)} rows -> {out}")
    print(df[["triplet","precision","cuda_graph","lat_p50_ms","lat_p99_ms","lat_mean_ms"]].to_string(index=False))


if __name__ == "__main__":
    main()
