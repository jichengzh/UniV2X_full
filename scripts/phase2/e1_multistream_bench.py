"""E1 — multi-stream concurrency bench on 4090 (A4 / A7 cross-frame pipeline).

Goal: fill the long-standing "0 latency 实测" gap for A4 multi-stream by
真测 throughput (FPS) AND single-frame latency separately, on the SAME engine
/ triplet as cudagraph_bench (T_baseline 064_128_256 fp16) so the口径 is
directly comparable.

Design (correctness notes):
  * GPU cross-frame pipelining is THROUGHPUT optimization, not single-frame
    latency. We must report both, and expect FPS up ~1.1-1.4x while
    single-frame latency stays ~1.0x (may even rise slightly).
  * Each stream gets its OWN execution context + IO buffers (TRT contexts are
    not safe to drive concurrently on multiple streams).
  * Throughput: enqueue `iters_per_stream` inferences on each of N streams
    (all in-flight), synchronize ALL streams, divide total inferences by wall
    time -> qps. baseline = N=1 (sequential submit).
  * Single-frame latency: under the concurrent load, time each individual
    enqueue with per-stream CUDA events and take the median across all frames
    on all streams. This is the latency a single frame observes while N streams
    contend for the SAME SMs.

Engine reused: models/p0_random_cache/engine_064_128_256_fp16.engine
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = REPO_ROOT / "models/p0_random_cache"
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

INPUT_SHAPE = (2, 64, 128, 256)
EXTRA_SHAPES = {"t_ego": (2, 2, 3)}

_TRT2TORCH = {
    trt.float32: torch.float32, trt.float16: torch.float16,
    trt.int32: torch.int32, trt.int8: torch.int8, trt.int64: torch.int64,
}


def make_ctx(engine):
    """Create an execution context + its own IO buffers, addresses bound."""
    context = engine.create_execution_context()
    n_io = engine.num_io_tensors
    names = [engine.get_tensor_name(i) for i in range(n_io)]
    input_names = [n for n in names
                   if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
    for name in input_names:
        context.set_input_shape(name, tuple(EXTRA_SHAPES.get(name, INPUT_SHAPE)))
    bufs = {}
    for name in names:
        dtype = engine.get_tensor_dtype(name)
        shape = tuple(context.get_tensor_shape(name))
        tdt = _TRT2TORCH.get(dtype, torch.float32)
        bufs[name] = torch.empty(shape, dtype=tdt, device="cuda")
        context.set_tensor_address(name, int(bufs[name].data_ptr()))
    torch.manual_seed(42)
    for nm in input_names:
        bufs[nm].copy_(torch.randn_like(bufs[nm].float()).to(bufs[nm].dtype))
    return context, bufs


def bench_streams(engine, n_streams: int, iters_per_stream: int = 400,
                  n_warmup: int = 100):
    """Run iters_per_stream inferences on each of n_streams concurrently.

    Returns dict with throughput_qps and single-frame latency stats.
    """
    ctxs = [make_ctx(engine) for _ in range(n_streams)]
    streams = [torch.cuda.Stream() for _ in range(n_streams)]

    # warmup every stream/context
    for (ctx, _), st in zip(ctxs, streams):
        with torch.cuda.stream(st):
            for _ in range(n_warmup):
                ctx.execute_async_v3(st.cuda_stream)
        st.synchronize()
    torch.cuda.synchronize()

    # ---- THROUGHPUT: all streams in-flight, single wall-clock window ----
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for k in range(iters_per_stream):
        for (ctx, _), st in zip(ctxs, streams):
            ctx.execute_async_v3(st.cuda_stream)
    for st in streams:
        st.synchronize()
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    total_inf = iters_per_stream * n_streams
    qps = total_inf / wall

    # ---- SINGLE-FRAME LATENCY under concurrent load ----
    # Each stream times its own per-inference events while all N stay busy.
    n_lat = 200
    ev_start = [[torch.cuda.Event(enable_timing=True) for _ in range(n_lat)]
                for _ in range(n_streams)]
    ev_end = [[torch.cuda.Event(enable_timing=True) for _ in range(n_lat)]
              for _ in range(n_streams)]
    for k in range(n_lat):
        for si, ((ctx, _), st) in enumerate(zip(ctxs, streams)):
            ev_start[si][k].record(st)
            ctx.execute_async_v3(st.cuda_stream)
            ev_end[si][k].record(st)
    for st in streams:
        st.synchronize()
    torch.cuda.synchronize()
    lat = []
    for si in range(n_streams):
        for k in range(n_lat):
            lat.append(ev_start[si][k].elapsed_time(ev_end[si][k]))
    lat = np.array(lat, dtype=np.float64)

    return {
        "streams": n_streams,
        "throughput_qps": qps,
        "wall_s": wall,
        "total_inf": total_inf,
        "lat_mean_ms": float(lat.mean()),
        "lat_p50_ms": float(np.percentile(lat, 50)),
        "lat_p99_ms": float(np.percentile(lat, 99)),
        "lat_std_ms": float(lat.std()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine",
                    default="models/p0_random_cache/engine_064_128_256_fp16.engine")
    ap.add_argument("--triplet", default="T_baseline")
    ap.add_argument("--triplet_sig", default="064_128_256")
    ap.add_argument("--precision", default="fp16")
    ap.add_argument("--streams", default="1,2,4")
    ap.add_argument("--iters_per_stream", type=int, default=400)
    ap.add_argument("--output", default="results/E1_multistream_4090.csv")
    ap.add_argument("--gpu", default="6")
    ap.add_argument("--ts_note", default="")
    args = ap.parse_args()

    engine_path = REPO_ROOT / args.engine
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    stream_list = [int(s) for s in args.streams.split(",")]
    rows = []
    base = None
    for ns in stream_list:
        print(f"[streams={ns}] benching ...")
        r = bench_streams(engine, ns, args.iters_per_stream)
        if base is None:
            base = r
        r["fps_speedup_vs_1"] = r["throughput_qps"] / base["throughput_qps"]
        r["lat_ratio_vs_1"] = r["lat_p50_ms"] / base["lat_p50_ms"]
        print(f"  qps={r['throughput_qps']:.1f} "
              f"fps_x={r['fps_speedup_vs_1']:.3f}  "
              f"lat_p50={r['lat_p50_ms']:.4f}ms "
              f"lat_x={r['lat_ratio_vs_1']:.3f}")
        rows.append({
            "engine": Path(args.engine).name,
            "triplet": args.triplet,
            "precision": args.precision,
            "streams": ns,
            "throughput_qps": round(r["throughput_qps"], 2),
            "latency_p50_ms": round(r["lat_p50_ms"], 4),
            "fps_speedup_vs_1": round(r["fps_speedup_vs_1"], 3),
            "lat_ratio_vs_1": round(r["lat_ratio_vs_1"], 3),
            "gpu": args.gpu,
            "ts_note": args.ts_note,
        })

    import pandas as pd
    df = pd.DataFrame(rows)
    out = REPO_ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nwrote {len(df)} rows -> {out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
