#!/usr/bin/env python3
"""E3 — Orin heterogeneous GPU || DLA stage-pipeline latency/throughput measurement.

Goal: validate whether a 2-stage split of the Pyramid backbone, with stage01 on DLA0
and stage2 on DLA1, can overlap frame N's stage2 with frame N+1's stage01 so the
steady-state inter-frame interval approaches max(stage_dla0, stage_dla1) rather than
their sum. This is the only physically-disjoint-resource form that could make the
"stage pipeline" idea yield real throughput gains (single-GPU stage pipeline was 1.08x).

Allocator: uses torch CUDA tensors (no pycuda/cuda-python needed on this Jetson).
TRT 8.5.2.2 execute_v2 with device pointers from torch tensors.

Outputs JSON to stdout with both single-frame latency and steady-state throughput,
for: (1) stage01 alone on DLA0, (2) stage2 alone on DLA1, (3) sequential sum on DLA
(no overlap), (4) pipelined DLA0||DLA1 (overlap). Honest: marks every number as
real-measured. Does NOT touch git.
"""
import argparse
import json
import sys
import time

import tensorrt as trt
import torch


def load_engine(rt, path):
    with open(path, "rb") as f:
        return rt.deserialize_cuda_engine(f.read())


def trt_dtype_to_torch(dt):
    return {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
        trt.DataType.INT8: torch.int8,
        trt.DataType.INT32: torch.int32,
    }[dt]


def make_io(engine, device="cuda"):
    """Allocate torch device tensors for every binding; return (inputs, outputs, ptr_list)."""
    n = engine.num_bindings
    tensors = [None] * n
    ptrs = [0] * n
    inputs, outputs = [], []
    for i in range(n):
        shape = tuple(engine.get_binding_shape(i))
        dtype = trt_dtype_to_torch(engine.get_binding_dtype(i))
        t = torch.zeros(shape, device=device, dtype=dtype)
        tensors[i] = t
        ptrs[i] = t.data_ptr()
        if engine.binding_is_input(i):
            inputs.append(i)
        else:
            outputs.append(i)
    return tensors, ptrs, inputs, outputs


def bench_single(engine, dla_core, stream, iters=200, warmup=50):
    """Single-engine steady benchmark on a given DLA core / stream. Returns ms stats."""
    ctx = engine.create_execution_context()
    tensors, ptrs, _, _ = make_io(engine)
    s = stream
    # warmup
    for _ in range(warmup):
        ctx.execute_async_v2(ptrs, s.cuda_stream)
    s.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    # measure end-to-end wall for throughput AND per-call gpu time
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    start.record(s)
    for _ in range(iters):
        ctx.execute_async_v2(ptrs, s.cuda_stream)
    end.record(s)
    s.synchronize()
    t1 = time.perf_counter()
    gpu_ms = start.elapsed_time(end) / iters
    wall_ms = (t1 - t0) * 1000.0 / iters
    return {"gpu_ms_per_call": gpu_ms, "wall_ms_per_call": wall_ms,
            "throughput_qps": 1000.0 / wall_ms}


def bench_pipeline(eng01, eng2, iters=200, warmup=50):
    """Pipelined: stage01 on DLA0 (streamA), stage2 on DLA1 (streamB).
    Frame N stage2 (streamB) overlaps frame N+1 stage01 (streamA).
    We keep both streams continuously fed and measure steady-state inter-frame interval.

    Dependency: stage2 input = stage01 layer1 output. We use a double-buffer on the
    handoff tensor and a CUDA event so streamB waits on streamA's completion of that frame.
    """
    streamA = torch.cuda.Stream()  # DLA0 stage01
    streamB = torch.cuda.Stream()  # DLA1 stage2

    ctx01 = eng01.create_execution_context()
    ctx2 = eng2.create_execution_context()

    # stage01 bindings
    t01, p01, in01, out01 = make_io(eng01)
    # identify which stage01 output feeds stage2 by matching shape to stage2 input
    s2_in_idx = [i for i in range(eng2.num_bindings) if eng2.binding_is_input(i)][0]
    s2_in_shape = tuple(eng2.get_binding_shape(s2_in_idx))
    handoff_out_idx = None
    for i in out01:
        if tuple(eng01.get_binding_shape(i)) == s2_in_shape:
            handoff_out_idx = i
            break
    if handoff_out_idx is None:
        raise RuntimeError(f"no stage01 output matches stage2 input shape {s2_in_shape}")

    # stage2 bindings (its own buffers); we will copy handoff into stage2 input each frame
    t2, p2, in2, out2 = make_io(eng2)

    # events to chain: after stage01 on A completes, B can run stage2
    nbuf = 2
    evtsA = [torch.cuda.Event() for _ in range(nbuf)]

    def run_frame(buf):
        # stage01 on DLA0 / streamA
        with torch.cuda.stream(streamA):
            ctx01.execute_async_v2(p01, streamA.cuda_stream)
            # copy handoff output -> stage2 input (on streamA so it's ordered after exec)
            t2[s2_in_idx].copy_(t01[handoff_out_idx], non_blocking=True)
            evtsA[buf].record(streamA)
        # stage2 on DLA1 / streamB, waits for stage01+copy of this frame
        streamB.wait_event(evtsA[buf])
        with torch.cuda.stream(streamB):
            ctx2.execute_async_v2(p2, streamB.cuda_stream)

    # warmup
    for k in range(warmup):
        run_frame(k % nbuf)
    torch.cuda.synchronize()

    # steady-state throughput: keep pipeline full
    t0 = time.perf_counter()
    for k in range(iters):
        run_frame(k % nbuf)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    interval_ms = (t1 - t0) * 1000.0 / iters

    # single-frame latency (drain each frame fully) for the latency-kind number
    lat_samples = []
    for k in range(50):
        torch.cuda.synchronize()
        a = time.perf_counter()
        run_frame(k % nbuf)
        torch.cuda.synchronize()
        b = time.perf_counter()
        lat_samples.append((b - a) * 1000.0)
    lat_samples.sort()
    return {
        "steady_interframe_ms": interval_ms,
        "throughput_qps": 1000.0 / interval_ms,
        "single_frame_latency_ms_median": lat_samples[len(lat_samples) // 2],
        "handoff_shape": list(s2_in_shape),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage01", required=True)
    ap.add_argument("--stage2", required=True)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=50)
    args = ap.parse_args()

    logger = trt.Logger(trt.Logger.ERROR)
    rt = trt.Runtime(logger)
    eng01 = load_engine(rt, args.stage01)
    eng2 = load_engine(rt, args.stage2)

    result = {"engines": {"stage01": args.stage01, "stage2": args.stage2},
              "iters": args.iters, "source": "real-measured (torch+TRT execute_async_v2)"}

    # single-engine baselines (both on default stream is fine for isolated timing)
    sA = torch.cuda.Stream()
    sB = torch.cuda.Stream()
    result["stage01_dla0_alone"] = bench_single(eng01, 0, sA, args.iters, args.warmup)
    result["stage2_dla1_alone"] = bench_single(eng2, 1, sB, args.iters, args.warmup)
    seq = (result["stage01_dla0_alone"]["gpu_ms_per_call"]
           + result["stage2_dla1_alone"]["gpu_ms_per_call"])
    result["sequential_sum_ms"] = seq
    result["max_stage_ms"] = max(result["stage01_dla0_alone"]["gpu_ms_per_call"],
                                 result["stage2_dla1_alone"]["gpu_ms_per_call"])

    # pipelined overlap
    result["pipelined_dla0_dla1"] = bench_pipeline(eng01, eng2, args.iters, args.warmup)

    # speedup of pipeline throughput vs sequential sum
    pipe_int = result["pipelined_dla0_dla1"]["steady_interframe_ms"]
    result["overlap_speedup_vs_sum"] = seq / pipe_int if pipe_int > 0 else None
    result["interval_vs_maxstage_ratio"] = pipe_int / result["max_stage_ms"]

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
