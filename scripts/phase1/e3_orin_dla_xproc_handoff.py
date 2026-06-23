#!/usr/bin/env python3
"""E3+ — Orin TRUE cross-process DLA0->DLA1 pipelined handoff.

Difference vs e3_orin_dla_pipeline.py (which measured two INDEPENDENT throughput
loops side by side = an upper-bound estimate, NO real frame data crossing):

This builds a REAL pipeline. A producer process runs stage01 on DLA0, copies the
layer1 handoff tensor (the actual stage2 input) into a POSIX shared-memory ring
buffer, and a consumer process running stage2 on DLA1 reads frame N's tensor out
of shared memory and runs stage2 on it. Frame N's stage2 (consumer/DLA1) overlaps
frame N+1's stage01 (producer/DLA0). Every byte of the intermediate feature map
physically crosses the process boundary through /dev/shm.

We measure:
  (a) true steady-state inter-frame interval (pipeline throughput) = how often a
      completed frame exits stage2 once the pipe is full.
  (b) true end-to-end single-frame latency = produce_start[N] -> consume_done[N],
      including the shm handoff (upload + memcpy + download) cost.
  (c) handoff transfer cost alone (device->pinned->shm->pinned->device).

Honest labeling: all reported numbers are real-measured wall clock on Orin.
TRT 8.5.2.2, torch CUDA tensors, execute_async_v2. DLA core fixed at deserialize
via runtime.DLA_core (engines were built with --useDLACore=0 / =1).

Does NOT touch git.
"""
import argparse
import json
import multiprocessing as mp
import struct
import sys
import time
from multiprocessing import shared_memory

import numpy as np

# torch / tensorrt imported lazily inside child procs (each needs its own CUDA ctx)

# ---- control-block layout (in a separate small shm) ----
# We use a single-producer / single-consumer ring with two 64-bit monotonic
# counters (produced, consumed) + per-slot float64 produce_start timestamp.
# Layout of control shm (little endian):
#   [0:8]   produced  (uint64)   number of frames the producer has published
#   [8:16]  consumed  (uint64)   number of frames the consumer has finished
#   [16:24] producer_done (uint64) flag, 1 when producer finished all frames
# Per-slot timestamps stored in a separate float64 array shm (n_slots entries):
#   ts_produce_start[slot] = perf_counter() when producer began frame for that slot

CTRL_PRODUCED = 0
CTRL_CONSUMED = 8
CTRL_PRODONE = 16
CTRL_SIZE = 24


def trt_dtype_to_torch(trt, dt):
    import torch
    return {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
        trt.DataType.INT8: torch.int8,
        trt.DataType.INT32: torch.int32,
    }[dt]


def deserialize(trt, path, dla_core):
    rt = trt.Runtime(trt.Logger(trt.Logger.ERROR))
    if dla_core is not None:
        rt.DLA_core = dla_core
    with open(path, "rb") as f:
        return rt, rt.deserialize_cuda_engine(f.read())


def find_handoff_output(eng01, eng2, trt):
    """Return (stage01_output_binding_idx, n_elements, np_dtype) for the tensor
    that feeds stage2's input (matched by shape)."""
    s2_in = [i for i in range(eng2.num_bindings) if eng2.binding_is_input(i)][0]
    s2_shape = tuple(eng2.get_binding_shape(s2_in))
    for i in range(eng01.num_bindings):
        if not eng01.binding_is_input(i) and tuple(eng01.get_binding_shape(i)) == s2_shape:
            return i, s2_in, s2_shape
    raise RuntimeError(f"no stage01 output matches stage2 input shape {s2_shape}")


def producer_proc(args, ctrl_name, ring_name, ts_name, ready_ev, go_ev):
    import torch
    import tensorrt as trt

    rt, eng = deserialize(trt, args.stage01, 0)  # DLA0
    ctx = eng.create_execution_context()

    # allocate device buffers for all bindings
    tensors, ptrs = [], []
    for i in range(eng.num_bindings):
        t = torch.zeros(tuple(eng.get_binding_shape(i)), device="cuda",
                        dtype=trt_dtype_to_torch(trt, eng.get_binding_dtype(i)))
        tensors.append(t)
        ptrs.append(t.data_ptr())

    eng2_rt, eng2 = deserialize(trt, args.stage2, None)  # just to introspect shape
    handoff_idx, _, handoff_shape = find_handoff_output(eng, eng2, trt)
    del eng2, eng2_rt
    n_elem = int(np.prod(handoff_shape))
    nbytes = n_elem * 2  # fp16

    # pinned host staging buffer for D->H copy of handoff
    pinned = torch.empty(handoff_shape, dtype=torch.float16, device="cpu").pin_memory()

    ctrl = shared_memory.SharedMemory(name=ctrl_name)
    ring = shared_memory.SharedMemory(name=ring_name)
    ts = shared_memory.SharedMemory(name=ts_name)
    ring_np = np.ndarray((args.slots, n_elem), dtype=np.float16, buffer=ring.buf)
    ts_np = np.ndarray((args.iters + args.warmup,), dtype=np.float64, buffer=ts.buf)

    stream = torch.cuda.Stream()

    def publish(v):
        struct.pack_into("<Q", ctrl.buf, CTRL_PRODUCED, v)

    def read_consumed():
        return struct.unpack_from("<Q", ctrl.buf, CTRL_CONSUMED)[0]

    total = args.iters + args.warmup
    ready_ev.set()
    go_ev.wait()

    for n in range(total):
        slot = n % args.slots
        # backpressure: don't overwrite a slot the consumer hasn't drained
        while n - read_consumed() >= args.slots:
            pass
        t_start = time.perf_counter()
        if n >= args.warmup:
            ts_np[n] = t_start
        with torch.cuda.stream(stream):
            ctx.execute_async_v2(ptrs, stream.cuda_stream)
            pinned.copy_(tensors[handoff_idx], non_blocking=True)
        stream.synchronize()
        # memcpy pinned -> shared ring slot (real cross-process byte transfer)
        ring_np[slot, :] = pinned.numpy().reshape(-1)
        publish(n + 1)

    struct.pack_into("<Q", ctrl.buf, CTRL_PRODONE, 1)
    ctrl.close(); ring.close(); ts.close()


def consumer_proc(args, ctrl_name, ring_name, ts_name, ready_ev, go_ev, result_q):
    import torch
    import tensorrt as trt

    rt, eng = deserialize(trt, args.stage2, 1)  # DLA1
    ctx = eng.create_execution_context()

    in_idx = [i for i in range(eng.num_bindings) if eng.binding_is_input(i)][0]
    in_shape = tuple(eng.get_binding_shape(in_idx))
    n_elem = int(np.prod(in_shape))

    tensors, ptrs = [], []
    for i in range(eng.num_bindings):
        t = torch.zeros(tuple(eng.get_binding_shape(i)), device="cuda",
                        dtype=trt_dtype_to_torch(trt, eng.get_binding_dtype(i)))
        tensors.append(t)
        ptrs.append(t.data_ptr())
    # pinned staging buffer for H->D upload of handoff
    pinned = torch.empty(in_shape, dtype=torch.float16, device="cpu").pin_memory()

    ctrl = shared_memory.SharedMemory(name=ctrl_name)
    ring = shared_memory.SharedMemory(name=ring_name)
    ts = shared_memory.SharedMemory(name=ts_name)
    ring_np = np.ndarray((args.slots, n_elem), dtype=np.float16, buffer=ring.buf)
    ts_produce = np.ndarray((args.iters + args.warmup,), dtype=np.float64, buffer=ts.buf)

    stream = torch.cuda.Stream()

    def read_produced():
        return struct.unpack_from("<Q", ctrl.buf, CTRL_PRODUCED)[0]

    def set_consumed(v):
        struct.pack_into("<Q", ctrl.buf, CTRL_CONSUMED, v)

    total = args.iters + args.warmup
    consume_done_ts = np.zeros(total, dtype=np.float64)
    e2e_lat = np.zeros(total, dtype=np.float64)

    ready_ev.set()
    go_ev.wait()

    n = 0
    pipeline_start = None
    while n < total:
        # wait for frame n to be available
        while read_produced() <= n:
            pass
        slot = n % args.slots
        # read from shared ring -> pinned host (real cross-process read)
        pinned.numpy().reshape(-1)[:] = ring_np[slot, :]
        with torch.cuda.stream(stream):
            tensors[in_idx].copy_(pinned, non_blocking=True)
            ctx.execute_async_v2(ptrs, stream.cuda_stream)
        stream.synchronize()
        t_done = time.perf_counter()
        consume_done_ts[n] = t_done
        if n >= args.warmup:
            e2e_lat[n] = (t_done - ts_produce[n]) * 1000.0
            if pipeline_start is None:
                pipeline_start = consume_done_ts[args.warmup]
        set_consumed(n + 1)
        n += 1

    # steady-state interframe interval over measured (post-warmup) frames
    meas = consume_done_ts[args.warmup:total]
    interframe_ms = (meas[-1] - meas[0]) * 1000.0 / (len(meas) - 1)
    e2e = e2e_lat[args.warmup:total]
    e2e_sorted = np.sort(e2e)
    result_q.put({
        "steady_interframe_ms": float(interframe_ms),
        "throughput_qps": float(1000.0 / interframe_ms),
        "e2e_latency_ms_median": float(np.median(e2e)),
        "e2e_latency_ms_mean": float(np.mean(e2e)),
        "e2e_latency_ms_p99": float(e2e_sorted[int(0.99 * len(e2e_sorted))]),
        "frames_measured": int(len(meas)),
    })
    ctrl.close(); ring.close(); ts.close()


def measure_handoff_cost(args):
    """Measure just the shm round-trip cost (D->pinned->shm->pinned->D) single proc."""
    import torch
    import tensorrt as trt
    rt, eng2 = deserialize(trt, args.stage2, None)
    in_idx = [i for i in range(eng2.num_bindings) if eng2.binding_is_input(i)][0]
    shape = tuple(eng2.get_binding_shape(in_idx))
    n_elem = int(np.prod(shape))
    dev = torch.randn(shape, device="cuda").half()
    pin = torch.empty(shape, dtype=torch.float16, device="cpu").pin_memory()
    shm = shared_memory.SharedMemory(create=True, size=n_elem * 2)
    arr = np.ndarray((n_elem,), dtype=np.float16, buffer=shm.buf)
    s = torch.cuda.Stream()
    # warmup
    for _ in range(20):
        with torch.cuda.stream(s):
            pin.copy_(dev, non_blocking=True)
        s.synchronize()
        arr[:] = pin.numpy().reshape(-1)
        pin.numpy().reshape(-1)[:] = arr
        with torch.cuda.stream(s):
            dev.copy_(pin, non_blocking=True)
        s.synchronize()
    N = 200
    t0 = time.perf_counter()
    for _ in range(N):
        with torch.cuda.stream(s):
            pin.copy_(dev, non_blocking=True)
        s.synchronize()
        arr[:] = pin.numpy().reshape(-1)
        pin.numpy().reshape(-1)[:] = arr
        with torch.cuda.stream(s):
            dev.copy_(pin, non_blocking=True)
        s.synchronize()
    t1 = time.perf_counter()
    shm.close(); shm.unlink()
    return {"handoff_roundtrip_ms": (t1 - t0) * 1000.0 / N,
            "handoff_bytes": n_elem * 2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage01", required=True)
    ap.add_argument("--stage2", required=True)
    ap.add_argument("--config", required=True, help="split tag for CSV, e.g. 016_032_064")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--slots", type=int, default=4)
    args = ap.parse_args()

    # handoff cost (single proc, before spawning pipeline)
    handoff = measure_handoff_cost(args)

    # figure handoff element count for shm sizing
    import tensorrt as trt
    rt, eng2 = deserialize(trt, args.stage2, None)
    in_idx = [i for i in range(eng2.num_bindings) if eng2.binding_is_input(i)][0]
    shape = tuple(eng2.get_binding_shape(in_idx))
    n_elem = int(np.prod(shape))
    del eng2, rt

    total = args.iters + args.warmup
    ctrl = shared_memory.SharedMemory(create=True, size=CTRL_SIZE)
    ctrl.buf[:CTRL_SIZE] = b"\x00" * CTRL_SIZE
    ring = shared_memory.SharedMemory(create=True, size=args.slots * n_elem * 2)
    ts = shared_memory.SharedMemory(create=True, size=total * 8)

    ctx = mp.get_context("spawn")
    ready_p = ctx.Event(); ready_c = ctx.Event(); go = ctx.Event()
    result_q = ctx.Queue()

    p = ctx.Process(target=producer_proc,
                    args=(args, ctrl.name, ring.name, ts.name, ready_p, go))
    c = ctx.Process(target=consumer_proc,
                    args=(args, ctrl.name, ring.name, ts.name, ready_c, go, result_q))
    p.start(); c.start()
    ready_p.wait(); ready_c.wait()
    go.set()  # release both simultaneously
    result = result_q.get()
    p.join(); c.join()

    ctrl.close(); ctrl.unlink()
    ring.close(); ring.unlink()
    ts.close(); ts.unlink()

    out = {
        "config": args.config,
        "engines": {"stage01_dla0": args.stage01, "stage2_dla1": args.stage2},
        "iters": args.iters, "warmup": args.warmup, "slots": args.slots,
        "handoff_bytes": handoff["handoff_bytes"],
        "handoff_roundtrip_ms": handoff["handoff_roundtrip_ms"],
        "pipeline": result,
        "source": "real-measured (xproc shm handoff, torch+TRT execute_async_v2)",
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
