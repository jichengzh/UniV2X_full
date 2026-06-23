"""Task #12 (revised) — collab2 REQUEST-LEVEL batch (RSU serving N vehicles).

Each "request" = one full collab2 inference (input [2,64,128,256], 2-agent fusion
intact). RSU spatial-dim batch = N independent collab2 requests concurrent. This
is the LEGITIMATE deployment throughput lever (NOT subnet batch, which is
withdrawn). Tests whether request-level concurrency DECOUPLES throughput from
1/latency, or whether the conv-dense collab2 model is already SM-saturated at
batch=1 (E1 multi-stream on subnet = 1.13× suggests saturation -> likely 2D).

Reports (data's early indicator): SM%/util @ batch=1 FIRST, then N∈{1,2,4,8}:
  - total throughput (req/s), decoupling = tput_N / tput_1
  - per-request latency p50 (queue/contention inflation = the real cost)
  - GPU util sampling.
口径: body_subnet_collab2 (real collab2 engine). throughput_kind=batched(request).
Output: results/P12_collab2_request_batch.csv
"""
import os, sys, csv, time, threading, subprocess
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("HW_GPU", "0"))
import numpy as np
import tensorrt as trt
import torch

LOG = trt.Logger(trt.Logger.ERROR)
ENGINE = "models/stage_a_cache/base_fp16.engine"
NS = [1, 2, 4, 8]
PHYS = int(os.environ.get("HW_GPU", "0"))


def load_engine(p):
    with open(p, "rb") as f:
        return trt.Runtime(LOG).deserialize_cuda_engine(f.read())


def make_ctx(engine):
    ctx = engine.create_execution_context()
    bufs = {}
    for i in range(engine.num_io_tensors):
        nm = engine.get_tensor_name(i)
        if engine.get_tensor_mode(nm) == trt.TensorIOMode.INPUT:
            shp = tuple(engine.get_tensor_shape(nm))
            ctx.set_input_shape(nm, shp)
    for i in range(engine.num_io_tensors):
        nm = engine.get_tensor_name(i)
        shp = tuple(ctx.get_tensor_shape(nm))
        t = torch.zeros(shp, dtype=torch.float32, device="cuda")
        bufs[nm] = t
        ctx.set_tensor_address(nm, int(t.data_ptr()))
    return ctx, bufs


def util_sampler(stop, samples):
    while not stop.is_set():
        try:
            o = subprocess.run(["nvidia-smi", "-i", str(PHYS),
                                "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=2).stdout.strip()
            samples.append(int(o.splitlines()[0]))
        except Exception:
            pass
        time.sleep(0.05)


def run_n(engine, n, hold_s=8):
    ctxs = [make_ctx(engine) for _ in range(n)]
    streams = [torch.cuda.Stream() for _ in range(n)]
    # warmup
    for (ctx, _), st in zip(ctxs, streams):
        with torch.cuda.stream(st):
            for _ in range(30):
                ctx.execute_async_v3(st.cuda_stream)
    for st in streams:
        st.synchronize()
    torch.cuda.synchronize()
    # per-request latency (single context, isolated) via events while others idle? -> measure under load:
    # sustained loop: all n streams in flight, measure wall + count
    stop = threading.Event(); usamp = []
    th = threading.Thread(target=util_sampler, args=(stop, usamp), daemon=True); th.start()
    # latency of one request under concurrent load: time each iteration on stream0
    ev_s = [torch.cuda.Event(enable_timing=True) for _ in range(200)]
    ev_e = [torch.cuda.Event(enable_timing=True) for _ in range(200)]
    t0 = time.perf_counter(); it = 0; k = 0
    while time.perf_counter() - t0 < hold_s:
        for j, ((ctx, _), st) in enumerate(zip(ctxs, streams)):
            with torch.cuda.stream(st):
                if j == 0 and k < 200:
                    ev_s[k].record(st); ctx.execute_async_v3(st.cuda_stream); ev_e[k].record(st)
                else:
                    ctx.execute_async_v3(st.cuda_stream)
        for st in streams:
            st.synchronize()
        if k < 200:
            k += 1
        it += 1
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    stop.set(); th.join(timeout=1)
    total_req = it * n
    tput = total_req / wall
    lat = np.array([ev_s[i].elapsed_time(ev_e[i]) for i in range(min(k, 200))])
    req_p50 = float(np.percentile(lat, 50)) if len(lat) else float("nan")
    util = float(np.mean(usamp[len(usamp)//5:])) if len(usamp) > 10 else float("nan")
    return tput, req_p50, util


def main():
    torch.cuda.set_device(0)
    eng = load_engine(str(REPO / ENGINE))
    rows = []
    tput1 = None
    for n in NS:
        tput, req_p50, util = run_n(eng, n)
        if n == 1:
            tput1 = tput
            print(f"★ SM%/util @ batch=1 (EARLY INDICATOR) = {util:.0f}%  "
                  f"(tput={tput:.1f} req/s, req_lat={req_p50:.3f}ms)")
        decoup = tput / tput1 if tput1 else 1.0
        print(f"  N={n}: total_tput={tput:7.1f} req/s  decoupling={decoup:.3f}x  "
              f"req_lat_p50={req_p50:.3f}ms  util~{util:.0f}%")
        rows.append({"n_concurrent_requests": n, "latency_kind": "body_subnet_collab2",
                     "throughput_kind": "batched_request", "engine": ENGINE,
                     "total_throughput_req_s": round(tput, 1),
                     "decoupling_vs_n1": round(decoup, 3),
                     "per_request_lat_p50_ms": round(req_p50, 4),
                     "gpu_util_pct": round(util, 1), "gpu": PHYS,
                     "source": "p12_collab2_request_batch;multi-context;CUDA-Event;util-smi"})
    out = REPO / "results/P12_collab2_request_batch.csv"
    keys = sorted({k for r in rows for k in r})
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows: w.writerow(r)
    peak = max(r["decoupling_vs_n1"] for r in rows)
    print(f"\n[verdict] peak decoupling = {peak:.3f}x -> "
          f"{'DECOUPLES (3D throughput)' if peak > 1.2 else 'WEAK/NO decouple -> 2D, throughput≈1/lat constraint'}")
    print(f"[done] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
