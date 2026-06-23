"""E4 — energy (perf/watt, J/frame) bench on RTX 4090.

Goal: give already-measured-latency TRT engines a REAL energy column so the
energy axis becomes a Pareto first-class metric (was 0 real data).

Power 口径 (board-level, NOT dynamic-only):
  * P_mean = steady-state board-average power (W), sampled via NVML
    nvmlDeviceGetPowerUsage at ~50 ms interval while a sustained inference
    loop saturates the engine on the target GPU.
  * idle_power_w = board-average power over a 5 s idle window (no inference)
    on the SAME GPU, recorded for reference. Board power is the
    deployment-relevant quantity; idle baseline lets a reader derive dynamic
    power = P_mean - idle if desired.
  * energy_per_frame (J) = P_mean (W) * latency_p50 (s)   [formula source: J=P*t]
  * perf_per_watt (fps/W) = throughput_fps / P_mean

CORRECTNESS REQUIREMENTS (enforced):
  * The target GPU MUST be idle before measuring (util ~0%, mem <= guard MiB),
    otherwise board power reflects OTHER tenants' load and J/frame is garbage.
    Script aborts with a clear message unless --allow_dirty is passed (which is
    only for pipeline validation, NOT for reportable numbers — output is then
    flagged invalid_dirty_gpu=True).
  * latency_p50 reused via the SAME CUDA-Event口径 as E1
    (scripts/phase2/e1_multistream_bench.py), single-stream.

Engine input shapes/names are auto-detected from the engine so this works for
both the doe_dataset_v1 subnet engines (input spatial_features [1,64,128,256])
and the p0_random_cache cudagraph engines ([2,64,128,256] + t_ego).
"""
from __future__ import annotations

import argparse
import os
import threading
import time
from pathlib import Path

import numpy as np
import pynvml
import tensorrt as trt
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

_TRT2TORCH = {
    trt.float32: torch.float32, trt.float16: torch.float16,
    trt.int32: torch.int32, trt.int8: torch.int8, trt.int64: torch.int64,
}


def load_engine(path: Path):
    runtime = trt.Runtime(TRT_LOGGER)
    with open(path, "rb") as f:
        return runtime.deserialize_cuda_engine(f.read())


def make_ctx(engine):
    """Execution context + IO buffers with addresses bound. Auto-detect shapes."""
    context = engine.create_execution_context()
    n_io = engine.num_io_tensors
    names = [engine.get_tensor_name(i) for i in range(n_io)]
    input_names = [n for n in names
                   if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
    # set dynamic input shapes to their engine min==opt==max (these engines are
    # built static; if profile has range, take the max/opt shape).
    for name in input_names:
        shp = engine.get_tensor_shape(name)
        if any(d < 0 for d in shp):
            # dynamic: use opt shape from profile 0
            _, opt, _ = engine.get_tensor_profile_shape(name, 0)
            shp = opt
        context.set_input_shape(name, tuple(shp))
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
    # batch = first dim of the largest 4D input (frames/inference)
    batch = 1
    for nm in input_names:
        s = tuple(bufs[nm].shape)
        if len(s) == 4:
            batch = s[0]
            break
    return context, bufs, batch, input_names


def measure_latency_p50(context, stream, n_warmup=200, n_measure=200):
    """Single-stream CUDA-Event latency (same口径 as E1). Returns p50 ms."""
    for _ in range(n_warmup):
        context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    torch.cuda.synchronize()
    ev_s = [torch.cuda.Event(enable_timing=True) for _ in range(n_measure)]
    ev_e = [torch.cuda.Event(enable_timing=True) for _ in range(n_measure)]
    for k in range(n_measure):
        ev_s[k].record(stream)
        context.execute_async_v3(stream.cuda_stream)
        ev_e[k].record(stream)
    stream.synchronize()
    torch.cuda.synchronize()
    lat = np.array([ev_s[k].elapsed_time(ev_e[k]) for k in range(n_measure)])
    return float(np.percentile(lat, 50)), float(lat.mean())


class PowerSampler(threading.Thread):
    """Background NVML board-power sampler at fixed interval."""

    def __init__(self, handle, interval_s=0.05):
        super().__init__(daemon=True)
        self.handle = handle
        self.interval_s = interval_s
        self._stop_evt = threading.Event()
        self.samples = []  # watts

    def run(self):
        while not self._stop_evt.is_set():
            try:
                mw = pynvml.nvmlDeviceGetPowerUsage(self.handle)
                self.samples.append(mw / 1000.0)
            except pynvml.NVMLError:
                pass
            time.sleep(self.interval_s)

    def stop_sampling(self):
        self._stop_evt.set()


def gpu_status(handle):
    util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
    mem = pynvml.nvmlDeviceGetMemoryInfo(handle).used / (1024 * 1024)  # MiB total
    # Memory used by OTHER processes only (exclude our own ~488 MiB context,
    # which is created at import and would otherwise self-trip the clean-check).
    own = os.getpid()
    others_bytes = 0
    try:
        for p in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
            if p.pid != own and p.usedGpuMemory:
                others_bytes += p.usedGpuMemory
    except pynvml.NVMLError:
        others_bytes = int(mem * 1024 * 1024)  # fallback: can't enumerate
    others_mib = others_bytes / (1024 * 1024)
    return util, mem, others_mib


def measure_idle_power(handle, secs=5.0, interval_s=0.05):
    sampler = PowerSampler(handle, interval_s)
    sampler.start()
    time.sleep(secs)
    sampler.stop_sampling()
    sampler.join()
    arr = np.array(sampler.samples)
    return float(arr.mean()), len(arr)


def measure_engine_energy(engine, handle, hold_s=12.0, interval_s=0.05):
    """Sustained inference loop while sampling board power.

    Returns dict with steady-state mean power, throughput, latency_p50.
    """
    context, bufs, batch, input_names = make_ctx(engine)
    stream = torch.cuda.Stream()

    # latency first (own warmup)
    with torch.cuda.stream(stream):
        lat_p50, lat_mean = measure_latency_p50(context, stream)

    # ---- sustained load + power sampling ----
    # warmup to reach steady clocks/thermals
    with torch.cuda.stream(stream):
        for _ in range(500):
            context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
    torch.cuda.synchronize()

    sampler = PowerSampler(handle, interval_s)
    sampler.start()
    # board energy counter (mJ, monotonic) for a sampling-independent J/frame
    # cross-check. NOTE: still board-level (includes any co-resident tenant).
    try:
        e_start_mj = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
    except pynvml.NVMLError:
        e_start_mj = None
    t0 = time.perf_counter()
    n_inf = 0
    # keep GPU saturated: submit in bursts then sync so the loop body cost is
    # negligible vs GPU work; count inferences for throughput.
    burst = 200
    with torch.cuda.stream(stream):
        while time.perf_counter() - t0 < hold_s:
            for _ in range(burst):
                context.execute_async_v3(stream.cuda_stream)
            stream.synchronize()
            n_inf += burst
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    try:
        e_end_mj = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
    except pynvml.NVMLError:
        e_end_mj = None
    sampler.stop_sampling()
    sampler.join()

    power = np.array(sampler.samples)
    # drop first 10% of samples (ramp) for steady-state mean
    if len(power) > 20:
        power_steady = power[len(power) // 10:]
    else:
        power_steady = power
    mean_power = float(power_steady.mean())
    # throughput counts frames = inferences * batch
    fps = (n_inf * batch) / wall
    lat_s = lat_p50 / 1000.0
    energy_per_frame_j = mean_power * lat_s
    perf_per_watt = fps / mean_power

    # cross-check: board energy counter delta / frames in the window. Under a
    # clean GPU this should agree with mean_power*lat_s within a few %.
    counter_j_per_frame = None
    if e_start_mj is not None and e_end_mj is not None and e_end_mj > e_start_mj:
        frames = n_inf * batch
        counter_j_per_frame = (e_end_mj - e_start_mj) / 1000.0 / frames

    return {
        "lat_p50_ms": lat_p50,
        "lat_mean_ms": lat_mean,
        "throughput_fps": fps,
        "mean_power_w": mean_power,
        "power_p50_w": float(np.percentile(power_steady, 50)),
        "power_std_w": float(power_steady.std()),
        "n_power_samples": len(power_steady),
        "n_inf": n_inf,
        "batch": batch,
        "wall_s": wall,
        "energy_per_frame_j": energy_per_frame_j,
        "perf_per_watt_fps_per_w": perf_per_watt,
        "counter_j_per_frame": counter_j_per_frame,
    }


# (engine_path, triplet, precision) tuples to measure
def build_targets():
    cg = REPO_ROOT / "models/p0_random_cache"
    doe = REPO_ROOT / "output/doe_dataset_v1/real_engines"
    targets = []
    # --- doe_dataset_v1: the "6+ complete points" (lat+AP already measured) ---
    doe_set = [
        ("T1_base_fp32.engine", "T1_base", "fp32", "doe6"),
        ("T1_base_fp16.engine", "T1_base", "fp16", "doe6"),
        ("T1_base_int8.engine", "T1_base", "int8", "doe6"),
        ("pruned_p25_fp16.engine", "T2_p25", "fp16", "doe6"),
        ("pruned_p50_fp16.engine", "T4_p50", "fp16", "doe6"),
        ("pruned_p75_fp16.engine", "T6_p75", "fp16", "doe6"),
        ("pruned_p50_int8.engine", "T4_p50", "int8", "doe6"),
    ]
    for fn, tri, prec, src in doe_set:
        p = doe / fn
        if p.exists():
            targets.append((p, tri, prec, src))
    # --- cudagraph triplets (E1-comparable口径) ---
    cg_set = [
        ("engine_064_128_256_fp16.engine", "T_baseline", "fp16", "cudagraph"),
        ("engine_064_128_256_int8.engine", "T_baseline", "int8", "cudagraph"),
        ("engine_032_064_136_fp16.engine", "T_prune50", "fp16", "cudagraph"),
        ("engine_032_064_136_int8.engine", "T_prune50", "int8", "cudagraph"),
        ("engine_016_032_064_fp16.engine", "T_prune75", "fp16", "cudagraph"),
        ("engine_016_032_064_int8.engine", "T_prune75", "int8", "cudagraph"),
    ]
    for fn, tri, prec, src in cg_set:
        p = cg / fn
        if p.exists():
            targets.append((p, tri, prec, src))
    return targets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=6)
    ap.add_argument("--hold_s", type=float, default=12.0)
    ap.add_argument("--mem_guard_mib", type=float, default=50.0)
    ap.add_argument("--allow_dirty", action="store_true",
                    help="bypass idle-GPU check (pipeline validation ONLY; "
                         "output flagged invalid)")
    ap.add_argument("--output", default="results/E4_energy_4090.csv")
    ap.add_argument("--ts_note", default="")
    args = ap.parse_args()

    # CLEAN-CHECK FIRST, before any CUDA context creation.
    # torch.cuda.set_device allocates a ~488 MiB context; if done before this
    # check it self-trips the mem_guard (script's own context mistaken for a
    # co-tenant). So query NVML on the physical device while no context exists.
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(args.gpu)

    util, mem, others_mib = gpu_status(handle)
    print(f"[GPU{args.gpu}] pre-check: util={util}%  total_mem={mem:.0f} MiB  "
          f"other_procs_mem={others_mib:.0f} MiB (own context excluded)")
    dirty = (util > 5) or (others_mib > args.mem_guard_mib)
    if dirty and not args.allow_dirty:
        print(f"ABORT: GPU{args.gpu} is NOT clean (util>{5}% or "
              f"mem>{args.mem_guard_mib} MiB). Energy numbers would reflect "
              f"other tenants' load. Re-run when idle, or --allow_dirty for "
              f"pipeline validation only.")
        pynvml.nvmlShutdown()
        return 2

    # bind torch to the target GPU (must match NVML index = physical index)
    torch.cuda.set_device(args.gpu)

    idle_power, idle_n = measure_idle_power(handle, secs=5.0)
    print(f"[GPU{args.gpu}] idle board power = {idle_power:.1f} W "
          f"({idle_n} samples)  dirty={dirty}")

    targets = build_targets()
    print(f"measuring {len(targets)} engines ...")
    rows = []
    for path, tri, prec, src in targets:
        print(f"  -> {src}/{tri}/{prec}: {path.name}")
        try:
            engine = load_engine(path)
            r = measure_engine_energy(engine, handle, hold_s=args.hold_s)
        except Exception as e:  # noqa: BLE001
            print(f"     FAILED: {e}")
            rows.append({
                "engine": path.name, "triplet": tri, "source": src,
                "precision": prec, "fail_reason": str(e)[:200],
                "gpu": args.gpu, "idle_power_w": round(idle_power, 2),
                "invalid_dirty_gpu": dirty, "ts_note": args.ts_note,
            })
            continue
        print(f"     P_mean={r['mean_power_w']:.1f}W  "
              f"lat_p50={r['lat_p50_ms']:.4f}ms  "
              f"fps={r['throughput_fps']:.0f}  "
              f"J/frame={r['energy_per_frame_j']*1000:.3f}mJ  "
              f"perf/W={r['perf_per_watt_fps_per_w']:.1f} fps/W")
        rows.append({
            "engine": path.name,
            "triplet": tri,
            "source": src,
            "precision": prec,
            "latency_p50_ms": round(r["lat_p50_ms"], 4),
            "throughput_fps": round(r["throughput_fps"], 1),
            "mean_power_w": round(r["mean_power_w"], 2),
            "power_p50_w": round(r["power_p50_w"], 2),
            "power_std_w": round(r["power_std_w"], 2),
            "idle_power_w": round(idle_power, 2),
            "energy_per_frame_mj": round(r["energy_per_frame_j"] * 1000, 4),
            "energy_per_frame_mj_counter": (
                round(r["counter_j_per_frame"] * 1000, 4)
                if r["counter_j_per_frame"] is not None else None),
            "perf_per_watt_fps_per_w": round(r["perf_per_watt_fps_per_w"], 2),
            "batch": r["batch"],
            "gpu": args.gpu,
            "n_samples": r["n_power_samples"],
            "invalid_dirty_gpu": dirty,
            "ts_note": args.ts_note,
        })

    pynvml.nvmlShutdown()

    import pandas as pd
    df = pd.DataFrame(rows)
    out = REPO_ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nwrote {len(df)} rows -> {out}")
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
