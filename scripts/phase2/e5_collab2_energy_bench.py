"""E5 — collab2 (2-agent) board energy bench on RTX 4090 (GPU 1).

Gives the 28 collab2 TRT engines a REAL energy column so they become
5-metric-complete (AP+latency+throughput+size+ENERGY).

Reuses the E4 measurement core (NVML power sampling, idle subtraction,
nvmlDeviceGetTotalEnergyConsumption counter cross-check, CUDA-Event latency).

CRITICAL FRAME DEFINITION (differs from E4):
  The collab2 engine input is (2,64,128,256) = 2 agents fused into ONE
  collaborative detection frame -> ONE fused output. A V2X "frame" = one
  2-agent collaborative inference. Therefore frames_per_inference = 1 and
  energy_per_frame = P_mean * latency_p50 (energy of ONE forward pass).
  We do NOT divide by the leading dim (2 = agents, not frames).

  This is the ONLY semantic difference from E4 (which divides by batch).
  Throughput here is frames/s = inferences/s (NOT inferences*2).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pynvml
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "phase2"))

# Reuse E4's primitives verbatim.
from e4_energy_bench import (  # noqa: E402
    load_engine,
    make_ctx,
    measure_latency_p50,
    PowerSampler,
    gpu_status,
    measure_idle_power,
)
import pynvml as _pynvml  # noqa: E402  (alias kept for clarity)
import time  # noqa: E402


def measure_engine_energy_collab2(engine, handle, hold_s=12.0, interval_s=0.05):
    """Same as E4 measure_engine_energy, but frames_per_inference = 1.

    The leading input dim is 2 (agents). One forward = one V2X frame.
    energy_per_frame = P_mean * latency_p50 (per ONE forward pass).
    throughput_fps = inferences / wall (NOT * batch).
    """
    context, bufs, batch_lead, input_names = make_ctx(engine)

    # detect the real 4D input shape for documentation / sanity flag
    input_shape = None
    for nm in input_names:
        s = tuple(bufs[nm].shape)
        if len(s) == 4:
            input_shape = s
            break
    frames_per_inference = 1  # collab2: 2-agent fused -> ONE frame

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        lat_p50, lat_mean = measure_latency_p50(context, stream)

    # warmup to steady clocks/thermals
    with torch.cuda.stream(stream):
        for _ in range(500):
            context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
    torch.cuda.synchronize()

    sampler = PowerSampler(handle, interval_s)
    sampler.start()
    try:
        e_start_mj = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
    except pynvml.NVMLError:
        e_start_mj = None
    t0 = time.perf_counter()
    n_inf = 0
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
    power_steady = power[len(power) // 10:] if len(power) > 20 else power
    mean_power = float(power_steady.mean())

    # throughput: ONE forward = ONE frame (do NOT multiply by leading dim)
    fps = n_inf / wall
    lat_s = lat_p50 / 1000.0
    energy_per_frame_j = mean_power * lat_s  # per ONE forward pass
    perf_per_watt = fps / mean_power

    # board energy counter cross-check: delta / inferences (frames)
    counter_j_per_frame = None
    if e_start_mj is not None and e_end_mj is not None and e_end_mj > e_start_mj:
        counter_j_per_frame = (e_end_mj - e_start_mj) / 1000.0 / n_inf

    return {
        "lat_p50_ms": lat_p50,
        "lat_mean_ms": lat_mean,
        "throughput_fps": fps,
        "mean_power_w": mean_power,
        "power_p50_w": float(np.percentile(power_steady, 50)),
        "power_std_w": float(power_steady.std()),
        "n_power_samples": len(power_steady),
        "n_inf": n_inf,
        "leading_dim": batch_lead,
        "input_shape": input_shape,
        "frames_per_inference": frames_per_inference,
        "wall_s": wall,
        "energy_per_frame_j": energy_per_frame_j,
        "perf_per_watt_fps_per_w": perf_per_watt,
        "counter_j_per_frame": counter_j_per_frame,
    }


def build_targets():
    """28 collab2 rows from dataset_v2, keyed by unique engine_path."""
    df = pd.read_parquet(REPO_ROOT / "multi_agent/data/dataset_v2.parquet")
    c = df[df["latency_kind"] == "body_subnet_collab2"].copy()
    targets = []
    for _, r in c.iterrows():
        prec = f"{r['stage0_prec']}_{r['stage1_prec']}_{r['stage2_prec']}"
        if r["stage0_prec"] == r["stage1_prec"] == r["stage2_prec"]:
            prec = r["stage0_prec"]  # uniform -> single token
        planes = f"{r['stage0_planes']}_{r['stage1_planes']}_{r['stage2_planes']}"
        targets.append({
            "config_label": r["config_label"],
            "engine_path": r["engine_path"],
            "precision": prec,
            "planes": planes,
            "ref_lat_p50_ms": r["lat_p50_ms"],
            "source_row": r["source"],
        })
    return targets


def reverify_idle(handle, gpu, mem_guard_mib, check_util=True):
    """Detect a co-tenant on the GPU.

    `check_util` True for the PRE-run check (GPU must be quiet before we start).
    Between engines we set it False: OUR OWN sustained loop drives util to 100%,
    so util is meaningless mid-run — only OTHER-process memory signals a real
    co-tenant. (other_procs_mem excludes our own context, see gpu_status.)
    """
    util, mem, others = gpu_status(handle)
    if check_util:
        dirty = (util > 5) or (others > mem_guard_mib)
    else:
        dirty = others > mem_guard_mib
    return dirty, util, others


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--hold_s", type=float, default=12.0)
    ap.add_argument("--mem_guard_mib", type=float, default=50.0)
    ap.add_argument("--n_warmup", type=int, default=200)
    ap.add_argument("--output", default="results/E5_collab2_energy.csv")
    args = ap.parse_args()

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(args.gpu)

    dirty, util, others = reverify_idle(handle, args.gpu, args.mem_guard_mib)
    print(f"[GPU{args.gpu}] pre-check: util={util}%  other_procs_mem={others:.0f} MiB")
    if dirty:
        print(f"ABORT: GPU{args.gpu} NOT exclusively idle (util>5% or "
              f"other_mem>{args.mem_guard_mib} MiB). Energy invalid on shared GPU.")
        pynvml.nvmlShutdown()
        return 2

    torch.cuda.set_device(args.gpu)
    idle_power, idle_n = measure_idle_power(handle, secs=5.0)
    print(f"[GPU{args.gpu}] idle board power = {idle_power:.1f} W ({idle_n} samples)")

    targets = build_targets()
    print(f"measuring {len(targets)} collab2 engines on GPU{args.gpu} ...")
    rows = []
    for i, t in enumerate(targets):
        path = REPO_ROOT / t["engine_path"]
        tag = f"{t['config_label']}/{t['precision']}/p{t['planes']}"
        print(f"  [{i+1}/{len(targets)}] {tag}: {path.name}")

        # periodic re-verify GPU exclusivity (every measurement). Mid-run we
        # only flag OTHER-process memory: our own sustained loop legitimately
        # holds util at 100%, so util is not a contention signal here.
        d2, u2, o2 = reverify_idle(handle, args.gpu, args.mem_guard_mib,
                                   check_util=False)
        if d2:
            print(f"     ABORT mid-run: GPU{args.gpu} got a co-tenant "
                  f"(other_proc_mem={o2:.0f}MiB > {args.mem_guard_mib}). "
                  f"Energy invalid; STOP.")
            break

        if not path.exists():
            print("     FAILED: engine file missing")
            rows.append({**_fail_row(t, idle_power, args.gpu,
                                     "engine_file_missing")})
            continue
        try:
            engine = load_engine(path)
            r = measure_engine_energy_collab2(engine, handle, hold_s=args.hold_s)
        except Exception as e:  # noqa: BLE001
            print(f"     FAILED: {e}")
            rows.append({**_fail_row(t, idle_power, args.gpu, str(e)[:200])})
            continue

        ishape = r["input_shape"]
        shape_str = "x".join(str(x) for x in ishape) if ishape else "NA"
        agent_flag = ""
        if ishape is not None and ishape[0] != 2:
            agent_flag = (f" [WARN leading_dim={ishape[0]} != 2 — "
                          f"NOT 2-agent collab2 shape]")
        print(f"     P_mean={r['mean_power_w']:.1f}W  "
              f"lat_p50={r['lat_p50_ms']:.4f}ms  "
              f"fps={r['throughput_fps']:.0f}  "
              f"J/frame={r['energy_per_frame_j']*1000:.3f}mJ  "
              f"in={shape_str}{agent_flag}")

        note = ("collab2 frame = ONE 2-agent fused forward; "
                "energy_per_frame = P_mean*lat_p50 (NOT /leading_dim=2)")
        if agent_flag:
            note += agent_flag
        rows.append({
            "config_label": t["config_label"],
            "engine_path": t["engine_path"],
            "precision": t["precision"],
            "planes(s0_s1_s2)": t["planes"],
            "latency_p50_ms": round(r["lat_p50_ms"], 4),
            "mean_power_w": round(r["mean_power_w"], 2),
            "power_p50_w": round(r["power_p50_w"], 2),
            "idle_power_w": round(idle_power, 2),
            "energy_per_frame_mj": round(r["energy_per_frame_j"] * 1000, 4),
            "energy_per_frame_mj_counter": (
                round(r["counter_j_per_frame"] * 1000, 4)
                if r["counter_j_per_frame"] is not None else None),
            "perf_per_watt_fps_per_w": round(r["perf_per_watt_fps_per_w"], 2),
            "throughput_fps": round(r["throughput_fps"], 1),
            "input_shape": shape_str,
            "frames_per_inference": r["frames_per_inference"],
            "gpu": args.gpu,
            "n_warmup": args.n_warmup,
            "n_measure": r["n_power_samples"],
            "source": "E5_collab2_energy;NVML;GPU1;idle-exclusive",
            "note": note,
        })

    pynvml.nvmlShutdown()

    df = pd.DataFrame(rows)
    out = REPO_ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nwrote {len(df)} rows -> {out}")
    with pd.option_context("display.width", 220, "display.max_columns", 60):
        cols = ["config_label", "precision", "planes(s0_s1_s2)",
                "latency_p50_ms", "mean_power_w", "energy_per_frame_mj",
                "perf_per_watt_fps_per_w", "input_shape"]
        cols = [c for c in cols if c in df.columns]
        print(df[cols].to_string(index=False))
    return 0


def _fail_row(t, idle_power, gpu, reason):
    return {
        "config_label": t["config_label"],
        "engine_path": t["engine_path"],
        "precision": t["precision"],
        "planes(s0_s1_s2)": t["planes"],
        "latency_p50_ms": None,
        "mean_power_w": None,
        "power_p50_w": None,
        "idle_power_w": round(idle_power, 2),
        "energy_per_frame_mj": None,
        "energy_per_frame_mj_counter": None,
        "perf_per_watt_fps_per_w": None,
        "throughput_fps": None,
        "input_shape": "NA",
        "frames_per_inference": 1,
        "gpu": gpu,
        "n_warmup": 200,
        "n_measure": 0,
        "source": "E5_collab2_energy;FAILED",
        "note": f"FAILED: {reason}",
    }


if __name__ == "__main__":
    raise SystemExit(main())
