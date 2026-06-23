"""P0-1·hw — p25 (48/96/192) INT8 collab2 latency + energy + alignment-trap.

Reuses verbatim:
  - m4_9_bench_perstage_v2.benchmark_engine  -> collab2 lat p50/p99 (CUDA-Event,
    warmup200/measure200, input (2,64,128,256) + t_ego (2,2,3))
  - e5_collab2_energy_bench.measure_engine_energy_collab2 -> NVML board energy
    (frame = ONE 2-agent fused forward, energy = P_mean*lat_p50, NOT /2)
  - e4_energy_bench.{load_engine,gpu_status,measure_idle_power}

口径: body_subnet_collab2 (== 主表 33 点). NOT e2e (no voxelize/NMS).
Device: physical GPU set by CUDA_VISIBLE_DEVICES (default 1). NVML handle uses
the PHYSICAL index (passed via --phys-gpu) since NVML ignores CUDA_VISIBLE_DEVICES.

Alignment-trap thesis: p25 stage0=48 is NOT a multiple of the INT8 IMMA tile
(32/64) -> TRT pads 48->64, so INT8 buys little over FP16. Aligned controls
p50(32/64/128) & p75(16/32/64) INT8 (already in主表) speed up cleanly. We report
ABSOLUTE lat/energy + effective-compute-normalized ratios, NOT a raw FP16->INT8
ratio (p25 FP16 baseline is itself padding-slowed = unclean denominator).
"""
from __future__ import annotations
import argparse, csv, os, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "phase1"))
sys.path.insert(0, str(REPO / "scripts" / "phase2"))

# default to GPU1 BEFORE torch import
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("HW_GPU", "1"))

import pynvml  # noqa: E402
import torch  # noqa: E402
from m4_9_bench_perstage_v2 import benchmark_engine  # noqa: E402
from e4_energy_bench import load_engine, gpu_status, measure_idle_power  # noqa: E402
from e5_collab2_energy_bench import measure_engine_energy_collab2  # noqa: E402

# (label, engine_rel_path, planes, prec_token, quant_mode)
TARGETS = [
    ("p25_fp16_rebaseline", "models/stage_a_cache/pruned25_fp16.engine",
     (48, 96, 192), "FP16", "automix_fp16"),
    ("p25_int8_automix", "models/stage_a_cache/pruned25_int8.engine",
     (48, 96, 192), "INT8", "automix_int8(TRT-auto)"),
    ("p25_c_all_int8_forced",
     "models/perstage_quant_ap_cache_v2/pruned25_c_all_int8_forced.engine",
     (48, 96, 192), "INT8", "forced_all_int8(PREFER_PRECISION_CONSTRAINTS)"),
]


def gate(handle, phys, mem_guard=50.0):
    util, mem, others = gpu_status(handle)
    if util > 2 or others > mem_guard:
        raise RuntimeError(
            f"GPU{phys} NOT idle: util={util}% other_proc_mem={others:.0f}MiB "
            f"(>2% or >{mem_guard}MiB) — abort (红线: 只在空闲卡测)")
    return util, others


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phys-gpu", type=int, default=int(os.environ.get("HW_GPU", "1")))
    ap.add_argument("--hold_s", type=float, default=12.0)
    ap.add_argument("--out", default=str(REPO / "results/P0_1_p25_int8_trap_4090.csv"))
    args = ap.parse_args()

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(args.phys_gpu)
    u0, o0 = gate(handle, args.phys_gpu)
    print(f"[gate] GPU{args.phys_gpu} idle: util={u0}% other_proc_mem={o0:.0f}MiB — proceed")
    torch.cuda.set_device(0)  # phys gpu is the only visible device
    idle_w, idle_n = measure_idle_power(handle, secs=5.0)
    print(f"[idle] board power = {idle_w:.1f}W ({idle_n} samples)")

    rows = []
    for label, rel, planes, prec, qmode in TARGETS:
        path = REPO / rel
        if not path.exists():
            print(f"[skip] {label}: engine missing -> {rel}")
            rows.append({"config_label": label, "engine_path": rel,
                         "status": "MISSING_ENGINE", "quant_mode": qmode,
                         "planes": "_".join(map(str, planes))})
            continue
        # foreign-process recheck before each engine
        _, _, others = gpu_status(handle)
        if others > 50.0:
            print(f"[ABORT] co-tenant appeared (other_mem={others:.0f}MiB) — stop")
            break
        print(f"[bench] {label}  ({rel})")
        lat = benchmark_engine(str(path), input_shape=(2, 64, 128, 256),
                               extra_input_shapes={"t_ego": (2, 2, 3)},
                               n_warmup=200, n_measure=200)
        eng = load_engine(path)
        en = measure_engine_energy_collab2(eng, handle, hold_s=args.hold_s)
        ishape = en["input_shape"]
        shape_str = "x".join(map(str, ishape)) if ishape else "NA"
        print(f"   lat_p50={lat['p50_ms']:.4f} p99={lat['p99_ms']:.4f}ms  "
              f"P={en['mean_power_w']:.1f}W  "
              f"J/frame={en['energy_per_frame_j']*1000:.3f}mJ  "
              f"size={lat['engine_size_mb']:.2f}MB  in={shape_str}")
        rows.append({
            "config_label": label, "quant_mode": qmode,
            "planes(s0_s1_s2)": "_".join(map(str, planes)),
            "precision": prec, "latency_kind": "body_subnet_collab2",
            "lat_p50_ms": round(lat["p50_ms"], 4),
            "lat_p99_ms": round(lat["p99_ms"], 4),
            "lat_mean_ms": round(lat["mean_ms"], 4),
            "throughput_fps": round(1000.0 / lat["mean_ms"], 1),
            "mean_power_w": round(en["mean_power_w"], 2),
            "idle_power_w": round(idle_w, 2),
            "energy_per_frame_mj": round(en["energy_per_frame_j"] * 1000, 4),
            "energy_per_frame_mj_counter": (
                round(en["counter_j_per_frame"] * 1000, 4)
                if en["counter_j_per_frame"] is not None else None),
            "perf_per_watt_fps_per_w": round(en["perf_per_watt_fps_per_w"], 2),
            "engine_size_mb": round(lat["engine_size_mb"], 3),
            "input_shape": shape_str,
            "engine_path": rel, "gpu": args.phys_gpu,
            "status": "OK",
            "source": ("P0_1_p25_int8_trap;lat:m4_9_benchmark_engine;"
                       "energy:e5_collab2_core;CUDA-Event;NVML-idle-excl;"
                       f"GPU{args.phys_gpu}"),
        })

    pynvml.nvmlShutdown()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n[done] {len(rows)} rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
