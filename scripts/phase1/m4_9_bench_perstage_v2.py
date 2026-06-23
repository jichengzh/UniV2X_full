"""Bench the 21 cached per-stage quant engines (perstage_quant_ap_cache_v2/).

口径: body-subnet, CUDA-Event, warmup 200 / measure 200 — same warmup/measure
protocol as complete_points (scripts/phase1/m4_8_trt_build_bench.benchmark_engine).

★ 重要口径差异: 这些引擎的输入是 collab 2-agent (spatial_features (2,64,128,256)
+ t_ego (2,2,3)) 与 fusion, 而 complete_points 是单 agent (1,64,128,256)。
故 latency_kind 标 `body_subnet_collab2`, NOT 与 complete_points 同列可比。
本 perstage 集合内部彼此可比 (同 collab2 口径), 满足 Pareto verdict 需求。

只在完全空闲 GPU 上测 (util<=2% & mem<=50MiB), 由 CUDA_VISIBLE_DEVICES 选定。
"""
import argparse
import os
import subprocess
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch

ROOT = Path("/home/jichengzhi/UniV2X")
CACHE = ROOT / "models/perstage_quant_ap_cache_v2"
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

TRIPLETS = ["base", "pruned50", "pruned75"]
CONFIGS = ["c3_I_F_F", "c4_F_I_F", "c5_F_F_I", "c6_I_I_F", "c7_I_F_I",
           "c8_F_I_I", "c_all_int8_forced"]


def _phys_id():
    return os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()


def gpu_util_mem():
    """(util%, mem_used_MiB) of the physical GPU selected via CUDA_VISIBLE_DEVICES."""
    phys = _phys_id()
    out = subprocess.check_output(
        ["nvidia-smi", f"--id={phys}",
         "--query-gpu=utilization.gpu,memory.used",
         "--format=csv,noheader,nounits"]).decode().strip().splitlines()
    util, mem = out[0].split(",")
    return int(util.strip()), int(mem.strip())


def foreign_procs():
    """List (pid, mem_MiB) of compute processes on the target GPU other than us."""
    phys = _phys_id()
    out = subprocess.check_output(
        ["nvidia-smi", f"--id={phys}",
         "--query-compute-apps=pid,used_memory",
         "--format=csv,noheader,nounits"]).decode().strip()
    me = os.getpid()
    procs = []
    for line in out.splitlines():
        if not line.strip():
            continue
        pid, mem = line.split(",")
        if int(pid.strip()) != me:
            procs.append((int(pid.strip()), int(mem.strip())))
    return procs


def gpu_idle_evidence():
    """Return (util%, mem_used_MiB). Idle = our own process only (no foreign load)."""
    return gpu_util_mem()


def benchmark_engine(engine_path, input_shape, extra_input_shapes,
                     n_warmup=200, n_measure=200):
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    n_io = engine.num_io_tensors
    tensor_names = [engine.get_tensor_name(i) for i in range(n_io)]
    input_names = [n for n in tensor_names
                   if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
    output_names = [n for n in tensor_names
                    if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]

    context.set_input_shape(input_names[0], input_shape)
    for nm in input_names[1:]:
        if nm not in extra_input_shapes:
            raise ValueError(f"missing shape for extra input '{nm}'")
        context.set_input_shape(nm, extra_input_shapes[nm])

    bufs = {}
    for name in tensor_names:
        shape = tuple(context.get_tensor_shape(name))
        dtype = engine.get_tensor_dtype(name)
        torch_dtype = {
            trt.float32: torch.float32, trt.float16: torch.float16,
            trt.int32: torch.int32, trt.int8: torch.int8,
            trt.int64: torch.int64,
        }.get(dtype, torch.float32)
        bufs[name] = torch.empty(shape, dtype=torch_dtype, device="cuda")
        context.set_tensor_address(name, int(bufs[name].data_ptr()))

    torch.manual_seed(42)
    for nm in input_names:
        bufs[nm].copy_(torch.randn_like(bufs[nm].float()).to(bufs[nm].dtype))

    stream = torch.cuda.Stream()
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    with torch.cuda.stream(stream):
        for _ in range(n_warmup):
            context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()

    times_ms = np.empty(n_measure, dtype=np.float64)
    with torch.cuda.stream(stream):
        for k in range(n_measure):
            start_evt.record(stream)
            context.execute_async_v3(stream.cuda_stream)
            end_evt.record(stream)
            end_evt.synchronize()
            times_ms[k] = start_evt.elapsed_time(end_evt)

    return {
        "n_warmup": n_warmup, "n_measure": n_measure,
        "mean_ms": float(times_ms.mean()), "std_ms": float(times_ms.std()),
        "p50_ms": float(np.percentile(times_ms, 50)),
        "p99_ms": float(np.percentile(times_ms, 99)),
        "min_ms": float(times_ms.min()), "max_ms": float(times_ms.max()),
        "engine_size_mb": Path(engine_path).stat().st_size / 1e6,
        "input_shape": list(input_shape),
        "input_names": input_names, "output_names": output_names,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-warmup", type=int, default=200)
    ap.add_argument("--n-measure", type=int, default=200)
    ap.add_argument("--out", default=str(ROOT / "results/perstage_quant_latency_v2.csv"))
    args = ap.parse_args()

    # Strict baseline gate: physical card must be fully idle before we touch it.
    base_util, base_mem = gpu_util_mem()
    if base_util > 2 or base_mem > 50:
        raise RuntimeError(
            f"target GPU (phys {_phys_id()}) NOT idle at start: "
            f"util={base_util}% mem={base_mem}MiB — abort (用户纪律: 只在空闲卡测)")
    print(f"[gate] start baseline: phys {_phys_id()} util={base_util}% "
          f"mem={base_mem}MiB — fully idle, proceed")

    rows = []
    for trip in TRIPLETS:
        for cfg in CONFIGS:
            eng = CACHE / f"{trip}_{cfg}.engine"
            if not eng.exists():
                print(f"[skip] missing {eng}")
                continue
            # Between configs: ensure no FOREIGN process appeared on the card.
            # (our own bench process holds CUDA ctx + a few hundred MiB — expected.)
            fp = foreign_procs()
            if fp:
                raise RuntimeError(
                    f"foreign process(es) {fp} on GPU phys {_phys_id()} before "
                    f"{trip}_{cfg} — abort (用户纪律: 只在空闲卡测)")
            util, mem = gpu_idle_evidence()
            print(f"[bench] {trip}_{cfg}  (gpu util={util}% mem={mem}MiB, "
                  f"foreign_procs=none)")
            stats = benchmark_engine(
                str(eng), input_shape=(2, 64, 128, 256),
                extra_input_shapes={"t_ego": (2, 2, 3)},
                n_warmup=args.n_warmup, n_measure=args.n_measure)
            util2, mem2 = gpu_idle_evidence()
            shape_str = "x".join(str(s) for s in stats["input_shape"])
            rows.append({
                "config_label": cfg, "triplet": trip,
                "lat_p50_ms": round(stats["p50_ms"], 4),
                "lat_p99_ms": round(stats["p99_ms"], 4),
                "lat_mean_ms": round(stats["mean_ms"], 4),
                "lat_std_ms": round(stats["std_ms"], 4),
                "throughput_fps": round(1000.0 / stats["mean_ms"], 1),
                "input_shape": shape_str,
                "n_inputs": len(stats["input_names"]),
                "latency_kind": "body_subnet_collab2",
                "gpu_idle_verified": (f"phys{_phys_id()};baseline:util{base_util}%/mem{base_mem}MiB;"
                                      f"pre:util{util}%/mem{mem}MiB(foreign=none);"
                                      f"post:util{util2}%/mem{mem2}MiB"),
                "n_warmup": stats["n_warmup"], "n_measure": stats["n_measure"],
                "engine_size_mb": round(stats["engine_size_mb"], 3),
                "engine_path": str(eng.relative_to(ROOT)),
                "source": "m4_9_bench_perstage_v2;CUDA-Event;collab2_shape",
            })
            print(f"        p50={stats['p50_ms']:.4f} p99={stats['p99_ms']:.4f} "
                  f"mean={stats['mean_ms']:.4f} fps={1000.0/stats['mean_ms']:.1f}")

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\n[done] {len(df)} rows -> {args.out}")
    print(df[["triplet", "config_label", "lat_p50_ms", "lat_mean_ms",
              "throughput_fps"]].to_string())


if __name__ == "__main__":
    main()
