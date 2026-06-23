"""Task #12 (b) — subnet-level throughput batch sweep (GPU saturation point).

collab2 frame-batching needs model-forward surgery (per-frame 2-agent fusion) =
deferred (>2h). This (b) measures the throughput-vs-batch curve + GPU saturation
at SUBNET level (single-agent body, clean batching), which is the precision/
fusion-independent PHYSICS of "throughput decouples from 1/latency": batch until
SMs saturate, then throughput stops scaling linearly.

口径: latency_kind=body_subnet (NOT collab2) -> does NOT enter collab2 Pareto;
it's the saturation-physics evidence for the throughput axis.

Builds a dynamic-batch FP16 engine from the subnet ONNX, runs batch∈{1,2,4,8},
reports throughput(fps), per-frame latency(ms), GPU util sampling.
Output: results/P12_subnet_batch_sweep.csv
"""
import os, sys, csv, time, threading, subprocess
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("HW_GPU", "1"))
import numpy as np
import tensorrt as trt
import torch

LOG = trt.Logger(trt.Logger.ERROR)
CONFIGS = [
    ("base", "models/pyramid_dair_m1_subnet_fp32.onnx"),
    ("p50", "models/pyramid_dair_m1_pruned50_subnet_fp32.onnx"),
]
BATCHES = [1, 2, 4, 8, 16]


def build_dyn(onnx_path, max_b=16):
    b = trt.Builder(LOG)
    net = b.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    trt.OnnxParser(net, LOG).parse(Path(onnx_path).read_bytes())
    # force input batch dim dynamic
    inp = net.get_input(0)
    shp = list(inp.shape)
    inp.shape = [-1] + shp[1:]
    cfg = b.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)
    cfg.set_flag(trt.BuilderFlag.FP16)
    prof = b.create_optimization_profile()
    prof.set_shape(inp.name, (1,)+tuple(shp[1:]), (max_b//2,)+tuple(shp[1:]), (max_b,)+tuple(shp[1:]))
    cfg.add_optimization_profile(prof)
    ser = b.build_serialized_network(net, cfg)
    return trt.Runtime(LOG).deserialize_cuda_engine(ser), inp.name, tuple(shp[1:])


def gpu_util_sampler(stop_evt, samples, gpu):
    while not stop_evt.is_set():
        try:
            out = subprocess.run(["nvidia-smi", "-i", str(gpu),
                                  "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=2).stdout.strip()
            samples.append(int(out.splitlines()[0]))
        except Exception:
            pass
        time.sleep(0.05)


def sweep(engine, in_name, chw, phys_gpu, hold_s=8):
    ctx = engine.create_execution_context()
    out_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)
                 if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT]
    rows = []
    for bsz in BATCHES:
        ctx.set_input_shape(in_name, (bsz,)+chw)
        inp = torch.randn((bsz,)+chw, dtype=torch.float32, device="cuda")
        ctx.set_tensor_address(in_name, int(inp.data_ptr()))
        outs = []
        for nm in out_names:
            s = tuple(ctx.get_tensor_shape(nm))
            t = torch.empty(s, dtype=torch.float32, device="cuda"); outs.append(t)
            ctx.set_tensor_address(nm, int(t.data_ptr()))
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            for _ in range(50): ctx.execute_async_v3(stream.cuda_stream)
            stream.synchronize()
        # per-inference latency (CUDA event)
        ev_s = [torch.cuda.Event(enable_timing=True) for _ in range(100)]
        ev_e = [torch.cuda.Event(enable_timing=True) for _ in range(100)]
        with torch.cuda.stream(stream):
            for k in range(100):
                ev_s[k].record(stream); ctx.execute_async_v3(stream.cuda_stream); ev_e[k].record(stream)
            stream.synchronize()
        torch.cuda.synchronize()
        lat_inf = np.array([ev_s[k].elapsed_time(ev_e[k]) for k in range(100)])
        inf_p50 = float(np.percentile(lat_inf, 50))   # per-inference (batch) ms
        per_frame_ms = inf_p50 / bsz
        # throughput sustained + util sampling
        stop = threading.Event(); usamp = []
        th = threading.Thread(target=gpu_util_sampler, args=(stop, usamp, phys_gpu), daemon=True); th.start()
        t0 = time.perf_counter(); n = 0
        with torch.cuda.stream(stream):
            while time.perf_counter()-t0 < hold_s:
                for _ in range(50): ctx.execute_async_v3(stream.cuda_stream)
                stream.synchronize(); n += 50
        torch.cuda.synchronize(); wall = time.perf_counter()-t0
        stop.set(); th.join(timeout=1)
        fps = n*bsz/wall
        util = float(np.mean(usamp[len(usamp)//5:])) if len(usamp) > 10 else float("nan")
        print(f"  batch={bsz:2d}: throughput={fps:7.1f} fps  per_frame={per_frame_ms:.4f}ms  "
              f"inf_p50={inf_p50:.4f}ms  util~{util:.0f}%")
        rows.append({"batch": bsz, "throughput_fps": round(fps, 1),
                     "per_frame_lat_ms": round(per_frame_ms, 4),
                     "per_inference_lat_ms": round(inf_p50, 4),
                     "gpu_util_pct": round(util, 1)})
    return rows


def main():
    phys = int(os.environ.get("HW_GPU", "1"))
    torch.cuda.set_device(0)
    allrows = []
    for tag, onnx in CONFIGS:
        print(f"=== {tag} subnet ({onnx}) ===")
        eng, in_name, chw = build_dyn(str(REPO / onnx))
        for r in sweep(eng, in_name, chw, phys):
            base_fps = None
            r2 = {"config": tag, "latency_kind": "body_subnet", "precision": "fp16",
                  "throughput_kind": "batched", "gpu": phys, **r,
                  "source": "p12_subnet_batch_sweep;dyn-batch;CUDA-Event;util-nvidia-smi"}
            allrows.append(r2)
        # speedup vs batch=1
        b1 = next(x for x in allrows if x["config"] == tag and x["batch"] == 1)
        for x in allrows:
            if x["config"] == tag:
                x["throughput_speedup_vs_b1"] = round(x["throughput_fps"]/b1["throughput_fps"], 3)
    out = REPO / "results/P12_subnet_batch_sweep.csv"
    keys = sorted({k for r in allrows for k in r})
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in allrows: w.writerow(r)
    print(f"\n[done] {len(allrows)} rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
