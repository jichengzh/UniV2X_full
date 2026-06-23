"""P0.2 (lat half) — real INT8 vs FP16 vs PyTorch latency of get_multiscale.

P0.2 built FP16/INT8 multi-output engines and verified AP (−0.14%), but the GPU
was contended so the *latency* was never measured. This is the missing half:
does INT8 actually accelerate the ResNeXt (giving the quantization axis a real
latency spread for the co-design search), or does it fall back to ~FP16?

Pure submodule latency on the real DAIR input (1,64,128,256), CUDA-Event timed:
  PyTorch fp32  |  PyTorch fp16 (autocast)  |  TRT fp16  |  TRT int8

Run on idle GPU (C10):
  cd /home/jichengzhi/heal_research/HEAL && CUDA_VISIBLE_DEVICES=6 python \
    /home/jichengzhi/UniV2X/scripts/phase2/p0_2d_int8_latency.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import tensorrt as trt

REPO = Path("/home/jichengzhi/UniV2X")
DATA_DIR = REPO / "paper_learning" / "2. AAAI最终故事" / "data"
sys.path.insert(0, str(REPO / "scripts" / "phase2"))

from p0_2_multiscale_trt import boot_pyramid, ENGINE_DIR, P64_BASELINE  # noqa: E402

CALIB_DIR = Path("/tmp/plan6_p0_2_calib")
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
N_WARMUP = 200
N_RUNS = 300
INPUT_SHAPE = (1, 64, 128, 256)


def _stats(lat):
    a = np.asarray(lat, dtype=float)
    return {"mean": round(float(a.mean()), 4), "p50": round(float(np.percentile(a, 50)), 4),
            "p99": round(float(np.percentile(a, 99)), 4), "std": round(float(a.std()), 4)}


def bench_pytorch(pyramid, x, amp=False):
    def run():
        if amp:
            with torch.autocast("cuda", dtype=torch.float16):
                return pyramid.get_multiscale_feature(x)
        return pyramid.get_multiscale_feature(x)
    with torch.no_grad():
        for _ in range(N_WARMUP):
            run()
        torch.cuda.synchronize()
        s = [torch.cuda.Event(enable_timing=True) for _ in range(N_RUNS)]
        e = [torch.cuda.Event(enable_timing=True) for _ in range(N_RUNS)]
        for i in range(N_RUNS):
            s[i].record(); run(); e[i].record()
        torch.cuda.synchronize()
    return _stats([s[i].elapsed_time(e[i]) for i in range(N_RUNS)])


def bench_engine(engine_path, x):
    with open(engine_path, "rb") as f:
        engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(f.read())
    ctx = engine.create_execution_context()
    in_name, out_names = None, []
    for i in range(engine.num_io_tensors):
        n = engine.get_tensor_name(i)
        if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT:
            in_name = n
        else:
            out_names.append(n)
    d_outs = [torch.empty(tuple(ctx.get_tensor_shape(n)), dtype=torch.float32, device="cuda")
              for n in out_names]
    ctx.set_tensor_address(in_name, x.data_ptr())
    for j, n in enumerate(out_names):
        ctx.set_tensor_address(n, d_outs[j].data_ptr())
    stream = torch.cuda.current_stream().cuda_stream
    for _ in range(N_WARMUP):
        ctx.execute_async_v3(stream)
    torch.cuda.synchronize()
    s = [torch.cuda.Event(enable_timing=True) for _ in range(N_RUNS)]
    e = [torch.cuda.Event(enable_timing=True) for _ in range(N_RUNS)]
    for i in range(N_RUNS):
        s[i].record(); ctx.execute_async_v3(stream); e[i].record()
    torch.cuda.synchronize()
    return _stats([s[i].elapsed_time(e[i]) for i in range(N_RUNS)])


def main():
    tag, model_dir, num_filters, ckpt_name = P64_BASELINE
    print("[lat] boot pyramid", flush=True)
    pyramid = boot_pyramid(model_dir, num_filters, ckpt_name)

    # real input: one harvested DAIR sample
    calib = np.load(CALIB_DIR / f"{tag}_calib.npy")
    x = torch.from_numpy(np.ascontiguousarray(calib[:1])).cuda()
    assert tuple(x.shape) == INPUT_SHAPE, f"unexpected input {tuple(x.shape)}"

    res = {}
    print("[lat] PyTorch fp32 ...", flush=True)
    res["pytorch_fp32"] = bench_pytorch(pyramid, x, amp=False)
    print("[lat] PyTorch fp16 (autocast) ...", flush=True)
    res["pytorch_fp16_amp"] = bench_pytorch(pyramid, x, amp=True)
    for q in ["fp16", "int8"]:
        eng = ENGINE_DIR / f"{tag}_{q}.engine"
        if eng.exists():
            print(f"[lat] TRT {q} ...", flush=True)
            res[f"trt_{q}"] = bench_engine(eng, x)

    base = res["pytorch_fp32"]["mean"]
    for k in res:
        res[k]["speedup_vs_torch_fp32"] = round(base / res[k]["mean"], 2)
    out = DATA_DIR / "p0_2d_int8_latency.json"
    out.write_text(json.dumps(res, indent=2))

    print("\n============== get_multiscale latency (DAIR 1x64x128x256) ==============")
    print(f"{'variant':20s} {'mean(ms)':>10s} {'p50':>8s} {'p99':>8s} {'vs torch fp32':>14s}")
    for k in ["pytorch_fp32", "pytorch_fp16_amp", "trt_fp16", "trt_int8"]:
        if k in res:
            r = res[k]
            print(f"{k:20s} {r['mean']:10.4f} {r['p50']:8.4f} {r['p99']:8.4f} "
                  f"{r['speedup_vs_torch_fp32']:13.2f}x")
    if "trt_int8" in res and "trt_fp16" in res:
        print(f"\nINT8 vs FP16 (TRT): "
              f"{res['trt_fp16']['mean'] / res['trt_int8']['mean']:.2f}x")
    print(f"[done] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
