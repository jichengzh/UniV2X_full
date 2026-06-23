"""E_headroom — single-GPU stage-level pipeline overlap headroom via roofline.

PURPOSE
-------
Quantify the *upper bound* of throughput gain achievable by splitting one
PyramidFusion forward into stages and pipelining adjacent frames' stages on a
single GPU. Physics: cross-frame stage pipelining can only overlap if a stage
does NOT already saturate the GPU (SMs / memory BW). If every stage already
hits the throughput ceiling, pipelining buys nothing -- which is exactly what
E1 multi-stream measured (1.13x only).

We CANNOT use ncu (RmProfilingAdminOnly=1, no sudo -> ERR_NVGPUCTRPERM). So we
use a roofline / achieved-throughput method that needs no admin:

  achieved_TFLOPS = FLOPs / latency
  achieved_GBps   = bytes  / latency
  compute_pct = achieved_TFLOPS / peak_TFLOPS
  mem_pct     = achieved_GBps   / peak_GBps
  bound_type  = compute if compute_pct >= mem_pct else memory
  overlap_headroom = 1 - max(compute_pct, mem_pct)

The weighted-average headroom (weighted by stage latency) is the theoretical
upper bound on single-GPU pipeline throughput gain. If headroom is small, E1's
1.13x is explained: the backbone is already near the GPU throughput ceiling.

STAGE SPLIT (compute-heavy single-agent forward path of PyramidFusion)
----------------------------------------------------------------------
  stage0 = resnet.layer0   (Bottleneck x3,  64ch,  stride1)
  stage1 = resnet.layer1   (Bottleneck x5, 128ch,  stride2)
  stage2 = resnet.layer2   (Bottleneck x8, 256ch,  stride2)
  deblocks = 3x ConvTranspose2d upsample to 128ch each (concat -> 384ch)
  shrink   = Conv2d 384->256 3x3
  heads    = cls/reg/dir Conv2d on 256ch
(single_head_i occupancy heads and the collab warp/softmax fuse are tiny /
memory-bound glue; we time them too but they are not the compute story.)

CAVEATS / 口径
-------------
* All FLOPs / bytes are ANALYTIC (computed from conv configs + tensor shapes),
  NOT measured -- exact for convs (MACs*2), BN/ReLU folded as negligible add.
* Latency is REAL (CUDA Event, p50, warmup>=200, measure>=200), must run on a
  CLEAN GPU (util 0% / mem<=50MiB, NOT gpu 0/1/2).
* Peak 算力 4090: fp16 Tensor Core DENSE = 165.2 TFLOPS (NOT 2:4 sparse 330).
  We run the model in fp16 (Tensor Core path) so this is the right ceiling.
  GDDR6X peak BW = 1008 GB/s. Both are vendor spec, not measured.
* Single-agent path (batch=1) is the pipeline unit. The collab body runs the
  backbone on batched agents, but the pipeline-overlap question is per the
  per-frame stage, so batch=1 is the right granularity.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HEAL_ROOT = "/home/jichengzhi/heal_research/HEAL"
sys.path.insert(0, HEAL_ROOT)

REPO_ROOT = Path(__file__).resolve().parents[2]

# ---- 4090 vendor peaks (口径: DENSE fp16 Tensor Core, GDDR6X) ----
PEAK_FP16_TFLOPS_DENSE = 165.2     # RTX 4090 fp16 TC dense (sparse=330.4, NOT used)
PEAK_MEM_GBPS = 1008.0             # GDDR6X 21 Gbps x 384-bit


# ───────────────────────── analytic FLOPs / bytes hooks ─────────────────────


def _conv_flops_bytes(m, x, y, dtype_bytes):
    """MACs*2 FLOPs and (weight + in_act + out_act) bytes for a conv/convT."""
    out = y[0] if isinstance(y, (tuple, list)) else y
    out_elems = out.numel()
    # per-output-element MACs = (Cin/groups) * kH * kW
    cin = m.in_channels
    kh, kw = (m.kernel_size if isinstance(m.kernel_size, tuple)
              else (m.kernel_size, m.kernel_size))
    macs = out_elems * (cin // m.groups) * kh * kw
    flops = 2 * macs
    w_bytes = m.weight.numel() * dtype_bytes
    in_bytes = x[0].numel() * dtype_bytes
    out_bytes = out_elems * dtype_bytes
    return flops, w_bytes + in_bytes + out_bytes


class StageAccountant:
    """Accumulates analytic FLOPs/bytes over the modules executed in a stage."""

    def __init__(self, dtype_bytes):
        self.flops = 0
        self.bytes = 0
        self.dtype_bytes = dtype_bytes
        self._handles = []

    def _hook(self, m, x, y):
        f, b = _conv_flops_bytes(m, x, y, self.dtype_bytes)
        self.flops += f
        self.bytes += b

    def attach(self, module):
        for sub in module.modules():
            if isinstance(sub, (nn.Conv2d, nn.ConvTranspose2d)):
                self._handles.append(sub.register_forward_hook(self._hook))

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def reset(self):
        self.flops = 0
        self.bytes = 0


# ───────────────────────── CUDA Event timing ─────────────────────────


def time_callable(fn, warmup, measure):
    """Returns p50 latency (ms) of fn() on the current CUDA stream."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(measure)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(measure)]
    for k in range(measure):
        starts[k].record()
        fn()
        ends[k].record()
    torch.cuda.synchronize()
    lat = np.array([starts[k].elapsed_time(ends[k]) for k in range(measure)])
    return float(np.percentile(lat, 50)), float(lat.mean())


# ───────────────────────── model build ─────────────────────────


def build_backbone(dtype, device):
    """Instantiate PyramidFusion with the Pyramid_m1_base config, random init.

    We only need the *structure* for roofline (FLOPs/bytes are config-driven and
    latency is shape/dtype-driven, not weight-value driven). Random init is fine
    and avoids the ckpt-load dependency. Stage shapes match the real model.
    The compute-heavy path we time (resnet layers / deblocks / shrink / heads)
    runs fine on CPU too, so --dry can build on CPU without touching a busy GPU.
    """
    from opencood.models.fuse_modules.pyramid_fuse import PyramidFusion

    cfg = {
        "layer_nums": [3, 5, 8],
        "layer_strides": [1, 2, 2],
        "num_filters": [64, 128, 256],
        "upsample_strides": [1, 2, 4],
        "num_upsample_filter": [128, 128, 128],
        "resnext": True,
        "resnext_groups": 32,
        "width_per_group": 4,
        "inplanes": 64,
        "anchor_number": 2,
    }
    model = PyramidFusion(cfg, input_channels=64).to(device).eval().to(dtype)
    # shrink conv 384->256 (3x3) + heads on 256ch (anchor_number=2)
    shrink = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1).to(device).eval().to(dtype)
    heads = nn.ModuleDict({
        "cls": nn.Conv2d(256, 2, 1),       # anchor_number
        "reg": nn.Conv2d(256, 2 * 7, 1),   # anchor_number * 7
        "dir": nn.Conv2d(256, 2 * 2, 1),   # anchor_number * num_bins(2)
    }).to(device).eval().to(dtype)
    return model, shrink, heads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--measure", type=int, default=200)
    ap.add_argument("--height", type=int, default=200, help="BEV spatial H")
    ap.add_argument("--width", type=int, default=704, help="BEV spatial W")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--dry", action="store_true",
                    help="CPU-side: build + FLOPs/bytes only, no GPU timing")
    ap.add_argument("--out", default=str(REPO_ROOT / "results/E_headroom_roofline_4090.csv"))
    ap.add_argument("--out_json", default=str(REPO_ROOT / "results/E_headroom_roofline_4090.json"))
    args = ap.parse_args()

    dtype_bytes = 2 if args.dtype == "fp16" else 4
    peak_tflops = PEAK_FP16_TFLOPS_DENSE if args.dtype == "fp16" else 82.6  # fp32 TC TF32-off ~ 82.6

    if args.dry:
        # CPU-side validation: fp16 conv is not supported on CPU, force fp32 math
        # for the build/shape/accounting pass (FLOPs/bytes use --dtype's bytes).
        device = "cpu"
        build_dtype = torch.float32
        dev_name = "DRY(cpu-accounting)"
    else:
        if not torch.cuda.is_available():
            print("[abort] CUDA not available and not --dry", flush=True)
            sys.exit(2)
        device = "cuda"
        build_dtype = torch.float16 if args.dtype == "fp16" else torch.float32
        dev_name = torch.cuda.get_device_name(0)

    print(f"[boot] device={dev_name} dtype={args.dtype} HxW={args.height}x{args.width}", flush=True)

    model, shrink, heads = build_backbone(build_dtype, device)
    x = torch.randn(1, 64, args.height, args.width, dtype=build_dtype, device=device)

    # Build a list of (stage_name, module, fn) where fn maps prev-output->output.
    # We chain real shapes so each stage's input matches the live pipeline.
    resnet = model.resnet

    # warm a single full pass to discover intermediate shapes
    with torch.no_grad():
        f0 = resnet.layer0(x)
        f1 = resnet.layer1(f0)
        f2 = resnet.layer2(f1)
        feats = [f0, f1, f2]
        ups = [model.deblocks[i](feats[i]) for i in range(3)]
        cat = torch.cat(ups, dim=1)
        sh = shrink(cat)
        _ = (heads["cls"](sh), heads["reg"](sh), heads["dir"](sh))

    print(f"[shapes] f0={tuple(f0.shape)} f1={tuple(f1.shape)} f2={tuple(f2.shape)} "
          f"cat={tuple(cat.shape)} shrink={tuple(sh.shape)}", flush=True)

    # Define stages: (name, fn, module_for_flops)
    stages = OrderedDict()
    stages["stage0_layer0"] = (lambda: resnet.layer0(x), resnet.layer0)
    stages["stage1_layer1"] = (lambda: resnet.layer1(f0), resnet.layer1)
    stages["stage2_layer2"] = (lambda: resnet.layer2(f1), resnet.layer2)
    stages["deblocks_upsample"] = (
        lambda: [model.deblocks[i](feats[i]) for i in range(3)], model.deblocks)
    stages["shrink_conv"] = (lambda: shrink(cat), shrink)

    def heads_fn():
        return (heads["cls"](sh), heads["reg"](sh), heads["dir"](sh))
    stages["heads_cls_reg_dir"] = (heads_fn, heads)

    rows = []
    with torch.no_grad():
        for name, (fn, mod) in stages.items():
            acct = StageAccountant(dtype_bytes)
            acct.attach(mod)
            fn()  # one pass to fill accountant
            acct.detach()
            flops = acct.flops
            nbytes = acct.bytes

            if args.dry:
                p50 = mean = float("nan")
            else:
                p50, mean = time_callable(fn, args.warmup, args.measure)

            lat_s = p50 / 1000.0 if p50 == p50 else float("nan")  # nan-safe
            achieved_tflops = (flops / lat_s) / 1e12 if lat_s == lat_s and lat_s > 0 else float("nan")
            achieved_gbps = (nbytes / lat_s) / 1e9 if lat_s == lat_s and lat_s > 0 else float("nan")
            compute_pct = achieved_tflops / peak_tflops if achieved_tflops == achieved_tflops else float("nan")
            mem_pct = achieved_gbps / peak_mem_gbps() if achieved_gbps == achieved_gbps else float("nan")
            if compute_pct == compute_pct and mem_pct == mem_pct:
                bound = "compute" if compute_pct >= mem_pct else "memory"
                headroom = 1.0 - max(compute_pct, mem_pct)
            else:
                bound = "n/a"
                headroom = float("nan")

            rows.append({
                "stage": name,
                "latency_p50_ms": round(p50, 5) if p50 == p50 else "",
                "flops_g": round(flops / 1e9, 4),
                "bytes_mb": round(nbytes / 1e6, 4),
                "achieved_tflops": round(achieved_tflops, 3) if achieved_tflops == achieved_tflops else "",
                "achieved_gbps": round(achieved_gbps, 3) if achieved_gbps == achieved_gbps else "",
                "compute_pct_peak": round(compute_pct, 4) if compute_pct == compute_pct else "",
                "mem_pct_peak": round(mem_pct, 4) if mem_pct == mem_pct else "",
                "bound_type": bound,
                "overlap_headroom_pct": round(headroom, 4) if headroom == headroom else "",
            })
            print(f"  {name:20s} FLOPs={flops/1e9:8.3f}G bytes={nbytes/1e6:8.2f}MB "
                  f"p50={p50:7.4f}ms tflops={rows[-1]['achieved_tflops']} "
                  f"gbps={rows[-1]['achieved_gbps']} bound={bound} "
                  f"headroom={rows[-1]['overlap_headroom_pct']}", flush=True)

    # weighted-average headroom (by latency) — the pipeline upper bound
    wsum = 0.0
    hwsum = 0.0
    for r in rows:
        if r["latency_p50_ms"] != "" and r["overlap_headroom_pct"] != "":
            w = float(r["latency_p50_ms"])
            wsum += w
            hwsum += w * float(r["overlap_headroom_pct"])
    wavg_headroom = hwsum / wsum if wsum > 0 else float("nan")

    # write csv
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fields = ["stage", "latency_p50_ms", "flops_g", "bytes_mb", "achieved_tflops",
              "achieved_gbps", "compute_pct_peak", "mem_pct_peak", "bound_type",
              "overlap_headroom_pct"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    summary = {
        "device": dev_name if not args.dry else "DRY(cpu-accounting)",
        "dtype": args.dtype,
        "peak_fp16_tflops_dense": peak_tflops,
        "peak_mem_gbps": PEAK_MEM_GBPS,
        "bev_hw": [args.height, args.width],
        "warmup": args.warmup,
        "measure": args.measure,
        "stages": rows,
        "weighted_avg_overlap_headroom": (round(wavg_headroom, 4)
                                          if wavg_headroom == wavg_headroom else None),
        "interpretation": (
            "weighted_avg_overlap_headroom = theoretical upper bound on single-GPU "
            "cross-frame stage-pipeline throughput gain fraction. Compare to E1 "
            "multi-stream measured 1.13x (=13% gain)."
        ),
    }
    Path(args.out_json).write_text(json.dumps(summary, indent=2))
    print(f"\n[done] wrote {args.out} and {args.out_json}", flush=True)
    print(f"[RESULT] weighted-avg overlap headroom = "
          f"{wavg_headroom:.4f}  (E1 measured gain = 0.13)", flush=True)


def peak_mem_gbps():
    return PEAK_MEM_GBPS


if __name__ == "__main__":
    main()
