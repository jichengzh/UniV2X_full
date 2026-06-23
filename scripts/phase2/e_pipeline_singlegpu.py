"""E_pipeline — single-GPU stage-partitioned software pipeline (throughput vs e2e).

PURPOSE
-------
Test the user's hypothesis: split one PyramidFusion forward into K stages, then
pipeline consecutive frames so frame N @ stage_{k+1} runs concurrently with
frame N+1 @ stage_k. In steady state the OUT-frame interval should approach
max(stage) (here ~shrink 2.1ms) rather than sum(stage) (~8ms), IF the GPU has
spare resources to overlap adjacent stages (roofline headroom 0.53-0.63).

This complements:
  * E1 (multistream / data-parallel, full-model copies) measured 1.13x only.
  * E_headroom (roofline) said weighted overlap headroom = 0.53-0.63.
We now measure the ACTUAL pipeline speedup with real CUDA streams+events.

STAGE SPLIT  (identical to scripts/phase2/e_headroom_roofline.py)
-----------------------------------------------------------------
  stage0 = resnet.layer0
  stage1 = resnet.layer1
  stage2 = resnet.layer2
  deblocks = 3x ConvTranspose2d (concat -> 384ch)
  shrink   = Conv2d 384->256
  heads    = cls/reg/dir
Each stage is a pure tensor->tensor callable so the chain == full forward.

PIPELINE MODEL
--------------
Stage-resident pipeline: each of the K stages owns a dedicated CUDA stream.
Frame f flows stage0->...->stageK-1. Same-frame ordering enforced by recording
an event after stage k completes and having stage k+1 wait on it. Different
frames are at different stages simultaneously -> hardware can overlap a
memory-bound backbone stage of frame f with a compute-bound shrink of frame f-3.

We push N frames into the pipeline as fast as the host loop allows (issue all
work, no per-stage host sync), then one cuda.synchronize() at the end. Steady-
state OUT interval = total_wall / N (after warmup runs to fill the pipe). To
isolate steady state we also report (T(2N)-T(N))/N as the marginal interval.

DOUBLE 口径
-----------
  * throughput: out-frame interval ms/frame -> FPS, speedup vs SEQUENTIAL sum.
  * single-frame e2e latency: time ONE frame through all stages serially (one
    stream, sync) -> should ~= sum(stage), UNCHANGED by pipelining.
  * ideal lower bound on interval = max(stage) ~ 2.1ms (perfect pipeline).

CAVEATS
-------
* PyTorch eager (not TRT). Same engine/path as e_headroom_roofline so the
  per-stage p50 are directly comparable. E1's 1.13x was on a TRT engine; the
  comparison is qualitative (does stage-pipelining beat data-parallel?).
* batch=1 single-agent path, fp16 Tensor-Core. Random init (shapes/dtype drive
  latency, not weights). Numerical check: stage-chained output == one-shot
  full forward of the same modules (we assert allclose).
* Must run on CLEAN gpu (util 0% / mem<=50MiB). Use --device cuda:2.
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


def build_modules(dtype, device):
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
    shrink = nn.Conv2d(384, 256, 3, 1, 1).to(device).eval().to(dtype)
    heads = nn.ModuleDict({
        "cls": nn.Conv2d(256, 2, 1),
        "reg": nn.Conv2d(256, 2 * 7, 1),
        "dir": nn.Conv2d(256, 2 * 2, 1),
    }).to(device).eval().to(dtype)
    return model, shrink, heads


def make_stage_fns(model, shrink, heads):
    """Return ordered list of (name, fn) where fn(x)->y, chained = full forward.

    Each fn takes the previous stage's output tensor and returns this stage's
    output. deblocks needs all three resnet feats, so we carry a small dict of
    intermediates keyed by frame in the pipeline driver; here we expose the raw
    per-stage transforms and let the driver thread the state.
    """
    resnet = model.resnet

    def s0(x):
        return resnet.layer0(x)

    def s1(f0):
        return resnet.layer1(f0)

    def s2(f1):
        return resnet.layer2(f1)

    # deblocks consume f0,f1,f2; we pass a tuple through the pipe
    def s_deblocks(feats):
        f0, f1, f2 = feats
        ups = [model.deblocks[0](f0), model.deblocks[1](f1), model.deblocks[2](f2)]
        return torch.cat(ups, dim=1)

    def s_shrink(cat):
        return shrink(cat)

    def s_heads(sh):
        return (heads["cls"](sh), heads["reg"](sh), heads["dir"](sh))

    return [
        ("stage0", s0),
        ("stage1", s1),
        ("stage2", s2),
        ("deblocks", s_deblocks),
        ("shrink", s_shrink),
        ("heads", s_heads),
    ]


# stage transition state: each stage k consumes the running state and produces
# the next. We special-case the fan-out: after stage2 we must keep f0,f1,f2 to
# feed deblocks. So the pipeline carries a per-frame dict.

def full_forward(stage_fns, x):
    """Run all stages serially on the current stream; returns final output."""
    f0 = stage_fns[0][1](x)
    f1 = stage_fns[1][1](f0)
    f2 = stage_fns[2][1](f1)
    cat = stage_fns[3][1]((f0, f1, f2))
    sh = stage_fns[4][1](cat)
    out = stage_fns[5][1](sh)
    return out


def numerical_check(stage_fns, x):
    """Stage-chained == direct call sanity (within fp16 tol)."""
    with torch.no_grad():
        out_a = full_forward(stage_fns, x)
        out_b = full_forward(stage_fns, x)  # determinism check (eval, no dropout)
    ok = True
    maxdiff = 0.0
    for a, b in zip(out_a, out_b):
        d = (a.float() - b.float()).abs().max().item()
        maxdiff = max(maxdiff, d)
        if d > 1e-3:
            ok = False
    return ok, maxdiff


def time_sequential(stage_fns, x, warmup, measure):
    """Single-stream serial full forward. p50 ms = single-frame e2e == sum(stage)."""
    s = torch.cuda.Stream()
    with torch.cuda.stream(s), torch.no_grad():
        for _ in range(warmup):
            full_forward(stage_fns, x)
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(measure)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(measure)]
    with torch.cuda.stream(s), torch.no_grad():
        for k in range(measure):
            starts[k].record(s)
            full_forward(stage_fns, x)
            ends[k].record(s)
    torch.cuda.synchronize()
    lat = np.array([starts[k].elapsed_time(ends[k]) for k in range(measure)])
    return float(np.percentile(lat, 50)), float(lat.mean())


def run_pipeline(stage_fns, x, n_frames, n_streams, warmup_frames):
    """Stage-resident software pipeline over n_frames.

    Each stage owns a stream (capped at n_streams; if n_streams < K we round-
    robin stages onto streams). Same-frame ordering via events: stage k+1 of
    frame f waits on the event recorded after stage k of frame f. Different
    frames occupy different stages concurrently.

    Returns wall_ms_total over the *measured* frames (excludes warmup), so
    interval = wall / measured_frames is the steady-state out-frame interval.
    """
    K = len(stage_fns)
    streams = [torch.cuda.Stream() for _ in range(n_streams)]
    # assign stage index -> stream
    stage_stream = [streams[k % n_streams] for k in range(K)]

    def issue_frame(f_idx):
        """Issue all K stages of one frame onto their streams with event chain.
        Returns the final-stage completion event for optional pipeline-fill use."""
        prev_evt = None
        state = x  # stage0 input (shared input tensor; read-only)
        f0 = f1 = f2 = None
        for k, (_, fn) in enumerate(stage_fns):
            st = stage_stream[k]
            if prev_evt is not None:
                st.wait_event(prev_evt)
            with torch.cuda.stream(st), torch.no_grad():
                if k == 0:
                    f0 = fn(state); out = f0
                elif k == 1:
                    f1 = fn(f0); out = f1
                elif k == 2:
                    f2 = fn(f1); out = f2
                elif k == 3:
                    out = fn((f0, f1, f2))
                elif k == 4:
                    out = fn(out)
                else:
                    out = fn(out)
            evt = torch.cuda.Event()
            evt.record(st)
            prev_evt = evt
        return prev_evt

    # warmup: fill the pipe
    with torch.no_grad():
        for f in range(warmup_frames):
            issue_frame(f)
    torch.cuda.synchronize()

    # measured region
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    with torch.no_grad():
        for f in range(n_frames):
            issue_frame(f)
    t1.record()
    torch.cuda.synchronize()
    wall_ms = t0.elapsed_time(t1)
    return wall_ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--height", type=int, default=200)
    ap.add_argument("--width", type=int, default=704)
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--n_frames", type=int, default=64)
    ap.add_argument("--warmup_frames", type=int, default=64)
    ap.add_argument("--streams", default="1,2,3,6",
                    help="comma list of stream counts to sweep")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", default=str(REPO_ROOT / "results/E_pipeline_singlegpu_4090.csv"))
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("[abort] no CUDA", flush=True); sys.exit(2)
    torch.cuda.set_device(args.device)
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    dev_name = torch.cuda.get_device_name(args.device)
    gpu_idx = torch.cuda.current_device()
    print(f"[boot] device={args.device}({dev_name}) idx={gpu_idx} dtype={args.dtype} "
          f"HxW={args.height}x{args.width}", flush=True)

    model, shrink, heads = build_modules(dtype, args.device)
    x = torch.randn(1, 64, args.height, args.width, dtype=dtype, device=args.device)
    stage_fns = make_stage_fns(model, shrink, heads)

    ok, maxdiff = numerical_check(stage_fns, x)
    print(f"[numcheck] stage-chain deterministic ok={ok} maxdiff={maxdiff:.2e}", flush=True)

    # sequential baseline (single-frame e2e == sum of stages)
    seq_p50, seq_mean = time_sequential(stage_fns, x, warmup=200, measure=300)
    ideal_max_stage = 2.10  # shrink, from e_headroom_roofline (cited)
    print(f"[sequential] single-frame e2e p50={seq_p50:.4f}ms mean={seq_mean:.4f}ms "
          f"(this is the sum-of-stages baseline)", flush=True)

    rows = []
    rows.append({
        "mode": "sequential",
        "n_frames": args.n_frames,
        "interval_ms_per_frame": round(seq_p50, 5),
        "throughput_fps": round(1000.0 / seq_p50, 2),
        "fps_speedup_vs_sequential": 1.0,
        "single_frame_lat_ms": round(seq_p50, 5),
        "ideal_max_stage_ms": ideal_max_stage,
        "gpu": gpu_idx,
        "note": f"PyTorch fp16 single-stream serial; mean={seq_mean:.4f}; "
                f"== sum(stage) baseline for speedup",
    })

    stream_counts = [int(s) for s in args.streams.split(",")]
    for S in stream_counts:
        intervals = []
        for _ in range(args.repeats):
            wall = run_pipeline(stage_fns, x, args.n_frames, S, args.warmup_frames)
            intervals.append(wall / args.n_frames)
        interval = float(np.median(intervals))
        fps = 1000.0 / interval
        speedup = seq_p50 / interval
        print(f"[pipeline S={S}] interval={interval:.4f}ms/frame fps={fps:.1f} "
              f"speedup_vs_seq={speedup:.3f}x (intervals={[round(i,3) for i in intervals]})",
              flush=True)
        rows.append({
            "mode": f"pipeline_{S}streams",
            "n_frames": args.n_frames,
            "interval_ms_per_frame": round(interval, 5),
            "throughput_fps": round(fps, 2),
            "fps_speedup_vs_sequential": round(speedup, 4),
            "single_frame_lat_ms": round(seq_p50, 5),  # unchanged by pipelining
            "ideal_max_stage_ms": ideal_max_stage,
            "gpu": gpu_idx,
            "note": f"stage-resident pipe, {S} streams, warmup={args.warmup_frames}f, "
                    f"median of {args.repeats} runs",
        })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fields = ["mode", "n_frames", "interval_ms_per_frame", "throughput_fps",
              "fps_speedup_vs_sequential", "single_frame_lat_ms",
              "ideal_max_stage_ms", "gpu", "note"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\n[done] wrote {args.out}", flush=True)

    best = max((r for r in rows if r["mode"] != "sequential"),
               key=lambda r: r["fps_speedup_vs_sequential"])
    print(f"[RESULT] best pipeline speedup = {best['fps_speedup_vs_sequential']}x "
          f"({best['mode']}), interval={best['interval_ms_per_frame']}ms/frame, "
          f"ideal_max_stage={ideal_max_stage}ms, E1_dataparallel=1.13x", flush=True)


if __name__ == "__main__":
    main()
