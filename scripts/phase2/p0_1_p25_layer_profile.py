"""P0-1·hw mechanism — per-layer TRT profile to confirm the p25(48) tactic cliff.

4090 host has NO trtexec CLI, so we use the TRT Python IProfiler (equivalent
per-layer timing). For each engine we attach a profiler, run N iters, accumulate
per-layer ms, then sort + classify reformat/copy layers.

Goal: show WHY p25(stage0=48) is 2.28x slower than base despite fewer channels.
Hypothesis (after padding-ratio was empirically falsified): TRT picks a bad
tactic / inserts heavy reformat at the 48-channel geometry. We compare:
  - p25 INT8  (the anomaly)
  - p25 FP16  (also slow 2.9ms -> cliff is NOT INT8-specific?)
  - p50 INT8  (aligned control, fast 0.80ms)

口径: body_subnet_collab2 (input 2x64x128x256 + t_ego 2x2x3). Clean GPU.
Output: results/P0_1_p25_layer_profile.csv (+ console top-layers).
"""
from __future__ import annotations
import os, sys, csv
from pathlib import Path
from collections import defaultdict

REPO = Path(__file__).resolve().parents[2]
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("HW_GPU", "7"))
sys.path.insert(0, str(REPO / "scripts" / "phase2"))

import tensorrt as trt  # noqa: E402
import torch  # noqa: E402
import pynvml  # noqa: E402
from e4_energy_bench import load_engine, make_ctx, gpu_status  # noqa: E402

TARGETS = [
    ("p25_int8", "models/stage_a_cache/pruned25_int8.engine"),
    ("p25_fp16", "models/stage_a_cache/pruned25_fp16.engine"),
    ("p50_int8_aligned_ctrl", "models/stage_a_cache/pruned50_int8.engine"),
]


class LayerProfiler(trt.IProfiler):
    def __init__(self):
        trt.IProfiler.__init__(self)
        self.acc = defaultdict(float)
        self.cnt = defaultdict(int)

    def report_layer_time(self, name, ms):
        self.acc[name] += ms
        self.cnt[name] += 1


def profile_engine(path, n_iter=50):
    eng = load_engine(Path(path))
    context, bufs, _, _ = make_ctx(eng)
    prof = LayerProfiler()
    context.profiler = prof
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(20):  # warmup (not profiled meaningfully; reset after)
            context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
    prof.acc.clear(); prof.cnt.clear()
    with torch.cuda.stream(stream):
        for _ in range(n_iter):
            context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
    torch.cuda.synchronize()
    # per-iter ms
    layers = {k: prof.acc[k] / max(prof.cnt[k], 1) for k in prof.acc}
    total = sum(layers.values())
    return layers, total


def classify(name):
    n = name.lower()
    if "reformat" in n or "copy" in n:
        return "reformat/copy"
    if "conv" in n:
        return "conv"
    if "pool" in n:
        return "pool"
    return "other"


def main():
    pynvml.nvmlInit()
    phys = int(os.environ.get("HW_GPU", "7"))
    h = pynvml.nvmlDeviceGetHandleByIndex(phys)
    u, m, o = gpu_status(h)
    if u > 2 or o > 50:
        print(f"ABORT: GPU{phys} not idle util={u}% other_mem={o:.0f}MiB")
        return 2
    print(f"[gate] GPU{phys} idle util={u}% other_mem={o:.0f}MiB — proceed")
    torch.cuda.set_device(0)

    rows = []
    summary = {}
    for tag, rel in TARGETS:
        p = REPO / rel
        if not p.exists():
            print(f"[skip] {tag} missing {rel}")
            continue
        layers, total = profile_engine(str(p))
        # aggregate by class
        by_cls = defaultdict(float)
        for k, v in layers.items():
            by_cls[classify(k)] += v
        n_reformat = sum(1 for k in layers if classify(k) == "reformat/copy")
        summary[tag] = (total, dict(by_cls), n_reformat, len(layers))
        print(f"\n=== {tag} ({rel}) total/iter={total:.4f}ms  "
              f"layers={len(layers)}  reformat_layers={n_reformat} ===")
        for cls, t in sorted(by_cls.items(), key=lambda x: -x[1]):
            print(f"   {cls:14s} {t:.4f}ms  ({t/total*100:4.1f}%)")
        top = sorted(layers.items(), key=lambda x: -x[1])[:8]
        print("   -- top layers --")
        for k, v in top:
            print(f"   {v:.4f}ms  [{classify(k)}]  {k[:90]}")
        for k, v in layers.items():
            rows.append({"engine": tag, "engine_path": rel,
                         "layer": k, "class": classify(k),
                         "ms_per_iter": round(v, 5),
                         "pct_of_total": round(v / total * 100, 2),
                         "engine_total_ms": round(total, 4)})

    pynvml.nvmlShutdown()
    out = REPO / "results/P0_1_p25_layer_profile.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["engine", "engine_path", "layer",
                                          "class", "ms_per_iter", "pct_of_total",
                                          "engine_total_ms"])
        w.writeheader()
        for r in sorted(rows, key=lambda x: (x["engine"], -x["ms_per_iter"])):
            w.writerow(r)
    print(f"\n[done] {len(rows)} layer rows -> {out}")
    print("\n=== reformat/copy overhead comparison (the tactic-cliff signal) ===")
    for tag, (total, by_cls, nref, nl) in summary.items():
        rf = by_cls.get("reformat/copy", 0.0)
        print(f"  {tag:24s} total={total:.4f}ms  reformat={rf:.4f}ms "
              f"({rf/total*100:.1f}%)  #reformat_layers={nref}/{nl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
