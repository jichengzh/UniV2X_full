"""Phase A.2 — PyTorch sub-module baseline benchmark.

Measures the *exact same sub-module* as Phase A.2 TRT engine
(pyramid_backbone + shrink_conv + cls/reg/dir heads) so the FP16/INT8
acceleration ratios are fair.

Anchors:
    PyTorch FP32 (eager mode)
    PyTorch FP16 (model.half(), input.half())
    PyTorch FP16 (autocast)        # what M4.6.0 measured

Output: results/m4_8_pytorch_subnet_bench.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.export_onnx_pyramid import build_pyramid_from_ckpt, PyramidSubnet  # noqa: E402

CKPT = "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_m1_base_2023_08_14_04_28_12/net_epoch_bestval_at23.pth"
HYPES = "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_m1_base_2023_08_14_04_28_12/config.yaml"


def bench(name: str, fn, n_warmup=200, n_measure=500) -> dict:
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    print(f"[{name}] warmup {n_warmup} ...")
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    print(f"[{name}] measure {n_measure} ...")
    t = np.empty(n_measure, dtype=np.float64)
    for k in range(n_measure):
        start_evt.record()
        fn()
        end_evt.record()
        end_evt.synchronize()
        t[k] = start_evt.elapsed_time(end_evt)
    s = {
        "anchor": name,
        "n_warmup": n_warmup,
        "n_measure": n_measure,
        "mean_ms": float(t.mean()),
        "std_ms": float(t.std()),
        "p50_ms": float(np.percentile(t, 50)),
        "p95_ms": float(np.percentile(t, 95)),
        "p99_ms": float(np.percentile(t, 99)),
        "min_ms": float(t.min()),
        "max_ms": float(t.max()),
    }
    print(f"[{name}] mean={s['mean_ms']:.3f}ms p50={s['p50_ms']:.3f} "
          f"p95={s['p95_ms']:.3f} p99={s['p99_ms']:.3f} std={s['std_ms']:.3f}")
    return s


def main():
    torch.manual_seed(42)
    print("[1] building model + subnet")
    full = build_pyramid_from_ckpt(HYPES, CKPT, device="cuda")
    subnet_fp32 = PyramidSubnet(full).cuda().eval()
    x = torch.randn(1, 64, 256, 256, device="cuda")

    results = []

    # FP32
    with torch.inference_mode():
        results.append(bench("pytorch_fp32", lambda: subnet_fp32(x)))

    # FP16 autocast (matches M4.6.0 inference path)
    with torch.inference_mode():
        results.append(bench(
            "pytorch_fp16_autocast",
            lambda: subnet_fp32(x) if False else _autocast_fwd(subnet_fp32, x),
        ))

    # FP16 .half() (true fp16 weights+input — what TRT FP16 corresponds to)
    subnet_fp16 = PyramidSubnet(full).cuda().eval().half()
    x_h = x.half()
    with torch.inference_mode():
        results.append(bench("pytorch_fp16_half", lambda: subnet_fp16(x_h)))

    out = REPO_ROOT / "results/m4_8_pytorch_subnet_bench.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"anchors": results, "input_shape": [1, 64, 256, 256]}, f, indent=2)
    print(f"\n  -> {out}")

    # Summary table
    print("\n=== Summary (sub-module pyramid_backbone + shrink + heads only) ===")
    print(f"{'anchor':<25} {'p50':>8} {'mean':>8} {'p99':>8}")
    for r in results:
        print(f"{r['anchor']:<25} {r['p50_ms']:>7.2f}ms {r['mean_ms']:>7.2f}ms {r['p99_ms']:>7.2f}ms")


def _autocast_fwd(net, x):
    with torch.cuda.amp.autocast(dtype=torch.float16):
        return net(x)


if __name__ == "__main__":
    main()
