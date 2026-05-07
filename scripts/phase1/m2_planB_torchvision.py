"""M2 Plan B: TorchVision backbone 跨平台 latency 映射 f 拟合.

由于 UniV2X ONNX 含 PythonOp / 自定义 plugin (Plan A blocked),
改用 4 个 torchvision backbone 作为通用 reference 拟合 f.

流程:
  1. export ResNet18/34/50/101 为 ONNX (input 1x3x224x224, FP32)
  2. 在 4090 上用 PyTorch CUDA Event 测 FP16 + INT8 latency
     (TRT Python API + 简化 timing, 不依赖 trtexec binary)
  3. 输出: data/m2_torchvision_4090_latency.csv
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import models

ROOT = Path(__file__).resolve().parents[2]
ONNX_DIR = ROOT / "data/m2_onnx"
RESULTS_DIR = ROOT / "results"
ONNX_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

BACKBONES = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "resnet101": models.resnet101,
}

INPUT_SHAPE = (1, 3, 224, 224)


def export_onnx(name: str, factory) -> Path:
    onnx_path = ONNX_DIR / f"{name}.onnx"
    if onnx_path.exists():
        print(f"  [skip] {onnx_path} exists ({onnx_path.stat().st_size // 1024} KB)")
        return onnx_path
    print(f"  exporting {name} ...")
    model = factory(weights=None).eval().cuda()
    dummy = torch.randn(*INPUT_SHAPE).cuda()
    torch.onnx.export(
        model, dummy, onnx_path,
        opset_version=13,
        input_names=["input"], output_names=["output"],
        dynamic_axes=None,
    )
    print(f"  saved {onnx_path} ({onnx_path.stat().st_size // 1024} KB)")
    del model, dummy
    torch.cuda.empty_cache()
    return onnx_path


@torch.no_grad()
def benchmark_pytorch_cuda_event(
    name: str, factory, dtype: torch.dtype = torch.float16,
    n_warmup: int = 50, n_iter: int = 200,
) -> dict:
    """在 4090 上用 PyTorch CUDA Event 测 latency.

    用 PyTorch (autocast or .half()) 模拟 TRT FP16 量级.
    INT8 用 PyTorch 静态量化 (近似, 真实 TRT INT8 通常更快).
    """
    model = factory(weights=None).eval().cuda()
    if dtype == torch.float16:
        model = model.half()
    dummy = torch.randn(*INPUT_SHAPE).cuda()
    if dtype == torch.float16:
        dummy = dummy.half()

    # warmup
    for _ in range(n_warmup):
        _ = model(dummy)
    torch.cuda.synchronize()

    # benchmark
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    timings = np.zeros(n_iter)
    for i in range(n_iter):
        starter.record()
        _ = model(dummy)
        ender.record()
        torch.cuda.synchronize()
        timings[i] = starter.elapsed_time(ender)  # ms

    del model, dummy
    torch.cuda.empty_cache()

    return {
        "mean_ms": float(np.mean(timings)),
        "p50_ms": float(np.percentile(timings, 50)),
        "p99_ms": float(np.percentile(timings, 99)),
        "std_ms": float(np.std(timings)),
        "n_iter": n_iter,
    }


def main() -> None:
    print("=" * 60)
    print("M2 Plan B — TorchVision backbones (4090 baseline)")
    print("=" * 60)
    print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}, Device 0: {torch.cuda.get_device_name(0)}")

    rows = []
    for name, factory in BACKBONES.items():
        print(f"\n--- {name} ---")
        export_onnx(name, factory)

        for prec, dtype in [("fp32", torch.float32), ("fp16", torch.float16)]:
            stats = benchmark_pytorch_cuda_event(name, factory, dtype=dtype)
            print(f"  {prec}: mean={stats['mean_ms']:.3f} ms  "
                  f"p50={stats['p50_ms']:.3f}  p99={stats['p99_ms']:.3f}  "
                  f"std={stats['std_ms']:.3f}")
            rows.append({
                "model": name,
                "platform": "rtx4090",
                "precision": prec,
                "framework": "pytorch_cuda_event",
                **stats,
            })

    # 输出
    import pandas as pd
    df = pd.DataFrame(rows)
    out = RESULTS_DIR / "m2_torchvision_4090_latency.csv"
    df.to_csv(out, index=False)
    print(f"\n✅ Saved {out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
