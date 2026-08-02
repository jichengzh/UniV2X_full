#!/usr/bin/env python3
"""Measure the fixed Pyramid native-PyTorch FP32 Stage6 reference on H800."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.stage2_h800_export_checkpoint_multiscale_onnx import (  # noqa: E402
    PyramidMultiscaleBackbone,
    load_heal_model,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def latency_repeat(model: Any, tensor: torch.Tensor, warmup: int, iters: int) -> dict[str, float]:
    for _ in range(warmup):
        model(tensor)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = np.empty(iters, dtype=np.float64)
    for index in range(iters):
        start.record()
        model(tensor)
        end.record()
        end.synchronize()
        samples[index] = start.elapsed_time(end)
    return {
        "mean_ms": float(samples.mean()),
        "p50_ms": float(np.percentile(samples, 50)),
        "p90_ms": float(np.percentile(samples, 90)),
        "p99_ms": float(np.percentile(samples, 99)),
        "std_ms": float(samples.std()),
    }


def measure_power(model: Any, tensor: torch.Tensor, gpu: int, min_seconds: float) -> dict[str, float | int]:
    import pynvml

    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu)
        samples = []
        deadline = time.monotonic() + min_seconds
        iterations = 0
        while time.monotonic() < deadline or iterations < 300:
            model(tensor)
            iterations += 1
            if iterations % 10 == 0:
                torch.cuda.synchronize()
                samples.append(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0)
        torch.cuda.synchronize()
    finally:
        pynvml.nvmlShutdown()
    return {
        "watt_mean": float(statistics.mean(samples)),
        "watt_p50": float(statistics.median(samples)),
        "power_sample_count": len(samples),
        "active_iterations": iterations,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--input-shape", default="2,64,128,256")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--energy-secs", type=float, default=5.0)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = args.out.resolve()
    if args.repeats < 3:
        raise ValueError("formal native baseline requires at least three independent repeats")
    torch.cuda.set_device(args.gpu)
    load_args = argparse.Namespace(
        heal_root="/home/jichengzhi/heal_research/HEAL",
        ckpt_dir=str(args.ckpt_dir),
        checkpoint_path=str(args.checkpoint_path),
        eval_range="102.4,51.2",
    )
    full_model, checkpoint_epoch, checkpoint = load_heal_model(load_args)
    model = PyramidMultiscaleBackbone(full_model).cuda().eval()
    shape = tuple(int(value) for value in args.input_shape.split(","))
    torch.manual_seed(42)
    tensor = torch.randn(*shape, device="cuda", dtype=torch.float32)
    with torch.inference_mode():
        outputs = tuple(model(tensor))
        repeats = [
            latency_repeat(model, tensor, args.warmup, args.iters)
            for _ in range(args.repeats)
        ]
        power = measure_power(model, tensor, args.gpu, args.energy_secs)
    aggregate_p50 = float(statistics.median(row["p50_ms"] for row in repeats))
    payload = {
        "schema_version": "stage6_native_fp32_baseline_v1",
        "hardware": torch.cuda.get_device_name(args.gpu),
        "gpu_abs": args.gpu,
        "scope": "pyramid_multiscale_backbone",
        "input_shape": list(shape),
        "width": [64, 128, 256],
        "precision": "fp32",
        "backend": "pytorch_eager",
        "backend_tuning": False,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(Path(checkpoint)),
        "checkpoint_epoch": checkpoint_epoch,
        "output_shapes": [list(output.shape) for output in outputs],
        "independent_repeat_count": len(repeats),
        "latency_repeats": repeats,
        "latency_p50_ms": aggregate_p50,
        "watt_avg": power["watt_mean"],
        "energy_j": power["watt_mean"] * aggregate_p50 / 1000.0,
        "energy_metric": "total_power_x_latency",
        "power": power,
        "full_network_claim": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
