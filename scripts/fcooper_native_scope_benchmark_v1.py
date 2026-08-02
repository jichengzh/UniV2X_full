#!/usr/bin/env python3
"""Benchmark the native FP32 F-Cooper dense body on one H800 GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import numpy as np
import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_measurements(
    latency_repeats_ms: Sequence[float], power_samples_w: Sequence[float]
) -> dict[str, float]:
    if not latency_repeats_ms or not power_samples_w:
        raise ValueError("latency and power measurements must be non-empty")
    latency = float(statistics.median(latency_repeats_ms))
    power = float(statistics.mean(power_samples_w))
    return {
        "latency_ms": latency,
        "power_w": power,
        "energy_j": power * latency / 1000.0,
    }


def gpu_binding(nvml_gpu: int, visible_devices: str | None) -> dict[str, object]:
    return {
        "gpu_abs": int(nvml_gpu),
        "cuda_visible_devices": visible_devices,
        "cuda_device_index": 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--nvml-gpu", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--energy-secs", type=float, default=5.0)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from opencood.hypes_yaml import yaml_utils
    from opencood.tools import train_utils
    from fcooper_materialize_source_v1 import FCooperDenseEncoder

    hypes = yaml_utils.load_yaml(
        str(args.config), SimpleNamespace(model_dir=None)
    )
    model = train_utils.create_model(hypes)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state, strict=True)
    dense = FCooperDenseEncoder(model).cuda().eval()
    generator = torch.Generator(device="cpu").manual_seed(20260723)
    sample = torch.randn(
        5, 64, 512, 512, generator=generator, dtype=torch.float32
    ).cuda()

    def run_once() -> torch.Tensor:
        with torch.no_grad():
            return dense(sample)

    for _ in range(args.warmup):
        run_once()
    torch.cuda.synchronize()
    latency_repeats = []
    for _ in range(args.repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iters):
            output = run_once()
        end.record()
        torch.cuda.synchronize()
        latency_repeats.append(float(start.elapsed_time(end)) / args.iters)
    reference = run_once()
    repeated = run_once()
    torch.cuda.synchronize()
    numerical_repeat_max_abs = float((reference - repeated).abs().max().item())

    import pynvml

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(args.nvml_gpu)
    powers = []
    stop = time.monotonic() + args.energy_secs
    while time.monotonic() < stop:
        run_once()
        torch.cuda.synchronize()
        powers.append(float(pynvml.nvmlDeviceGetPowerUsage(handle)) / 1000.0)
    pynvml.nvmlShutdown()
    summary = summarize_measurements(latency_repeats, powers)
    output_np = np.ascontiguousarray(reference.detach().cpu().numpy())
    report = {
        "schema_version": "fcooper_native_scope_benchmark_v1",
        "status": "success",
        "backend": "pytorch_cuda_cudnn",
        "precision": "fp32",
        "optimized_scope": "post_scatter_backbone_shrinker",
        "input_shape": list(sample.shape),
        "output_shape": list(reference.shape),
        "warmup": args.warmup,
        "iters": args.iters,
        "repeat": args.repeat,
        "latency_repeats_ms": latency_repeats,
        "power_sample_count": len(powers),
        "numerical_repeat_max_abs": numerical_repeat_max_abs,
        "output_sha256": hashlib.sha256(output_np.tobytes()).hexdigest(),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "config_sha256": sha256_file(args.config),
        **gpu_binding(args.nvml_gpu, os.environ.get("CUDA_VISIBLE_DEVICES")),
        **summary,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
