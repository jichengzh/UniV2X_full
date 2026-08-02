#!/usr/bin/env python3
"""Measure F-Cooper MaxFusion coverage on a real OPV2V sample."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--nvml-gpu", type=int, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from opencood.data_utils.datasets import build_dataset
    from opencood.hypes_yaml import yaml_utils
    from opencood.tools import train_utils

    hypes = yaml_utils.load_yaml(str(args.config), SimpleNamespace(model_dir=None))
    hypes["validate_dir"] = hypes["test_dir"]
    dataset = build_dataset(hypes, visualize=False, train=False)
    loader = DataLoader(
        Subset(dataset, [args.sample_index]),
        batch_size=1,
        num_workers=0,
        collate_fn=dataset.collate_batch_test,
    )
    model = train_utils.create_model(hypes)
    _, model = train_utils.load_saved_model(str(args.checkpoint_dir), model)
    model.cuda().eval()
    captured = []

    def hook(_module, inputs):
        captured.append(tuple(item.detach() for item in inputs))

    handle = model.fusion_net.register_forward_pre_hook(hook)
    batch = train_utils.to_device(next(iter(loader)), torch.device("cuda"))
    with torch.no_grad():
        model(batch["ego"])
    handle.remove()
    if len(captured) != 1:
        raise RuntimeError("expected one MaxFusion invocation")
    features, record_len, affine = captured[0]

    def run_once() -> torch.Tensor:
        with torch.no_grad():
            return model.fusion_net(features, record_len, affine)

    for _ in range(args.warmup):
        run_once()
    torch.cuda.synchronize()
    samples = []
    for _ in range(args.repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iters):
            output = run_once()
        end.record()
        torch.cuda.synchronize()
        samples.append(float(start.elapsed_time(end)) / args.iters)
    reference = run_once()
    repeat = run_once()
    torch.cuda.synchronize()
    max_abs = float((reference - repeat).abs().max().item())

    import pynvml

    pynvml.nvmlInit()
    device = pynvml.nvmlDeviceGetHandleByIndex(args.nvml_gpu)
    powers = []
    stop = time.time() + 3.0
    while time.time() < stop:
        run_once()
        torch.cuda.synchronize()
        powers.append(pynvml.nvmlDeviceGetPowerUsage(device) / 1000.0)
    pynvml.nvmlShutdown()
    latency = statistics.median(samples)
    output_np = np.ascontiguousarray(reference.detach().cpu().numpy())
    report = {
        "schema_version": "fcooper_maxfusion_probe_v1",
        "status": "success",
        "sample_index": args.sample_index,
        "agents": int(features.shape[0]),
        "channels": int(features.shape[1]),
        "feature_shape": list(features.shape),
        "output_shape": list(reference.shape),
        "latency_ms": latency,
        "latency_repeats_ms": samples,
        "power_w": statistics.mean(powers),
        "energy_j": statistics.mean(powers) * latency / 1000.0,
        "numerical_repeat_max_abs": max_abs,
        "output_sha256": sha256_bytes(output_np.tobytes()),
        "actual_graph_features": {
            "graph_feature_provenance": "runtime_maxfusion_probe_v1",
            "maxfusion_count": 1,
            "fusion_agents": int(features.shape[0]),
            "fusion_channels": int(features.shape[1]),
            "fusion_output_elements": int(reference.numel()),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
