#!/usr/bin/env python3
"""Measure an Orin native-PyTorch multiscale backbone with the Stage6 protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

try:
    from .lane_c_backbone_parity_runner import (
        _start_tegrastats,
        _stop_tegrastats,
        build_power_measurement,
        parse_tegrastats,
        summarize_latency_samples,
    )
except ImportError:
    from lane_c_backbone_parity_runner import (
        _start_tegrastats,
        _stop_tegrastats,
        build_power_measurement,
        parse_tegrastats,
        summarize_latency_samples,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_native_input(inputs: np.ndarray) -> np.ndarray:
    array = np.asarray(inputs)
    if array.ndim == 5:
        if array.shape[0] != 1:
            raise ValueError("native latency input file must contain exactly one batch")
        array = array[0]
    if array.ndim != 4 or array.shape[0] != 2 or array.shape[1] != 64:
        raise ValueError(f"invalid native input shape: {list(array.shape)}")
    return array


def load_pyramid(args: argparse.Namespace) -> tuple[Any, Callable[[Any], Any], Path]:
    heal_root = args.repo_root.resolve()
    if str(heal_root) not in sys.path:
        sys.path.insert(0, str(heal_root))
    os.chdir(heal_root)
    import opencood.hypes_yaml.yaml_utils as yaml_utils
    from opencood.tools import train_utils

    opt = argparse.Namespace(
        model_dir=str(args.model_dir),
        fusion_method="intermediate",
        save_vis_interval=10**9,
        save_npy=False,
        range="102.4,51.2",
        no_score=True,
        note="lane_c_native_backbone_orin",
    )
    hypes = yaml_utils.load_yaml(None, opt)
    model = train_utils.create_model(hypes)
    state = torch.load(args.checkpoint, map_location="cpu")
    train_utils.check_missing_key(model.state_dict(), state)
    model.load_state_dict(state, strict=False)
    model = model.cuda().eval()
    return model, model.pyramid_backbone.get_multiscale_feature, args.checkpoint


def load_codriving(args: argparse.Namespace) -> tuple[Any, Callable[[Any], Any], Path]:
    repo_root = args.repo_root.resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    os.chdir(repo_root)
    from opencood.hypes_yaml.yaml_utils import load_yaml
    from opencood.tools import train_utils

    hypes = load_yaml(str(args.model_dir / "config.yaml"))
    model = train_utils.create_model(hypes)
    _, model = train_utils.load_saved_model(str(args.model_dir), model)
    model = model.cuda().eval()
    return model, model.backbone.resnet, args.checkpoint


def measure_callable(
    function: Callable[[torch.Tensor], Any],
    tensor: torch.Tensor,
    *,
    warmup: int,
    iters: int,
    repeat: int,
) -> tuple[list[float], list[dict[str, Any]], float]:
    if (warmup, iters, repeat) != (20, 300, 5):
        raise ValueError("primary protocol requires warmup=20, iters=300, repeat=5")
    with torch.inference_mode():
        for _ in range(warmup):
            function(tensor)
        torch.cuda.synchronize()
        all_samples: list[float] = []
        repeats = []
        started = time.perf_counter()
        for repeat_index in range(repeat):
            samples = []
            for _ in range(iters):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                function(tensor)
                end.record()
                end.synchronize()
                samples.append(float(start.elapsed_time(end)))
            all_samples.extend(samples)
            repeats.append(
                {
                    "repeat_index": repeat_index,
                    "sample_count": len(samples),
                    **summarize_latency_samples(samples),
                }
            )
        elapsed = time.perf_counter() - started
    return all_samples, repeats, elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("pyramid", "codriving"), required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--inputs-npy", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--power-log", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    array = select_native_input(np.load(args.inputs_npy, allow_pickle=False))
    tensor = torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32)).cuda()
    loader = load_pyramid if args.model == "pyramid" else load_codriving
    model, function, checkpoint = loader(args)
    with torch.inference_mode():
        outputs = tuple(function(tensor))
    expected_spatial = (
        (128, 256), (64, 128), (32, 64)
    ) if args.model == "pyramid" else (
        (128, 256), (64, 128), (32, 64)
    )
    if len(outputs) != 3 or tuple(tuple(item.shape[-2:]) for item in outputs) != expected_spatial:
        raise ValueError("native backbone output contract mismatch")

    process, handle, privileged = _start_tegrastats(args.power_log)
    try:
        samples, repeats, elapsed = measure_callable(
            function,
            tensor,
            warmup=args.warmup,
            iters=args.iters,
            repeat=args.repeat,
        )
    finally:
        _stop_tegrastats(process, handle, privileged=privileged)
    raw_power = args.power_log.read_text(encoding="utf-8")
    try:
        power = parse_tegrastats(raw_power)
    except ValueError:
        power = None
    latency = summarize_latency_samples(samples)
    power_measurement = build_power_measurement(
        power=power,
        power_source="tegrastats",
        latency_summary=latency,
        raw_power_sha256=sha256_file(args.power_log),
        used_sudo=privileged,
    )
    report = {
        "schema_version": "lane_c_stage6_orin_native_backbone_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "runtime": "native_pytorch",
        "precision": "fp32",
        "scope": f"{args.model}_multiscale_backbone_compute_no_data_transfer",
        "protocol": {
            "batch": 2,
            "warmup": args.warmup,
            "iters": args.iters,
            "repeat": args.repeat,
            "timing": "CUDA_event",
            "data_transfer_inside_timed_region": False,
        },
        "input_shape": list(array.shape),
        "input_sha256": sha256_file(args.inputs_npy),
        "output_shapes": [list(item.shape) for item in outputs],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "sample_count": len(samples),
        "measurement_wall_seconds": elapsed,
        **latency,
        "per_repeat": repeats,
        "power_measurement": power_measurement,
        "power_log_sha256": sha256_file(args.power_log),
        "hardware": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "model_kept_alive": bool(model is not None),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
