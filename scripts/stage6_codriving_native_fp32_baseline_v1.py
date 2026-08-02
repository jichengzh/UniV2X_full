#!/usr/bin/env python3
"""Measure fixed CoDriving native-PyTorch FP32 Stage6 reference on H800."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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

from scripts.stage2_h800_true_fp16_ap_eval import best_checkpoint, write_json  # noqa: E402


DEFAULT_REPO_ROOT = Path("/exdata/jichengzhi/V2Xverse_pyramid")
DEFAULT_MODEL_DIR = DEFAULT_REPO_ROOT / "output/codriving_v2_gold_ap_20260709/64x128x256"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    root = path
    for item in files:
        digest.update(item.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with item.open("rb") as handle:
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


def run_ap(args: argparse.Namespace) -> dict[str, Any]:
    from torch.utils.data import DataLoader

    repo_root = Path(args.repo_root).resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    os.chdir(repo_root)
    from opencood.data_utils.datasets import build_dataset
    from opencood.hypes_yaml.yaml_utils import load_yaml
    from opencood.tools import train_utils
    from opencood.utils import eval_utils

    model_dir = Path(args.model_dir).resolve()
    config_path = model_dir / "config.yaml"
    checkpoint_path = best_checkpoint(model_dir)
    hypes = load_yaml(str(config_path))
    hypes["validate_dir"] = hypes["test_dir"]
    dataset_path = Path(str(hypes["validate_dir"])).resolve()
    model = train_utils.create_model(hypes)
    _, model = train_utils.load_saved_model(str(model_dir), model)
    model = model.cuda().eval()
    dataset = build_dataset(hypes, visualize=False, train=False)
    loader = DataLoader(dataset, batch_size=1, num_workers=args.num_workers, collate_fn=dataset.collate_batch_test, shuffle=False, pin_memory=False, drop_last=False)
    result_stat = {iou: {"tp": [], "fp": [], "gt": 0, "score": []} for iou in (0.3, 0.5, 0.7)}
    processed = 0
    started = time.time()
    with torch.inference_mode():
        for batch_data in loader:
            if processed >= args.ap_samples:
                break
            if batch_data is None:
                continue
            batch_data = train_utils.to_device(batch_data, "cuda")
            output = model(batch_data["ego"])
            pred_box, pred_score, gt_box = dataset.post_process(batch_data, {"ego": output})
            for iou in (0.3, 0.5, 0.7):
                eval_utils.caluclate_tp_fp(pred_box, pred_score, gt_box, result_stat, iou)
            processed += 1
    if processed != args.ap_samples:
        raise RuntimeError(f"AP sample contract mismatch: {processed} != {args.ap_samples}")
    args.ap_out.parent.mkdir(parents=True, exist_ok=True)
    eval_dir = args.ap_out.parent / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    ap30, ap50, ap70 = eval_utils.eval_final_results(result_stat, str(eval_dir), "stage6_codriving_native_fp32")
    report = {
        "schema": "stage6_codriving_native_fp32_ap_v1",
        "status": "success",
        "pipeline_scope": "codriving_full_detection_native_pytorch_for_ap",
        "processed_samples": processed,
        "ap30": float(ap30),
        "ap50": float(ap50),
        "ap70": float(ap70),
        "elapsed_secs": time.time() - started,
        "model_dir": str(model_dir),
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "dataset_path": str(dataset_path),
        "sha256": {
            "config": sha256_path(config_path),
            "checkpoint": sha256_path(checkpoint_path),
            "dataset": sha256_path(dataset_path),
        },
    }
    write_json(args.ap_out, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--input-shape", default="2,64,256,512")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--energy-secs", type=float, default=5.0)
    parser.add_argument("--ap-samples", type=int, default=1789)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ap-out", type=Path, required=True)
    args = parser.parse_args()

    if args.repeats < 3:
        raise ValueError("formal native baseline requires at least three independent repeats")
    torch.cuda.set_device(args.gpu)
    repo_root = Path(args.repo_root).resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    os.chdir(repo_root)
    from opencood.hypes_yaml.yaml_utils import load_yaml
    from opencood.tools import train_utils

    model_dir = Path(args.model_dir).resolve()
    config_path = model_dir / "config.yaml"
    checkpoint_path = best_checkpoint(model_dir)
    hypes = load_yaml(str(config_path))
    model = train_utils.create_model(hypes)
    _, model = train_utils.load_saved_model(str(model_dir), model)
    model = model.cuda().eval()
    resnet = model.backbone.resnet.eval()
    shape = tuple(int(value) for value in args.input_shape.split(","))
    torch.manual_seed(42)
    tensor = torch.randn(*shape, device="cuda", dtype=torch.float32)
    with torch.inference_mode():
        outputs = tuple(resnet(tensor))
        repeats = [latency_repeat(resnet, tensor, args.warmup, args.iters) for _ in range(args.repeats)]
        power = measure_power(resnet, tensor, args.gpu, args.energy_secs)
    aggregate_p50 = float(statistics.median(row["p50_ms"] for row in repeats))
    ap_report = run_ap(args)
    payload = {
        "schema_version": "stage6_native_fp32_baseline_v3",
        "hardware": torch.cuda.get_device_name(args.gpu),
        "gpu_abs": args.gpu,
        "scope": "codriving_backbone_resnet",
        "input_shape": list(shape),
        "width": [64, 128, 256],
        "precision": "fp32",
        "backend": "pytorch_eager",
        "backend_tuning": False,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_path(checkpoint_path),
        "output_shapes": [list(output.shape) for output in outputs],
        "independent_repeat_count": len(repeats),
        "latency_repeats": repeats,
        "latency_p50_ms": aggregate_p50,
        "watt_avg": power["watt_mean"],
        "energy_j": power["watt_mean"] * aggregate_p50 / 1000.0,
        "energy_metric": "total_power_x_backbone_latency",
        "power": power,
        "full_network_claim": False,
        "ap_report_path": str(args.ap_out.resolve()),
        "ap_report_sha256": sha256_path(args.ap_out.resolve()),
        "ap_num_samples": int(ap_report["processed_samples"]),
        "ap30": float(ap_report["ap30"]),
        "ap50": float(ap_report["ap50"]),
        "ap70": float(ap_report["ap70"]),
        "ap_evidence_status": "full_h800_backend_execution_bound",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
