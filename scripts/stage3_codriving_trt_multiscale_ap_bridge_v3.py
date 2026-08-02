#!/usr/bin/env python3
"""Run CoDriving AP with TensorRT replacing only ``model.backbone.resnet``."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.stage2_h800_true_fp16_ap_eval import best_checkpoint, write_json  # noqa: E402
from stage3_trt_multiscale_ap_bridge_v3 import (  # noqa: E402
    TrtMultiscaleRunner,
    output_error_record,
    prepare_spatial_features_for_engine,
    slice_output_to_record_len,
    sort_output_specs,
    summarize_error_records,
)


SCHEMA = "stage3_codriving_trt_multiscale_ap_bridge_v3"
PIPELINE_SCOPE = "codriving_resnet_trt_in_full_pytorch_fusion_head_postprocess"
DEFAULT_REPO_ROOT = Path("/exdata/jichengzhi/V2Xverse_pyramid")
SANITY_SAMPLES = 16
FULL_SAMPLES = 1789


def sha256_path(path: Path) -> str:
    """Hash a file or a directory tree, including relative file names."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
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


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def protocol_payload(*, precision_tag: str) -> dict[str, Any]:
    return {
        "schema": f"{SCHEMA}_protocol_v1",
        "precision_tag": str(precision_tag),
        "engine_batch": 2,
        "engine_inputs": 1,
        "engine_outputs": 3,
        "patch_target": "model.backbone.resnet",
        "pipeline_scope": PIPELINE_SCOPE,
        "fallback_policy": "forbidden",
        "sanity_samples": SANITY_SAMPLES,
        "full_samples": FULL_SAMPLES,
    }


class CoDrivingTrtResnetBridge(torch.nn.Module):
    """PyTorch-compatible ResNet replacement backed by a fixed-batch TRT engine."""

    def __init__(self, *, reference_resnet: Callable[[Any], Any], trt_runner: Callable[[np.ndarray], Any]) -> None:
        super().__init__()
        self.reference_resnet = reference_resnet
        self.trt_runner = trt_runner
        self.output_error_records: list[dict[str, Any]] = []
        self.call_count = 0

    def forward(self, spatial_features: Any) -> tuple[Any, ...]:
        import torch

        record_len = int(spatial_features.shape[0])
        reference_outputs = tuple(self.reference_resnet(spatial_features))
        source = spatial_features.detach().to(dtype=torch.float32, device="cpu").numpy()
        engine_input, _ = prepare_spatial_features_for_engine(source, record_len=record_len, engine_batch=2)
        runner_outputs = list(self.trt_runner(engine_input))
        if len(runner_outputs) != 3:
            raise RuntimeError(f"expected exactly three TensorRT outputs, got {len(runner_outputs)}")
        by_name = {str(name): np.asarray(value) for name, value in runner_outputs}
        specs = sort_output_specs([{"name": name, "shape": list(value.shape)} for name, value in by_name.items()])
        outputs = []
        for output_index, spec in enumerate(specs):
            name = str(spec["name"])
            sliced = slice_output_to_record_len(by_name[name], record_len=record_len)
            self.output_error_records.append(
                {
                    **output_error_record(
                        name, reference_outputs[output_index], sliced
                    ),
                    "call_index": self.call_count,
                    "output_index": output_index,
                }
            )
            outputs.append(torch.from_numpy(sliced).to(device=spatial_features.device, dtype=torch.float32))
        self.call_count += 1
        return tuple(outputs)

def patch_model_resnet(model: Any, trt_runner: Callable[[np.ndarray], Any]) -> CoDrivingTrtResnetBridge:
    if not hasattr(model, "backbone") or not hasattr(model.backbone, "resnet"):
        raise AttributeError("CoDriving model must expose model.backbone.resnet")
    bridge = CoDrivingTrtResnetBridge(reference_resnet=model.backbone.resnet, trt_runner=trt_runner)
    model.backbone.resnet = bridge
    return bridge


def evaluate_gates(
    *,
    processed_samples: int,
    engine_samples: int,
    fallback_samples: int,
    failed_samples: int,
    output_error_summary: dict[str, Any],
) -> dict[str, Any]:
    blockers: list[str] = []
    if int(fallback_samples):
        blockers.append("fallback_forbidden")
    if int(failed_samples):
        blockers.append("failed_samples_present")
    if int(engine_samples) != int(processed_samples):
        blockers.append("engine_samples_mismatch")
    expected_records = int(processed_samples) * 6
    if output_error_summary.get("all_finite") is not True:
        blockers.append("output_not_all_finite")
    if int(output_error_summary.get("shape_mismatch_count", -1)) != 0:
        blockers.append("output_shape_mismatch")
    if int(output_error_summary.get("nonfinite_count", -1)) != 0:
        blockers.append("output_nonfinite")
    if int(output_error_summary.get("num_records", -1)) != expected_records:
        blockers.append("output_record_count_mismatch")
    if int(output_error_summary.get("num_compared", -1)) != expected_records:
        blockers.append("output_compared_count_mismatch")
    if int(processed_samples) < FULL_SAMPLES:
        blockers.append("full_samples_below_1789")
    clean_engine_path = not any(item != "full_samples_below_1789" for item in blockers)
    return {
        "sanity_16": int(processed_samples) >= SANITY_SAMPLES and clean_engine_path,
        "full_1789": int(processed_samples) >= FULL_SAMPLES and not blockers,
        "blockers": blockers,
    }


def build_report(
    *, model_dir: Path, engine_path: Path, checkpoint_path: Path, config_path: Path, dataset_path: Path,
    shas: dict[str, str], protocol: dict[str, Any], processed_samples: int, engine_samples: int,
    fallback_samples: int, failed_samples: int, ap30: float, ap50: float, ap70: float,
    output_error_summary: dict[str, Any], elapsed_secs: float, engine_calls: int | None = None,
) -> dict[str, Any]:
    bound_shas = {**dict(shas), "protocol": sha256_json(protocol)}
    report = {
        "schema": SCHEMA,
        "status": "success",
        "pipeline_scope": PIPELINE_SCOPE,
        "model_dir": str(model_dir),
        "engine_path": str(engine_path),
        "checkpoint_path": str(checkpoint_path),
        "config_path": str(config_path),
        "dataset_path": str(dataset_path),
        "sha256": bound_shas,
        "protocol": protocol,
        "processed_samples": int(processed_samples),
        "engine_samples": int(engine_samples),
        "fallback_samples": int(fallback_samples),
        "failed_samples": int(failed_samples),
        "ap30": float(ap30),
        "ap50": float(ap50),
        "ap70": float(ap70),
        "output_vs_reference_error": dict(output_error_summary),
        "gates": evaluate_gates(
            processed_samples=processed_samples, engine_samples=engine_samples,
            fallback_samples=fallback_samples, failed_samples=failed_samples,
            output_error_summary=output_error_summary,
        ),
        "elapsed_secs": float(elapsed_secs),
    }
    if engine_calls is not None:
        report["engine_calls"] = int(engine_calls)
        report["engine_calls_per_sample"] = float(engine_calls) / int(processed_samples)
    return report


def run_bridge(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    resolve_output_paths(args)
    repo_root = Path(args.repo_root).resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    os.chdir(repo_root)
    from opencood.data_utils.datasets import build_dataset
    from opencood.hypes_yaml.yaml_utils import load_yaml
    from opencood.tools import train_utils
    from opencood.utils import eval_utils

    model_dir = Path(args.model_dir).resolve()
    engine_path = Path(args.engine).resolve()
    config_path = model_dir / "config.yaml"
    checkpoint_path = best_checkpoint(model_dir)
    hypes = load_yaml(str(config_path))
    hypes["validate_dir"] = hypes["test_dir"]
    dataset_path = Path(str(hypes["validate_dir"])).resolve()
    model = train_utils.create_model(hypes)
    _, model = train_utils.load_saved_model(str(model_dir), model)
    model = model.cuda().eval()
    bridge = patch_model_resnet(model, TrtMultiscaleRunner(engine_path))

    dataset = build_dataset(hypes, visualize=False, train=False)
    loader = DataLoader(dataset, batch_size=1, num_workers=args.num_workers,
                        collate_fn=dataset.collate_batch_test, shuffle=False, pin_memory=False, drop_last=False)
    result_stat = {iou: {"tp": [], "fp": [], "gt": 0, "score": []} for iou in (0.3, 0.5, 0.7)}
    processed = 0
    started = time.time()
    with torch.inference_mode():
        for batch_data in loader:
            if processed >= args.n_samples:
                break
            if batch_data is None:
                continue
            batch_data = train_utils.to_device(batch_data, "cuda")
            output = model(batch_data["ego"])
            pred_box, pred_score, gt_box = dataset.post_process(batch_data, {"ego": output})
            for iou in (0.3, 0.5, 0.7):
                eval_utils.caluclate_tp_fp(pred_box, pred_score, gt_box, result_stat, iou)
            processed += 1

    if processed == 0:
        raise RuntimeError("no CoDriving samples were processed")
    args.eval_dir.mkdir(parents=True, exist_ok=True)
    ap30, ap50, ap70 = eval_utils.eval_final_results(result_stat, str(args.eval_dir), args.precision_tag)
    protocol = protocol_payload(precision_tag=args.precision_tag)
    report = build_report(
        model_dir=model_dir, engine_path=engine_path, checkpoint_path=checkpoint_path,
        config_path=config_path, dataset_path=dataset_path,
        shas={"engine": sha256_path(engine_path), "checkpoint": sha256_path(checkpoint_path),
              "config": sha256_path(config_path), "dataset": sha256_path(dataset_path)},
        protocol=protocol, processed_samples=processed, engine_samples=processed,
        engine_calls=bridge.call_count,
        fallback_samples=0, failed_samples=0, ap30=ap30, ap50=ap50, ap70=ap70,
        output_error_summary=summarize_error_records(bridge.output_error_records),
        elapsed_secs=time.time() - started,
    )
    write_json(args.out_json, report)
    return report


def resolve_output_paths(args: argparse.Namespace) -> None:
    args.eval_dir = Path(args.eval_dir).resolve()
    args.out_json = Path(args.out_json).resolve()
    args.engine = Path(args.engine).resolve()
    args.model_dir = Path(args.model_dir).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--precision-tag", required=True)
    parser.add_argument("--gate", choices=("sanity", "full"), default="sanity")
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()
    expected = SANITY_SAMPLES if args.gate == "sanity" else FULL_SAMPLES
    args.n_samples = expected if args.n_samples is None else args.n_samples
    if args.n_samples != expected:
        parser.error(f"--gate {args.gate} requires --n-samples {expected}")
    return args


def main() -> int:
    args = parse_args()
    try:
        report = run_bridge(args)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        blocker = {"schema": f"{SCHEMA}_blocker_v1", "status": "failed",
                   "failure_reason": f"{type(exc).__name__}:{exc}", "traceback": traceback.format_exc()}
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.out_json, blocker)
        print(json.dumps(blocker, indent=2, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
