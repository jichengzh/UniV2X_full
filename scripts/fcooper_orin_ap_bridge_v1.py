#!/usr/bin/env python3
"""Evaluate full-2170 F-Cooper AP with native or Orin TensorRT dense scope."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import tensorrt as trt
import torch
from torch import nn
from torch.utils.data import DataLoader


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TorchTensorRTRunner:
    def __init__(self, engine_path: Path):
        runtime = trt.Runtime(TRT_LOGGER)
        self.engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize engine: {engine_path}")
        self.context = self.engine.create_execution_context()
        self.inputs = [
            self.engine.get_tensor_name(index)
            for index in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(index))
            == trt.TensorIOMode.INPUT
        ]
        self.outputs = [
            self.engine.get_tensor_name(index)
            for index in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(index))
            == trt.TensorIOMode.OUTPUT
        ]
        if len(self.inputs) != 1 or len(self.outputs) != 1:
            raise ValueError("F-Cooper dense engine must have one input and one output")
        self.input_shape = tuple(self.engine.get_tensor_shape(self.inputs[0]))
        self.output_shape = tuple(self.engine.get_tensor_shape(self.outputs[0]))
        if any(value <= 0 for value in (*self.input_shape, *self.output_shape)):
            raise ValueError("F-Cooper evidence requires static positive engine shapes")
        input_numpy = np.empty(
            (), dtype=trt.nptype(self.engine.get_tensor_dtype(self.inputs[0]))
        )
        output_numpy = np.empty(
            (), dtype=trt.nptype(self.engine.get_tensor_dtype(self.outputs[0]))
        )
        self.input_dtype = torch.from_numpy(input_numpy).dtype
        self.output_dtype = torch.from_numpy(output_numpy).dtype
        self.output = torch.empty(
            self.output_shape, device="cuda", dtype=self.output_dtype
        )
        self.call_count = 0

    def __call__(self, source: torch.Tensor) -> torch.Tensor:
        if source.dtype != self.input_dtype:
            raise ValueError(
                f"engine input dtype drift: {source.dtype} != {self.input_dtype}"
            )
        if not source.is_contiguous():
            source = source.contiguous()
        if not self.context.set_tensor_address(self.inputs[0], source.data_ptr()):
            raise RuntimeError("failed to bind TensorRT input address")
        if not self.context.set_tensor_address(
            self.outputs[0], self.output.data_ptr()
        ):
            raise RuntimeError("failed to bind TensorRT output address")
        stream = torch.cuda.current_stream()
        if not self.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT execute_async_v3 returned false")
        self.call_count += 1
        return self.output


class EngineBackbone(nn.Module):
    def __init__(self, runner: TorchTensorRTRunner):
        super().__init__()
        self.runner = runner

    def forward(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        source = data["spatial_features"]
        agents = int(source.shape[0])
        if agents > self.runner.input_shape[0]:
            raise ValueError(f"record has {agents} agents, engine supports five")
        padded = torch.zeros(
            self.runner.input_shape, device=source.device, dtype=source.dtype
        )
        padded[:agents] = source
        encoded = self.runner(padded)
        return {"spatial_features_2d": encoded[:agents].float()}


class IdentityShrinker(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value


def update_prediction_hash(
    digest: hashlib._Hash, index: int, result: dict[str, torch.Tensor]
) -> None:
    digest.update(index.to_bytes(8, byteorder="little", signed=False))
    for name in ("pred_box_tensor", "pred_score", "gt_box_tensor"):
        value = result[name]
        digest.update(name.encode())
        if value is None:
            digest.update(b"none")
        else:
            array = value.detach().cpu().contiguous().numpy()
            digest.update(str(array.dtype).encode())
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--engine", type=Path)
    parser.add_argument("--build-report", type=Path)
    parser.add_argument("--allow-diagnostic", action="store_true")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from opencood.data_utils.datasets import build_dataset
    from opencood.hypes_yaml import yaml_utils
    from opencood.tools import inference_utils, train_utils
    from opencood.utils import eval_utils

    hypes = yaml_utils.load_yaml(str(args.config), SimpleNamespace(model_dir=None))
    hypes["validate_dir"] = hypes["test_dir"]
    dataset = build_dataset(hypes, visualize=False, train=False)
    if len(dataset) != 2170:
        raise ValueError(f"OPV2V full-test contract drift: {len(dataset)}")
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_batch_test,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )
    model = train_utils.create_model(hypes)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"), strict=True)
    model.cuda().eval()
    runner = None
    build_report = None
    if args.engine is not None:
        if args.build_report is None:
            raise ValueError("TensorRT AP requires --build-report")
        build_report = json.loads(args.build_report.read_text())
        if build_report.get("status") != "formal" and not args.allow_diagnostic:
            raise RuntimeError(
                "non-formal engine rejected; --allow-diagnostic is required"
            )
        runner = TorchTensorRTRunner(args.engine)
        model.backbone_m1 = EngineBackbone(runner)
        model.shrinker_m1 = IdentityShrinker()
    stats = {
        threshold: {"tp": [], "fp": [], "gt": 0, "score": []}
        for threshold in (0.3, 0.5, 0.7)
    }
    prediction_digest = hashlib.sha256()
    processed = failed = fallback = 0
    started = time.time()
    for index, batch in enumerate(loader):
        if batch is None:
            failed += 1
            raise RuntimeError(f"collate returned None at sample {index}")
        batch = train_utils.to_device(batch, torch.device("cuda"))
        with torch.no_grad():
            result = inference_utils.inference_intermediate_fusion(
                batch, model, dataset
            )
        update_prediction_hash(prediction_digest, index, result)
        for threshold in stats:
            eval_utils.caluclate_tp_fp(
                result["pred_box_tensor"],
                result["pred_score"],
                result["gt_box_tensor"],
                stats,
                threshold,
            )
        processed += 1
    ap30, ap50, ap70 = (
        eval_utils.calculate_ap(stats, threshold)[0]
        for threshold in (0.3, 0.5, 0.7)
    )
    engine_calls = runner.call_count if runner is not None else 0
    full_engine_execution = (
        runner is None or runner.call_count == processed == 2170
    )
    report = {
        "schema_version": "fcooper_trt_ap_report_v1",
        "status": (
            "success_full"
            if processed == 2170
            and failed == 0
            and fallback == 0
            and full_engine_execution
            else "failure"
        ),
        "dataset": "OPV2V",
        "split": "test",
        "dataset_samples": len(dataset),
        "processed_samples": processed,
        "failed_samples": failed,
        "fallback_samples": fallback,
        "engine_samples": engine_calls,
        "engine_calls": engine_calls,
        "backend": "tensorrt" if runner is not None else "pytorch_cuda_native",
        "ap30": float(ap30),
        "ap50": float(ap50),
        "ap70": float(ap70),
        "elapsed_seconds": time.time() - started,
        "prediction_stream_sha256": prediction_digest.hexdigest(),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "config_sha256": sha256_file(args.config),
        "engine_sha256": sha256_file(args.engine) if args.engine else None,
        "build_evidence_status": (
            build_report.get("status") if build_report is not None else "native"
        ),
        "numerical_contract": {
            "full_dataset_execution": processed == 2170,
            "full_dataset_engine_execution": full_engine_execution,
            "requested_samples": 2170,
            "silent_fallback_forbidden": True,
            "fallback_samples": fallback,
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
