#!/usr/bin/env python3
"""Build and measure the frozen F-Cooper dense scope on Jetson Orin."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

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


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def load_model(config: Path, checkpoint: Path) -> tuple[nn.Module, dict[str, Any]]:
    from opencood.hypes_yaml import yaml_utils
    from opencood.tools import train_utils

    hypes = yaml_utils.load_yaml(str(config), SimpleNamespace(model_dir=None))
    model = train_utils.create_model(hypes)
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state, strict=True)
    return model.cuda().eval(), hypes


class DenseEncoder(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.backbone = model.backbone_m1
        self.shrinker = model.shrinker_m1

    def forward(self, spatial_features: torch.Tensor) -> torch.Tensor:
        encoded = self.backbone({"spatial_features": spatial_features})
        return self.shrinker(encoded["spatial_features_2d"])


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

    def __call__(self, source: torch.Tensor) -> torch.Tensor:
        if source.dtype != self.input_dtype:
            raise ValueError(
                f"engine input dtype drift: {source.dtype} != {self.input_dtype}"
            )
        if not source.is_contiguous():
            raise ValueError("engine input must be contiguous before timed execution")
        if tuple(source.shape) != self.input_shape:
            raise ValueError(f"engine input shape drift: {tuple(source.shape)}")
        if not self.context.set_tensor_address(self.inputs[0], source.data_ptr()):
            raise RuntimeError("failed to bind TensorRT input address")
        if not self.context.set_tensor_address(
            self.outputs[0], self.output.data_ptr()
        ):
            raise RuntimeError("failed to bind TensorRT output address")
        stream = torch.cuda.current_stream()
        if not self.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT execute_async_v3 returned false")
        return self.output


def numeric_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    if reference.shape != candidate.shape:
        raise ValueError(f"shape mismatch: {reference.shape} != {candidate.shape}")
    ref = reference.astype(np.float64, copy=False).ravel()
    cand = candidate.astype(np.float64, copy=False).ravel()
    finite = np.isfinite(ref) & np.isfinite(cand)
    difference = cand[finite] - ref[finite]
    ref_finite = ref[finite]
    cand_finite = cand[finite]
    denominator = np.linalg.norm(ref_finite) * np.linalg.norm(cand_finite)
    cosine = float(np.dot(ref_finite, cand_finite) / denominator) if denominator else 1.0
    rmse = float(np.sqrt(np.mean(np.square(difference))))
    reference_rms = float(np.sqrt(np.mean(np.square(ref_finite))))
    return {
        "shape": list(reference.shape),
        "reference_dtype": str(reference.dtype),
        "candidate_dtype": str(candidate.dtype),
        "cosine": cosine,
        "nrmse": rmse / max(reference_rms, np.finfo(np.float64).eps),
        "mae": float(np.mean(np.abs(difference))),
        "max_abs": float(np.max(np.abs(difference))),
        "finite_ratio": float(np.mean(finite)),
        "reference_min": float(np.min(ref_finite)),
        "reference_max": float(np.max(ref_finite)),
        "candidate_min": float(np.min(cand_finite)),
        "candidate_max": float(np.max(cand_finite)),
        "reference_zero_ratio": float(np.mean(ref_finite == 0)),
        "candidate_zero_ratio": float(np.mean(cand_finite == 0)),
    }


def summarize_latency(repeats: list[list[float]]) -> dict[str, Any]:
    def summary(values: np.ndarray) -> dict[str, float]:
        return {
            "median_ms": float(np.median(values)),
            "p90_ms": float(np.percentile(values, 90)),
            "p99_ms": float(np.percentile(values, 99)),
            "mean_ms": float(np.mean(values)),
        }

    aggregate = np.concatenate([np.asarray(row) for row in repeats])
    return {
        "protocol": {
            "warmup": 20,
            "iterations": 300,
            "repeat": 5,
            "timing": "cuda_event",
            "scope": "compute_only_no_data_transfer",
            "sample_batch": 1,
            "dense_agent_batch": 5,
        },
        "sample_count": int(aggregate.size),
        "repeats": [
            {"repeat": index, **summary(np.asarray(row))}
            for index, row in enumerate(repeats)
        ],
        "aggregate": summary(aggregate),
        "raw_ms": repeats,
    }


def validate_inspector(inspector_text: str, precision: str) -> dict[str, Any]:
    normalized = inspector_text.lower()
    reduced = {
        "fp16": "fp16" in normalized or "half" in normalized,
        "int8": "int8" in normalized,
        "bf16": "bf16" in normalized or "bfloat16" in normalized,
        "tf32": "tf32" in normalized,
    }
    if precision == "fp32":
        valid = not any(reduced.values()) and (
            "fp32" in normalized or "float" in normalized
        )
    else:
        valid = reduced["fp16"] and not reduced["int8"] and not reduced["bf16"]
    return {
        "expected_precision": precision,
        "tf32_disabled": True,
        "findings": reduced,
        "valid": bool(valid),
    }


def require_diagnostic_authorization(
    build_report: Path | None, allow_diagnostic: bool
) -> dict[str, Any] | None:
    if build_report is None:
        raise ValueError("TensorRT measurement requires --build-report")
    report = json.loads(build_report.read_text())
    if report.get("status") != "formal" and not allow_diagnostic:
        raise RuntimeError(
            "non-formal engine rejected; pass --allow-diagnostic only for "
            "explicitly downgraded evidence"
        )
    return report


def command_build(args: argparse.Namespace) -> None:
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)
    source = args.onnx.read_bytes()
    if not parser.parse(source):
        errors = [str(parser.get_error(index)) for index in range(parser.num_errors)]
        raise RuntimeError("ONNX parse failed:\n" + "\n".join(errors))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_gib << 30)
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    config.clear_flag(trt.BuilderFlag.TF32)
    if args.precision == "fp16":
        if not builder.platform_has_fast_fp16:
            raise RuntimeError("Orin reports no fast FP16 support")
        config.set_flag(trt.BuilderFlag.FP16)
    fresh_cache = config.create_timing_cache(b"")
    config.set_timing_cache(fresh_cache, False)
    builder_level_supported = hasattr(config, "builder_optimization_level")
    builder_level_applied = False
    if builder_level_supported:
        config.builder_optimization_level = args.builder_level
        builder_level_applied = True
    started = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT build returned no serialized network")
    args.engine.parent.mkdir(parents=True, exist_ok=True)
    args.engine.write_bytes(bytes(serialized))
    cache = config.get_timing_cache()
    args.cache.write_bytes(bytes(cache.serialize()))
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(args.engine.read_bytes())
    if engine is None:
        raise RuntimeError("built engine cannot be deserialized")
    inspector = engine.create_engine_inspector()
    inspector_text = inspector.get_engine_information(
        trt.LayerInformationFormat.JSON
    )
    args.inspector.write_text(inspector_text + "\n")
    precision_audit = validate_inspector(inspector_text, args.precision)
    if not precision_audit["valid"]:
        raise RuntimeError(f"inspector precision audit failed: {precision_audit}")
    report = {
        "schema_version": "fcooper_orin_engine_build_v1",
        "status": "formal" if builder_level_applied else "diagnostic_only",
        "diagnostic_reason": (
            None
            if builder_level_applied
            else "tensorrt_8_5_builder_optimization_level_unsupported"
        ),
        "tensorrt_version": trt.__version__,
        "precision": args.precision,
        "tf32_disabled": True,
        "requested_builder_level": args.builder_level,
        "builder_level_supported": builder_level_supported,
        "builder_level_applied": builder_level_applied,
        "onnx_sha256": sha256_file(args.onnx),
        "engine_sha256": sha256_file(args.engine),
        "timing_cache_sha256": sha256_file(args.cache),
        "inspector_sha256": sha256_file(args.inspector),
        "inspector_precision_audit": precision_audit,
        "build_seconds": time.time() - started,
        "input_shape": list(engine.get_tensor_shape("spatial_features")),
        "output_shape": list(engine.get_tensor_shape("encoded_features")),
    }
    write_json(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))


def command_capture(args: argparse.Namespace) -> None:
    from opencood.data_utils.datasets import build_dataset
    from opencood.tools import train_utils

    if sha256_file(args.test_manifest) != args.expected_manifest_sha:
        raise ValueError("frozen OPV2V test manifest SHA mismatch")
    model, hypes = load_model(args.config, args.checkpoint)
    hypes["validate_dir"] = hypes["test_dir"]
    dataset = build_dataset(hypes, visualize=False, train=False)
    if len(dataset) != 2170:
        raise ValueError(f"OPV2V full-test contract drift: {len(dataset)}")
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        collate_fn=dataset.collate_batch_test,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )
    batch = next(iter(loader))
    batch = train_utils.to_device(batch, torch.device("cuda"))
    captured: dict[str, torch.Tensor] = {}

    def pre_hook(_module: nn.Module, hook_args: tuple[Any, ...]) -> None:
        captured["input"] = hook_args[0]["spatial_features"].detach()

    handle = model.backbone_m1.register_forward_pre_hook(pre_hook)
    with torch.no_grad():
        model(batch["ego"])
    handle.remove()
    source = captured["input"]
    if source.shape[0] > 5:
        raise ValueError(f"held-out sample has {source.shape[0]} agents")
    padded = torch.zeros((5, *source.shape[1:]), device="cuda", dtype=source.dtype)
    padded[: source.shape[0]] = source
    dense = DenseEncoder(model).eval()
    with torch.no_grad():
        reference = dense(padded)
    args.input.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.input, padded.cpu().numpy())
    np.save(args.reference, reference.cpu().numpy())
    report = {
        "schema_version": "fcooper_orin_heldout_capture_v1",
        "status": "success",
        "dataset": "OPV2V",
        "split": "test",
        "dataset_samples": len(dataset),
        "sample_index": 0,
        "source": "real_opv2v_post_scatter",
        "heldout_basis": "frozen_opv2v_test_split_not_used_by_training_or_tuning",
        "test_manifest_sha256": sha256_file(args.test_manifest),
        "actual_agents": int(source.shape[0]),
        "padded_agents": 5,
        "input_shape": list(padded.shape),
        "reference_shape": list(reference.shape),
        "input_sha256": sha256_file(args.input),
        "reference_sha256": sha256_file(args.reference),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "config_sha256": sha256_file(args.config),
    }
    write_json(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))


def _benchmark(callable_object: Any, source: torch.Tensor) -> list[list[float]]:
    with torch.no_grad():
        for _ in range(20):
            callable_object(source)
    torch.cuda.synchronize()
    repeats: list[list[float]] = []
    with torch.no_grad():
        for _repeat in range(5):
            samples = []
            for _iteration in range(300):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                callable_object(source)
                end.record()
                end.synchronize()
                samples.append(float(start.elapsed_time(end)))
            repeats.append(samples)
    return repeats


def command_measure(args: argparse.Namespace) -> None:
    source_numpy = np.load(args.input)
    source = torch.from_numpy(source_numpy).cuda()
    reference = np.load(args.reference)
    if args.engine:
        build_report = require_diagnostic_authorization(
            args.build_report, args.allow_diagnostic
        )
        runner = TorchTensorRTRunner(args.engine)
        with torch.no_grad():
            candidate = runner(source).float().cpu().numpy().copy()
        repeats = _benchmark(runner, source)
        implementation = "tensorrt"
        implementation_sha = sha256_file(args.engine)
    else:
        build_report = None
        model, _hypes = load_model(args.config, args.checkpoint)
        runner = DenseEncoder(model).eval()
        with torch.no_grad():
            candidate = runner(source).float().cpu().numpy()
        repeats = _benchmark(runner, source)
        implementation = "pytorch_cuda_native"
        implementation_sha = sha256_file(args.checkpoint)
    report = {
        "schema_version": "fcooper_orin_dense_measurement_v1",
        "status": "success",
        "implementation": implementation,
        "implementation_sha256": implementation_sha,
        "build_evidence_status": (
            build_report.get("status") if build_report is not None else "native"
        ),
        "input_sha256": sha256_file(args.input),
        "reference_sha256": sha256_file(args.reference),
        "numerical": numeric_metrics(reference, candidate),
        "latency": summarize_latency(repeats),
    }
    write_json(args.report, report)
    print(json.dumps({key: value for key, value in report.items() if key != "latency"}, indent=2))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    subparsers = root.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--onnx", type=Path, required=True)
    build.add_argument("--engine", type=Path, required=True)
    build.add_argument("--cache", type=Path, required=True)
    build.add_argument("--inspector", type=Path, required=True)
    build.add_argument("--report", type=Path, required=True)
    build.add_argument("--precision", choices=("fp32", "fp16"), required=True)
    build.add_argument("--builder-level", type=int, required=True)
    build.add_argument("--workspace-gib", type=int, default=2)
    build.set_defaults(function=command_build)
    capture = subparsers.add_parser("capture")
    capture.add_argument("--config", type=Path, required=True)
    capture.add_argument("--checkpoint", type=Path, required=True)
    capture.add_argument("--input", type=Path, required=True)
    capture.add_argument("--reference", type=Path, required=True)
    capture.add_argument("--test-manifest", type=Path, required=True)
    capture.add_argument("--expected-manifest-sha", required=True)
    capture.add_argument("--report", type=Path, required=True)
    capture.set_defaults(function=command_capture)
    measure = subparsers.add_parser("measure")
    measure.add_argument("--config", type=Path)
    measure.add_argument("--checkpoint", type=Path)
    measure.add_argument("--engine", type=Path)
    measure.add_argument("--build-report", type=Path)
    measure.add_argument("--allow-diagnostic", action="store_true")
    measure.add_argument("--input", type=Path, required=True)
    measure.add_argument("--reference", type=Path, required=True)
    measure.add_argument("--report", type=Path, required=True)
    measure.set_defaults(function=command_measure)
    return root


def main() -> None:
    args = parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
