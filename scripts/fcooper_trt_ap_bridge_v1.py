#!/usr/bin/env python3
"""Evaluate F-Cooper full-test AP with an exact native or Orin TRT scope."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np


INPUT_NAME = "spatial_features"
OUTPUT_NAME = "encoded_features"
INPUT_SHAPE = (5, 64, 512, 512)
SCOPE = "post_scatter_backbone_shrinker"
SECRET_MARKERS = ("secret", "token", "password", "credential", "api_key")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path | str, value: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def orin_platform_probe() -> tuple[str, str, str]:
    model_path = Path("/proc/device-tree/model")
    try:
        board = (
            model_path.read_text(encoding="utf-8", errors="replace")
            .replace("\x00", " ")
            .strip()
        )
    except OSError:
        board = ""
    return platform.system(), platform.machine(), board


def require_orin(
    probe: Any = orin_platform_probe,
) -> tuple[str, str, str]:
    system, machine, description = probe()
    if (
        system != "Linux"
        or machine.lower() not in {"aarch64", "arm64"}
        or "orin" not in description.lower()
    ):
        raise RuntimeError("TRT AP requires Linux aarch64 NVIDIA Orin")
    return system, machine, description


def _load_json_object(path: Path | str, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"invalid {label}: expected a JSON object")
    return value


def _receipt_platform_is_orin(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) >= 3
        and value[0] == "Linux"
        and str(value[1]).lower() in {"aarch64", "arm64"}
        and "orin" in " ".join(map(str, value[2:])).lower()
    )


def validate_trt_build_identity(
    *,
    engine_path: Path | str,
    checkpoint_path: Path | str,
    config_path: Path | str,
    manifest_path: Path | str,
    receipt_path: Path | str,
    arm_name: str,
) -> dict[str, Any]:
    paths = {
        "engine": Path(engine_path),
        "checkpoint": Path(checkpoint_path),
        "config": Path(config_path),
        "canonical manifest": Path(manifest_path),
        "build receipt": Path(receipt_path),
    }
    for label, path in paths.items():
        if not path.is_file():
            raise ValueError(f"missing {label}: {path}")
    manifest = _load_json_object(paths["canonical manifest"], "canonical manifest")
    receipt = _load_json_object(paths["build receipt"], "build receipt")
    if manifest.get("status") != "ready":
        raise ValueError("canonical manifest status is not ready")
    if manifest.get("scope") != SCOPE:
        raise ValueError("canonical manifest scope mismatch")
    arms = manifest.get("arms")
    if not isinstance(arms, dict) or arm_name not in arms:
        raise ValueError("canonical manifest arm mismatch")
    arm = arms[arm_name]
    if not isinstance(arm, dict) or arm.get("runtime") != "trt":
        raise ValueError("canonical manifest arm is not TRT")
    if (
        arm.get("builder_policy") != "trt85_default"
        or arm.get("builder_optimization_level") is not None
    ):
        raise ValueError("TRT 8.5 default builder contract mismatch")

    onnx_path = Path(str(arm.get("onnx_path", "")))
    if not onnx_path.is_file():
        raise ValueError("canonical ONNX artifact is missing")
    actual = {
        "manifest": sha256_file(paths["canonical manifest"]),
        "engine": sha256_file(paths["engine"]),
        "checkpoint": sha256_file(paths["checkpoint"]),
        "config": sha256_file(paths["config"]),
        "ONNX": sha256_file(onnx_path),
    }
    for label, arm_key in (
        ("checkpoint", "checkpoint_sha256"),
        ("config", "config_sha256"),
        ("ONNX", "onnx_sha256"),
    ):
        if not arm.get(arm_key) or actual[label] != arm[arm_key]:
            raise ValueError(f"{label} identity mismatch")

    required_receipt = {
        "status": "built",
        "scope": SCOPE,
        "arm": arm_name,
        "manifest_sha256": actual["manifest"],
        "engine_sha256": actual["engine"],
        "checkpoint_sha256": actual["checkpoint"],
        "onnx_sha256": actual["ONNX"],
    }
    for key, expected in required_receipt.items():
        if receipt.get(key) != expected:
            label = {
                "manifest_sha256": "manifest",
                "engine_sha256": "engine",
                "checkpoint_sha256": "checkpoint",
                "onnx_sha256": "ONNX",
            }.get(key, key)
            raise ValueError(f"build receipt {label} mismatch")
    if receipt.get("config_sha256") != actual["config"]:
        raise ValueError("build receipt config identity mismatch")
    if (
        not receipt.get("engine")
        or Path(receipt["engine"]).resolve() != paths["engine"].resolve()
    ):
        raise ValueError("build receipt engine path mismatch")
    if not _receipt_platform_is_orin(receipt.get("platform")):
        raise ValueError("build receipt platform is not Linux aarch64 Orin")
    if not str(receipt.get("tensorrt_version", "")).startswith("8.5."):
        raise ValueError("build receipt is not TensorRT 8.5.x")
    if receipt.get("h800_engine_or_cache_read") is not False:
        raise ValueError("build receipt does not exclude H800 engine/cache reuse")
    if receipt.get("calibration_cache_read") is not False:
        raise ValueError("build receipt does not exclude calibration cache reads")
    builder_contract = {
        "builder_policy": arm.get("builder_policy"),
        "builder_optimization_level": arm.get("builder_optimization_level"),
        "q_mode": arm.get("q_mode"),
    }
    build_arguments = receipt.get("build_arguments")
    if not isinstance(build_arguments, dict) or any(
        build_arguments.get(key) != value
        for key, value in builder_contract.items()
    ):
        raise ValueError("build receipt builder contract mismatch")
    return {
        "arm": arm_name,
        "scope": SCOPE,
        "manifest_path": str(paths["canonical manifest"].resolve()),
        "manifest_sha256": actual["manifest"],
        "build_receipt_path": str(paths["build receipt"].resolve()),
        "build_receipt_sha256": sha256_file(paths["build receipt"]),
        "engine_path": str(paths["engine"].resolve()),
        "engine_sha256": actual["engine"],
        "checkpoint_sha256": actual["checkpoint"],
        "config_sha256": actual["config"],
        "onnx_path": str(onnx_path.resolve()),
        "onnx_sha256": actual["ONNX"],
        "build_platform": receipt["platform"],
        "builder_contract": builder_contract,
        "tensorrt_version": receipt.get("tensorrt_version"),
    }


def validate_engine_binding_contract(
    *,
    input_names: Sequence[str],
    output_names: Sequence[str],
    input_shape: Sequence[int],
    output_shape: Sequence[int],
) -> None:
    if (
        list(input_names) != [INPUT_NAME]
        or list(output_names) != [OUTPUT_NAME]
        or tuple(input_shape) != INPUT_SHAPE
        or len(output_shape) != 4
        or int(output_shape[0]) != 5
        or any(int(value) <= 0 for value in output_shape)
    ):
        raise ValueError(
            "TensorRT binding contract must be one spatial_features input "
            "[5,64,512,512] and one static batch-5 encoded_features output"
        )


class TorchTensorRTRunner:
    """Lazy TensorRT adapter retaining CUDA tensors at the exact dense scope."""

    def __init__(self, engine_path: Path):
        try:
            import tensorrt as trt
            import torch
        except ImportError as error:
            raise RuntimeError("TRT mode requires TensorRT and PyTorch") from error
        self._trt = trt
        self._torch = torch
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize engine: {engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("failed to create TensorRT execution context")
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
        self.input_shape = tuple(self.engine.get_tensor_shape(self.inputs[0])) if len(self.inputs) == 1 else ()
        self.output_shape = tuple(self.engine.get_tensor_shape(self.outputs[0])) if len(self.outputs) == 1 else ()
        validate_engine_binding_contract(
            input_names=self.inputs,
            output_names=self.outputs,
            input_shape=self.input_shape,
            output_shape=self.output_shape,
        )
        if any(value <= 0 for value in self.output_shape):
            raise ValueError("F-Cooper AP bridge requires a static output shape")
        input_dtype = trt.nptype(self.engine.get_tensor_dtype(self.inputs[0]))
        output_dtype = trt.nptype(self.engine.get_tensor_dtype(self.outputs[0]))
        if input_dtype != np.float32 or output_dtype != np.float32:
            raise ValueError("TensorRT bindings must both expose float32 tensors")
        self.output_dtype = torch.from_numpy(
            np.empty((), dtype=output_dtype)
        ).dtype
        self.output_buffer = torch.empty(
            self.output_shape, device="cuda", dtype=self.output_dtype
        )
        self.call_count = 0

    @property
    def tensorrt_version(self) -> str:
        return str(getattr(self._trt, "__version__", "unknown"))

    def __call__(self, source: Any) -> Any:
        if tuple(source.shape) != self.input_shape:
            raise ValueError(f"engine input shape drift: {tuple(source.shape)}")
        if source.dtype != self._torch.float32 or not source.is_cuda:
            raise ValueError("engine input must be a CUDA float32 tensor")
        source = source.contiguous()
        self.context.set_tensor_address(self.inputs[0], source.data_ptr())
        self.context.set_tensor_address(
            self.outputs[0], self.output_buffer.data_ptr()
        )
        stream = self._torch.cuda.current_stream()
        if not self.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT execute_async_v3 returned false")
        self.call_count += 1
        return self.output_buffer


class _EngineBackboneCore:
    def __init__(self, runner: Any, tensor_ops: Any):
        self.runner = runner
        self.tensor_ops = tensor_ops

    def _padded(self, source: Any, agents: int) -> Any:
        if self.tensor_ops is np:
            raw = source.value if hasattr(source, "value") else source
            padded_array = np.zeros(self.runner.input_shape, dtype=raw.dtype)
            padded_array[:agents] = raw
            return type(source)(padded_array) if hasattr(source, "value") else padded_array
        padded = self.tensor_ops.zeros(
            self.runner.input_shape, device=source.device, dtype=source.dtype
        )
        padded[:agents] = source
        return padded

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        source = data[INPUT_NAME]
        agents = int(source.shape[0])
        engine_batch = int(self.runner.input_shape[0])
        if agents < 1 or agents > engine_batch or engine_batch != 5:
            raise ValueError(
                f"record has {agents} agents, engine requires 1..5"
            )
        encoded = self.runner(self._padded(source, agents))
        return {"spatial_features_2d": encoded[:agents].float()}

    __call__ = forward


class EngineBackbone:
    """Construct a real ``nn.Module`` at runtime, or a fake-friendly core."""

    def __new__(cls, runner: Any, tensor_ops: Any = None) -> Any:
        if tensor_ops is not None:
            return _EngineBackboneCore(runner, tensor_ops)
        import torch
        from torch import nn

        class _TorchEngineBackbone(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.core = _EngineBackboneCore(runner, torch)

            def forward(self, data: dict[str, Any]) -> dict[str, Any]:
                return self.core.forward(data)

        return _TorchEngineBackbone()


class IdentityShrinker:
    def __new__(cls) -> Any:
        from torch import nn

        class _IdentityShrinker(nn.Module):
            def forward(self, value: Any) -> Any:
                return value

        return _IdentityShrinker()


def _canonical_id_bytes(sample_ids: Sequence[str]) -> bytes:
    return json.dumps(
        list(sample_ids), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")


def build_prediction_manifest(
    sample_ids: Sequence[str], *, dataset_samples: int
) -> dict[str, Any]:
    ids = list(sample_ids)
    if any(not isinstance(value, str) or not value for value in ids):
        raise ValueError("prediction sample IDs must be non-empty strings")
    if len(set(ids)) != len(ids):
        raise ValueError("prediction sample IDs must be unique")
    return {
        "schema_version": "fcooper_prediction_manifest_v1",
        "dataset": "OPV2V",
        "split": "test",
        "dataset_samples": int(dataset_samples),
        "processed_samples": len(ids),
        "evaluation_order": "dataset_order_no_shuffle",
        "sample_ids": ids,
        "sample_ids_sha256": hashlib.sha256(
            _canonical_id_bytes(ids)
        ).hexdigest(),
    }


def validate_formal_completion(
    *,
    runtime: str,
    dataset_samples: int,
    processed_samples: int,
    failed_samples: int,
    fallback_samples: int,
    engine_calls: int,
    expected_dataset_samples: int,
) -> dict[str, Any]:
    expected_calls = 2170 if runtime == "trt" else 0
    valid_runtime = runtime in {"trt", "native_fp32"}
    complete = (
        valid_runtime
        and expected_dataset_samples == 2170
        and dataset_samples == 2170
        and processed_samples == 2170
        and failed_samples == 0
        and fallback_samples == 0
        and engine_calls == expected_calls
    )
    if not complete:
        raise ValueError(
            "formal full2170 contract failed: exact dataset/processed/failure/"
            "fallback/engine-call counts are required"
        )
    return {
        "status": "success_full",
        "runtime": runtime,
        "dataset_samples": dataset_samples,
        "processed_samples": processed_samples,
        "failed_samples": failed_samples,
        "fallback_samples": fallback_samples,
        "engine_calls": engine_calls,
    }


def sanitize_arguments(value: Any, key: str = "") -> Any:
    if any(marker in key.lower() for marker in SECRET_MARKERS):
        return None
    if isinstance(value, argparse.Namespace):
        return {
            name: sanitize_arguments(child, name)
            for name, child in vars(value).items()
            if not any(marker in name.lower() for marker in SECRET_MARKERS)
        }
    if isinstance(value, Mapping):
        return {
            str(name): sanitize_arguments(child, str(name))
            for name, child in value.items()
            if not any(
                marker in str(name).lower() for marker in SECRET_MARKERS
            )
        }
    if isinstance(value, (list, tuple)):
        return [sanitize_arguments(child) for child in value]
    if isinstance(value, Path):
        return str(value)
    return value


def build_failure_receipt(
    *,
    args: argparse.Namespace,
    error: BaseException,
    started_at: str,
    ended_at: str,
) -> dict[str, Any]:
    secret_values = [
        str(value)
        for name, value in vars(args).items()
        if value is not None
        and any(marker in name.lower() for marker in SECRET_MARKERS)
    ]
    message = str(error)
    for secret_value in secret_values:
        message = message.replace(secret_value, "[REDACTED]")
    identities: dict[str, Any] = {
        "dataset": "OPV2V-test",
        "expected_dataset_samples": getattr(
            args, "expected_dataset_samples", None
        ),
        "config_sha256": None,
        "checkpoint_sha256": None,
        "engine_sha256": None,
        "native_identity": (
            "exact_checkpoint_pytorch_cuda_fp32_unchanged"
            if getattr(args, "native", False)
            else None
        ),
        "arm": getattr(args, "arm", None),
    }
    for name in ("config", "checkpoint", "engine", "manifest", "build_receipt"):
        candidate = getattr(args, name, None)
        if candidate is not None and Path(candidate).is_file():
            identities[f"{name}_sha256"] = sha256_file(candidate)
    return {
        "schema_version": "fcooper_ap_run_log_v1",
        "status": "failure",
        "exit_state": "exception",
        "started_at": started_at,
        "ended_at": ended_at,
        "scope": SCOPE,
        "arguments": sanitize_arguments(args),
        "identities": identities,
        "failure": {
            "type": type(error).__name__,
            "message": message,
        },
    }


def _unwrap_singleton(value: Any) -> Any:
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    return value


def sample_identity_from_batch(batch: dict[str, Any]) -> tuple[int, list[str]]:
    record = batch.get("ego", batch)
    if "sample_idx" not in record or "cav_id_list" not in record:
        raise ValueError("batch lacks sample_idx or cav_id_list provenance")
    sample_idx = int(_unwrap_singleton(record["sample_idx"]))
    cav_ids = record["cav_id_list"]
    if (
        isinstance(cav_ids, (list, tuple))
        and len(cav_ids) == 1
        and isinstance(cav_ids[0], (list, tuple))
    ):
        cav_ids = cav_ids[0]
    ids = [str(value) for value in cav_ids]
    if not ids:
        raise ValueError("batch cav_id_list is empty")
    return sample_idx, ids


def stable_sample_id(sample_idx: int, cav_ids: Sequence[str]) -> str:
    return f"opv2v-test:{sample_idx}:{','.join(map(str, cav_ids))}"


def _agent_count(batch: dict[str, Any], cav_ids: Sequence[str]) -> int:
    record = batch.get("ego", batch)
    record_len = record.get("record_len")
    if record_len is None:
        return len(cav_ids)
    value = _unwrap_singleton(record_len)
    if hasattr(value, "item"):
        value = value.item()
    agents = int(value)
    if agents != len(cav_ids):
        raise ValueError("record_len and cav_id_list disagree")
    return agents


def _checkpoint_state(torch: Any, checkpoint: Path) -> dict[str, Any]:
    loaded = torch.load(checkpoint, map_location="cpu")
    if not isinstance(loaded, dict):
        raise ValueError("checkpoint must contain a state dictionary")
    for key in ("state_dict", "model_state_dict", "model"):
        state = loaded.get(key)
        if isinstance(state, dict):
            return state
    return loaded


def pre_scan_dataset_contract(
    dataset: Any, *, expected_dataset_samples: int
) -> list[str]:
    dataset_samples = len(dataset)
    if dataset_samples != expected_dataset_samples:
        raise ValueError(
            f"dataset length mismatch: expected {expected_dataset_samples}, "
            f"got {dataset_samples}"
        )
    sample_ids: list[str] = []
    for index in range(dataset_samples):
        batch = dataset.collate_batch_test([dataset[index]])
        if batch is None:
            raise ValueError(f"dataset produced null batch at index {index}")
        sample_idx, cav_ids = sample_identity_from_batch(batch)
        agents = _agent_count(batch, cav_ids)
        if agents < 1 or agents > 5:
            raise ValueError(
                f"record {sample_idx} has {agents} agents; contract requires 1..5"
            )
        sample_ids.append(stable_sample_id(sample_idx, cav_ids))
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("dataset pre-scan found duplicate stable sample IDs")
    return sample_ids


def validate_evaluated_sequence(
    sample_ids: Sequence[str], *, expected_dataset_samples: int
) -> list[str]:
    ids = list(sample_ids)
    if len(ids) != expected_dataset_samples:
        raise ValueError(
            "evaluated sample ID count mismatch: expected "
            f"{expected_dataset_samples}, got {len(ids)}"
        )
    if len(set(ids)) != len(ids):
        raise ValueError("evaluated sample sequence contains duplicate IDs")
    return ids


def validate_checkpoint_location(
    checkpoint: Path | str, checkpoint_dir: Path | str
) -> Path:
    resolved = Path(checkpoint).resolve()
    try:
        resolved.relative_to(Path(checkpoint_dir).resolve())
    except ValueError as error:
        raise ValueError(
            "--checkpoint must resolve under --checkpoint-dir"
        ) from error
    return resolved


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    runtime = parser.add_mutually_exclusive_group(required=True)
    runtime.add_argument("--engine", type=Path)
    runtime.add_argument("--native", action="store_true")
    parser.add_argument("--arm")
    parser.add_argument("--build-receipt", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--prediction-manifest", type=Path, required=True)
    parser.add_argument("--run-log-manifest", type=Path, required=True)
    parser.add_argument("--require-full2170", action="store_true")
    parser.add_argument("--expected-dataset-samples", type=int, default=2170)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args(argv)
    if args.require_full2170 and args.max_samples is not None:
        parser.error("--max-samples is incompatible with --require-full2170")
    if args.require_full2170 and args.expected_dataset_samples != 2170:
        parser.error("--require-full2170 requires --expected-dataset-samples 2170")
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be positive")
    trt_identity_args = (args.arm, args.build_receipt, args.manifest)
    if args.engine is not None and any(value is None for value in trt_identity_args):
        parser.error("TRT mode requires --arm, --build-receipt, and --manifest")
    if args.native and any(value is not None for value in trt_identity_args):
        parser.error("native mode rejects --arm, --build-receipt, and --manifest")
    return args


def _runtime_versions(torch: Any, trt_version: str | None) -> dict[str, str]:
    return {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "pytorch": str(getattr(torch, "__version__", "unknown")),
        "tensorrt": trt_version or "not_loaded_native",
    }


def _write_failure_outputs(
    args: argparse.Namespace, receipt: dict[str, Any]
) -> None:
    for path in (args.output_json, args.run_log_manifest):
        try:
            write_json(path, receipt)
        except OSError:
            pass
    try:
        prediction = {
            "schema_version": "fcooper_prediction_manifest_v1",
            "status": "failure",
            "sample_ids": [],
            "sample_ids_sha256": hashlib.sha256(b"[]").hexdigest(),
        }
        write_json(args.prediction_manifest, prediction)
    except OSError:
        pass


def _failure_args_from_tokens(
    argv: Sequence[str],
) -> argparse.Namespace | None:
    tokens = list(argv)

    def value(flag: str) -> str | None:
        for index in range(len(tokens) - 1, -1, -1):
            token = tokens[index]
            if token.startswith(flag + "="):
                return token.split("=", 1)[1]
            if token == flag:
                return tokens[index + 1] if index + 1 < len(tokens) else None
        return None

    output = value("--output-json")
    prediction = value("--prediction-manifest")
    run_log = value("--run-log-manifest")
    if not output or not prediction or not run_log:
        return None
    expected = value("--expected-dataset-samples")
    try:
        expected_samples = int(expected) if expected else 2170
    except ValueError:
        expected_samples = None
    return argparse.Namespace(
        config=Path(value("--config")) if value("--config") else None,
        checkpoint=(
            Path(value("--checkpoint")) if value("--checkpoint") else None
        ),
        checkpoint_dir=(
            Path(value("--checkpoint-dir"))
            if value("--checkpoint-dir")
            else None
        ),
        engine=Path(value("--engine")) if value("--engine") else None,
        native="--native" in tokens,
        arm=value("--arm"),
        build_receipt=(
            Path(value("--build-receipt"))
            if value("--build-receipt")
            else None
        ),
        manifest=Path(value("--manifest")) if value("--manifest") else None,
        output_json=Path(output),
        prediction_manifest=Path(prediction),
        run_log_manifest=Path(run_log),
        require_full2170="--require-full2170" in tokens,
        expected_dataset_samples=expected_samples,
        max_samples=value("--max-samples"),
    )


def run(
    args: argparse.Namespace,
    *,
    started_at: str,
    orin_probe: Any = orin_platform_probe,
) -> dict[str, Any]:
    validate_checkpoint_location(args.checkpoint, args.checkpoint_dir)
    if args.engine is not None:
        require_orin(orin_probe)
    build_identity = (
        validate_trt_build_identity(
            engine_path=args.engine,
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            manifest_path=args.manifest,
            receipt_path=args.build_receipt,
            arm_name=args.arm,
        )
        if args.engine is not None
        else None
    )
    import torch
    from torch.utils.data import DataLoader

    from opencood.data_utils.datasets import build_dataset
    from opencood.hypes_yaml import yaml_utils
    from opencood.tools import inference_utils, train_utils
    from opencood.utils import eval_utils

    hypes = yaml_utils.load_yaml(
        str(args.config), SimpleNamespace(model_dir=None)
    )
    if "test_dir" not in hypes:
        raise ValueError("configuration lacks test_dir")
    hypes["validate_dir"] = hypes["test_dir"]
    dataset = build_dataset(hypes, visualize=False, train=False)
    dataset_samples = len(dataset)
    if dataset_samples != args.expected_dataset_samples:
        raise ValueError(
            f"dataset length mismatch: expected {args.expected_dataset_samples}, "
            f"got {dataset_samples}"
        )
    if args.require_full2170 and dataset_samples != 2170:
        raise ValueError("formal AP requires the full OPV2V test set of 2170")
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
    model.load_state_dict(
        _checkpoint_state(torch, args.checkpoint), strict=True
    )
    model.cuda().eval()
    runner = None
    runtime = "native_fp32"
    if args.engine is not None:
        runner = TorchTensorRTRunner(args.engine)
        model.backbone_m1 = EngineBackbone(runner)
        model.shrinker_m1 = IdentityShrinker()
        runtime = "trt"

    stats = {
        threshold: {"tp": [], "fp": [], "gt": 0, "score": []}
        for threshold in (0.3, 0.5, 0.7)
    }
    sample_ids: list[str] = []
    failed_samples = 0
    fallback_samples = 0
    started_seconds = time.time()
    for batch in loader:
        if batch is None:
            failed_samples += 1
            raise ValueError("dataset produced a null batch")
        try:
            sample_idx, cav_ids = sample_identity_from_batch(batch)
            agents = _agent_count(batch, cav_ids)
            if agents < 1 or agents > 5:
                raise ValueError(
                    f"record has {agents} agents; dense contract requires 1..5"
                )
            stable_id = stable_sample_id(sample_idx, cav_ids)
            sample_ids.append(stable_id)
            batch = train_utils.to_device(batch, torch.device("cuda"))
            with torch.no_grad():
                result = inference_utils.inference_intermediate_fusion(
                    batch, model, dataset
                )
            for threshold in stats:
                eval_utils.caluclate_tp_fp(
                    result["pred_box_tensor"],
                    result["pred_score"],
                    result["gt_box_tensor"],
                    stats,
                    threshold,
                )
            if args.max_samples is not None and len(sample_ids) >= args.max_samples:
                break
            if args.require_full2170 and len(sample_ids) % 100 == 0:
                print(
                    json.dumps(
                        {
                            "progress_samples": len(sample_ids),
                            "dataset_samples": dataset_samples,
                            "runtime": runtime,
                            "engine_calls": runner.call_count if runner else 0,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        except Exception:
            failed_samples += 1
            raise

    ap30, ap50, ap70 = (
        eval_utils.calculate_ap(stats, threshold)[0]
        for threshold in (0.3, 0.5, 0.7)
    )
    engine_calls = runner.call_count if runner is not None else 0
    expected_processed = (
        min(dataset_samples, args.max_samples)
        if args.max_samples is not None
        else dataset_samples
    )
    if args.require_full2170:
        sample_ids = validate_evaluated_sequence(
            sample_ids,
            expected_dataset_samples=args.expected_dataset_samples,
        )
        completion = validate_formal_completion(
            runtime=runtime,
            dataset_samples=dataset_samples,
            processed_samples=len(sample_ids),
            failed_samples=failed_samples,
            fallback_samples=fallback_samples,
            engine_calls=engine_calls,
            expected_dataset_samples=args.expected_dataset_samples,
        )
    elif failed_samples or len(sample_ids) != expected_processed:
        raise ValueError("smoke AP did not process the requested sample count")
    else:
        completion = {"status": "success_sanity"}

    prediction_manifest = build_prediction_manifest(
        sample_ids, dataset_samples=dataset_samples
    )
    if args.require_full2170 and len(prediction_manifest["sample_ids"]) != 2170:
        raise ValueError("formal prediction manifest must contain 2170 IDs")
    write_json(args.prediction_manifest, prediction_manifest)
    report = {
        "schema_version": "fcooper_ap_report_v2",
        "status": completion["status"],
        "dataset": "OPV2V",
        "split": "test",
        "dataset_samples": dataset_samples,
        "processed_samples": len(sample_ids),
        "failed_samples": failed_samples,
        "fallback_samples": fallback_samples,
        "engine_calls": engine_calls,
        "runtime": runtime,
        "requested_runtime": "native" if args.native else "trt",
        "scope": SCOPE,
        "ap30": float(ap30),
        "ap50": float(ap50),
        "ap70": float(ap70),
        "elapsed_seconds": time.time() - started_seconds,
        "checkpoint_path": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "config_path": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config),
        "engine_path": str(args.engine.resolve()) if args.engine else None,
        "engine_sha256": sha256_file(args.engine) if args.engine else None,
        "arm": args.arm,
        "build_identity": build_identity,
        "native_identity": (
            "exact_checkpoint_pytorch_cuda_fp32_unchanged"
            if args.native
            else None
        ),
        "versions": _runtime_versions(
            torch, runner.tensorrt_version if runner else None
        ),
        "prediction_manifest_path": str(args.prediction_manifest.resolve()),
        "prediction_manifest_sha256": sha256_file(args.prediction_manifest),
        "numerical_contract": {
            "dense_scope_engine_execution": (
                engine_calls == len(sample_ids) if runner else True
            ),
            "silent_fallback_forbidden": True,
            "fallback_samples": fallback_samples,
        },
    }
    write_json(args.output_json, report)
    run_log = {
        "schema_version": "fcooper_ap_run_log_v1",
        "status": "success",
        "started_at": started_at,
        "ended_at": utc_now(),
        "arguments": sanitize_arguments(args),
        "scope": SCOPE,
        "runtime": runtime,
        "dataset_samples": dataset_samples,
        "checkpoint_sha256": report["checkpoint_sha256"],
        "config_sha256": report["config_sha256"],
        "engine_sha256": report["engine_sha256"],
        "arm": args.arm,
        "build_identity": build_identity,
        "prediction_manifest_sha256": report["prediction_manifest_sha256"],
        "output_json_sha256": sha256_file(args.output_json),
    }
    write_json(args.run_log_manifest, run_log)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    started_at = utc_now()
    tokens = list(argv) if argv is not None else sys.argv[1:]
    try:
        args = parse_args(tokens)
    except SystemExit as error:
        if error.code:
            failure_args = _failure_args_from_tokens(tokens)
            if failure_args is not None:
                receipt = build_failure_receipt(
                    args=failure_args,
                    error=error,
                    started_at=started_at,
                    ended_at=utc_now(),
                )
                _write_failure_outputs(failure_args, receipt)
        raise
    try:
        report = run(args, started_at=started_at)
    except Exception as error:
        receipt = build_failure_receipt(
            args=args,
            error=error,
            started_at=started_at,
            ended_at=utc_now(),
        )
        _write_failure_outputs(args, receipt)
        raise
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
