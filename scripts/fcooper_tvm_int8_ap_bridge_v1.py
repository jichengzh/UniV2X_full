#!/usr/bin/env python3
"""Evaluate F-Cooper OPV2V AP through a Route-B TVM Relax VM INT8 artifact."""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import math
import os
import re
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from scripts.fcooper_prediction_artifact_v1 import PredictionArtifactWriter

from scripts.fcooper_tvm_subprocess_runner_v1 import (
    SharedMemoryTvmClient,
    build_worker_env,
)


SCHEMA_VERSION = "fcooper_tvm_int8_ap_report_v1"
PIPELINE_SCOPE = "fcooper_post_scatter_backbone_m1_plus_shrinker_m1"
SANITY_SAMPLES = 16
FULL_AP_SAMPLES = 2170
ARTIFACT_BATCH = 5
GRAPH_INPUT_NAME = "spatial_features"
ALLOWED_RUNTIME_ROLES = {"graph_input", "weight_input", "bias_input", "graph_output"}
DYNAMIC_QUANT_KEYS = ("dynamic", "dynamic_per_chunk", "per_chunk", "allow_fallback")


def sha256_path(path: Path) -> str:
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
    root = path.parent if path.is_file() else path
    for item in files:
        digest.update(item.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def quantize_uint8(value: np.ndarray, *, scale: float, zero_point: int) -> np.ndarray:
    scale, zero_point = _validate_quant_values(scale, zero_point)
    source = np.asarray(value, dtype=np.float32)
    return np.clip(np.rint(source / scale) + zero_point, 0, 255).astype(np.uint8)


def dequantize_uint8(value: np.ndarray, *, scale: float, zero_point: int) -> np.ndarray:
    scale, zero_point = _validate_quant_values(scale, zero_point)
    source = np.asarray(value)
    if source.dtype != np.uint8:
        raise ValueError(f"dequantization requires uint8 input, got {source.dtype}")
    return ((source.astype(np.float32) - zero_point) * scale).astype(np.float32, copy=False)


def _validate_quant_values(scale: Any, zero_point: Any) -> tuple[float, int]:
    try:
        resolved_scale = float(scale)
        resolved_zero_point = int(zero_point)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("static quant params require numeric scale and zero_point") from exc
    if not math.isfinite(resolved_scale) or resolved_scale <= 0.0:
        raise ValueError(f"static quant scale must be finite and positive, got {resolved_scale}")
    if not 0 <= resolved_zero_point <= 255:
        raise ValueError(f"static quant zero_point must be in [0,255], got {resolved_zero_point}")
    return resolved_scale, resolved_zero_point


@dataclass(frozen=True)
class QuantParams:
    scale: float
    zero_point: int

    @classmethod
    def from_payload(cls, name: str, payload: Any) -> "QuantParams":
        if not isinstance(payload, Mapping):
            raise ValueError(f"missing static quant params for {name}")
        source = str(payload.get("source") or "").lower().replace("-", "_")
        if any(token in source for token in ("dynamic", "per_chunk", "fallback")) or any(
            bool(payload.get(key)) for key in DYNAMIC_QUANT_KEYS
        ):
            raise ValueError(f"dynamic quantization is forbidden for {name}")
        try:
            scale, zero_point = _validate_quant_values(payload["scale"], payload["zero_point"])
        except KeyError as exc:
            raise ValueError(f"missing static quant params for {name}") from exc
        return cls(scale=scale, zero_point=zero_point)


@dataclass(frozen=True)
class RuntimeTensor:
    name: str
    role: str
    shape: tuple[int, ...]
    dtype: str
    initializer_name: str | None = None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "RuntimeTensor":
        name = str(payload.get("arg_name") or "")
        role = str(payload.get("role") or "")
        if not name:
            raise ValueError("runtime_arg_plan item is missing arg_name")
        if role not in ALLOWED_RUNTIME_ROLES:
            raise ValueError(f"unsupported runtime role for {name}: {role}")
        try:
            shape = tuple(int(value) for value in payload["shape"])
            dtype = np.dtype(str(payload["dtype"])).name
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid runtime tensor contract for {name}") from exc
        if not shape or any(value <= 0 for value in shape):
            raise ValueError(f"runtime tensor {name} requires positive static dimensions")
        initializer = payload.get("initializer_name")
        return cls(name, role, shape, dtype, str(initializer) if initializer is not None else None)


@dataclass(frozen=True)
class RuntimeContract:
    tensors: tuple[RuntimeTensor, ...]
    graph_input: RuntimeTensor
    graph_output: RuntimeTensor
    input_quant: QuantParams
    output_quant: QuantParams

    @property
    def input_shape(self) -> tuple[int, ...]:
        return self.graph_input.shape

    @property
    def output_shape(self) -> tuple[int, ...]:
        return self.graph_output.shape

    @property
    def runtime_inputs(self) -> tuple[RuntimeTensor, ...]:
        return tuple(item for item in self.tensors if item.role != "graph_output")

    @property
    def weight_inputs(self) -> tuple[RuntimeTensor, ...]:
        return tuple(item for item in self.tensors if item.role in {"weight_input", "bias_input"})

    @classmethod
    def from_payloads(
        cls,
        runtime_arg_plan: Iterable[Mapping[str, Any]],
        quant_payload: Mapping[str, Any],
    ) -> "RuntimeContract":
        tensors = tuple(RuntimeTensor.from_payload(item) for item in runtime_arg_plan)
        inputs = [item for item in tensors if item.role == "graph_input"]
        outputs = [item for item in tensors if item.role == "graph_output"]
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("Route-B artifact must expose exactly one graph input and one graph output")
        graph_input, graph_output = inputs[0], outputs[0]
        if graph_input.name != GRAPH_INPUT_NAME:
            raise ValueError(f"graph input must be {GRAPH_INPUT_NAME}, got {graph_input.name}")
        if graph_input.dtype != "uint8" or graph_output.dtype != "uint8":
            raise ValueError("Route-B graph input and output must both be uint8")
        if len(graph_input.shape) != 4 or len(graph_output.shape) != 4:
            raise ValueError("F-Cooper graph input and output must be rank-4 NCHW tensors")
        if graph_input.shape[0] != ARTIFACT_BATCH or graph_output.shape[0] != ARTIFACT_BATCH:
            raise ValueError(f"F-Cooper Route-B artifact batch must be {ARTIFACT_BATCH}")
        params = quant_payload.get("params", quant_payload)
        if not isinstance(params, Mapping):
            raise ValueError("static tensor quant contract must contain a params mapping")
        return cls(
            tensors=tensors,
            graph_input=graph_input,
            graph_output=graph_output,
            input_quant=QuantParams.from_payload(graph_input.name, params.get(graph_input.name)),
            output_quant=QuantParams.from_payload(graph_output.name, params.get(graph_output.name)),
        )


def _declared_sha256(payload: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, Mapping):
            value = value.get("sha256")
        if value:
            return str(value).lower()
    return None


def _require_declared_sha(
    payload: Mapping[str, Any],
    path: Path,
    *,
    label: str,
    keys: Sequence[str],
) -> str:
    actual = sha256_path(path)
    expected = _declared_sha256(payload, *keys)
    if expected is not None and expected != actual.lower():
        raise ValueError(f"{label} SHA256 mismatch")
    return actual


def _archive_key_for_tensor(
    tensor: RuntimeTensor,
    declared_bindings: Mapping[str, Any],
    archive_keys: set[str],
) -> str:
    declared = declared_bindings.get(tensor.name)
    if declared is not None:
        key = str(declared)
        if key not in archive_keys:
            raise ValueError(f"runtime weight archive missing declared key for {tensor.name}: {key}")
        return key
    for candidate in (tensor.name, tensor.initializer_name):
        if candidate is not None and candidate in archive_keys:
            return candidate
    raise ValueError(f"runtime weight archive missing archive binding for {tensor.name}")


def load_runtime_weights(
    route_result: Mapping[str, Any],
    archive_path: Path,
) -> tuple[tuple[np.ndarray, ...], dict[str, str]]:
    archive_path = Path(archive_path)
    _require_declared_sha(
        route_result,
        archive_path,
        label="runtime weight",
        keys=("runtime_weight_archive_sha256", "runtime_weights_sha256", "runtime_weights"),
    )
    plan = route_result.get("runtime_arg_plan")
    if not isinstance(plan, list) or not plan:
        raise ValueError("Route-B result runtime_arg_plan must be a non-empty list")
    tensors = tuple(RuntimeTensor.from_payload(item) for item in plan)
    weight_tensors = tuple(
        item for item in tensors if item.role in {"weight_input", "bias_input"}
    )
    declared = route_result.get("runtime_weight_archive_keys") or {}
    if not isinstance(declared, Mapping):
        raise ValueError("runtime_weight_archive_keys must be a mapping")
    arrays: list[np.ndarray] = []
    bindings: dict[str, str] = {}
    with np.load(archive_path, allow_pickle=False) as archive:
        keys = set(archive.files)
        for tensor in weight_tensors:
            key = _archive_key_for_tensor(tensor, declared, keys)
            value = np.asarray(archive[key])
            if tuple(value.shape) != tensor.shape:
                raise ValueError(
                    f"runtime weight shape mismatch for {tensor.name}: "
                    f"{tuple(value.shape)} != {tensor.shape}"
                )
            if value.dtype != np.dtype(tensor.dtype):
                raise ValueError(
                    f"runtime weight dtype mismatch for {tensor.name}: "
                    f"{value.dtype} != {tensor.dtype}"
                )
            arrays.append(np.ascontiguousarray(value))
            bindings[tensor.name] = key
    return tuple(arrays), bindings


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    if hasattr(value, "asnumpy"):
        return np.asarray(value.asnumpy())
    return np.asarray(value)


def _single_vm_output(value: Any) -> np.ndarray:
    if hasattr(value, "numpy") or hasattr(value, "asnumpy") or isinstance(value, np.ndarray):
        return _as_numpy(value)
    if isinstance(value, (list, tuple)):
        outputs = list(value)
    else:
        try:
            outputs = [value[index] for index in range(len(value))]
        except TypeError as exc:
            raise ValueError("TVM Relax VM result is not a tensor or output tuple") from exc
    if len(outputs) != 1:
        raise ValueError(f"Route-B Relax VM returned {len(outputs)} outputs; expected exactly one")
    return _as_numpy(outputs[0])


class TvmRelaxVmInt8Runner:
    """Strict one-input/one-output Relax VM runner with static uint8 boundaries."""

    def __init__(
        self,
        artifact_path: Path,
        contract: RuntimeContract,
        runtime_weights: Sequence[np.ndarray],
        *,
        gpu_id: int = 0,
    ) -> None:
        artifact_path = Path(artifact_path)
        if not artifact_path.is_file():
            raise FileNotFoundError(artifact_path)
        if len(runtime_weights) != len(contract.weight_inputs):
            raise ValueError("runtime weight count does not match runtime_arg_plan bindings")
        self.artifact_path = artifact_path
        self.contract = contract
        self.input_shape = contract.input_shape
        self.output_shape = contract.output_shape
        self.gpu_id = int(gpu_id)
        self.call_count = 0
        self._runtime_weights = tuple(np.ascontiguousarray(value) for value in runtime_weights)
        self._tvm, self._device, self._main = self._load_backend()

    def _load_backend(self) -> tuple[Any, Any, Any]:
        try:
            import tvm
            from tvm import relax
        except ImportError as exc:
            raise RuntimeError("TVM is required to execute the F-Cooper Route-B artifact") from exc
        device = tvm.cuda(self.gpu_id)
        if hasattr(device, "exist") and not device.exist:
            raise RuntimeError(f"TVM CUDA device {self.gpu_id} is unavailable")
        if self.artifact_path.suffix == ".vmexec":
            library = tvm.get_global_func("ffi.Module.load_from_file.so")(
                str(self.artifact_path),
                "so",
            )
        else:
            library = tvm.runtime.load_module(str(self.artifact_path))
        vm = relax.VirtualMachine(library, device)
        return tvm, device, vm["main"]

    def _make_tvm_array(self, value: np.ndarray) -> Any:
        tensor_constructor = getattr(self._tvm.runtime, "tensor", None)
        if tensor_constructor is not None:
            return tensor_constructor(value, device=self._device)
        return self._tvm.nd.array(value, self._device)

    def _execute_host_input(self, host_input: np.ndarray) -> np.ndarray:
        if tuple(host_input.shape) != self.input_shape:
            raise ValueError(f"VM input shape drift: {tuple(host_input.shape)} != {self.input_shape}")
        if host_input.dtype != np.float32:
            raise ValueError(f"VM model-boundary input must be float32, got {host_input.dtype}")
        quant = self.contract.input_quant
        graph_input = quantize_uint8(host_input, scale=quant.scale, zero_point=quant.zero_point)
        weights = iter(self._runtime_weights)
        call_args = tuple(
            graph_input if tensor.role == "graph_input" else next(weights)
            for tensor in self.contract.runtime_inputs
        )
        self.call_count += 1
        returned = self._main(*(self._make_tvm_array(value) for value in call_args))
        self._device.sync()
        graph_output = np.ascontiguousarray(_single_vm_output(returned))
        if tuple(graph_output.shape) != self.output_shape:
            raise ValueError(
                f"VM output shape drift: {tuple(graph_output.shape)} != {self.output_shape}"
            )
        if graph_output.dtype != np.uint8:
            raise ValueError(f"VM graph output dtype drift: {graph_output.dtype} != uint8")
        quant = self.contract.output_quant
        return dequantize_uint8(graph_output, scale=quant.scale, zero_point=quant.zero_point)

    def __call__(self, source: torch.Tensor) -> torch.Tensor:
        if tuple(source.shape) != self.input_shape:
            raise ValueError(f"VM input shape drift: {tuple(source.shape)} != {self.input_shape}")
        if source.dtype != torch.float32:
            raise ValueError(f"F-Cooper model-boundary input must be float32, got {source.dtype}")
        if not source.is_cuda:
            raise ValueError("F-Cooper TVM INT8 AP bridge requires a CUDA input tensor")
        host_input = np.ascontiguousarray(source.detach().cpu().numpy(), dtype=np.float32)
        host_output = self._execute_host_input(host_input)
        return torch.from_numpy(host_output).to(device=source.device, dtype=torch.float32)


class SubprocessTvmRelaxVmInt8Runner:
    """Execute the INT8 Relax VM under its Python 3.10 ABI."""

    def __init__(
        self,
        artifact_path: Path,
        contract: RuntimeContract,
        *,
        route_result: Path,
        runtime_weights: Path,
        quant_contract: Path,
        gpu_id: int,
        worker_python: Path,
        tvm_site: Path,
        tvm_lib_dirs: Sequence[Path],
        workspace: Path,
    ) -> None:
        self.input_shape = contract.input_shape
        self.output_shape = contract.output_shape
        worker = Path(__file__).with_name("fcooper_tvm_relax_worker_v1.py")
        command = [
            str(worker_python),
            str(worker),
            "--mode",
            "int8",
            "--artifact",
            str(artifact_path),
            "--input-shape",
            ",".join(map(str, self.input_shape)),
            "--output-shape",
            ",".join(map(str, self.output_shape)),
            "--input-dtype",
            "float32",
            "--output-dtype",
            "float32",
            "--route-result",
            str(route_result),
            "--runtime-weights",
            str(runtime_weights),
            "--quant-contract",
            str(quant_contract),
            "--gpu-id",
            "0",
        ]
        self._client = SharedMemoryTvmClient(
            worker_command=command,
            input_shape=self.input_shape,
            output_shape=self.output_shape,
            input_dtype="float32",
            output_dtype="float32",
            workspace=workspace,
            env=build_worker_env(
                tvm_site=tvm_site,
                tvm_lib_dirs=tvm_lib_dirs,
                gpu_id=gpu_id,
            ),
        )

    @property
    def call_count(self) -> int:
        return self._client.call_count

    def __call__(self, source: torch.Tensor) -> torch.Tensor:
        if tuple(source.shape) != self.input_shape:
            raise ValueError(f"VM input shape drift: {tuple(source.shape)} != {self.input_shape}")
        if source.dtype != torch.float32:
            raise ValueError(f"F-Cooper model-boundary input must be float32, got {source.dtype}")
        if not source.is_cuda:
            raise ValueError("F-Cooper TVM INT8 AP bridge requires a CUDA input tensor")
        host_input = np.ascontiguousarray(source.detach().cpu().numpy(), dtype=np.float32)
        host_output = self._client.execute(host_input)
        return torch.from_numpy(host_output).to(device=source.device, dtype=torch.float32)

    def close(self) -> None:
        self._client.close()


def _numeric_record(actual: torch.Tensor, reference: torch.Tensor, sample_index: int) -> dict[str, Any]:
    actual_host = actual.detach().float().cpu().numpy().reshape(-1)
    reference_host = reference.detach().float().cpu().numpy().reshape(-1)
    if actual_host.size != reference_host.size:
        raise ValueError("numerical oracle shape mismatch")
    finite = bool(np.isfinite(actual_host).all() and np.isfinite(reference_host).all())
    if actual_host.size > 1 and np.std(actual_host) > 0.0 and np.std(reference_host) > 0.0:
        corrcoef = float(np.corrcoef(actual_host, reference_host)[0, 1])
    else:
        corrcoef = 1.0 if np.array_equal(actual_host, reference_host) else 0.0
    return {
        "sample_index": int(sample_index),
        "finite": finite,
        "corrcoef": corrcoef,
        "mae": float(np.mean(np.abs(actual_host - reference_host))),
        "max_abs_error": float(np.max(np.abs(actual_host - reference_host))),
    }


class _ReferenceDenseBody(nn.Module):
    def __init__(self, backbone: nn.Module, shrinker: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.shrinker = shrinker

    def forward(self, data: dict[str, torch.Tensor]) -> torch.Tensor:
        output = self.backbone(dict(data))
        return self.shrinker(output["spatial_features_2d"]).float()


class TvmInt8DenseBody(nn.Module):
    """Zero-pad agents to five, execute once, dequantize, and unpad."""

    def __init__(self, runner: Any, reference: nn.Module | None = None) -> None:
        super().__init__()
        self.runner = runner
        self.reference = reference
        self.numerical_records: list[dict[str, Any]] = []

    def forward(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        source = data["spatial_features"]
        agents = int(source.shape[0])
        expected_tail = tuple(self.runner.input_shape[1:])
        if tuple(source.shape[1:]) != expected_tail:
            raise ValueError(
                f"dense-body input shape drift: {tuple(source.shape[1:])} != {expected_tail}"
            )
        artifact_batch = int(self.runner.input_shape[0])
        if artifact_batch != ARTIFACT_BATCH:
            raise ValueError(f"artifact batch must be {ARTIFACT_BATCH}, got {artifact_batch}")
        if agents > artifact_batch:
            raise ValueError(f"record has {agents} agents, artifact supports {artifact_batch}")
        padded = torch.zeros(self.runner.input_shape, device=source.device, dtype=torch.float32)
        padded[:agents] = source.float()
        encoded = self.runner(padded)
        if tuple(encoded.shape) != tuple(self.runner.output_shape):
            raise ValueError(
                f"dense-body output shape drift: {tuple(encoded.shape)} "
                f"!= {tuple(self.runner.output_shape)}"
            )
        unpadded = encoded[:agents].float()
        if self.reference is not None:
            with torch.no_grad():
                reference = self.reference(data)
            self.numerical_records.append(
                _numeric_record(unpadded, reference, len(self.numerical_records))
            )
        return {"spatial_features_2d": unpadded}


class IdentityShrinker(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value


def evaluate_numerical_sanity(
    records: Sequence[Mapping[str, Any]],
    *,
    processed_samples: int,
) -> dict[str, Any]:
    blockers: list[str] = []
    if processed_samples != SANITY_SAMPLES:
        blockers.append(f"numerical_sanity_requires_{SANITY_SAMPLES}_samples")
    if len(records) != processed_samples:
        blockers.append("numerical_record_coverage_mismatch")
    correlations = [
        float(record.get("corrcoef"))
        for record in records
        if isinstance(record.get("corrcoef"), (int, float))
    ]
    if any(record.get("finite") is not True for record in records):
        blockers.append("non_finite_numerical_output")
    mean_correlation = float(np.mean(correlations)) if correlations else None
    if mean_correlation is None or not math.isfinite(mean_correlation) or mean_correlation < 0.5:
        blockers.append("output_correlation_below_0.5")
    return {
        "passed": not blockers,
        "records": len(records),
        "corrcoef_mean": mean_correlation,
        "blockers": blockers,
    }


def evaluate_execution_gates(
    *,
    mode: str,
    requested_samples: int,
    processed_samples: int,
    failed_samples: int,
    vm_calls: int,
    fallback_samples: int,
    numerical_sanity_passed: bool,
) -> dict[str, Any]:
    if mode not in {"sanity", "full"}:
        raise ValueError(f"unsupported mode: {mode}")
    required = SANITY_SAMPLES if mode == "sanity" else FULL_AP_SAMPLES
    blockers: list[str] = []
    if requested_samples != required:
        blockers.append(f"{mode}_requested_samples_must_equal_{required}")
    if processed_samples != requested_samples:
        blockers.append("requested_samples_not_processed")
    if failed_samples:
        blockers.append("failed_samples_present")
    if fallback_samples:
        blockers.append("fallback_forbidden")
    if vm_calls != processed_samples:
        blockers.append("vm_calls_mismatch")
    if not numerical_sanity_passed:
        blockers.append("numerical_sanity_not_passed")
    passed = not blockers
    return {
        "mode": mode,
        "passed": passed,
        "publish_ap": mode == "full" and passed,
        "sanity_16": passed if mode == "sanity" else numerical_sanity_passed,
        "full_2170": passed if mode == "full" else False,
        "blockers": blockers,
    }


def build_report(
    *,
    gates: Mapping[str, Any],
    ap: tuple[float, float, float] | None = None,
    **fields: Any,
) -> dict[str, Any]:
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "success" if gates.get("passed") else "failure",
        **fields,
        "gates": dict(gates),
        "ap_measured": bool(gates.get("publish_ap")),
        "execution_device": {
            "physical_gpu_id": (
                int(os.environ["CUDA_VISIBLE_DEVICES"])
                if os.environ.get("CUDA_VISIBLE_DEVICES", "").isdigit()
                else None
            ),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    }
    if gates.get("publish_ap"):
        if ap is None:
            raise ValueError("passing full gate requires AP30/AP50/AP70")
        ap30, ap50, ap70 = (float(value) for value in ap)
        report.update(
            {
                "ap": {"ap30": ap30, "ap50": ap50, "ap70": ap70},
                "ap30": ap30,
                "ap50": ap50,
                "ap70": ap70,
            }
        )
    return report


def require_bound_sanity_report(
    report_path: Path,
    expected_sha256: str,
    expected_bindings: Mapping[str, Any],
) -> dict[str, Any]:
    report_path = Path(report_path)
    if sha256_path(report_path).lower() != str(expected_sha256).lower():
        raise ValueError("sanity report SHA256 mismatch")
    report = _load_json(report_path)
    if (
        report.get("gates", {}).get("sanity_16") is not True
        or int(report.get("processed_samples") or 0) != SANITY_SAMPLES
        or int(report.get("vm_calls") or 0) != SANITY_SAMPLES
    ):
        raise ValueError("passing 16-sample numerical sanity report is required before full AP")
    for key, expected in expected_bindings.items():
        if report.get(key) != expected:
            raise ValueError(f"sanity binding mismatch:{key}")
    return report


def _checkpoint_for_hash(args: argparse.Namespace) -> Path:
    if args.checkpoint is not None:
        checkpoint = Path(args.checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        return checkpoint
    best_validation = sorted(args.checkpoint_dir.glob("net_epoch_bestval_at*.pth"))
    if len(best_validation) != 1:
        raise ValueError("checkpoint directory must contain exactly one best-validation checkpoint")
    return best_validation[0]


def _resolve_runtime_contract(
    route_result_path: Path,
    artifact_path: Path,
    weights_path: Path,
    quant_path: Path,
) -> tuple[dict[str, Any], RuntimeContract, tuple[np.ndarray, ...], dict[str, str]]:
    result = _load_json(route_result_path)
    if result.get("status") != "success":
        raise ValueError("Route-B result status is not success")
    plan = result.get("runtime_arg_plan")
    if not isinstance(plan, list) or not plan:
        raise ValueError("Route-B result runtime_arg_plan must be a non-empty list")
    _require_declared_sha(
        result,
        artifact_path,
        label="artifact",
        keys=("artifact_sha256", "compiled_artifact_sha256", "source_compiled_artifact_sha256"),
    )
    _require_declared_sha(
        result,
        quant_path,
        label="quant contract",
        keys=("tensor_quant_params_sha256", "quant_contract_sha256"),
    )
    contract = RuntimeContract.from_payloads(plan, _load_json(quant_path))
    weights, bindings = load_runtime_weights(result, weights_path)
    return result, contract, weights, bindings


def _run_bindings(
    *,
    artifact: Path,
    route_result: Path,
    runtime_weights: Path,
    quant_contract: Path,
    checkpoint: Path,
    config: Path,
) -> dict[str, str]:
    paths = {
        "artifact": artifact,
        "result": route_result,
        "runtime_weights": runtime_weights,
        "quant_contract": quant_contract,
        "checkpoint": checkpoint,
        "config": config,
    }
    return {
        key: value
        for name, path in paths.items()
        for key, value in (
            (f"{name}_path", str(Path(path).resolve())),
            (f"{name}_sha256", sha256_path(path)),
        )
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--route-result", type=Path, required=True)
    parser.add_argument("--runtime-weights", type=Path, required=True)
    parser.add_argument("--quant-contract", type=Path, required=True)
    parser.add_argument("--mode", choices=("sanity", "full"), required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--sanity-report", type=Path)
    parser.add_argument("--sanity-report-sha256")
    parser.add_argument("--prediction-path", type=Path)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--tvm-worker-python", type=Path)
    parser.add_argument("--tvm-site", type=Path)
    parser.add_argument("--tvm-lib-dir", type=Path, action="append", default=[])
    args = parser.parse_args(argv)
    if args.mode == "full" and (args.sanity_report is None or not args.sanity_report_sha256):
        parser.error("--sanity-report and --sanity-report-sha256 are required in full mode")
    worker_fields = (args.tvm_worker_python, args.tvm_site, args.tvm_lib_dir)
    if any(worker_fields) and not all(worker_fields):
        parser.error(
            "--tvm-worker-python, --tvm-site, and at least one --tvm-lib-dir "
            "must be provided together"
        )
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    from opencood.data_utils.datasets import build_dataset
    from opencood.hypes_yaml import yaml_utils
    from opencood.tools import inference_utils, train_utils
    from opencood.utils import eval_utils

    checkpoint = _checkpoint_for_hash(args).resolve()
    bindings = _run_bindings(
        artifact=args.artifact,
        route_result=args.route_result,
        runtime_weights=args.runtime_weights,
        quant_contract=args.quant_contract,
        checkpoint=checkpoint,
        config=args.config,
    )
    _, contract, runtime_weights, weight_bindings = _resolve_runtime_contract(
        args.route_result, args.artifact, args.runtime_weights, args.quant_contract
    )
    requested = SANITY_SAMPLES if args.mode == "sanity" else FULL_AP_SAMPLES
    if args.mode == "full":
        require_bound_sanity_report(
            args.sanity_report,
            args.sanity_report_sha256,
            bindings,
        )

    hypes = yaml_utils.load_yaml(str(args.config), SimpleNamespace(model_dir=None))
    hypes["validate_dir"] = hypes["test_dir"]
    dataset = build_dataset(hypes, visualize=False, train=False)
    if len(dataset) != FULL_AP_SAMPLES:
        raise ValueError(f"OPV2V test split must contain {FULL_AP_SAMPLES} samples, got {len(dataset)}")
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
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state, strict=True)
    torch.cuda.set_device(args.gpu_id)
    model.cuda().eval()

    if args.tvm_worker_python is not None:
        runner = SubprocessTvmRelaxVmInt8Runner(
            args.artifact,
            contract,
            route_result=args.route_result,
            runtime_weights=args.runtime_weights,
            quant_contract=args.quant_contract,
            gpu_id=args.gpu_id,
            worker_python=args.tvm_worker_python,
            tvm_site=args.tvm_site,
            tvm_lib_dirs=args.tvm_lib_dir,
            workspace=args.output_json.parent / f"tvm_worker_{args.mode}_ipc",
        )
    else:
        runner = TvmRelaxVmInt8Runner(
            args.artifact,
            contract,
            runtime_weights,
            gpu_id=args.gpu_id,
        )
    close_runner = getattr(runner, "close", None)
    if close_runner is not None:
        atexit.register(close_runner)
    reference = None
    if args.mode == "sanity":
        reference = _ReferenceDenseBody(model.backbone_m1, model.shrinker_m1)
    dense_body = TvmInt8DenseBody(runner, reference)
    model.backbone_m1 = dense_body
    model.shrinker_m1 = IdentityShrinker()

    stats = {
        threshold: {"tp": [], "fp": [], "gt": 0, "score": []}
        for threshold in (0.3, 0.5, 0.7)
    }
    processed = failed = 0
    started = time.time()
    prediction_writer = (
        PredictionArtifactWriter(args.prediction_path)
        if args.prediction_path is not None
        else nullcontext(None)
    )
    with prediction_writer as writer:
        for batch in loader:
            if processed >= requested:
                break
            if batch is None:
                continue
            try:
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
                if writer is not None:
                    writer.append(sample_index=processed, result=result)
                processed += 1
            except Exception:
                failed += 1
                raise

    if args.mode == "sanity":
        numerical = evaluate_numerical_sanity(
            dense_body.numerical_records,
            processed_samples=processed,
        )
        sanity_passed = numerical["passed"]
    else:
        numerical = {"passed": True, "source": str(args.sanity_report.resolve())}
        sanity_passed = True
    gates = evaluate_execution_gates(
        mode=args.mode,
        requested_samples=requested,
        processed_samples=processed,
        failed_samples=failed,
        vm_calls=runner.call_count,
        fallback_samples=0,
        numerical_sanity_passed=sanity_passed,
    )
    ap = None
    if gates["publish_ap"]:
        ap = tuple(
            eval_utils.calculate_ap(stats, threshold)[0]
            for threshold in (0.3, 0.5, 0.7)
        )
    report = build_report(
        gates=gates,
        ap=ap,
        dataset="OPV2V",
        split="test",
        pipeline_scope=PIPELINE_SCOPE,
        requested_samples=requested,
        processed_samples=processed,
        failed_samples=failed,
        vm_calls=runner.call_count,
        fallback_samples=0,
        elapsed_seconds=time.time() - started,
        runtime_weight_bindings=weight_bindings,
        numerical_sanity=numerical,
        numerical_contract={
            "io_cardinality": {"inputs": 1, "outputs": 1},
            "input_shape": list(contract.input_shape),
            "output_shape": list(contract.output_shape),
            "artifact_batch": ARTIFACT_BATCH,
            "graph_input_dtype": "uint8",
            "graph_output_dtype": "uint8",
            "model_boundary_dtype": "float32",
            "quantization": "round_clip_static_per_tensor_scale_zero_point",
            "agent_batch_padding": "zero_pad_to_5_then_unpad",
            "one_vm_call_per_opv2v_sample": True,
            "fallback_forbidden": True,
        },
        **bindings,
    )
    if args.prediction_path is not None:
        report["prediction_path"] = str(args.prediction_path.resolve())
        report["prediction_sha256"] = sha256_path(args.prediction_path)
    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if close_runner is not None:
        close_runner()
        atexit.unregister(close_runner)
    if not gates["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
