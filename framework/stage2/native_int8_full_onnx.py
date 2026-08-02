"""Contracts for Stage2 native INT8 full ONNX topology route evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Any


FULL_ONNX_LATENCY_ROWS = "native_int8_full_onnx_latency_rows_v1.jsonl"
FULL_ONNX_ENERGY_ROWS = "native_int8_full_onnx_energy_rows_v1.jsonl"
REQUIRED_FULL_ONNX_OPS = ("Conv", "Relu", "Add", "Identity")
REQUIRED_FULL_ONNX_BLOCKER_FIELDS = (
    "op_name",
    "op_type",
    "shape",
    "group",
    "stride",
    "pad",
    "stdout_path",
    "stderr_path",
    "traceback",
    "build_status",
    "run_status",
    "failure_reason",
)
TVM_WORKER_REQUEST_SCHEMA = "native_int8_tvm_worker_request_v1"
TVM_WORKER_RESPONSE_SCHEMA = "native_int8_tvm_worker_response_v1"


class NativeInt8FullOnnxError(ValueError):
    """Raised when full ONNX native INT8 route evidence is incomplete."""


def expected_full_onnx_row_paths(repo_root: Path) -> tuple[Path, Path]:
    rows_dir = Path(repo_root) / "rows"
    return rows_dir / FULL_ONNX_LATENCY_ROWS, rows_dir / FULL_ONNX_ENERGY_ROWS


def build_conv_op_spec(
    *,
    name: str,
    input_name: str,
    output_name: str,
    input_shape: list[int],
    output_shape: list[int],
    weight_shape: list[int],
    group: int,
    strides: list[int],
    pads: list[int],
) -> dict[str, Any]:
    return {
        "op_name": name,
        "op_type": "Conv",
        "input_name": input_name,
        "output_name": output_name,
        "input_shape": [int(item) for item in input_shape],
        "output_shape": [int(item) for item in output_shape],
        "weight_shape": [int(item) for item in weight_shape],
        "group": int(group),
        "strides": [int(item) for item in strides],
        "pads": [int(item) for item in pads],
    }


def parse_input_shape_overrides(items: list[str] | tuple[str, ...] | None) -> dict[str, list[int]]:
    overrides: dict[str, list[int]] = {}
    for item in items or []:
        if "=" not in str(item):
            raise NativeInt8FullOnnxError(f"shape override must be name=N,C,H,W: {item}")
        name, value = str(item).split("=", 1)
        name = name.strip()
        if not name:
            raise NativeInt8FullOnnxError("shape override input name is empty")
        parts = [part.strip() for part in value.split(",") if part.strip()]
        if len(parts) != 4:
            raise NativeInt8FullOnnxError(f"shape override must be NCHW with 4 dims: {item}")
        try:
            shape = [int(part) for part in parts]
        except ValueError as exc:
            raise NativeInt8FullOnnxError(f"shape override contains non-integer dim: {item}") from exc
        if any(dim <= 0 for dim in shape):
            raise NativeInt8FullOnnxError(f"shape override dims must be positive: {item}")
        overrides[name] = shape
    return overrides


def build_runtime_arg_plan(
    *,
    graph_input_shapes: dict[str, list[int]],
    weight_inputs: list[dict[str, Any]],
    output_shapes: dict[str, list[int]],
) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    for name, shape in graph_input_shapes.items():
        plan.append(
            {
                "role": "graph_input",
                "arg_name": str(name),
                "shape": [int(item) for item in shape],
                "dtype": "uint8",
                "source": "runtime_activation_quantized_uint8",
            }
        )
    for item in weight_inputs:
        quantization = item.get("quantization") or {}
        role = str(item.get("role") or "weight_input")
        dtype = str(item.get("dtype") or ("int32" if role == "bias_input" else "int8"))
        source = (
            "onnx_initializer_quantized_int32_bias"
            if role == "bias_input"
            else "onnx_initializer_quantized_int8"
        )
        plan.append(
            {
                "role": role,
                "arg_name": str(item["arg_name"]),
                "initializer_name": str(item["initializer_name"]),
                "shape": [int(dim) for dim in item["shape"]],
                "dtype": dtype,
                "source": source,
                "quantization": {
                    "scheme": str(quantization.get("scheme") or "symmetric_absmax_int8"),
                    "scale": float(quantization.get("scale") or 1.0),
                    "zero_point": int(quantization.get("zero_point") or 0),
                },
            }
        )
    for name, shape in output_shapes.items():
        plan.append(
            {
                "role": "graph_output",
                "arg_name": str(name),
                "shape": [int(item) for item in shape],
                "dtype": "uint8",
                "source": "tvm_native_int8_route_output",
            }
        )
    return plan


def build_tvm_worker_request(
    *,
    label: str,
    run_id: str,
    artifact_path: Path,
    inventory_path: Path,
    runtime_weight_archive_path: Path,
    activation_npy_path: Path,
    output_dir: Path,
    gpu: int,
    runtime_arg_plan: list[dict[str, Any]],
) -> dict[str, Any]:
    expected_output_shapes = {
        str(item["arg_name"]): [int(dim) for dim in item["shape"]]
        for item in runtime_arg_plan
        if item.get("role") == "graph_output"
    }
    if not expected_output_shapes:
        raise NativeInt8FullOnnxError("worker request requires at least one graph_output arg")
    return {
        "schema": TVM_WORKER_REQUEST_SCHEMA,
        "label": str(label),
        "run_id": str(run_id),
        "gpu": int(gpu),
        "artifact_path": str(artifact_path),
        "inventory_path": str(inventory_path),
        "runtime_weight_archive_path": str(runtime_weight_archive_path),
        "activation_npy_path": str(activation_npy_path),
        "output_dir": str(output_dir),
        "runtime_arg_plan": runtime_arg_plan,
        "expected_output_shapes": expected_output_shapes,
    }


def validate_tvm_worker_response(
    response: dict[str, Any],
    *,
    expected_output_shapes: dict[str, list[int]],
) -> None:
    if response.get("schema") != TVM_WORKER_RESPONSE_SCHEMA:
        raise NativeInt8FullOnnxError("invalid worker response schema")
    if response.get("status") != "success":
        raise NativeInt8FullOnnxError(f"worker response status is not success: {response.get('status')}")
    outputs = response.get("outputs")
    if not isinstance(outputs, list):
        raise NativeInt8FullOnnxError("worker response outputs must be a list")
    output_by_name = {str(item.get("arg_name")): item for item in outputs if isinstance(item, dict)}
    for name, expected_shape in expected_output_shapes.items():
        if name not in output_by_name:
            raise NativeInt8FullOnnxError(f"missing worker output: {name}")
        output = output_by_name[name]
        if [int(dim) for dim in output.get("shape", [])] != [int(dim) for dim in expected_shape]:
            raise NativeInt8FullOnnxError(f"worker output shape mismatch: {name}")
        if str(output.get("dtype")) != "uint8":
            raise NativeInt8FullOnnxError(f"worker output dtype mismatch: {name}")
        if not output.get("path"):
            raise NativeInt8FullOnnxError(f"worker output path is empty: {name}")


def quantize_activation_uint8(values: Any, *, tensor_name: str) -> tuple[Any, dict[str, Any]]:
    import numpy as np

    array = np.asarray(values, dtype=np.float32)
    if array.size == 0:
        raise NativeInt8FullOnnxError("cannot quantize empty activation tensor")
    min_value = float(np.min(array))
    max_value = float(np.max(array))
    if min_value == max_value:
        quantized = np.zeros(array.shape, dtype=np.uint8)
        scale = 1.0
        zero_point = 0
    else:
        scale = float((max_value - min_value) / 255.0)
        zero_point = int(np.clip(np.round(-min_value / scale), 0, 255))
        quant_float = (array.astype(np.float64) - min_value) / scale
        quantized = np.clip(np.floor(quant_float + 0.5), 0, 255).astype(np.uint8)
    summary = {
        "schema": "native_int8_activation_quant_summary_v1",
        "tensor_name": str(tensor_name),
        "scheme": "asymmetric_minmax_uint8",
        "shape": [int(dim) for dim in array.shape],
        "dtype": str(array.dtype),
        "min": min_value,
        "max": max_value,
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "scale": scale,
        "zero_point": zero_point,
        "quantized_dtype": "uint8",
    }
    return quantized, summary


def quantize_activation_uint8_static(
    values: Any,
    *,
    tensor_name: str,
    scale: float,
    zero_point: int,
    source: str,
) -> tuple[Any, dict[str, Any]]:
    import numpy as np

    array = np.asarray(values, dtype=np.float32)
    if array.size == 0:
        raise NativeInt8FullOnnxError("cannot quantize empty activation tensor")
    scale = float(scale)
    if scale <= 0.0:
        raise NativeInt8FullOnnxError(f"static activation quantization scale must be positive: {scale}")
    zero_point = int(zero_point)
    if zero_point < 0 or zero_point > 255:
        raise NativeInt8FullOnnxError(f"static activation quantization zero_point must be in [0,255]: {zero_point}")
    quantized = np.clip(np.rint(array.astype(np.float64) / scale + zero_point), 0, 255).astype(np.uint8)
    summary = {
        "schema": "native_int8_activation_quant_summary_v1",
        "tensor_name": str(tensor_name),
        "scheme": "calibration_static_uint8",
        "source": str(source),
        "shape": [int(dim) for dim in array.shape],
        "dtype": str(array.dtype),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "scale": scale,
        "zero_point": zero_point,
        "quantized_dtype": "uint8",
    }
    return quantized, summary


def resolve_initializer_name(name: str, aliases: dict[str, str]) -> str:
    current = str(name)
    seen: set[str] = set()
    while current in aliases:
        if current in seen:
            raise NativeInt8FullOnnxError(f"initializer alias cycle at: {current}")
        seen.add(current)
        current = str(aliases[current])
    return current


def validate_full_onnx_blocker(blocker: dict[str, Any]) -> None:
    for field in REQUIRED_FULL_ONNX_BLOCKER_FIELDS:
        if field not in blocker:
            raise NativeInt8FullOnnxError(f"missing full ONNX blocker field: {field}")
        value = blocker[field]
        if value is None or value == "":
            raise NativeInt8FullOnnxError(f"empty full ONNX blocker field: {field}")
    if str(blocker["op_type"]) not in REQUIRED_FULL_ONNX_OPS:
        raise NativeInt8FullOnnxError(f"unsupported blocker op_type: {blocker['op_type']}")
