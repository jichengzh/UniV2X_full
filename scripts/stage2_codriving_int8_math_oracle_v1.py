#!/usr/bin/env python3
"""Independent NumPy oracle for CoDriving Route-B INT8 convolution math."""

from __future__ import annotations

import argparse
import copy
import json
import socket
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np


def _positive_scale(scale: float, *, name: str) -> float:
    value = float(scale)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def quantize_s8_reference(values: Any, *, scale: float) -> tuple[np.ndarray, dict[str, float | int]]:
    """Apply symmetric signed INT8 quantization using round-to-nearest-even."""

    scale_value = _positive_scale(scale, name="scale")
    normalized = np.asarray(values, dtype="float32") / scale_value
    rounded = np.rint(normalized)
    saturated = np.logical_or(rounded < -127.0, rounded > 127.0)
    quantized = np.clip(rounded, -127.0, 127.0).astype("int8")
    element_count = int(quantized.size)
    saturation_count = int(np.count_nonzero(saturated))
    return quantized, {
        "element_count": element_count,
        "saturation_count": saturation_count,
        "saturation_ratio": saturation_count / element_count if element_count else 0.0,
    }


def _conv_parameters(
    activation: np.ndarray,
    weight: np.ndarray,
    *,
    strides: Sequence[int],
    pads: Sequence[int],
    group: int,
) -> tuple[int, int, int, int, int, int, int, int, int, int, int]:
    if activation.ndim != 4 or weight.ndim != 4:
        raise ValueError("activation and weight must be NCHW/OIHW rank-4 tensors")
    if len(strides) != 2 or len(pads) != 4:
        raise ValueError("strides must have 2 values and pads must have 4 values")
    if group <= 0:
        raise ValueError("group must be positive")
    n, input_channels, input_height, input_width = map(int, activation.shape)
    output_channels, channels_per_group, kernel_height, kernel_width = map(int, weight.shape)
    if input_channels != channels_per_group * group:
        raise ValueError("activation channels do not match grouped weight channels")
    if output_channels % group != 0:
        raise ValueError("output channels must be divisible by group")
    stride_height, stride_width = map(int, strides)
    pad_top, pad_left, pad_bottom, pad_right = map(int, pads)
    if stride_height <= 0 or stride_width <= 0 or min(pads) < 0:
        raise ValueError("strides must be positive and pads must be nonnegative")
    output_height = (input_height + pad_top + pad_bottom - kernel_height) // stride_height + 1
    output_width = (input_width + pad_left + pad_right - kernel_width) // stride_width + 1
    if output_height <= 0 or output_width <= 0:
        raise ValueError("convolution output has a nonpositive spatial dimension")
    return (
        n,
        input_channels,
        input_height,
        input_width,
        output_channels,
        channels_per_group,
        kernel_height,
        kernel_width,
        output_height,
        output_width,
        output_channels // group,
    )


def sample_conv2d_accumulators(
    activation: Any,
    weight: Any,
    *,
    samples: Sequence[Sequence[int]],
    strides: Sequence[int],
    pads: Sequence[int],
    group: int,
) -> list[int]:
    """Compute exact int32 accumulators only at requested NCHW outputs."""

    activation_array = np.asarray(activation, dtype="int8")
    weight_array = np.asarray(weight, dtype="int8")
    (
        batch_size,
        _,
        input_height,
        input_width,
        output_channels,
        channels_per_group,
        kernel_height,
        kernel_width,
        output_height,
        output_width,
        output_channels_per_group,
    ) = _conv_parameters(
        activation_array,
        weight_array,
        strides=strides,
        pads=pads,
        group=group,
    )
    stride_height, stride_width = map(int, strides)
    pad_top, pad_left, _, _ = map(int, pads)
    values: list[int] = []
    for sample in samples:
        if len(sample) != 4:
            raise ValueError("each output sample must contain N, C, H, W")
        batch, output_channel, output_y, output_x = map(int, sample)
        if not (
            0 <= batch < batch_size
            and 0 <= output_channel < output_channels
            and 0 <= output_y < output_height
            and 0 <= output_x < output_width
        ):
            raise IndexError(f"output sample is out of bounds: {tuple(sample)}")
        group_index = output_channel // output_channels_per_group
        input_channel_start = group_index * channels_per_group
        accumulator = 0
        for channel_offset in range(channels_per_group):
            input_channel = input_channel_start + channel_offset
            for kernel_y in range(kernel_height):
                input_y = output_y * stride_height + kernel_y - pad_top
                if input_y < 0 or input_y >= input_height:
                    continue
                for kernel_x in range(kernel_width):
                    input_x = output_x * stride_width + kernel_x - pad_left
                    if input_x < 0 or input_x >= input_width:
                        continue
                    accumulator += int(activation_array[batch, input_channel, input_y, input_x]) * int(
                        weight_array[output_channel, channel_offset, kernel_y, kernel_x]
                    )
        values.append(accumulator)
    return values


def conv2d_nchw_int32_reference(
    activation: Any,
    weight: Any,
    *,
    strides: Sequence[int],
    pads: Sequence[int],
    group: int,
) -> np.ndarray:
    """Compute a full NCHW convolution with an exact int32 accumulator."""

    activation_array = np.asarray(activation, dtype="int8")
    weight_array = np.asarray(weight, dtype="int8")
    parameters = _conv_parameters(
        activation_array,
        weight_array,
        strides=strides,
        pads=pads,
        group=group,
    )
    output_shape = (parameters[0], parameters[4], parameters[8], parameters[9])
    samples = list(np.ndindex(output_shape))
    values = sample_conv2d_accumulators(
        activation_array,
        weight_array,
        samples=samples,
        strides=strides,
        pads=pads,
        group=group,
    )
    return np.asarray(values, dtype="int32").reshape(output_shape)


def requantize_s8_reference(
    accumulator: Any,
    bias: Any,
    *,
    input_scale: float,
    weight_scale: float,
    output_scale: float,
) -> np.ndarray:
    """Apply FP bias and requantize an int32 NCHW accumulator to signed INT8."""

    input_scale_value = _positive_scale(input_scale, name="input_scale")
    weight_scale_value = _positive_scale(weight_scale, name="weight_scale")
    output_scale_value = _positive_scale(output_scale, name="output_scale")
    accumulator_array = np.asarray(accumulator, dtype="int32")
    bias_array = np.asarray(bias, dtype="float16").astype("float32")
    if accumulator_array.ndim != 4 or bias_array.ndim != 1:
        raise ValueError("accumulator must be NCHW rank-4 and bias must be rank-1")
    if accumulator_array.shape[1] != bias_array.shape[0]:
        raise ValueError("bias length must equal accumulator channels")
    fp32 = (
        accumulator_array.astype("float32") * input_scale_value * weight_scale_value
        + bias_array.reshape(1, -1, 1, 1)
    )
    return np.clip(np.rint(fp32 / output_scale_value), -127.0, 127.0).astype("int8")


def deterministic_output_samples(shape: Sequence[int], *, count: int, seed: int) -> list[tuple[int, int, int, int]]:
    """Select unique deterministic NCHW coordinates and always include both corners."""

    if len(shape) != 4 or any(int(dimension) <= 0 for dimension in shape):
        raise ValueError("shape must contain four positive NCHW dimensions")
    if count <= 0:
        raise ValueError("count must be positive")
    dimensions = tuple(map(int, shape))
    element_count = int(np.prod(dimensions, dtype="int64"))
    target_count = min(int(count), element_count)
    first = (0, 0, 0, 0)
    last = tuple(dimension - 1 for dimension in dimensions)
    chosen = {first, last}
    rng = np.random.default_rng(int(seed))
    while len(chosen) < target_count:
        flat_index = int(rng.integers(0, element_count))
        chosen.add(tuple(map(int, np.unravel_index(flat_index, dimensions))))
    ordered = [first]
    ordered.extend(sorted(chosen - {first, last}))
    if last != first and len(ordered) < target_count:
        ordered.append(last)
    return ordered[:target_count]


def extract_tvm_accumulator_samples(
    accumulator: Any,
    *,
    samples: Sequence[Sequence[int]],
    output_shape: Sequence[int],
    group: int,
) -> list[int]:
    """Read NCHW points from Route-B's grouped, flattened and OC-padded output."""

    if len(output_shape) != 4:
        raise ValueError("output_shape must be NCHW")
    batch_size, output_channels, output_height, output_width = map(int, output_shape)
    if group <= 0 or output_channels % group != 0:
        raise ValueError("output channels must be divisible by a positive group")
    array = np.asarray(accumulator, dtype="int32")
    output_channels_per_group = output_channels // group
    expected_rows = batch_size * output_height * output_width
    if array.ndim != 3 or array.shape[0] != group or array.shape[1] != expected_rows:
        raise ValueError("unexpected TVM accumulator layout")
    if array.shape[2] < output_channels_per_group:
        raise ValueError("TVM accumulator lacks logical output channels")
    values: list[int] = []
    for sample in samples:
        if len(sample) != 4:
            raise ValueError("each output sample must contain N, C, H, W")
        batch, output_channel, output_y, output_x = map(int, sample)
        if not (
            0 <= batch < batch_size
            and 0 <= output_channel < output_channels
            and 0 <= output_y < output_height
            and 0 <= output_x < output_width
        ):
            raise IndexError(f"output sample is out of bounds: {tuple(sample)}")
        group_index = output_channel // output_channels_per_group
        channel_in_group = output_channel % output_channels_per_group
        row = batch * output_height * output_width + output_y * output_width + output_x
        values.append(int(array[group_index, row, channel_in_group]))
    return values


def compare_sampled_accumulators(
    *,
    samples: Sequence[Sequence[int]],
    reference: Sequence[int],
    candidate: Sequence[int],
) -> dict[str, Any]:
    """Produce a compact, exact comparison report for sampled int32 outputs."""

    if not (len(samples) == len(reference) == len(candidate)):
        raise ValueError("samples, reference, and candidate must have equal lengths")
    mismatches = []
    max_abs_error = 0
    for coordinate, expected, observed in zip(samples, reference, candidate):
        error = abs(int(observed) - int(expected))
        max_abs_error = max(max_abs_error, error)
        if error:
            mismatches.append(
                {
                    "coordinate": list(map(int, coordinate)),
                    "reference": int(expected),
                    "candidate": int(observed),
                    "abs_error": error,
                }
            )
    return {
        "sample_count": len(samples),
        "exact_match": not mismatches,
        "mismatch_count": len(mismatches),
        "max_abs_error": max_abs_error,
        "mismatches": mismatches,
    }


def _sampled_epilogue_fp32(
    accumulators: Sequence[int],
    *,
    samples: Sequence[Sequence[int]],
    bias: Any,
    accumulator_scale: float,
) -> np.ndarray:

    if len(accumulators) != len(samples):
        raise ValueError("accumulators and samples must have equal lengths")
    scale = _positive_scale(accumulator_scale, name="accumulator_scale")
    bias_array = np.asarray(bias, dtype="float16").astype("float32")
    values = []
    for accumulator, sample in zip(accumulators, samples):
        if len(sample) != 4:
            raise ValueError("each output sample must contain N, C, H, W")
        output_channel = int(sample[1])
        if not 0 <= output_channel < len(bias_array):
            raise IndexError(f"output channel is out of bounds: {output_channel}")
        values.append(np.float32(accumulator) * np.float32(scale) + bias_array[output_channel])
    return np.asarray(values, dtype="float32")


def dequantize_sampled_fp16_reference(
    accumulators: Sequence[int],
    *,
    samples: Sequence[Sequence[int]],
    bias: Any,
    accumulator_scale: float,
) -> np.ndarray:
    """Apply the production FP32 bias epilogue and cast sampled outputs to FP16."""

    return _sampled_epilogue_fp32(
        accumulators,
        samples=samples,
        bias=bias,
        accumulator_scale=accumulator_scale,
    ).astype("float16")


def requantize_sampled_s8_reference(
    accumulators: Sequence[int],
    *,
    samples: Sequence[Sequence[int]],
    bias: Any,
    accumulator_scale: float,
    output_scale: float,
) -> np.ndarray:
    """Apply bias and symmetric signed requantization to sampled outputs."""

    output_scale_value = _positive_scale(output_scale, name="output_scale")
    dequantized = _sampled_epilogue_fp32(
        accumulators,
        samples=samples,
        bias=bias,
        accumulator_scale=accumulator_scale,
    )
    return np.clip(np.rint(dequantized / output_scale_value), -127.0, 127.0).astype("int8")


def _load_lowering() -> Any:
    import stage2_codriving_mixed_auto_lowering_v1 as lowering

    return lowering


def _read_input_sample(input_npz: Path, expected_shape: Sequence[int]) -> np.ndarray:
    with np.load(input_npz, allow_pickle=False) as payload:
        if "spatial_features" not in payload:
            raise ValueError("input NPZ lacks spatial_features")
        value = np.asarray(payload["spatial_features"], dtype="float32")
    expected = tuple(map(int, expected_shape))
    if value.shape == expected:
        return value
    if value.ndim == len(expected) + 1 and tuple(value.shape[1:]) == expected:
        return np.asarray(value[0], dtype="float32")
    raise ValueError(f"input NPZ shape {value.shape} does not match model input {expected}")


def _capture_conv_inputs(
    onnx_path: Path,
    input_npz: Path,
    records: Sequence[dict[str, Any]],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    import onnx
    import onnxruntime as ort
    from onnx import helper

    inferred = onnx.shape_inference.infer_shapes(onnx.load(str(onnx_path)))
    initializer_names = {str(item.name) for item in inferred.graph.initializer}
    runtime_inputs = [item for item in inferred.graph.input if str(item.name) not in initializer_names]
    if len(runtime_inputs) != 1:
        raise ValueError(f"oracle currently requires one runtime ONNX input, found {len(runtime_inputs)}")
    input_info = runtime_inputs[0]
    input_shape = [int(dim.dim_value) for dim in input_info.type.tensor_type.shape.dim]
    model_input = _read_input_sample(input_npz, input_shape)
    graph_input_name = str(input_info.name)

    shape_map: dict[str, list[int]] = {}
    for item in [*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output]:
        if item.type.HasField("tensor_type"):
            shape_map[str(item.name)] = [int(dim.dim_value) for dim in item.type.tensor_type.shape.dim]
    requested = list(dict.fromkeys(str(record["input_name"]) for record in records))
    internal = [name for name in requested if name != graph_input_name]
    augmented = copy.deepcopy(inferred)
    existing_outputs = {str(item.name) for item in augmented.graph.output}
    for name in internal:
        if name not in shape_map:
            raise ValueError(f"shape inference lacks selected Conv input: {name}")
        if name not in existing_outputs:
            augmented.graph.output.append(helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape_map[name]))
    session = ort.InferenceSession(augmented.SerializeToString(), providers=["CPUExecutionProvider"])
    captured_values = session.run(internal, {graph_input_name: model_input}) if internal else []
    captured = {graph_input_name: model_input, **dict(zip(internal, captured_values))}
    return (
        {name: np.asarray(captured[name], dtype="float32") for name in requested},
        {
            "provider": "CPUExecutionProvider",
            "model_input_name": graph_input_name,
            "model_input_shape": list(model_input.shape),
            "captured_tensor_names": requested,
        },
    )


def _conv_runtime_specs(onnx_path: Path, records: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    import onnx
    from onnx import helper, numpy_helper

    model = onnx.load(str(onnx_path))
    initializers = {str(item.name): item for item in model.graph.initializer}
    specs: dict[str, dict[str, Any]] = {}
    for record in records:
        node = model.graph.node[int(record["node_index"])]
        attrs = {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}
        raw_weight = np.asarray(numpy_helper.to_array(initializers[str(record["weight_name"])]), dtype="float32")
        if len(node.input) >= 3 and str(node.input[2]) in initializers:
            bias = np.asarray(numpy_helper.to_array(initializers[str(node.input[2])]), dtype="float16")
        else:
            bias = np.zeros((int(record["weight_shape"][0]),), dtype="float16")
        specs[str(record["node_id"])] = {
            "input_shape": list(map(int, record["input_shape"])),
            "weight_shape": list(map(int, record["weight_shape"])),
            "output_shape": list(map(int, record["output_shape"])),
            "group": int(record["group"]),
            "strides": list(map(int, attrs.get("strides", [1, 1]))),
            "pads": list(map(int, attrs.get("pads", [0, 0, 0, 0]))),
            "weight": raw_weight,
            "bias": bias,
        }
    return specs


def _run_tvm_accumulator(
    stack: dict[str, Any],
    *,
    spec: dict[str, Any],
    activation: np.ndarray,
    weight: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    lowering = _load_lowering()
    tvm = stack["tvm"]
    primfunc = lowering.make_s8_conv_accum_primfunc(stack, spec, "main")
    scheduled, schedule_records = lowering.schedule_mixed_module(
        stack,
        tvm.IRModule({"main": primfunc}),
        {"main": "int8_accum"},
    )
    with stack["target"], tvm.transform.PassContext(opt_level=3):
        try:
            runtime_module = tvm.build(scheduled, target=stack["target"])
        except Exception:
            runtime_module = tvm.compile(scheduled, target=stack["target"])
    output_shape = list(map(int, spec["output_shape"]))
    group = int(spec["group"])
    nper = output_shape[1] // group
    accumulator_shape = (group, output_shape[0] * output_shape[2] * output_shape[3], max(128, nper))
    output = tvm.runtime.tensor(np.zeros(accumulator_shape, dtype="int32"), device=stack["dev"])
    runtime_module["main"](
        tvm.runtime.tensor(activation, device=stack["dev"]),
        tvm.runtime.tensor(weight, device=stack["dev"]),
        output,
    )
    stack["dev"].sync()
    return np.asarray(output.numpy(), dtype="int32"), schedule_records


def _run_tvm_primfunc(
    stack: dict[str, Any],
    *,
    primfunc: Any,
    role: str,
    inputs: Sequence[np.ndarray],
    output_shape: Sequence[int],
    output_dtype: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    lowering = _load_lowering()
    tvm = stack["tvm"]
    scheduled, schedule_records = lowering.schedule_mixed_module(
        stack,
        tvm.IRModule({"main": primfunc}),
        {"main": role},
    )
    with stack["target"], tvm.transform.PassContext(opt_level=3):
        try:
            runtime_module = tvm.build(scheduled, target=stack["target"])
        except Exception:
            runtime_module = tvm.compile(scheduled, target=stack["target"])
    output = tvm.runtime.tensor(
        np.zeros(tuple(map(int, output_shape)), dtype=output_dtype),
        device=stack["dev"],
    )
    runtime_inputs = [tvm.runtime.tensor(value, device=stack["dev"]) for value in inputs]
    runtime_module["main"](*runtime_inputs, output)
    stack["dev"].sync()
    return np.asarray(output.numpy(), dtype=output_dtype), schedule_records


def _exact_array_comparison(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    reference_array = np.asarray(reference)
    candidate_array = np.asarray(candidate)
    if reference_array.shape != candidate_array.shape:
        return {
            "exact_match": False,
            "shape_match": False,
            "reference_shape": list(reference_array.shape),
            "candidate_shape": list(candidate_array.shape),
        }
    mismatch = reference_array != candidate_array
    mismatch_count = int(np.count_nonzero(mismatch))
    difference = np.abs(reference_array.astype("float64") - candidate_array.astype("float64"))
    return {
        "exact_match": mismatch_count == 0,
        "shape_match": True,
        "element_count": int(reference_array.size),
        "mismatch_count": mismatch_count,
        "max_abs_error": float(difference.max()) if difference.size else 0.0,
        "mean_abs_error": float(difference.mean()) if difference.size else 0.0,
    }


def build_oracle_report(
    *,
    onnx_path: Path,
    calibration_path: Path,
    input_npz: Path,
    width: str,
    selected_conv_limit: int,
    sample_count: int,
    seed: int,
    gpu: str | None,
) -> dict[str, Any]:
    lowering = _load_lowering()
    if selected_conv_limit <= 0:
        raise ValueError("selected_conv_limit must be positive")
    records = lowering.collect_onnx_conv_records(onnx_path)
    onnx_sha256 = lowering.sha256_file(onnx_path)
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    lowering.validate_per_node_calibration_for_lowering(
        calibration,
        records=records,
        onnx_sha256=onnx_sha256,
    )
    ranked = sorted(records, key=lambda record: (-int(record["macs"]), str(record["node_id"])))
    selected_records = ranked[:selected_conv_limit]
    captured, capture_evidence = _capture_conv_inputs(onnx_path, input_npz, selected_records)
    runtime_specs = _conv_runtime_specs(onnx_path, selected_records)

    stack = None
    if gpu is not None:
        auto = lowering._load_auto_decomp()
        auto.cap.configure_tvm_env(str(gpu))
        stack = auto.cap.import_tvm_stack()

    conv_reports = []
    selected_scales = {
        str(record["node_id"]): lowering.resolve_node_scales(
            calibration,
            node_id=str(record["node_id"]),
            input_name=str(record["input_name"]),
            weight_name=str(record["weight_name"]),
            onnx_sha256=onnx_sha256,
        )
        for record in selected_records
    }
    for rank, record in enumerate(selected_records, start=1):
        node_id = str(record["node_id"])
        spec = runtime_specs[node_id]
        scales = selected_scales[node_id]
        # The production graph casts ONNX activations to FP16 before quantization.
        activation_fp16 = np.asarray(captured[str(record["input_name"])], dtype="float16")
        activation_s8, activation_stats = quantize_s8_reference(
            activation_fp16.astype("float32"),
            scale=float(scales["input_scale"]),
        )
        weight_s8, weight_stats = quantize_s8_reference(
            spec["weight"],
            scale=float(scales["weight_scale"]),
        )
        samples = deterministic_output_samples(spec["output_shape"], count=sample_count, seed=seed + rank)
        reference = sample_conv2d_accumulators(
            activation_s8,
            weight_s8,
            samples=samples,
            strides=spec["strides"],
            pads=spec["pads"],
            group=int(spec["group"]),
        )
        comparison = None
        schedule_records = None
        quantize_comparison = None
        quantize_schedule_records = None
        epilogue_comparison = None
        epilogue_schedule_records = None
        epilogue_kind = "dequant_fp16"
        epilogue_output_scale = None
        if stack is not None:
            quantize_primfunc = lowering.make_quantize_s8_primfunc(
                stack,
                spec["input_shape"],
                float(scales["input_scale"]),
                "main",
            )
            tvm_quantized, quantize_schedule_records = _run_tvm_primfunc(
                stack,
                primfunc=quantize_primfunc,
                role="quantize",
                inputs=[activation_fp16],
                output_shape=spec["input_shape"],
                output_dtype="int8",
            )
            quantize_comparison = _exact_array_comparison(activation_s8, tvm_quantized)
            tvm_accumulator, schedule_records = _run_tvm_accumulator(
                stack,
                spec=spec,
                activation=activation_s8,
                weight=weight_s8,
            )
            candidate = extract_tvm_accumulator_samples(
                tvm_accumulator,
                samples=samples,
                output_shape=spec["output_shape"],
                group=int(spec["group"]),
            )
            comparison = compare_sampled_accumulators(
                samples=samples,
                reference=reference,
                candidate=candidate,
            )
            if rank == 1 and len(selected_records) >= 2:
                next_node_id = str(selected_records[1]["node_id"])
                epilogue_kind = "requant_s8_for_next_selected_conv"
                epilogue_output_scale = float(selected_scales[next_node_id]["input_scale"])
            fused_primfunc = lowering.make_s8_conv_fused_primfunc(
                stack,
                spec,
                float(scales["input_scale"]) * float(scales["weight_scale"]),
                "main",
                requant_output_scale=epilogue_output_scale,
            )
            epilogue_dtype = "int8" if epilogue_output_scale is not None else "float16"
            tvm_epilogue, epilogue_schedule_records = _run_tvm_primfunc(
                stack,
                primfunc=fused_primfunc,
                role="int8_fused",
                inputs=[activation_s8, weight_s8, np.asarray(spec["bias"], dtype="float16")],
                output_shape=spec["output_shape"],
                output_dtype=epilogue_dtype,
            )
            tvm_epilogue_samples = np.asarray([tvm_epilogue[item] for item in samples], dtype=epilogue_dtype)
            if epilogue_output_scale is None:
                reference_epilogue = dequantize_sampled_fp16_reference(
                    reference,
                    samples=samples,
                    bias=spec["bias"],
                    accumulator_scale=float(scales["input_scale"]) * float(scales["weight_scale"]),
                )
            else:
                reference_epilogue = requantize_sampled_s8_reference(
                    reference,
                    samples=samples,
                    bias=spec["bias"],
                    accumulator_scale=float(scales["input_scale"]) * float(scales["weight_scale"]),
                    output_scale=epilogue_output_scale,
                )
            epilogue_comparison = _exact_array_comparison(reference_epilogue, tvm_epilogue_samples)
        conv_reports.append(
            {
                "flops_rank": rank,
                "node_id": node_id,
                "node_name": record["node_name"],
                "macs": int(record["macs"]),
                "input_name": record["input_name"],
                "input_shape": spec["input_shape"],
                "weight_shape": spec["weight_shape"],
                "output_shape": spec["output_shape"],
                "group": int(spec["group"]),
                "strides": spec["strides"],
                "pads": spec["pads"],
                "input_scale": float(scales["input_scale"]),
                "weight_scale": float(scales["weight_scale"]),
                "activation_quantization": activation_stats,
                "weight_quantization": weight_stats,
                "sample_coordinates": [list(item) for item in samples],
                "cpu_reference_accumulators": reference,
                "quantize_comparison": quantize_comparison,
                "quantize_schedule_records": quantize_schedule_records,
                "tvm_comparison": comparison,
                "schedule_records": schedule_records,
                "epilogue_kind": epilogue_kind,
                "epilogue_output_scale": epilogue_output_scale,
                "epilogue_comparison": epilogue_comparison,
                "epilogue_schedule_records": epilogue_schedule_records,
            }
        )
    exact_match = stack is not None and all(bool(item["tvm_comparison"]["exact_match"]) for item in conv_reports)
    full_math_match = stack is not None and all(
        bool(item["quantize_comparison"]["exact_match"])
        and bool(item["tvm_comparison"]["exact_match"])
        and bool(item["epilogue_comparison"]["exact_match"])
        for item in conv_reports
    )
    status = "passed_tvm_exact_math_gate" if full_math_match else "failed_tvm_exact_math_gate" if stack is not None else "cpu_reference_generated"
    return {
        "schema": "codriving_int8_math_oracle_v1",
        "status": status,
        "trusted_for_final_frontier": False,
        "scope": "E0_per_conv_quantization_and_int32_accumulator",
        "host": socket.gethostname(),
        "width": width,
        "gpu": gpu,
        "onnx": str(onnx_path),
        "onnx_sha256": onnx_sha256,
        "calibration": str(calibration_path),
        "calibration_sha256": lowering.sha256_file(calibration_path),
        "input_npz": str(input_npz),
        "input_npz_sha256": lowering.sha256_file(input_npz),
        "activation_semantics": "ORT_FP32_intermediate_cast_to_FP16_then_symmetric_s8",
        "rounding_semantics": "round_to_nearest_even_then_clip_-127_127",
        "capture_evidence": capture_evidence,
        "selected_conv_limit": selected_conv_limit,
        "sample_count_per_conv": sample_count,
        "seed": seed,
        "exact_accumulator_gate_passed": exact_match,
        "exact_quantize_accumulator_epilogue_gate_passed": full_math_match,
        "conv_reports": conv_reports,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--per-node-calibration", type=Path, required=True)
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--width", required=True)
    parser.add_argument("--selected-conv-limit", type=int, default=2)
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.onnx, args.per_node_calibration, args.input_npz):
        if not path.is_file():
            parser.error(f"input file not found: {path}")
    return args


def main() -> int:
    args = parse_args()
    started = time.time()
    report = build_oracle_report(
        onnx_path=args.onnx,
        calibration_path=args.per_node_calibration,
        input_npz=args.input_npz,
        width=args.width,
        selected_conv_limit=args.selected_conv_limit,
        sample_count=args.sample_count,
        seed=args.seed,
        gpu=args.gpu,
    )
    report = {**report, "elapsed_s": time.time() - started}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] != "failed_tvm_exact_math_gate" else 2


if __name__ == "__main__":
    raise SystemExit(main())
