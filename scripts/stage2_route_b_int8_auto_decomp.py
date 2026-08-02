#!/usr/bin/env python3
"""Route B-int8 automatic per-block decomposition experiment.

This is an experimental backend-closure probe, not the historical hand-written
full-engine im2col route. It parses the real Pyramid backbone ONNX graph,
creates one Relax call_tir PrimFunc per Conv/Relu/Add block, rewrites each Conv
as an int8 im2col->matmul block, applies MatmulInt8Tensorization per Conv
PrimFunc, then compiles/measures the composed full graph.

Correctness is checked against the existing native direct INT8 full-ONNX TE
route using the same quantized ONNX weights and the same runtime inputs.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import importlib.util
import io
import json
import math
import os
import socket
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(os.environ.get("STAGE2_V2X_ROOT", "/home/jichengzhi/V2X"))
ROUTE_PATH = ROOT / (
    "multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/"
    "raw/int8_native_route/stage2_h800_native_int8_full_onnx_route.py"
)
CAPABILITY_PATH = ROOT / (
    "multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/"
    "raw/int8_native_route/stage2_h800_native_int8_capability_probe.py"
)
DEFAULT_OUT_DIR = Path("/exdata/jichengzhi/s2_tvm/route_b_int8_auto_decomp_20260707")
DEFAULT_ONNX = Path("/exdata/jichengzhi/s2_tvm/models/smbo_64x128x256_backbone.onnx")


def load_module(path: Path, name: str) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


route = load_module(ROUTE_PATH, "route_b_int8_native_full_onnx_route")
cap = load_module(CAPABILITY_PATH, "route_b_int8_capability_probe")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def sha256(path: Path | None) -> str:
    return cap.sha256(path)


def make_tuning_budget_config(max_trials: int) -> dict[str, Any]:
    """Translate the CLI budget into strict MetaSchedule limits."""
    requested = int(max_trials)
    if requested < 0:
        raise ValueError("max_trials must be non-negative")
    if requested == 0:
        return {
            "enabled": False,
            "requested_trials": 0,
            "max_trials_global": 0,
            "max_trials_per_task": 0,
            "num_trials_per_iter": 0,
            "allocation_policy": "default_per_primfunc_tensorization_no_metaschedule",
        }
    return {
        "enabled": True,
        "requested_trials": requested,
        "max_trials_global": requested,
        "max_trials_per_task": requested,
        # Gradient scheduling initializes multiple workloads independently.  A
        # larger batch can cross the global cap before the scheduler observes
        # the completed records (for example, 1 + 1 + 1 + 64 records for a
        # nominal 64-trial run).  Single-trial dispatch keeps the on-disk
        # measurement count inside the frozen outer contract.
        "num_trials_per_iter": 1,
        "allocation_policy": "metaschedule_global_gradient_with_per_task_cap",
    }


def _record_run_secs(payload: Any) -> list[float]:
    """Read run seconds from TVM 0.20 JSONDatabase's stable line format."""
    try:
        values = payload[1][1]
    except (IndexError, KeyError, TypeError):
        return []
    if not isinstance(values, list):
        return []
    out: list[float] = []
    for value in values:
        try:
            out.append(float(value))
        except (TypeError, ValueError):
            continue
    return out


def audit_metaschedule_database(work_dir: Path, *, requested_trials: int) -> dict[str, Any]:
    """Audit the on-disk JSON database without hiding failed measurements."""
    directory = Path(work_dir)
    workload_path = directory / "database_workload.json"
    record_path = directory / "database_tuning_record.json"
    if not workload_path.is_file() or not record_path.is_file():
        raise RuntimeError(f"MetaSchedule database is incomplete: {directory}")

    attempted = 0
    valid = 0
    for line_number, raw_line in enumerate(record_path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        attempted += 1
        try:
            payload = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"invalid MetaSchedule tuning record at line {line_number}: {record_path}"
            ) from exc
        run_secs = _record_run_secs(payload)
        if run_secs and all(math.isfinite(value) and 0.0 < value < 1e9 for value in run_secs):
            valid += 1

    requested = int(requested_trials)
    if attempted > requested:
        raise RuntimeError(
            f"MetaSchedule attempted trials exceeded fixed budget: {attempted} > {requested}"
        )
    return {
        "requested_trials": requested,
        "attempted_trials": attempted,
        "valid_trials": valid,
        "database_path": str(directory),
        "database_workload_path": str(workload_path),
        "database_workload_sha256": sha256(workload_path),
        "database_tuning_record_path": str(record_path),
        "database_tuning_record_sha256": sha256(record_path),
    }


def persist_runtime_weights(
    path: Path,
    *,
    runtime_arg_plan: list[dict[str, Any]],
    weight_runtime_values: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Persist only quantized initializer arguments required by the VM ABI."""
    output_path = Path(path)
    expected = {
        str(item["arg_name"])
        for item in runtime_arg_plan
        if item.get("role") in {"weight_input", "bias_input"}
    }
    actual = {str(name) for name in weight_runtime_values}
    if actual != expected:
        raise ValueError(
            "runtime weight contract mismatch: "
            f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
        )

    arrays = {
        name: np.ascontiguousarray(np.asarray(weight_runtime_values[name]))
        for name in sorted(expected)
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    with temporary_path.open("wb") as handle:
        np.savez(handle, **arrays)
    os.replace(temporary_path, output_path)
    return {
        "path": str(output_path),
        "sha256": sha256(output_path),
        "array_names": sorted(arrays),
        "arrays": {
            name: {
                "shape": [int(value) for value in arrays[name].shape],
                "dtype": str(arrays[name].dtype),
                "nbytes": int(arrays[name].nbytes),
            }
            for name in sorted(arrays)
        },
        "random_inputs_persisted": False,
    }


def counts(text: str) -> dict[str, int]:
    keys = ["wmma", "tvm_mma_sync", "mma_sync", "ldmatrix", "mma", "dp4a", "__dp4a", "int8", "uint8", "int32"]
    return {key: text.count(key) for key in keys}


def summarize_times_ms(values_s: list[float]) -> dict[str, Any]:
    vals = sorted(float(item) * 1e3 for item in values_s)
    if not vals:
        return {}
    mid = len(vals) // 2
    p50 = vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2.0
    p95 = vals[min(len(vals) - 1, int(round(0.95 * (len(vals) - 1))))]
    p99 = vals[min(len(vals) - 1, int(round(0.99 * (len(vals) - 1))))]
    return {
        "latency_ms_p50": float(p50),
        "latency_ms_p95": float(p95),
        "latency_ms_p99": float(p99),
        "latency_ms_mean": float(sum(vals) / len(vals)),
        "latency_ms_min": float(vals[0]),
        "latency_ms_max": float(vals[-1]),
        "repeats_ms": [float(item) for item in vals],
    }


def prod_int(values: list[int] | tuple[int, ...]) -> int:
    out = 1
    for value in values:
        out *= int(value)
    return int(out)


def ceil_div(a: int, b: int) -> int:
    return (int(a) + int(b) - 1) // int(b)


def depth_to_space_source_index(
    *,
    input_channels: int,
    output_channel: int,
    output_y: int,
    output_x: int,
    block_size: int,
) -> tuple[int, int, int]:
    """Map one ONNX DepthToSpace DCR output index back to its input."""
    block = int(block_size)
    if block <= 0:
        raise ValueError("block_size must be positive")
    channels = int(input_channels)
    if channels % (block * block):
        raise ValueError("input_channels must be divisible by block_size squared")
    output_channels = channels // (block * block)
    offset = (int(output_y) % block) * block + int(output_x) % block
    channel = offset * output_channels + int(output_channel)
    return channel, int(output_y) // block, int(output_x) // block


def validate_depth_to_space_spec(
    input_shape: list[int],
    block_size: int,
    mode: str,
) -> list[int]:
    if str(mode).upper() != "DCR":
        raise ValueError("only ONNX DepthToSpace DCR mode is supported")
    if len(input_shape) != 4:
        raise ValueError("DepthToSpace input must be NCHW")
    block = int(block_size)
    if block <= 0:
        raise ValueError("DepthToSpace block_size must be positive")
    n, channels, height, width = [int(value) for value in input_shape]
    divisor = block * block
    if channels % divisor:
        raise ValueError("DepthToSpace channels must be divisible by block_size squared")
    return [n, channels // divisor, height * block, width * block]


def validate_concat_shapes(input_shapes: list[list[int]], axis: int) -> list[int]:
    if int(axis) != 1:
        raise ValueError("Route B Concat currently supports the channel axis only")
    if not input_shapes or any(len(shape) != 4 for shape in input_shapes):
        raise ValueError("Concat inputs must be nonempty NCHW tensors")
    reference = [int(value) for value in input_shapes[0]]
    for shape in input_shapes[1:]:
        normalized = [int(value) for value in shape]
        if (normalized[0], normalized[2], normalized[3]) != (
            reference[0],
            reference[2],
            reference[3],
        ):
            raise ValueError("Concat non-channel dimensions must match")
    return [
        reference[0],
        sum(int(shape[1]) for shape in input_shapes),
        reference[2],
        reference[3],
    ]


def dtype_nbytes(dtype: str) -> int:
    text = str(dtype)
    if text in {"float16", "int16", "uint16"}:
        return 2
    if text in {"float32", "int32", "uint32"}:
        return 4
    if text in {"float64", "int64", "uint64"}:
        return 8
    return 1


def conv_static_metrics(
    *,
    input_shape: list[int],
    output_shape: list[int],
    weight_shape: list[int],
    group: int,
    m_dim: int,
    k_total: int,
    nper: int,
    nper_compute: int,
) -> dict[str, Any]:
    logical_accum_elements = int(group) * int(m_dim) * int(nper)
    padded_accum_elements = int(group) * int(m_dim) * int(nper_compute)
    real_output_elements = prod_int([int(v) for v in output_shape])
    mma_tiles_padded = int(group) * ceil_div(m_dim, 16) * ceil_div(nper_compute, 16) * ceil_div(k_total, 16)
    mma_tiles_logical = int(group) * ceil_div(m_dim, 16) * ceil_div(nper, 16) * ceil_div(k_total, 16)
    return {
        "input_shape": [int(v) for v in input_shape],
        "output_shape": [int(v) for v in output_shape],
        "weight_shape": [int(v) for v in weight_shape],
        "group": int(group),
        "m_dim": int(m_dim),
        "k_total": int(k_total),
        "nper": int(nper),
        "nper_compute": int(nper_compute),
        "nper_padding_ratio": float(nper_compute) / float(nper) if nper else None,
        "logical_accum_elements": logical_accum_elements,
        "padded_accum_elements": padded_accum_elements,
        "accum_padding_ratio": float(padded_accum_elements) / float(logical_accum_elements)
        if logical_accum_elements
        else None,
        "real_output_elements": real_output_elements,
        "accum_bytes_logical": logical_accum_elements * 4,
        "accum_bytes_padded": padded_accum_elements * 4,
        "padded_extra_accum_bytes": (padded_accum_elements - logical_accum_elements) * 4,
        "real_output_bytes_uint8": real_output_elements,
        "real_output_bytes_fp16": real_output_elements * 2,
        "accum_padded_bytes_vs_uint8_output": float(padded_accum_elements * 4) / float(real_output_elements)
        if real_output_elements
        else None,
        "estimated_mma_tiles_m16n16k16_padded": mma_tiles_padded,
        "estimated_mma_tiles_m16n16k16_logical": mma_tiles_logical,
        "estimated_mma_tile_padding_ratio": float(mma_tiles_padded) / float(mma_tiles_logical)
        if mma_tiles_logical
        else None,
    }


def conv2d_out_hw(h: int, w: int, kh: int, kw: int, strides: list[int], pads: list[int]) -> tuple[int, int]:
    pt, pl, pb, pr = [int(item) for item in pads]
    sh, sw = [int(item) for item in strides]
    oh = (h + pt + pb - kh) // sh + 1
    ow = (w + pl + pr - kw) // sw + 1
    return oh, ow


def clip_u8_expr(te: Any, value: Any) -> Any:
    hi = te.if_then_else(value > te.const(255, "int32"), te.const(255, "int32"), value)
    lo = te.if_then_else(hi < te.const(0, "int32"), te.const(0, "int32"), hi)
    return lo.astype("uint8")


def make_int8_tc_conv_primfunc(stack: dict[str, Any], spec: dict[str, Any], name: str) -> Any:
    """Create one uint8/int8/int32 -> uint8 conv block exposed as int8 matmul."""
    te = stack["te"]
    n, cin, h, w = [int(v) for v in spec["input_shape"]]
    cout, cpg, kh, kw = [int(v) for v in spec["weight_shape"]]
    group = int(spec["group"])
    strides = [int(v) for v in spec["strides"]]
    pads = [int(v) for v in spec["pads"]]
    oh, ow = conv2d_out_hw(h, w, kh, kw, strides, pads)
    pt, pl, _, _ = pads
    sh, sw = strides
    m = n * oh * ow
    k_total = cpg * kh * kw
    nper = cout // group
    # MatmulInt8Tensorization pads the N/J axis aggressively. If the logical
    # per-group channel count is small, keeping matmul's storage at nper lets
    # padded columns alias following groups when the consumer is inlined. Make
    # the padded J storage explicit, and have the final NCHW output read only
    # the real per-group columns.
    nper_compute = 128 if nper < 128 else nper
    input_zp = int(spec.get("input_zero_point", 128))
    output_zp = int(spec.get("output_zero_point", 128))
    input_scale = spec.get("input_scale")
    weight_scale = spec.get("weight_scale")
    output_scale = spec.get("output_scale")
    scale_aware = input_scale is not None and weight_scale is not None and output_scale is not None

    x = te.placeholder((n, cin, h, w), "uint8", name="x")
    weight = te.placeholder((cout, cpg, kh, kw), "int8", name="weight")
    bias = te.placeholder((cout,), "int32", name="bias")

    def im2col_compute(g: Any, row: Any, kk: Any) -> Any:
        nn = row // (oh * ow)
        rem = row % (oh * ow)
        yy = rem // ow
        xx = rem % ow
        ci = kk // (kh * kw)
        rem_k = kk % (kh * kw)
        ry = rem_k // kw
        rx = rem_k % kw
        in_y = yy * sh + ry - pt
        in_x = xx * sw + rx - pl
        in_c = g * cpg + ci
        in_bounds = (in_y >= 0) & (in_y < h) & (in_x >= 0) & (in_x < w)
        centered = x[nn, in_c, in_y, in_x].astype("int32") - te.const(input_zp, "int32")
        return te.if_then_else(in_bounds, centered.astype("int8"), te.const(0, "int8"))

    x_col = te.compute((group, m, k_total), im2col_compute, name="x_col")

    def weight_compute(g: Any, kk: Any, ocg: Any) -> Any:
        ci = kk // (kh * kw)
        rem_k = kk % (kh * kw)
        ry = rem_k // kw
        rx = rem_k % kw
        valid = ocg < te.const(nper, "int32")
        safe_ocg = te.if_then_else(valid, ocg, te.const(0, "int32"))
        value = weight[g * nper + safe_ocg, ci, ry, rx]
        return te.if_then_else(valid, value, te.const(0, "int8"))

    w_mat = te.compute((group, k_total, nper_compute), weight_compute, name="w_mat")
    rk = te.reduce_axis((0, k_total), name="rk")
    matmul = te.compute(
        (group, m, nper_compute),
        lambda g, row, ocg: te.sum(
            x_col[g, row, rk].astype("int32") * w_mat[g, rk, ocg].astype("int32"),
            axis=rk,
        ),
        name="matmul",
    )

    def out_compute(nn: Any, oc: Any, yy: Any, xx: Any) -> Any:
        g = oc // nper
        ocg = oc % nper
        row = nn * (oh * ow) + yy * ow + xx
        acc = matmul[g, row, ocg] + bias[oc].astype("int32")
        if scale_aware:
            effective_scale = float(input_scale) * float(weight_scale) / float(output_scale)
            scaled = acc.astype("float32") * te.const(effective_scale, "float32") + te.const(
                float(output_zp), "float32"
            )
            rounded = route.round_expr(stack, scaled).astype("int32")
            return clip_u8_expr(te, rounded)
        shifted = acc // te.const(256, "int32") + te.const(128, "int32")
        return clip_u8_expr(te, shifted)

    out = te.compute((n, cout, oh, ow), out_compute, name="out_nchw")
    return te.create_prim_func([x, weight, bias, out]).with_attr("global_symbol", name)


def make_int8_tc_conv_accum_primfunc(stack: dict[str, Any], spec: dict[str, Any], name: str) -> Any:
    """Create uint8/int8 -> padded int32 grouped matmul accumulator.

    The final NCHW requant/pack is intentionally a separate PrimFunc. The dlight
    int8 tensorization pass inlines consumer chains and pads the J axis; keeping
    the real NCHW store outside this PrimFunc prevents padded J columns from
    aliasing later channel groups.
    """
    te = stack["te"]
    n, cin, h, w = [int(v) for v in spec["input_shape"]]
    cout, cpg, kh, kw = [int(v) for v in spec["weight_shape"]]
    group = int(spec["group"])
    strides = [int(v) for v in spec["strides"]]
    pads = [int(v) for v in spec["pads"]]
    oh, ow = conv2d_out_hw(h, w, kh, kw, strides, pads)
    pt, pl, _, _ = pads
    sh, sw = strides
    m = n * oh * ow
    k_total = cpg * kh * kw
    nper = cout // group
    nper_compute = 128 if nper < 128 else nper
    input_zp = int(spec.get("input_zero_point", 128))

    x = te.placeholder((n, cin, h, w), "uint8", name="x")
    weight = te.placeholder((cout, cpg, kh, kw), "int8", name="weight")

    def im2col_compute(g: Any, row: Any, kk: Any) -> Any:
        nn = row // (oh * ow)
        rem = row % (oh * ow)
        yy = rem // ow
        xx = rem % ow
        ci = kk // (kh * kw)
        rem_k = kk % (kh * kw)
        ry = rem_k // kw
        rx = rem_k % kw
        in_y = yy * sh + ry - pt
        in_x = xx * sw + rx - pl
        in_c = g * cpg + ci
        in_bounds = (in_y >= 0) & (in_y < h) & (in_x >= 0) & (in_x < w)
        centered = x[nn, in_c, in_y, in_x].astype("int32") - te.const(input_zp, "int32")
        return te.if_then_else(in_bounds, centered.astype("int8"), te.const(0, "int8"))

    x_col = te.compute((group, m, k_total), im2col_compute, name="x_col")

    def weight_compute(g: Any, kk: Any, ocg: Any) -> Any:
        ci = kk // (kh * kw)
        rem_k = kk % (kh * kw)
        ry = rem_k // kw
        rx = rem_k % kw
        valid = ocg < te.const(nper, "int32")
        safe_ocg = te.if_then_else(valid, ocg, te.const(0, "int32"))
        value = weight[g * nper + safe_ocg, ci, ry, rx]
        return te.if_then_else(valid, value, te.const(0, "int8"))

    w_mat = te.compute((group, k_total, nper_compute), weight_compute, name="w_mat")
    rk = te.reduce_axis((0, k_total), name="rk")
    accum = te.compute(
        (group, m, nper_compute),
        lambda g, row, ocg: te.sum(
            x_col[g, row, rk].astype("int32") * w_mat[g, rk, ocg].astype("int32"),
            axis=rk,
        ),
        name="matmul",
    )
    return te.create_prim_func([x, weight, accum]).with_attr("global_symbol", name)


def make_int8_conv_pack_primfunc(stack: dict[str, Any], spec: dict[str, Any], name: str) -> Any:
    """Pack padded per-group int32 accumulators into real uint8 NCHW output."""
    te = stack["te"]
    n, _, h, w = [int(v) for v in spec["input_shape"]]
    cout, _, kh, kw = [int(v) for v in spec["weight_shape"]]
    group = int(spec["group"])
    strides = [int(v) for v in spec["strides"]]
    pads = [int(v) for v in spec["pads"]]
    oh, ow = conv2d_out_hw(h, w, kh, kw, strides, pads)
    m = n * oh * ow
    nper = cout // group
    nper_compute = 128 if nper < 128 else nper
    output_zp = int(spec.get("output_zero_point", 128))
    input_scale = spec.get("input_scale")
    weight_scale = spec.get("weight_scale")
    output_scale = spec.get("output_scale")
    scale_aware = input_scale is not None and weight_scale is not None and output_scale is not None

    accum = te.placeholder((group, m, nper_compute), "int32", name="accum")
    bias = te.placeholder((cout,), "int32", name="bias")

    def out_compute(nn: Any, oc: Any, yy: Any, xx: Any) -> Any:
        g = oc // nper
        ocg = oc % nper
        row = nn * (oh * ow) + yy * ow + xx
        acc = accum[g, row, ocg] + bias[oc].astype("int32")
        if scale_aware:
            effective_scale = float(input_scale) * float(weight_scale) / float(output_scale)
            scaled = acc.astype("float32") * te.const(effective_scale, "float32") + te.const(
                float(output_zp), "float32"
            )
            rounded = route.round_expr(stack, scaled).astype("int32")
            return clip_u8_expr(te, rounded)
        shifted = acc // te.const(256, "int32") + te.const(128, "int32")
        return clip_u8_expr(te, shifted)

    out = te.compute((n, cout, oh, ow), out_compute, name="out_nchw")
    return te.create_prim_func([accum, bias, out]).with_attr("global_symbol", name)


def make_fp16_conv_direct_primfunc(stack: dict[str, Any], spec: dict[str, Any], name: str) -> Any:
    """Create a same-shape fp16 im2col->matmul Conv block for per-block comparison."""
    te = stack["te"]
    n, cin, h, w = [int(v) for v in spec["input_shape"]]
    cout, cpg, kh, kw = [int(v) for v in spec["weight_shape"]]
    group = int(spec["group"])
    strides = [int(v) for v in spec["strides"]]
    pads = [int(v) for v in spec["pads"]]
    oh, ow = conv2d_out_hw(h, w, kh, kw, strides, pads)
    pt, pl, _, _ = pads
    sh, sw = strides
    m = n * oh * ow
    k_total = cpg * kh * kw
    nper = cout // group

    x = te.placeholder((n, cin, h, w), "float16", name="x")
    weight = te.placeholder((cout, cpg, kh, kw), "float16", name="weight")
    bias = te.placeholder((cout,), "float16", name="bias")

    def im2col_compute(g: Any, row: Any, kk: Any) -> Any:
        nn = row // (oh * ow)
        rem = row % (oh * ow)
        yy = rem // ow
        xx = rem % ow
        ci = kk // (kh * kw)
        rem_k = kk % (kh * kw)
        ry = rem_k // kw
        rx = rem_k % kw
        in_y = yy * sh + ry - pt
        in_x = xx * sw + rx - pl
        in_c = g * cpg + ci
        in_bounds = (in_y >= 0) & (in_y < h) & (in_x >= 0) & (in_x < w)
        return te.if_then_else(in_bounds, x[nn, in_c, in_y, in_x], te.const(0, "float16"))

    x_col = te.compute((group, m, k_total), im2col_compute, name="x_col")

    def weight_compute(g: Any, kk: Any, ocg: Any) -> Any:
        ci = kk // (kh * kw)
        rem_k = kk % (kh * kw)
        ry = rem_k // kw
        rx = rem_k % kw
        return weight[g * nper + ocg, ci, ry, rx]

    w_mat = te.compute((group, k_total, nper), weight_compute, name="w_mat")
    rk = te.reduce_axis((0, k_total), name="rk")
    matmul = te.compute(
        (group, m, nper),
        lambda g, row, ocg: te.sum(x_col[g, row, rk] * w_mat[g, rk, ocg], axis=rk),
        name="matmul",
    )

    def out_compute(nn: Any, oc: Any, yy: Any, xx: Any) -> Any:
        g = oc // nper
        ocg = oc % nper
        row = nn * (oh * ow) + yy * ow + xx
        return (matmul[g, row, ocg] + bias[oc]).astype("float16")

    out = te.compute((n, cout, oh, ow), out_compute, name="out_nchw")
    return te.create_prim_func([x, weight, bias, out]).with_attr("global_symbol", name)


def make_native_direct_conv_primfunc(stack: dict[str, Any], spec: dict[str, Any], name: str) -> Any:
    """Native uint8/int8 Conv fallback for blocks not representable as s8 TensorCore."""
    te = stack["te"]
    n, cin, h, w = [int(v) for v in spec["input_shape"]]
    cout, cpg, kh, kw = [int(v) for v in spec["weight_shape"]]
    x = te.placeholder((n, cin, h, w), "uint8", name="x")
    weight = te.placeholder((cout, cpg, kh, kw), "int8", name="weight")
    bias = te.placeholder((cout,), "int32", name="bias")
    out = route.conv2d_native_int8(
        stack,
        x,
        weight,
        bias=bias,
        group=int(spec["group"]),
        strides=[int(v) for v in spec["strides"]],
        pads=[int(v) for v in spec["pads"]],
        name=name,
        input_zero_point=int(spec.get("input_zero_point", 128)),
        input_scale=spec.get("input_scale"),
        weight_scale=spec.get("weight_scale"),
        output_scale=spec.get("output_scale"),
        output_zero_point=int(spec.get("output_zero_point", 128)),
    )
    return te.create_prim_func([x, weight, bias, out]).with_attr("global_symbol", name)


def make_relu_primfunc(
    stack: dict[str, Any], shape: list[int], *, input_scale: float,
    input_zero_point: int, output_scale: float, output_zero_point: int, name: str,
) -> Any:
    te = stack["te"]
    x = te.placeholder(tuple(int(v) for v in shape), "uint8", name="x")
    out = route.relu_u8_scale_aware(
        stack, x, name="out_relu", input_scale=input_scale,
        input_zero_point=input_zero_point, output_scale=output_scale,
        output_zero_point=output_zero_point,
    )
    return te.create_prim_func([x, out]).with_attr("global_symbol", name)


def make_add_primfunc(
    stack: dict[str, Any], shape: list[int], *, lhs_scale: float,
    lhs_zero_point: int, rhs_scale: float, rhs_zero_point: int,
    output_scale: float, output_zero_point: int, name: str,
) -> Any:
    te = stack["te"]
    lhs = te.placeholder(tuple(int(v) for v in shape), "uint8", name="lhs")
    rhs = te.placeholder(tuple(int(v) for v in shape), "uint8", name="rhs")
    out = route.add_u8_scale_aware(
        stack, lhs, rhs, name="out_add", lhs_scale=lhs_scale,
        lhs_zero_point=lhs_zero_point, rhs_scale=rhs_scale,
        rhs_zero_point=rhs_zero_point, output_scale=output_scale,
        output_zero_point=output_zero_point,
    )
    return te.create_prim_func([lhs, rhs, out]).with_attr("global_symbol", name)


def make_depth_to_space_primfunc(
    stack: dict[str, Any],
    input_shape: list[int],
    *,
    block_size: int,
    name: str,
) -> Any:
    te = stack["te"]
    output_shape = validate_depth_to_space_spec(input_shape, block_size, "DCR")
    block = int(block_size)
    output_channels = int(output_shape[1])
    x = te.placeholder(tuple(int(value) for value in input_shape), "uint8", name="x")
    out = te.compute(
        tuple(output_shape),
        lambda nn, cc, yy, xx: x[
            nn,
            ((yy % block) * block + xx % block) * output_channels + cc,
            yy // block,
            xx // block,
        ],
        name="depth_to_space_dcr",
    )
    return te.create_prim_func([x, out]).with_attr("global_symbol", name)


def make_concat_requant_primfunc(
    stack: dict[str, Any],
    input_shapes: list[list[int]],
    *,
    input_scales: list[float],
    input_zero_points: list[int],
    output_scale: float,
    output_zero_point: int,
    name: str,
) -> Any:
    te = stack["te"]
    output_shape = validate_concat_shapes(input_shapes, axis=1)
    if len(input_scales) != len(input_shapes) or len(input_zero_points) != len(input_shapes):
        raise ValueError("Concat quantization metadata must match its inputs")
    tensors = [
        te.placeholder(tuple(int(value) for value in shape), "uint8", name=f"x{index}")
        for index, shape in enumerate(input_shapes)
    ]
    boundaries: list[int] = []
    total = 0
    for shape in input_shapes:
        total += int(shape[1])
        boundaries.append(total)
    starts = [0, *boundaries[:-1]]

    def compute(nn: Any, cc: Any, yy: Any, xx: Any) -> Any:
        expression = None
        for index, boundary in reversed(list(enumerate(boundaries))):
            source = tensors[index][nn, cc - starts[index], yy, xx]
            scaled = (
                (source.astype("float32") - te.const(int(input_zero_points[index]), "float32"))
                * te.const(float(input_scales[index]) / float(output_scale), "float32")
                + te.const(int(output_zero_point), "float32")
            )
            value = route.clip_u8_from_i32(
                stack,
                route.round_expr(stack, scaled).astype("int32"),
            )
            expression = value if expression is None else te.if_then_else(cc < boundary, value, expression)
        assert expression is not None
        return expression

    out = te.compute(tuple(output_shape), compute, name="concat_channel_requant")
    return te.create_prim_func([*tensors, out]).with_attr("global_symbol", name)


def relax_tensor_info(relax: Any, shape: list[int], dtype: str) -> Any:
    return relax.TensorStructInfo(tuple(int(v) for v in shape), dtype)


def build_auto_decomp_relax_module(
    stack: dict[str, Any],
    *,
    label: str,
    onnx_path: Path,
    tensor_quant_params: dict[str, Any] | None = None,
    graph_input_quant_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import onnx
    from tvm import relax

    model, shape_map, initializers = route.load_onnx_model(onnx_path)
    te = stack["te"]
    bb = relax.BlockBuilder()
    quant_params = dict(tensor_quant_params or {})
    input_quant_params = dict(graph_input_quant_params or quant_params)
    scale_aware = bool(quant_params)
    initializer_names = set(initializers)
    initializer_aliases: dict[str, str] = {}
    tensor_map: dict[str, Any] = {}
    tensor_scales: dict[str, float] = {}
    tensor_zero_points: dict[str, int] = {}
    inputs: list[Any] = []
    input_specs: list[dict[str, Any]] = []
    weight_runtime_values: dict[str, np.ndarray] = {}
    weight_inputs: list[dict[str, Any]] = []
    op_records: list[dict[str, Any]] = []
    op_counts: Counter[str] = Counter()
    block_reports: list[dict[str, Any]] = []

    for item in model.graph.input:
        if item.name in initializer_names:
            continue
        shape = shape_map.get(item.name)
        if not shape:
            raise RuntimeError(f"missing graph input shape: {item.name}")
        var_name = route.sanitize_name(item.name, "input")
        var = relax.Var(var_name, relax_tensor_info(relax, shape, "uint8"))
        tensor_map[item.name] = var
        scale, zp = route.tensor_quant_params_for(
            input_quant_params,
            item.name,
            default_scale=1.0,
            default_zero_point=0,
        )
        tensor_scales[item.name] = scale
        tensor_zero_points[item.name] = zp
        inputs.append(var)
        input_specs.append({"role": "graph_input", "arg_name": var_name, "source_name": item.name, "shape": shape, "dtype": "uint8"})

    conv_index = 0
    relu_index = 0
    add_index = 0
    depth_to_space_index = 0
    concat_index = 0
    supported_ops = {*route.REQUIRED_FULL_ONNX_OPS, "DepthToSpace", "Concat"}
    with bb.function("main", inputs):
        with bb.dataflow():
            for node_index, node in enumerate(model.graph.node):
                op_type = str(node.op_type)
                op_counts[op_type] += 1
                op_name = str(node.name or f"{op_type}_{node_index}")
                if op_type not in supported_ops:
                    raise RuntimeError(f"unsupported op type {op_type}: {op_name}")

                if op_type == "Identity":
                    resolved_input = route.resolve_initializer_name(node.input[0], initializer_aliases)
                    if resolved_input in initializers:
                        initializer_aliases[node.output[0]] = resolved_input
                        op_records.append(
                            {
                                "op_name": op_name,
                                "op_type": op_type,
                                "input_name": node.input[0],
                                "output_name": node.output[0],
                                "identity_kind": "initializer_alias",
                                "output_shape": shape_map.get(resolved_input, []),
                            }
                        )
                        continue
                    tensor_map[node.output[0]] = tensor_map[resolved_input]
                    tensor_scales[node.output[0]] = tensor_scales.get(resolved_input, 1.0)
                    tensor_zero_points[node.output[0]] = tensor_zero_points.get(resolved_input, 128)
                    op_records.append(
                        {
                            "op_name": op_name,
                            "op_type": op_type,
                            "input_name": resolved_input,
                            "output_name": node.output[0],
                            "identity_kind": "activation_alias",
                            "output_shape": shape_map.get(node.output[0], []),
                        }
                    )
                    continue

                if op_type == "DepthToSpace":
                    depth_to_space_index += 1
                    attrs = route.attr_dict(node)
                    block_size = int(attrs.get("blocksize", 0) or 0)
                    raw_mode = next(
                        (
                            onnx.helper.get_attribute_value(attribute)
                            for attribute in node.attribute
                            if attribute.name == "mode"
                        ),
                        "DCR",
                    )
                    mode = raw_mode.decode("utf-8") if isinstance(raw_mode, bytes) else str(raw_mode)
                    input_name = node.input[0]
                    output_name = node.output[0]
                    input_shape = shape_map[input_name]
                    output_shape = validate_depth_to_space_spec(input_shape, block_size, mode)
                    if output_shape != [int(value) for value in shape_map[output_name]]:
                        raise ValueError(f"DepthToSpace inferred shape drift for {op_name}")
                    prim_name = route.sanitize_name(
                        f"routeb_int8_depth_to_space_{depth_to_space_index}_{op_name}",
                        f"routeb_int8_depth_to_space_{depth_to_space_index}",
                    )
                    prim_gv = bb.add_func(
                        make_depth_to_space_primfunc(
                            stack,
                            input_shape,
                            block_size=block_size,
                            name=prim_name,
                        ),
                        prim_name,
                    )
                    out = bb.emit(
                        relax.call_tir(
                            prim_gv,
                            [tensor_map[input_name]],
                            relax_tensor_info(relax, output_shape, "uint8"),
                        )
                    )
                    tensor_map[output_name] = out
                    tensor_scales[output_name] = float(tensor_scales.get(input_name, 1.0))
                    tensor_zero_points[output_name] = int(tensor_zero_points.get(input_name, 128))
                    record = {
                        "op_name": op_name,
                        "op_type": op_type,
                        "input_name": input_name,
                        "output_name": output_name,
                        "input_shape": input_shape,
                        "output_shape": output_shape,
                        "block_size": block_size,
                        "mode": mode,
                        "primfunc": prim_name,
                        "stage": "layout_materialization",
                    }
                    op_records.append(record)
                    block_reports.append({**record, "tensorization_status": "fallback_pending"})
                    continue

                if op_type == "Concat":
                    concat_index += 1
                    attrs = route.attr_dict(node)
                    axis = int(attrs.get("axis", 1))
                    input_names = [str(value) for value in node.input]
                    output_name = node.output[0]
                    input_shapes = [shape_map[name] for name in input_names]
                    output_shape = validate_concat_shapes(input_shapes, axis)
                    if output_shape != [int(value) for value in shape_map[output_name]]:
                        raise ValueError(f"Concat inferred shape drift for {op_name}")
                    input_scales = [float(tensor_scales.get(name, 1.0)) for name in input_names]
                    input_zero_points = [int(tensor_zero_points.get(name, 128)) for name in input_names]
                    output_scale, output_zp = route.tensor_quant_params_for(
                        quant_params,
                        output_name,
                        default_scale=max(input_scales),
                        default_zero_point=128,
                    )
                    prim_name = route.sanitize_name(
                        f"routeb_int8_concat_{concat_index}_{op_name}",
                        f"routeb_int8_concat_{concat_index}",
                    )
                    prim_gv = bb.add_func(
                        make_concat_requant_primfunc(
                            stack,
                            input_shapes,
                            input_scales=input_scales,
                            input_zero_points=input_zero_points,
                            output_scale=output_scale,
                            output_zero_point=output_zp,
                            name=prim_name,
                        ),
                        prim_name,
                    )
                    out = bb.emit(
                        relax.call_tir(
                            prim_gv,
                            [tensor_map[name] for name in input_names],
                            relax_tensor_info(relax, output_shape, "uint8"),
                        )
                    )
                    tensor_map[output_name] = out
                    tensor_scales[output_name] = output_scale
                    tensor_zero_points[output_name] = output_zp
                    record = {
                        "op_name": op_name,
                        "op_type": op_type,
                        "input_names": input_names,
                        "output_name": output_name,
                        "input_shapes": input_shapes,
                        "output_shape": output_shape,
                        "input_scales": input_scales,
                        "output_scale": output_scale,
                        "primfunc": prim_name,
                        "stage": "concat_requant_materialization",
                    }
                    op_records.append(record)
                    block_reports.append({**record, "tensorization_status": "fallback_pending"})
                    continue

                if op_type == "Relu":
                    relu_index += 1
                    input_name = node.input[0]
                    output_name = node.output[0]
                    shape = shape_map[output_name]
                    out_scale, out_zp = route.tensor_quant_params_for(
                        quant_params,
                        output_name,
                        default_scale=float(tensor_scales.get(input_name, 1.0)),
                        default_zero_point=int(tensor_zero_points.get(input_name, 128)),
                    )
                    input_scale = float(tensor_scales.get(input_name, 1.0))
                    input_zp = int(tensor_zero_points.get(input_name, 128))
                    prim_name = route.sanitize_name(f"routeb_int8_relu_{relu_index}_{op_name}", f"routeb_int8_relu_{relu_index}")
                    prim_gv = bb.add_func(
                        make_relu_primfunc(
                            stack, shape, input_scale=input_scale,
                            input_zero_point=input_zp, output_scale=out_scale,
                            output_zero_point=out_zp, name=prim_name,
                        ),
                        prim_name,
                    )
                    out = bb.emit(relax.call_tir(prim_gv, [tensor_map[input_name]], relax_tensor_info(relax, shape, "uint8")))
                    tensor_map[output_name] = out
                    tensor_scales[output_name] = out_scale
                    tensor_zero_points[output_name] = out_zp
                    op_records.append(
                        {
                            "op_name": op_name,
                            "op_type": op_type,
                            "input_name": input_name,
                            "output_name": output_name,
                            "output_shape": shape,
                            "output_zero_point": out_zp,
                            "primfunc": prim_name,
                        }
                    )
                    block_reports.append(
                        {
                            "op_name": op_name,
                            "op_type": op_type,
                            "primfunc": prim_name,
                            "stage": "relu",
                            "shape": shape,
                            "static_metrics": {
                                "read_bytes_uint8": prod_int([int(v) for v in shape]),
                                "write_bytes_uint8": prod_int([int(v) for v in shape]),
                            },
                        }
                    )
                    continue

                if op_type == "Add":
                    add_index += 1
                    lhs_name, rhs_name = node.input[:2]
                    output_name = node.output[0]
                    shape = shape_map[output_name]
                    lhs_scale = float(tensor_scales.get(lhs_name, 1.0))
                    lhs_zp = int(tensor_zero_points.get(lhs_name, 128))
                    rhs_scale = float(tensor_scales.get(rhs_name, 1.0))
                    rhs_zp = int(tensor_zero_points.get(rhs_name, 128))
                    out_scale, out_zp = route.tensor_quant_params_for(
                        quant_params,
                        output_name,
                        default_scale=min(lhs_scale, rhs_scale),
                        default_zero_point=128,
                    )
                    prim_name = route.sanitize_name(f"routeb_int8_add_{add_index}_{op_name}", f"routeb_int8_add_{add_index}")
                    prim_gv = bb.add_func(make_add_primfunc(
                        stack, shape, lhs_scale=lhs_scale, lhs_zero_point=lhs_zp,
                        rhs_scale=rhs_scale, rhs_zero_point=rhs_zp,
                        output_scale=out_scale, output_zero_point=out_zp, name=prim_name,
                    ), prim_name)
                    out = bb.emit(relax.call_tir(
                        prim_gv, [tensor_map[lhs_name], tensor_map[rhs_name]],
                        relax_tensor_info(relax, shape, "uint8"),
                    ))
                    tensor_map[output_name] = out
                    tensor_scales[output_name] = out_scale
                    tensor_zero_points[output_name] = out_zp
                    op_records.append(
                        {
                            "op_name": op_name,
                            "op_type": op_type,
                            "input_names": [lhs_name, rhs_name],
                            "output_name": output_name,
                            "output_shape": shape,
                            "primfunc": prim_name,
                        }
                    )
                    block_reports.append(
                        {
                            "op_name": op_name,
                            "op_type": op_type,
                            "primfunc": prim_name,
                            "stage": "add",
                            "shape": shape,
                            "static_metrics": {
                                "lhs_read_bytes_uint8": prod_int([int(v) for v in shape]),
                                "rhs_read_bytes_uint8": prod_int([int(v) for v in shape]),
                                "write_bytes_uint8": prod_int([int(v) for v in shape]),
                            },
                        }
                    )
                    continue

                if op_type != "Conv":
                    raise RuntimeError(f"unsupported lowering for op type {op_type}: {op_name}")
                conv_index += 1
                attrs = route.attr_dict(node)
                group = int(attrs.get("group", 1) or 1)
                strides = [int(v) for v in attrs.get("strides", [1, 1])]
                pads = [int(v) for v in attrs.get("pads", [0, 0, 0, 0])]
                input_name = node.input[0]
                output_name = node.output[0]
                weight_name = route.resolve_initializer_name(node.input[1], initializer_aliases)
                weight_shape = shape_map.get(weight_name)
                if not weight_shape:
                    raise RuntimeError(f"missing weight shape for {op_name}: {weight_name}")
                input_shape = shape_map[input_name]
                output_shape = shape_map[output_name]
                input_zp = int(tensor_zero_points.get(input_name, 128))
                input_scale = float(tensor_scales.get(input_name, 1.0))
                quantized_weight, quantization = route.quantize_initializer_int8(initializers[weight_name])
                weight_scale = float(quantization.get("scale") or 1.0)
                weight_var_name = route.sanitize_name(f"weight_{conv_index}_{weight_name}", f"weight_{conv_index}")
                weight_var = relax.Var(weight_var_name, relax_tensor_info(relax, weight_shape, "int8"))
                inputs.append(weight_var)
                weight_runtime_values[weight_var_name] = quantized_weight
                weight_inputs.append(
                    {
                        "role": "weight_input",
                        "arg_name": weight_var_name,
                        "initializer_name": weight_name,
                        "shape": weight_shape,
                        "dtype": "int8",
                        "quantization": quantization,
                    }
                )

                bias_name: str | None = None
                if len(node.input) >= 3 and str(node.input[2] or ""):
                    bias_name = route.resolve_initializer_name(node.input[2], initializer_aliases)
                if bias_name and bias_name in initializers:
                    bias_shape = shape_map[bias_name]
                    quantized_bias, bias_quantization = route.quantize_bias_int32(
                        initializers[bias_name],
                        input_scale=input_scale,
                        weight_scale=weight_scale,
                    )
                else:
                    bias_shape = [int(weight_shape[0])]
                    quantized_bias = np.zeros((int(weight_shape[0]),), dtype=np.int32)
                    bias_quantization = {
                        "scheme": "zero_bias_int32",
                        "scale": float(input_scale) * float(weight_scale),
                        "zero_point": 0,
                        "source_shape": bias_shape,
                    }
                bias_var_name = route.sanitize_name(f"bias_{conv_index}_{bias_name or 'zero'}", f"bias_{conv_index}")
                bias_var = relax.Var(bias_var_name, relax_tensor_info(relax, bias_shape, "int32"))
                inputs.append(bias_var)
                weight_runtime_values[bias_var_name] = quantized_bias
                weight_inputs.append(
                    {
                        "role": "bias_input",
                        "arg_name": bias_var_name,
                        "initializer_name": str(bias_name or "synthetic_zero_bias"),
                        "shape": bias_shape,
                        "dtype": "int32",
                        "quantization": bias_quantization,
                    }
                )
                output_scale, output_zp = route.tensor_quant_params_for(
                    quant_params,
                    output_name,
                    default_scale=input_scale * weight_scale,
                    default_zero_point=128,
                )
                conv_spec = {
                    "input_shape": input_shape,
                    "output_shape": output_shape,
                    "weight_shape": weight_shape,
                    "group": group,
                    "strides": strides,
                    "pads": pads,
                    "input_zero_point": input_zp,
                    "input_scale": input_scale if scale_aware else None,
                    "weight_scale": weight_scale if scale_aware else None,
                    "output_scale": output_scale if scale_aware else None,
                    "output_zero_point": output_zp,
                }
                tensorcore_eligible = input_zp == 128
                n, _, ih, iw = [int(v) for v in input_shape]
                _, _, kh, kw = [int(v) for v in weight_shape]
                oh, ow = conv2d_out_hw(ih, iw, kh, kw, strides, pads)
                k_total = int(weight_shape[1]) * int(weight_shape[2]) * int(weight_shape[3])
                nper = int(weight_shape[0]) // int(group)
                nper_compute = 128 if nper < 128 else nper
                m_dim = n * oh * ow
                accum_shape = [group, m_dim, nper_compute]
                static_metrics = conv_static_metrics(
                    input_shape=input_shape,
                    output_shape=output_shape,
                    weight_shape=weight_shape,
                    group=group,
                    m_dim=m_dim,
                    k_total=k_total,
                    nper=nper,
                    nper_compute=nper_compute,
                )
                if tensorcore_eligible:
                    accum_prim_name = route.sanitize_name(
                        f"routeb_int8_conv_{conv_index}_{op_name}",
                        f"routeb_int8_conv_{conv_index}",
                    )
                    accum_prim_gv = bb.add_func(
                        make_int8_tc_conv_accum_primfunc(stack, conv_spec, accum_prim_name),
                        accum_prim_name,
                    )
                    accum = bb.emit(
                        relax.call_tir(
                            accum_prim_gv,
                            [tensor_map[input_name], weight_var],
                            relax_tensor_info(relax, accum_shape, "int32"),
                        )
                    )
                    pack_prim_name = route.sanitize_name(
                        f"routeb_int8_pack_{conv_index}_{op_name}",
                        f"routeb_int8_pack_{conv_index}",
                    )
                    pack_prim_gv = bb.add_func(
                        make_int8_conv_pack_primfunc(stack, conv_spec, pack_prim_name),
                        pack_prim_name,
                    )
                    out = bb.emit(
                        relax.call_tir(
                            pack_prim_gv,
                            [accum, bias_var],
                            relax_tensor_info(relax, output_shape, "uint8"),
                        )
                    )
                    prim_name_for_record = accum_prim_name
                    pack_name_for_record = pack_prim_name
                    tensorization_status = "pending"
                else:
                    fallback_prim_name = route.sanitize_name(
                        f"routeb_int8_nativefallback_{conv_index}_{op_name}",
                        f"routeb_int8_nativefallback_{conv_index}",
                    )
                    fallback_prim_gv = bb.add_func(
                        make_native_direct_conv_primfunc(stack, conv_spec, fallback_prim_name),
                        fallback_prim_name,
                    )
                    out = bb.emit(
                        relax.call_tir(
                            fallback_prim_gv,
                            [tensor_map[input_name], weight_var, bias_var],
                            relax_tensor_info(relax, output_shape, "uint8"),
                        )
                    )
                    prim_name_for_record = fallback_prim_name
                    pack_name_for_record = None
                    tensorization_status = "native_fallback_uint8_input_not_s8_tensorcore_eligible"
                tensor_map[output_name] = out
                tensor_scales[output_name] = output_scale
                tensor_zero_points[output_name] = output_zp
                conv_record = route.build_conv_op_spec(
                    name=op_name,
                    input_name=input_name,
                    output_name=output_name,
                    input_shape=input_shape,
                    output_shape=output_shape,
                    weight_shape=weight_shape,
                    group=group,
                    strides=strides,
                    pads=pads,
                ) | {
                    "input_zero_point": input_zp,
                    "input_scale": input_scale,
                    "weight_scale": weight_scale,
                    "output_scale": output_scale,
                    "output_zero_point": output_zp,
                    "bias_name": bias_name,
                    "bias_shape": bias_shape,
                    "conv_index": conv_index,
                    "primfunc": prim_name_for_record,
                    "pack_primfunc": pack_name_for_record,
                    "accum_shape": accum_shape,
                    "k_total": k_total,
                    "nper": nper,
                    "nper_compute": nper_compute,
                    "tensorcore_eligible": tensorcore_eligible,
                    "stage": "conv_accum_tensorized" if tensorcore_eligible else "conv_native_fallback",
                    "static_metrics": static_metrics,
                }
                op_records.append(conv_record)
                block_reports.append({**conv_record, "tensorization_status": tensorization_status})
                if tensorcore_eligible:
                    block_reports.append(
                        {
                            "op_name": op_name,
                            "op_type": "ConvPack",
                            "primfunc": pack_name_for_record,
                            "conv_index": conv_index,
                            "stage": "conv_pack_requant",
                            "shape": output_shape,
                            "accum_shape": accum_shape,
                            "static_metrics": {
                                "accum_read_bytes_padded": static_metrics["accum_bytes_padded"],
                                "bias_read_bytes_int32": int(weight_shape[0]) * 4,
                                "real_output_write_bytes_uint8": static_metrics["real_output_bytes_uint8"],
                                "pack_total_bytes_approx": static_metrics["accum_bytes_padded"]
                                + int(weight_shape[0]) * 4
                                + static_metrics["real_output_bytes_uint8"],
                                "accum_padded_bytes_vs_uint8_output": static_metrics[
                                    "accum_padded_bytes_vs_uint8_output"
                                ],
                            },
                        }
                    )

            outputs = []
            output_shapes: dict[str, list[int]] = {}
            for item in model.graph.output:
                outputs.append(tensor_map[item.name])
                output_shapes[item.name] = shape_map[item.name]
            if len(outputs) == 1:
                gv = bb.emit_output(outputs[0])
            else:
                gv = bb.emit_output(relax.Tuple(outputs))
        bb.emit_func_output(gv)

    input_shapes = {
        item["source_name"]: [int(dim) for dim in item["shape"]]
        for item in input_specs
        if item["role"] == "graph_input"
    }
    runtime_arg_plan = route.build_runtime_arg_plan(
        graph_input_shapes=input_shapes,
        weight_inputs=weight_inputs,
        output_shapes=output_shapes,
    )
    source_by_arg = {
        route.sanitize_name(name, "input"): name for name in input_shapes
    } | {
        route.sanitize_name(name, "output"): name for name in output_shapes
    }
    for item in runtime_arg_plan:
        if item.get("role") not in {"graph_input", "graph_output"}:
            continue
        source_name = source_by_arg.get(str(item.get("arg_name")))
        if source_name and source_name in quant_params:
            scale, zero_point = route.tensor_quant_params_for(
                quant_params, source_name, default_scale=1.0, default_zero_point=128,
            )
            item["source_name"] = source_name
            item["quantization"] = {
                "scale": scale, "zero_point": zero_point,
                "source": "stage3_static_quant_contract",
            }
    return {
        "mod": bb.finalize(),
        "inputs": inputs,
        "input_specs": input_specs,
        "weight_runtime_values": weight_runtime_values,
        "weight_inputs": weight_inputs,
        "runtime_arg_plan": runtime_arg_plan,
        "op_records": op_records,
        "block_reports": block_reports,
        "op_counts": dict(sorted(op_counts.items())),
        "input_shapes": input_shapes,
        "output_shapes": output_shapes,
        "onnx_path": str(onnx_path),
        "onnx_digest": sha256(onnx_path),
        "route_spec": "route_b_int8_auto_decomp_per_block_matmul_tensorization_v1",
        "tensor_quant_params_count": len(quant_params),
    }


def apply_per_block_schedules(tvm: Any, target: Any, mod: Any, block_reports: list[dict[str, Any]]) -> tuple[Any, list[dict[str, Any]]]:
    import tvm.s_tir.dlight as dl
    from tvm.s_tir.dlight.gpu.matmul import MatmulInt8Tensorization

    out = tvm.IRModule(dict(mod.functions), attrs=mod.attrs)
    by_prim = {str(item.get("primfunc")): item for item in block_reports}
    updated_reports: list[dict[str, Any]] = []
    for gv, func in list(out.functions_items()):
        if not hasattr(func, "script"):
            continue
        name = gv.name_hint
        pre = counts(func.script())
        report = by_prim.get(name, {"primfunc": name, "op_type": "unknown"})
        item = {**report, "pre_counts": pre}
        if str(name).startswith("routeb_int8_conv_"):
            try:
                with target, tvm.transform.PassContext(opt_level=3):
                    sched = dl.ApplyDefaultSchedule(MatmulInt8Tensorization())(tvm.IRModule({gv: func}))
                post_func = sched[gv]
                post = counts(post_func.script())
                out.update_func(gv, post_func)
                item.update(
                    {
                        "tensorization_status": (
                            "tensorized"
                            if post.get("tvm_mma_sync", 0) or post.get("mma_sync", 0) or post.get("wmma", 0)
                            else "scheduled_without_mma"
                        ),
                        "post_counts": post,
                    }
                )
            except Exception as exc:
                item.update(
                    {
                        "tensorization_status": "failed",
                        "tensorization_error": repr(exc),
                        "tensorization_traceback": traceback.format_exc(),
                    }
                )
        else:
            try:
                with target, tvm.transform.PassContext(opt_level=3):
                    sched = dl.ApplyDefaultSchedule(dl.gpu.Fallback())(tvm.IRModule({gv: func}))
                post_func = sched[gv]
                post = counts(post_func.script())
                out.update_func(gv, post_func)
                item.update({"tensorization_status": "fallback_scheduled", "post_counts": post})
            except Exception as exc:
                item.update(
                    {
                        "tensorization_status": "fallback_failed",
                        "tensorization_error": repr(exc),
                        "tensorization_traceback": traceback.format_exc(),
                    }
                )
        updated_reports.append(item)
    return out, updated_reports


def apply_fixed_budget_tuning(
    tvm: Any,
    relax: Any,
    target: Any,
    mod: Any,
    *,
    default_scheduled_mod: Any,
    max_trials: int,
    work_dir: Path,
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    """Tune the decomposed Relax/TIR module under one auditable global cap."""
    budget = make_tuning_budget_config(max_trials)
    audit: dict[str, Any] = {
        **budget,
        "work_dir": str(work_dir),
        "attempted_trials": 0,
        "valid_trials": 0,
        "database_path": None,
        "database_workload_path": None,
        "database_workload_sha256": None,
        "database_tuning_record_path": None,
        "database_tuning_record_sha256": None,
        "applied": False,
        "task_count": 0,
        "tasks": [],
    }
    if not budget["enabled"]:
        return default_scheduled_mod, audit

    tuning_dir = Path(work_dir)
    if tuning_dir.exists() and any(tuning_dir.iterdir()):
        raise RuntimeError(
            f"tuning work dir must be fresh; refusing prior or stale records: {tuning_dir}"
        )
    tuning_dir.mkdir(parents=True, exist_ok=True)

    from tvm.s_tir.meta_schedule import relax_integration as ri

    tasks = ri.extract_tasks(mod, target, params={})
    task_rows = [
        {
            "task_name": str(getattr(task, "task_name", f"task_{index}")),
            "weight": float(getattr(task, "weight", 1.0)),
        }
        for index, task in enumerate(tasks)
    ]
    audit.update({"task_count": len(task_rows), "tasks": task_rows})
    if not task_rows:
        raise RuntimeError("MetaSchedule extracted no tunable PrimFunc from automatic decomposition")

    tune_started = time.time()
    ri.tune_relax(
        mod=mod,
        params={},
        target=target,
        work_dir=str(tuning_dir),
        max_trials_global=int(budget["max_trials_global"]),
        max_trials_per_task=int(budget["max_trials_per_task"]),
        num_trials_per_iter=int(budget["num_trials_per_iter"]),
        task_scheduler="gradient",
        seed=int(seed),
    )
    audit.update(
        audit_metaschedule_database(
            tuning_dir,
            requested_trials=int(budget["requested_trials"]),
        )
    )
    audit["tuning_elapsed_s"] = round(time.time() - tune_started, 6)
    if int(audit["attempted_trials"]) == 0:
        raise RuntimeError("MetaSchedule produced no attempted trial; default schedule was not substituted")
    if int(audit["valid_trials"]) == 0:
        raise RuntimeError("MetaSchedule produced no valid trial; default schedule was not substituted")

    with target, tvm.transform.PassContext(opt_level=3):
        database_applied_mod = relax.transform.MetaScheduleApplyDatabase(
            work_dir=str(tuning_dir)
        )(mod)

    original_by_name = {gv.name_hint: func for gv, func in mod.functions_items()}
    applied_by_name = {
        gv.name_hint: func for gv, func in database_applied_mod.functions_items()
    }
    execution_mod = tvm.IRModule(
        dict(default_scheduled_mod.functions),
        attrs=default_scheduled_mod.attrs,
    )
    applied_primfuncs: list[str] = []
    for gv, default_func in list(execution_mod.functions_items()):
        name = gv.name_hint
        original_func = original_by_name.get(name)
        tuned_func = applied_by_name.get(name)
        if original_func is None or tuned_func is None:
            continue
        if tvm.ir.structural_equal(original_func, tuned_func):
            continue
        execution_mod.update_func(gv, tuned_func)
        applied_primfuncs.append(name)

    audit["applied"] = bool(applied_primfuncs)
    audit["applied_primfunc_count"] = len(applied_primfuncs)
    audit["applied_primfuncs"] = sorted(applied_primfuncs)
    audit["unmatched_tasks_kept_default_scheduled"] = True
    audit["post_tuning_counts"] = counts(execution_mod.script())
    if not applied_primfuncs:
        raise RuntimeError("MetaSchedule database was valid but no tuned schedule was applied")
    return execution_mod, audit


def make_runtime_args(
    tvm: Any,
    dev: Any,
    spec: dict[str, Any],
    rng_seed: int,
    *,
    graph_input_max: int,
) -> tuple[list[Any], dict[str, str]]:
    rng = np.random.RandomState(rng_seed)
    args = []
    sources: dict[str, str] = {}
    weight_runtime_values = dict(spec["weight_runtime_values"])
    for var in spec["inputs"]:
        name = str(var.name_hint)
        sinfo = var.struct_info
        shape = tuple(int(v) for v in sinfo.shape.values)
        dtype = str(sinfo.dtype)
        if name in weight_runtime_values:
            value = np.asarray(weight_runtime_values[name], dtype=dtype)
            sources[name] = "onnx_initializer_quantized"
        elif dtype == "int8":
            value = rng.randint(-4, 4, size=shape).astype("int8")
            sources[name] = "synthetic_random_int8"
        elif dtype == "int32":
            value = np.zeros(shape, dtype="int32")
            sources[name] = "synthetic_zero_int32"
        else:
            value = rng.randint(0, int(graph_input_max) + 1, size=shape).astype(dtype)
            sources[name] = f"synthetic_random_uint8_activation_0_{int(graph_input_max)}"
        args.append(cap.runtime_tensor(tvm, value, dev))
    return args, sources


def make_output_buffers(tvm: Any, dev: Any, output_tensors: list[Any]) -> list[Any]:
    buffers = []
    for tensor in output_tensors:
        shape = tuple(int(item) for item in tensor.shape)
        dtype = str(tensor.dtype)
        buffers.append(cap.runtime_tensor(tvm, np.zeros(shape, dtype=dtype), dev))
    return buffers


def unpack_outputs(obj: Any) -> list[Any]:
    try:
        from tvm.runtime.container import ADT

        if isinstance(obj, ADT):
            return [obj[i] for i in range(len(obj))]
    except Exception:
        pass
    if isinstance(obj, (tuple, list)):
        return list(obj)
    if not (hasattr(obj, "numpy") or hasattr(obj, "asnumpy")) and hasattr(obj, "__len__") and hasattr(obj, "__getitem__"):
        try:
            return [obj[i] for i in range(len(obj))]
        except Exception:
            pass
    return [obj]


def compare_outputs(reference: list[np.ndarray], candidate: list[np.ndarray]) -> list[dict[str, Any]]:
    rows = []
    for idx, (ref, got) in enumerate(zip(reference, candidate)):
        ref_i = np.asarray(ref).astype(np.int32)
        got_i = np.asarray(got).astype(np.int32)
        diff = got_i - ref_i
        rows.append(
            {
                "output": idx,
                "shape": [int(v) for v in ref_i.shape],
                "dtype_ref": str(ref.dtype),
                "dtype_candidate": str(got.dtype),
                "exact_equal": bool(np.array_equal(ref, got)),
                "max_abs_diff": int(np.max(np.abs(diff))) if diff.size else 0,
                "mean_abs_diff": float(np.mean(np.abs(diff))) if diff.size else 0.0,
                "mismatch_rate": float(np.mean(diff != 0)) if diff.size else 0.0,
                "ref_mean": float(np.mean(ref_i)) if diff.size else 0.0,
                "candidate_mean": float(np.mean(got_i)) if diff.size else 0.0,
            }
        )
    return rows


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "numpy"):
        return value.numpy()
    if hasattr(value, "asnumpy"):
        return value.asnumpy()
    return np.asarray(value)


def measure_vm_latency(vm: Any, args: list[Any], dev: Any, *, warmup: int, number: int, repeat: int) -> dict[str, Any]:
    for _ in range(warmup):
        vm["main"](*args)
        dev.sync()
    evaluator = vm.time_evaluator("main", dev, number=number, repeat=repeat)
    return summarize_times_ms(list(evaluator(*args).results))


def shape_to_ints(shape: Any) -> list[int]:
    out = []
    for item in shape:
        if hasattr(item, "value"):
            out.append(int(item.value))
        else:
            out.append(int(item))
    return out


def profile_stage_for_report(report: dict[str, Any]) -> str:
    stage = report.get("stage")
    if stage:
        return str(stage)
    op_type = str(report.get("op_type", "unknown"))
    primfunc = str(report.get("primfunc", ""))
    status = str(report.get("tensorization_status", ""))
    if op_type == "ConvPack" or primfunc.startswith("routeb_int8_pack_"):
        return "conv_pack_requant"
    if op_type == "Conv" and "native_fallback" in status:
        return "conv_native_fallback"
    if op_type == "Conv":
        return "conv_accum_tensorized"
    if op_type == "Relu":
        return "relu"
    if op_type == "Add":
        return "add"
    return op_type.lower() if op_type else "unknown"


def primfunc_buffer_specs(func: Any) -> list[dict[str, Any]]:
    specs = []
    buffer_map = getattr(func, "buffer_map", {})
    for index, param in enumerate(getattr(func, "params", [])):
        try:
            buf = buffer_map[param]
        except Exception:
            try:
                buf = dict(buffer_map).get(param)
            except Exception:
                buf = None
        if buf is None:
            specs.append({"index": index, "name": str(param), "kind": "scalar_or_unmapped"})
            continue
        shape = shape_to_ints(buf.shape)
        dtype = str(buf.dtype)
        specs.append(
            {
                "index": index,
                "name": str(buf.name),
                "shape": shape,
                "dtype": dtype,
                "nbytes": prod_int(shape) * dtype_nbytes(dtype),
            }
        )
    return specs


def random_profile_array(rng: np.random.RandomState, shape: list[int], dtype: str) -> np.ndarray:
    dtype = str(dtype)
    if dtype == "uint8":
        return rng.randint(0, 256, size=shape).astype(dtype)
    if dtype == "int8":
        return rng.randint(-8, 8, size=shape).astype(dtype)
    if dtype in {"int32", "uint32", "int64", "uint64"}:
        return rng.randint(-128, 128, size=shape).astype(dtype)
    if dtype == "float16":
        return rng.randn(*shape).astype(dtype)
    if dtype == "float32":
        return rng.randn(*shape).astype(dtype)
    return np.zeros(shape, dtype=dtype)


def make_primfunc_profile_args(tvm: Any, dev: Any, func: Any, seed: int) -> tuple[list[Any], list[dict[str, Any]]]:
    rng = np.random.RandomState(seed)
    args = []
    specs = primfunc_buffer_specs(func)
    for spec in specs:
        if spec.get("kind") == "scalar_or_unmapped":
            raise RuntimeError(f"cannot profile scalar/unmapped PrimFunc argument: {spec}")
        arr = random_profile_array(rng, list(spec["shape"]), str(spec["dtype"]))
        args.append(cap.runtime_tensor(tvm, arr, dev))
    return args, specs


def runtime_entry_name(rt_mod: Any, preferred: str) -> str:
    try:
        rt_mod[preferred]
        return preferred
    except Exception:
        pass
    entry = getattr(rt_mod, "entry_name", None)
    if entry:
        return str(entry)
    return preferred


def profile_one_primfunc(
    tvm: Any,
    target: Any,
    dev: Any,
    gv: Any,
    func: Any,
    report: dict[str, Any],
    *,
    warmup: int,
    number: int,
    repeat: int,
    seed: int,
) -> dict[str, Any]:
    name = str(getattr(gv, "name_hint", report.get("primfunc", "unknown")))
    record = {
        **report,
        "primfunc": name,
        "stage": profile_stage_for_report(report),
        "profile_status": "started",
    }
    try:
        build_started = time.time()
        single = tvm.IRModule({gv: func.with_attr("global_symbol", name)})
        with target, tvm.transform.PassContext(opt_level=3):
            try:
                rt_mod = tvm.build(single, target=target)
            except Exception:
                rt_mod = tvm.compile(single, target=target)
        record["profile_build_time_s"] = round(time.time() - build_started, 6)
        profile_args, arg_specs = make_primfunc_profile_args(tvm, dev, func, seed)
        record["profile_arg_specs"] = arg_specs
        entry = runtime_entry_name(rt_mod, name)
        for _ in range(warmup):
            rt_mod[entry](*profile_args)
            dev.sync()
        evaluator = rt_mod.time_evaluator(entry, dev, number=number, repeat=repeat)
        record.update(
            {
                "profile_status": "success",
                "latency": summarize_times_ms(list(evaluator(*profile_args).results)),
            }
        )
    except Exception as exc:
        record.update(
            {
                "profile_status": "failed",
                "profile_error": repr(exc),
                "profile_traceback": traceback.format_exc(),
            }
        )
    finally:
        try:
            del profile_args  # type: ignore[name-defined]
            del rt_mod  # type: ignore[name-defined]
        except Exception:
            pass
        gc.collect()
    return record


def aggregate_profile_records(records: list[dict[str, Any]], *, vm_latency_ms_p50: float | None) -> dict[str, Any]:
    by_stage: dict[str, dict[str, Any]] = {}
    successful = [item for item in records if item.get("profile_status") == "success"]
    for item in successful:
        stage = profile_stage_for_report(item)
        lat = item.get("latency") or {}
        p50 = float(lat.get("latency_ms_p50") or 0.0)
        mean = float(lat.get("latency_ms_mean") or 0.0)
        entry = by_stage.setdefault(
            stage,
            {
                "stage": stage,
                "n_blocks": 0,
                "latency_ms_p50_sum": 0.0,
                "latency_ms_mean_sum": 0.0,
                "latency_ms_p50_max": 0.0,
            },
        )
        entry["n_blocks"] += 1
        entry["latency_ms_p50_sum"] += p50
        entry["latency_ms_mean_sum"] += mean
        entry["latency_ms_p50_max"] = max(float(entry["latency_ms_p50_max"]), p50)

    total_p50 = sum(float(item["latency_ms_p50_sum"]) for item in by_stage.values())
    for item in by_stage.values():
        item["share_of_isolated_profile_p50"] = (
            float(item["latency_ms_p50_sum"]) / total_p50 if total_p50 > 0 else None
        )
        item["share_of_vm_latency_p50"] = (
            float(item["latency_ms_p50_sum"]) / float(vm_latency_ms_p50)
            if vm_latency_ms_p50 and vm_latency_ms_p50 > 0
            else None
        )

    ordered = sorted(by_stage.values(), key=lambda row: float(row["latency_ms_p50_sum"]), reverse=True)
    return {
        "n_records": len(records),
        "n_success": len(successful),
        "n_failed": len(records) - len(successful),
        "vm_latency_ms_p50": vm_latency_ms_p50,
        "isolated_profile_latency_ms_p50_sum": total_p50,
        "isolated_profile_vs_vm_ratio": (
            total_p50 / float(vm_latency_ms_p50) if vm_latency_ms_p50 and vm_latency_ms_p50 > 0 else None
        ),
        "stage_rows": ordered,
    }


def flatten_for_csv(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, dict):
            for sub_key, sub_value in flatten_for_csv(value).items():
                out[f"{key}.{sub_key}"] = sub_value
        elif isinstance(value, (list, tuple)):
            out[key] = json.dumps(value, ensure_ascii=False)
        else:
            out[key] = value
    return out


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = [flatten_for_csv(row) for row in rows]
    fieldnames = sorted({key for row in flat for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in flat:
            writer.writerow(row)


def profile_scheduled_primfuncs(
    tvm: Any,
    target: Any,
    dev: Any,
    scheduled_mod: Any,
    block_reports: list[dict[str, Any]],
    *,
    warmup: int,
    number: int,
    repeat: int,
    seed: int,
    limit: int,
) -> list[dict[str, Any]]:
    by_prim = {str(item.get("primfunc")): item for item in block_reports if item.get("primfunc")}
    records: list[dict[str, Any]] = []
    for gv, func in scheduled_mod.functions_items():
        if limit and len(records) >= limit:
            break
        if not hasattr(func, "buffer_map") or not hasattr(func, "params"):
            continue
        name = str(gv.name_hint)
        if name == "main":
            continue
        report = by_prim.get(name, {"primfunc": name, "op_type": "unknown", "stage": "unknown"})
        records.append(
            profile_one_primfunc(
                tvm,
                target,
                dev,
                gv,
                func,
                report,
                warmup=warmup,
                number=number,
                repeat=repeat,
                seed=seed + len(records),
            )
        )
    return records


def schedule_fp16_conv_primfunc(tvm: Any, target: Any, prim_name: str, primfunc: Any) -> tuple[Any, dict[str, int], str, str, str | None]:
    import tvm.s_tir.dlight as dl
    from tvm.s_tir.dlight.gpu.matmul import MatmulTensorization

    mod = tvm.IRModule({prim_name: primfunc})
    gv = next(iter(mod.functions.keys()))
    tensorize_error = None
    try:
        with target, tvm.transform.PassContext(opt_level=3):
            scheduled = dl.ApplyDefaultSchedule(MatmulTensorization())(mod)
        schedule_status = "matmul_tensorization_schedule"
    except Exception as exc:
        tensorize_error = repr(exc)
        with target, tvm.transform.PassContext(opt_level=3):
            scheduled = dl.ApplyDefaultSchedule(dl.gpu.Fallback())(mod)
        schedule_status = "fallback_schedule_after_matmul_tensorization_failed"
    func = scheduled[gv]
    return scheduled, counts(func.script()), str(gv.name_hint), schedule_status, tensorize_error


def profile_fp16_conv_blocks(
    tvm: Any,
    target: Any,
    dev: Any,
    stack: dict[str, Any],
    op_records: list[dict[str, Any]],
    *,
    warmup: int,
    number: int,
    repeat: int,
    seed: int,
    limit: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    convs = [item for item in op_records if item.get("op_type") == "Conv"]
    for item in convs:
        if limit and len(records) >= limit:
            break
        conv_index = int(item.get("conv_index") or (len(records) + 1))
        prim_name = route.sanitize_name(f"fp16_profile_conv_{conv_index}_{item.get('name') or item.get('op_name')}", f"fp16_profile_conv_{conv_index}")
        spec = {
            "input_shape": item["input_shape"],
            "output_shape": item["output_shape"],
            "weight_shape": item["weight_shape"],
            "group": int(item.get("group") or item.get("groups") or 1),
            "strides": item["strides"],
            "pads": item["pads"],
        }
        report = {
            "op_name": item.get("name") or item.get("op_name"),
            "op_type": "Conv",
            "conv_index": conv_index,
            "stage": "fp16_conv_direct",
            "primfunc": prim_name,
            "input_shape": spec["input_shape"],
            "output_shape": spec["output_shape"],
            "weight_shape": spec["weight_shape"],
            "group": spec["group"],
            "static_metrics": item.get("static_metrics"),
        }
        try:
            prim = make_fp16_conv_direct_primfunc(stack, spec, prim_name)
            scheduled, post_counts, scheduled_name, schedule_status, tensorize_error = schedule_fp16_conv_primfunc(
                tvm,
                target,
                prim_name,
                prim,
            )
            gv, func = next(iter(scheduled.functions_items()))
            report.update(
                {
                    "primfunc": scheduled_name,
                    "fp16_schedule_status": schedule_status,
                    "fp16_tensorize_error": tensorize_error,
                    "tensorization_status": "tensorized"
                    if post_counts.get("tvm_mma_sync") or post_counts.get("mma_sync") or post_counts.get("wmma")
                    else "scheduled_without_mma",
                    "post_counts": post_counts,
                }
            )
            records.append(
                profile_one_primfunc(
                    tvm,
                    target,
                    dev,
                    gv,
                    func,
                    report,
                    warmup=warmup,
                    number=number,
                    repeat=repeat,
                    seed=seed + len(records),
                )
            )
        except Exception as exc:
            records.append(
                {
                    **report,
                    "profile_status": "failed",
                    "profile_error": repr(exc),
                    "profile_traceback": traceback.format_exc(),
                }
            )
    return records


class VmCallable:
    def __init__(self, vm: Any) -> None:
        self.vm = vm

    def __call__(self, *args: Any) -> Any:
        return self.vm["main"](*args)


def run(args: argparse.Namespace) -> dict[str, Any]:
    cap.configure_tvm_env(str(args.gpu))
    import tvm
    from tvm import relax

    out_dir = Path(args.out_dir)
    label_dir = out_dir / args.label
    label_dir.mkdir(parents=True, exist_ok=True)
    if args.wait_idle:
        cap.wait_gpu_idle(str(args.gpu), label_dir / "gpu_idle_gate", timeout_s=args.idle_timeout_s)
    stack = cap.import_tvm_stack()
    dev = stack["dev"]
    target = stack["target"]
    started = time.time()
    result: dict[str, Any] = {
        "schema": "route_b_int8_auto_decomp_result_v1",
        "status": "started",
        "created_at": cap.utc_now(),
        "host": socket.gethostname(),
        "label": args.label,
        "width": [int(v) for v in args.width.split(",")],
        "gpu": str(args.gpu),
        "onnx_path": str(args.onnx),
        "out_dir": str(label_dir),
        "method": (
            "automatic decomposition + fixed-budget whole-module MetaSchedule"
            if int(getattr(args, "max_trials", 0)) > 0
            else "automatic decomposition + per-PrimFunc default MatmulInt8Tensorization"
        ),
        "not_used": [
            "QDQ/default lowering",
            "hand-written full-engine im2col+MMA as final route",
            "historical hand-rewrite prior",
        ],
        "tuning": {
            **make_tuning_budget_config(int(getattr(args, "max_trials", 0))),
            "applied": False,
            "attempted_trials": 0,
            "valid_trials": 0,
        },
    }
    try:
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        quant_params_path = Path(args.tensor_quant_params_json).resolve() if args.tensor_quant_params_json else None
        quant_params = route.load_tensor_quant_params(quant_params_path)
        with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
            auto_spec = build_auto_decomp_relax_module(
                stack, label=args.label, onnx_path=Path(args.onnx),
                tensor_quant_params=quant_params, graph_input_quant_params=quant_params,
            )
        write_text(label_dir / "build_auto_spec_stdout.txt", stdout_buf.getvalue() or "no stdout\n")
        write_text(label_dir / "build_auto_spec_stderr.txt", stderr_buf.getvalue() or "no stderr\n")
        write_text(label_dir / "auto_decomp_pre_schedule.py", auto_spec["mod"].script())
        scheduled_mod, block_reports = apply_per_block_schedules(tvm, target, auto_spec["mod"], auto_spec["block_reports"])
        write_text(label_dir / "auto_decomp_scheduled.py", scheduled_mod.script())
        write_json(label_dir / "block_reports.json", block_reports)
        tensorized = [item for item in block_reports if item.get("tensorization_status") == "tensorized"]
        failed = [item for item in block_reports if str(item.get("tensorization_status", "")).endswith("failed")]
        result.update(
            {
                "op_counts": auto_spec["op_counts"],
                "n_blocks": len(block_reports),
                "n_conv_blocks": sum(1 for item in block_reports if item.get("op_type") == "Conv"),
                "n_tensorized_conv_blocks": len(tensorized),
                "n_schedule_failures": len(failed),
                "block_reports_path": str(label_dir / "block_reports.json"),
                "pre_schedule_path": str(label_dir / "auto_decomp_pre_schedule.py"),
                "scheduled_path": str(label_dir / "auto_decomp_scheduled.py"),
                "scheduled_counts": counts(scheduled_mod.script()),
                "route_spec": auto_spec["route_spec"],
                "runtime_arg_plan": auto_spec["runtime_arg_plan"],
                "tensor_quant_params_path": str(quant_params_path) if quant_params_path else None,
                "tensor_quant_params_sha256": sha256(quant_params_path) if quant_params_path else None,
                "tensor_quant_params_count": len(quant_params),
            }
        )
        if failed and not args.allow_schedule_failures:
            result.update({"status": "failed_schedule", "first_blocker": failed[0]})
            return result

        tuning_work_dir = (
            Path(args.tuning_work_dir).resolve()
            if getattr(args, "tuning_work_dir", None)
            else label_dir / "metaschedule"
        )
        execution_mod, tuning_audit = apply_fixed_budget_tuning(
            tvm,
            relax,
            target,
            auto_spec["mod"] if int(getattr(args, "max_trials", 0)) > 0 else scheduled_mod,
            default_scheduled_mod=scheduled_mod,
            max_trials=int(getattr(args, "max_trials", 0)),
            work_dir=tuning_work_dir,
            seed=int(args.rng_seed),
        )
        result["tuning"] = tuning_audit
        execution_path = (
            label_dir / "auto_decomp_metaschedule_tuned.py"
            if tuning_audit["applied"]
            else label_dir / "auto_decomp_scheduled.py"
        )
        write_text(execution_path, execution_mod.script())
        result.update(
            {
                "execution_module_path": str(execution_path),
                "execution_module_counts": counts(execution_mod.script()),
            }
        )

        build_start = time.time()
        with target, tvm.transform.PassContext(opt_level=3):
            ex = tvm.compile(execution_mod, target=target)
        build_s = time.time() - build_start
        artifact_path = label_dir / "route_b_int8_auto_decomp.vmexec"
        ex.export_library(str(artifact_path))
        runtime_weights_audit = persist_runtime_weights(
            label_dir / "runtime_weights_int8.npz",
            runtime_arg_plan=auto_spec["runtime_arg_plan"],
            weight_runtime_values=auto_spec["weight_runtime_values"],
        )
        result["runtime_weights"] = runtime_weights_audit
        vm = relax.VirtualMachine(ex, dev)
        runtime_args, runtime_sources = make_runtime_args(
            tvm,
            dev,
            auto_spec,
            args.rng_seed,
            graph_input_max=args.graph_input_max,
        )

        direct_dir = label_dir / "native_direct_reference"
        extended_ops = sorted(set(auto_spec["op_counts"]) - set(route.REQUIRED_FULL_ONNX_OPS))
        if extended_ops:
            direct_dir.mkdir(parents=True, exist_ok=True)
            direct_build_start = time.time()
            # Compare the tuned module against the same automatically
            # decomposed graph under its deterministic per-block schedule.
            # Compiling the raw unscheduled full graph is not a stronger
            # numerical oracle and can spend close to an hour in codegen for
            # F-Cooper.  The scheduled reference preserves the exact graph and
            # quantization contract while isolating MetaSchedule's effect.
            with target, tvm.transform.PassContext(opt_level=3):
                direct_ex = tvm.compile(scheduled_mod, target=target)
            direct_artifact = direct_dir / "route_b_int8_auto_default_scheduled_ref.so"
            direct_ex.export_library(str(direct_artifact))
            direct_vm = relax.VirtualMachine(direct_ex, dev)
            direct_obj = direct_vm["main"](*runtime_args)
            dev.sync()
            ref_outputs = [to_numpy(item) for item in unpack_outputs(direct_obj)]
            direct_latency = measure_vm_latency(
                direct_vm,
                runtime_args,
                dev,
                warmup=max(1, min(args.warmup, 3)),
                number=max(1, min(args.number, 5)),
                repeat=max(1, min(args.repeat, 2)),
            )
            direct_build = {
                "reference_kind": "same_quantized_graph_default_automatic_schedule",
                "extended_ops": extended_ops,
                "artifact_path": str(direct_artifact),
                "artifact_digest": sha256(direct_artifact),
                "build_time_s": time.time() - direct_build_start,
                "latency_stats": direct_latency,
                "lowered_text_path": None,
            }
        else:
            # Preserve the reviewed native-direct comparator for legacy graphs.
            direct_spec = route.build_full_onnx_te_spec(
                stack, label=args.label, onnx_path=Path(args.onnx),
                tensor_quant_params=quant_params, graph_input_quant_params=quant_params,
            )
            direct_build = route.build_full_onnx_module(
                stack,
                direct_spec,
                direct_dir,
                f"{args.label}_native_direct_reference",
                number=max(1, min(args.number, 5)),
                repeat=max(1, min(args.repeat, 2)),
            )
            direct_tvm_args = [
                *runtime_args,
                *make_output_buffers(tvm, dev, list(direct_spec["outputs"])),
            ]
            direct_build["module"](*direct_tvm_args)
            dev.sync()
            n_outputs = len(auto_spec["output_shapes"])
            ref_outputs = [
                to_numpy(direct_tvm_args[-n_outputs + idx])
                for idx in range(n_outputs)
            ]
        cand_obj = vm["main"](*runtime_args)
        dev.sync()
        cand_outputs = [to_numpy(item) for item in unpack_outputs(cand_obj)]
        correctness = compare_outputs(ref_outputs, cand_outputs)
        correctness_all_exact = bool(
            correctness and all(item["exact_equal"] for item in correctness)
        )
        if int(getattr(args, "max_trials", 0)) > 0 and not correctness_all_exact:
            result.update(
                {
                    "status": "failed_tuned_correctness",
                    "build_success": True,
                    "numerical_success": False,
                    "build_time_s": build_s,
                    "artifact_path": str(artifact_path),
                    "artifact_digest": sha256(artifact_path),
                    "artifact_sha256": sha256(artifact_path),
                    "runtime_sources": runtime_sources,
                    "graph_input_max": int(args.graph_input_max),
                    "input_shapes": auto_spec["input_shapes"],
                    "output_shapes": auto_spec["output_shapes"],
                    "correctness_vs_native_direct": correctness,
                    "correctness_all_exact": False,
                    "first_blocker": correctness[0] if correctness else {
                        "reason": "reference comparison produced no outputs"
                    },
                }
            )
            return result

        latency = measure_vm_latency(vm, runtime_args, dev, warmup=args.warmup, number=args.number, repeat=args.repeat)
        energy = None
        if args.measure_energy:
            energy = cap.measure_energy(label_dir, VmCallable(vm), dev, runtime_args, gpu=str(args.gpu), measure_iters=args.energy_iters)

        profiling = None
        if args.profile_blocks:
            profile_dir = label_dir / "rootcause_profile"
            profile_dir.mkdir(parents=True, exist_ok=True)
            static_rows = [item for item in block_reports if item.get("static_metrics")]
            write_json(profile_dir / "static_block_metrics.json", static_rows)
            write_csv_rows(profile_dir / "static_block_metrics.csv", static_rows)
            int8_profile_records = profile_scheduled_primfuncs(
                tvm,
                target,
                dev,
                execution_mod,
                block_reports,
                warmup=args.profile_warmup,
                number=args.profile_number,
                repeat=args.profile_repeat,
                seed=args.rng_seed + 1000,
                limit=args.profile_limit,
            )
            int8_summary = aggregate_profile_records(
                int8_profile_records,
                vm_latency_ms_p50=latency.get("latency_ms_p50"),
            )
            write_json(profile_dir / "int8_block_profile.json", int8_profile_records)
            write_csv_rows(profile_dir / "int8_block_profile.csv", int8_profile_records)
            write_json(profile_dir / "int8_stage_summary.json", int8_summary)
            write_csv_rows(profile_dir / "int8_stage_summary.csv", int8_summary.get("stage_rows", []))

            fp16_profile_records: list[dict[str, Any]] = []
            fp16_summary = None
            if args.profile_fp16_blocks:
                fp16_profile_records = profile_fp16_conv_blocks(
                    tvm,
                    target,
                    dev,
                    stack,
                    auto_spec["op_records"],
                    warmup=args.profile_warmup,
                    number=args.profile_number,
                    repeat=args.profile_repeat,
                    seed=args.rng_seed + 2000,
                    limit=args.profile_fp16_limit,
                )
                fp16_summary = aggregate_profile_records(
                    fp16_profile_records,
                    vm_latency_ms_p50=args.route_b_fp16_ms,
                )
                write_json(profile_dir / "fp16_conv_block_profile.json", fp16_profile_records)
                write_csv_rows(profile_dir / "fp16_conv_block_profile.csv", fp16_profile_records)
                write_json(profile_dir / "fp16_conv_stage_summary.json", fp16_summary)
                write_csv_rows(profile_dir / "fp16_conv_stage_summary.csv", fp16_summary.get("stage_rows", []))
            profiling = {
                "profile_dir": str(profile_dir),
                "profile_warmup": int(args.profile_warmup),
                "profile_number": int(args.profile_number),
                "profile_repeat": int(args.profile_repeat),
                "int8_block_profile_path": str(profile_dir / "int8_block_profile.json"),
                "int8_stage_summary_path": str(profile_dir / "int8_stage_summary.json"),
                "int8_stage_summary": int8_summary,
                "static_block_metrics_path": str(profile_dir / "static_block_metrics.json"),
                "fp16_conv_block_profile_path": str(profile_dir / "fp16_conv_block_profile.json")
                if args.profile_fp16_blocks
                else None,
                "fp16_conv_stage_summary_path": str(profile_dir / "fp16_conv_stage_summary.json")
                if args.profile_fp16_blocks
                else None,
                "fp16_conv_stage_summary": fp16_summary,
            }

        comparator = {
            "native_direct_int8_ms": args.native_direct_int8_ms,
            "route_b_fp16_ms": args.route_b_fp16_ms,
            "default_fp16_no_ms_ms": args.default_fp16_no_ms_ms,
        }
        lat_p50 = latency.get("latency_ms_p50")
        speedup = {}
        if lat_p50:
            for key, value in comparator.items():
                if value and float(value) > 0:
                    speedup[f"speedup_vs_{key}"] = float(value) / float(lat_p50)
        result.update(
            {
                "status": "success",
                "build_success": True,
                "build_time_s": build_s,
                "artifact_path": str(artifact_path),
                "artifact_digest": sha256(artifact_path),
                "artifact_sha256": sha256(artifact_path),
                "runtime_sources": runtime_sources,
                "graph_input_max": int(args.graph_input_max),
                "input_shapes": auto_spec["input_shapes"],
                "output_shapes": auto_spec["output_shapes"],
                "correctness_vs_native_direct": correctness,
                "correctness_all_exact": correctness_all_exact,
                "latency": latency,
                "energy": energy,
                "profiling": profiling,
                "comparators": comparator,
                "speedup": speedup,
                "native_direct_reference": {
                    "artifact_path": direct_build.get("artifact_path"),
                    "artifact_digest": direct_build.get("artifact_digest"),
                    "latency_stats": direct_build.get("latency_stats"),
                    "lowered_text_path": direct_build.get("lowered_text_path"),
                },
            }
        )
    except Exception as exc:
        result.update(
            {
                "status": "failed",
                "build_success": False,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        result["elapsed_s"] = round(time.time() - started, 6)
        write_json(label_dir / "route_b_int8_auto_decomp_result.json", result)
        write_json(out_dir / "route_b_int8_auto_decomp_latest.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="smbo_64x128x256")
    parser.add_argument("--width", default="64,128,256")
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--gpu", default="4")
    parser.add_argument("--wait-idle", action="store_true")
    parser.add_argument("--idle-timeout-s", type=int, default=900)
    parser.add_argument("--rng-seed", type=int, default=20260707)
    parser.add_argument("--graph-input-max", type=int, default=255)
    parser.add_argument("--tensor-quant-params-json", type=Path)
    parser.add_argument(
        "--max-trials",
        type=int,
        default=0,
        help="Strict global MetaSchedule trial cap; 0 preserves default tensorization.",
    )
    parser.add_argument(
        "--tuning-work-dir",
        type=Path,
        help="Fresh MetaSchedule database directory (default: LABEL_DIR/metaschedule).",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--number", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--measure-energy", action="store_true")
    parser.add_argument("--energy-iters", type=int, default=300)
    parser.add_argument("--profile-blocks", action="store_true")
    parser.add_argument("--profile-fp16-blocks", action="store_true")
    parser.add_argument("--profile-warmup", type=int, default=3)
    parser.add_argument("--profile-number", type=int, default=20)
    parser.add_argument("--profile-repeat", type=int, default=3)
    parser.add_argument("--profile-limit", type=int, default=0)
    parser.add_argument("--profile-fp16-limit", type=int, default=0)
    parser.add_argument("--allow-schedule-failures", action="store_true")
    parser.add_argument("--native-direct-int8-ms", type=float, default=11.099446)
    parser.add_argument("--route-b-fp16-ms", type=float, default=3.5576)
    parser.add_argument("--default-fp16-no-ms-ms", type=float, default=53.248542)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.profile_fp16_blocks:
        args.profile_blocks = True
    result = run(args)
    print(json.dumps({k: v for k, v in result.items() if k != "traceback"}, ensure_ascii=False, indent=2, sort_keys=True))
    if result.get("status") != "success":
        print(result.get("traceback", ""), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
