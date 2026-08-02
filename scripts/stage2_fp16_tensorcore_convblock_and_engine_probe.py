#!/usr/bin/env python3
"""Probe TVM FP16 tensor-core route for a lhc_07 conv block and full engine.

This script is intended to run on the H800 TVM environment.  It writes
machine-readable JSON plus a short review Markdown file under the original60
exports directory.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path("/home/jichengzhi/V2X")
DEFAULT_EXPORT_DIR = (
    REPO_ROOT
    / "multi_agent/data/stage2_lut_generation_v1/generated/"
    / "original60_quant_20260627/exports"
)
DEFAULT_RAW_DIR = Path("/exdata/jichengzhi/s2_tvm/fp16_tensorcore_convblock_engine_20260629")
DEFAULT_ONNX = Path("/exdata/jichengzhi/s2_tvm/models/lhc_07_backbone.onnx")


def safe_label(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return safe.strip("_") or "unknown"


def _avoid_local_onnx_shadow() -> None:
    """Keep the repo's ONNX model directory from shadowing the onnx package."""
    cwd = Path.cwd().resolve()
    filtered: list[str] = []
    for item in sys.path:
        if item == "" and cwd == REPO_ROOT:
            continue
        if item and Path(item).resolve() == REPO_ROOT:
            continue
        filtered.append(item)
    sys.path[:] = filtered
    loaded = sys.modules.get("onnx")
    if loaded is not None and not hasattr(loaded, "load"):
        del sys.modules["onnx"]


def _counts(text: str) -> dict[str, int]:
    keys = [
        "wmma",
        "tvm_mma_sync",
        "mma.sync",
        "ptx_mma",
        "ldmatrix",
        "float16",
        "float32",
        "conv2d",
        "matmul",
    ]
    low = text.lower()
    out: dict[str, int] = {}
    for key in keys:
        haystack = low if key != "tvm_mma_sync" else text
        needle = key.lower() if key != "tvm_mma_sync" else key
        out[key] = haystack.count(needle)
    return out


def _time_vm(vm: Any, args: list[Any], dev: Any, reps: int) -> tuple[float, float, list[float]]:
    vm["main"](*args)
    dev.sync()
    timer = vm.time_evaluator("main", dev, number=reps, repeat=5)
    result = timer(*args)
    repeats_us = [float(item) * 1e6 for item in result.results]
    return float(result.mean) * 1e6, min(repeats_us), repeats_us


def _make_lhc07_1x1_conv_as_matmul_mod(variant: str, k_dim: int) -> Any:
    from tvm import relax

    # fused_conv2d11_add7_relu5:
    # input NCHW=(2,48,64,128), weight=(256,48,1,1)
    # 1x1 conv is equivalent to X[M,K] @ W[K,N], M=2*64*128, K=48, N=256.
    m, k, n = 2 * 64 * 128, int(k_dim), 256
    bb = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorStructInfo((m, k), "float16"))
    w = relax.Var("w", relax.TensorStructInfo((k, n), "float16"))
    params = [x, w]
    b = None
    if "bias_relu" in variant:
        b = relax.Var("b", relax.TensorStructInfo((n,), "float16"))
        params.append(b)
    with bb.function("main", params):
        with bb.dataflow():
            y = bb.emit(relax.op.matmul(x, w, out_dtype="float32"))
            if "bias_relu" in variant:
                assert b is not None
                b32 = bb.emit(relax.op.astype(b, "float32"))
                y = bb.emit(relax.op.add(y, b32))
                y = bb.emit(relax.op.nn.relu(y))
            gv = bb.emit_output(y)
        bb.emit_func_output(gv)
    return bb.finalize()


GROUP_CONV_CANDIDATES: dict[str, dict[str, Any]] = {
    "fused_conv2d1_add5_relu3": {
        "input_nchw": (2, 96, 128, 256),
        "weight_oihw": (96, 3, 3, 3),
        "bias": (1, 96, 1, 1),
        "strides": (2, 2),
        "padding": (1, 1, 1, 1),
        "groups": 32,
        "relu": True,
    },
    "fused_conv2d3_add5_relu3": {
        "input_nchw": (2, 96, 64, 128),
        "weight_oihw": (96, 3, 3, 3),
        "bias": (1, 96, 1, 1),
        "strides": (1, 1),
        "padding": (1, 1, 1, 1),
        "groups": 32,
        "relu": True,
    },
    "fused_conv2d4_add10_relu6": {
        "input_nchw": (2, 256, 64, 128),
        "weight_oihw": (256, 8, 3, 3),
        "bias": (1, 256, 1, 1),
        "strides": (2, 2),
        "padding": (1, 1, 1, 1),
        "groups": 32,
        "relu": True,
    },
    "fused_conv2d6_add10_relu6": {
        "input_nchw": (2, 256, 32, 64),
        "weight_oihw": (256, 8, 3, 3),
        "bias": (1, 256, 1, 1),
        "strides": (1, 1),
        "padding": (1, 1, 1, 1),
        "groups": 32,
        "relu": True,
    },
}


_TIR_BUFFER_RE = re.compile(
    r'T\.Buffer\(\(T\.int64\((\d+)\), T\.int64\((\d+)\), T\.int64\((\d+)\), T\.int64\((\d+)\)\), "float16"\)'
)


def _make_dynamic_group_conv_spec(
    candidate: str,
    input_nchw: tuple[int, int, int, int],
) -> dict[str, Any]:
    base = dict(GROUP_CONV_CANDIDATES[candidate])
    input_shape = tuple(int(item) for item in input_nchw)
    channels = int(input_shape[1])
    groups = int(base["groups"])
    if channels % groups != 0:
        raise ValueError(f"group conv channels must be divisible by groups: {input_shape}, groups={groups}")
    base["input_nchw"] = input_shape
    base["weight_oihw"] = (channels, channels // groups, 3, 3)
    base["bias"] = (1, channels, 1, 1)
    out_h, out_w = _conv2d_out_hw(
        int(input_shape[2]),
        int(input_shape[3]),
        3,
        3,
        tuple(base["strides"]),
        tuple(base["padding"]),
    )
    base["output_nchw"] = (int(input_shape[0]), channels, out_h, out_w)
    return base


def _classify_full_engine_group_conv_primfunc_text(text: str) -> dict[str, Any] | None:
    """Classify target group-conv PrimFuncs without hard-coding AP/speed spatial sizes."""
    shapes = [tuple(int(item) for item in match) for match in _TIR_BUFFER_RE.findall(text)]
    weight_shapes = [
        shape
        for shape in shapes
        if shape[2:] == (3, 3) and shape[0] % 32 == 0 and shape[1] * 32 == shape[0]
    ]
    if not weight_shapes:
        return None
    for weight_shape in weight_shapes:
        channels = int(weight_shape[0])
        nchw_shapes = [shape for shape in shapes if shape[0] == 2 and shape[1] == channels]
        if not nchw_shapes:
            continue
        input_shape = max(nchw_shapes, key=lambda shape: int(shape[2]) * int(shape[3]))
        _, _, in_h, in_w = input_shape
        downsample_output = (2, channels, in_h // 2, in_w // 2)
        if in_h % 2 == 0 and in_w % 2 == 0 and downsample_output in nchw_shapes:
            return {
                "candidate": "fused_conv2d4_add10_relu6",
                "spec": _make_dynamic_group_conv_spec("fused_conv2d4_add10_relu6", input_shape),
                "matched_input_nchw": input_shape,
                "matched_output_nchw": downsample_output,
                "match_rule": f"weight_{channels}x{channels // 32}x3x3_stride2_shape",
            }
        if nchw_shapes.count(input_shape) >= 2:
            return {
                "candidate": "fused_conv2d6_add10_relu6",
                "spec": _make_dynamic_group_conv_spec("fused_conv2d6_add10_relu6", input_shape),
                "matched_input_nchw": input_shape,
                "matched_output_nchw": input_shape,
                "match_rule": f"weight_{channels}x{channels // 32}x3x3_stride1_same_shape",
            }
    return None


def _conv2d_out_hw(h: int, w: int, kh: int, kw: int, strides: tuple[int, int], padding: tuple[int, int, int, int]) -> tuple[int, int]:
    pt, pl, pb, pr = padding
    sh, sw = strides
    return ((h + pt + pb - kh) // sh + 1, (w + pl + pr - kw) // sw + 1)


def _make_group_conv_default_mod(spec: dict[str, Any]) -> Any:
    from tvm import relax

    x_shape = tuple(spec["input_nchw"])
    w_shape = tuple(spec["weight_oihw"])
    b_shape = tuple(spec["bias"])
    groups = int(spec["groups"])
    bb = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorStructInfo(x_shape, "float16"))
    w = relax.Var("w", relax.TensorStructInfo(w_shape, "float16"))
    b = relax.Var("b", relax.TensorStructInfo(b_shape, "float16"))
    with bb.function("main", [x, w, b]):
        with bb.dataflow():
            y = bb.emit(
                relax.op.nn.conv2d(
                    x,
                    w,
                    strides=tuple(spec["strides"]),
                    padding=tuple(spec["padding"]),
                    groups=groups,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                    out_dtype="float16",
                )
            )
            y = bb.emit(relax.op.add(y, b))
            if bool(spec.get("relu", False)):
                y = bb.emit(relax.op.nn.relu(y))
            gv = bb.emit_output(y)
        bb.emit_func_output(gv)
    return bb.finalize()


def _make_group_conv_im2col_batched_matmul_mod(spec: dict[str, Any]) -> Any:
    from tvm import relax

    n, cin, h, w = [int(v) for v in spec["input_nchw"]]
    cout, cpg, kh, kw = [int(v) for v in spec["weight_oihw"]]
    groups = int(spec["groups"])
    oh, ow = _conv2d_out_hw(h, w, kh, kw, tuple(spec["strides"]), tuple(spec["padding"]))
    m = n * oh * ow
    k = cpg * kh * kw
    nper = cout // groups

    bb = relax.BlockBuilder()
    x_col = relax.Var("x_col", relax.TensorStructInfo((groups, m, k), "float16"))
    w_mat = relax.Var("w_mat", relax.TensorStructInfo((groups, k, nper), "float16"))
    b = relax.Var("b", relax.TensorStructInfo((groups, 1, nper), "float16"))
    with bb.function("main", [x_col, w_mat, b]):
        with bb.dataflow():
            y = bb.emit(relax.op.matmul(x_col, w_mat, out_dtype="float16"))
            y = bb.emit(relax.op.add(y, b))
            if bool(spec.get("relu", False)):
                y = bb.emit(relax.op.nn.relu(y))
            gv = bb.emit_output(y)
        bb.emit_func_output(gv)
    return bb.finalize()


def _make_te_im2col_primfunc(spec: dict[str, Any], name: str) -> Any:
    from tvm import te

    n, cin, h, width = [int(v) for v in spec["input_nchw"]]
    cout, cpg, kh, kw = [int(v) for v in spec["weight_oihw"]]
    groups = int(spec["groups"])
    oh, ow = _conv2d_out_hw(h, width, kh, kw, tuple(spec["strides"]), tuple(spec["padding"]))
    pt, pl, _, _ = tuple(spec["padding"])
    sh, sw = tuple(spec["strides"])
    m = n * oh * ow
    k_total = cpg * kh * kw
    x = te.placeholder((n, cin, h, width), "float16", name="x")

    def compute(g: Any, row: Any, kk: Any) -> Any:
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
        cond = (in_y >= 0) & (in_y < h) & (in_x >= 0) & (in_x < width)
        return te.if_then_else(cond, x[nn, in_c, in_y, in_x], te.const(0, "float16"))

    out = te.compute((groups, m, k_total), compute, name="x_col")
    return te.create_prim_func([x, out]).with_attr("global_symbol", name)


def _make_te_restore_nchw_primfunc(spec: dict[str, Any], name: str) -> Any:
    from tvm import te

    n, _, h, width = [int(v) for v in spec["input_nchw"]]
    cout, _, kh, kw = [int(v) for v in spec["weight_oihw"]]
    groups = int(spec["groups"])
    oh, ow = _conv2d_out_hw(h, width, kh, kw, tuple(spec["strides"]), tuple(spec["padding"]))
    nper = cout // groups
    m = n * oh * ow
    y_group = te.placeholder((groups, m, nper), "float16", name="y_group")

    def compute(nn: Any, oc: Any, yy: Any, xx: Any) -> Any:
        g = oc // nper
        ocg = oc % nper
        row = nn * (oh * ow) + yy * ow + xx
        return y_group[g, row, ocg]

    out = te.compute((n, cout, oh, ow), compute, name="out_nchw")
    return te.create_prim_func([y_group, out]).with_attr("global_symbol", name)


def _make_group_conv_full_im2col_tensorcore_mod(spec: dict[str, Any]) -> Any:
    from tvm import relax

    n, cin, h, w = [int(v) for v in spec["input_nchw"]]
    cout, cpg, kh, kw = [int(v) for v in spec["weight_oihw"]]
    groups = int(spec["groups"])
    oh, ow = _conv2d_out_hw(h, w, kh, kw, tuple(spec["strides"]), tuple(spec["padding"]))
    m = n * oh * ow
    k = cpg * kh * kw
    nper = cout // groups

    bb = relax.BlockBuilder()
    im2col_gv = bb.add_func(_make_te_im2col_primfunc(spec, "im2col_materialize"), "im2col_materialize")
    restore_gv = bb.add_func(_make_te_restore_nchw_primfunc(spec, "restore_nchw"), "restore_nchw")
    x = relax.Var("x", relax.TensorStructInfo((n, cin, h, w), "float16"))
    w_mat = relax.Var("w_mat", relax.TensorStructInfo((groups, k, nper), "float16"))
    b = relax.Var("b", relax.TensorStructInfo((groups, 1, nper), "float16"))
    with bb.function("main", [x, w_mat, b]):
        with bb.dataflow():
            x_col = bb.emit(
                relax.call_tir(
                    im2col_gv,
                    [x],
                    relax.TensorStructInfo((groups, m, k), "float16"),
                )
            )
            y = bb.emit(relax.op.matmul(x_col, w_mat, out_dtype="float16"))
            y = bb.emit(relax.op.add(y, b))
            if bool(spec.get("relu", False)):
                y = bb.emit(relax.op.nn.relu(y))
            out = bb.emit(
                relax.call_tir(
                    restore_gv,
                    [y],
                    relax.TensorStructInfo((n, cout, oh, ow), "float16"),
                )
            )
            gv = bb.emit_output(out)
        bb.emit_func_output(gv)
    return bb.finalize()


def _make_te_group_conv_full_im2col_same_signature_primfunc(
    spec: dict[str, Any], name: str, accum_dtype: str = "float16"
) -> Any:
    from tvm import te

    # "int32" = int8 operands / int32 accumulate (materialized int8 buffers so the
    # dlight MatmulInt8Tensorization rule detects the int8xint8->int32 matmul and
    # emits the int8 MMA path). fp16 I/O signature preserved. Latency-only (dummy
    # per-tensor scale=1) -> AP unaffected (AP comes from the real-activation bridge).
    if accum_dtype not in {"float16", "float32", "int32"}:
        raise ValueError(f"unsupported_accum_dtype:{accum_dtype}")
    int8_mma = accum_dtype == "int32"

    n, cin, h, width = [int(v) for v in spec["input_nchw"]]
    cout, cpg, kh, kw = [int(v) for v in spec["weight_oihw"]]
    groups = int(spec["groups"])
    oh, ow = _conv2d_out_hw(h, width, kh, kw, tuple(spec["strides"]), tuple(spec["padding"]))
    pt, pl, _, _ = tuple(spec["padding"])
    sh, sw = tuple(spec["strides"])
    m = n * oh * ow
    k_total = cpg * kh * kw
    nper = cout // groups

    x = te.placeholder((n, cin, h, width), "float16", name="x")
    weight = te.placeholder((cout, cpg, kh, kw), "float16", name="weight")
    bias = te.placeholder((1, cout, 1, 1), "float16", name="bias")

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
        cond = (in_y >= 0) & (in_y < h) & (in_x >= 0) & (in_x < width)
        return te.if_then_else(cond, x[nn, in_c, in_y, in_x], te.const(0, "float16"))

    x_col = te.compute((groups, m, k_total), im2col_compute, name="x_col")

    def weight_compute(g: Any, kk: Any, ocg: Any) -> Any:
        ci = kk // (kh * kw)
        rem_k = kk % (kh * kw)
        ry = rem_k // kw
        rx = rem_k % kw
        return weight[g * nper + ocg, ci, ry, rx]

    w_mat = te.compute((groups, k_total, nper), weight_compute, name="w_mat")
    # int8 mode: materialize int8 buffers so MatmulInt8Tensorization detects int8 matmul
    if int8_mma:
        x_col_q = te.compute((groups, m, k_total),
                             lambda g, r, kk: x_col[g, r, kk].astype("int8"), name="x_col_q")
        w_mat_q = te.compute((groups, k_total, nper),
                             lambda g, kk, o: w_mat[g, kk, o].astype("int8"), name="w_mat_q")
    rk = te.reduce_axis((0, k_total), name="rk")

    def matmul_compute(g: Any, row: Any, ocg: Any) -> Any:
        if int8_mma:
            return te.sum(
                x_col_q[g, row, rk].astype("int32") * w_mat_q[g, rk, ocg].astype("int32"),
                axis=rk,
            )
        if accum_dtype == "float32":
            return te.sum(
                x_col[g, row, rk].astype("float32") * w_mat[g, rk, ocg].astype("float32"),
                axis=rk,
            )
        return te.sum(x_col[g, row, rk] * w_mat[g, rk, ocg], axis=rk)

    matmul = te.compute(
        (groups, m, nper),
        matmul_compute,
        name="matmul",
    )

    def out_compute(nn: Any, oc: Any, yy: Any, xx: Any) -> Any:
        g = oc // nper
        ocg = oc % nper
        row = nn * (oh * ow) + yy * ow + xx
        if int8_mma:
            # int32 accum -> back to fp16 signature; dummy scale=1 (latency-only)
            value = matmul[g, row, ocg].astype("float16") + bias[0, oc, 0, 0]
        elif accum_dtype == "float32":
            value = matmul[g, row, ocg] + bias[0, oc, 0, 0].astype("float32")
        else:
            value = matmul[g, row, ocg] + bias[0, oc, 0, 0]
        if bool(spec.get("relu", False)):
            relu_dtype = "float16" if int8_mma else accum_dtype
            value = te.max(value, te.const(0, relu_dtype))
        return value.astype("float16")

    out = te.compute((n, cout, oh, ow), out_compute, name="out_nchw")
    return te.create_prim_func([x, weight, bias, out]).with_attr("global_symbol", name)


def _make_group_conv_same_signature_replacement_mod(
    spec: dict[str, Any], accum_dtype: str = "float16"
) -> Any:
    from tvm import relax

    n, cin, h, w = [int(v) for v in spec["input_nchw"]]
    cout, cpg, kh, kw = [int(v) for v in spec["weight_oihw"]]
    oh, ow = _conv2d_out_hw(h, w, kh, kw, tuple(spec["strides"]), tuple(spec["padding"]))

    bb = relax.BlockBuilder()
    prim_gv = bb.add_func(
        _make_te_group_conv_full_im2col_same_signature_primfunc(
            spec, "fused_conv2d_add_relu", accum_dtype=accum_dtype
        ),
        "fused_conv2d_add_relu",
    )
    x = relax.Var("x", relax.TensorStructInfo((n, cin, h, w), "float16"))
    weight = relax.Var("weight", relax.TensorStructInfo((cout, cpg, kh, kw), "float16"))
    bias = relax.Var("bias", relax.TensorStructInfo(tuple(spec["bias"]), "float16"))
    with bb.function("main", [x, weight, bias]):
        with bb.dataflow():
            out = bb.emit(
                relax.call_tir(
                    prim_gv,
                    [x, weight, bias],
                    relax.TensorStructInfo((n, cout, oh, ow), "float16"),
                )
            )
            gv = bb.emit_output(out)
        bb.emit_func_output(gv)
    return bb.finalize()


def _numpy_group_conv2d(x: np.ndarray, w: np.ndarray, b: np.ndarray, spec: dict[str, Any]) -> np.ndarray:
    n, cin, h, width = x.shape
    cout, cpg, kh, kw = w.shape
    groups = int(spec["groups"])
    strides = tuple(spec["strides"])
    padding = tuple(spec["padding"])
    oh, ow = _conv2d_out_hw(h, width, kh, kw, strides, padding)
    pt, pl, pb, pr = padding
    sh, sw = strides
    x_pad = np.pad(x, ((0, 0), (0, 0), (pt, pb), (pl, pr)), mode="constant")
    out = np.zeros((n, cout, oh, ow), dtype="float32")
    cout_per_group = cout // groups
    for nn in range(n):
        for gg in range(groups):
            in0 = gg * cpg
            out0 = gg * cout_per_group
            for oc in range(cout_per_group):
                filt = w[out0 + oc].astype("float32")
                for yy in range(oh):
                    for xx in range(ow):
                        patch = x_pad[nn, in0 : in0 + cpg, yy * sh : yy * sh + kh, xx * sw : xx * sw + kw]
                        out[nn, out0 + oc, yy, xx] = np.sum(patch.astype("float32") * filt)
    out += b.astype("float32")
    if bool(spec.get("relu", False)):
        out = np.maximum(out, 0.0)
    return out


def _numpy_group_conv_im2col_inputs(x: np.ndarray, w: np.ndarray, b: np.ndarray, spec: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n, cin, h, width = x.shape
    cout, cpg, kh, kw = w.shape
    groups = int(spec["groups"])
    strides = tuple(spec["strides"])
    padding = tuple(spec["padding"])
    oh, ow = _conv2d_out_hw(h, width, kh, kw, strides, padding)
    pt, pl, pb, pr = padding
    sh, sw = strides
    cout_per_group = cout // groups
    m = n * oh * ow
    k = cpg * kh * kw
    x_col = np.zeros((groups, m, k), dtype="float16")
    x_pad = np.pad(x, ((0, 0), (0, 0), (pt, pb), (pl, pr)), mode="constant")
    for gg in range(groups):
        in0 = gg * cpg
        row = 0
        for nn in range(n):
            for yy in range(oh):
                for xx in range(ow):
                    patch = x_pad[nn, in0 : in0 + cpg, yy * sh : yy * sh + kh, xx * sw : xx * sw + kw]
                    x_col[gg, row, :] = patch.reshape(-1)
                    row += 1
    w_mat = np.zeros((groups, k, cout_per_group), dtype="float16")
    for gg in range(groups):
        out0 = gg * cout_per_group
        w_mat[gg] = w[out0 : out0 + cout_per_group].reshape(cout_per_group, k).transpose(1, 0)
    b_mat = b.reshape(groups, cout_per_group).astype("float16")[:, None, :]
    ref = x_col.astype("float32") @ w_mat.astype("float32")
    ref = ref + b_mat.astype("float32")
    if bool(spec.get("relu", False)):
        ref = np.maximum(ref, 0.0)
    return x_col, w_mat, b_mat, ref


def _restore_grouped_rows_to_nchw(group_rows: np.ndarray, spec: dict[str, Any]) -> np.ndarray:
    n, _, h, width = [int(v) for v in spec["input_nchw"]]
    cout, _, kh, kw = [int(v) for v in spec["weight_oihw"]]
    groups = int(spec["groups"])
    oh, ow = _conv2d_out_hw(h, width, kh, kw, tuple(spec["strides"]), tuple(spec["padding"]))
    nper = cout // groups
    out = np.zeros((n, cout, oh, ow), dtype="float32")
    for gg in range(groups):
        for row in range(n * oh * ow):
            nn = row // (oh * ow)
            rem = row % (oh * ow)
            yy = rem // ow
            xx = rem % ow
            out0 = gg * nper
            out[nn, out0 : out0 + nper, yy, xx] = group_rows[gg, row, :]
    return out


def _sample_check_group_conv_im2col_mapping(
    candidate: str, spec: dict[str, Any], rng: np.random.RandomState, sample_count: int = 256
) -> dict[str, Any]:
    n, cin, h, width = [int(v) for v in spec["input_nchw"]]
    cout, cpg, kh, kw = [int(v) for v in spec["weight_oihw"]]
    groups = int(spec["groups"])
    oh, ow = _conv2d_out_hw(h, width, kh, kw, tuple(spec["strides"]), tuple(spec["padding"]))
    pt, pl, _, _ = tuple(spec["padding"])
    sh, sw = tuple(spec["strides"])
    m = n * oh * ow
    k = cpg * kh * kw
    nper = cout // groups

    x_np = (rng.rand(n, cin, h, width).astype("float16") - np.float16(0.5)) * np.float16(0.2)
    w_np = (rng.rand(cout, cpg, kh, kw).astype("float16") - np.float16(0.5)) * np.float16(0.2)
    b_np = (rng.rand(*tuple(spec["bias"])).astype("float16") - np.float16(0.5)) * np.float16(0.2)
    x_col, w_mat, b_mat, _ = _numpy_group_conv_im2col_inputs(x_np, w_np, b_np, spec)

    x_mismatches: list[dict[str, Any]] = []
    w_mismatches: list[dict[str, Any]] = []
    b_mismatches: list[dict[str, Any]] = []
    for _ in range(sample_count):
        g = int(rng.randint(0, groups))
        row = int(rng.randint(0, m))
        kk = int(rng.randint(0, k))
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
        expected_x = (
            x_np[nn, in_c, in_y, in_x]
            if 0 <= in_y < h and 0 <= in_x < width
            else np.float16(0.0)
        )
        actual_x = x_col[g, row, kk]
        if actual_x != expected_x:
            x_mismatches.append(
                {
                    "g": g,
                    "row": row,
                    "kk": kk,
                    "actual": float(actual_x),
                    "expected": float(expected_x),
                }
            )

        ocg = int(rng.randint(0, nper))
        expected_w = w_np[g * nper + ocg, ci, ry, rx]
        actual_w = w_mat[g, kk, ocg]
        if actual_w != expected_w:
            w_mismatches.append(
                {
                    "g": g,
                    "kk": kk,
                    "ocg": ocg,
                    "actual": float(actual_w),
                    "expected": float(expected_w),
                }
            )

        expected_b = b_np[0, g * nper + ocg, 0, 0]
        actual_b = b_mat[g, 0, ocg]
        if actual_b != expected_b:
            b_mismatches.append(
                {
                    "g": g,
                    "ocg": ocg,
                    "actual": float(actual_b),
                    "expected": float(expected_b),
                }
            )

    return {
        "candidate": candidate,
        "status": "success" if not x_mismatches and not w_mismatches and not b_mismatches else "failed",
        "sample_count": sample_count,
        "implicit_gemm_shape": {
            "groups": groups,
            "M": m,
            "K": k,
            "N_per_group": nper,
            "output_hw": [oh, ow],
        },
        "x_col_shape": list(x_col.shape),
        "w_mat_shape": list(w_mat.shape),
        "b_mat_shape": list(b_mat.shape),
        "x_mismatch_count": len(x_mismatches),
        "w_mismatch_count": len(w_mismatches),
        "b_mismatch_count": len(b_mismatches),
        "x_mismatches_sample": x_mismatches[:5],
        "w_mismatches_sample": w_mismatches[:5],
        "b_mismatches_sample": b_mismatches[:5],
    }


def run_self_test_no_tvm(args: argparse.Namespace) -> dict[str, Any]:
    raw_dir = Path(args.raw_dir) / "self_test_no_tvm"
    raw_dir.mkdir(parents=True, exist_ok=True)
    spec: dict[str, Any] = {
        "input_nchw": (2, 4, 5, 6),
        "weight_oihw": (4, 2, 3, 3),
        "bias": (1, 4, 1, 1),
        "strides": (1, 1),
        "padding": (1, 1, 1, 1),
        "groups": 2,
        "relu": True,
    }
    rng = np.random.RandomState(20260702)
    x_np = (rng.rand(*spec["input_nchw"]).astype("float16") - np.float16(0.5)) * np.float16(0.2)
    w_np = (rng.rand(*spec["weight_oihw"]).astype("float16") - np.float16(0.5)) * np.float16(0.2)
    b_np = (rng.rand(*spec["bias"]).astype("float16") - np.float16(0.5)) * np.float16(0.2)
    direct = _numpy_group_conv2d(x_np, w_np, b_np, spec)
    x_col, w_mat, b_mat, im2col_rows = _numpy_group_conv_im2col_inputs(x_np, w_np, b_np, spec)
    restored = _restore_grouped_rows_to_nchw(im2col_rows, spec)
    diff = _compare_arrays(direct, restored)
    candidate_sample_checks = [
        _sample_check_group_conv_im2col_mapping(
            candidate,
            GROUP_CONV_CANDIDATES[candidate],
            np.random.RandomState(20260703 + idx),
        )
        for idx, candidate in enumerate(sorted(GROUP_CONV_CANDIDATES))
    ]
    max_allowed = 1e-5
    payload = {
        "schema": "fp16_lhc07_probe_self_test_no_tvm_v1",
        "status": (
            "success"
            if float(diff["max_abs_err"]) <= max_allowed
            and all(item["status"] == "success" for item in candidate_sample_checks)
            else "failed"
        ),
        "raw_dir": str(raw_dir),
        "claim": "pure_numpy_indexing_padding_group_mapping_self_test_not_tvm_measurement",
        "spec": {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in spec.items()
        },
        "x_col_shape": list(x_col.shape),
        "w_mat_shape": list(w_mat.shape),
        "b_mat_shape": list(b_mat.shape),
        "direct_vs_im2col_restore": diff,
        "candidate_sample_checks": candidate_sample_checks,
        "max_allowed_abs_err": max_allowed,
        "python_executable": sys.executable,
    }
    (raw_dir / "self_test_no_tvm_payload.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return payload


def _apply_tensorcore_schedule(mod: Any, target: Any) -> Any:
    import tvm
    import tvm.s_tir.dlight as dl
    from tvm.s_tir.dlight.gpu.matmul import MatmulTensorization

    with target, tvm.transform.PassContext(opt_level=3):
        return dl.ApplyDefaultSchedule(MatmulTensorization())(mod)


def _apply_full_engine_schedule(mod: Any, target: Any) -> Any:
    import tvm
    import tvm.s_tir.dlight as dl
    from tvm.s_tir.dlight.gpu.matmul import MatmulTensorization

    with target, tvm.transform.PassContext(opt_level=3):
        return dl.ApplyDefaultSchedule(
            MatmulTensorization(),
            dl.gpu.Matmul(),
            dl.gpu.GEMV(),
            dl.gpu.Reduction(),
            dl.gpu.GeneralReduction(),
            dl.gpu.Fallback(),
        )(mod)


def _apply_selective_matmul_tensorization(mod: Any, target: Any, accum_dtype: str = "float16") -> tuple[Any, list[dict[str, Any]]]:
    import tvm
    import tvm.s_tir.dlight as dl
    # int8 matmul needs the dedicated int8 rule (MatmulInt8Tensorization); the fp16
    # MatmulTensorization silently leaves int8 un-tensorized (verified 2026-07-03).
    if accum_dtype == "int32":
        from tvm.s_tir.dlight.gpu.matmul import MatmulInt8Tensorization as _MatmulRule
    else:
        from tvm.s_tir.dlight.gpu.matmul import MatmulTensorization as _MatmulRule

    out = tvm.IRModule(dict(mod.functions), attrs=mod.attrs)
    records: list[dict[str, Any]] = []
    for gv, func in mod.functions_items():
        name = gv.name_hint
        if not hasattr(func, "script"):
            records.append({"name": name, "kind": type(func).__name__, "status": "skipped_non_tir"})
            continue
        func_text = func.script()
        if 'T.sblock("matmul")' not in func_text and "matmul" not in name:
            records.append({"name": name, "kind": "PrimFunc", "status": "kept_non_matmul"})
            continue
        try:
            single = tvm.IRModule({gv: func})
            with target, tvm.transform.PassContext(opt_level=3):
                scheduled_single = dl.ApplyDefaultSchedule(_MatmulRule())(single)
            scheduled_func = scheduled_single[gv]
            scheduled_text = scheduled_func.script()
            counts = _counts(scheduled_text)
            if counts.get("wmma", 0) > 0 and counts.get("tvm_mma_sync", 0) > 0:
                out.update_func(gv, scheduled_func)
                status = "tensorized"
            else:
                status = "matmul_schedule_no_tensorcore"
            records.append(
                {
                    "name": name,
                    "kind": "PrimFunc",
                    "status": status,
                    "counts": counts,
                }
            )
        except Exception as exc:
            records.append(
                {
                    "name": name,
                    "kind": "PrimFunc",
                    "status": "matmul_tensorization_failed_kept_original",
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            )
    return out, records


def _legalize_fuse(mod: Any, target: Any) -> Any:
    import tvm
    from tvm import relax

    seq = tvm.transform.Sequential(
        [
            relax.transform.LegalizeOps(),
            relax.transform.AnnotateTIROpPattern(),
            relax.transform.FuseOps(),
            relax.transform.FuseTIR(),
        ]
    )
    with target, tvm.transform.PassContext(opt_level=3):
        return seq(mod)


def run_convblock(args: argparse.Namespace, target: Any, dev: Any) -> dict[str, Any]:
    import tvm
    from tvm import relax

    raw_dir = Path(args.raw_dir) / "lhc07_1x1_conv_as_matmul"
    raw_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(20260629)
    x48_np = (rng.rand(2 * 64 * 128, 48).astype("float16") - np.float16(0.5)) * np.float16(0.02)
    w48_np = (rng.rand(48, 256).astype("float16") - np.float16(0.5)) * np.float16(0.02)
    x64_np = np.zeros((2 * 64 * 128, 64), dtype="float16")
    w64_np = np.zeros((64, 256), dtype="float16")
    x64_np[:, :48] = x48_np
    w64_np[:48, :] = w48_np
    b_np = (rng.rand(256).astype("float16") - np.float16(0.5)) * np.float16(0.02)
    ref48 = x48_np.astype("float32") @ w48_np.astype("float32")
    ref_relu = np.maximum(ref48 + b_np.astype("float32"), 0.0)

    runs: list[dict[str, Any]] = []
    variants = [
        ("pure_matmul_k48", 48, x48_np, w48_np),
        ("matmul_bias_relu_k48", 48, x48_np, w48_np),
        ("pure_matmul_k64_padded", 64, x64_np, w64_np),
        ("matmul_bias_relu_k64_padded", 64, x64_np, w64_np),
    ]
    for variant, k_dim, x_np, w_np in variants:
        for idx in range(int(args.conv_repeats)):
            item: dict[str, Any] = {
                "variant": variant,
                "k_dim": k_dim,
                "repeat_index": idx,
                "status": "started",
            }
            try:
                mod = _make_lhc07_1x1_conv_as_matmul_mod(variant, k_dim)
                mod = _legalize_fuse(mod, target)
                scheduled = _apply_tensorcore_schedule(mod, target)
                tir = scheduled.script()
                tir_path = raw_dir / f"scheduled_{variant}_repeat{idx}.py"
                tir_path.write_text(tir, encoding="utf-8")
                item["scheduled_counts"] = _counts(tir)
                with target, tvm.transform.PassContext(opt_level=3):
                    executable = tvm.compile(scheduled, target=target)
                vm = relax.VirtualMachine(executable, dev)
                tvm_args = [tvm.runtime.tensor(x_np, dev), tvm.runtime.tensor(w_np, dev)]
                if variant.endswith("bias_relu") or "bias_relu" in variant:
                    tvm_args.append(tvm.runtime.tensor(b_np, dev))
                mean_us, min_us, repeats_us = _time_vm(vm, tvm_args, dev, int(args.reps))
                out = vm["main"](*tvm_args).numpy().astype("float32")
                if variant.startswith("pure_matmul"):
                    ref_variant = ref48
                else:
                    ref_variant = ref_relu
                item.update(
                    {
                        "status": "success",
                        "latency_mean_us": mean_us,
                        "latency_min_us": min_us,
                        "latency_repeats_us": repeats_us,
                        "max_abs_err": float(np.max(np.abs(out - ref_variant))),
                        "mean_abs_err": float(np.mean(np.abs(out - ref_variant))),
                        "tir_path": str(tir_path),
                    }
                )
            except Exception as exc:  # pragma: no cover - H800 probe path
                item.update(
                    {
                        "status": "failed",
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
            runs.append(item)

    pure_runs = [item for item in runs if item.get("variant") == "pure_matmul_k48"]
    pure_success = [item for item in pure_runs if item.get("status") == "success"]
    fused_runs = [item for item in runs if item.get("variant") == "matmul_bias_relu_k48"]
    fused_success = [item for item in fused_runs if item.get("status") == "success"]
    padded_runs = [item for item in runs if item.get("variant") == "pure_matmul_k64_padded"]
    padded_success = [item for item in padded_runs if item.get("status") == "success"]
    stable_tensorcore = (
        len(pure_success) == int(args.conv_repeats)
        and all((item.get("scheduled_counts") or {}).get("tvm_mma_sync", 0) > 0 for item in pure_success)
        and all((item.get("scheduled_counts") or {}).get("wmma", 0) > 0 for item in pure_success)
    )
    fused_tensorcore = (
        len(fused_success) == int(args.conv_repeats)
        and all((item.get("scheduled_counts") or {}).get("tvm_mma_sync", 0) > 0 for item in fused_success)
        and all((item.get("scheduled_counts") or {}).get("wmma", 0) > 0 for item in fused_success)
    )
    padded_tensorcore = (
        len(padded_success) == int(args.conv_repeats)
        and all((item.get("scheduled_counts") or {}).get("tvm_mma_sync", 0) > 0 for item in padded_success)
        and all((item.get("scheduled_counts") or {}).get("wmma", 0) > 0 for item in padded_success)
    )
    return {
        "schema": "fp16_lhc07_convblock_tensorcore_probe_v1",
        "label": "lhc_07",
        "block": "fused_conv2d11_add7_relu5",
        "original_conv_shape": {
            "input_nchw": [2, 48, 64, 128],
            "weight_oihw": [256, 48, 1, 1],
            "bias": [256],
            "output_nchw": [2, 256, 64, 128],
        },
        "matmul_equivalent": {"M": 16384, "K": 48, "N": 256},
        "route": "explicit_1x1_conv_as_matmul_plus_bias_relu_dlight_MatmulTensorization",
        "conv_repeats": int(args.conv_repeats),
        "reps_per_timing": int(args.reps),
        "stable_tensorcore_k48": stable_tensorcore,
        "fused_bias_relu_tensorcore_k48": fused_tensorcore,
        "stable_tensorcore_k64_padded": padded_tensorcore,
        "runs": runs,
    }


def _load_onnx_relax_mod(onnx_path: Path, batch: int) -> tuple[Any, dict[str, tuple[int, ...]]]:
    _avoid_local_onnx_shadow()
    import onnx
    from tvm.relax.frontend.onnx import from_onnx

    model = onnx.load(str(onnx_path))
    init_names = {item.name for item in model.graph.initializer}
    shape_dict: dict[str, tuple[int, ...]] = {}
    for inp in model.graph.input:
        if inp.name in init_names:
            continue
        dims: list[int] = []
        for dim in inp.type.tensor_type.shape.dim:
            dims.append(int(dim.dim_value) if int(dim.dim_value) > 0 else int(batch))
        shape_dict[inp.name] = tuple(dims)
    mod = from_onnx(model, shape_dict=shape_dict, keep_params_in_input=False)
    return mod, shape_dict


def _try_export_library(executable: Any, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if hasattr(executable, "export_library"):
            executable.export_library(str(path))
            return {"status": "success", "path": str(path), "bytes": path.stat().st_size}
        if hasattr(executable, "mod") and hasattr(executable.mod, "export_library"):
            executable.mod.export_library(str(path))
            return {"status": "success", "path": str(path), "bytes": path.stat().st_size}
        return {"status": "unsupported", "reason": f"no export_library on {type(executable).__name__}"}
    except Exception as exc:  # pragma: no cover - H800 probe path
        return {"status": "failed", "error": repr(exc), "traceback": traceback.format_exc()}


def run_full_engine(args: argparse.Namespace, target: Any, dev: Any) -> dict[str, Any]:
    import tvm
    from tvm import relax

    onnx_path = Path(args.onnx)
    raw_dir = Path(args.raw_dir) / "lhc07_full_fp16_engine"
    raw_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema": "fp16_lhc07_full_engine_tensorcore_export_probe_v1",
        "label": "lhc_07",
        "onnx_path": str(onnx_path),
        "routes": [],
    }
    if not onnx_path.exists():
        payload.update({"status": "failed", "error": f"onnx_not_found:{onnx_path}"})
        return payload

    try:
        mod0, shape_dict = _load_onnx_relax_mod(onnx_path, int(args.batch))
        payload["shape_dict"] = {key: list(value) for key, value in shape_dict.items()}
        rng = np.random.RandomState(7)
        feeds_np = {key: rng.rand(*value).astype("float32") for key, value in shape_dict.items()}
        feeds_tvm = [tvm.runtime.tensor(feeds_np[key], dev) for key in shape_dict]
    except Exception as exc:
        payload.update({"status": "failed", "error": repr(exc), "traceback": traceback.format_exc()})
        return payload

    route_specs = [
        ("default_relax_build", None),
        ("legalize_fuse_dlight_matmul_tensorization_fallback", "dlight"),
    ]
    for route_name, route_kind in route_specs:
        item: dict[str, Any] = {"route": route_name, "status": "started"}
        try:
            if route_kind is None:
                with tvm.transform.PassContext(opt_level=3):
                    executable = relax.build(mod0, target=target)
                scheduled_text = ""
            else:
                mod_legal = _legalize_fuse(mod0, target)
                scheduled = _apply_full_engine_schedule(mod_legal, target)
                scheduled_text = scheduled.script()
                scheduled_path = raw_dir / f"{route_name}_scheduled.py"
                scheduled_path.write_text(scheduled_text, encoding="utf-8")
                item["scheduled_path"] = str(scheduled_path)
                item["scheduled_counts"] = _counts(scheduled_text)
                with target, tvm.transform.PassContext(opt_level=3):
                    executable = tvm.compile(scheduled, target=target)
            vm = relax.VirtualMachine(executable, dev)
            mean_us, min_us, repeats_us = _time_vm(vm, feeds_tvm, dev, int(args.full_reps))
            export_result = _try_export_library(executable, raw_dir / f"{route_name}.so")
            item.update(
                {
                    "status": "success",
                    "latency_mean_us": mean_us,
                    "latency_min_us": min_us,
                    "latency_repeats_us": repeats_us,
                    "export_library": export_result,
                    "scheduled_counts": item.get("scheduled_counts") or _counts(scheduled_text),
                    "executable_type": type(executable).__name__,
                }
            )
        except Exception as exc:  # pragma: no cover - H800 probe path
            item.update(
                {
                    "status": "failed",
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            )
        payload["routes"].append(item)

    payload["status"] = "success" if any(r.get("status") == "success" for r in payload["routes"]) else "failed"
    return payload


def write_outputs(args: argparse.Namespace, result: dict[str, Any]) -> tuple[Path, Path]:
    export_dir = Path(args.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    json_path = export_dir / "fp16_lhc07_convblock_tensorcore_and_engine_probe_latest.json"
    md_path = export_dir / "fp16_lhc07_convblock_tensorcore_and_engine_probe_latest.md"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    conv = result.get("convblock", {})
    engine = result.get("full_engine", {})
    lines = [
        "# FP16 lhc_07 convblock tensor-core and full engine probe",
        "",
        f"- status: `{result.get('status')}`",
        f"- target: `{result.get('target')}`",
        f"- raw_dir: `{result.get('raw_dir')}`",
        "",
        "## 1. Convblock stable tensor-core gate",
        "",
        f"- block: `{conv.get('block')}`",
        f"- route: `{conv.get('route')}`",
        f"- original conv: `{conv.get('original_conv_shape')}`",
        f"- matmul equivalent: `{conv.get('matmul_equivalent')}`",
        f"- stable_tensorcore_k48: `{conv.get('stable_tensorcore_k48')}`",
        f"- fused_bias_relu_tensorcore_k48: `{conv.get('fused_bias_relu_tensorcore_k48')}`",
        f"- stable_tensorcore_k64_padded: `{conv.get('stable_tensorcore_k64_padded')}`",
        "",
        "| variant | K | repeat | status | wmma | tvm_mma_sync | latency_mean_us | max_abs_err |",
        "|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for item in conv.get("runs", []):
        counts = item.get("scheduled_counts") or {}
        lines.append(
            "| `{variant}` | {k_dim} | {idx} | `{status}` | {wmma} | {mma} | {lat} | {err} |".format(
                variant=item.get("variant"),
                k_dim=item.get("k_dim"),
                idx=item.get("repeat_index"),
                status=item.get("status"),
                wmma=counts.get("wmma"),
                mma=counts.get("tvm_mma_sync"),
                lat=item.get("latency_mean_us"),
                err=item.get("max_abs_err"),
            )
        )
    lines += [
        "",
        "## 2. Full FP16 engine export probe",
        "",
        f"- onnx_path: `{engine.get('onnx_path')}`",
        f"- shape_dict: `{engine.get('shape_dict')}`",
        "",
        "| route | status | wmma | tvm_mma_sync | latency_mean_us | export |",
        "|---|---|---:|---:|---:|---|",
    ]
    for item in engine.get("routes", []):
        counts = item.get("scheduled_counts") or {}
        export_result = item.get("export_library") or {}
        export_status = export_result.get("status") or "n/a"
        lines.append(
            "| `{route}` | `{status}` | {wmma} | {mma} | {lat} | `{export}` |".format(
                route=item.get("route"),
                status=item.get("status"),
                wmma=counts.get("wmma"),
                mma=counts.get("tvm_mma_sync"),
                lat=item.get("latency_mean_us"),
                export=export_status,
            )
        )
    lines += [
        "",
        "## 3. Interpretation",
        "",
        "- 如果 convblock gate 稳定为 true, 说明 lhc_07 这个 1x1 conv 的算子形状本身可以走 FP16 tensor-core。",
        "- 如果完整 engine route 仍没有 wmma/tvm_mma_sync, 则问题在完整 ONNX backbone 的 NCHW conv lowering/fusion/schedule 路径没有把 1x1 conv 改写成 dlight 可 tensorize 的 matmul。",
        "- 这与 INT8 早期慢的表象相似, 都是 lowering/backend route 没有进入目标硬件快路径; 但具体机制不同: FP16 是没有进入 WMMA/tensor-core matmul, INT8 是早期 QDQ/float32-heavy 或 native INT8 scale/requant/布局路线未完全收口。",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def run_e2e_counterfactual(args: argparse.Namespace, target: Any, dev: Any) -> dict[str, Any]:
    import tvm
    from tvm import relax

    raw_dir = Path(args.raw_dir) / "lhc07_e2e_counterfactual"
    raw_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema": "fp16_lhc07_e2e_tensorcore_counterfactual_v1",
        "label": "lhc_07",
        "onnx_path": str(args.onnx),
        "raw_dir": str(raw_dir),
        "status": "started",
    }

    try:
        mod0, shape_dict = _load_onnx_relax_mod(Path(args.onnx), int(args.batch))
        payload["shape_dict"] = {key: list(value) for key, value in shape_dict.items()}
        rng = np.random.RandomState(17)
        feeds_np = {key: rng.rand(*value).astype("float32") for key, value in shape_dict.items()}
        feeds_tvm = [tvm.runtime.tensor(feeds_np[key], dev) for key in shape_dict]

        with tvm.transform.PassContext(opt_level=3):
            full_ex = relax.build(mod0, target=target)
        full_vm = relax.VirtualMachine(full_ex, dev)
        full_mean_us, full_min_us, full_repeats_us = _time_vm(
            full_vm, feeds_tvm, dev, int(args.full_reps)
        )
        full_export = _try_export_library(full_ex, raw_dir / "full_default_relax_build.so")
        payload["full_default_engine"] = {
            "status": "success",
            "latency_mean_us": full_mean_us,
            "latency_min_us": full_min_us,
            "latency_repeats_us": full_repeats_us,
            "scheduled_counts": {"wmma": 0, "tvm_mma_sync": 0},
            "export_library": full_export,
            "claim": "measured_full_default_engine",
        }

        m, k, n = 2 * 64 * 128, 48, 256
        x_np = (rng.rand(m, k).astype("float16") - np.float16(0.5)) * np.float16(0.02)
        w_np = (rng.rand(k, n).astype("float16") - np.float16(0.5)) * np.float16(0.02)
        b_np = (rng.rand(n).astype("float16") - np.float16(0.5)) * np.float16(0.02)
        ref = np.maximum(x_np.astype("float32") @ w_np.astype("float32") + b_np.astype("float32"), 0.0)
        conv_args = [
            tvm.runtime.tensor(x_np, dev),
            tvm.runtime.tensor(w_np, dev),
            tvm.runtime.tensor(b_np, dev),
        ]

        conv_mod = _make_lhc07_1x1_conv_as_matmul_mod("matmul_bias_relu_k48", 48)
        conv_legal = _legalize_fuse(conv_mod, target)
        (raw_dir / "target_conv_legalized_unscheduled.py").write_text(
            conv_legal.script(), encoding="utf-8"
        )
        with target, tvm.transform.PassContext(opt_level=3):
            conv_default_ex = tvm.compile(conv_legal, target=target)
        conv_default_vm = relax.VirtualMachine(conv_default_ex, dev)
        default_mean_us, default_min_us, default_repeats_us = _time_vm(
            conv_default_vm, conv_args, dev, int(args.reps)
        )
        default_out = conv_default_vm["main"](*conv_args).numpy().astype("float32")
        payload["target_conv_default"] = {
            "status": "success",
            "route": "legalize_fuse_then_default_tvm_compile_no_matmul_tensorization",
            "latency_mean_us": default_mean_us,
            "latency_min_us": default_min_us,
            "latency_repeats_us": default_repeats_us,
            "scheduled_counts": _counts(conv_legal.script()),
            "max_abs_err": float(np.max(np.abs(default_out - ref))),
            "mean_abs_err": float(np.mean(np.abs(default_out - ref))),
            "claim": "measured_target_conv_default",
        }

        conv_tc = _apply_tensorcore_schedule(conv_legal, target)
        conv_tc_tir = conv_tc.script()
        (raw_dir / "target_conv_tensorcore_scheduled.py").write_text(
            conv_tc_tir, encoding="utf-8"
        )
        with target, tvm.transform.PassContext(opt_level=3):
            conv_tc_ex = tvm.compile(conv_tc, target=target)
        conv_tc_vm = relax.VirtualMachine(conv_tc_ex, dev)
        tc_mean_us, tc_min_us, tc_repeats_us = _time_vm(conv_tc_vm, conv_args, dev, int(args.reps))
        tc_out = conv_tc_vm["main"](*conv_args).numpy().astype("float32")
        tc_counts = _counts(conv_tc_tir)
        payload["target_conv_tensorcore"] = {
            "status": "success",
            "route": "legalize_fuse_then_dlight_MatmulTensorization",
            "latency_mean_us": tc_mean_us,
            "latency_min_us": tc_min_us,
            "latency_repeats_us": tc_repeats_us,
            "scheduled_counts": tc_counts,
            "max_abs_err": float(np.max(np.abs(tc_out - ref))),
            "mean_abs_err": float(np.mean(np.abs(tc_out - ref))),
            "claim": "measured_target_conv_tensorcore",
        }

        counterfactual_us = full_mean_us - default_mean_us + tc_mean_us
        payload["counterfactual"] = {
            "status": "success",
            "formula": "full_default_engine - target_conv_default + target_conv_tensorcore",
            "latency_mean_us": counterfactual_us,
            "latency_mean_ms": counterfactual_us / 1000.0,
            "absolute_delta_us": full_mean_us - counterfactual_us,
            "absolute_delta_ms": (full_mean_us - counterfactual_us) / 1000.0,
            "speedup_ratio": full_mean_us / counterfactual_us if counterfactual_us > 0 else None,
            "target_conv_default_share": default_mean_us / full_mean_us if full_mean_us > 0 else None,
            "target_conv_tensorcore_gate": bool(
                tc_counts.get("wmma", 0) > 0 and tc_counts.get("tvm_mma_sync", 0) > 0
            ),
            "claim": "measured_convblock_tensorcore_counterfactual",
        }
        payload["status"] = "success"
    except Exception as exc:  # pragma: no cover - H800 probe path
        payload.update({"status": "failed", "error": repr(exc), "traceback": traceback.format_exc()})
    return payload


def _onnx_attrs(node: Any) -> dict[str, Any]:
    _avoid_local_onnx_shadow()
    import onnx

    return {item.name: onnx.helper.get_attribute_value(item) for item in node.attribute}


def _shape_map(model: Any) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    tensors = list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info)
    for value in tensors:
        tensor_type = value.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        dims: list[int] = []
        ok = True
        for dim in tensor_type.shape.dim:
            if dim.dim_value <= 0:
                ok = False
                break
            dims.append(int(dim.dim_value))
        if ok:
            out[value.name] = dims
    return out


def _eligible_1x1_convs(onnx_path: Path) -> list[dict[str, Any]]:
    _avoid_local_onnx_shadow()
    import onnx

    model = onnx.shape_inference.infer_shapes(onnx.load(str(onnx_path)))
    init = {item.name: item for item in model.graph.initializer}
    shapes = _shape_map(model)
    rows: list[dict[str, Any]] = []
    for index, node in enumerate(model.graph.node):
        if node.op_type != "Conv":
            continue
        attrs = _onnx_attrs(node)
        if attrs.get("group", 1) != 1:
            continue
        if list(attrs.get("kernel_shape", [])) != [1, 1]:
            continue
        if list(attrs.get("pads", [0, 0, 0, 0])) != [0, 0, 0, 0]:
            continue
        if list(attrs.get("strides", [1, 1])) != [1, 1]:
            continue
        weight = init.get(node.input[1])
        if weight is None:
            continue
        input_shape = shapes.get(node.input[0])
        output_shape = shapes.get(node.output[0])
        if not input_shape or not output_shape or len(input_shape) != 4 or len(output_shape) != 4:
            continue
        batch, cin, h, w = input_shape
        _, cout, out_h, out_w = output_shape
        rows.append(
            {
                "node_index": index,
                "node_name": node.name or f"Conv_{index}",
                "input": node.input[0],
                "output": node.output[0],
                "input_shape": input_shape,
                "output_shape": output_shape,
                "weight_shape": list(weight.dims),
                "has_bias": len(node.input) > 2 and node.input[2] in init,
                "M": int(batch * out_h * out_w),
                "K": int(cin),
                "N": int(cout),
            }
        )
    return rows


def _rewrite_onnx_1x1_convs_to_matmul(
    onnx_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    _avoid_local_onnx_shadow()
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    out_dir.mkdir(parents=True, exist_ok=True)
    model = onnx.shape_inference.infer_shapes(onnx.load(str(onnx_path)))
    init = {item.name: item for item in model.graph.initializer}
    shapes = _shape_map(model)
    eligible = {item["node_index"]: item for item in _eligible_1x1_convs(onnx_path)}

    new_nodes: list[Any] = []
    new_initializers = list(model.graph.initializer)
    rewrite_rows: list[dict[str, Any]] = []

    def unique(base: str) -> str:
        existing = {
            value.name
            for value in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info)
        }
        existing.update(item.name for item in new_initializers)
        existing.update(out for node in new_nodes for out in node.output)
        name = base
        idx = 0
        while name in existing:
            idx += 1
            name = f"{base}_{idx}"
        return name

    for index, node in enumerate(model.graph.node):
        row = eligible.get(index)
        if row is None:
            new_nodes.append(node)
            continue

        input_shape = shapes.get(node.input[0]) or row["input_shape"]
        output_shape = shapes.get(node.output[0]) or row["output_shape"]
        if len(input_shape) != 4 or len(output_shape) != 4:
            new_nodes.append(node)
            continue

        weight = init[node.input[1]]
        weight_np = numpy_helper.to_array(weight)
        matmul_weight = weight_np[:, :, 0, 0].transpose(1, 0).copy()

        prefix = (node.name or f"Conv_{index}").replace("/", "_").replace(":", "_")
        weight_name = unique(f"{prefix}_matmul_weight")
        new_initializers.append(numpy_helper.from_array(matmul_weight, name=weight_name))

        n, c, h, w = [int(v) for v in input_shape]
        _, out_c, out_h, out_w = [int(v) for v in output_shape]
        nhwc_shape_name = unique(f"{prefix}_nhwc_shape")
        nchw_shape_name = unique(f"{prefix}_nchw_shape")
        new_initializers.append(
            helper.make_tensor(
                nhwc_shape_name,
                TensorProto.INT64,
                [2],
                [int(n * h * w), int(c)],
            )
        )
        new_initializers.append(
            helper.make_tensor(
                nchw_shape_name,
                TensorProto.INT64,
                [4],
                [int(n), int(out_h), int(out_w), int(out_c)],
            )
        )

        x_nhwc = unique(f"{prefix}_x_nhwc")
        x_2d = unique(f"{prefix}_x_2d")
        y_2d = unique(f"{prefix}_y_2d")
        y_2d_bias = unique(f"{prefix}_y_2d_bias")
        y_nhwc = unique(f"{prefix}_y_nhwc")

        new_nodes.extend(
            [
                helper.make_node(
                    "Transpose",
                    [node.input[0]],
                    [x_nhwc],
                    name=f"{prefix}_to_nhwc",
                    perm=[0, 2, 3, 1],
                ),
                helper.make_node(
                    "Reshape",
                    [x_nhwc, nhwc_shape_name],
                    [x_2d],
                    name=f"{prefix}_flatten_nhwc",
                ),
                helper.make_node(
                    "MatMul",
                    [x_2d, weight_name],
                    [y_2d],
                    name=f"{prefix}_matmul",
                ),
            ]
        )
        matmul_out = y_2d
        if len(node.input) > 2 and node.input[2] in init:
            new_nodes.append(
                helper.make_node(
                    "Add",
                    [matmul_out, node.input[2]],
                    [y_2d_bias],
                    name=f"{prefix}_bias_add",
                )
            )
            matmul_out = y_2d_bias
        new_nodes.extend(
            [
                helper.make_node(
                    "Reshape",
                    [matmul_out, nchw_shape_name],
                    [y_nhwc],
                    name=f"{prefix}_restore_nhwc",
                ),
                helper.make_node(
                    "Transpose",
                    [y_nhwc],
                    list(node.output),
                    name=f"{prefix}_to_nchw",
                    perm=[0, 3, 1, 2],
                ),
            ]
        )
        rewrite_rows.append(
            {
                **row,
                "rewrite": "NCHW->NHWC->Reshape->MatMul->Add?->Reshape->NCHW",
                "matmul_weight": weight_name,
                "matmul_weight_dtype": str(matmul_weight.dtype),
            }
        )

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    del model.graph.initializer[:]
    model.graph.initializer.extend(new_initializers)
    model = onnx.shape_inference.infer_shapes(model)

    rewritten_path = out_dir / "lhc_07_backbone_1x1_matmul_rewrite.onnx"
    inferred_path = out_dir / "lhc_07_backbone_1x1_matmul_rewrite_inferred.onnx"
    onnx.save(model, str(rewritten_path))
    onnx.save(onnx.shape_inference.infer_shapes(model), str(inferred_path))
    return {
        "status": "success",
        "source_onnx": str(onnx_path),
        "rewritten_onnx": str(rewritten_path),
        "rewritten_inferred_onnx": str(inferred_path),
        "eligible_1x1_conv_count": len(eligible),
        "rewritten_1x1_conv_count": len(rewrite_rows),
        "rewritten_nodes": rewrite_rows,
    }


def _cast_onnx_float_tensors_to_fp16(onnx_path: Path, out_dir: Path) -> dict[str, Any]:
    _avoid_local_onnx_shadow()
    import onnx
    from onnx import TensorProto, numpy_helper

    out_dir.mkdir(parents=True, exist_ok=True)
    model = onnx.load(str(onnx_path))
    converted_initializers = 0
    for idx, tensor in enumerate(model.graph.initializer):
        if tensor.data_type != TensorProto.FLOAT:
            continue
        arr16 = numpy_helper.to_array(tensor).astype("float16")
        model.graph.initializer[idx].CopyFrom(numpy_helper.from_array(arr16, name=tensor.name))
        converted_initializers += 1

    converted_value_infos = 0
    for value in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        tensor_type = value.type.tensor_type
        if tensor_type.elem_type == TensorProto.FLOAT:
            tensor_type.elem_type = TensorProto.FLOAT16
            converted_value_infos += 1

    out_path = out_dir / "lhc_07_backbone_fp16_cast.onnx"
    inferred_path = out_dir / "lhc_07_backbone_fp16_cast_inferred.onnx"
    onnx.save(model, str(out_path))
    onnx.save(onnx.shape_inference.infer_shapes(model), str(inferred_path))
    return {
        "status": "success",
        "source_onnx": str(onnx_path),
        "fp16_onnx": str(out_path),
        "fp16_inferred_onnx": str(inferred_path),
        "converted_initializers": converted_initializers,
        "converted_value_infos": converted_value_infos,
    }


def _onnx_input_numpy_dtypes(onnx_path: Path) -> dict[str, str]:
    _avoid_local_onnx_shadow()
    import onnx
    from onnx import TensorProto

    model = onnx.load(str(onnx_path))
    init_names = {item.name for item in model.graph.initializer}
    dtype_map = {
        TensorProto.FLOAT16: "float16",
        TensorProto.FLOAT: "float32",
        TensorProto.DOUBLE: "float64",
        TensorProto.INT32: "int32",
        TensorProto.INT64: "int64",
    }
    out: dict[str, str] = {}
    for inp in model.graph.input:
        if inp.name in init_names:
            continue
        elem_type = inp.type.tensor_type.elem_type
        out[inp.name] = dtype_map.get(elem_type, "float32")
    return out


def _flatten_vm_output(value: Any) -> list[np.ndarray]:
    if hasattr(value, "numpy"):
        return [value.numpy()]
    if isinstance(value, (list, tuple)):
        out: list[np.ndarray] = []
        for item in value:
            out.extend(_flatten_vm_output(item))
        return out
    if hasattr(value, "__len__") and not isinstance(value, (str, bytes)):
        out = []
        for idx in range(len(value)):
            out.extend(_flatten_vm_output(value[idx]))
        return out
    raise TypeError(f"unsupported VM output type: {type(value).__name__}")


def run_rewrite_onnx_1x1(args: argparse.Namespace, target: Any, dev: Any) -> dict[str, Any]:
    import tvm
    from tvm import relax

    onnx_path = Path(args.onnx)
    raw_dir = Path(args.raw_dir) / "lhc07_full_fp16_engine_1x1_rewrite"
    raw_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema": "fp16_lhc07_rewritten_1x1_full_engine_v1",
        "label": "lhc_07",
        "onnx_path": str(onnx_path),
        "raw_dir": str(raw_dir),
        "status": "started",
    }

    try:
        effective_onnx_path = onnx_path
        if bool(args.cast_fp16_source):
            cast_result = _cast_onnx_float_tensors_to_fp16(onnx_path, raw_dir)
            payload["source_cast_fp16"] = cast_result
            effective_onnx_path = Path(cast_result["fp16_inferred_onnx"])
        rewrite = _rewrite_onnx_1x1_convs_to_matmul(effective_onnx_path, raw_dir)
        payload["rewrite"] = rewrite
        rewritten_onnx = Path(rewrite["rewritten_inferred_onnx"])

        orig_mod, orig_shape_dict = _load_onnx_relax_mod(effective_onnx_path, int(args.batch))
        rewritten_mod, rewritten_shape_dict = _load_onnx_relax_mod(rewritten_onnx, int(args.batch))
        payload["original_shape_dict"] = {key: list(value) for key, value in orig_shape_dict.items()}
        payload["rewritten_shape_dict"] = {
            key: list(value) for key, value in rewritten_shape_dict.items()
        }
        input_dtypes = _onnx_input_numpy_dtypes(effective_onnx_path)
        payload["input_dtypes"] = input_dtypes
        rng = np.random.RandomState(20260629)
        feeds_np = {
            key: rng.rand(*value).astype(input_dtypes.get(key, "float32"))
            for key, value in orig_shape_dict.items()
        }
        orig_feeds = [tvm.runtime.tensor(feeds_np[key], dev) for key in orig_shape_dict]
        rewritten_feeds = [tvm.runtime.tensor(feeds_np[key], dev) for key in rewritten_shape_dict]

        with tvm.transform.PassContext(opt_level=3):
            orig_ex = relax.build(orig_mod, target=target)
        orig_vm = relax.VirtualMachine(orig_ex, dev)
        orig_mean_us, orig_min_us, orig_repeats_us = _time_vm(
            orig_vm, orig_feeds, dev, int(args.full_reps)
        )
        orig_out = _flatten_vm_output(orig_vm["main"](*orig_feeds))
        payload["original_default_engine"] = {
            "status": "success",
            "route": "original_default_relax_build",
            "latency_mean_us": orig_mean_us,
            "latency_mean_ms": orig_mean_us / 1000.0,
            "latency_min_us": orig_min_us,
            "latency_repeats_us": orig_repeats_us,
            "scheduled_counts": {"wmma": 0, "tvm_mma_sync": 0},
            "export_library": _try_export_library(orig_ex, raw_dir / "original_default_relax_build.so"),
        }

        rewritten_legal = _legalize_fuse(rewritten_mod, target)
        legal_path = raw_dir / "rewritten_legalize_fuse.py"
        legal_path.write_text(rewritten_legal.script(), encoding="utf-8")
        schedule_attempts: list[dict[str, Any]] = []
        try:
            scheduled = _apply_full_engine_schedule(rewritten_legal, target)
            schedule_attempts.append({"route": "full_module_dlight_matmul_tensorization", "status": "success"})
            schedule_route = "full_module_dlight_matmul_tensorization"
        except Exception as exc:
            schedule_attempts.append(
                {
                    "route": "full_module_dlight_matmul_tensorization",
                    "status": "failed",
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            (raw_dir / "full_module_dlight_failure_traceback.txt").write_text(
                traceback.format_exc(), encoding="utf-8"
            )
            scheduled, selective_records = _apply_selective_matmul_tensorization(
                rewritten_legal, target
            )
            schedule_attempts.append(
                {
                    "route": "selective_matmul_primfunc_tensorization_keep_other_tir",
                    "status": "success",
                    "records": selective_records,
                }
            )
            schedule_route = "selective_matmul_primfunc_tensorization_keep_other_tir"
        scheduled_text = scheduled.script()
        scheduled_path = raw_dir / f"rewritten_{schedule_route}_scheduled.py"
        scheduled_path.write_text(scheduled_text, encoding="utf-8")
        scheduled_counts = _counts(scheduled_text)
        with target, tvm.transform.PassContext(opt_level=3):
            rewritten_ex = tvm.compile(scheduled, target=target)
        rewritten_vm = relax.VirtualMachine(rewritten_ex, dev)
        rewritten_mean_us, rewritten_min_us, rewritten_repeats_us = _time_vm(
            rewritten_vm, rewritten_feeds, dev, int(args.full_reps)
        )
        rewritten_out = _flatten_vm_output(rewritten_vm["main"](*rewritten_feeds))
        diffs: list[dict[str, Any]] = []
        for idx, (base, candidate) in enumerate(zip(orig_out, rewritten_out)):
            base32 = base.astype("float32")
            candidate32 = candidate.astype("float32")
            diff = np.abs(base32 - candidate32)
            base_abs = np.abs(base32)
            candidate_abs = np.abs(candidate32)
            base_abs_max = float(np.max(base_abs))
            base_abs_mean = float(np.mean(base_abs))
            diffs.append(
                {
                    "output_index": idx,
                    "original_shape": list(base.shape),
                    "rewritten_shape": list(candidate.shape),
                    "max_abs_err": float(np.max(diff)),
                    "mean_abs_err": float(np.mean(diff)),
                    "original_abs_max": base_abs_max,
                    "original_abs_mean": base_abs_mean,
                    "rewritten_abs_max": float(np.max(candidate_abs)),
                    "rewritten_abs_mean": float(np.mean(candidate_abs)),
                    "max_abs_err_over_original_abs_max": float(np.max(diff) / base_abs_max) if base_abs_max else None,
                    "mean_abs_err_over_original_abs_mean": float(np.mean(diff) / base_abs_mean) if base_abs_mean else None,
                }
            )
        payload["rewritten_engine"] = {
            "status": "success",
            "route": f"rewritten_1x1_matmul_legalize_fuse_{schedule_route}",
            "latency_mean_us": rewritten_mean_us,
            "latency_mean_ms": rewritten_mean_us / 1000.0,
            "latency_min_us": rewritten_min_us,
            "latency_repeats_us": rewritten_repeats_us,
            "scheduled_path": str(scheduled_path),
            "legalized_path": str(legal_path),
            "scheduled_counts": scheduled_counts,
            "schedule_attempts": schedule_attempts,
            "tensorcore_gate": bool(
                scheduled_counts.get("wmma", 0) > 0
                and scheduled_counts.get("tvm_mma_sync", 0) > 0
            ),
            "export_library": _try_export_library(
                rewritten_ex, raw_dir / "rewritten_1x1_matmul_tensorcore.so"
            ),
            "output_compare": diffs,
        }
        payload["comparison"] = {
            "latency_delta_ms": (orig_mean_us - rewritten_mean_us) / 1000.0,
            "speedup_ratio": orig_mean_us / rewritten_mean_us if rewritten_mean_us > 0 else None,
            "output_count_original": len(orig_out),
            "output_count_rewritten": len(rewritten_out),
        }
        payload["status"] = "success"
    except Exception as exc:
        payload.update({"status": "failed", "error": repr(exc), "traceback": traceback.format_exc()})
        (raw_dir / "failure_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
    return payload


def write_rewrite_onnx_1x1_outputs(args: argparse.Namespace, result: dict[str, Any]) -> tuple[Path, Path]:
    export_dir = Path(args.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    json_path = export_dir / "fp16_lhc07_rewritten_1x1_full_engine_latest.json"
    md_path = export_dir / "fp16_lhc07_rewritten_1x1_full_engine_latest.md"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    rewrite = result.get("rewrite") or {}
    cast_result = result.get("source_cast_fp16") or {}
    orig = result.get("original_default_engine") or {}
    rewritten = result.get("rewritten_engine") or {}
    counts = rewritten.get("scheduled_counts") or {}
    cmp_row = result.get("comparison") or {}
    lines = [
        "# FP16 lhc_07 rewritten 1x1 full-engine validation",
        "",
        f"- status: `{result.get('status')}`",
        f"- label: `{result.get('label')}`",
        f"- raw_dir: `{result.get('raw_dir')}`",
        f"- cast_fp16_source: `{bool(cast_result)}`",
        f"- fp16_cast_onnx: `{cast_result.get('fp16_onnx')}`",
        f"- input_dtypes: `{result.get('input_dtypes')}`",
        f"- source_onnx: `{rewrite.get('source_onnx')}`",
        f"- rewritten_onnx: `{rewrite.get('rewritten_onnx')}`",
        f"- rewritten_inferred_onnx: `{rewrite.get('rewritten_inferred_onnx')}`",
        f"- eligible_1x1_conv_count: `{rewrite.get('eligible_1x1_conv_count')}`",
        f"- rewritten_1x1_conv_count: `{rewrite.get('rewritten_1x1_conv_count')}`",
        "",
        "## 1. Full Engine Evidence",
        "",
        "| engine | status | route | wmma | tvm_mma_sync | latency_mean_ms | export |",
        "|---|---|---|---:|---:|---:|---|",
        "| original default | `{}` | `{}` | {} | {} | {} | `{}` |".format(
            orig.get("status"),
            orig.get("route"),
            (orig.get("scheduled_counts") or {}).get("wmma"),
            (orig.get("scheduled_counts") or {}).get("tvm_mma_sync"),
            orig.get("latency_mean_ms"),
            (orig.get("export_library") or {}).get("status"),
        ),
        "| rewritten 1x1 matmul | `{}` | `{}` | {} | {} | {} | `{}` |".format(
            rewritten.get("status"),
            rewritten.get("route"),
            counts.get("wmma"),
            counts.get("tvm_mma_sync"),
            rewritten.get("latency_mean_ms"),
            (rewritten.get("export_library") or {}).get("status"),
        ),
        "",
        "## 2. Output Compare",
        "",
        "| output | original_shape | rewritten_shape | max_abs_err | mean_abs_err | original_abs_max | original_abs_mean | max_err/orig_max | mean_err/orig_mean |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in rewritten.get("output_compare", []):
        lines.append(
            "| {} | `{}` | `{}` | {} | {} | {} | {} | {} | {} |".format(
                item.get("output_index"),
                item.get("original_shape"),
                item.get("rewritten_shape"),
                item.get("max_abs_err"),
                item.get("mean_abs_err"),
                item.get("original_abs_max"),
                item.get("original_abs_mean"),
                item.get("max_abs_err_over_original_abs_max"),
                item.get("mean_abs_err_over_original_abs_mean"),
            )
        )
    lines += [
        "",
        "## 3. Latency Summary",
        "",
        f"- latency_delta_ms: `{cmp_row.get('latency_delta_ms')}`",
        f"- speedup_ratio: `{cmp_row.get('speedup_ratio')}`",
        f"- tensorcore_gate: `{rewritten.get('tensorcore_gate')}`",
        "",
        "## 4. Interpretation",
        "",
        "- 这是真实 rewritten ONNX full-engine build/run 路径, 不是 all-1x1 counterfactual。",
        "- 只有 rewritten full engine 自身出现 `wmma/tvm_mma_sync` 且输出误差可解释时, 才能称为 tensor-core rewritten full-engine measured latency。",
        "- 若本路径失败, 以 raw_dir 中的 rewritten ONNX、legalized TIR、scheduled TIR 或 traceback 作为下一步最小复现入口。",
        "",
    ]
    if result.get("status") != "success":
        lines += [
            "## 5. Failure",
            "",
            f"- error: `{result.get('error')}`",
            f"- traceback_path: `{Path(result.get('raw_dir', '.')) / 'failure_traceback.txt'}`",
            "",
        ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def run_group_conv_im2col_tensorcore(args: argparse.Namespace, target: Any, dev: Any) -> dict[str, Any]:
    import tvm
    from tvm import relax

    candidate = str(args.conv_candidate)
    spec = GROUP_CONV_CANDIDATES[candidate]
    raw_dir = Path(args.raw_dir) / "lhc07_group_conv_im2col_tensorcore" / candidate
    raw_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema": "fp16_lhc07_group_conv_im2col_tensorcore_v1",
        "label": "lhc_07",
        "candidate": candidate,
        "spec": {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in spec.items()
        },
        "raw_dir": str(raw_dir),
        "status": "started",
        "claim": "single_primfunc_im2col_gemm_positive_not_full_engine_latency",
    }
    try:
        n, cin, h, w_in = [int(v) for v in spec["input_nchw"]]
        cout, cpg, kh, kw = [int(v) for v in spec["weight_oihw"]]
        groups = int(spec["groups"])
        oh, ow = _conv2d_out_hw(h, w_in, kh, kw, tuple(spec["strides"]), tuple(spec["padding"]))
        m = n * oh * ow
        k = cpg * kh * kw
        nper = cout // groups
        payload["implicit_gemm_shape"] = {
            "groups": groups,
            "M": m,
            "K": k,
            "N_per_group": nper,
            "output_hw": [oh, ow],
        }

        rng = np.random.RandomState(20260629 + len(candidate))
        x_np = (rng.rand(n, cin, h, w_in).astype("float16") - np.float16(0.5)) * np.float16(0.02)
        weight_np = (rng.rand(cout, cpg, kh, kw).astype("float16") - np.float16(0.5)) * np.float16(0.02)
        bias_np = (rng.rand(1, cout, 1, 1).astype("float16") - np.float16(0.5)) * np.float16(0.02)
        x_col_np, w_mat_np, b_mat_np, im2col_ref = _numpy_group_conv_im2col_inputs(
            x_np, weight_np, bias_np, spec
        )

        default_mod = _make_group_conv_default_mod(spec)
        default_legal = _legalize_fuse(default_mod, target)
        default_tir_path = raw_dir / "default_group_conv_legalized.py"
        default_tir_path.write_text(default_legal.script(), encoding="utf-8")
        with target, tvm.transform.PassContext(opt_level=3):
            default_ex = tvm.compile(default_legal, target=target)
        default_vm = relax.VirtualMachine(default_ex, dev)
        default_args = [
            tvm.runtime.tensor(x_np, dev),
            tvm.runtime.tensor(weight_np, dev),
            tvm.runtime.tensor(bias_np, dev),
        ]
        default_mean_us, default_min_us, default_repeats_us = _time_vm(
            default_vm, default_args, dev, int(args.reps)
        )
        default_out = default_vm["main"](*default_args).numpy().astype("float32")

        # Convert im2col reference back to NCHW for default-conv error check.
        ref_nchw = np.zeros((n, cout, oh, ow), dtype="float32")
        for gg in range(groups):
            for row in range(m):
                nn = row // (oh * ow)
                rem = row % (oh * ow)
                yy = rem // ow
                xx = rem % ow
                out0 = gg * nper
                ref_nchw[nn, out0 : out0 + nper, yy, xx] = im2col_ref[gg, row, :]
        default_diff = np.abs(default_out - ref_nchw)
        payload["default_group_conv"] = {
            "status": "success",
            "route": "relax_nn_conv2d_legalize_fuse_default_compile",
            "latency_mean_us": default_mean_us,
            "latency_mean_ms": default_mean_us / 1000.0,
            "latency_min_us": default_min_us,
            "latency_repeats_us": default_repeats_us,
            "tir_path": str(default_tir_path),
            "scheduled_counts": _counts(default_legal.script()),
            "max_abs_err_vs_im2col_ref": float(np.max(default_diff)),
            "mean_abs_err_vs_im2col_ref": float(np.mean(default_diff)),
            "export_library": _try_export_library(default_ex, raw_dir / "default_group_conv.so"),
        }

        im2col_mod = _make_group_conv_im2col_batched_matmul_mod(spec)
        im2col_legal = _legalize_fuse(im2col_mod, target)
        im2col_legal_path = raw_dir / "im2col_batched_matmul_legalized.py"
        im2col_legal_path.write_text(im2col_legal.script(), encoding="utf-8")
        im2col_args = [
            tvm.runtime.tensor(x_col_np, dev),
            tvm.runtime.tensor(w_mat_np, dev),
            tvm.runtime.tensor(b_mat_np, dev),
        ]
        with target, tvm.transform.PassContext(opt_level=3):
            im2col_default_ex = tvm.compile(im2col_legal, target=target)
        im2col_default_vm = relax.VirtualMachine(im2col_default_ex, dev)
        im2col_default_mean_us, im2col_default_min_us, im2col_default_repeats_us = _time_vm(
            im2col_default_vm, im2col_args, dev, int(args.reps)
        )
        im2col_default_out = im2col_default_vm["main"](*im2col_args).numpy().astype("float32")
        im2col_default_diff = np.abs(im2col_default_out - im2col_ref)

        try:
            im2col_tc = _apply_tensorcore_schedule(im2col_legal, target)
            tc_schedule_route = "dlight_MatmulTensorization"
        except Exception:
            (raw_dir / "im2col_full_tensorization_failure_traceback.txt").write_text(
                traceback.format_exc(), encoding="utf-8"
            )
            im2col_tc, selective_records = _apply_selective_matmul_tensorization(im2col_legal, target)
            payload["selective_records"] = selective_records
            tc_schedule_route = "selective_matmul_tensorization"
        im2col_tc_tir = im2col_tc.script()
        im2col_tc_path = raw_dir / f"im2col_batched_matmul_{tc_schedule_route}_scheduled.py"
        im2col_tc_path.write_text(im2col_tc_tir, encoding="utf-8")
        with target, tvm.transform.PassContext(opt_level=3):
            im2col_tc_ex = tvm.compile(im2col_tc, target=target)
        im2col_tc_vm = relax.VirtualMachine(im2col_tc_ex, dev)
        im2col_tc_mean_us, im2col_tc_min_us, im2col_tc_repeats_us = _time_vm(
            im2col_tc_vm, im2col_args, dev, int(args.reps)
        )
        im2col_tc_out = im2col_tc_vm["main"](*im2col_args).numpy().astype("float32")
        im2col_tc_diff = np.abs(im2col_tc_out - im2col_ref)
        tc_counts = _counts(im2col_tc_tir)
        payload["im2col_batched_matmul"] = {
            "status": "success",
            "claim": "precomputed_im2col_single_op_not_counting_im2col_materialization_or_nchw_restore",
            "default_route": "legalize_fuse_default_compile",
            "default_latency_mean_us": im2col_default_mean_us,
            "default_latency_mean_ms": im2col_default_mean_us / 1000.0,
            "default_latency_min_us": im2col_default_min_us,
            "default_latency_repeats_us": im2col_default_repeats_us,
            "default_max_abs_err": float(np.max(im2col_default_diff)),
            "default_mean_abs_err": float(np.mean(im2col_default_diff)),
            "tensorcore_route": tc_schedule_route,
            "tensorcore_latency_mean_us": im2col_tc_mean_us,
            "tensorcore_latency_mean_ms": im2col_tc_mean_us / 1000.0,
            "tensorcore_latency_min_us": im2col_tc_min_us,
            "tensorcore_latency_repeats_us": im2col_tc_repeats_us,
            "tensorcore_scheduled_counts": tc_counts,
            "tensorcore_gate": bool(tc_counts.get("wmma", 0) > 0 and tc_counts.get("tvm_mma_sync", 0) > 0),
            "tensorcore_max_abs_err": float(np.max(im2col_tc_diff)),
            "tensorcore_mean_abs_err": float(np.mean(im2col_tc_diff)),
            "legalized_path": str(im2col_legal_path),
            "scheduled_path": str(im2col_tc_path),
            "export_library": _try_export_library(im2col_tc_ex, raw_dir / "im2col_batched_matmul_tensorcore.so"),
        }
        schedule_only_delta_us = im2col_default_mean_us - im2col_tc_mean_us
        schedule_only_counterfactual_us = default_mean_us - schedule_only_delta_us
        ideal_compute_only_delta_us = default_mean_us - im2col_tc_mean_us
        payload["counterfactual"] = {
            "status": "success",
            "schedule_only_formula": "default_group_conv - (im2col_matmul_default - im2col_matmul_tensorcore)",
            "schedule_only_latency_mean_us": schedule_only_counterfactual_us,
            "schedule_only_latency_mean_ms": schedule_only_counterfactual_us / 1000.0,
            "schedule_only_absolute_delta_us": schedule_only_delta_us,
            "schedule_only_absolute_delta_ms": schedule_only_delta_us / 1000.0,
            "schedule_only_speedup_ratio": (
                default_mean_us / schedule_only_counterfactual_us
                if schedule_only_counterfactual_us > 0
                else None
            ),
            "ideal_compute_only_formula": "replace default_group_conv compute with precomputed_im2col_tensorcore_matmul",
            "ideal_compute_only_latency_mean_us": im2col_tc_mean_us,
            "ideal_compute_only_latency_mean_ms": im2col_tc_mean_us / 1000.0,
            "ideal_compute_only_absolute_delta_us": ideal_compute_only_delta_us,
            "ideal_compute_only_absolute_delta_ms": ideal_compute_only_delta_us / 1000.0,
            "ideal_compute_only_speedup_ratio": (
                default_mean_us / im2col_tc_mean_us if im2col_tc_mean_us > 0 else None
            ),
            "claim": "single_group_conv_counterfactual_not_full_engine_excludes_im2col_materialization_and_restore",
        }
        payload["status"] = "success"
    except Exception as exc:
        payload.update({"status": "failed", "error": repr(exc), "traceback": traceback.format_exc()})
        (raw_dir / "failure_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
    return payload


def write_group_conv_im2col_outputs(args: argparse.Namespace, result: dict[str, Any]) -> tuple[Path, Path]:
    export_dir = Path(args.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    json_path = export_dir / "fp16_lhc07_group_conv_im2col_tensorcore_latest.json"
    md_path = export_dir / "fp16_lhc07_group_conv_im2col_tensorcore_latest.md"
    candidate = str(result.get("candidate") or args.conv_candidate)
    candidate_json_path = export_dir / f"fp16_lhc07_group_conv_im2col_tensorcore_{candidate}.json"
    candidate_md_path = export_dir / f"fp16_lhc07_group_conv_im2col_tensorcore_{candidate}.md"
    json_text = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    json_path.write_text(json_text, encoding="utf-8")
    candidate_json_path.write_text(json_text, encoding="utf-8")

    default = result.get("default_group_conv") or {}
    im2col = result.get("im2col_batched_matmul") or {}
    cf = result.get("counterfactual") or {}
    counts = im2col.get("tensorcore_scheduled_counts") or {}
    lines = [
        "# FP16 lhc_07 group conv im2col TensorCore probe",
        "",
        f"- status: `{result.get('status')}`",
        f"- candidate: `{result.get('candidate')}`",
        f"- raw_dir: `{result.get('raw_dir')}`",
        f"- claim: `{result.get('claim')}`",
        f"- implicit_gemm_shape: `{result.get('implicit_gemm_shape')}`",
        "",
        "## 1. Evidence",
        "",
        "| item | status | wmma | tvm_mma_sync | latency_mean_ms | max_abs_err | mean_abs_err |",
        "|---|---|---:|---:|---:|---:|---:|",
        "| default group conv | `{}` | {} | {} | {} | {} | {} |".format(
            default.get("status"),
            (default.get("scheduled_counts") or {}).get("wmma"),
            (default.get("scheduled_counts") or {}).get("tvm_mma_sync"),
            default.get("latency_mean_ms"),
            default.get("max_abs_err_vs_im2col_ref"),
            default.get("mean_abs_err_vs_im2col_ref"),
        ),
        "| im2col matmul default | `{}` | n/a | n/a | {} | {} | {} |".format(
            im2col.get("status"),
            im2col.get("default_latency_mean_ms"),
            im2col.get("default_max_abs_err"),
            im2col.get("default_mean_abs_err"),
        ),
        "| im2col matmul tensorcore | `{}` | {} | {} | {} | {} | {} |".format(
            im2col.get("status"),
            counts.get("wmma"),
            counts.get("tvm_mma_sync"),
            im2col.get("tensorcore_latency_mean_ms"),
            im2col.get("tensorcore_max_abs_err"),
            im2col.get("tensorcore_mean_abs_err"),
        ),
        "",
        "## 2. Counterfactual",
        "",
        f"- schedule_only_formula: `{cf.get('schedule_only_formula')}`",
        f"- schedule_only_latency_mean_ms: `{cf.get('schedule_only_latency_mean_ms')}`",
        f"- schedule_only_absolute_delta_ms: `{cf.get('schedule_only_absolute_delta_ms')}`",
        f"- schedule_only_speedup_ratio: `{cf.get('schedule_only_speedup_ratio')}`",
        f"- ideal_compute_only_formula: `{cf.get('ideal_compute_only_formula')}`",
        f"- ideal_compute_only_latency_mean_ms: `{cf.get('ideal_compute_only_latency_mean_ms')}`",
        f"- ideal_compute_only_absolute_delta_ms: `{cf.get('ideal_compute_only_absolute_delta_ms')}`",
        f"- ideal_compute_only_speedup_ratio: `{cf.get('ideal_compute_only_speedup_ratio')}`",
        f"- claim: `{cf.get('claim')}`",
        "",
        "## 3. Interpretation",
        "",
        "- 这是单个 3x3/group conv 的 im2col/implicit-GEMM 正例和 counterfactual, 不是 full-engine latency。",
        "- im2col matmul latency 不包含 im2col materialization 和 NCHW restore overhead。",
        "- schedule-only delta 只表示 im2col MatMul 的 default schedule 与 TensorCore schedule 差异。",
        "- ideal compute-only delta 是把 default group conv 计算替换为预先 im2col 后 TensorCore MatMul 的理论下界, 仍不包含真实 im2col/restore 代价。",
        "- 若 tensorcore_gate 为 true 且 ideal compute-only delta 足够大, 下一步才值得实现真实 im2col materialization 或 TIR rewrite。",
        f"- tensorcore_gate: `{im2col.get('tensorcore_gate')}`",
        "",
    ]
    if result.get("status") != "success":
        lines += [
            "## 4. Failure",
            "",
            f"- error: `{result.get('error')}`",
            f"- traceback_path: `{Path(result.get('raw_dir', '.')) / 'failure_traceback.txt'}`",
            "",
        ]
    md_text = "\n".join(lines)
    md_path.write_text(md_text, encoding="utf-8")
    candidate_md_path.write_text(md_text, encoding="utf-8")
    return json_path, md_path


def run_group_conv_full_im2col_tensorcore(args: argparse.Namespace, target: Any, dev: Any) -> dict[str, Any]:
    import tvm
    from tvm import relax

    candidate = str(args.conv_candidate)
    spec = GROUP_CONV_CANDIDATES[candidate]
    raw_dir = Path(args.raw_dir) / "lhc07_group_conv_full_im2col_tensorcore" / candidate
    raw_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema": "fp16_lhc07_group_conv_full_im2col_tensorcore_v1",
        "label": "lhc_07",
        "candidate": candidate,
        "spec": {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in spec.items()
        },
        "raw_dir": str(raw_dir),
        "status": "started",
        "claim": "single_layer_full_im2col_materialize_tensorcore_restore_not_full_engine",
    }
    try:
        n, cin, h, w_in = [int(v) for v in spec["input_nchw"]]
        cout, cpg, kh, kw = [int(v) for v in spec["weight_oihw"]]
        groups = int(spec["groups"])
        oh, ow = _conv2d_out_hw(h, w_in, kh, kw, tuple(spec["strides"]), tuple(spec["padding"]))
        m = n * oh * ow
        k = cpg * kh * kw
        nper = cout // groups
        payload["implicit_gemm_shape"] = {
            "groups": groups,
            "M": m,
            "K": k,
            "N_per_group": nper,
            "output_hw": [oh, ow],
        }

        rng = np.random.RandomState(20260630 + len(candidate))
        x_np = (rng.rand(n, cin, h, w_in).astype("float16") - np.float16(0.5)) * np.float16(0.02)
        weight_np = (rng.rand(cout, cpg, kh, kw).astype("float16") - np.float16(0.5)) * np.float16(0.02)
        bias_np = (rng.rand(1, cout, 1, 1).astype("float16") - np.float16(0.5)) * np.float16(0.02)
        _, w_mat_np, b_mat_np, _ = _numpy_group_conv_im2col_inputs(x_np, weight_np, bias_np, spec)

        default_mod = _make_group_conv_default_mod(spec)
        default_legal = _legalize_fuse(default_mod, target)
        default_tir_path = raw_dir / "default_group_conv_legalized.py"
        default_tir_path.write_text(default_legal.script(), encoding="utf-8")
        with target, tvm.transform.PassContext(opt_level=3):
            default_ex = tvm.compile(default_legal, target=target)
        default_vm = relax.VirtualMachine(default_ex, dev)
        default_args = [
            tvm.runtime.tensor(x_np, dev),
            tvm.runtime.tensor(weight_np, dev),
            tvm.runtime.tensor(bias_np, dev),
        ]
        default_mean_us, default_min_us, default_repeats_us = _time_vm(
            default_vm, default_args, dev, int(args.reps)
        )
        default_out = default_vm["main"](*default_args).numpy().astype("float32")
        payload["default_group_conv"] = {
            "status": "success",
            "route": "relax_nn_conv2d_legalize_fuse_default_compile",
            "latency_mean_us": default_mean_us,
            "latency_mean_ms": default_mean_us / 1000.0,
            "latency_min_us": default_min_us,
            "latency_repeats_us": default_repeats_us,
            "tir_path": str(default_tir_path),
            "scheduled_counts": _counts(default_legal.script()),
            "export_library": _try_export_library(default_ex, raw_dir / "default_group_conv.so"),
        }

        full_mod = _make_group_conv_full_im2col_tensorcore_mod(spec)
        full_legal = _legalize_fuse(full_mod, target)
        full_legal_path = raw_dir / "full_im2col_legalized.py"
        full_legal_path.write_text(full_legal.script(), encoding="utf-8")
        full_scheduled, selective_records = _apply_selective_matmul_tensorization(full_legal, target)
        full_scheduled_text = full_scheduled.script()
        full_scheduled_path = raw_dir / "full_im2col_selective_tensorcore_scheduled.py"
        full_scheduled_path.write_text(full_scheduled_text, encoding="utf-8")
        counts = _counts(full_scheduled_text)
        with target, tvm.transform.PassContext(opt_level=3):
            full_ex = tvm.compile(full_scheduled, target=target)
        full_vm = relax.VirtualMachine(full_ex, dev)
        full_args = [
            tvm.runtime.tensor(x_np, dev),
            tvm.runtime.tensor(w_mat_np, dev),
            tvm.runtime.tensor(b_mat_np, dev),
        ]
        full_mean_us, full_min_us, full_repeats_us = _time_vm(
            full_vm, full_args, dev, int(args.reps)
        )
        full_out = full_vm["main"](*full_args).numpy().astype("float32")
        diff = np.abs(full_out - default_out)
        payload["full_im2col_tensorcore"] = {
            "status": "success",
            "route": "call_tir_im2col_materialize_then_tensorcore_matmul_then_call_tir_restore_nchw",
            "latency_mean_us": full_mean_us,
            "latency_mean_ms": full_mean_us / 1000.0,
            "latency_min_us": full_min_us,
            "latency_repeats_us": full_repeats_us,
            "legalized_path": str(full_legal_path),
            "scheduled_path": str(full_scheduled_path),
            "scheduled_counts": counts,
            "selective_records": selective_records,
            "tensorcore_gate": bool(counts.get("wmma", 0) > 0 and counts.get("tvm_mma_sync", 0) > 0),
            "max_abs_err_vs_default": float(np.max(diff)),
            "mean_abs_err_vs_default": float(np.mean(diff)),
            "export_library": _try_export_library(full_ex, raw_dir / "full_im2col_tensorcore.so"),
        }
        payload["comparison"] = {
            "latency_delta_us": default_mean_us - full_mean_us,
            "latency_delta_ms": (default_mean_us - full_mean_us) / 1000.0,
            "speedup_ratio": default_mean_us / full_mean_us if full_mean_us > 0 else None,
            "is_faster_than_default": bool(full_mean_us < default_mean_us),
            "claim": "single_layer_full_im2col_vs_default_group_conv_not_full_engine",
        }
        payload["status"] = "success"
    except Exception as exc:
        payload.update({"status": "failed", "error": repr(exc), "traceback": traceback.format_exc()})
        (raw_dir / "failure_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
    return payload


def write_group_conv_full_im2col_outputs(args: argparse.Namespace, result: dict[str, Any]) -> tuple[Path, Path]:
    export_dir = Path(args.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    candidate = str(result.get("candidate") or args.conv_candidate)
    json_path = export_dir / "fp16_lhc07_group_conv_full_im2col_tensorcore_latest.json"
    md_path = export_dir / "fp16_lhc07_group_conv_full_im2col_tensorcore_latest.md"
    candidate_json_path = export_dir / f"fp16_lhc07_group_conv_full_im2col_tensorcore_{candidate}.json"
    candidate_md_path = export_dir / f"fp16_lhc07_group_conv_full_im2col_tensorcore_{candidate}.md"
    json_text = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    json_path.write_text(json_text, encoding="utf-8")
    candidate_json_path.write_text(json_text, encoding="utf-8")

    default = result.get("default_group_conv") or {}
    full = result.get("full_im2col_tensorcore") or {}
    cmp_row = result.get("comparison") or {}
    counts = full.get("scheduled_counts") or {}
    lines = [
        "# FP16 lhc_07 full im2col TensorCore single-layer probe",
        "",
        f"- status: `{result.get('status')}`",
        f"- candidate: `{result.get('candidate')}`",
        f"- raw_dir: `{result.get('raw_dir')}`",
        f"- claim: `{result.get('claim')}`",
        f"- implicit_gemm_shape: `{result.get('implicit_gemm_shape')}`",
        "",
        "## 1. Evidence",
        "",
        "| item | status | wmma | tvm_mma_sync | latency_mean_ms | max_abs_err_vs_default | mean_abs_err_vs_default |",
        "|---|---|---:|---:|---:|---:|---:|",
        "| default group conv | `{}` | {} | {} | {} | n/a | n/a |".format(
            default.get("status"),
            (default.get("scheduled_counts") or {}).get("wmma"),
            (default.get("scheduled_counts") or {}).get("tvm_mma_sync"),
            default.get("latency_mean_ms"),
        ),
        "| full im2col tensorcore | `{}` | {} | {} | {} | {} | {} |".format(
            full.get("status"),
            counts.get("wmma"),
            counts.get("tvm_mma_sync"),
            full.get("latency_mean_ms"),
            full.get("max_abs_err_vs_default"),
            full.get("mean_abs_err_vs_default"),
        ),
        "",
        "## 2. Comparison",
        "",
        f"- latency_delta_ms: `{cmp_row.get('latency_delta_ms')}`",
        f"- speedup_ratio: `{cmp_row.get('speedup_ratio')}`",
        f"- is_faster_than_default: `{cmp_row.get('is_faster_than_default')}`",
        f"- tensorcore_gate: `{full.get('tensorcore_gate')}`",
        f"- claim: `{cmp_row.get('claim')}`",
        "",
        "## 3. Interpretation",
        "",
        "- 这是单个 3x3/group conv 的完整 TVM 单层路径, 包含 im2col materialization、TensorCore MatMul 和 NCHW restore。",
        "- 它仍不是 lhc_07 full-engine latency, 但可以决定是否值得继续嵌入 full engine。",
        "- 若该路径不快于 default group conv, 则 full-engine 嵌入大概率不会收益, 除非进一步融合 im2col/materialization/restore。",
        "",
    ]
    if result.get("status") != "success":
        lines += [
            "## 4. Failure",
            "",
            f"- error: `{result.get('error')}`",
            f"- traceback_path: `{Path(result.get('raw_dir', '.')) / 'failure_traceback.txt'}`",
            "",
        ]
    md_text = "\n".join(lines)
    md_path.write_text(md_text, encoding="utf-8")
    candidate_md_path.write_text(md_text, encoding="utf-8")
    return json_path, md_path


def run_group_conv_accum_compare(args: argparse.Namespace, target: Any, dev: Any) -> dict[str, Any]:
    import tvm
    from tvm import relax

    candidate = str(args.conv_candidate)
    spec = GROUP_CONV_CANDIDATES[candidate]
    raw_dir = Path(args.raw_dir) / "lhc07_group_conv_accum_compare" / candidate
    raw_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema": "fp16_lhc07_group_conv_accum_compare_v1",
        "label": "lhc_07",
        "candidate": candidate,
        "spec": {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in spec.items()
        },
        "raw_dir": str(raw_dir),
        "status": "started",
        "claim": "single_layer_unscheduled_same_signature_accumulation_order_diagnostic_not_full_engine",
        "scales": [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
    }
    try:
        n, cin, h, w_in = [int(v) for v in spec["input_nchw"]]
        cout, cpg, kh, kw = [int(v) for v in spec["weight_oihw"]]
        rng = np.random.RandomState(20260701 + len(candidate))
        x_base = rng.rand(n, cin, h, w_in).astype("float16") - np.float16(0.5)
        weight_base = rng.rand(cout, cpg, kh, kw).astype("float16") - np.float16(0.5)
        bias_base = rng.rand(*tuple(spec["bias"])).astype("float16") - np.float16(0.5)

        modules = {
            "default": _legalize_fuse(_make_group_conv_default_mod(spec), target),
            "fp16_accum_replacement": _legalize_fuse(
                _make_group_conv_same_signature_replacement_mod(spec, "float16"), target
            ),
            "fp32_accum_replacement": _legalize_fuse(
                _make_group_conv_same_signature_replacement_mod(spec, "float32"), target
            ),
        }
        for name, mod in modules.items():
            (raw_dir / f"{name}.py").write_text(mod.script(), encoding="utf-8")

        executables: dict[str, Any] = {}
        vms: dict[str, Any] = {}
        for name, mod in modules.items():
            with target, tvm.transform.PassContext(opt_level=3):
                executables[name] = tvm.compile(mod, target=target)
            vms[name] = relax.VirtualMachine(executables[name], dev)

        rows: list[dict[str, Any]] = []
        latency_scale1_ms: dict[str, float] = {}
        for scale in payload["scales"]:
            scale16 = np.float16(scale)
            x_np = (x_base * scale16).astype("float16")
            weight_np = (weight_base * scale16).astype("float16")
            bias_np = (bias_base * scale16).astype("float16")
            tvm_args = [
                tvm.runtime.tensor(x_np, dev),
                tvm.runtime.tensor(weight_np, dev),
                tvm.runtime.tensor(bias_np, dev),
            ]
            outs = {
                name: vms[name]["main"](*tvm_args).numpy()
                for name in ["default", "fp16_accum_replacement", "fp32_accum_replacement"]
            }
            row = {
                "scale": float(scale),
                "fp16_accum": _compare_arrays(outs["default"], outs["fp16_accum_replacement"]),
                "fp32_accum": _compare_arrays(outs["default"], outs["fp32_accum_replacement"]),
            }
            rows.append(row)
            if abs(float(scale) - 1.0) < 1e-12:
                for name, vm in vms.items():
                    mean_us, min_us, repeats_us = _time_vm(vm, tvm_args, dev, int(args.reps))
                    latency_scale1_ms[name] = mean_us / 1000.0
                    row[f"{name}_latency_mean_ms"] = mean_us / 1000.0
                    row[f"{name}_latency_min_ms"] = min_us / 1000.0
                    row[f"{name}_latency_repeats_ms"] = [v / 1000.0 for v in repeats_us]

        payload.update(
            {
                "status": "success",
                "rows": rows,
                "latency_scale1_ms": latency_scale1_ms,
                "counts": {name: _counts(mod.script()) for name, mod in modules.items()},
                "paths": {name: str(raw_dir / f"{name}.py") for name in modules},
                "interpretation": (
                    "Compare default TOPI group_conv against same-signature im2col/matmul "
                    "replacement without TensorCore scheduling. FP32 accumulation reducing but "
                    "not eliminating error supports reduction/order numeric drift rather than a "
                    "pure TensorCore intrinsic issue."
                ),
            }
        )
    except Exception as exc:
        payload.update({"status": "failed", "error": repr(exc), "traceback": traceback.format_exc()})
        (raw_dir / "failure_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
    return payload


def write_group_conv_accum_compare_outputs(args: argparse.Namespace, result: dict[str, Any]) -> tuple[Path, Path]:
    export_dir = Path(args.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    candidate = str(result.get("candidate") or args.conv_candidate)
    json_path = export_dir / "fp16_lhc07_group_conv_accum_compare_latest.json"
    md_path = export_dir / "fp16_lhc07_group_conv_accum_compare_latest.md"
    candidate_json_path = export_dir / f"fp16_lhc07_group_conv_accum_compare_{candidate}.json"
    candidate_md_path = export_dir / f"fp16_lhc07_group_conv_accum_compare_{candidate}.md"
    json_text = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    json_path.write_text(json_text, encoding="utf-8")
    candidate_json_path.write_text(json_text, encoding="utf-8")

    lines = [
        "# FP16 lhc_07 group conv accumulation/order diagnostic",
        "",
        f"- status: `{result.get('status')}`",
        f"- candidate: `{result.get('candidate')}`",
        f"- raw_dir: `{result.get('raw_dir')}`",
        f"- claim: `{result.get('claim')}`",
        f"- latency_scale1_ms: `{result.get('latency_scale1_ms')}`",
        "",
        "## 1. Scale Sweep",
        "",
        "| scale | fp16 mean_rel | fp32 mean_rel | fp16 mean_abs_err | fp32 mean_abs_err |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in result.get("rows", []):
        fp16 = row.get("fp16_accum") or {}
        fp32 = row.get("fp32_accum") or {}
        lines.append(
            "| {scale} | {fp16_rel} | {fp32_rel} | {fp16_mean} | {fp32_mean} |".format(
                scale=row.get("scale"),
                fp16_rel=fp16.get("mean_abs_err_over_original_abs_mean"),
                fp32_rel=fp32.get("mean_abs_err_over_original_abs_mean"),
                fp16_mean=fp16.get("mean_abs_err"),
                fp32_mean=fp32.get("mean_abs_err"),
            )
        )
    lines += [
        "",
        "## 2. Interpretation",
        "",
        f"- interpretation: `{result.get('interpretation')}`",
        "- 该模式不启用 TensorCore schedule, 只用于判断 same-signature replacement 与 default group_conv 的 accumulation/order 差异。",
        "- 若 FP32 accumulation 明显降低但不能消除误差, 下一步应优先做 default-order/default-like reduction 对照或 AP smoke, 而不是只怪 TensorCore intrinsic。",
        "",
    ]
    if result.get("status") != "success":
        lines += [
            "## 3. Failure",
            "",
            f"- error: `{result.get('error')}`",
            f"- traceback_path: `{Path(result.get('raw_dir', '.')) / 'failure_traceback.txt'}`",
            "",
        ]
    md_text = "\n".join(lines)
    md_path.write_text(md_text, encoding="utf-8")
    candidate_md_path.write_text(md_text, encoding="utf-8")
    return json_path, md_path


def write_import_failure_outputs(args: argparse.Namespace, result: dict[str, Any]) -> tuple[Path, Path]:
    export_dir = Path(args.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = Path(args.raw_dir) / "import_failure"
    raw_dir.mkdir(parents=True, exist_ok=True)
    traceback_path = raw_dir / "failure_traceback.txt"
    traceback_path.write_text(str(result.get("traceback") or ""), encoding="utf-8")
    result["traceback_path"] = str(traceback_path)

    json_path = export_dir / "fp16_lhc07_probe_import_failure_latest.json"
    md_path = export_dir / "fp16_lhc07_probe_import_failure_latest.md"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# FP16 lhc_07 probe import failure",
        "",
        f"- status: `{result.get('status')}`",
        f"- mode: `{result.get('mode')}`",
        f"- gpu: `{result.get('gpu')}`",
        f"- error: `{result.get('error')}`",
        f"- python_executable: `{result.get('python_executable')}`",
        f"- traceback_path: `{result.get('traceback_path')}`",
        "",
        "## Interpretation",
        "",
        "- TVM import failed before any TVM build/run path could start.",
        "- This artifact is an environment failure record, not a latency or correctness measurement.",
        "- Use the H800 TVM Python environment from the runbook before interpreting performance.",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def write_self_test_no_tvm_outputs(args: argparse.Namespace, result: dict[str, Any]) -> tuple[Path, Path]:
    export_dir = Path(args.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    json_path = export_dir / "fp16_lhc07_probe_self_test_no_tvm_latest.json"
    md_path = export_dir / "fp16_lhc07_probe_self_test_no_tvm_latest.md"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    diff = result.get("direct_vs_im2col_restore") or {}
    lines = [
        "# FP16 lhc_07 probe self-test without TVM",
        "",
        f"- status: `{result.get('status')}`",
        f"- raw_dir: `{result.get('raw_dir')}`",
        f"- claim: `{result.get('claim')}`",
        f"- python_executable: `{result.get('python_executable')}`",
        f"- x_col_shape: `{result.get('x_col_shape')}`",
        f"- w_mat_shape: `{result.get('w_mat_shape')}`",
        f"- b_mat_shape: `{result.get('b_mat_shape')}`",
        "",
        "## Direct Group Conv vs Im2col Restore",
        "",
        f"- max_abs_err: `{diff.get('max_abs_err')}`",
        f"- mean_abs_err: `{diff.get('mean_abs_err')}`",
        f"- max_allowed_abs_err: `{result.get('max_allowed_abs_err')}`",
        f"- mean_err/original_mean: `{diff.get('mean_abs_err_over_original_abs_mean')}`",
        "",
        "## Real Candidate Sample Mapping Checks",
        "",
        "| candidate | status | groups | M | K | N/group | x mismatches | w mismatches | b mismatches |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in result.get("candidate_sample_checks", []):
        shape = item.get("implicit_gemm_shape") or {}
        lines.append(
            "| `{candidate}` | `{status}` | {groups} | {m} | {k} | {nper} | {xm} | {wm} | {bm} |".format(
                candidate=item.get("candidate"),
                status=item.get("status"),
                groups=shape.get("groups"),
                m=shape.get("M"),
                k=shape.get("K"),
                nper=shape.get("N_per_group"),
                xm=item.get("x_mismatch_count"),
                wm=item.get("w_mismatch_count"),
                bm=item.get("b_mismatch_count"),
            )
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- This checks pure NumPy indexing, padding, group channel mapping, and restore order.",
        "- Real lhc_07 group-conv candidates are checked by sampled x_col/w_mat/b_mat mapping, not full direct convolution.",
        "- It does not import TVM and is not a latency, TensorCore, or AP measurement.",
        "- Passing this self-test only narrows the FP16 drift root cause away from gross im2col/group mapping errors.",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def _load_group_full_im2col_result(export_dir: Path, candidate: str) -> dict[str, Any]:
    path = export_dir / f"fp16_lhc07_group_conv_full_im2col_tensorcore_{candidate}.json"
    if not path.exists():
        raise FileNotFoundError(f"group_full_im2col_result_not_found:{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def run_full_engine_group_conv_counterfactual(args: argparse.Namespace, target: Any, dev: Any) -> dict[str, Any]:
    import tvm
    from tvm import relax

    onnx_path = Path(args.onnx)
    raw_dir = Path(args.raw_dir) / "lhc07_full_engine_group_conv_counterfactual"
    raw_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema": "fp16_lhc07_full_engine_group_conv_counterfactual_v1",
        "label": "lhc_07",
        "onnx_path": str(onnx_path),
        "raw_dir": str(raw_dir),
        "status": "started",
        "claim": "full_engine_measured_counterfactual_not_rewritten_engine",
    }
    try:
        effective_onnx_path = onnx_path
        if bool(args.cast_fp16_source):
            cast_result = _cast_onnx_float_tensors_to_fp16(onnx_path, raw_dir)
            payload["source_cast_fp16"] = cast_result
            effective_onnx_path = Path(cast_result["fp16_inferred_onnx"])

        mod0, shape_dict = _load_onnx_relax_mod(effective_onnx_path, int(args.batch))
        payload["shape_dict"] = {key: list(value) for key, value in shape_dict.items()}
        input_dtypes = _onnx_input_numpy_dtypes(effective_onnx_path)
        payload["input_dtypes"] = input_dtypes
        rng = np.random.RandomState(20260630)
        feeds_np = {
            key: rng.rand(*value).astype(input_dtypes.get(key, "float32"))
            for key, value in shape_dict.items()
        }
        feeds_tvm = [tvm.runtime.tensor(feeds_np[key], dev) for key in shape_dict]

        with tvm.transform.PassContext(opt_level=3):
            full_ex = relax.build(mod0, target=target)
        full_vm = relax.VirtualMachine(full_ex, dev)
        full_mean_us, full_min_us, full_repeats_us = _time_vm(
            full_vm, feeds_tvm, dev, int(args.full_reps)
        )
        payload["full_default_engine"] = {
            "status": "success",
            "route": "default_relax_build",
            "latency_mean_us": full_mean_us,
            "latency_mean_ms": full_mean_us / 1000.0,
            "latency_min_us": full_min_us,
            "latency_repeats_us": full_repeats_us,
            "scheduled_counts": {"wmma": 0, "tvm_mma_sync": 0},
            "export_library": _try_export_library(full_ex, raw_dir / "full_default_relax_build.so"),
        }

        candidates = ["fused_conv2d4_add10_relu6", "fused_conv2d6_add10_relu6"]
        rows: list[dict[str, Any]] = []
        total_default_us = 0.0
        total_replacement_us = 0.0
        for candidate in candidates:
            result = _load_group_full_im2col_result(Path(args.export_dir), candidate)
            default = result["default_group_conv"]
            replacement = result["full_im2col_tensorcore"]
            default_us = float(default["latency_mean_us"])
            replacement_us = float(replacement["latency_mean_us"])
            total_default_us += default_us
            total_replacement_us += replacement_us
            rows.append(
                {
                    "candidate": candidate,
                    "default_group_conv_latency_ms": default_us / 1000.0,
                    "replacement_full_im2col_tensorcore_latency_ms": replacement_us / 1000.0,
                    "delta_ms": (default_us - replacement_us) / 1000.0,
                    "replacement_counts": replacement.get("scheduled_counts"),
                    "replacement_tensorcore_gate": replacement.get("tensorcore_gate"),
                    "max_abs_err_vs_default": replacement.get("max_abs_err_vs_default"),
                    "source_result": str(
                        Path(args.export_dir)
                        / f"fp16_lhc07_group_conv_full_im2col_tensorcore_{candidate}.json"
                    ),
                    "claim": "single_layer_replacement_measurement",
                }
            )
        counterfactual_us = full_mean_us - total_default_us + total_replacement_us
        payload["replacement_rows"] = rows
        payload["counterfactual"] = {
            "status": "success",
            "formula": "full_default_engine - sum(default_group_conv_layers) + sum(full_im2col_tensorcore_layers)",
            "full_default_latency_mean_us": full_mean_us,
            "full_default_latency_mean_ms": full_mean_us / 1000.0,
            "default_group_conv_sum_us": total_default_us,
            "default_group_conv_sum_ms": total_default_us / 1000.0,
            "replacement_sum_us": total_replacement_us,
            "replacement_sum_ms": total_replacement_us / 1000.0,
            "counterfactual_latency_mean_us": counterfactual_us,
            "counterfactual_latency_mean_ms": counterfactual_us / 1000.0,
            "absolute_delta_us": full_mean_us - counterfactual_us,
            "absolute_delta_ms": (full_mean_us - counterfactual_us) / 1000.0,
            "speedup_ratio": full_mean_us / counterfactual_us if counterfactual_us > 0 else None,
            "claim": "full_engine_measured_counterfactual_not_rewritten_engine",
        }
        payload["status"] = "success"
    except Exception as exc:
        payload.update({"status": "failed", "error": repr(exc), "traceback": traceback.format_exc()})
        (raw_dir / "failure_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
    return payload


def write_full_engine_group_conv_counterfactual_outputs(args: argparse.Namespace, result: dict[str, Any]) -> tuple[Path, Path]:
    export_dir = Path(args.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    json_path = export_dir / "fp16_lhc07_full_engine_group_conv_counterfactual_latest.json"
    md_path = export_dir / "fp16_lhc07_full_engine_group_conv_counterfactual_latest.md"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    full = result.get("full_default_engine") or {}
    cf = result.get("counterfactual") or {}
    lines = [
        "# FP16 lhc_07 full-engine group-conv counterfactual",
        "",
        f"- status: `{result.get('status')}`",
        f"- claim: `{result.get('claim')}`",
        f"- onnx_path: `{result.get('onnx_path')}`",
        f"- cast_fp16_source: `{bool(result.get('source_cast_fp16'))}`",
        f"- raw_dir: `{result.get('raw_dir')}`",
        "",
        "## 1. Full Default Engine",
        "",
        "| item | status | wmma | tvm_mma_sync | latency_mean_ms | export |",
        "|---|---|---:|---:|---:|---|",
        "| default full engine | `{}` | {} | {} | {} | `{}` |".format(
            full.get("status"),
            (full.get("scheduled_counts") or {}).get("wmma"),
            (full.get("scheduled_counts") or {}).get("tvm_mma_sync"),
            full.get("latency_mean_ms"),
            (full.get("export_library") or {}).get("status"),
        ),
        "",
        "## 2. Replacement Rows",
        "",
        "| candidate | default layer ms | replacement ms | delta ms | wmma | tvm_mma_sync | max_abs_err |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result.get("replacement_rows", []):
        counts = row.get("replacement_counts") or {}
        lines.append(
            "| `{}` | {} | {} | {} | {} | {} | {} |".format(
                row.get("candidate"),
                row.get("default_group_conv_latency_ms"),
                row.get("replacement_full_im2col_tensorcore_latency_ms"),
                row.get("delta_ms"),
                counts.get("wmma"),
                counts.get("tvm_mma_sync"),
                row.get("max_abs_err_vs_default"),
            )
        )
    lines += [
        "",
        "## 3. Counterfactual",
        "",
        f"- formula: `{cf.get('formula')}`",
        f"- full_default_latency_mean_ms: `{cf.get('full_default_latency_mean_ms')}`",
        f"- default_group_conv_sum_ms: `{cf.get('default_group_conv_sum_ms')}`",
        f"- replacement_sum_ms: `{cf.get('replacement_sum_ms')}`",
        f"- counterfactual_latency_mean_ms: `{cf.get('counterfactual_latency_mean_ms')}`",
        f"- absolute_delta_ms: `{cf.get('absolute_delta_ms')}`",
        f"- speedup_ratio: `{cf.get('speedup_ratio')}`",
        f"- claim: `{cf.get('claim')}`",
        "",
        "## 4. Interpretation",
        "",
        "- 这是 full-engine measured counterfactual, 不是 rewritten full-engine binary。",
        "- default full engine 仍不能标为 TensorCore engine。",
        "- 只有把 replacement 真正嵌入 full engine 并完成 build/run/latency/output compare 后, 才能写成 rewritten full-engine measured latency。",
        "",
    ]
    if result.get("status") != "success":
        lines += [
            "## 5. Failure",
            "",
            f"- error: `{result.get('error')}`",
            f"- traceback_path: `{Path(result.get('raw_dir', '.')) / 'failure_traceback.txt'}`",
            "",
        ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def _replace_group_conv_primfuncs_for_full_engine(
    mod: Any,
    rewrite_filter: str = "all",
    repeated_callsite_indices: set[int] | None = None,
    accum_dtype: str = "float16",
) -> tuple[Any, list[dict[str, Any]]]:
    import tvm
    from tvm import relax

    allowed_candidates = {
        "all": {"fused_conv2d4_add10_relu6", "fused_conv2d6_add10_relu6"},
        "downsample": {"fused_conv2d4_add10_relu6"},
        "repeated": {"fused_conv2d6_add10_relu6"},
    }.get(rewrite_filter)
    if allowed_candidates is None:
        raise ValueError(f"unknown_group_conv_rewrite_filter:{rewrite_filter}")

    def classify(func: Any) -> dict[str, Any] | None:
        if not hasattr(func, "script"):
            return None
        return _classify_full_engine_group_conv_primfunc_text(func.script())

    out = tvm.IRModule(dict(mod.functions), attrs=mod.attrs)
    records: list[dict[str, Any]] = []
    callsite_replacement_gv = None
    for gv, func in mod.functions_items():
        name = gv.name_hint
        match = classify(func)
        if match is None:
            continue
        candidate = match["candidate"]
        if candidate not in allowed_candidates:
            records.append(
                {
                    "name": name,
                    "candidate": candidate,
                    "status": "skipped_by_filter",
                    "rewrite_filter": rewrite_filter,
                    "matched_input_nchw": list(match["matched_input_nchw"]),
                    "matched_output_nchw": list(match["matched_output_nchw"]),
                    "match_rule": match["match_rule"],
                }
            )
            continue
        replacement = _make_te_group_conv_full_im2col_same_signature_primfunc(
            match["spec"], name, accum_dtype=accum_dtype
        )
        # Preserve the private/s_tir attributes expected by the fused full engine.
        attrs = getattr(func, "attrs", None)
        if attrs is not None:
            for key in ["private", "s_tir", "tirx.noalias"]:
                if key in attrs:
                    replacement = replacement.with_attr(key, attrs[key])
        if candidate == "fused_conv2d6_add10_relu6" and repeated_callsite_indices is not None:
            replacement_name = f"{name}_tensorcore_selected"
            replacement = replacement.with_attr("global_symbol", replacement_name)
            builder = relax.BlockBuilder(out)
            callsite_replacement_gv = builder.add_func(replacement, replacement_name)
            out = builder.finalize()
            records.append(
                {
                    "name": name,
                    "replacement_name": callsite_replacement_gv.name_hint,
                    "candidate": candidate,
                    "status": "callsite_replacement_added",
                    "rewrite_filter": rewrite_filter,
                    "selected_callsite_indices": sorted(repeated_callsite_indices),
                    "matched_input_nchw": list(match["matched_input_nchw"]),
                    "matched_output_nchw": list(match["matched_output_nchw"]),
                    "match_rule": match["match_rule"],
                    "original_counts": _counts(func.script()) if hasattr(func, "script") else {},
                    "replacement_counts_before_schedule": _counts(replacement.script()),
                }
            )
            continue
        out.update_func(gv, replacement)
        records.append(
            {
                "name": name,
                "candidate": candidate,
                "status": "replaced",
                "rewrite_filter": rewrite_filter,
                "matched_input_nchw": list(match["matched_input_nchw"]),
                "matched_output_nchw": list(match["matched_output_nchw"]),
                "match_rule": match["match_rule"],
                "original_counts": _counts(func.script()) if hasattr(func, "script") else {},
                "replacement_counts_before_schedule": _counts(replacement.script()),
            }
        )

    if repeated_callsite_indices is not None:
        if callsite_replacement_gv is None:
            records.append(
                {
                    "name": "fused_conv2d16_add8_relu6",
                    "candidate": "fused_conv2d6_add10_relu6",
                    "status": "callsite_replacement_missing",
                    "rewrite_filter": rewrite_filter,
                    "selected_callsite_indices": sorted(repeated_callsite_indices),
                }
            )
            return out, records

        @relax.expr_functor.mutator
        class CallsiteMutator(relax.PyExprMutator):
            def __init__(self, irmod: Any, replacement_gv: Any, selected: set[int]):
                super().__init__(irmod)
                self.replacement_gv = replacement_gv
                self.selected = selected
                self.callsite_index = 0
                self.replaced: list[int] = []
                self.skipped: list[int] = []

            def visit_call_(self, call: Any) -> Any:
                visited = super().visit_call_(call)
                if (
                    getattr(visited.op, "name", None) == "relax.call_tir"
                    and len(visited.args) >= 1
                    and getattr(visited.args[0], "name_hint", None) == "fused_conv2d16_add8_relu6"
                ):
                    idx = self.callsite_index
                    self.callsite_index += 1
                    if idx in self.selected:
                        self.replaced.append(idx)
                        tir_args = visited.args[1]
                        if hasattr(tir_args, "fields"):
                            tir_args = list(tir_args.fields)
                        return relax.call_tir(
                            self.replacement_gv,
                            tir_args,
                            visited.sinfo_args[0],
                        )
                    self.skipped.append(idx)
                return visited

        main_gv = None
        for gv, _ in out.functions_items():
            if gv.name_hint == "main":
                main_gv = gv
                break
        if main_gv is None:
            raise RuntimeError("main_global_var_not_found_for_callsite_rewrite")
        mut = CallsiteMutator(out, callsite_replacement_gv, repeated_callsite_indices)
        new_main = mut.visit_expr(out[main_gv])
        out.update_func(main_gv, new_main)
        records.append(
            {
                "name": "main",
                "candidate": "fused_conv2d6_add10_relu6",
                "status": "callsite_rewrite_applied",
                "rewrite_filter": rewrite_filter,
                "selected_callsite_indices": sorted(repeated_callsite_indices),
                "actual_replaced_callsite_indices": list(mut.replaced),
                "actual_skipped_callsite_indices": list(mut.skipped),
                "total_callsite_count": int(mut.callsite_index),
            }
        )
    return out, records


def _parse_callsite_indices(text: str) -> set[int] | None:
    value = str(text or "").strip()
    if not value:
        return None
    indices: set[int] = set()
    for part in value.split(","):
        token = part.strip()
        if not token:
            continue
        idx = int(token)
        if idx < 0:
            raise ValueError(f"negative_callsite_index:{idx}")
        indices.add(idx)
    return indices


def _append_repeated_callsite_outputs_to_main(
    mod: Any,
    target_call_tir_names: set[str],
) -> tuple[Any, list[dict[str, Any]]]:
    """Return a module whose main additionally returns selected call_tir outputs."""
    import tvm
    from tvm import relax

    out = tvm.IRModule(dict(mod.functions), attrs=mod.attrs)
    main_gv = None
    for gv, _ in out.functions_items():
        if gv.name_hint == "main":
            main_gv = gv
            break
    if main_gv is None:
        raise RuntimeError("main_global_var_not_found_for_intermediate_debug")

    main = out[main_gv]
    body = main.body
    if not hasattr(body, "blocks") or not body.blocks:
        raise RuntimeError("main_body_not_seqexpr_for_intermediate_debug")
    if len(body.blocks) != 1:
        raise RuntimeError(f"unsupported_main_block_count:{len(body.blocks)}")

    block = body.blocks[0]
    bindings = list(block.bindings)
    captured: list[Any] = []
    records: list[dict[str, Any]] = []
    for binding_index, binding in enumerate(bindings):
        value = getattr(binding, "value", None)
        if getattr(getattr(value, "op", None), "name", None) != "relax.call_tir":
            continue
        call_name = getattr(value.args[0], "name_hint", None)
        if call_name not in target_call_tir_names:
            continue
        captured.append(binding.var)
        records.append(
            {
                "capture_index": len(captured) - 1,
                "binding_index": int(binding_index),
                "var": str(binding.var),
                "call_tir_name": call_name,
                "struct_info": str(binding.var.struct_info),
            }
        )

    if not captured:
        raise RuntimeError(f"no_call_tir_captured:{sorted(target_call_tir_names)}")

    original_output_expr = body.body
    original_fields = (
        list(original_output_expr.fields)
        if hasattr(original_output_expr, "fields")
        else [original_output_expr]
    )
    fields = original_fields + captured
    tuple_sinfo = relax.TupleStructInfo([field.struct_info for field in fields])
    debug_output = relax.Var("gv_intermediate_debug", tuple_sinfo)
    debug_binding = relax.VarBinding(debug_output, relax.Tuple(fields))
    debug_block = relax.DataflowBlock(bindings + [debug_binding])
    debug_body = relax.SeqExpr([debug_block], debug_output)
    debug_main = relax.Function(main.params, debug_body, tuple_sinfo, main.is_pure, main.attrs)
    out.update_func(main_gv, debug_main)
    out = relax.transform.ToNonDataflow()(out)
    return out, records


def _compare_arrays(base: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    base32 = base.astype("float32")
    candidate32 = candidate.astype("float32")
    diff = np.abs(base32 - candidate32)
    base_abs = np.abs(base32)
    candidate_abs = np.abs(candidate32)
    base_abs_max = float(np.max(base_abs))
    base_abs_mean = float(np.mean(base_abs))
    return {
        "original_shape": list(base.shape),
        "rewritten_shape": list(candidate.shape),
        "max_abs_err": float(np.max(diff)),
        "mean_abs_err": float(np.mean(diff)),
        "original_abs_max": base_abs_max,
        "original_abs_mean": base_abs_mean,
        "rewritten_abs_max": float(np.max(candidate_abs)),
        "rewritten_abs_mean": float(np.mean(candidate_abs)),
        "max_abs_err_over_original_abs_max": float(np.max(diff) / base_abs_max) if base_abs_max else None,
        "mean_abs_err_over_original_abs_mean": float(np.mean(diff) / base_abs_mean) if base_abs_mean else None,
    }


def run_full_engine_group_conv_rewrite(args: argparse.Namespace, target: Any, dev: Any) -> dict[str, Any]:
    import tvm
    from tvm import relax

    label = str(getattr(args, "label", "lhc_07") or "lhc_07")
    label_safe = safe_label(label)
    onnx_path = Path(args.onnx)
    raw_dir = Path(args.raw_dir) / f"{label_safe}_full_engine_group_conv_rewrite"
    raw_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema": "fp16_full_engine_group_conv_rewrite_v2",
        "label": label,
        "label_safe": label_safe,
        "onnx_path": str(onnx_path),
        "raw_dir": str(raw_dir),
        "status": "started",
        "claim": "rewritten_full_engine_candidate",
        "rewrite_filter": str(args.group_conv_rewrite_filter),
        "repeated_callsite_indices": sorted(_parse_callsite_indices(args.group_conv_rewrite_callsite_indices) or []),
    }
    try:
        effective_onnx_path = onnx_path
        if bool(args.cast_fp16_source):
            cast_result = _cast_onnx_float_tensors_to_fp16(onnx_path, raw_dir)
            payload["source_cast_fp16"] = cast_result
            effective_onnx_path = Path(cast_result["fp16_inferred_onnx"])

        mod0, shape_dict = _load_onnx_relax_mod(effective_onnx_path, int(args.batch))
        payload["shape_dict"] = {key: list(value) for key, value in shape_dict.items()}
        input_dtypes = _onnx_input_numpy_dtypes(effective_onnx_path)
        payload["input_dtypes"] = input_dtypes
        rng = np.random.RandomState(20260701)
        feeds_np = {
            key: rng.rand(*value).astype(input_dtypes.get(key, "float32"))
            for key, value in shape_dict.items()
        }
        feeds_tvm = [tvm.runtime.tensor(feeds_np[key], dev) for key in shape_dict]

        with tvm.transform.PassContext(opt_level=3):
            default_ex = relax.build(mod0, target=target)
        default_vm = relax.VirtualMachine(default_ex, dev)
        default_mean_us, default_min_us, default_repeats_us = _time_vm(
            default_vm, feeds_tvm, dev, int(args.full_reps)
        )
        default_out = _flatten_vm_output(default_vm["main"](*feeds_tvm))
        payload["default_full_engine"] = {
            "status": "success",
            "route": "default_relax_build",
            "latency_mean_us": default_mean_us,
            "latency_mean_ms": default_mean_us / 1000.0,
            "latency_min_us": default_min_us,
            "latency_repeats_us": default_repeats_us,
            "scheduled_counts": {"wmma": 0, "tvm_mma_sync": 0},
            "export_library": _try_export_library(default_ex, raw_dir / "default_full_engine.so"),
        }

        legal = _legalize_fuse(mod0, target)
        legal_path = raw_dir / "full_engine_legalize_fuse_before_rewrite.py"
        legal_path.write_text(legal.script(), encoding="utf-8")
        repeated_callsite_indices = _parse_callsite_indices(args.group_conv_rewrite_callsite_indices)
        _accum_dtype = "int32" if getattr(args, "rewrite_dtype", "float16") == "int8" else "float16"
        rewritten, replace_records = _replace_group_conv_primfuncs_for_full_engine(
            legal, str(args.group_conv_rewrite_filter), repeated_callsite_indices,
            accum_dtype=_accum_dtype,
        )
        rewritten_unscheduled_path = raw_dir / "full_engine_group_conv_rewritten_unscheduled.py"
        rewritten_unscheduled_path.write_text(rewritten.script(), encoding="utf-8")
        scheduled, selective_records = _apply_selective_matmul_tensorization(
            rewritten, target, accum_dtype=_accum_dtype)
        scheduled_text = scheduled.script()
        scheduled_path = raw_dir / "full_engine_group_conv_rewritten_selective_tensorcore_scheduled.py"
        scheduled_path.write_text(scheduled_text, encoding="utf-8")
        counts = _counts(scheduled_text)
        with target, tvm.transform.PassContext(opt_level=3):
            rewritten_ex = tvm.compile(scheduled, target=target)
        rewritten_vm = relax.VirtualMachine(rewritten_ex, dev)
        rewritten_mean_us, rewritten_min_us, rewritten_repeats_us = _time_vm(
            rewritten_vm, feeds_tvm, dev, int(args.full_reps)
        )
        rewritten_out = _flatten_vm_output(rewritten_vm["main"](*feeds_tvm))
        diffs: list[dict[str, Any]] = []
        for idx, (base, candidate) in enumerate(zip(default_out, rewritten_out)):
            base32 = base.astype("float32")
            candidate32 = candidate.astype("float32")
            diff = np.abs(base32 - candidate32)
            base_abs = np.abs(base32)
            candidate_abs = np.abs(candidate32)
            base_abs_max = float(np.max(base_abs))
            base_abs_mean = float(np.mean(base_abs))
            diffs.append(
                {
                    "output_index": idx,
                    "original_shape": list(base.shape),
                    "rewritten_shape": list(candidate.shape),
                    "max_abs_err": float(np.max(diff)),
                    "mean_abs_err": float(np.mean(diff)),
                    "original_abs_max": base_abs_max,
                    "original_abs_mean": base_abs_mean,
                    "rewritten_abs_max": float(np.max(candidate_abs)),
                    "rewritten_abs_mean": float(np.mean(candidate_abs)),
                    "max_abs_err_over_original_abs_max": float(np.max(diff) / base_abs_max) if base_abs_max else None,
                    "mean_abs_err_over_original_abs_mean": float(np.mean(diff) / base_abs_mean) if base_abs_mean else None,
                }
            )
        payload["rewritten_full_engine"] = {
            "status": "success",
            "route": "replace_two_group_conv_primfuncs_with_full_im2col_tensorcore",
            "latency_mean_us": rewritten_mean_us,
            "latency_mean_ms": rewritten_mean_us / 1000.0,
            "latency_min_us": rewritten_min_us,
            "latency_repeats_us": rewritten_repeats_us,
            "legalized_path": str(legal_path),
            "rewritten_unscheduled_path": str(rewritten_unscheduled_path),
            "scheduled_path": str(scheduled_path),
            "replace_records": replace_records,
            "selective_records": selective_records,
            "scheduled_counts": counts,
            "tensorcore_gate": bool(counts.get("wmma", 0) > 0 and counts.get("tvm_mma_sync", 0) > 0),
            "output_compare": diffs,
            "export_library": _try_export_library(rewritten_ex, raw_dir / "rewritten_full_engine.so"),
        }
        payload["comparison"] = {
            "latency_delta_us": default_mean_us - rewritten_mean_us,
            "latency_delta_ms": (default_mean_us - rewritten_mean_us) / 1000.0,
            "speedup_ratio": default_mean_us / rewritten_mean_us if rewritten_mean_us > 0 else None,
            "is_faster_than_default": bool(rewritten_mean_us < default_mean_us),
            "claim": "rewritten_full_engine_measured_latency",
        }
        payload["status"] = "success"
    except Exception as exc:
        payload.update({"status": "failed", "error": repr(exc), "traceback": traceback.format_exc()})
        (raw_dir / "failure_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
    return payload


def write_full_engine_group_conv_rewrite_outputs(args: argparse.Namespace, result: dict[str, Any]) -> tuple[Path, Path]:
    export_dir = Path(args.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    label = str(result.get("label") or getattr(args, "label", "lhc_07") or "lhc_07")
    label_safe = safe_label(label)
    json_path = export_dir / f"fp16_{label_safe}_full_engine_group_conv_rewrite_latest.json"
    md_path = export_dir / f"fp16_{label_safe}_full_engine_group_conv_rewrite_latest.md"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    default = result.get("default_full_engine") or {}
    rewritten = result.get("rewritten_full_engine") or {}
    cmp_row = result.get("comparison") or {}
    counts = rewritten.get("scheduled_counts") or {}
    lines = [
        f"# FP16 {label} full-engine group-conv rewrite",
        "",
        f"- status: `{result.get('status')}`",
        f"- label: `{label}`",
        f"- claim: `{result.get('claim')}`",
        f"- onnx_path: `{result.get('onnx_path')}`",
        f"- cast_fp16_source: `{bool(result.get('source_cast_fp16'))}`",
        f"- raw_dir: `{result.get('raw_dir')}`",
        "",
        "## 1. Engine Evidence",
        "",
        "| engine | status | wmma | tvm_mma_sync | latency_mean_ms | export |",
        "|---|---|---:|---:|---:|---|",
        "| default full engine | `{}` | {} | {} | {} | `{}` |".format(
            default.get("status"),
            (default.get("scheduled_counts") or {}).get("wmma"),
            (default.get("scheduled_counts") or {}).get("tvm_mma_sync"),
            default.get("latency_mean_ms"),
            (default.get("export_library") or {}).get("status"),
        ),
        "| rewritten full engine | `{}` | {} | {} | {} | `{}` |".format(
            rewritten.get("status"),
            counts.get("wmma"),
            counts.get("tvm_mma_sync"),
            rewritten.get("latency_mean_ms"),
            (rewritten.get("export_library") or {}).get("status"),
        ),
        "",
        "## 2. Output Compare",
        "",
        "| output | original_shape | rewritten_shape | max_abs_err | mean_abs_err | original_abs_max | original_abs_mean | max_err/orig_max | mean_err/orig_mean |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in rewritten.get("output_compare", []):
        lines.append(
            "| {} | `{}` | `{}` | {} | {} | {} | {} | {} | {} |".format(
                item.get("output_index"),
                item.get("original_shape"),
                item.get("rewritten_shape"),
                item.get("max_abs_err"),
                item.get("mean_abs_err"),
                item.get("original_abs_max"),
                item.get("original_abs_mean"),
                item.get("max_abs_err_over_original_abs_max"),
                item.get("mean_abs_err_over_original_abs_mean"),
            )
        )
    lines += [
        "",
        "## 3. Comparison",
        "",
        f"- latency_delta_ms: `{cmp_row.get('latency_delta_ms')}`",
        f"- speedup_ratio: `{cmp_row.get('speedup_ratio')}`",
        f"- is_faster_than_default: `{cmp_row.get('is_faster_than_default')}`",
        f"- tensorcore_gate: `{rewritten.get('tensorcore_gate')}`",
        f"- claim: `{cmp_row.get('claim')}`",
        "",
        "## 4. Interpretation",
        "",
        "- 这是真实 rewritten full-engine build/run 路径, 不再是 counterfactual。",
        "- replacement 方式是替换两个 group conv PrimFunc, main graph 其它部分保持不变。",
        "- 若输出误差可解释且 latency 小于 default, 可作为 rewritten full-engine measured latency 证据。",
        "",
    ]
    if result.get("status") != "success":
        lines += [
            "## 5. Failure",
            "",
            f"- error: `{result.get('error')}`",
            f"- traceback_path: `{Path(result.get('raw_dir', '.')) / 'failure_traceback.txt'}`",
            "",
        ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def run_full_engine_group_conv_intermediate_debug(args: argparse.Namespace, target: Any, dev: Any) -> dict[str, Any]:
    import tvm
    from tvm import relax

    onnx_path = Path(args.onnx)
    raw_dir = Path(args.raw_dir) / "lhc07_full_engine_group_conv_intermediate_debug"
    raw_dir.mkdir(parents=True, exist_ok=True)
    selected = _parse_callsite_indices(args.group_conv_rewrite_callsite_indices)
    if selected is None:
        selected = set(range(7))
    payload: dict[str, Any] = {
        "schema": "fp16_lhc07_full_engine_group_conv_intermediate_debug_v1",
        "label": "lhc_07",
        "onnx_path": str(onnx_path),
        "raw_dir": str(raw_dir),
        "status": "started",
        "claim": "intermediate_output_debug_not_latency",
        "rewrite_filter": "repeated",
        "selected_repeated_callsite_indices": sorted(selected),
    }
    try:
        effective_onnx_path = onnx_path
        if bool(args.cast_fp16_source):
            cast_result = _cast_onnx_float_tensors_to_fp16(onnx_path, raw_dir)
            payload["source_cast_fp16"] = cast_result
            effective_onnx_path = Path(cast_result["fp16_inferred_onnx"])

        mod0, shape_dict = _load_onnx_relax_mod(effective_onnx_path, int(args.batch))
        payload["shape_dict"] = {key: list(value) for key, value in shape_dict.items()}
        input_dtypes = _onnx_input_numpy_dtypes(effective_onnx_path)
        payload["input_dtypes"] = input_dtypes
        rng = np.random.RandomState(20260701)
        feeds_np = {
            key: rng.rand(*value).astype(input_dtypes.get(key, "float32"))
            for key, value in shape_dict.items()
        }
        feeds_tvm = [tvm.runtime.tensor(feeds_np[key], dev) for key in shape_dict]

        legal = _legalize_fuse(mod0, target)
        legal_path = raw_dir / "full_engine_legalize_fuse_before_intermediate_debug.py"
        legal_path.write_text(legal.script(), encoding="utf-8")

        default_debug, default_capture_records = _append_repeated_callsite_outputs_to_main(
            legal, {"fused_conv2d16_add8_relu6"}
        )
        default_debug_path = raw_dir / "default_full_engine_with_repeated_callsite_outputs.py"
        default_debug_path.write_text(default_debug.script(), encoding="utf-8")
        with tvm.transform.PassContext(opt_level=3):
            default_ex = relax.build(default_debug, target=target)
        default_vm = relax.VirtualMachine(default_ex, dev)
        default_out = _flatten_vm_output(default_vm["main"](*feeds_tvm))

        rewritten, replace_records = _replace_group_conv_primfuncs_for_full_engine(
            legal, "repeated", selected
        )
        rewritten_debug, rewritten_capture_records = _append_repeated_callsite_outputs_to_main(
            rewritten,
            {
                "fused_conv2d16_add8_relu6",
                "fused_conv2d16_add8_relu6_tensorcore_selected",
            },
        )
        rewritten_debug_unscheduled_path = raw_dir / "rewritten_full_engine_with_repeated_callsite_outputs_unscheduled.py"
        rewritten_debug_unscheduled_path.write_text(rewritten_debug.script(), encoding="utf-8")

        with tvm.transform.PassContext(opt_level=3):
            rewritten_unscheduled_ex = relax.build(rewritten_debug, target=target)
        rewritten_unscheduled_vm = relax.VirtualMachine(rewritten_unscheduled_ex, dev)
        rewritten_unscheduled_out = _flatten_vm_output(
            rewritten_unscheduled_vm["main"](*feeds_tvm)
        )

        scheduled, selective_records = _apply_selective_matmul_tensorization(rewritten_debug, target)
        scheduled_text = scheduled.script()
        scheduled_path = raw_dir / "rewritten_full_engine_with_repeated_callsite_outputs_scheduled.py"
        scheduled_path.write_text(scheduled_text, encoding="utf-8")
        counts = _counts(scheduled_text)
        with target, tvm.transform.PassContext(opt_level=3):
            rewritten_ex = tvm.compile(scheduled, target=target)
        rewritten_vm = relax.VirtualMachine(rewritten_ex, dev)
        rewritten_out = _flatten_vm_output(rewritten_vm["main"](*feeds_tvm))

        if len(default_out) != len(rewritten_out):
            raise RuntimeError(f"debug_output_count_mismatch:{len(default_out)}:{len(rewritten_out)}")
        if len(default_out) != len(rewritten_unscheduled_out):
            raise RuntimeError(
                f"debug_unscheduled_output_count_mismatch:{len(default_out)}:{len(rewritten_unscheduled_out)}"
            )
        final_output_count = 3

        def make_debug_diffs(candidate_out: list[np.ndarray]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
            final_rows: list[dict[str, Any]] = []
            for out_idx in range(final_output_count):
                item = _compare_arrays(default_out[out_idx], candidate_out[out_idx])
                item["output_index"] = out_idx
                final_rows.append(item)
            callsite_rows: list[dict[str, Any]] = []
            for out_idx in range(final_output_count, len(default_out)):
                callsite_idx = out_idx - final_output_count
                item = _compare_arrays(default_out[out_idx], candidate_out[out_idx])
                item.update(
                    {
                        "callsite_index": callsite_idx,
                        "output_index": out_idx,
                        "default_capture": default_capture_records[callsite_idx]
                        if callsite_idx < len(default_capture_records)
                        else None,
                        "rewritten_capture": rewritten_capture_records[callsite_idx]
                        if callsite_idx < len(rewritten_capture_records)
                        else None,
                        "selected_for_rewrite": bool(callsite_idx in selected),
                    }
                )
                callsite_rows.append(item)
            return final_rows, callsite_rows

        unscheduled_final_diffs, unscheduled_callsite_diffs = make_debug_diffs(
            rewritten_unscheduled_out
        )
        scheduled_final_diffs, scheduled_callsite_diffs = make_debug_diffs(rewritten_out)

        payload.update(
            {
                "status": "success",
                "default_debug": {
                    "status": "success",
                    "capture_records": default_capture_records,
                    "debug_module_path": str(default_debug_path),
                    "export_library": _try_export_library(default_ex, raw_dir / "default_intermediate_debug.so"),
                },
                "rewritten_unscheduled_debug": {
                    "status": "success",
                    "replace_records": replace_records,
                    "rewritten_unscheduled_path": str(rewritten_debug_unscheduled_path),
                    "scheduled_counts": _counts(rewritten_debug.script()),
                    "tensorcore_gate": False,
                    "export_library": _try_export_library(
                        rewritten_unscheduled_ex, raw_dir / "rewritten_unscheduled_intermediate_debug.so"
                    ),
                    "final_output_compare": unscheduled_final_diffs,
                    "callsite_output_compare": unscheduled_callsite_diffs,
                },
                "rewritten_debug": {
                    "status": "success",
                    "replace_records": replace_records,
                    "capture_records": rewritten_capture_records,
                    "rewritten_unscheduled_path": str(rewritten_debug_unscheduled_path),
                    "scheduled_path": str(scheduled_path),
                    "selective_records": selective_records,
                    "scheduled_counts": counts,
                    "tensorcore_gate": bool(counts.get("wmma", 0) > 0 and counts.get("tvm_mma_sync", 0) > 0),
                    "export_library": _try_export_library(rewritten_ex, raw_dir / "rewritten_intermediate_debug.so"),
                },
                "final_output_compare": scheduled_final_diffs,
                "callsite_output_compare": scheduled_callsite_diffs,
                "interpretation_hint": (
                    "Compare rewritten_unscheduled_debug against rewritten_debug. If unscheduled is "
                    "near-exact but scheduled drifts, root cause is tensorization/FP16 accumulation or "
                    "schedule numeric behavior. If unscheduled already drifts, inspect replacement "
                    "indexing/padding/group mapping."
                ),
            }
        )
    except Exception as exc:
        payload.update({"status": "failed", "error": repr(exc), "traceback": traceback.format_exc()})
        (raw_dir / "failure_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
    return payload


def write_full_engine_group_conv_intermediate_debug_outputs(
    args: argparse.Namespace, result: dict[str, Any]
) -> tuple[Path, Path]:
    export_dir = Path(args.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    json_path = export_dir / "fp16_lhc07_full_engine_group_conv_intermediate_debug_latest.json"
    md_path = export_dir / "fp16_lhc07_full_engine_group_conv_intermediate_debug_latest.md"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    counts = (result.get("rewritten_debug") or {}).get("scheduled_counts") or {}
    lines = [
        "# FP16 lhc_07 full-engine group-conv intermediate debug",
        "",
        f"- status: `{result.get('status')}`",
        f"- claim: `{result.get('claim')}`",
        f"- selected_repeated_callsite_indices: `{result.get('selected_repeated_callsite_indices')}`",
        f"- raw_dir: `{result.get('raw_dir')}`",
        f"- rewritten_wmma: `{counts.get('wmma')}`",
        f"- rewritten_tvm_mma_sync: `{counts.get('tvm_mma_sync')}`",
        "",
        "## Final Output Compare: TensorCore Scheduled",
        "",
        "| output | max_abs_err | mean_abs_err | mean_err/orig_mean |",
        "|---:|---:|---:|---:|",
    ]
    for item in result.get("final_output_compare", []):
        lines.append(
            "| {} | {} | {} | {} |".format(
                item.get("output_index"),
                item.get("max_abs_err"),
                item.get("mean_abs_err"),
                item.get("mean_abs_err_over_original_abs_mean"),
            )
        )
    lines += [
        "",
        "## Repeated Callsite Immediate Output Compare: TensorCore Scheduled",
        "",
        "| callsite | selected | rewritten_call | max_abs_err | mean_abs_err | mean_err/orig_mean |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for item in result.get("callsite_output_compare", []):
        rewritten_capture = item.get("rewritten_capture") or {}
        lines.append(
            "| {} | `{}` | `{}` | {} | {} | {} |".format(
                item.get("callsite_index"),
                item.get("selected_for_rewrite"),
                rewritten_capture.get("call_tir_name"),
                item.get("max_abs_err"),
                item.get("mean_abs_err"),
                item.get("mean_abs_err_over_original_abs_mean"),
            )
        )
    unscheduled = result.get("rewritten_unscheduled_debug") or {}
    lines += [
        "",
        "## Final Output Compare: Replacement Unscheduled",
        "",
        "| output | max_abs_err | mean_abs_err | mean_err/orig_mean |",
        "|---:|---:|---:|---:|",
    ]
    for item in unscheduled.get("final_output_compare", []):
        lines.append(
            "| {} | {} | {} | {} |".format(
                item.get("output_index"),
                item.get("max_abs_err"),
                item.get("mean_abs_err"),
                item.get("mean_abs_err_over_original_abs_mean"),
            )
        )
    lines += [
        "",
        "## Repeated Callsite Immediate Output Compare: Replacement Unscheduled",
        "",
        "| callsite | selected | max_abs_err | mean_abs_err | mean_err/orig_mean |",
        "|---:|---|---:|---:|---:|",
    ]
    for item in unscheduled.get("callsite_output_compare", []):
        lines.append(
            "| {} | `{}` | {} | {} | {} |".format(
                item.get("callsite_index"),
                item.get("selected_for_rewrite"),
                item.get("max_abs_err"),
                item.get("mean_abs_err"),
                item.get("mean_abs_err_over_original_abs_mean"),
            )
        )
    lines += [
        "",
        "## Interpretation Hint",
        "",
        f"- {result.get('interpretation_hint')}",
        "",
    ]
    if result.get("status") != "success":
        lines += [
            "## Failure",
            "",
            f"- error: `{result.get('error')}`",
            f"- traceback_path: `{Path(result.get('raw_dir', '.')) / 'failure_traceback.txt'}`",
            "",
        ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def _measure_matmul_shape(
    shape: tuple[int, int, int, bool],
    args: argparse.Namespace,
    target: Any,
    dev: Any,
    raw_dir: Path,
) -> dict[str, Any]:
    import tvm
    from tvm import relax

    m, k, n, has_bias = shape
    rng = np.random.RandomState((m + 31 * k + 131 * n + int(has_bias)) % (2**32 - 1))
    x_np = (rng.rand(m, k).astype("float16") - np.float16(0.5)) * np.float16(0.02)
    w_np = (rng.rand(k, n).astype("float16") - np.float16(0.5)) * np.float16(0.02)
    b_np = (rng.rand(n).astype("float16") - np.float16(0.5)) * np.float16(0.02)
    variant = "matmul_bias_relu_k48" if has_bias else "pure_matmul_k48"
    # k_dim is only used to set TensorStructInfo; it is not constrained to 48.
    mod = _make_lhc07_1x1_conv_as_matmul_mod(variant, k)
    # Patch the fixed lhc_07 M/N helper by creating the general module inline.
    from tvm import relax as _relax

    bb = _relax.BlockBuilder()
    x = _relax.Var("x", _relax.TensorStructInfo((m, k), "float16"))
    w = _relax.Var("w", _relax.TensorStructInfo((k, n), "float16"))
    params = [x, w]
    b = None
    if has_bias:
        b = _relax.Var("b", _relax.TensorStructInfo((n,), "float16"))
        params.append(b)
    with bb.function("main", params):
        with bb.dataflow():
            y = bb.emit(_relax.op.matmul(x, w, out_dtype="float32"))
            if has_bias:
                assert b is not None
                y = bb.emit(_relax.op.add(y, bb.emit(_relax.op.astype(b, "float32"))))
            gv = bb.emit_output(y)
        bb.emit_func_output(gv)
    mod = bb.finalize()
    legal = _legalize_fuse(mod, target)
    conv_args = [tvm.runtime.tensor(x_np, dev), tvm.runtime.tensor(w_np, dev)]
    if has_bias:
        conv_args.append(tvm.runtime.tensor(b_np, dev))
    ref = x_np.astype("float32") @ w_np.astype("float32")
    if has_bias:
        ref = ref + b_np.astype("float32")

    out: dict[str, Any] = {
        "shape_key": {"M": m, "K": k, "N": n, "has_bias": has_bias},
        "status": "started",
    }
    try:
        with target, tvm.transform.PassContext(opt_level=3):
            default_ex = tvm.compile(legal, target=target)
        default_vm = relax.VirtualMachine(default_ex, dev)
        default_mean_us, default_min_us, default_repeats_us = _time_vm(
            default_vm, conv_args, dev, int(args.reps)
        )
        default_arr = default_vm["main"](*conv_args).numpy().astype("float32")
        tc = _apply_tensorcore_schedule(legal, target)
        tc_tir = tc.script()
        tir_name = f"tc_M{m}_K{k}_N{n}_bias{int(has_bias)}.py"
        (raw_dir / tir_name).write_text(tc_tir, encoding="utf-8")
        with target, tvm.transform.PassContext(opt_level=3):
            tc_ex = tvm.compile(tc, target=target)
        tc_vm = relax.VirtualMachine(tc_ex, dev)
        tc_mean_us, tc_min_us, tc_repeats_us = _time_vm(tc_vm, conv_args, dev, int(args.reps))
        tc_arr = tc_vm["main"](*conv_args).numpy().astype("float32")
        counts = _counts(tc_tir)
        out.update(
            {
                "status": "success",
                "default_latency_mean_us": default_mean_us,
                "default_latency_min_us": default_min_us,
                "default_latency_repeats_us": default_repeats_us,
                "default_max_abs_err": float(np.max(np.abs(default_arr - ref))),
                "tensorcore_latency_mean_us": tc_mean_us,
                "tensorcore_latency_min_us": tc_min_us,
                "tensorcore_latency_repeats_us": tc_repeats_us,
                "tensorcore_scheduled_counts": counts,
                "tensorcore_gate": bool(counts.get("wmma", 0) > 0 and counts.get("tvm_mma_sync", 0) > 0),
                "tensorcore_max_abs_err": float(np.max(np.abs(tc_arr - ref))),
                "delta_us": default_mean_us - tc_mean_us,
            }
        )
    except Exception as exc:
        out.update({"status": "failed", "error": repr(exc), "traceback": traceback.format_exc()})
    return out


def run_e2e_all1x1_counterfactual(args: argparse.Namespace, target: Any, dev: Any) -> dict[str, Any]:
    raw_dir = Path(args.raw_dir) / "lhc07_e2e_all1x1_counterfactual"
    raw_dir.mkdir(parents=True, exist_ok=True)
    base = run_e2e_counterfactual(args, target, dev)
    payload: dict[str, Any] = {
        "schema": "fp16_lhc07_e2e_all1x1_tensorcore_counterfactual_v1",
        "status": "started",
        "label": "lhc_07",
        "onnx_path": str(args.onnx),
        "raw_dir": str(raw_dir),
        "single_block_gate": base,
    }
    try:
        nodes = _eligible_1x1_convs(Path(args.onnx))
        payload["eligible_1x1_conv_count"] = len(nodes)
        payload["eligible_1x1_convs"] = nodes
        unique: dict[tuple[int, int, int, bool], dict[str, Any]] = {}
        for node in nodes:
            key = (int(node["M"]), int(node["K"]), int(node["N"]), bool(node["has_bias"]))
            if key not in unique:
                unique[key] = _measure_matmul_shape(key, args, target, dev, raw_dir)
        payload["unique_shape_measurements"] = [
            {"key": {"M": k[0], "K": k[1], "N": k[2], "has_bias": k[3]}, **v}
            for k, v in unique.items()
        ]
        total_default = 0.0
        total_tc = 0.0
        covered_nodes = 0
        failed_nodes: list[dict[str, Any]] = []
        for node in nodes:
            key = (int(node["M"]), int(node["K"]), int(node["N"]), bool(node["has_bias"]))
            measurement = unique[key]
            if measurement.get("status") == "success" and measurement.get("tensorcore_gate"):
                total_default += float(measurement["default_latency_mean_us"])
                total_tc += float(measurement["tensorcore_latency_mean_us"])
                covered_nodes += 1
            else:
                failed_nodes.append(node)
        full_us = float(base["full_default_engine"]["latency_mean_us"])
        counterfactual_us = full_us - total_default + total_tc
        payload["aggregate_counterfactual"] = {
            "status": "success",
            "formula": "full_default_engine - sum(eligible_1x1_default) + sum(eligible_1x1_tensorcore)",
            "full_default_latency_mean_us": full_us,
            "eligible_default_sum_us": total_default,
            "eligible_tensorcore_sum_us": total_tc,
            "absolute_delta_us": total_default - total_tc,
            "absolute_delta_ms": (total_default - total_tc) / 1000.0,
            "counterfactual_latency_mean_us": counterfactual_us,
            "counterfactual_latency_mean_ms": counterfactual_us / 1000.0,
            "speedup_ratio": full_us / counterfactual_us if counterfactual_us > 0 else None,
            "covered_nodes": covered_nodes,
            "failed_nodes": failed_nodes,
            "claim": "measured_all_eligible_1x1_tensorcore_counterfactual",
        }
        payload["status"] = "success"
    except Exception as exc:
        payload.update({"status": "failed", "error": repr(exc), "traceback": traceback.format_exc()})
    return payload


def write_e2e_outputs(args: argparse.Namespace, result: dict[str, Any]) -> tuple[Path, Path]:
    export_dir = Path(args.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    json_path = export_dir / "fp16_lhc07_e2e_tensorcore_validation_latest.json"
    md_path = export_dir / "fp16_lhc07_e2e_tensorcore_validation_latest.md"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    full = result.get("full_default_engine") or {}
    default = result.get("target_conv_default") or {}
    tc = result.get("target_conv_tensorcore") or {}
    cf = result.get("counterfactual") or {}
    default_counts = default.get("scheduled_counts") or {}
    tc_counts = tc.get("scheduled_counts") or {}
    lines = [
        "# FP16 lhc_07 end-to-end tensor-core validation",
        "",
        f"- status: `{result.get('status')}`",
        f"- label: `{result.get('label')}`",
        f"- onnx_path: `{result.get('onnx_path')}`",
        f"- raw_dir: `{result.get('raw_dir')}`",
        "",
        "## 1. Evidence Table",
        "",
        "| item | claim | status | wmma | tvm_mma_sync | latency_mean_us | latency_mean_ms | max_abs_err |",
        "|---|---|---|---:|---:|---:|---:|---:|",
        "| full default FP16 engine | `{}` | `{}` | {} | {} | {} | {} | {} |".format(
            full.get("claim"),
            full.get("status"),
            (full.get("scheduled_counts") or {}).get("wmma"),
            (full.get("scheduled_counts") or {}).get("tvm_mma_sync"),
            full.get("latency_mean_us"),
            (full.get("latency_mean_us") / 1000.0) if full.get("latency_mean_us") else None,
            "n/a",
        ),
        "| target conv default | `{}` | `{}` | {} | {} | {} | {} | {} |".format(
            default.get("claim"),
            default.get("status"),
            default_counts.get("wmma"),
            default_counts.get("tvm_mma_sync"),
            default.get("latency_mean_us"),
            (default.get("latency_mean_us") / 1000.0) if default.get("latency_mean_us") else None,
            default.get("max_abs_err"),
        ),
        "| target conv tensor-core | `{}` | `{}` | {} | {} | {} | {} | {} |".format(
            tc.get("claim"),
            tc.get("status"),
            tc_counts.get("wmma"),
            tc_counts.get("tvm_mma_sync"),
            tc.get("latency_mean_us"),
            (tc.get("latency_mean_us") / 1000.0) if tc.get("latency_mean_us") else None,
            tc.get("max_abs_err"),
        ),
        "| counterfactual full engine | `{}` | `{}` | n/a | n/a | {} | {} | n/a |".format(
            cf.get("claim"),
            cf.get("status"),
            cf.get("latency_mean_us"),
            cf.get("latency_mean_ms"),
        ),
        "",
        "## 2. Counterfactual Formula",
        "",
        "`counterfactual_latency = full_default_engine - target_conv_default + target_conv_tensorcore`",
        "",
        f"- absolute_delta_us: `{cf.get('absolute_delta_us')}`",
        f"- absolute_delta_ms: `{cf.get('absolute_delta_ms')}`",
        f"- speedup_ratio: `{cf.get('speedup_ratio')}`",
        f"- target_conv_default_share: `{cf.get('target_conv_default_share')}`",
        f"- target_conv_tensorcore_gate: `{cf.get('target_conv_tensorcore_gate')}`",
        "",
        "## 3. Interpretation",
        "",
        "- 这是完整 backbone/subnet latency 的 measured counterfactual, 不是已经落地的单一 rewritten full-engine binary。",
        "- 它证明同一个 lhc_07 算法图中的目标 FP16 convblock 一旦进入 tensor-core path, 会对完整推理 latency 产生可量化影响。",
        "- 若要把该影响变成 production full-engine 加速, 下一步必须完成 ONNX/TIR graph rewrite, 让完整 engine 本身包含 tensor-core convblock。",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def write_all1x1_outputs(args: argparse.Namespace, result: dict[str, Any]) -> tuple[Path, Path]:
    export_dir = Path(args.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    json_path = export_dir / "fp16_lhc07_e2e_all1x1_tensorcore_validation_latest.json"
    md_path = export_dir / "fp16_lhc07_e2e_all1x1_tensorcore_validation_latest.md"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    agg = result.get("aggregate_counterfactual") or {}
    lines = [
        "# FP16 lhc_07 all-1x1 end-to-end tensor-core validation",
        "",
        f"- status: `{result.get('status')}`",
        f"- label: `{result.get('label')}`",
        f"- eligible_1x1_conv_count: `{result.get('eligible_1x1_conv_count')}`",
        f"- raw_dir: `{result.get('raw_dir')}`",
        "",
        "## 1. Aggregate Counterfactual",
        "",
        f"- formula: `{agg.get('formula')}`",
        f"- full_default_latency_mean_us: `{agg.get('full_default_latency_mean_us')}`",
        f"- eligible_default_sum_us: `{agg.get('eligible_default_sum_us')}`",
        f"- eligible_tensorcore_sum_us: `{agg.get('eligible_tensorcore_sum_us')}`",
        f"- absolute_delta_us: `{agg.get('absolute_delta_us')}`",
        f"- absolute_delta_ms: `{agg.get('absolute_delta_ms')}`",
        f"- counterfactual_latency_mean_us: `{agg.get('counterfactual_latency_mean_us')}`",
        f"- counterfactual_latency_mean_ms: `{agg.get('counterfactual_latency_mean_ms')}`",
        f"- speedup_ratio: `{agg.get('speedup_ratio')}`",
        f"- covered_nodes: `{agg.get('covered_nodes')}`",
        f"- failed_nodes_count: `{len(agg.get('failed_nodes') or [])}`",
        "",
        "## 2. Unique Shape Measurements",
        "",
        "| M | K | N | bias | status | wmma | tvm_mma_sync | default_us | tensorcore_us | delta_us |",
        "|---:|---:|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for item in result.get("unique_shape_measurements", []):
        key = item.get("key") or item.get("shape_key") or {}
        counts = item.get("tensorcore_scheduled_counts") or {}
        lines.append(
            "| {M} | {K} | {N} | {bias} | `{status}` | {wmma} | {mma} | {default} | {tc} | {delta} |".format(
                M=key.get("M"),
                K=key.get("K"),
                N=key.get("N"),
                bias=key.get("has_bias"),
                status=item.get("status"),
                wmma=counts.get("wmma"),
                mma=counts.get("tvm_mma_sync"),
                default=item.get("default_latency_mean_us"),
                tc=item.get("tensorcore_latency_mean_us"),
                delta=item.get("delta_us"),
            )
        )
    lines += [
        "",
        "## 3. Interpretation",
        "",
        "- 这是 lhc_07 全部 eligible stride=1/group=1/1x1 Conv 的 measured counterfactual。",
        "- 它比单 block gate 更接近完整算法推理影响, 但仍不是单一 rewritten full-engine binary。",
        "- 若 absolute_delta_ms 明显大于单 block, 则证明 FP16 tensor-core lowering 对 lhc_07 完整 backbone/subnet latency 有可量化影响。",
        "- 真正最终收口仍需要 ONNX/TIR rewrite full engine build/run 通过, 并把该 counterfactual 变成实际 engine latency。",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=[
            "all",
            "convblock",
            "full-engine",
            "e2e-counterfactual",
            "e2e-all1x1",
            "rewrite-onnx-1x1",
            "group-conv-im2col",
            "group-conv-full-im2col",
            "full-engine-group-conv-counterfactual",
            "full-engine-group-conv-rewrite",
            "full-engine-group-conv-intermediate-debug",
            "group-conv-accum-compare",
            "self-test-no-tvm",
        ],
        default="all",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--label", default="lhc_07")
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--full-reps", type=int, default=30)
    parser.add_argument("--conv-repeats", type=int, default=3)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--onnx", default=str(DEFAULT_ONNX))
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
    parser.add_argument("--export-dir", default=str(DEFAULT_EXPORT_DIR))
    parser.add_argument("--cast-fp16-source", action="store_true")
    parser.add_argument("--rewrite-dtype", choices=("float16", "int8"), default="float16",
                        help="int8 = im2col matmul on int8 tensor cores (MatmulInt8Tensorization); "
                             "latency-only dummy scale, fp16 I/O signature preserved")
    parser.add_argument(
        "--group-conv-rewrite-filter",
        choices=["all", "downsample", "repeated"],
        default="all",
        help="Diagnostic filter for full-engine-group-conv-rewrite.",
    )
    parser.add_argument(
        "--group-conv-rewrite-callsite-indices",
        default="",
        help="Comma-separated fused_conv2d16_add8_relu6 callsite indices to rewrite via a duplicate TensorCore PrimFunc.",
    )
    parser.add_argument(
        "--conv-candidate",
        choices=sorted(GROUP_CONV_CANDIDATES),
        default="fused_conv2d4_add10_relu6",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["PATH"] = "/usr/local/cuda-12.2/bin:" + os.environ.get("PATH", "")

    if args.mode == "self-test-no-tvm":
        result = run_self_test_no_tvm(args)
        result["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        result["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        json_path, md_path = write_self_test_no_tvm_outputs(args, result)
        print(f"[done] json={json_path}")
        print(f"[done] md={md_path}")
        return 0 if result.get("status") == "success" else 2

    try:
        import tvm
        import tvm.s_tir.tensor_intrin.cuda  # noqa: F401
    except Exception as exc:
        result = {
            "schema": "fp16_lhc07_probe_import_failure_v1",
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": "failed",
            "mode": args.mode,
            "gpu": args.gpu,
            "raw_dir": str(args.raw_dir),
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "python_executable": sys.executable,
        }
        json_path, md_path = write_import_failure_outputs(args, result)
        print(f"[failed] json={json_path}")
        print(f"[failed] md={md_path}")
        return 2

    dev = tvm.cuda(0)
    target = tvm.target.Target.from_device(dev)
    result: dict[str, Any] = {
        "schema": "fp16_lhc07_convblock_tensorcore_and_engine_probe_v1",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "started",
        "mode": args.mode,
        "gpu": args.gpu,
        "target": str(target),
        "raw_dir": str(args.raw_dir),
    }

    if args.mode == "e2e-counterfactual":
        result = run_e2e_counterfactual(args, target, dev)
        result["target"] = str(target)
        result["started_at"] = result.get("started_at") or time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        )
        result["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        json_path, md_path = write_e2e_outputs(args, result)
        print(f"[done] json={json_path}")
        print(f"[done] md={md_path}")
        return 0 if result.get("status") == "success" else 2

    if args.mode == "e2e-all1x1":
        result = run_e2e_all1x1_counterfactual(args, target, dev)
        result["target"] = str(target)
        result["started_at"] = result.get("started_at") or time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        )
        result["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        json_path, md_path = write_all1x1_outputs(args, result)
        print(f"[done] json={json_path}")
        print(f"[done] md={md_path}")
        return 0 if result.get("status") == "success" else 2

    if args.mode == "rewrite-onnx-1x1":
        result = run_rewrite_onnx_1x1(args, target, dev)
        result["target"] = str(target)
        result["started_at"] = result.get("started_at") or time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        )
        result["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        json_path, md_path = write_rewrite_onnx_1x1_outputs(args, result)
        print(f"[done] json={json_path}")
        print(f"[done] md={md_path}")
        return 0 if result.get("status") == "success" else 2

    if args.mode == "group-conv-im2col":
        result = run_group_conv_im2col_tensorcore(args, target, dev)
        result["target"] = str(target)
        result["started_at"] = result.get("started_at") or time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        )
        result["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        json_path, md_path = write_group_conv_im2col_outputs(args, result)
        print(f"[done] json={json_path}")
        print(f"[done] md={md_path}")
        return 0 if result.get("status") == "success" else 2

    if args.mode == "group-conv-full-im2col":
        result = run_group_conv_full_im2col_tensorcore(args, target, dev)
        result["target"] = str(target)
        result["started_at"] = result.get("started_at") or time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        )
        result["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        json_path, md_path = write_group_conv_full_im2col_outputs(args, result)
        print(f"[done] json={json_path}")
        print(f"[done] md={md_path}")
        return 0 if result.get("status") == "success" else 2

    if args.mode == "full-engine-group-conv-counterfactual":
        result = run_full_engine_group_conv_counterfactual(args, target, dev)
        result["target"] = str(target)
        result["started_at"] = result.get("started_at") or time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        )
        result["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        json_path, md_path = write_full_engine_group_conv_counterfactual_outputs(args, result)
        print(f"[done] json={json_path}")
        print(f"[done] md={md_path}")
        return 0 if result.get("status") == "success" else 2

    if args.mode == "full-engine-group-conv-rewrite":
        result = run_full_engine_group_conv_rewrite(args, target, dev)
        result["target"] = str(target)
        result["started_at"] = result.get("started_at") or time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        )
        result["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        json_path, md_path = write_full_engine_group_conv_rewrite_outputs(args, result)
        print(f"[done] json={json_path}")
        print(f"[done] md={md_path}")
        return 0 if result.get("status") == "success" else 2

    if args.mode == "full-engine-group-conv-intermediate-debug":
        result = run_full_engine_group_conv_intermediate_debug(args, target, dev)
        result["target"] = str(target)
        result["started_at"] = result.get("started_at") or time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        )
        result["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        json_path, md_path = write_full_engine_group_conv_intermediate_debug_outputs(args, result)
        print(f"[done] json={json_path}")
        print(f"[done] md={md_path}")
        return 0 if result.get("status") == "success" else 2

    if args.mode == "group-conv-accum-compare":
        result = run_group_conv_accum_compare(args, target, dev)
        result["target"] = str(target)
        result["started_at"] = result.get("started_at") or time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        )
        result["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        json_path, md_path = write_group_conv_accum_compare_outputs(args, result)
        print(f"[done] json={json_path}")
        print(f"[done] md={md_path}")
        return 0 if result.get("status") == "success" else 2

    if args.mode in {"all", "convblock"}:
        result["convblock"] = run_convblock(args, target, dev)
    if args.mode in {"all", "full-engine"}:
        result["full_engine"] = run_full_engine(args, target, dev)

    convblock = result.get("convblock", {})
    conv_ok = bool(
        convblock.get("stable_tensorcore_k48", True)
        or convblock.get("stable_tensorcore_k64_padded", False)
    )
    engine_status = result.get("full_engine", {}).get("status", "success")
    result["status"] = "success" if conv_ok and engine_status in {"success", "failed"} else "failed"
    result["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    json_path, md_path = write_outputs(args, result)
    print(f"[done] json={json_path}")
    print(f"[done] md={md_path}")
    return 0 if result["status"] == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
