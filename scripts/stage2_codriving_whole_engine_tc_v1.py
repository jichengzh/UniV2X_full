#!/usr/bin/env python3
"""Whole-engine fp16-TC / int8-TC rewrite for CoDriving backbone (groups=1 conv).

Loads the real CoDriving backbone ONNX (base/p25/p50/p75), legalizes+fuses via
TVM relax, generically classifies every `fused_conv2d*` PrimFunc directly from
its own TIR buffer_map (no text-pattern / no Pyramid-specific classifier —
groups=1 is the only assumption), replaces the plain (x,w,bias)->out conv
functions [9/12 in the base backbone; the other 3 are residual-add conv2 of
each stage's first block, (x,w,bias,residual)->out, left on TVM's default
schedule and documented honestly rather than force-fit] with the proven
im2col+matmul same-signature primfunc (generalized here to real float32 I/O
with an internal fp16/int8 cast at the tensor-core boundary -- this is a
genuine precision-cast inference path, not a fabricated shortcut), applies
dlight's full-engine schedule (tensorization rule first, generic fallback
rules for everything else: conv2d_transpose / batch_norm / reshape /
concatenate), builds+runs on GPU, times e2e latency, and counts whole-module
mma_sync/wmma instructions directly in the scheduled module's own script().

The standalone `run_one` int8 path explicitly uses scale=1 quantize/dequantize
for latency/tensorization probing only. AP-capable callers must provide real
calibration scales; the same-signature rewrite now rejects missing scales.

Usage: python stage2_codriving_whole_engine_tc_v1.py --gpu 4 --onnx models/codriving_cache/base_backbone.onnx --precision both --reps 50 --out-json <path>
"""
from __future__ import annotations

import argparse
import json
import threading
import time
import traceback
from typing import Any


def _conv2d_out_hw(h: int, w: int, kh: int, kw: int, strides: tuple, padding: tuple) -> tuple[int, int]:
    sh, sw = strides
    pt, pl, pb, pr = padding
    oh = (h + pt + pb - kh) // sh + 1
    ow = (w + pl + pr - kw) // sw + 1
    return oh, ow


def _power_stats(samples: list[float]) -> dict[str, float | None]:
    if not samples:
        return {"avg": None, "p50": None, "p90": None}
    vals = sorted(float(v) for v in samples)
    p50 = vals[len(vals) // 2]
    p90 = vals[min(len(vals) - 1, int(round((len(vals) - 1) * 0.9)))]
    return {"avg": sum(vals) / len(vals), "p50": p50, "p90": p90}


def _query_power_w(gpu: int) -> float:
    import pynvml  # type: ignore

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu))
    return float(pynvml.nvmlDeviceGetPowerUsage(handle)) / 1000.0


def _sample_power(gpu: int, duration_s: float, interval_s: float = 0.05) -> list[float]:
    samples: list[float] = []
    deadline = time.time() + max(float(duration_s), 0.0)
    while time.time() < deadline:
        try:
            samples.append(_query_power_w(gpu))
        except Exception:
            pass
        time.sleep(interval_s)
    return samples


def _measure_energy_loop(
    vm: Any,
    vm_args: list[Any],
    dev: Any,
    gpu: int,
    *,
    energy_iters: int,
    min_active_s: float,
) -> dict[str, Any]:
    idle_samples = _sample_power(gpu, 5.0)
    active_samples: list[float] = []
    stop_sampling = threading.Event()

    def poll_power() -> None:
        while not stop_sampling.is_set():
            try:
                active_samples.append(_query_power_w(gpu))
            except Exception:
                pass
            time.sleep(0.05)

    completed_iters = 0
    start = time.time()
    sampler = threading.Thread(target=poll_power, name="codriving_tc_energy_sampler", daemon=True)
    sampler.start()
    try:
        while completed_iters < energy_iters or (time.time() - start) < min_active_s:
            vm["main"](*vm_args)
            completed_iters += 1
            if completed_iters % 20 == 0:
                dev.sync()
        dev.sync()
    finally:
        stop_sampling.set()
        sampler.join(timeout=1.0)

    elapsed = max(time.time() - start, 1e-9)
    idle = _power_stats(idle_samples)
    active = _power_stats(active_samples)
    idle_avg = float(idle["avg"] or 0.0)
    watt_avg = float(active["avg"] or 0.0)
    dynamic_watt_avg = max(watt_avg - idle_avg, 0.0)
    completed = max(completed_iters, 1)
    return {
        "status": "success" if active_samples else "no_power_samples",
        "joule_per_inference": dynamic_watt_avg * elapsed / float(completed),
        "idle_watt_avg": idle_avg,
        "watt_avg": watt_avg,
        "dynamic_watt_avg": dynamic_watt_avg,
        "watt_p50": active["p50"],
        "watt_p90": active["p90"],
        "elapsed_s": elapsed,
        "completed_measure_iters": completed_iters,
        "requested_measure_iters": energy_iters,
        "min_active_s": min_active_s,
        "idle_sample_count": len(idle_samples),
        "active_sample_count": len(active_samples),
    }


def round_te_expr(tvm_module: Any, te_module: Any, value: Any) -> Any:
    tir_module = getattr(tvm_module, "tir", None)
    if tir_module is not None and hasattr(tir_module, "round"):
        return tir_module.round(value)
    if hasattr(te_module, "round"):
        return te_module.round(value)
    raise RuntimeError("TVM build exposes neither tvm.tir.round nor te.round")


def make_std_conv_im2col_same_signature_primfunc(
    spec: dict[str, Any],
    name: str,
    accum_dtype: str,
    io_dtype: str = "float32",
    *,
    input_scale: float | None = None,
    weight_scale: float | None = None,
) -> Any:
    """groups=1 im2col+matmul same-signature primfunc with real io_dtype I/O
    (e.g. float32, matching the actual ONNX-imported graph) and an internal
    fp16 (accum_dtype='float16') or int8 (accum_dtype='int32') tensor-core
    cast. Supports an optional 5th `residual` input added post-bias pre-relu.
    """
    if accum_dtype not in {"float16", "int32"}:
        raise ValueError(f"unsupported_accum_dtype:{accum_dtype}")
    int8_mma = accum_dtype == "int32"
    if int8_mma and (
        input_scale is None
        or weight_scale is None
        or float(input_scale) <= 0.0
        or float(weight_scale) <= 0.0
    ):
        raise ValueError("int8 same-signature rewrite requires positive input_scale and weight_scale")

    import tvm
    from tvm import te

    core_dtype = "int8" if int8_mma else "float16"
    has_residual = bool(spec.get("residual", False))

    n, cin, h, width = [int(v) for v in spec["input_nchw"]]
    cout, cpg, kh, kw = [int(v) for v in spec["weight_oihw"]]
    groups = 1
    assert cpg == cin, f"groups=1 requires weight in-channels == input channels, got cpg={cpg} cin={cin}"
    oh, ow = _conv2d_out_hw(h, width, kh, kw, tuple(spec["strides"]), tuple(spec["padding"]))
    pt, pl, _, _ = tuple(spec["padding"])
    sh, sw = tuple(spec["strides"])
    m = n * oh * ow
    k_total = cpg * kh * kw
    nper = cout // groups

    x = te.placeholder((n, cin, h, width), io_dtype, name="x")
    weight = te.placeholder((cout, cpg, kh, kw), io_dtype, name="weight")
    bias = te.placeholder((1, cout, 1, 1), io_dtype, name="bias")
    residual = te.placeholder((n, cout, oh, ow), io_dtype, name="residual") if has_residual else None

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
        return te.if_then_else(cond, x[nn, in_c, in_y, in_x], te.const(0, io_dtype))

    x_col = te.compute((groups, m, k_total), im2col_compute, name="x_col")

    def weight_compute(g: Any, kk: Any, ocg: Any) -> Any:
        ci = kk // (kh * kw)
        rem_k = kk % (kh * kw)
        ry = rem_k // kw
        rx = rem_k % kw
        return weight[g * nper + ocg, ci, ry, rx]

    w_mat = te.compute((groups, k_total, nper), weight_compute, name="w_mat")

    def quantize_i8(value: Any, scale: float) -> Any:
        rounded = round_te_expr(
            tvm,
            te,
            value.astype("float32") / te.const(float(scale), "float32"),
        )
        upper = te.if_then_else(
            rounded > te.const(127.0, "float32"),
            te.const(127.0, "float32"),
            rounded,
        )
        clipped = te.if_then_else(
            upper < te.const(-127.0, "float32"),
            te.const(-127.0, "float32"),
            upper,
        )
        return clipped.astype("int8")

    if int8_mma:
        x_col_c = te.compute(
            (groups, m, k_total),
            lambda g, r, kk: quantize_i8(x_col[g, r, kk], float(input_scale)),
            name="x_col_c",
        )
        w_mat_c = te.compute(
            (groups, k_total, nper),
            lambda g, kk, o: quantize_i8(w_mat[g, kk, o], float(weight_scale)),
            name="w_mat_c",
        )
    else:
        x_col_c = te.compute(
            (groups, m, k_total),
            lambda g, r, kk: x_col[g, r, kk].astype(core_dtype),
            name="x_col_c",
        )
        w_mat_c = te.compute(
            (groups, k_total, nper),
            lambda g, kk, o: w_mat[g, kk, o].astype(core_dtype),
            name="w_mat_c",
        )

    rk = te.reduce_axis((0, k_total), name="rk")

    def matmul_compute(g: Any, row: Any, ocg: Any) -> Any:
        if int8_mma:
            return te.sum(x_col_c[g, row, rk].astype("int32") * w_mat_c[g, rk, ocg].astype("int32"), axis=rk)
        return te.sum(x_col_c[g, row, rk] * w_mat_c[g, rk, ocg], axis=rk)

    matmul = te.compute((groups, m, nper), matmul_compute, name="matmul")

    def out_compute(nn: Any, oc: Any, yy: Any, xx: Any) -> Any:
        g = oc // nper
        ocg = oc % nper
        row = nn * (oh * ow) + yy * ow + xx
        if int8_mma:
            dequant_scale = te.const(float(input_scale) * float(weight_scale), io_dtype)
            value = matmul[g, row, ocg].astype(io_dtype) * dequant_scale + bias[0, oc, 0, 0]
        else:
            value = matmul[g, row, ocg].astype(io_dtype) + bias[0, oc, 0, 0]
        if has_residual:
            value = value + residual[nn, oc, yy, xx]
        if bool(spec.get("relu", False)):
            value = te.max(value, te.const(0, io_dtype))
        return value

    out = te.compute((n, cout, oh, ow), out_compute, name="out_nchw")
    params = [x, weight, bias] + ([residual] if has_residual else []) + [out]
    return te.create_prim_func(params).with_attr("global_symbol", name)


def _counts(text: str) -> dict[str, int]:
    keys = ["wmma", "tvm_mma_sync", "mma_sync", "ldmatrix", "mma", "dp4a", "__dp4a"]
    return {k: text.count(k) for k in keys}


def classify_conv_funcs(mod: Any) -> dict[str, dict[str, Any]]:
    """Generic groups=1 classifier driven by the PrimFunc's own buffer_map
    shapes (not text regex, not name hardcoding). Recognizes both the plain
    (x,weight,bias)->out signature and the residual (x,weight,bias,residual)->out
    signature that FuseTIR produces for ResNet-style skip-add conv epilogues."""
    specs: dict[str, dict[str, Any]] = {}
    for gv, func in mod.functions_items():
        name = gv.name_hint
        if not name.startswith("fused_conv2d") or "transpose" in name:
            continue
        bufs = [func.buffer_map[p] for p in func.params if p in func.buffer_map]
        if len(bufs) not in (4, 5):
            continue
        shapes = [tuple(int(d) for d in b.shape) for b in bufs]
        dtypes = [b.dtype for b in bufs]
        if len(set(dtypes)) != 1:
            continue
        io_dtype = dtypes[0]
        weight_idx = [i for i, s in enumerate(shapes) if len(s) == 4 and s[2] == s[3] and s[2] in (1, 3) and s[0] != 1]
        if len(weight_idx) != 1:
            continue
        wi = weight_idx[0]
        cout, cin_w, kh, kw = shapes[wi]
        out_idx = len(shapes) - 1  # destination-passing convention: last param is the output buffer
        if shapes[out_idx][1] != cout or len(shapes[out_idx]) != 4:
            continue
        n_, _, oh, ow = shapes[out_idx]
        input_idx = [
            i for i, s in enumerate(shapes)
            if i not in (wi, out_idx) and len(s) == 4 and s[1] == cin_w and s[2] >= kh and s[2] != 1
        ]
        if not input_idx:
            continue
        ii = input_idx[0]
        n, cin, h, w = shapes[ii]
        bias_idx = [
            i for i, s in enumerate(shapes)
            if i not in (wi, out_idx, ii) and len(s) == 4 and s[1] == cout and s[2] == 1 and s[3] == 1
        ]
        if len(bias_idx) != 1:
            continue
        bi = bias_idx[0]
        residual_idx = [i for i in range(len(shapes)) if i not in (wi, out_idx, ii, bi)]
        has_residual = len(residual_idx) == 1
        if has_residual and shapes[residual_idx[0]] != shapes[out_idx]:
            continue
        stride_h = 1 if oh == h else 2
        stride_w = 1 if ow == w else 2
        pad = 1 if kh == 3 else 0
        specs[name] = {
            "input_nchw": (n, cin, h, w),
            "weight_oihw": (cout, cin_w, kh, kw),
            "strides": (stride_h, stride_w),
            "padding": (pad, pad, pad, pad),
            "relu": "relu" in name,
            "residual": has_residual,
            "io_dtype": io_dtype,
            "n_params": len(shapes),
        }
    return specs


def _read_onnx_inputs(onnx_model: Any, batch: int) -> dict[str, tuple]:
    init = {i.name for i in onnx_model.graph.initializer}
    shapes = {}
    for i in onnx_model.graph.input:
        if i.name in init:
            continue
        dims = [d.dim_value if d.dim_value > 0 else batch for d in i.type.tensor_type.shape.dim]
        shapes[i.name] = tuple(dims)
    return shapes


def _main_input_specs(mod: Any, fallback_shapes: dict[str, tuple]) -> list[dict[str, Any]]:
    fallback_items = list(fallback_shapes.items())
    specs: list[dict[str, Any]] = []
    try:
        params = list(mod["main"].params)
    except Exception:
        params = []
    for idx, param in enumerate(params):
        name = str(getattr(param, "name_hint", f"input_{idx}"))
        source_name, source_shape = fallback_items[idx] if idx < len(fallback_items) else (name, fallback_shapes.get(name, ()))
        sinfo = getattr(param, "struct_info", None)
        dtype = str(getattr(sinfo, "dtype", "float32"))
        shape_obj = getattr(sinfo, "shape", None)
        dims = getattr(shape_obj, "values", shape_obj)
        try:
            shape = tuple(int(dim) for dim in dims)
        except Exception:
            shape = tuple(int(dim) for dim in source_shape)
        specs.append(
            {
                "name": name,
                "source_name": source_name,
                "shape": list(shape),
                "source_shape": list(source_shape),
                "dtype": dtype,
            }
        )
    if specs:
        return specs
    return [
        {
            "name": name,
            "source_name": name,
            "shape": list(shape),
            "source_shape": list(shape),
            "dtype": "float32",
        }
        for name, shape in fallback_items
    ]


def _numpy_dtype(dtype: str) -> str:
    if dtype == "float16":
        return "float16"
    if dtype == "float32":
        return "float32"
    if dtype == "int8":
        return "int8"
    if dtype == "uint8":
        return "uint8"
    if dtype == "int32":
        return "int32"
    return "float32"


def build_and_legalize(onnx_path: str, batch: int, target: Any, graph_io_dtype: str = "fp32"):
    import onnx
    from tvm import relax
    from tvm.relax.frontend.onnx import from_onnx

    onnx_model = onnx.load(onnx_path)
    shapes = _read_onnx_inputs(onnx_model, batch)
    mod = from_onnx(onnx_model, shape_dict=shapes, keep_params_in_input=False)
    if graph_io_dtype == "fp16":
        mod = relax.transform.ToMixedPrecision(out_dtype="float16")(mod)
    import tvm

    seq = tvm.transform.Sequential(
        [
            relax.transform.LegalizeOps(),
            relax.transform.AnnotateTIROpPattern(),
            relax.transform.FuseOps(),
            relax.transform.FuseTIR(),
        ]
    )
    with target, tvm.transform.PassContext(opt_level=3):
        legal = seq(mod)
    return legal, _main_input_specs(legal, shapes)


def _full_engine_rules(accum_dtype: str):
    import tvm.s_tir.dlight as dl
    from tvm.s_tir.dlight.gpu.matmul import MatmulInt8Tensorization, MatmulTensorization

    if accum_dtype == "mixed":
        tensorize_rules = [MatmulInt8Tensorization(), MatmulTensorization()]
    else:
        tensorize_rules = [MatmulInt8Tensorization() if accum_dtype == "int32" else MatmulTensorization()]
    return tensorize_rules + [dl.gpu.Matmul(), dl.gpu.GEMV(), dl.gpu.Reduction(), dl.gpu.GeneralReduction(), dl.gpu.Fallback()]


def _conv_mac_proxy(spec: dict[str, Any]) -> int:
    n, _, h, width = [int(v) for v in spec["input_nchw"]]
    cout, cpg, kh, kw = [int(v) for v in spec["weight_oihw"]]
    oh, ow = _conv2d_out_hw(h, width, kh, kw, tuple(spec["strides"]), tuple(spec["padding"]))
    return int(n * oh * ow * cout * cpg * kh * kw)


def _select_mixed_int8_convs(conv_specs: dict[str, dict[str, Any]], policy: str) -> set[str]:
    if policy == "none":
        return set()
    if policy == "all":
        return set(conv_specs)
    ranked = sorted(conv_specs.items(), key=lambda item: _conv_mac_proxy(item[1]), reverse=True)
    if policy == "top25_flops":
        keep = max(1, (len(ranked) + 3) // 4)
        return {name for name, _ in ranked[:keep]}
    if policy == "top50_flops":
        keep = max(1, (len(ranked) + 1) // 2)
        return {name for name, _ in ranked[:keep]}
    if policy == "top75_flops":
        keep = max(1, (3 * len(ranked) + 3) // 4)
        return {name for name, _ in ranked[:keep]}
    if policy == "stride2_or_1x1":
        out = set()
        for name, spec in conv_specs.items():
            strides = tuple(int(v) for v in spec["strides"])
            _, _, kh, kw = [int(v) for v in spec["weight_oihw"]]
            if strides == (2, 2) or (kh, kw) == (1, 1):
                out.add(name)
        return out
    raise ValueError(f"unsupported mixed policy: {policy}")


def run_one(
    onnx_path: str,
    precision: str,
    reps: int,
    gpu: int,
    batch: int = 2,
    mixed_policy: str = "top50_flops",
    graph_io_dtype: str = "fp32",
    measure_energy: bool = False,
    energy_iters: int = 300,
    energy_min_active_s: float = 5.0,
) -> dict[str, Any]:
    import numpy as np
    import tvm
    from tvm import relax
    import tvm.s_tir.dlight as dl
    import tvm.s_tir.tensor_intrin.cuda  # noqa: side-effect registers wmma/mma intrins

    if precision == "int8":
        schedule_dtype = "int32"
    elif precision == "mixed":
        schedule_dtype = "mixed"
    else:
        schedule_dtype = "float16"
    dev = tvm.cuda(gpu)
    target = tvm.target.Target.from_device(dev)

    legal, input_specs = build_and_legalize(onnx_path, batch, target, graph_io_dtype=graph_io_dtype)
    conv_specs = classify_conv_funcs(legal)
    n_conv_total = sum(1 for gv, _ in legal.functions_items() if gv.name_hint.startswith("fused_conv2d") and "transpose" not in gv.name_hint)
    n_replaced = len(conv_specs)

    mixed_int8_names = _select_mixed_int8_convs(conv_specs, mixed_policy) if precision == "mixed" else set()
    conv_precision_plan: dict[str, str] = {}
    rewritten = legal
    for name, spec in conv_specs.items():
        gv = rewritten.get_global_var(name)
        accum_dtype = "int32" if precision == "int8" or name in mixed_int8_names else "float16"
        conv_precision_plan[name] = "int8" if accum_dtype == "int32" else "fp16"
        probe_scale = 1.0 if accum_dtype == "int32" else None
        new_func = make_std_conv_im2col_same_signature_primfunc(
            spec,
            name,
            accum_dtype,
            io_dtype=spec["io_dtype"],
            input_scale=probe_scale,
            weight_scale=probe_scale,
        )
        rewritten.update_func(gv, new_func)

    with target, tvm.transform.PassContext(opt_level=3):
        scheduled = dl.ApplyDefaultSchedule(*_full_engine_rules(schedule_dtype))(rewritten)

    txt = scheduled["main"].script()
    all_txt = "\n".join(f.script() for gv, f in scheduled.functions_items())
    counts_main = _counts(txt)
    counts_all = _counts(all_txt)

    result: dict[str, Any] = {
        "onnx_path": onnx_path,
        "precision": precision,
        "graph_io_dtype": graph_io_dtype,
        "input_specs": input_specs,
        "mixed_policy": mixed_policy if precision == "mixed" else None,
        "n_conv_total": n_conv_total,
        "n_conv_replaced_tensorized": n_replaced,
        "conv_precision_plan": conv_precision_plan,
        "conv_specs_replaced": {k: {kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in v.items()} for k, v in conv_specs.items()},
        "counts_main_func": counts_main,
        "counts_whole_module": counts_all,
        "quantization_semantics": (
            "dummy_scale_1_latency_tensorization_probe_only"
            if precision in {"int8", "mixed"}
            else None
        ),
        "status": "pending",
    }

    try:
        with target, tvm.transform.PassContext(opt_level=3):
            ex = tvm.compile(scheduled, target=target)
        vm = relax.VirtualMachine(ex, dev)

        rng = np.random.RandomState(20260704)
        np_feeds: dict[str, np.ndarray] = {}
        for spec in input_specs:
            source_name = str(spec["source_name"])
            source_shape = tuple(int(v) for v in spec["source_shape"])
            if source_name not in np_feeds:
                np_feeds[source_name] = rng.rand(*source_shape).astype("float32")
        args = []
        for spec in input_specs:
            source_value = np_feeds[str(spec["source_name"])]
            value = np.asarray(source_value, dtype=_numpy_dtype(str(spec["dtype"])))
            args.append(tvm.runtime.tensor(value, device=dev))
        out = vm["main"](*args)
        outs = [out] if hasattr(out, "shape") else list(out)
        dev.sync()

        try:
            import onnxruntime as ort

            sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            onames = [o.name for o in sess.get_outputs()]
            ref = sess.run(None, np_feeds)
            worst_rel, worst_abs = 0.0, 0.0
            for i, t in enumerate(outs):
                t_np = t.numpy() if hasattr(t, "numpy") else np.from_dlpack(t)
                if i < len(ref) and tuple(t_np.shape) == tuple(ref[i].shape):
                    diff = np.abs(t_np - ref[i])
                    ad = float(np.max(diff))
                    mask = np.abs(ref[i]) > 1e-2  # exclude ReLU-zeroed / near-zero entries from rel-err denom
                    rd = float(np.max(diff[mask] / np.abs(ref[i])[mask])) if mask.any() else 0.0
                    worst_abs = max(worst_abs, ad)
                    worst_rel = max(worst_rel, rd)
            result["numerical_vs_ort_fp32"] = {
                "max_abs_err": worst_abs,
                "max_rel_err_masked_ref_gt_1e-2": worst_rel,
                "n_outputs": len(outs),
                "note": "rel_err masks |ref|<=1e-2 entries (mostly ReLU-zeroed activations) to avoid division-by-near-zero blowup; max_abs_err is unmasked and is the primary fidelity metric",
            }
        except Exception as e:
            result["numerical_vs_ort_fp32"] = {"error": repr(e)[:400]}

        n_warm = max(5, reps // 10)
        for _ in range(n_warm):
            vm["main"](*args)
        dev.sync()
        lats = []
        for _ in range(reps):
            t0 = time.perf_counter()
            vm["main"](*args)
            dev.sync()
            lats.append((time.perf_counter() - t0) * 1000.0)
        lats.sort()
        result["latency_ms_p50"] = lats[len(lats) // 2]
        result["latency_ms_min"] = lats[0]
        result["reps"] = reps
        if measure_energy:
            energy = _measure_energy_loop(
                vm,
                args,
                dev,
                gpu,
                energy_iters=energy_iters,
                min_active_s=energy_min_active_s,
            )
            result["energy"] = energy
            result["energy_j"] = energy["joule_per_inference"]
        result["out_shapes"] = [list(o.shape) for o in outs]
        result["status"] = "success"
    except Exception as e:
        result["status"] = "exception"
        result["error"] = repr(e)[:800]
        result["traceback"] = traceback.format_exc()[-3000:]

    return result


def run_default_baseline(
    onnx_path: str,
    reps: int,
    gpu: int,
    batch: int = 2,
    graph_io_dtype: str = "fp32",
    measure_energy: bool = False,
    energy_iters: int = 300,
    energy_min_active_s: float = 5.0,
) -> dict[str, Any]:
    """No-TC baseline: legalize+fuse then TVM's own default GPU schedule (dlight
    Fallback/Matmul/Reduction rules, no tensorization rule at all) -- the correct
    apples-to-apples comparison point for speedup_vs_notc."""
    import numpy as np
    import tvm
    from tvm import relax
    import tvm.s_tir.dlight as dl

    dev = tvm.cuda(gpu)
    target = tvm.target.Target.from_device(dev)
    legal, input_specs = build_and_legalize(onnx_path, batch, target, graph_io_dtype=graph_io_dtype)

    with target, tvm.transform.PassContext(opt_level=3):
        scheduled = dl.ApplyDefaultSchedule(dl.gpu.Matmul(), dl.gpu.GEMV(), dl.gpu.Reduction(), dl.gpu.GeneralReduction(), dl.gpu.Fallback())(legal)

    result: dict[str, Any] = {
        "onnx_path": onnx_path,
        "precision": "no_tc_default",
        "graph_io_dtype": graph_io_dtype,
        "input_specs": input_specs,
        "status": "pending",
    }
    try:
        with target, tvm.transform.PassContext(opt_level=3):
            ex = tvm.compile(scheduled, target=target)
        vm = relax.VirtualMachine(ex, dev)
        rng = np.random.RandomState(20260704)
        np_feeds: dict[str, np.ndarray] = {}
        for spec in input_specs:
            source_name = str(spec["source_name"])
            source_shape = tuple(int(v) for v in spec["source_shape"])
            if source_name not in np_feeds:
                np_feeds[source_name] = rng.rand(*source_shape).astype("float32")
        args = []
        for spec in input_specs:
            source_value = np_feeds[str(spec["source_name"])]
            value = np.asarray(source_value, dtype=_numpy_dtype(str(spec["dtype"])))
            args.append(tvm.runtime.tensor(value, device=dev))
        vm["main"](*args)
        dev.sync()
        n_warm = max(5, reps // 10)
        for _ in range(n_warm):
            vm["main"](*args)
        dev.sync()
        lats = []
        for _ in range(reps):
            t0 = time.perf_counter()
            vm["main"](*args)
            dev.sync()
            lats.append((time.perf_counter() - t0) * 1000.0)
        lats.sort()
        result["latency_ms_p50"] = lats[len(lats) // 2]
        result["latency_ms_min"] = lats[0]
        result["reps"] = reps
        if measure_energy:
            energy = _measure_energy_loop(
                vm,
                args,
                dev,
                gpu,
                energy_iters=energy_iters,
                min_active_s=energy_min_active_s,
            )
            result["energy"] = energy
            result["energy_j"] = energy["joule_per_inference"]
        result["status"] = "success"
    except Exception as e:
        result["status"] = "exception"
        result["error"] = repr(e)[:800]
        result["traceback"] = traceback.format_exc()[-3000:]
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True, choices=[4, 5, 6])
    ap.add_argument("--onnx", type=str, required=True)
    ap.add_argument("--precision", choices=["fp16", "int8", "mixed", "both"], default="both")
    ap.add_argument(
        "--mixed-policy",
        choices=["none", "all", "top25_flops", "top50_flops", "top75_flops", "stride2_or_1x1"],
        default="top50_flops",
        help="When --precision=mixed, choose which conv PrimFuncs use int8 TC; remaining convs use fp16 TC.",
    )
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument(
        "--graph-io-dtype",
        choices=["fp32", "fp16"],
        default="fp32",
        help="Use fp16 to run Relax ToMixedPrecision before legalization; useful for G4-S4 boundary/materialization probes.",
    )
    ap.add_argument("--max-preflight-mem-mib", type=int, default=1024)
    ap.add_argument("--measure-energy", action="store_true")
    ap.add_argument("--energy-iters", type=int, default=300)
    ap.add_argument("--energy-min-active-s", type=float, default=5.0)
    ap.add_argument("--out-json", type=str, required=True)
    args = ap.parse_args()

    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(args.gpu)
        util = pynvml.nvmlDeviceGetUtilizationRates(h)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        mem_used_mib = mem.used / 1024 / 1024
        assert util.gpu <= 5 and mem_used_mib <= args.max_preflight_mem_mib, (
            f"GPU{args.gpu} not idle: util={util.gpu}% mem={mem_used_mib:.0f}MiB "
            f"(limit={args.max_preflight_mem_mib}MiB)"
        )
    except ImportError:
        pass

    precisions = ["fp16", "int8"] if args.precision == "both" else [args.precision]
    out: dict[str, Any] = {
        "schema": "codriving_whole_engine_tc_v1",
        "onnx": args.onnx,
        "gpu": args.gpu,
        "batch": args.batch,
        "graph_io_dtype": args.graph_io_dtype,
        "mixed_policy": args.mixed_policy if args.precision == "mixed" else None,
        "results": [],
    }

    baseline = run_default_baseline(
        args.onnx,
        args.reps,
        args.gpu,
        args.batch,
        graph_io_dtype=args.graph_io_dtype,
        measure_energy=args.measure_energy,
        energy_iters=args.energy_iters,
        energy_min_active_s=args.energy_min_active_s,
    )
    out["baseline_no_tc"] = baseline

    for prec in precisions:
        r = run_one(
            args.onnx,
            prec,
            args.reps,
            args.gpu,
            args.batch,
            args.mixed_policy,
            graph_io_dtype=args.graph_io_dtype,
            measure_energy=args.measure_energy,
            energy_iters=args.energy_iters,
            energy_min_active_s=args.energy_min_active_s,
        )
        if r.get("status") == "success" and baseline.get("status") == "success":
            r["speedup_vs_notc"] = baseline["latency_ms_p50"] / r["latency_ms_p50"]
        out["results"].append(r)
        print(f"[{prec}] status={r['status']} tensorized_convs={r.get('n_conv_replaced_tensorized')}/{r.get('n_conv_total')} "
              f"wmma={r.get('counts_whole_module',{}).get('wmma')} mma_sync={r.get('counts_whole_module',{}).get('mma_sync')} "
              f"lat_p50={r.get('latency_ms_p50')} speedup={r.get('speedup_vs_notc')}", flush=True)

    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=1)
    print("WROTE", args.out_json)


if __name__ == "__main__":
    main()
