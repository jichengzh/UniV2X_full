#!/usr/bin/env python3
"""De-risking milestone: prove CoDriving's STANDARD (groups=1) conv can be
lowered to GPU Tensor Cores via TVM, in both fp16 and int8.

CoDriving's backbone (ResNetModified/BasicBlock, opencood/models/sub_modules/
resblock.py) uses plain groups=1 3x3/1x1 conv -- unlike Pyramid's grouped
bottleneck conv. dlight's default schedule for a standard conv2d does NOT
tensorize (fp16 speed == fp32 speed on CoDriving per prior HANDOFF). This
script rebuilds one real CoDriving conv layer as an im2col + batched-matmul
primfunc (groups=1 is just degenerate groups dim) and applies dlight's
MatmulTensorization (fp16) / MatmulInt8Tensorization (int8) rule directly,
then PROVES tensorization by literally counting wmma/mma_sync tokens in the
scheduled primfunc's own script() text (not inferred from MS-DB traces).

fp16 path is checked for numerical fidelity against a real fp32 NCHW conv
reference (max_rel_err should be ~0). int8 path here uses dummy per-tensor
scale=1 (latency/tensorization-only, per project convention) -- it is NOT a
claim of real quantized accuracy; real int8 AP requires a calibrated
real-activation bridge (separate follow-on step), and this script never
reports an AP number.

Usage (H800, GPU 4-6 only, confirm idle via nvidia-smi first):
  python stage2_codriving_tc_derisk_v1.py --gpu 4 --layer stage0_conv2_base \
      --reps 200 --out-json results/plan3_codriving_tc/derisk_stage0_conv2_base.json
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from typing import Any


# Real CoDriving BasicBlock conv shapes (groups=1), base width num_filters=[64,128,256].
# Derived from opencood/models/sub_modules/resblock.py (BasicBlock/ResNetModified) +
# codriving_multiclass_config.yaml (layer_nums=[3,4,5], layer_strides=[2,2,2]).
# spatial_features input (2,64,256,512) -> stage0 out (2,64,128,256) -> stage1 out
# (2,128,64,128) -> stage2 out (2,256,32,64).
CODRIVING_LAYERS: dict[str, dict[str, Any]] = {
    "stage0_conv2_base": {
        "input_nchw": (2, 64, 128, 256), "weight_oihw": (64, 64, 3, 3),
        "strides": (1, 1), "padding": (1, 1, 1, 1), "bias": (1, 64, 1, 1), "relu": True,
    },
    "stage0_conv1_base": {
        # first block's conv1 takes inplanes=64 at full (256,512) res, stride 2
        "input_nchw": (2, 64, 256, 512), "weight_oihw": (64, 64, 3, 3),
        "strides": (2, 2), "padding": (1, 1, 1, 1), "bias": (1, 64, 1, 1), "relu": True,
    },
    "stage0_downsample_base": {
        "input_nchw": (2, 64, 256, 512), "weight_oihw": (64, 64, 1, 1),
        "strides": (2, 2), "padding": (0, 0, 0, 0), "bias": (1, 64, 1, 1), "relu": False,
    },
    "stage1_conv1_base": {
        "input_nchw": (2, 64, 128, 256), "weight_oihw": (128, 64, 3, 3),
        "strides": (2, 2), "padding": (1, 1, 1, 1), "bias": (1, 128, 1, 1), "relu": True,
    },
    "stage1_conv2_base": {
        "input_nchw": (2, 128, 64, 128), "weight_oihw": (128, 128, 3, 3),
        "strides": (1, 1), "padding": (1, 1, 1, 1), "bias": (1, 128, 1, 1), "relu": True,
    },
    "stage2_conv1_base": {
        "input_nchw": (2, 128, 64, 128), "weight_oihw": (256, 128, 3, 3),
        "strides": (2, 2), "padding": (1, 1, 1, 1), "bias": (1, 256, 1, 1), "relu": True,
    },
    "stage2_conv2_base": {
        "input_nchw": (2, 256, 32, 64), "weight_oihw": (256, 256, 3, 3),
        "strides": (1, 1), "padding": (1, 1, 1, 1), "bias": (1, 256, 1, 1), "relu": True,
    },
    # p25 width (num_filters=[48,96,192]) stage0, cross-checkable against
    # results/codriving_int8_verify.json cin48_p25_s0 entry.
    "stage0_conv2_p25": {
        "input_nchw": (2, 48, 128, 256), "weight_oihw": (48, 48, 3, 3),
        "strides": (1, 1), "padding": (1, 1, 1, 1), "bias": (1, 48, 1, 1), "relu": True,
    },
}


def _conv2d_out_hw(h: int, w: int, kh: int, kw: int, strides, padding) -> tuple[int, int]:
    pt, pl, pb, pr = padding
    sh, sw = strides
    return (h + pt + pb - kh) // sh + 1, (w + pl + pr - kw) // sw + 1


def make_std_conv_im2col_same_signature_primfunc(spec: dict[str, Any], name: str, accum_dtype: str) -> Any:
    """groups=1 specialization of the proven Pyramid full-engine im2col+matmul
    same-signature primfunc (stage2_fp16_tensorcore_convblock_and_engine_probe.py
    ::_make_te_group_conv_full_im2col_same_signature_primfunc). groups is left as
    an explicit (degenerate =1) tensor dim so the exact same dlight tensorization
    code path applies unmodified."""
    from tvm import te

    if accum_dtype not in {"float16", "float32", "int32"}:
        raise ValueError(f"unsupported_accum_dtype:{accum_dtype}")
    int8_mma = accum_dtype == "int32"

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

    if int8_mma:
        x_col_q = te.compute((groups, m, k_total), lambda g, r, kk: x_col[g, r, kk].astype("int8"), name="x_col_q")
        w_mat_q = te.compute((groups, k_total, nper), lambda g, kk, o: w_mat[g, kk, o].astype("int8"), name="w_mat_q")

    rk = te.reduce_axis((0, k_total), name="rk")

    def matmul_compute(g: Any, row: Any, ocg: Any) -> Any:
        if int8_mma:
            return te.sum(x_col_q[g, row, rk].astype("int32") * w_mat_q[g, rk, ocg].astype("int32"), axis=rk)
        if accum_dtype == "float32":
            return te.sum(x_col[g, row, rk].astype("float32") * w_mat[g, rk, ocg].astype("float32"), axis=rk)
        return te.sum(x_col[g, row, rk] * w_mat[g, rk, ocg], axis=rk)

    matmul = te.compute((groups, m, nper), matmul_compute, name="matmul")

    def out_compute(nn: Any, oc: Any, yy: Any, xx: Any) -> Any:
        g = oc // nper
        ocg = oc % nper
        row = nn * (oh * ow) + yy * ow + xx
        if int8_mma:
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


def _counts(text: str) -> dict[str, int]:
    keys = ["wmma", "tvm_mma_sync", "mma_sync", "ldmatrix", "mma", "dp4a", "__dp4a"]
    return {k: text.count(k) for k in keys}


def _numpy_conv2d_ref(x, w, b, strides, padding, relu: bool):
    import numpy as np

    n, cin, h, width = x.shape
    cout, cpg, kh, kw = w.shape
    assert cpg == cin
    pt, pl, pb, pr = padding
    sh, sw = strides
    xp = np.pad(x, ((0, 0), (0, 0), (pt, pb), (pl, pr)), mode="constant")
    oh = (h + pt + pb - kh) // sh + 1
    ow = (width + pl + pr - kw) // sw + 1
    out = np.zeros((n, cout, oh, ow), dtype=np.float32)
    for yy in range(oh):
        for xx in range(ow):
            patch = xp[:, :, yy * sh:yy * sh + kh, xx * sw:xx * sw + kw]  # (n,cin,kh,kw)
            out[:, :, yy, xx] = np.einsum("ncij,ocij->no", patch, w)
    out = out + b.reshape(1, cout, 1, 1)
    if relu:
        out = np.maximum(out, 0.0)
    return out


def run_one(layer_name: str, spec: dict[str, Any], precision: str, reps: int, gpu: int) -> dict[str, Any]:
    import numpy as np
    import tvm
    from tvm import relax
    import tvm.s_tir.dlight as dl

    accum_dtype = "int32" if precision == "int8" else "float16"
    dev = tvm.cuda(gpu)
    target = tvm.target.Target.from_device(dev)
    result: dict[str, Any] = {"layer": layer_name, "spec": spec, "precision": precision, "status": "started"}
    try:
        prim = make_std_conv_im2col_same_signature_primfunc(spec, "codriving_std_conv", accum_dtype)
        prim_gv = tvm.ir.GlobalVar("codriving_std_conv")
        pre_counts = _counts(prim.script())
        result["pre_schedule_counts"] = pre_counts

        if precision == "int8":
            from tvm.s_tir.dlight.gpu.matmul import MatmulInt8Tensorization as Rule
        else:
            from tvm.s_tir.dlight.gpu.matmul import MatmulTensorization as Rule

        # schedule the bare TIR primfunc in isolation (this is what the counted
        # proof below refers to), then wrap it in a relax module with a `main`
        # entry so tvm.compile emits a VM-loadable executable (relax.build/compile
        # only attaches VM metadata for a Relax IRModule, not a bare-TIR one).
        tir_only_mod = tvm.IRModule({prim_gv: prim})
        with target, tvm.transform.PassContext(opt_level=3):
            scheduled_tir_only = dl.ApplyDefaultSchedule(Rule())(tir_only_mod)
        scheduled_func = scheduled_tir_only[prim_gv]
        scheduled_text = scheduled_func.script()
        post_counts = _counts(scheduled_text)
        result["post_schedule_counts"] = post_counts
        tensorized = post_counts.get("wmma", 0) > 0 and (
            post_counts.get("tvm_mma_sync", 0) > 0 or post_counts.get("mma_sync", 0) > 0
        )
        result["tensorized"] = bool(tensorized)

        n, cin, h, width = spec["input_nchw"]
        cout, cpg, kh, kw = spec["weight_oihw"]
        x_var = relax.Var("x", relax.TensorStructInfo((n, cin, h, width), "float16"))
        w_var = relax.Var("weight", relax.TensorStructInfo((cout, cpg, kh, kw), "float16"))
        b_var = relax.Var("bias", relax.TensorStructInfo((1, cout, 1, 1), "float16"))
        oh, ow = _conv2d_out_hw(h, width, kh, kw, tuple(spec["strides"]), tuple(spec["padding"]))
        bb = relax.BlockBuilder()
        conv_gv = bb.add_func(scheduled_func, "codriving_std_conv")
        with bb.function("main", [x_var, w_var, b_var]):
            with bb.dataflow():
                out = bb.emit(relax.call_tir(conv_gv, [x_var, w_var, b_var],
                              relax.TensorStructInfo((n, cout, oh, ow), "float16")))
                gv_out = bb.emit_output(out)
            bb.emit_func_output(gv_out)
        relax_mod = bb.finalize()

        with target, tvm.transform.PassContext(opt_level=3):
            executable = tvm.compile(relax_mod, target=target)
        vm = relax.VirtualMachine(executable, dev)

        n, cin, h, width = spec["input_nchw"]
        cout, cpg, kh, kw = spec["weight_oihw"]
        rng = np.random.RandomState(20260704)
        x_np = ((rng.rand(n, cin, h, width).astype("float32") - 0.5) * 0.2).astype("float16")
        w_np = ((rng.rand(cout, cpg, kh, kw).astype("float32") - 0.5) * 0.2).astype("float16")
        b_np = ((rng.rand(cout).astype("float32") - 0.5) * 0.1).astype("float16")
        b_np4 = b_np.reshape(1, cout, 1, 1)

        feeds = [
            tvm.runtime.tensor(x_np, dev),
            tvm.runtime.tensor(w_np, dev),
            tvm.runtime.tensor(b_np4, dev),
        ]
        for _ in range(20):
            vm["main"](*feeds)
        dev.sync()
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            vm["main"](*feeds)
            dev.sync()
            ts.append((time.perf_counter() - t0) * 1e3)
        ts.sort()
        out_tvm = vm["main"](*feeds).numpy().astype("float32")

        if precision == "fp16":
            ref = _numpy_conv2d_ref(
                x_np.astype("float32"), w_np.astype("float32"), b_np.astype("float32"),
                tuple(spec["strides"]), tuple(spec["padding"]), bool(spec.get("relu", False)),
            )
            max_abs_err = float(np.max(np.abs(out_tvm - ref)))
            denom = float(np.max(np.abs(ref))) + 1e-8
            max_rel_err = max_abs_err / denom
            result["numerical"] = {
                "max_abs_err": max_abs_err,
                "max_rel_err": max_rel_err,
                "ref_max_abs": denom,
                "note": "fp16-TC vs fp32 NCHW conv reference (numpy einsum), same random x/w/b seed=20260704",
            }
        else:
            result["numerical"] = {
                "note": "int8-TC uses dummy per-tensor scale=1 cast (latency/tensorization-only per project "
                        "convention) -- NOT a real-quant accuracy claim. Real int8 AP requires a separate "
                        "calibrated real-activation bridge (not run in this de-risking probe).",
            }

        result.update({
            "status": "success",
            "latency_ms_p50": ts[len(ts) // 2],
            "latency_ms_min": ts[0],
            "reps": reps,
        })
    except Exception as exc:
        result.update({"status": "failed", "error": repr(exc), "traceback": traceback.format_exc()})
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True, choices=[4, 5, 6])
    ap.add_argument("--layer", default="stage0_conv2_base", choices=list(CODRIVING_LAYERS.keys()))
    ap.add_argument("--precision", default="both", choices=["fp16", "int8", "both"])
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    spec = CODRIVING_LAYERS[args.layer]
    precisions = ["fp16", "int8"] if args.precision == "both" else [args.precision]
    results = [run_one(args.layer, spec, prec, args.reps, args.gpu) for prec in precisions]
    payload = {"schema": "codriving_tc_derisk_v1", "layer": args.layer, "gpu": args.gpu, "results": results}
    print(json.dumps({k: v for k, v in payload.items()}, indent=1, default=str))
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(payload, f, indent=1, default=str)
    ok = all(r.get("status") == "success" for r in results)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
