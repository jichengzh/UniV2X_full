#!/usr/bin/env python3
"""INT8 Tensor-Core feasibility gate for the Pyramid group-conv bottleneck.

Question this gate answers (before any 60-width rollout):
  Can the SAME im2col->batched-matmul rewrite that made fp16 hit WMMA (6.08ms)
  be built in int8 so it tensorizes to the int8 MMA path (m16n16k32 s8s8s32)
  and beats the naive un-tensorized int8 conv (schedule_block, 8.73ms)?

Method (mirrors the fp16 rewrite's proven pipeline, dtype swapped to int8):
  group conv -> im2col batched matmul; operands int8, accumulate int32; then
  dlight MatmulTensorization (auto-selects the int8 MMA intrinsic). Latency-only
  (dummy quant scales) -> AP unaffected (AP comes from the real-activation bridge).

Usage:
  python stage2_int8_tensorcore_gate_v1.py --gpu 3 --reps 200 \
      --input-nchw 2,256,128,256 --weight-oihw 256,8,3,3 --groups 32
"""
from __future__ import annotations

import argparse
import json
import time
from typing import Any


def _conv2d_out_hw(h, w, kh, kw, strides, padding):
    pt, pl, pb, pr = padding
    sh, sw = strides
    oh = (h + pt + pb - kh) // sh + 1
    ow = (w + pl + pr - kw) // sw + 1
    return oh, ow


def make_int8_im2col_matmul_primfunc(spec: dict[str, Any], name: str) -> Any:
    """int8xint8->int32 batched-matmul form of a group conv (fp16 I/O signature).

    Heavy matmul runs on int8 tensor cores; im2col columns/weights are cast to
    int8 (dummy per-tensor scale=1, latency-only), accumulate int32, cast back
    to fp16 on output so the block plugs into the surrounding fp16 graph exactly
    like the fp16 rewrite does.
    """
    from tvm import te

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

    def im2col_compute(g, row, kk):
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

    x_col_f16 = te.compute((groups, m, k_total), im2col_compute, name="x_col_f16")
    # dummy quant to int8 (scale=1) — latency-only
    x_col = te.compute((groups, m, k_total),
                       lambda g, r, kk: x_col_f16[g, r, kk].astype("int8"), name="x_col")

    def weight_compute(g, kk, ocg):
        ci = kk // (kh * kw)
        rem_k = kk % (kh * kw)
        ry = rem_k // kw
        rx = rem_k % kw
        return weight[g * nper + ocg, ci, ry, rx]

    w_mat_f16 = te.compute((groups, k_total, nper), weight_compute, name="w_mat_f16")
    w_mat = te.compute((groups, k_total, nper),
                       lambda g, kk, o: w_mat_f16[g, kk, o].astype("int8"), name="w_mat")

    rk = te.reduce_axis((0, k_total), name="rk")
    # canonical int8 matmul: int8 operands cast to int32 in the product, reduce int32
    matmul = te.compute(
        (groups, m, nper),
        lambda g, row, ocg: te.sum(
            x_col[g, row, rk].astype("int32") * w_mat[g, rk, ocg].astype("int32"), axis=rk),
        name="matmul",
    )

    def out_compute(nn, oc, yy, xx):
        g = oc // nper
        ocg = oc % nper
        row = nn * (oh * ow) + yy * ow + xx
        value = matmul[g, row, ocg].astype("float16") + bias[0, oc, 0, 0]
        if bool(spec.get("relu", False)):
            value = te.max(value, te.const(0, "float16"))
        return value

    out = te.compute((n, cout, oh, ow), out_compute, name="out_nchw")
    return te.create_prim_func([x, weight, bias, out]).with_attr("global_symbol", name)


def make_replacement_mod(spec: dict[str, Any]) -> Any:
    from tvm import relax

    n, cin, h, w = [int(v) for v in spec["input_nchw"]]
    cout, cpg, kh, kw = [int(v) for v in spec["weight_oihw"]]
    oh, ow = _conv2d_out_hw(h, w, kh, kw, tuple(spec["strides"]), tuple(spec["padding"]))
    bb = relax.BlockBuilder()
    prim_gv = bb.add_func(make_int8_im2col_matmul_primfunc(spec, "int8_group_conv"), "int8_group_conv")
    x = relax.Var("x", relax.TensorStructInfo((n, cin, h, w), "float16"))
    weight = relax.Var("weight", relax.TensorStructInfo((cout, cpg, kh, kw), "float16"))
    bias = relax.Var("bias", relax.TensorStructInfo(tuple(spec["bias"]), "float16"))
    with bb.function("main", [x, weight, bias]):
        with bb.dataflow():
            out = bb.emit(relax.call_tir(prim_gv, [x, weight, bias],
                          relax.TensorStructInfo((n, cout, oh, ow), "float16")))
            gv = bb.emit_output(out)
        bb.emit_func_output(gv)
    return bb.finalize()


def _counts(text: str) -> dict[str, int]:
    keys = ["wmma", "tvm_mma_sync", "mma_sync", "ldmatrix", "mma", "dp4a", "__dp4a"]
    return {k: text.count(k) for k in keys}


def main() -> int:
    import numpy as np
    import tvm
    from tvm import relax
    import tvm.s_tir.dlight as dl
    from tvm.s_tir.dlight.gpu.matmul import MatmulInt8Tensorization

    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--input-nchw", default="2,256,128,256")
    ap.add_argument("--weight-oihw", default="256,8,3,3")
    ap.add_argument("--groups", type=int, default=32)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    spec = {
        "input_nchw": [int(v) for v in args.input_nchw.split(",")],
        "weight_oihw": [int(v) for v in args.weight_oihw.split(",")],
        "groups": args.groups, "strides": [1, 1], "padding": [1, 1, 1, 1],
        "bias": [1, int(args.weight_oihw.split(",")[0]), 1, 1], "relu": True,
    }
    dev = tvm.cuda(args.gpu)
    target = tvm.target.Target.from_device(dev)
    result: dict[str, Any] = {"spec": spec, "status": "started"}
    try:
        mod = make_replacement_mod(spec)
        seq = tvm.transform.Sequential([
            relax.transform.LegalizeOps(), relax.transform.AnnotateTIROpPattern(),
            relax.transform.FuseOps(), relax.transform.FuseTIR()])
        with target, tvm.transform.PassContext(opt_level=3):
            legal = seq(mod)
        # apply MatmulTensorization to the matmul primfunc(s)
        out = tvm.IRModule(dict(legal.functions), attrs=legal.attrs)
        tensorized = False
        diag = []
        for gv, func in legal.functions_items():
            if not hasattr(func, "script"):
                continue
            txt = func.script()
            has_mm = ('sblock("matmul")' in txt) or ('block("matmul")' in txt) or ("matmul" in gv.name_hint)
            diag.append({"func": gv.name_hint, "has_matmul_block": has_mm,
                         "pre_counts": _counts(txt)})
            if not has_mm:
                continue
            single = tvm.IRModule({gv: func})
            with target, tvm.transform.PassContext(opt_level=3):
                sched = dl.ApplyDefaultSchedule(MatmulInt8Tensorization())(single)
            counts = _counts(sched[gv].script())
            result["tensorized_counts"] = counts
            diag[-1]["post_counts"] = counts
            if counts.get("tvm_mma_sync", 0) > 0 or counts.get("mma_sync", 0) > 0 or counts.get("wmma", 0) > 0:
                out.update_func(gv, sched[gv])
                tensorized = True
        result["diag"] = diag
        # fill any remaining un-scheduled prims with a default gpu schedule
        with target, tvm.transform.PassContext(opt_level=3):
            out = dl.ApplyDefaultSchedule(dl.gpu.Fallback())(out)
            ex = tvm.compile(out, target=target)
        vm = relax.VirtualMachine(ex, dev)
        n, cin, h, w = spec["input_nchw"]
        cout, cpg, kh, kw = spec["weight_oihw"]
        feeds = [
            tvm.runtime.tensor(np.random.randn(n, cin, h, w).astype("float16"), dev),
            tvm.runtime.tensor(np.random.randn(cout, cpg, kh, kw).astype("float16"), dev),
            tvm.runtime.tensor(np.random.randn(*spec["bias"]).astype("float16"), dev),
        ]
        for _ in range(20):
            vm["main"](*feeds)
        dev.sync()
        ts = []
        for _ in range(args.reps):
            t0 = time.perf_counter(); vm["main"](*feeds); dev.sync()
            ts.append((time.perf_counter() - t0) * 1e3)
        ts.sort()
        result.update({"status": "success", "tensorized": tensorized,
                       "latency_ms_p50": ts[len(ts) // 2], "latency_ms_min": ts[0]})
    except Exception as exc:
        import traceback
        result.update({"status": "failed", "error": repr(exc), "traceback": traceback.format_exc()})
    print(json.dumps({k: v for k, v in result.items() if k != "traceback"}, indent=1))
    if result.get("status") == "failed":
        print(result.get("traceback", ""))
    if args.out_json:
        open(args.out_json, "w").write(json.dumps(result, indent=1))
    return 0 if result.get("status") == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
