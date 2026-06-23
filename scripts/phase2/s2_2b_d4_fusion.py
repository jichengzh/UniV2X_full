"""S2.2b option-2: D4 fusion x Q/DQ-placement coupling (exploratory).

Hypothesis (dims_hardware_v4 §2 T3-D4, mirrors our TRT P4 observation): a
quantize/dequantize boundary inserted between layers BREAKS operator fusion,
forcing an extra activation round-trip to memory -> extra traffic that can
offset the quant compute saving. The optimal fusion grouping therefore depends
on where Q/DQ is placed (= the quantization-granularity decision), which is the
prune/quant x fusion coupling.

relax 0.20 has no INT8 quant pass (R1), so we express Q/DQ as explicit
elementwise ops (round/clip/astype) -- a faithful proxy for a per-tensor QDQ
pair. We build a conv->relu->conv chain in two regimes and compare:
  FUSE : conv1 -> relu -> conv2                       (no quant boundary)
  QDQ  : conv1 -> relu -> quantize -> dequantize -> conv2

Metric = number of fused TIR kernels (PrimFuncs after FuseOps+FuseTIR) and clean
latency. If QDQ raises kernel count / latency, the fusion-break coupling is real.

Usage: python s2_2b_d4_fusion.py <regime fuse|qdq> <C> <H> <W> <work_root> <label> [trials]
"""
from __future__ import annotations
import sys, os, time, traceback

REGIME = sys.argv[1]; C = int(sys.argv[2]); H = int(sys.argv[3]); W = int(sys.argv[4])
WORK_ROOT = sys.argv[5]; LABEL = sys.argv[6]
TRIALS = int(sys.argv[7]) if len(sys.argv) > 7 else 64


def build_chain(regime, c, h, w):
    import tvm
    from tvm import relax
    from tvm.relax.op import nn, astype, multiply, round as r_round, clip
    bb = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorStructInfo((1, c, h, w), "float16"))
    w1 = relax.Var("w1", relax.TensorStructInfo((c, c, 3, 3), "float16"))
    w2 = relax.Var("w2", relax.TensorStructInfo((c, c, 3, 3), "float16"))
    with bb.function("main", [x, w1, w2]):
        with bb.dataflow():
            y = bb.emit(nn.conv2d(x, w1, padding=(1, 1), out_dtype="float16"))
            y = bb.emit(nn.relu(y))
            if regime == "qdq":
                # quantize: clip(round(y*scale)) -> int8 ; dequantize: int8->fp16 * inv
                q = bb.emit(multiply(y, relax.const(8.0, "float16")))
                q = bb.emit(r_round(q))
                q = bb.emit(clip(q, -128.0, 127.0))
                q = bb.emit(astype(q, "int8"))
                y = bb.emit(astype(q, "float16"))
                y = bb.emit(multiply(y, relax.const(0.125, "float16")))
            y = bb.emit(nn.conv2d(y, w2, padding=(1, 1), out_dtype="float16"))
            gv = bb.emit_output(y)
        bb.emit_func_output(gv)
    return bb.finalize()


def count_primfuncs(mod):
    import tvm
    n = 0; names = []
    for gv, func in mod.functions.items():
        if isinstance(func, tvm.tir.PrimFunc) if hasattr(tvm, "tir") else False:
            n += 1; names.append(gv.name_hint)
    # robust: count by type name (renamed Unity build)
    n2 = 0; names2 = []
    for gv, func in mod.functions.items():
        if type(func).__name__ == "PrimFunc":
            n2 += 1; names2.append(gv.name_hint)
    return (n2 or n), (names2 or names)


def main():
    import tvm
    from tvm import relax
    from tvm.s_tir.meta_schedule import relax_integration as ri
    import tvm.s_tir.tensor_intrin.cuda  # noqa

    print(f"[{LABEL}] D4 regime={REGIME} C={C} {H}x{W} trials={TRIALS}", flush=True)
    work_dir = os.path.join(WORK_ROOT, LABEL); os.makedirs(work_dir, exist_ok=True)
    dev = tvm.cuda(0)
    target = tvm.target.Target.from_device(dev)
    mod = build_chain(REGIME, C, H, W)
    seq = tvm.transform.Sequential([
        relax.transform.LegalizeOps(), relax.transform.AnnotateTIROpPattern(),
        relax.transform.FuseOps(), relax.transform.FuseTIR(),
    ])
    with target, tvm.transform.PassContext(opt_level=3):
        mod = seq(mod)
    nfunc, names = count_primfuncs(mod)
    # count call_tir sites in main = number of kernel launches
    main_txt = str(mod["main"]) if "main" in [g.name_hint for g in mod.functions] else ""
    n_calltir = main_txt.count("call_tir")
    print(f"[{LABEL}] fused_kernels(PrimFuncs)={nfunc} call_tir_sites={n_calltir} names={names}", flush=True)

    t0 = time.time()
    db = ri.tune_relax(mod=mod, params={}, target=target, work_dir=work_dir,
                       max_trials_global=TRIALS, seed=0)
    print(f"[{LABEL}] tune done {time.time()-t0:.0f}s", flush=True)
    best = None
    for r in db.get_all_tuning_records():
        rs = getattr(r, "run_secs", None)
        if rs:
            vals = [float(x) for x in rs if x is not None]
            if vals:
                m = sum(vals)/len(vals); best = m if best is None or m < best else best
    best_us = best*1e6 if best else -1.0
    print(f"[{LABEL}] RESULT label={LABEL} regime={REGIME} fused_kernels={nfunc} "
          f"call_tir_sites={n_calltir} best_us={best_us:.3f}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        traceback.print_exc(); print(f"[{LABEL}] EXCEPTION {repr(e)}"); sys.exit(3)
