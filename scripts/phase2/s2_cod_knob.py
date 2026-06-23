"""P4 (CoDriving) per-knob scan on REAL conv shapes — L1/L2/L3/L4.

Mirrors s2_2f_gemm_knob.py but uses the real CoDriving 3x3-conv im2col-GEMM dims
(M=2048 spatial, N=W output channels, K=9*W reduction depth) instead of the
model-agnostic square (K=N=W). For each knob, vary ONLY that knob over the pruning
width W in {64,128,256}; argmin(knob|W) shift => that knob couples with pruning.

Knobs: staging(L1 shared) / ttile(L2 thread tile) / align(L3 bank pad) / pipe(L4 sw-pipeline)
Usage: python s2_cod_knob.py <KNOB> <SETTING> <W> <out_csv> [reps]
"""
from __future__ import annotations
import sys, os, traceback

KNOB = sys.argv[1]; SETTING = int(sys.argv[2]); W = int(sys.argv[3])
OUT = sys.argv[4]; REPS = int(sys.argv[5]) if len(sys.argv) > 5 else 1000
M = 2048; N = W; K = 9 * W           # real 3x3 conv im2col GEMM

TM = TN = 16; BK = 16; SHARED = 1; ALIGN = 0; PIPE = 0
if KNOB == "staging":   SHARED = SETTING
elif KNOB == "ttile":   TM = TN = SETTING
elif KNOB == "align":   ALIGN = SETTING
elif KNOB == "pipe":    PIPE = SETTING
else:
    print(f"unknown knob {KNOB}"); sys.exit(2)


def record(us, status):
    hdr = not os.path.exists(OUT)
    with open(OUT, "a") as f:
        if hdr:
            f.write("knob,setting,W,K,N,TM,TN,BK,shared,align,pipe,status,mean_us\n")
        f.write(f"{KNOB},{SETTING},{W},{K},{N},{TM},{TN},{BK},{SHARED},{ALIGN},{PIPE},{status},{us:.3f}\n")


def valid():
    if M % TM or N % TN or K % BK:
        return False, "indivisible"
    if TM * TN > 1024:
        return False, "too_many_threads"
    return True, "ok"


def build():
    import tvm
    from tvm import relax, s_tir
    bb = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorStructInfo((M, K), "float32"))
    w = relax.Var("w", relax.TensorStructInfo((K, N), "float32"))
    with bb.function("main", [x, w]):
        with bb.dataflow():
            y = bb.emit(relax.op.matmul(x, w, out_dtype="float32")); gv = bb.emit_output(y)
        bb.emit_func_output(gv)
    mod = tvm.transform.Sequential([relax.transform.LegalizeOps(),
                                    relax.transform.FuseTIR()])(bb.finalize())
    sch = s_tir.Schedule(mod)
    mm = sch.get_sblock("matmul", func_name="matmul")
    i, j, k = sch.get_loops(mm)
    io, ii = sch.split(i, factors=[None, TM]); jo, ji = sch.split(j, factors=[None, TN])
    if not SHARED:
        sch.reorder(io, jo, ii, ji, k)
        sch.bind(io, "blockIdx.y"); sch.bind(jo, "blockIdx.x")
        sch.bind(ii, "threadIdx.y"); sch.bind(ji, "threadIdx.x")
    else:
        ko, ki = sch.split(k, factors=[None, BK])
        sch.reorder(io, jo, ii, ji, ko, ki)
        sch.bind(io, "blockIdx.y"); sch.bind(jo, "blockIdx.x")
        sch.bind(ii, "threadIdx.y"); sch.bind(ji, "threadIdx.x")
        a_sh = sch.cache_read(mm, 0, "shared"); sch.compute_at(a_sh, ko)
        b_sh = sch.cache_read(mm, 1, "shared"); sch.compute_at(b_sh, ko)
        for sh in (a_sh, b_sh):
            ls = sch.get_loops(sh); fused = sch.fuse(*ls[-2:])
            _, ty, tx = sch.split(fused, factors=[None, TM, TN])
            sch.bind(ty, "threadIdx.y"); sch.bind(tx, "threadIdx.x")
        if ALIGN > 0:
            sch.storage_align(a_sh, 0, axis=-2, factor=32, offset=ALIGN)
            sch.storage_align(b_sh, 0, axis=-2, factor=32, offset=ALIGN)
        if PIPE > 0:
            sch.annotate(ko, "software_pipeline_stage", [0, 0, 1])
            sch.annotate(ko, "software_pipeline_order", [0, 1, 2])
    m2 = sch.mod
    gv = [g for g, f in m2.functions.items() if type(f).__name__ == "PrimFunc"][0]
    m2 = m2.clone(); m2.update_func(gv, m2[gv].with_attr("tirx.is_scheduled", True))
    return m2


def main():
    import tvm, numpy as np
    from tvm import relax
    ok, why = valid()
    print(f"[{KNOB}={SETTING} W={W}] K={K} N={N} tile=({TM},{TN},{BK}) shared={SHARED} align={ALIGN} pipe={PIPE} valid={ok}", flush=True)
    if not ok:
        record(-1.0, why); print(f"SKIP {why}", flush=True); return
    us = -1.0; status = "ok"
    try:
        m2 = build(); dev = tvm.cuda(0); tgt = tvm.target.Target.from_device(dev)
        ex = tvm.compile(m2, target=tgt); vm = relax.VirtualMachine(ex, dev)
        a = tvm.runtime.tensor(np.random.randn(M, K).astype("float32"), device=dev)
        b = tvm.runtime.tensor(np.random.randn(K, N).astype("float32"), device=dev)
        vm["main"](a, b); dev.sync()
        vf = vm.time_evaluator("main", dev, number=REPS, repeat=5); r = vf(a, b)
        us = r.mean * 1e6
        print(f"[{KNOB}={SETTING} W={W}] mean_us={us:.3f} min_us={min(r.results)*1e6:.3f}", flush=True)
    except Exception as e:
        status = "build_fail"; traceback.print_exc(); print(f"FAILED {repr(e)[:120]}", flush=True)
    record(us, status); print("DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        traceback.print_exc(); print(f"EXCEPTION {repr(e)}"); sys.exit(3)
