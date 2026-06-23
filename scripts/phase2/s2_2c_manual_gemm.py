"""S2.2c rigorous per-knob ablation via DETERMINISTIC manual GEMM scheduling.

The rule-drop ablation was confounded (search-space size vs fixed budget:
cuda_full was SLOWEST). Here we remove the tuner: hand-build ONE fp32 GEMM
schedule, add exactly one knob per level, compile (marking tirx.is_scheduled so
dlight skips re-scheduling) and time on an idle GPU. Marginal latency drop
level i->i+1 = that knob's contribution. No search => no budget confound.

Cumulative levels (each adds one knob over the previous):
  0 bind   : block+thread binding, k serial, reads from global   [L2 bind, foundational]
  1 shared : + cache_read A,B into shared memory (staging)        [L1 memory staging]
  2 align  : + storage_align on shared (bank-conflict padding)    [L3 bank-conflict]
  3 pipe   : + software_pipeline on K loop (prefetch overlap)     [L4 software pipeline]

Usage: python s2_2c_manual_gemm.py <M> <N> <K> <level 0..3> <label> <out_csv> [reps]
"""
from __future__ import annotations
import sys, os, traceback

M = int(sys.argv[1]); N = int(sys.argv[2]); K = int(sys.argv[3])
LEVEL = int(sys.argv[4]); LABEL = sys.argv[5]; OUT_CSV = sys.argv[6]
REPS = int(sys.argv[7]) if len(sys.argv) > 7 else 1000

TM, TN, BK = 16, 16, 16   # thread grid + reduction tile


def build_relax_mod():
    import tvm
    from tvm import relax
    bb = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorStructInfo((M, K), "float32"))
    w = relax.Var("w", relax.TensorStructInfo((K, N), "float32"))
    with bb.function("main", [x, w]):
        with bb.dataflow():
            y = bb.emit(relax.op.matmul(x, w, out_dtype="float32")); gv = bb.emit_output(y)
        bb.emit_func_output(gv)
    return tvm.transform.Sequential([relax.transform.LegalizeOps(),
                                     relax.transform.FuseTIR()])(bb.finalize())


def schedule_mod(level):
    import tvm
    from tvm import s_tir
    mod = build_relax_mod()
    sch = s_tir.Schedule(mod)
    mm = sch.get_sblock("matmul", func_name="matmul")
    i, j, k = sch.get_loops(mm)
    io, ii = sch.split(i, factors=[None, TM])
    jo, ji = sch.split(j, factors=[None, TN])
    if level == 0:
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
        # cooperative fetch: bind the shared-load tile loops across the TMxTN
        # thread grid so threads load shared collaboratively (NOT serially).
        for sh in (a_sh, b_sh):
            ls = sch.get_loops(sh)
            fused = sch.fuse(*ls[-2:])
            _, ty, tx = sch.split(fused, factors=[None, TM, TN])
            sch.bind(ty, "threadIdx.y"); sch.bind(tx, "threadIdx.x")
        if level >= 2:
            sch.storage_align(a_sh, 0, axis=-2, factor=32, offset=8)
            sch.storage_align(b_sh, 0, axis=-2, factor=32, offset=8)
        if level >= 3:
            sch.annotate(ko, "software_pipeline_stage", [0, 0, 1])
            sch.annotate(ko, "software_pipeline_order", [0, 1, 2])
    m2 = sch.mod
    gv = [g for g, f in m2.functions.items() if type(f).__name__ == "PrimFunc"][0]
    m2 = m2.clone(); m2.update_func(gv, m2[gv].with_attr("tirx.is_scheduled", True))
    return m2


def main():
    import tvm, numpy as np
    from tvm import relax
    knob = {0: "L2_bind", 1: "L1_shared", 2: "L3_align", 3: "L4_pipe"}[LEVEL]
    print(f"[{LABEL}] manual GEMM M={M} N={N} K={K} level={LEVEL} knob={knob}", flush=True)
    m2 = schedule_mod(LEVEL)
    dev = tvm.cuda(0); tgt = tvm.target.Target.from_device(dev)
    us = -1.0
    try:
        ex = tvm.compile(m2, target=tgt); vm = relax.VirtualMachine(ex, dev)
        a = tvm.runtime.tensor(np.random.randn(M, K).astype("float32"), device=dev)
        b = tvm.runtime.tensor(np.random.randn(K, N).astype("float32"), device=dev)
        vm["main"](a, b); dev.sync()
        vf = vm.time_evaluator("main", dev, number=REPS, repeat=3)
        r = vf(a, b); us = r.mean * 1e6
        mn = min(r.results) * 1e6
        print(f"[{LABEL}] RESULT level={LEVEL} knob={knob} mean_us={us:.2f} min_us={mn:.2f}", flush=True)
    except Exception as e:
        traceback.print_exc(); print(f"[{LABEL}] BUILD/RUN FAILED {repr(e)[:140]}", flush=True)
    hdr = not os.path.exists(OUT_CSV)
    with open(OUT_CSV, "a") as f:
        if hdr:
            f.write("label,M,N,K,level,knob,mean_us\n")
        f.write(f"{LABEL},{M},{N},{K},{LEVEL},{knob},{us:.3f}\n")
    print(f"[{LABEL}] DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        traceback.print_exc(); print(f"[{LABEL}] EXCEPTION {repr(e)}"); sys.exit(3)
