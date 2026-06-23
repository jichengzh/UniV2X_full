"""S2.2d / E-couple-A — HW(tile) x SW(pruning-width) coupling test.

Question (route2 GO/NO-GO): does the OPTIMAL tile (TM,TN,BK) for a GEMM
depend on the software decision (pruning width W)? If argmin_tile(latency|W)
SHIFTS across W, the HW knob couples with the SW choice -> co-design has value
(GO). If the same tile wins at every W, the axes are separable -> route2
NO-GO (tune HW once, independent of pruning).

Workload = pruned conv-as-GEMM proxy: matmul (M x K) @ (K x N), with
K = N = W (the pruned channel width), M = fixed spatial-ish batch.
HW knob = thread tile (TM,TN) + reduction tile (BK), hand-scheduled with the
proven level-1 schedule (bind + shared staging + cooperative fetch), marked
tirx.is_scheduled so dlight does not re-schedule. Timed on an IDLE gpu.

A (tile,width) combo is INVALID when a dim is not divisible by its tile
(e.g. W=48 not divisible by TM=32) -> build is skipped & recorded as NA.
Invalidity itself is a coupling signal (the 48-misalignment story).

Usage (single point):
  python s2_2d_couple_tile.py <M> <W> <TM> <TN> <BK> <label> <out_csv> [reps]
"""
from __future__ import annotations
import sys, os, traceback

M = int(sys.argv[1]); W = int(sys.argv[2])
TM = int(sys.argv[3]); TN = int(sys.argv[4]); BK = int(sys.argv[5])
LABEL = sys.argv[6]; OUT_CSV = sys.argv[7]
REPS = int(sys.argv[8]) if len(sys.argv) > 8 else 1000

K = W; N = W   # pruned width drives both reduction (K) and output (N) dims


def record(us, status):
    hdr = not os.path.exists(OUT_CSV)
    with open(OUT_CSV, "a") as f:
        if hdr:
            f.write("label,M,W,K,N,TM,TN,BK,status,mean_us\n")
        f.write(f"{LABEL},{M},{W},{K},{N},{TM},{TN},{BK},{status},{us:.3f}\n")


def valid():
    # split factors=[None,T] require exact divisibility; thread budget <=1024.
    if M % TM or N % TN or K % BK:
        return False, "indivisible"
    if TM * TN > 1024:
        return False, "too_many_threads"
    return True, "ok"


def build_scheduled():
    import tvm
    from tvm import relax, s_tir
    bb = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorStructInfo((M, K), "float32"))
    w = relax.Var("w", relax.TensorStructInfo((K, N), "float32"))
    with bb.function("main", [x, w]):
        with bb.dataflow():
            y = bb.emit(relax.op.matmul(x, w, out_dtype="float32"))
            gv = bb.emit_output(y)
        bb.emit_func_output(gv)
    mod = tvm.transform.Sequential([relax.transform.LegalizeOps(),
                                    relax.transform.FuseTIR()])(bb.finalize())
    sch = s_tir.Schedule(mod)
    mm = sch.get_sblock("matmul", func_name="matmul")
    i, j, k = sch.get_loops(mm)
    io, ii = sch.split(i, factors=[None, TM])
    jo, ji = sch.split(j, factors=[None, TN])
    ko, ki = sch.split(k, factors=[None, BK])
    sch.reorder(io, jo, ii, ji, ko, ki)
    sch.bind(io, "blockIdx.y"); sch.bind(jo, "blockIdx.x")
    sch.bind(ii, "threadIdx.y"); sch.bind(ji, "threadIdx.x")
    a_sh = sch.cache_read(mm, 0, "shared"); sch.compute_at(a_sh, ko)
    b_sh = sch.cache_read(mm, 1, "shared"); sch.compute_at(b_sh, ko)
    for sh in (a_sh, b_sh):
        ls = sch.get_loops(sh)
        fused = sch.fuse(*ls[-2:])
        _, ty, tx = sch.split(fused, factors=[None, TM, TN])
        sch.bind(ty, "threadIdx.y"); sch.bind(tx, "threadIdx.x")
    m2 = sch.mod
    gv = [g for g, f in m2.functions.items() if type(f).__name__ == "PrimFunc"][0]
    m2 = m2.clone(); m2.update_func(gv, m2[gv].with_attr("tirx.is_scheduled", True))
    return m2


def main():
    import tvm, numpy as np
    from tvm import relax
    ok, why = valid()
    print(f"[{LABEL}] M={M} W={W} tile=({TM},{TN},{BK}) valid={ok}({why})", flush=True)
    if not ok:
        record(-1.0, why)
        print(f"[{LABEL}] SKIP {why}", flush=True)
        return
    us = -1.0; status = "ok"
    try:
        m2 = build_scheduled()
        dev = tvm.cuda(0); tgt = tvm.target.Target.from_device(dev)
        ex = tvm.compile(m2, target=tgt); vm = relax.VirtualMachine(ex, dev)
        a = tvm.runtime.tensor(np.random.randn(M, K).astype("float32"), device=dev)
        b = tvm.runtime.tensor(np.random.randn(K, N).astype("float32"), device=dev)
        vm["main"](a, b); dev.sync()
        vf = vm.time_evaluator("main", dev, number=REPS, repeat=5)
        r = vf(a, b); us = r.mean * 1e6; mn = min(r.results) * 1e6
        print(f"[{LABEL}] RESULT tile=({TM},{TN},{BK}) mean_us={us:.2f} min_us={mn:.2f}", flush=True)
    except Exception as e:
        status = "build_fail"; traceback.print_exc()
        print(f"[{LABEL}] BUILD/RUN FAILED {repr(e)[:140]}", flush=True)
    record(us, status)
    print(f"[{LABEL}] DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        traceback.print_exc(); print(f"[{LABEL}] EXCEPTION {repr(e)}"); sys.exit(3)
