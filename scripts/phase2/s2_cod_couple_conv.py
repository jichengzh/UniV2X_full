"""P4 / CoDriving E-couple — HW(tile) x SW(pruning-width) coupling on REAL CoDriving
conv shapes (not the model-agnostic square GEMM proxy used for Pyramid).

CoDriving backbone = standard 3x3 ResNet BasicBlock convs (group=1). The heaviest
layer (layer2) is Cin=Cout=W @ H*W=32x64, batch=2. As im2col-GEMM:
  M = H_out*W_out*batch (spatial, fixed)   N = Cout = W (pruning width)
  K = Cin*KH*KW = 9*W                      (reduction depth = 9x the output width)
This K=9W vs N=W ASYMMETRY is the real-conv difference from Pyramid's square proxy
(K=N=W); it can shift which tile wins, so re-running the coupling on it tests whether
the Pyramid coupling table transfers to a standard-conv model.

SW axis = pruning width W in {64(p75),128(p50),256(base)} (layer2 channel count).
HW knob = thread tile (TM,TN) + reduction tile (BK), proven level-1 hand schedule
(bind + shared staging + cooperative fetch, tirx.is_scheduled). Timed on IDLE gpu.

Usage: python s2_cod_couple_conv.py <W> <TM> <TN> <BK> <label> <out_csv> [M=2048] [reps=1000]
"""
from __future__ import annotations
import sys, os, traceback

W = int(sys.argv[1])
TM = int(sys.argv[2]); TN = int(sys.argv[3]); BK = int(sys.argv[4])
LABEL = sys.argv[5]; OUT_CSV = sys.argv[6]
M = int(sys.argv[7]) if len(sys.argv) > 7 else 2048
REPS = int(sys.argv[8]) if len(sys.argv) > 8 else 1000

N = W            # output channels = pruning width
K = 9 * W        # 3x3 conv im2col reduction depth = 9 * input width


def record(us, status):
    hdr = not os.path.exists(OUT_CSV)
    with open(OUT_CSV, "a") as f:
        if hdr:
            f.write("label,M,W,K,N,TM,TN,BK,status,mean_us\n")
        f.write(f"{LABEL},{M},{W},{K},{N},{TM},{TN},{BK},{status},{us:.3f}\n")


def valid():
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
    print(f"[{LABEL}] M={M} W={W} K={K} N={N} tile=({TM},{TN},{BK}) valid={ok}({why})", flush=True)
    if not ok:
        record(-1.0, why); print(f"[{LABEL}] SKIP {why}", flush=True); return
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
