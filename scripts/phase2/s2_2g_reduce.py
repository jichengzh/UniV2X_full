"""S2.2g / E-couple L5 cross-thread-reduction knob vs software (output width).

L5 knob: reduction strategy in {serial-k, cross-thread-k (bind reduction axis
to threadIdx => allreduce)}. Software axis: output width N=Cout (pruning shrinks
Cout). Reduction-bound layers (small output, large K=Cin*k^2) benefit from
cross-thread; output-rich layers have enough spatial parallelism for serial.
argmin(reduction|Cout) flip => the reduction knob couples with pruning.

matmul (M x K) @ (K x N): M=64 (spatial), K=2048 fixed (reduction-heavy), N=Cout.

Usage: python s2_2g_reduce.py <serial|cross> <Cout> <out_csv> [reps]
"""
from __future__ import annotations
import sys, os, traceback

MODE = sys.argv[1]; COUT = int(sys.argv[2]); OUT = sys.argv[3]
REPS = int(sys.argv[4]) if len(sys.argv) > 4 else 1000
M = 64; K = 2048; N = COUT
TR = 64  # threads on reduction axis for cross-thread mode


def record(us, status):
    hdr = not os.path.exists(OUT)
    with open(OUT, "a") as f:
        if hdr:
            f.write("mode,Cout,M,K,N,TR,status,mean_us\n")
        f.write(f"{MODE},{COUT},{M},{K},{N},{TR},{status},{us:.3f}\n")


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
    fij = sch.fuse(i, j)
    if MODE == "serial":
        # all spatial parallelism on threads, k serial in each thread
        bo, bi = sch.split(fij, factors=[None, 256])
        sch.bind(bo, "blockIdx.x"); sch.bind(bi, "threadIdx.x")
    elif MODE == "cross":
        # spatial on blocks; split reduction, bind chunk to threadIdx => allreduce
        sch.bind(fij, "blockIdx.x")
        ko, ki = sch.split(k, factors=[TR, None])
        sch.bind(ko, "threadIdx.x")
    else:
        raise ValueError(MODE)
    m2 = sch.mod
    gv = [g for g, f in m2.functions.items() if type(f).__name__ == "PrimFunc"][0]
    m2 = m2.clone(); m2.update_func(gv, m2[gv].with_attr("tirx.is_scheduled", True))
    return m2


def main():
    import tvm, numpy as np
    from tvm import relax
    print(f"[L5 {MODE} Cout={COUT}] M={M} K={K} N={N}", flush=True)
    us = -1.0; status = "ok"
    try:
        m2 = build(); dev = tvm.cuda(0); tgt = tvm.target.Target.from_device(dev)
        ex = tvm.compile(m2, target=tgt); vm = relax.VirtualMachine(ex, dev)
        a = tvm.runtime.tensor(np.random.randn(M, K).astype("float32"), device=dev)
        b = tvm.runtime.tensor(np.random.randn(K, N).astype("float32"), device=dev)
        vm["main"](a, b); dev.sync()
        vf = vm.time_evaluator("main", dev, number=REPS, repeat=5); r = vf(a, b)
        us = r.mean * 1e6
        print(f"[L5 {MODE} Cout={COUT}] mean_us={us:.3f} min_us={min(r.results)*1e6:.3f}", flush=True)
    except Exception as e:
        status = "build_fail"; traceback.print_exc(); print(f"FAILED {repr(e)[:120]}", flush=True)
    record(us, status); print("DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        traceback.print_exc(); print(f"EXCEPTION {repr(e)}"); sys.exit(3)
