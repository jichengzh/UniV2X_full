"""S2.2h — VFE (encoder_m1 compute) absolute latency in TVM, to test whether
the VFE *computation* is a RSU-stage bottleneck or whether the measured
encoder=17.49ms (Orin eager) is dominated by scatter + eager launch overhead.

Real VFE structure (pillar_vfe.py, DAIR m1): single Linear(10->64) + BN + ReLU
+ max over the point dim. Synthesize the dominant compute:
  voxel_features [M, 32, 10] @ W[10,64] -> [M,32,64] -> max over dim=1 -> [M,64]
M = number of active pillars (~1000-2000 for DAIR). dlight vs MS-tuned, abs us.
If dlight VFE compute is already microseconds, the 17.49ms encoder is NOT the
dense compute -> it is scatter(data-dependent) + eager overhead -> TVM cannot
help it; RSU-stage TVM value is backbone-only. (compute density argument is
HW-relative but the order-of-magnitude conclusion transfers.)

Usage: python s2_2h_vfe.py <M> <trials> <out_csv> [reps]
"""
from __future__ import annotations
import sys, os, time, traceback
import numpy as np

M = int(sys.argv[1]); TRIALS = int(sys.argv[2]); OUT = sys.argv[3]
REPS = int(sys.argv[4]) if len(sys.argv) > 4 else 1000
P = 32   # max points per pillar
DIN = 10; DOUT = 64
WORK = f"/exdata/jichengzhi/s2_tvm/ms_work_2h_M{M}"


def build_mod():
    import tvm
    from tvm import relax
    bb = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorStructInfo((M, P, DIN), "float32"))
    w = relax.Var("w", relax.TensorStructInfo((DIN, DOUT), "float32"))
    with bb.function("main", [x, w]):
        with bb.dataflow():
            h = bb.emit(relax.op.matmul(x, w, out_dtype="float32"))   # [M,32,64]
            h = bb.emit(relax.op.nn.relu(h))
            y = bb.emit(relax.op.max(h, axis=1))                       # [M,64]
            gv = bb.emit_output(y)
        bb.emit_func_output(gv)
    return bb.finalize()


def time_vm(vm, args, dev, reps):
    vm["main"](*args); dev.sync()
    vf = vm.time_evaluator("main", dev, number=reps, repeat=5)
    r = vf(*args)
    return r.mean * 1e6, min(r.results) * 1e6


def main():
    import tvm
    from tvm import relax
    from tvm.s_tir.meta_schedule import relax_integration as ri
    dev = tvm.cuda(0); tgt = tvm.target.Target.from_device(dev)
    feeds = [np.random.randn(M, P, DIN).astype("float32"),
             np.random.randn(DIN, DOUT).astype("float32")]
    mod0 = build_mod()
    print(f"[VFE M={M}] Linear({DIN}->{DOUT}) on [{M},{P},{DIN}] + max -> [{M},{DOUT}]", flush=True)

    def_us = -1.0
    try:
        with tvm.transform.PassContext(opt_level=3):
            ex = relax.build(mod0, target="cuda")
        vm = relax.VirtualMachine(ex, dev)
        args = [tvm.runtime.tensor(f, device=dev) for f in feeds]
        def_us, def_mn = time_vm(vm, args, dev, REPS)
        print(f"[VFE M={M}] DEFAULT mean_us={def_us:.2f} min_us={def_mn:.2f}", flush=True)
    except Exception:
        traceback.print_exc(); print("DEFAULT FAILED", flush=True)

    tun_us = -1.0
    try:
        seq = tvm.transform.Sequential([
            relax.transform.LegalizeOps(), relax.transform.AnnotateTIROpPattern(),
            relax.transform.FuseOps(), relax.transform.FuseTIR()])
        with tgt, tvm.transform.PassContext(opt_level=3):
            modt = seq(mod0)
        os.makedirs(WORK, exist_ok=True)
        ri.tune_relax(mod=modt, params={}, target=tgt, work_dir=WORK,
                      max_trials_global=TRIALS, seed=0)
        with tgt, tvm.transform.PassContext(opt_level=3):
            sched = relax.transform.MetaScheduleApplyDatabase(work_dir=WORK)(modt)
            ex2 = tvm.compile(sched, target=tgt)
        vm2 = relax.VirtualMachine(ex2, dev)
        args = [tvm.runtime.tensor(f, device=dev) for f in feeds]
        tun_us, tun_mn = time_vm(vm2, args, dev, REPS)
        print(f"[VFE M={M}] TUNED mean_us={tun_us:.2f} min_us={tun_mn:.2f}", flush=True)
    except Exception:
        traceback.print_exc(); print("TUNED FAILED", flush=True)

    ratio = def_us / tun_us if (def_us > 0 and tun_us > 0) else -1.0
    hdr = not os.path.exists(OUT)
    with open(OUT, "a") as f:
        if hdr:
            f.write("M,trials,default_us,tuned_us,ratio\n")
        f.write(f"{M},{TRIALS},{def_us:.2f},{tun_us:.2f},{ratio:.3f}\n")
    print(f"[VFE M={M}] default/tuned={ratio:.3f}x DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        traceback.print_exc(); print(f"EXCEPTION {repr(e)}"); sys.exit(3)
