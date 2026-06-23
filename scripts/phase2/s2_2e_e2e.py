"""S2.2e / E-e2e — does the per-op knob speedup fold to END-TO-END?

Critique #1: the knob wins (D1 2.3x, L1 1.46x, L5 67x) are single-op
microbenchmarks. Amdahl says a 67x on a 1%-of-model op is invisible e2e.
Here we time the WHOLE Pyramid conv backbone subgraph two ways on an idle GPU:
  arm DEFAULT : relax.build default GPU lowering (dlight)  -> minimal knobs
  arm TUNED   : MetaSchedule tune (full knob search)       -> all knobs
e2e ratio = default/tuned = the REAL end-to-end value of the knob stack.
Run per SW config ({base, p50}) so the e2e number also connects to coupling.

Usage: python s2_2e_e2e.py <onnx_path> <label> <trials> <out_csv> [reps] [seed]
"""
from __future__ import annotations
import sys, os, time, traceback
import numpy as np

ONNX = sys.argv[1]; LABEL = sys.argv[2]; TRIALS = int(sys.argv[3]); OUT = sys.argv[4]
REPS = int(sys.argv[5]) if len(sys.argv) > 5 else 500
SEED = int(sys.argv[6]) if len(sys.argv) > 6 else 0
WORK = f"/exdata/jichengzhi/s2_tvm/ms_work_2e_{LABEL}"


def read_inputs(m):
    init = {i.name for i in m.graph.initializer}
    return {i.name: tuple(d.dim_value for d in i.type.tensor_type.shape.dim)
            for i in m.graph.input if i.name not in init}


def time_vm(vm, args, dev, reps):
    vm["main"](*args); dev.sync()
    vf = vm.time_evaluator("main", dev, number=reps, repeat=5)
    r = vf(*args)
    return r.mean * 1e6, min(r.results) * 1e6


def main():
    import onnx, tvm
    from tvm import relax
    from tvm.relax.frontend.onnx import from_onnx
    from tvm.s_tir.meta_schedule import relax_integration as ri
    import tvm.s_tir.tensor_intrin.cuda  # noqa: registers wmma/mma intrins

    dev = tvm.cuda(0); tgt = tvm.target.Target.from_device(dev)
    m = onnx.load(ONNX); SH = read_inputs(m)
    print(f"[{LABEL}] inputs {SH} trials={TRIALS}", flush=True)
    rng = np.random.RandomState(0)
    feeds = {k: rng.rand(*v).astype("float32") for k, v in SH.items()}
    mod0 = from_onnx(m, shape_dict=SH, keep_params_in_input=False)

    # ---- arm DEFAULT (dlight) ----
    def_us = -1.0
    try:
        with tvm.transform.PassContext(opt_level=3):
            ex = relax.build(mod0, target="cuda")
        vm = relax.VirtualMachine(ex, dev)
        args = [tvm.runtime.tensor(feeds[k], device=dev) for k in SH]
        def_us, def_mn = time_vm(vm, args, dev, REPS)
        print(f"[{LABEL}] DEFAULT mean_us={def_us:.1f} min_us={def_mn:.1f}", flush=True)
    except Exception:
        traceback.print_exc(); print(f"[{LABEL}] DEFAULT FAILED", flush=True)

    # ---- arm TUNED (MetaSchedule) ----
    tun_us = -1.0; tune_s = -1
    try:
        seq = tvm.transform.Sequential([
            relax.transform.LegalizeOps(), relax.transform.AnnotateTIROpPattern(),
            relax.transform.FuseOps(), relax.transform.FuseTIR()])
        with tgt, tvm.transform.PassContext(opt_level=3):
            modt = seq(mod0)
        os.makedirs(WORK, exist_ok=True)
        t0 = time.time()
        ri.tune_relax(mod=modt, params={}, target=tgt, work_dir=WORK,
                      max_trials_global=TRIALS, seed=SEED)
        tune_s = time.time() - t0
        with tgt, tvm.transform.PassContext(opt_level=3):
            sched = relax.transform.MetaScheduleApplyDatabase(work_dir=WORK)(modt)
            ex2 = tvm.compile(sched, target=tgt)
        vm2 = relax.VirtualMachine(ex2, dev)
        args = [tvm.runtime.tensor(feeds[k], device=dev) for k in SH]
        tun_us, tun_mn = time_vm(vm2, args, dev, REPS)
        print(f"[{LABEL}] TUNED mean_us={tun_us:.1f} min_us={tun_mn:.1f} tune_s={tune_s:.0f}", flush=True)
    except Exception:
        traceback.print_exc(); print(f"[{LABEL}] TUNED FAILED", flush=True)

    ratio = def_us / tun_us if (def_us > 0 and tun_us > 0) else -1.0
    print(f"[{LABEL}] E2E default/tuned = {ratio:.3f}x", flush=True)
    hdr = not os.path.exists(OUT)
    with open(OUT, "a") as f:
        if hdr:
            f.write("label,onnx,trials,default_us,tuned_us,e2e_ratio,tune_s\n")
        f.write(f"{LABEL},{os.path.basename(ONNX)},{TRIALS},{def_us:.2f},{tun_us:.2f},{ratio:.3f},{tune_s:.0f}\n")
    print(f"[{LABEL}] DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        traceback.print_exc(); print(f"[{LABEL}] EXCEPTION {repr(e)}"); sys.exit(3)
