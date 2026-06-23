"""CoDriving whole-pipeline TVM — E-e2e for the CoDriving dense core (backbone+heads).

Mirrors Pyramid s2_2e_e2e.py: time the CoDriving conv core subgraph two ways on an
idle GPU and report default/tuned = the real e2e value of the TVM knob stack.
  arm DEFAULT : dlight GPU lowering (minimal knobs); CoDriving 384->1 cls head trips
                relax.build default reduction schedule => fall back to dlight+Fallback.
  arm TUNED   : MetaSchedule tune (full knob search).
CoDriving ONNX exports batch as a symbolic dim (dim_value=0) => substitute BATCH.

Usage: python s2_codriving_e2e.py <onnx> <label> <trials> <out_csv> [batch=2] [reps=500] [seed=0]
"""
from __future__ import annotations
import sys, os, time, traceback
import numpy as np

ONNX = sys.argv[1]; LABEL = sys.argv[2]; TRIALS = int(sys.argv[3]); OUT = sys.argv[4]
BATCH = int(sys.argv[5]) if len(sys.argv) > 5 else 2
REPS = int(sys.argv[6]) if len(sys.argv) > 6 else 500
SEED = int(sys.argv[7]) if len(sys.argv) > 7 else 0
WORK = f"/exdata/jichengzhi/s2_tvm/ms_work_cod_{LABEL}"


def read_inputs(m):
    init = {i.name for i in m.graph.initializer}
    sh = {}
    for i in m.graph.input:
        if i.name in init:
            continue
        sh[i.name] = tuple((d.dim_value if d.dim_value > 0 else BATCH)
                           for d in i.type.tensor_type.shape.dim)
    return sh


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
    print(f"[{LABEL}] inputs {SH} trials={TRIALS} batch={BATCH}", flush=True)
    rng = np.random.RandomState(0)
    feeds = {k: rng.rand(*v).astype("float32") for k, v in SH.items()}
    mod0 = from_onnx(m, shape_dict=SH, keep_params_in_input=False)

    # ---- arm DEFAULT ----
    def_us = -1.0
    try:
        with tvm.transform.PassContext(opt_level=3):
            ex = relax.build(mod0, target="cuda")
        vm = relax.VirtualMachine(ex, dev)
        args = [tvm.runtime.tensor(feeds[k], device=dev) for k in SH]
        def_us, def_mn = time_vm(vm, args, dev, REPS)
        print(f"[{LABEL}] DEFAULT(relax.build) mean_us={def_us:.1f} min_us={def_mn:.1f}", flush=True)
    except Exception:
        print(f"[{LABEL}] DEFAULT relax.build failed (likely 384->1 head); trying dlight+Fallback", flush=True)
        try:
            import tvm.s_tir.dlight as dl
            seq = tvm.transform.Sequential([
                relax.transform.LegalizeOps(), relax.transform.AnnotateTIROpPattern(),
                relax.transform.FuseOps(), relax.transform.FuseTIR()])
            with tgt, tvm.transform.PassContext(opt_level=3):
                modd = seq(mod0)
                modd = dl.ApplyDefaultSchedule(
                    dl.gpu.Matmul(), dl.gpu.GEMV(), dl.gpu.Reduction(),
                    dl.gpu.GeneralReduction(), dl.gpu.Fallback())(modd)
                exd = tvm.compile(modd, target=tgt)
            vmd = relax.VirtualMachine(exd, dev)
            args = [tvm.runtime.tensor(feeds[k], device=dev) for k in SH]
            def_us, def_mn = time_vm(vmd, args, dev, REPS)
            print(f"[{LABEL}] DEFAULT(dlight+Fallback) mean_us={def_us:.1f} min_us={def_mn:.1f}", flush=True)
        except Exception:
            traceback.print_exc(); print(f"[{LABEL}] DEFAULT FAILED both paths", flush=True)

    # ---- arm TUNED ----
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
        import tvm.s_tir.dlight as dl
        with tgt, tvm.transform.PassContext(opt_level=3):
            sched = relax.transform.MetaScheduleApplyDatabase(work_dir=WORK)(modt)
            # dlight Fallback schedules leftover blocks MS didn't tune (BN/transpose/
            # elementwise) — CoDriving backbone has BatchNorm which Pyramid's lacked;
            # un-scheduled blocks else host-access GPU mem => "Memory verification failed".
            sched = dl.ApplyDefaultSchedule(
                dl.gpu.Matmul(), dl.gpu.GEMV(), dl.gpu.Reduction(),
                dl.gpu.GeneralReduction(), dl.gpu.Fallback())(sched)
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
            f.write("label,onnx,trials,batch,default_us,tuned_us,e2e_ratio,tune_s\n")
        f.write(f"{LABEL},{os.path.basename(ONNX)},{TRIALS},{BATCH},{def_us:.2f},{tun_us:.2f},{ratio:.3f},{tune_s:.0f}\n")
    print(f"[{LABEL}] DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        traceback.print_exc(); print(f"[{LABEL}] EXCEPTION {repr(e)}"); sys.exit(3)
