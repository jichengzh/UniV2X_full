"""L3 Recovery: Apply best TVM schedule from existing workdir + measure.
Tuning completed (1000+ records) but process died before writing CSV.
This script loads the workdir and runs ONLY apply+measure (no re-tuning).

Usage: python cod_l3_recover.py <onnx> <label> <workdir> <out_csv> <gpu> [reps=500] [batch=2]
"""
import sys, os, time, traceback
import numpy as np

ONNX    = sys.argv[1]
LABEL   = sys.argv[2]
WORKDIR = sys.argv[3]
OUT_CSV = sys.argv[4]
GPU_ID  = int(sys.argv[5]) if len(sys.argv) > 5 else 4
REPS    = int(sys.argv[6]) if len(sys.argv) > 6 else 500
BATCH   = int(sys.argv[7]) if len(sys.argv) > 7 else 2

os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
print(f"[RECOVER] label={LABEL} onnx={ONNX} workdir={WORKDIR} gpu={GPU_ID}", flush=True)

# Check workdir
record_file = os.path.join(WORKDIR, "database_tuning_record.json")
if os.path.exists(record_file):
    with open(record_file) as f:
        n_records = sum(1 for _ in f)
    print(f"[RECOVER] Found {n_records} tuning records in {WORKDIR}", flush=True)
else:
    print(f"[RECOVER] WARNING: No record file at {record_file}", flush=True)

import onnx as onnx_mod
import tvm
from tvm import relax
from tvm.relax.frontend.onnx import from_onnx
import tvm.s_tir.dlight as dl
import tvm.s_tir.tensor_intrin.cuda  # noqa: registers wmma/mma intrins

dev = tvm.cuda(0)
tgt = tvm.target.Target.from_device(dev)

# Load ONNX (same as s2_codriving_e2e.py)
m = onnx_mod.load(ONNX)
init_names = {i.name for i in m.graph.initializer}
SH = {}
for inp in m.graph.input:
    if inp.name in init_names:
        continue
    SH[inp.name] = tuple(
        (d.dim_value if d.dim_value > 0 else BATCH)
        for d in inp.type.tensor_type.shape.dim
    )
print(f"[RECOVER] inputs {SH}", flush=True)

rng = np.random.RandomState(0)
feeds = {k: rng.rand(*v).astype("float32") for k, v in SH.items()}
mod0 = from_onnx(m, shape_dict=SH, keep_params_in_input=False)


def time_vm(vm, args_tvm, dev, reps):
    vm["main"](*args_tvm); dev.sync()
    vf = vm.time_evaluator("main", dev, number=reps, repeat=5)
    r = vf(*args_tvm)
    return r.mean * 1e6, min(r.results) * 1e6


# ---- Arm DEFAULT (relax.build → fallback to dlight) ----
def_us = -1.0
print("[RECOVER] Measuring DEFAULT...", flush=True)
try:
    with tvm.transform.PassContext(opt_level=3):
        ex = relax.build(mod0, target="cuda")
    vm = relax.VirtualMachine(ex, dev)
    args_tvm = [tvm.runtime.tensor(feeds[k], device=dev) for k in SH]
    def_us, def_mn = time_vm(vm, args_tvm, dev, REPS)
    print(f"[RECOVER] DEFAULT(relax.build) mean_us={def_us:.1f} min_us={def_mn:.1f}", flush=True)
except Exception as e:
    print(f"[RECOVER] DEFAULT relax.build failed: {e}", flush=True)
    try:
        seq = tvm.transform.Sequential([
            relax.transform.LegalizeOps(), relax.transform.AnnotateTIROpPattern(),
            relax.transform.FuseOps(), relax.transform.FuseTIR()])
        with tgt, tvm.transform.PassContext(opt_level=3):
            modd = seq(mod0)
            modd = dl.ApplyDefaultSchedule(
                dl.gpu.Matmul(), dl.gpu.GEMV(), dl.gpu.Reduction(),
                dl.gpu.GeneralReduction(), dl.gpu.Fallback())(modd)
            exd = tvm.compile(modd, target=tgt)
        vm = relax.VirtualMachine(exd, dev)
        args_tvm = [tvm.runtime.tensor(feeds[k], device=dev) for k in SH]
        def_us, def_mn = time_vm(vm, args_tvm, dev, REPS)
        print(f"[RECOVER] DEFAULT(dlight+Fallback) mean_us={def_us:.1f}", flush=True)
    except Exception as e2:
        print(f"[RECOVER] DEFAULT dlight also failed: {e2}", flush=True)

# ---- Arm TUNED (apply from existing workdir, NO re-tuning) ----
tuned_us = -1.0
tune_s = 0.0
print(f"[RECOVER] Applying schedule from workdir (no re-tuning)...", flush=True)
t0 = time.time()
try:
    # Legalize (same as tuning pipeline)
    seq = tvm.transform.Sequential([
        relax.transform.LegalizeOps(), relax.transform.AnnotateTIROpPattern(),
        relax.transform.FuseOps(), relax.transform.FuseTIR()])
    with tgt, tvm.transform.PassContext(opt_level=3):
        modt = seq(mod0)

    # Apply best schedule from workdir
    sched = relax.transform.MetaScheduleApplyDatabase(work_dir=WORKDIR)(modt)
    sched = dl.ApplyDefaultSchedule(
        dl.gpu.Matmul(), dl.gpu.GEMV(), dl.gpu.Reduction(),
        dl.gpu.GeneralReduction(), dl.gpu.Fallback()
    )(sched)
    ex_tuned = tvm.compile(sched, target=tgt)
    tune_s = time.time() - t0

    vm_tuned = relax.VirtualMachine(ex_tuned, dev)
    args_tvm = [tvm.runtime.tensor(feeds[k], device=dev) for k in SH]
    tuned_us, tuned_mn = time_vm(vm_tuned, args_tvm, dev, REPS)
    print(f"[RECOVER] TUNED mean_us={tuned_us:.1f} min_us={tuned_mn:.1f} (apply_s={tune_s:.1f})", flush=True)
except Exception:
    traceback.print_exc()
    print("[RECOVER] TUNED apply failed", flush=True)

# ---- Write result ----
if def_us > 0 and tuned_us > 0:
    ratio = def_us / tuned_us
else:
    ratio = -1.0

print(f"[RECOVER] FINAL: label={LABEL} default_us={def_us:.1f} tuned_us={tuned_us:.1f} ratio={ratio:.3f}", flush=True)

header = "label,onnx,trials,batch,default_us,tuned_us,e2e_ratio,tune_s"
row = f"{LABEL},{os.path.basename(ONNX)},RECOVERED,{BATCH},{def_us:.1f},{tuned_us:.1f},{ratio:.3f},{tune_s:.0f}"

needs_header = not os.path.exists(OUT_CSV)
with open(OUT_CSV, "a") as f:
    if needs_header:
        f.write(header + "\n")
    f.write(row + "\n")

print(f"[RECOVER] Written to {OUT_CSV}", flush=True)
print("[RECOVER] DONE", flush=True)
