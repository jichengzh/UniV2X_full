"""L3 STEP 1+2: Recover fp16-tuned latencies from existing workdirs.
Fixed: MetaScheduleApplyDatabase must be INSIDE with tgt, PassContext block.

Usage: python l3_recover_v2.py <onnx_path> <label> <workdir> <out_csv> <gpu> [batch=2]
"""
import sys, os, time, traceback
import numpy as np

ONNX    = sys.argv[1]
LABEL   = sys.argv[2]
WORKDIR = sys.argv[3]
OUT_CSV = sys.argv[4]
GPU_ID  = int(sys.argv[5])
BATCH   = int(sys.argv[6]) if len(sys.argv) > 6 else 2
REPS    = 500

os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
print(f"[RECOVER-V2] label={LABEL} gpu={GPU_ID} batch={BATCH}", flush=True)
print(f"[RECOVER-V2] onnx={ONNX}", flush=True)
print(f"[RECOVER-V2] workdir={WORKDIR}", flush=True)

record_file = os.path.join(WORKDIR, "database_tuning_record.json")
if os.path.exists(record_file):
    with open(record_file) as f:
        n_records = sum(1 for _ in f)
    print(f"[RECOVER-V2] {n_records} tuning records in workdir", flush=True)
else:
    print(f"[RECOVER-V2] ERROR: no records at {record_file}", flush=True)
    sys.exit(1)

import onnx as onnx_mod
import tvm
from tvm import relax
from tvm.relax.frontend.onnx import from_onnx
import tvm.s_tir.dlight as dl
import tvm.s_tir.tensor_intrin.cuda  # noqa

dev = tvm.cuda(0)
tgt = tvm.target.Target.from_device(dev)

m = onnx_mod.load(ONNX)
init_names = {n.name for n in m.graph.initializer}
SH = {}
for inp in m.graph.input:
    if inp.name in init_names:
        continue
    SH[inp.name] = tuple(
        (d.dim_value if d.dim_value > 0 else BATCH)
        for d in inp.type.tensor_type.shape.dim)
print(f"[RECOVER-V2] inputs={SH}", flush=True)

rng = np.random.RandomState(0)
feeds = {k: rng.rand(*v).astype("float32") for k, v in SH.items()}
mod0 = from_onnx(m, shape_dict=SH, keep_params_in_input=False)


def time_vm(vm, args_tvm, dev, reps):
    vm["main"](*args_tvm); dev.sync()
    vf = vm.time_evaluator("main", dev, number=reps, repeat=5)
    r = vf(*args_tvm)
    return r.mean * 1e6, min(r.results) * 1e6


# ---- DEFAULT ----
def_us = -1.0
print("[RECOVER-V2] Measuring DEFAULT...", flush=True)
try:
    with tvm.transform.PassContext(opt_level=3):
        ex = relax.build(mod0, target="cuda")
    vm = relax.VirtualMachine(ex, dev)
    args = [tvm.runtime.tensor(feeds[k], device=dev) for k in SH]
    def_us, _ = time_vm(vm, args, dev, REPS)
    print(f"[RECOVER-V2] DEFAULT mean_us={def_us:.1f}", flush=True)
except Exception as e:
    print(f"[RECOVER-V2] relax.build failed ({e}); dlight fallback", flush=True)
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
        vmd = relax.VirtualMachine(exd, dev)
        args = [tvm.runtime.tensor(feeds[k], device=dev) for k in SH]
        def_us, _ = time_vm(vmd, args, dev, REPS)
        print(f"[RECOVER-V2] DEFAULT(dlight) mean_us={def_us:.1f}", flush=True)
    except Exception:
        traceback.print_exc()

# ---- TUNED (apply from existing workdir) ----
tuned_us = -1.0
print(f"[RECOVER-V2] Applying from workdir (no re-tune)...", flush=True)
t0 = time.time()
try:
    seq = tvm.transform.Sequential([
        relax.transform.LegalizeOps(), relax.transform.AnnotateTIROpPattern(),
        relax.transform.FuseOps(), relax.transform.FuseTIR()])
    with tgt, tvm.transform.PassContext(opt_level=3):
        modt = seq(mod0)
        # ★ MetaScheduleApplyDatabase MUST be INSIDE with tgt, PassContext
        sched = relax.transform.MetaScheduleApplyDatabase(work_dir=WORKDIR)(modt)
        sched = dl.ApplyDefaultSchedule(
            dl.gpu.Matmul(), dl.gpu.GEMV(), dl.gpu.Reduction(),
            dl.gpu.GeneralReduction(), dl.gpu.Fallback())(sched)
        ex_tuned = tvm.compile(sched, target=tgt)
    apply_s = time.time() - t0
    vm_tuned = relax.VirtualMachine(ex_tuned, dev)
    args = [tvm.runtime.tensor(feeds[k], device=dev) for k in SH]
    tuned_us, tuned_mn = time_vm(vm_tuned, args, dev, REPS)
    print(f"[RECOVER-V2] TUNED mean_us={tuned_us:.1f} min_us={tuned_mn:.1f} (apply_s={apply_s:.1f})", flush=True)
except Exception:
    traceback.print_exc()
    print("[RECOVER-V2] TUNED FAILED", flush=True)

ratio = (def_us / tuned_us) if (def_us > 0 and tuned_us > 0) else -1.0
print(f"[RECOVER-V2] FINAL: {LABEL}: default={def_us:.1f} tuned={tuned_us:.1f} ratio={ratio:.3f}", flush=True)

header = "label,onnx,trials,batch,default_us,tuned_us,e2e_ratio,tune_s,source"
row = f"{LABEL},{os.path.basename(ONNX)},RECOVERED,{BATCH},{def_us:.1f},{tuned_us:.1f},{ratio:.3f},0,H800_TVM_fp16_real"

needs_header = not os.path.exists(OUT_CSV)
with open(OUT_CSV, "a") as f:
    if needs_header:
        f.write(header + "\n")
    f.write(row + "\n")
print(f"[RECOVER-V2] Written to {OUT_CSV}", flush=True)
print("[RECOVER-V2] DONE", flush=True)
