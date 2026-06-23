"""L3: Fresh tune + measure for CoDriving backbone widths.
Fix for CUDA illegal-access: use seed=42 + fresh workdir (avoids bad kernels from seed=0).

This script runs in a subprocess-safe way: each width gets its own process context.

Usage: python l3_fresh_tune.py <onnx> <label> <out_csv> <gpu> [batch=2] [trials=500]
"""
import sys, os, time, traceback
import numpy as np

ONNX   = sys.argv[1]
LABEL  = sys.argv[2]
OUT_CSV = sys.argv[3]
GPU_ID = int(sys.argv[4])
BATCH  = int(sys.argv[5]) if len(sys.argv) > 5 else 2
TRIALS = int(sys.argv[6]) if len(sys.argv) > 6 else 500
REPS   = 500
SEED   = 42  # Fixed seed to avoid bad kernels

os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

BASE     = "/exdata/jichengzhi/s2_tvm"
# Fresh workdir - NEVER reuse old (old workdir = source of illegal-access kernels)
WORKDIR  = f"{BASE}/ms_work_l3_fresh_{LABEL}_s42"

print(f"[L3-FRESH] label={LABEL} gpu={GPU_ID} batch={BATCH} trials={TRIALS} seed={SEED}", flush=True)
print(f"[L3-FRESH] onnx={ONNX}", flush=True)
print(f"[L3-FRESH] workdir={WORKDIR} (FRESH, seed=42)", flush=True)

if not os.path.exists(ONNX):
    print(f"[L3-FRESH] ERROR: ONNX not found: {ONNX}", flush=True)
    sys.exit(1)

import onnx as onnx_mod
import tvm
from tvm import relax
from tvm.relax.frontend.onnx import from_onnx
from tvm.s_tir.meta_schedule import relax_integration as ri
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
print(f"[L3-FRESH] inputs={SH}", flush=True)

rng = np.random.RandomState(0)
feeds = {k: rng.rand(*v).astype("float32") for k, v in SH.items()}
mod0 = from_onnx(m, shape_dict=SH, keep_params_in_input=False)


def time_vm(vm, args_tvm, dev, reps):
    vm["main"](*args_tvm); dev.sync()
    vf = vm.time_evaluator("main", dev, number=reps, repeat=5)
    r = vf(*args_tvm)
    return r.mean * 1e6, min(r.results) * 1e6


def prepare_seq(mod0, tgt):
    seq = tvm.transform.Sequential([
        relax.transform.LegalizeOps(), relax.transform.AnnotateTIROpPattern(),
        relax.transform.FuseOps(), relax.transform.FuseTIR()])
    with tgt, tvm.transform.PassContext(opt_level=3):
        return seq(mod0)


# ---- DEFAULT ----
def_us = -1.0
print("[L3-FRESH] Measuring DEFAULT...", flush=True)
try:
    with tvm.transform.PassContext(opt_level=3):
        ex = relax.build(mod0, target="cuda")
    vm = relax.VirtualMachine(ex, dev)
    args = [tvm.runtime.tensor(feeds[k], device=dev) for k in SH]
    def_us, _ = time_vm(vm, args, dev, REPS)
    print(f"[L3-FRESH] DEFAULT mean_us={def_us:.1f}", flush=True)
except Exception as e:
    print(f"[L3-FRESH] DEFAULT relax.build failed ({e}); dlight fallback", flush=True)
    try:
        modd = prepare_seq(mod0, tgt)
        with tgt, tvm.transform.PassContext(opt_level=3):
            modd = dl.ApplyDefaultSchedule(
                dl.gpu.Matmul(), dl.gpu.GEMV(), dl.gpu.Reduction(),
                dl.gpu.GeneralReduction(), dl.gpu.Fallback())(modd)
            exd = tvm.compile(modd, target=tgt)
        vmd = relax.VirtualMachine(exd, dev)
        args = [tvm.runtime.tensor(feeds[k], device=dev) for k in SH]
        def_us, _ = time_vm(vmd, args, dev, REPS)
        print(f"[L3-FRESH] DEFAULT(dlight) mean_us={def_us:.1f}", flush=True)
    except Exception:
        traceback.print_exc()

# ---- TUNE (fresh workdir, seed=42) ----
print(f"[L3-FRESH] Starting FRESH tune ({TRIALS} trials, seed={SEED})...", flush=True)
os.makedirs(WORKDIR, exist_ok=True)
modt = prepare_seq(mod0, tgt)

t0 = time.time()
ri.tune_relax(mod=modt, params={}, target=tgt, work_dir=WORKDIR,
              max_trials_global=TRIALS, seed=SEED)
tune_s = time.time() - t0
print(f"[L3-FRESH] Tune DONE in {tune_s:.0f}s ({TRIALS} trials)", flush=True)

# ---- APPLY + MEASURE ----
tuned_us = -1.0
print("[L3-FRESH] Applying best schedule...", flush=True)
t1 = time.time()
try:
    with tgt, tvm.transform.PassContext(opt_level=3):
        modt2 = prepare_seq(mod0, tgt)
        sched = relax.transform.MetaScheduleApplyDatabase(work_dir=WORKDIR)(modt2)
        sched = dl.ApplyDefaultSchedule(
            dl.gpu.Matmul(), dl.gpu.GEMV(), dl.gpu.Reduction(),
            dl.gpu.GeneralReduction(), dl.gpu.Fallback())(sched)
        ex_tuned = tvm.compile(sched, target=tgt)
    apply_s = time.time() - t1
    vm_tuned = relax.VirtualMachine(ex_tuned, dev)
    args = [tvm.runtime.tensor(feeds[k], device=dev) for k in SH]
    tuned_us, tuned_mn = time_vm(vm_tuned, args, dev, REPS)
    print(f"[L3-FRESH] TUNED mean_us={tuned_us:.1f} min_us={tuned_mn:.1f} (apply_s={apply_s:.1f})", flush=True)
except Exception:
    traceback.print_exc()
    print("[L3-FRESH] TUNED apply FAILED", flush=True)

ratio = (def_us / tuned_us) if (def_us > 0 and tuned_us > 0) else -1.0
print(f"[L3-FRESH] RESULT: {LABEL}: default={def_us:.1f} tuned={tuned_us:.1f} ratio={ratio:.3f}", flush=True)

header = "label,onnx,trials,batch,default_us,tuned_us,e2e_ratio,tune_s,source"
row = f"{LABEL},{os.path.basename(ONNX)},{TRIALS},{BATCH},{def_us:.1f},{tuned_us:.1f},{ratio:.3f},{tune_s:.0f},H800_TVM_fp16_real_seed42"

needs_header = not os.path.exists(OUT_CSV)
with open(OUT_CSV, "a") as f:
    if needs_header:
        f.write(header + "\n")
    f.write(row + "\n")
print(f"[L3-FRESH] Written to {OUT_CSV}", flush=True)
print("[L3-FRESH] DONE", flush=True)
