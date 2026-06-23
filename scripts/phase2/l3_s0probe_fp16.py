"""L3 STEP 2: s0-only variation probe for CoDriving fp16.

Tests rank-flip by varying ONLY s0 while fixing s1=128, s2=256 (base).
- s0=32 (label=s0_32): misaligned ÷32 BUT ÷16 aligned → should work fine fp16
- s0=48 (label=s0_48): recover from existing workdir
- s0=64 = base (already measured: tuned=8057µs)

If fp16 has NO rank-flip → CoDriving fp16 is separable.
Under INT8: s0=48 K=432÷32=13.5 ✗ vs s0=64 K=576÷32=18 ✓ → expect alignment effect.

Usage:
  python l3_s0probe_fp16.py s0_48 <gpu>       # recover from workdir
  python l3_s0probe_fp16.py s0_32 <gpu>       # fresh tune seed=42 (hours!)
  python l3_s0probe_fp16.py s0_32 <gpu> tune  # force fresh tune
"""
import sys, os, time, traceback
import numpy as np

LABEL  = sys.argv[1]  # s0_32 | s0_48
GPU_ID = int(sys.argv[2])
MODE   = sys.argv[3] if len(sys.argv) > 3 else "recover"
REPS   = 500
BATCH  = 2
TRIALS = 750  # for fresh tune

os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

BASE = "/exdata/jichengzhi/s2_tvm"
ONNX_PATH = f"{BASE}/models/codriving_cache/cod_{LABEL}_backbone.onnx"
WORKDIR_OLD = f"{BASE}/ms_work_cod_cod_{LABEL}"
WORKDIR_NEW = f"{BASE}/ms_work_cod_l3_{LABEL}_fresh"  # fresh workdir
OUT_CSV = f"{BASE}/results/codriving_s0probe_fp16.csv"

print(f"[L3-S0PROBE] label={LABEL} gpu={GPU_ID} mode={MODE}", flush=True)
print(f"[L3-S0PROBE] onnx={ONNX_PATH}", flush=True)

if not os.path.exists(ONNX_PATH):
    print(f"[L3-S0PROBE] ERROR: ONNX not found: {ONNX_PATH}", flush=True)
    sys.exit(1)

import onnx as onnx_mod
import tvm
from tvm import relax
from tvm.relax.frontend.onnx import from_onnx
import tvm.s_tir.dlight as dl
import tvm.s_tir.tensor_intrin.cuda  # noqa

dev = tvm.cuda(0)
tgt = tvm.target.Target.from_device(dev)

m = onnx_mod.load(ONNX_PATH)
init_names = {i.name for i in m.graph.initializer}
SH = {}
for inp in m.graph.input:
    if inp.name in init_names:
        continue
    SH[inp.name] = tuple(
        (d.dim_value if d.dim_value > 0 else BATCH)
        for d in inp.type.tensor_type.shape.dim)
print(f"[L3-S0PROBE] inputs={SH}", flush=True)

rng = np.random.RandomState(0)
feeds = {k: rng.rand(*v).astype("float32") for k, v in SH.items()}
mod0 = from_onnx(m, shape_dict=SH, keep_params_in_input=False)


def time_vm(vm, args_tvm, dev, reps):
    vm["main"](*args_tvm); dev.sync()
    vf = vm.time_evaluator("main", dev, number=reps, repeat=5)
    r = vf(*args_tvm)
    return r.mean * 1e6, min(r.results) * 1e6


def prepare_modt(mod0, tgt):
    seq = tvm.transform.Sequential([
        relax.transform.LegalizeOps(), relax.transform.AnnotateTIROpPattern(),
        relax.transform.FuseOps(), relax.transform.FuseTIR()])
    with tgt, tvm.transform.PassContext(opt_level=3):
        return seq(mod0)


# ---- DEFAULT ----
def_us = -1.0
print("[L3-S0PROBE] Measuring DEFAULT...", flush=True)
try:
    with tvm.transform.PassContext(opt_level=3):
        ex = relax.build(mod0, target="cuda")
    vm = relax.VirtualMachine(ex, dev)
    args_tvm = [tvm.runtime.tensor(feeds[k], device=dev) for k in SH]
    def_us, _ = time_vm(vm, args_tvm, dev, REPS)
    print(f"[L3-S0PROBE] DEFAULT mean_us={def_us:.1f}", flush=True)
except Exception as e:
    print(f"[L3-S0PROBE] DEFAULT relax.build failed ({e}); dlight fallback", flush=True)
    try:
        modd = prepare_modt(mod0, tgt)
        with tgt, tvm.transform.PassContext(opt_level=3):
            modd = dl.ApplyDefaultSchedule(
                dl.gpu.Matmul(), dl.gpu.GEMV(), dl.gpu.Reduction(),
                dl.gpu.GeneralReduction(), dl.gpu.Fallback())(modd)
            exd = tvm.compile(modd, target=tgt)
        vmd = relax.VirtualMachine(exd, dev)
        args_tvm = [tvm.runtime.tensor(feeds[k], device=dev) for k in SH]
        def_us, _ = time_vm(vmd, args_tvm, dev, REPS)
        print(f"[L3-S0PROBE] DEFAULT(dlight) mean_us={def_us:.1f}", flush=True)
    except Exception:
        traceback.print_exc()

# ---- TUNED ----
tuned_us = -1.0
workdir_used = ""
tune_s = 0

# Decide: recover from old workdir OR fresh tune
use_workdir = None
if MODE == "recover":
    record_file = os.path.join(WORKDIR_OLD, "database_tuning_record.json")
    if os.path.exists(record_file):
        with open(record_file) as f:
            n = sum(1 for _ in f)
        print(f"[L3-S0PROBE] Found old workdir with {n} records; recovering", flush=True)
        use_workdir = WORKDIR_OLD
    else:
        print(f"[L3-S0PROBE] No old workdir; doing fresh tune (may take hours)", flush=True)
        MODE = "tune"

if MODE == "tune" or use_workdir is None:
    # Fresh tune with seed=42 to avoid illegal-access crash
    print(f"[L3-S0PROBE] FRESH TUNE: workdir={WORKDIR_NEW} trials={TRIALS} seed=42", flush=True)
    os.makedirs(WORKDIR_NEW, exist_ok=True)
    modt = prepare_modt(mod0, tgt)
    from tvm.s_tir.meta_schedule import relax_integration as ri
    t0 = time.time()
    ri.tune_relax(mod=modt, params={}, target=tgt, work_dir=WORKDIR_NEW,
                  max_trials_global=TRIALS, seed=42)
    tune_s = time.time() - t0
    print(f"[L3-S0PROBE] Tune done in {tune_s:.0f}s", flush=True)
    use_workdir = WORKDIR_NEW

# Apply schedule
print(f"[L3-S0PROBE] Applying schedule from {use_workdir}...", flush=True)
t0 = time.time()
try:
    modt = prepare_modt(mod0, tgt)
    sched = relax.transform.MetaScheduleApplyDatabase(work_dir=use_workdir)(modt)
    sched = dl.ApplyDefaultSchedule(
        dl.gpu.Matmul(), dl.gpu.GEMV(), dl.gpu.Reduction(),
        dl.gpu.GeneralReduction(), dl.gpu.Fallback())(sched)
    ex_tuned = tvm.compile(sched, target=tgt)
    apply_s = time.time() - t0
    vm_tuned = relax.VirtualMachine(ex_tuned, dev)
    args_tvm = [tvm.runtime.tensor(feeds[k], device=dev) for k in SH]
    tuned_us, tuned_mn = time_vm(vm_tuned, args_tvm, dev, REPS)
    workdir_used = os.path.basename(use_workdir)
    print(f"[L3-S0PROBE] TUNED mean_us={tuned_us:.1f} min_us={tuned_mn:.1f} (apply_s={apply_s:.1f})", flush=True)
except Exception:
    traceback.print_exc()
    print("[L3-S0PROBE] TUNED FAILED", flush=True)

ratio = (def_us / tuned_us) if (def_us > 0 and tuned_us > 0) else -1.0
print(f"[L3-S0PROBE] RESULT: {LABEL} default={def_us:.1f} tuned={tuned_us:.1f} ratio={ratio:.3f} workdir={workdir_used}", flush=True)

header = "label,onnx,trials,batch,default_us,tuned_us,e2e_ratio,tune_s,workdir,source"
row = f"{LABEL},cod_{LABEL}_backbone.onnx,{TRIALS},{BATCH},{def_us:.1f},{tuned_us:.1f},{ratio:.3f},{tune_s:.0f},{workdir_used},H800_TVM_fp16_real"

needs_header = not os.path.exists(OUT_CSV)
with open(OUT_CSV, "a") as f:
    if needs_header:
        f.write(header + "\n")
    f.write(row + "\n")
print(f"[L3-S0PROBE] Written to {OUT_CSV}", flush=True)
print("[L3-S0PROBE] DONE", flush=True)
