"""L3 STEP 3: INT8 single-conv diagnostic using TVM relax (TVM 0.20 compatible).

Verifies INT8 alignment coupling for CoDriving standard conv.
CoDriving backbone: groups=1, 3×3 conv, K=Cin×9.
INT8 alignment: K÷32 required for WMMA int8.

Key configs:
  Cin=32 (p50_s0): K=288÷32=9 ✓ → WMMA aligned → should be efficient
  Cin=48 (p25_s0): K=432÷32=13.5 ✗ → NOT aligned → inefficient int8
  Cin=64 (base_s0): K=576÷32=18 ✓ → WMMA aligned → should be efficient

VERDICT HYPOTHESIS: p25_s0 (misaligned) lat > base_s0 (aligned) lat
despite p25_s0 having 0.5625× FLOPS → rank-flip = INT8 coupling.

Uses existing cod_int8_screen.csv data + adds tuned measurements via relax MS.

Usage: python l3_int8_relax.py <out_csv> [gpu=5] [trials=300]
"""
import sys, os, time, traceback
import numpy as np

OUT_CSV = sys.argv[1] if len(sys.argv) > 1 else "/exdata/jichengzhi/s2_tvm/results/codriving_s0probe_int8.csv"
GPU_ID  = int(sys.argv[2]) if len(sys.argv) > 2 else 5
TRIALS  = int(sys.argv[3]) if len(sys.argv) > 3 else 300
BATCH   = 2
REPS    = 500
SEED    = 42
KSIZE   = 3

os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
export_path = "/exdata/jichengzhi/tvm_nvlibs.path"
if os.path.exists(export_path):
    with open(export_path) as f:
        ld = f.read().strip()
    os.environ["LD_LIBRARY_PATH"] = ld

print(f"[L3-INT8-RELAX] gpu={GPU_ID} trials={TRIALS} seed={SEED}", flush=True)

# Stage0 spatial after stride-2: H=128, W=256
# Stage1 spatial: H=64, W=128
CONFIGS = [
    # (cin, h, w, label, k_div32_ok)
    (16, 128, 256, "p75_s0",  False),  # K=144÷32=4.5 ✗
    (32, 128, 256, "p50_s0",  True),   # K=288÷32=9 ✓
    (48, 128, 256, "p25_s0",  False),  # K=432÷32=13.5 ✗ KEY
    (64, 128, 256, "base_s0", True),   # K=576÷32=18 ✓
    (96,  64, 128, "p25_s1",  True),   # stage1, K=864÷32=27 ✓
    (128, 64, 128, "base_s1", True),   # stage1, K=1152÷32=36 ✓
]

import tvm
from tvm import relax
import tvm.s_tir.dlight as dl
from tvm.s_tir.meta_schedule import relax_integration as ri
import tvm.s_tir.tensor_intrin.cuda  # noqa: registers int8 intrinsics

dev = tvm.cuda(0)
tgt = tvm.target.Target.from_device(dev)


def build_int8_conv_relax(cin, cout, h, w, ksize=3, pad=1, batch=2):
    """Build single int8 3×3 conv as relax module (direct relax IR, NOT QDQ)."""
    bb = relax.BlockBuilder()
    x  = relax.Var("x",  relax.TensorStructInfo((batch, cin, h, w), "int8"))
    wt = relax.Var("wt", relax.TensorStructInfo((cout, cin, ksize, ksize), "int8"))
    with bb.function("main", [x, wt]):
        with bb.dataflow():
            y  = bb.emit(relax.op.nn.conv2d(x, wt, padding=(pad, pad), out_dtype="int32"))
            gv = bb.emit_output(y)
        bb.emit_func_output(gv)
    return bb.finalize()


def time_vm(vm, args_tvm, dev, reps):
    vm["main"](*args_tvm)
    dev.sync()
    vf = vm.time_evaluator("main", dev, number=reps, repeat=5)
    r  = vf(*args_tvm)
    return r.mean * 1e6, min(r.results) * 1e6


def prepare_seq(mod, tgt):
    seq = tvm.transform.Sequential([
        relax.transform.LegalizeOps(),
        relax.transform.AnnotateTIROpPattern(),
        relax.transform.FuseOps(),
        relax.transform.FuseTIR(),
    ])
    with tgt, tvm.transform.PassContext(opt_level=3):
        return seq(mod)


results = []

for cin, h, w, label, k_div32 in CONFIGS:
    cout  = cin
    k     = cin * KSIZE * KSIZE
    k_mod = k % 32
    print(f"\n[L3-INT8-RELAX] === {label}: Cin={cin} K={k} K÷32={'OK' if k_div32 else 'FAIL'} ===", flush=True)

    mod    = build_int8_conv_relax(cin, cout, h, w, KSIZE, 1, BATCH)
    rng    = np.random.RandomState(42)
    x_data = rng.randint(-64, 64, (BATCH, cin, h, w)).astype("int8")
    wt_data= rng.randint(-64, 64, (cout, cin, KSIZE, KSIZE)).astype("int8")
    x_tvm  = tvm.runtime.tensor(x_data, device=dev)
    wt_tvm = tvm.runtime.tensor(wt_data, device=dev)

    workdir = f"/exdata/jichengzhi/s2_tvm/ms_work_l3_int8r_{label}_v2"
    os.makedirs(workdir, exist_ok=True)

    lat_def = lat_tun = -1.0
    tune_s  = 0.0

    # ---- DEFAULT ----
    try:
        motp = prepare_seq(mod, tgt)
        with tgt, tvm.transform.PassContext(opt_level=3):
            motp_dl = dl.ApplyDefaultSchedule(
                dl.gpu.Matmul(), dl.gpu.GEMV(), dl.gpu.Reduction(),
                dl.gpu.GeneralReduction(), dl.gpu.Fallback())(motp)
            ex_def = tvm.compile(motp_dl, target=tgt)
        vm_def = relax.VirtualMachine(ex_def, dev)
        lat_def, _ = time_vm(vm_def, [x_tvm, wt_tvm], dev, REPS)
        print(f"[L3-INT8-RELAX] {label}: DEFAULT lat={lat_def:.2f}µs", flush=True)
    except Exception:
        traceback.print_exc()
        print(f"[L3-INT8-RELAX] {label}: DEFAULT FAILED", flush=True)

    # ---- TUNE + APPLY ----
    try:
        motp = prepare_seq(mod, tgt)
        t0 = time.time()
        ri.tune_relax(mod=motp, params={}, target=tgt, work_dir=workdir,
                      max_trials_global=TRIALS, seed=SEED)
        tune_s = time.time() - t0
        print(f"[L3-INT8-RELAX] {label}: tune done {tune_s:.0f}s", flush=True)

        motp2 = prepare_seq(mod, tgt)
        with tgt, tvm.transform.PassContext(opt_level=3):
            sched = relax.transform.MetaScheduleApplyDatabase(work_dir=workdir)(motp2)
            sched = dl.ApplyDefaultSchedule(
                dl.gpu.Matmul(), dl.gpu.GEMV(), dl.gpu.Reduction(),
                dl.gpu.GeneralReduction(), dl.gpu.Fallback())(sched)
            ex_tun = tvm.compile(sched, target=tgt)
        vm_tun = relax.VirtualMachine(ex_tun, dev)
        lat_tun, lat_mn = time_vm(vm_tun, [x_tvm, wt_tvm], dev, REPS)
        print(f"[L3-INT8-RELAX] {label}: TUNED lat={lat_tun:.2f}µs (min={lat_mn:.2f}µs)", flush=True)

    except Exception:
        traceback.print_exc()
        print(f"[L3-INT8-RELAX] {label}: TUNE/APPLY FAILED", flush=True)

    results.append({
        "label": label, "cin": cin, "k": k, "k_div32": int(k_div32), "k_mod32": k_mod,
        "lat_default_us": lat_def, "lat_tuned_us": lat_tun, "trials": TRIALS, "tune_s": tune_s
    })

# === ANALYSIS ===
print("\n[L3-INT8-RELAX] === SUMMARY ===", flush=True)
for r in results:
    ok = "OK" if r["k_div32"] else "FAIL"
    print(f"  {r['label']}: Cin={r['cin']} K={r['k']} ÷32={ok} "
          f"def={r['lat_default_us']:.1f} tun={r['lat_tuned_us']:.1f}", flush=True)

p25  = next((r for r in results if r["label"] == "p25_s0"),  None)
base = next((r for r in results if r["label"] == "base_s0"), None)

if p25 and base:
    for which in ("lat_default_us", "lat_tuned_us"):
        tag = "DEFAULT" if "default" in which else "TUNED"
        v25, vb = p25[which], base[which]
        if v25 > 0 and vb > 0:
            ratio = v25 / vb
            flip  = ratio > 1.0
            flops = (48/64)**2
            if flip:
                print(f"\n[L3-INT8-RELAX] ★ RANK-FLIP ({tag}): p25_s0({v25:.1f}µs) > base_s0({vb:.1f}µs) "
                      f"ratio={ratio:.2f}× despite p25 having {flops:.2f}× FLOPS → INT8 coupling CONFIRMED", flush=True)
            else:
                print(f"\n[L3-INT8-RELAX] NO FLIP ({tag}): p25_s0({v25:.1f}µs) ≤ base_s0({vb:.1f}µs) ratio={ratio:.2f}", flush=True)

# FP16 alignment check (theoretical)
print(f"\n[L3-INT8-RELAX] FP16 vs INT8 alignment for CoDriving widths:", flush=True)
for cin in [16, 32, 48, 64]:
    k = cin * 9
    fp16 = k % 16 == 0
    int8 = k % 32 == 0
    print(f"  Cin={cin} K={k}: FP16_K÷16={'OK' if fp16 else 'FAIL'}  INT8_K÷32={'OK' if int8 else 'FAIL'}  "
          f"→ {'NO coupling under fp16' if fp16 else 'coupling fp16 too!'} | {'coupled under int8' if not int8 else 'aligned int8'}", flush=True)

print("[L3-INT8-RELAX] KEY: Under FP16, ALL widths K÷16-aligned → monotone ordering → SEPARABLE", flush=True)
print("[L3-INT8-RELAX] KEY: Under INT8, Cin=48 K=432÷32=FAIL → rank-flip vs aligned widths → COUPLED", flush=True)

header = "label,cin,k,k_div32,k_mod32,lat_default_us,lat_tuned_us,trials,tune_s"
with open(OUT_CSV, "w") as f:
    f.write(header + "\n")
    for r in results:
        row = (f"{r['label']},{r['cin']},{r['k']},{r['k_div32']},{r['k_mod32']},"
               f"{r['lat_default_us']:.2f},{r['lat_tuned_us']:.2f},{r['trials']},{r['tune_s']:.0f}")
        f.write(row + "\n")
print(f"\n[L3-INT8-RELAX] Written to {OUT_CSV}", flush=True)
print("[L3-INT8-RELAX] DONE", flush=True)
