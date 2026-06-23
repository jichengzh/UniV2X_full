"""L3 STEP 3 - INT8 gate using TVM relay.qnn (proven dp4a path for dense conv).

CoDriving backbone has STANDARD conv (groups=1, dense).
For 3×3 conv: reduction dim K = Cin × 9.
INT8 WMMA requires K ÷ 32:
  Cin=32 (p50): K=288÷32=9 ✓ → dp4a/WMMA works → fast
  Cin=48 (p25): K=432÷32=13.5 ✗ → dp4a/WMMA fails → slow
  Cin=64 (base): K=576÷32=18 ✓ → dp4a/WMMA works → fast

KEY TEST: Does p25_s0 (Cin=48, misaligned) > base_s0 (Cin=64, aligned) in tuned latency?
If YES → INT8 alignment creates rank-flip (COUPLING that fp16 doesn't have).

Also runs full backbone int8 using per-stage conv simulation.

Usage: python l3_int8_qnn.py <out_csv> [gpu=5] [trials=300]
"""
import sys, os, time, traceback
import numpy as np

OUT_CSV = sys.argv[1] if len(sys.argv) > 1 else "/exdata/jichengzhi/s2_tvm/results/codriving_s0probe_int8.csv"
GPU_ID  = int(sys.argv[2]) if len(sys.argv) > 2 else 5
TRIALS  = int(sys.argv[3]) if len(sys.argv) > 3 else 300
BATCH   = 2
REPS    = 300
SEED    = 42

os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
print(f"[L3-INT8-QNN] gpu={GPU_ID} trials={TRIALS} seed={SEED}", flush=True)

# All relevant configs (stage0 spatial dims after stride-2 from [batch=2, cin=64, H=256, W=512])
STAGE0_H, STAGE0_W = 128, 256
STAGE1_H, STAGE1_W =  64, 128
CONFIGS = [
    # (cin, h, w, label, k_div32_ok, note)
    (16, STAGE0_H, STAGE0_W, "p75_s0",  False, "p75 stage0, K=144÷32=4.5✗"),
    (32, STAGE0_H, STAGE0_W, "p50_s0",  True,  "p50 stage0, K=288÷32=9✓"),
    (48, STAGE0_H, STAGE0_W, "p25_s0",  False, "p25 stage0, K=432÷32=13.5✗ KEY"),
    (64, STAGE0_H, STAGE0_W, "base_s0", True,  "base stage0, K=576÷32=18✓"),
    (96, STAGE1_H, STAGE1_W, "p25_s1",  True,  "p25 stage1, K=864÷32=27✓ (aligned)"),
    (128, STAGE1_H, STAGE1_W, "base_s1", True,  "base stage1, K=1152÷32=36✓ (aligned)"),
]

import tvm
import tvm.relay as relay
import tvm.relay.qnn as qnn
from tvm import auto_scheduler
from tvm.contrib import graph_executor

dev = tvm.cuda(0)
target = tvm.target.Target("cuda -arch=sm_90")  # H800 = sm_90 Hopper

DP4A_PATTERNS = ["dp4a", "__dp4a", "vdotq", "mma.sync", ".s8", "s8s8s32"]


def build_int8_3x3_qnn(cin, cout, h, w, batch=2):
    """Build single 3×3 int8 conv as QNN relay module (uses dp4a on aligned K)."""
    data = relay.var("data", shape=(batch, cin, h, w), dtype="int8")
    wt   = relay.var("weight", shape=(cout, cin, 3, 3), dtype="int8")
    out = qnn.op.conv2d(
        data, wt,
        input_zero_point=relay.const(0, "int32"),
        kernel_zero_point=relay.const(0, "int32"),
        input_scale=relay.const(1.0, "float32"),
        kernel_scale=relay.const(1.0, "float32"),
        channels=cout, kernel_size=(3, 3), padding=(1, 1),
        data_layout="NCHW", kernel_layout="OIHW", out_dtype="int32",
    )
    func = relay.Function([data, wt], out)
    mod  = tvm.IRModule.from_expr(func)
    return relay.transform.InferType()(mod), {}


def get_cuda_src(lib):
    """Try multiple paths to get CUDA source from compiled lib."""
    for getter in [
        lambda: lib.lib.get_source("ptx"),
        lambda: lib.lib.get_source("cu"),
        lambda: lib.lib.get_source(),
        lambda: "\n".join(m.get_source() for m in lib.lib.imported_modules if m.type_key == "cuda"),
        lambda: lib.lib.imported_modules[0].get_source(),
    ]:
        try:
            src = getter()
            if src and len(src) > 10:
                return src
        except Exception:
            pass
    return ""


def run_and_time(m, inputs, dev, reps=300):
    for name, arr in inputs.items():
        m.set_input(name, tvm.nd.array(arr, dev))
    m.run()  # warmup
    timer = m.module.time_evaluator("run", dev, number=reps, repeat=5)
    r = timer()
    return r.mean * 1e6, min(r.results) * 1e6


results = []

for cin, h, w, label, k_div32, note in CONFIGS:
    cout = cin
    k = cin * 9
    k_mod32 = k % 32
    print(f"\n[L3-INT8-QNN] === {label} ({note}) ===", flush=True)
    print(f"[L3-INT8-QNN]   Cin={cin} K={k} K÷32={'OK' if k_div32 else 'FAIL'} K÷4={'OK' if k%4==0 else 'FAIL'}", flush=True)

    mod, params = build_int8_3x3_qnn(cin, cout, h, w, BATCH)
    rng = np.random.RandomState(42)
    x_data  = rng.randint(-64, 64, (BATCH, cin, h, w)).astype("int8")
    wt_data = rng.randint(-64, 64, (cout, cin, 3, 3)).astype("int8")
    inputs = {"data": x_data, "weight": wt_data}

    workdir = f"/exdata/jichengzhi/s2_tvm/ms_work_l3_qnn_{label}"
    os.makedirs(workdir, exist_ok=True)
    log_file = f"{workdir}/ansor.json"

    lat_def_us = lat_tun_us = -1.0
    has_dp4a = False
    dp4a_key = ""
    tune_s = 0.0

    # ---- Default build ----
    try:
        with tvm.transform.PassContext(opt_level=3):
            lib_def = relay.build(mod, target=target, params=params)
        m_def = graph_executor.GraphModule(lib_def["default"](dev))
        lat_def_us, _ = run_and_time(m_def, inputs, dev, REPS)
        print(f"[L3-INT8-QNN] {label}: DEFAULT lat={lat_def_us:.2f}µs", flush=True)
    except Exception:
        traceback.print_exc()
        print(f"[L3-INT8-QNN] {label}: DEFAULT FAILED", flush=True)

    # ---- Tuned build (Ansor) ----
    try:
        tasks, weights = auto_scheduler.extract_tasks(mod, params, target)
        print(f"[L3-INT8-QNN] {label}: {len(tasks)} Ansor tasks", flush=True)
        tuner = auto_scheduler.TaskScheduler(tasks, weights)
        tune_opt = auto_scheduler.TuningOptions(
            num_measure_trials=TRIALS,
            runner=auto_scheduler.LocalRunner(repeat=3, timeout=20),
            measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        )
        t0 = time.time()
        tuner.tune(tune_opt)
        tune_s = time.time() - t0
        print(f"[L3-INT8-QNN] {label}: Ansor done in {tune_s:.0f}s", flush=True)

        with auto_scheduler.ApplyHistoryBest(log_file):
            with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
                lib_tun = relay.build(mod, target=target, params=params)

        m_tun = graph_executor.GraphModule(lib_tun["default"](dev))
        lat_tun_us, lat_tun_mn = run_and_time(m_tun, inputs, dev, REPS)
        print(f"[L3-INT8-QNN] {label}: TUNED lat={lat_tun_us:.2f}µs (min={lat_tun_mn:.2f}µs)", flush=True)

        # Get CUDA source for dp4a check
        cuda_src = get_cuda_src(lib_tun)
        if not cuda_src:
            cuda_src = get_cuda_src(lib_def)
        if cuda_src:
            cuda_lower = cuda_src.lower()
            for kw in DP4A_PATTERNS:
                if kw.lower() in cuda_lower:
                    has_dp4a = True
                    dp4a_key = kw
                    break
            dump = f"/exdata/jichengzhi/s2_tvm/results/cod_qnn_diag_{label}.ptx"
            with open(dump, "w") as f:
                f.write(cuda_src[:100000])
            print(f"[L3-INT8-QNN] {label}: CUDA src {len(cuda_src)} chars, has_dp4a={has_dp4a} key={dp4a_key!r}", flush=True)

    except Exception:
        traceback.print_exc()
        print(f"[L3-INT8-QNN] {label}: TUNE FAILED", flush=True)

    results.append({
        "label": label, "cin": cin, "k": k, "k_div32": int(k_div32), "k_mod32": k_mod32,
        "lat_default_us": lat_def_us, "lat_tuned_us": lat_tun_us,
        "has_dp4a": int(has_dp4a), "dp4a_key": dp4a_key, "trials": TRIALS, "tune_s": tune_s,
        "note": note
    })

# === SUMMARY + VERDICT ===
print("\n[L3-INT8-QNN] === FINAL SUMMARY ===", flush=True)
for r in results:
    k_ok = "OK" if r["k_div32"] else "FAIL"
    print(f"  {r['label']}: Cin={r['cin']} K={r['k']} ÷32={k_ok} "
          f"def={r['lat_default_us']:.1f}µs tun={r['lat_tuned_us']:.1f}µs "
          f"dp4a={r['has_dp4a']} key={r['dp4a_key']!r}", flush=True)

p25_r  = next((r for r in results if r["label"] == "p25_s0"), None)
base_r = next((r for r in results if r["label"] == "base_s0"), None)
p50_r  = next((r for r in results if r["label"] == "p50_s0"), None)

if p25_r and base_r and p25_r["lat_tuned_us"] > 0 and base_r["lat_tuned_us"] > 0:
    ratio = p25_r["lat_tuned_us"] / base_r["lat_tuned_us"]
    flops = (48/64)**2
    flip  = ratio > 1.0
    print(f"\n[L3-INT8-QNN] RANK-FLIP TEST (p25_s0 vs base_s0):", flush=True)
    print(f"  p25_s0 tuned = {p25_r['lat_tuned_us']:.1f}µs (Cin=48, K=432÷32✗, has_dp4a={p25_r['has_dp4a']})", flush=True)
    print(f"  base_s0 tuned = {base_r['lat_tuned_us']:.1f}µs (Cin=64, K=576÷32✓, has_dp4a={base_r['has_dp4a']})", flush=True)
    print(f"  p25/base ratio = {ratio:.3f}× (FLOPS ratio = {flops:.3f}×)", flush=True)
    if flip:
        print(f"  ★ RANK-FLIP CONFIRMED: p25_s0 is {ratio:.2f}× SLOWER than base_s0 despite {flops:.2f}× FLOPS", flush=True)
        print(f"     → INT8 K÷32 alignment creates prune×quant coupling in standard conv", flush=True)
    else:
        print(f"  NO rank-flip: p25_s0 is FASTER than base_s0 (monotone by size)", flush=True)

# FP16 comparison: under FP16, all widths K÷16-aligned → no coupling
print(f"\n[L3-INT8-QNN] FP16 ALIGNMENT CHECK (theory):", flush=True)
for cin in [16, 32, 48, 64]:
    k = cin * 9
    fp16_ok = (k % 16 == 0)
    int8_ok  = (k % 32 == 0)
    print(f"  Cin={cin}: K={k} FP16_K÷16={'OK' if fp16_ok else 'FAIL'} INT8_K÷32={'OK' if int8_ok else 'FAIL'}", flush=True)
print("  → Under FP16: ALL widths aligned → monotone by size → SEPARABLE (no coupling)", flush=True)
print("  → Under INT8: Cin=48 misaligned → creates alignment coupling not present in FP16", flush=True)

# Write CSV
header = "label,cin,k,k_div32,k_mod32,lat_default_us,lat_tuned_us,has_dp4a,dp4a_key,trials,tune_s"
with open(OUT_CSV, "w") as f:
    f.write(header + "\n")
    for r in results:
        row = (f"{r['label']},{r['cin']},{r['k']},{r['k_div32']},{r['k_mod32']},"
               f"{r['lat_default_us']:.2f},{r['lat_tuned_us']:.2f},{r['has_dp4a']},{r['dp4a_key']},{r['trials']},{r['tune_s']:.0f}")
        f.write(row + "\n")
print(f"\n[L3-INT8-QNN] Written to {OUT_CSV}", flush=True)
print("[L3-INT8-QNN] DONE", flush=True)
