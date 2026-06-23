"""L3 STEP 3 - INT8 Diagnostic Gate for CoDriving standard conv.

Verifies whether TVM int8 codegen uses real dp4a/WMMA for:
  Cin=32 (p50 stage0, K=288÷32=9 ✓ aligned)
  Cin=48 (p25 stage0, K=432÷32=13.5 ✗ misaligned)
  Cin=64 (base stage0, K=576÷32=18 ✓ aligned)

Uses TVM relay (not relax) for reliable CUDA PTX dump.
CoDriving is groups=1 standard conv → dp4a should work for aligned K.

Key asymmetry vs Pyramid:
  Pyramid: grouped conv (groups=32), in_per_g=3/4 → dp4a impossible for in_per_g=3
  CoDriving: groups=1, standard → dp4a works for K÷4-aligned K

Output: /exdata/jichengzhi/s2_tvm/results/codriving_s0probe_int8.csv
        + CUDA dump files for inspection

Usage: python l3_int8_gate.py <out_csv> [gpu=5] [trials=300]
"""
import sys, os, time, traceback, json
import numpy as np

OUT_CSV = sys.argv[1] if len(sys.argv) > 1 else "/exdata/jichengzhi/s2_tvm/results/codriving_s0probe_int8.csv"
GPU_ID  = int(sys.argv[2]) if len(sys.argv) > 2 else 5
TRIALS  = int(sys.argv[3]) if len(sys.argv) > 3 else 300
BATCH   = 2
REPS    = 300
SEED    = 42

os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
print(f"[L3-INT8-GATE] gpu={GPU_ID} trials={TRIALS} seed={SEED}", flush=True)

# Stage0 configs: spatial dims after stride-2 from 256×512
CONFIGS = [
    # (cin, label, k_div32_ok, expected_tc)
    (16, "p75_s0", False, "no"),    # K=144÷32=4.5 ✗
    (32, "p50_s0", True,  "yes"),   # K=288÷32=9 ✓
    (48, "p25_s0", False, "no"),    # K=432÷32=13.5 ✗ (KEY: does this rank-flip vs base?)
    (64, "base_s0", True,  "yes"),  # K=576÷32=18 ✓
]
COUT  = None  # same as cin (square conv)
H, W  = 128, 256  # stage0 spatial after stride-2 from 256×512
KSIZE = 3
PAD   = 1

import tvm
import tvm.relay as relay
from tvm.relay import testing
from tvm.relay.backend import Executor, Runtime
from tvm import auto_scheduler
from tvm.s_tir import meta_schedule as ms

# Also need standard TVM AutoTVM for simpler CUDA dump
import tvm.auto_scheduler as ansor
from tvm.contrib import graph_executor

dev = tvm.cuda(0)
target = tvm.target.Target("cuda")

def build_single_conv_relay(cin, cout, h, w, ksize=3, pad=1, batch=2, dtype="int8"):
    """Build relay module for a single int8 conv2d."""
    data = relay.var("data", shape=(batch, cin, h, w), dtype=dtype)
    weight = relay.var("weight", shape=(cout, cin, ksize, ksize), dtype=dtype)
    out = relay.nn.conv2d(
        data, weight,
        channels=cout,
        kernel_size=(ksize, ksize),
        padding=(pad, pad),
        data_layout="NCHW",
        kernel_layout="OIHW",
        out_dtype="int32",
    )
    func = relay.Function([data, weight], out)
    mod = tvm.IRModule.from_expr(func)
    return mod


def get_cuda_ptx(lib):
    """Extract CUDA PTX from compiled relay module."""
    try:
        src = lib.get_source("ptx")
        return src
    except Exception:
        pass
    try:
        src = lib.get_source("cu")
        return src
    except Exception:
        pass
    try:
        for k in dir(lib):
            if 'source' in k.lower():
                src = getattr(lib, k)
                if callable(src):
                    try:
                        return src()
                    except Exception:
                        pass
    except Exception:
        pass
    # Try imported modules
    try:
        for im in lib.imported_modules:
            try:
                src = im.get_source()
                if src:
                    return src
            except Exception:
                pass
    except Exception:
        pass
    return ""


DP4A_KEYS = ["dp4a", "__dp4a", "s8", "int8", "wmma", "mma.sync", "tvm_int8"]

results = []

for cin, label, k_div32, expected_tc in CONFIGS:
    cout = cin
    k = cin * KSIZE * KSIZE
    k_mod32 = k % 32
    k_div4  = k % 4 == 0  # dp4a requires K÷4
    print(f"\n[L3-INT8-GATE] === {label}: Cin={cin} K={k} K÷32={'OK' if k_div32 else 'FAIL'} K÷4={'OK' if k_div4 else 'FAIL'} ===", flush=True)

    mod = build_single_conv_relay(cin, cout, H, W, ksize=KSIZE, pad=PAD, batch=BATCH, dtype="int8")
    params = {}

    # Build with default CUDA target (TVM selects best schedule)
    workdir = f"/exdata/jichengzhi/s2_tvm/ms_work_l3_int8gate_{label}"
    os.makedirs(workdir, exist_ok=True)

    lat_default_us = -1.0
    lat_tuned_us   = -1.0
    has_dp4a = False
    dp4a_key_found = ""
    cuda_src = ""
    tune_s = 0.0

    # Build default (no tuning)
    try:
        with tvm.transform.PassContext(opt_level=3):
            lib_default = relay.build(mod, target=target, params=params)
        m_default = graph_executor.GraphModule(lib_default["default"](dev))
        rng = np.random.RandomState(42)
        x_data  = rng.randint(-64, 64, (BATCH, cin, H, W)).astype("int8")
        wt_data = rng.randint(-64, 64, (cout, cin, KSIZE, KSIZE)).astype("int8")
        m_default.set_input("data", tvm.nd.array(x_data, dev))
        m_default.set_input("weight", tvm.nd.array(wt_data, dev))
        m_default.run()
        timer = m_default.module.time_evaluator("run", dev, number=REPS, repeat=5)
        r = timer()
        lat_default_us = r.mean * 1e6
        print(f"[L3-INT8-GATE] {label}: DEFAULT lat={lat_default_us:.2f}µs", flush=True)

        # Get CUDA source from default build for dp4a check
        cuda_src = get_cuda_ptx(lib_default.lib)
    except Exception:
        traceback.print_exc()
        print(f"[L3-INT8-GATE] {label}: DEFAULT FAILED", flush=True)

    # Tune with AutoScheduler (Ansor) - gives access to CUDA source
    try:
        tasks, task_weights = auto_scheduler.extract_tasks(mod, params, target)
        print(f"[L3-INT8-GATE] {label}: {len(tasks)} tasks for Ansor tuning", flush=True)

        log_file = f"{workdir}/ansor_{label}.json"
        tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
        tune_opt = auto_scheduler.TuningOptions(
            num_measure_trials=TRIALS,
            runner=auto_scheduler.LocalRunner(repeat=3, timeout=15),
            measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        )
        t0 = time.time()
        tuner.tune(tune_opt)
        tune_s = time.time() - t0
        print(f"[L3-INT8-GATE] {label}: Ansor tune done in {tune_s:.0f}s", flush=True)

        with auto_scheduler.ApplyHistoryBest(log_file):
            with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
                lib_tuned = relay.build(mod, target=target, params=params)

        m_tuned = graph_executor.GraphModule(lib_tuned["default"](dev))
        rng = np.random.RandomState(42)
        x_data  = rng.randint(-64, 64, (BATCH, cin, H, W)).astype("int8")
        wt_data = rng.randint(-64, 64, (cout, cin, KSIZE, KSIZE)).astype("int8")
        m_tuned.set_input("data", tvm.nd.array(x_data, dev))
        m_tuned.set_input("weight", tvm.nd.array(wt_data, dev))
        m_tuned.run()
        timer = m_tuned.module.time_evaluator("run", dev, number=REPS, repeat=5)
        r = timer()
        lat_tuned_us = r.mean * 1e6
        print(f"[L3-INT8-GATE] {label}: TUNED lat={lat_tuned_us:.2f}µs", flush=True)

        # Get CUDA source from tuned build
        cuda_src_tuned = get_cuda_ptx(lib_tuned.lib)
        if cuda_src_tuned:
            cuda_src = cuda_src_tuned

    except Exception:
        traceback.print_exc()
        print(f"[L3-INT8-GATE] {label}: TUNE FAILED", flush=True)

    # Check CUDA source for dp4a/int8 signatures
    if cuda_src:
        cuda_lower = cuda_src.lower()
        for k_str in DP4A_KEYS:
            if k_str.lower() in cuda_lower:
                has_dp4a = True
                dp4a_key_found = k_str
                break
        # Save for inspection
        dump_path = f"/exdata/jichengzhi/s2_tvm/results/cod_int8gate_{label}.cu"
        with open(dump_path, "w") as f:
            f.write(cuda_src[:50000])
        print(f"[L3-INT8-GATE] {label}: CUDA dumped ({len(cuda_src)} chars), has_dp4a={has_dp4a} key={dp4a_key_found!r}", flush=True)
    else:
        print(f"[L3-INT8-GATE] {label}: no CUDA source obtained", flush=True)

    results.append({
        "label": label, "cin": cin, "k": k, "k_div32": int(k_div32), "k_mod32": k_mod32,
        "lat_default_us": lat_default_us, "lat_tuned_us": lat_tuned_us,
        "has_dp4a": int(has_dp4a), "dp4a_key": dp4a_key_found,
        "trials": TRIALS, "tune_s": tune_s
    })

    print(f"[L3-INT8-GATE] {label}: SUMMARY: default={lat_default_us:.1f} tuned={lat_tuned_us:.1f} dp4a={has_dp4a}", flush=True)

# === ANALYSIS ===
print("\n[L3-INT8-GATE] === ANALYSIS ===", flush=True)
header = "label,cin,k,k_div32,k_mod32,lat_default_us,lat_tuned_us,has_dp4a,dp4a_key,trials,tune_s"
rows = []
for r in results:
    print(f"  {r['label']}: Cin={r['cin']} K={r['k']} K÷32={'OK' if r['k_div32'] else 'FAIL'} "
          f"default={r['lat_default_us']:.1f} tuned={r['lat_tuned_us']:.1f} dp4a={r['has_dp4a']}", flush=True)
    row = (f"{r['label']},{r['cin']},{r['k']},{r['k_div32']},{r['k_mod32']},"
           f"{r['lat_default_us']:.2f},{r['lat_tuned_us']:.2f},{r['has_dp4a']},{r['dp4a_key']},{r['trials']},{r['tune_s']:.0f}")
    rows.append(row)

# Key rank-flip analysis
p25  = next((r for r in results if r["label"] == "p25_s0"),  None)
base = next((r for r in results if r["label"] == "base_s0"), None)
if p25 and base and p25["lat_tuned_us"] > 0 and base["lat_tuned_us"] > 0:
    ratio = p25["lat_tuned_us"] / base["lat_tuned_us"]
    flip  = ratio > 1.0
    flops_ratio = (48/64)**2  # ≈0.5625
    verdict = (
        f"RANK-FLIP CONFIRMED: p25_s0({p25['lat_tuned_us']:.1f}µs) > base_s0({base['lat_tuned_us']:.1f}µs) "
        f"ratio={ratio:.2f}× despite p25 having {flops_ratio:.2f}× FLOPS → INT8 K÷32 alignment couples prune×quant"
        if flip else
        f"NO RANK-FLIP: p25_s0({p25['lat_tuned_us']:.1f}µs) ≤ base_s0({base['lat_tuned_us']:.1f}µs) ratio={ratio:.2f}"
    )
    print(f"\n[L3-INT8-GATE] VERDICT: {verdict}", flush=True)

with open(OUT_CSV, "w") as f:
    f.write(header + "\n")
    f.write("\n".join(rows) + "\n")
print(f"\n[L3-INT8-GATE] Results written to {OUT_CSV}", flush=True)
print("[L3-INT8-GATE] DONE", flush=True)
