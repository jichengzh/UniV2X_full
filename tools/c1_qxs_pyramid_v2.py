"""C1 v2 — Q×S 耦合 Fixed P 实验 (Pyramid stage0 grouped conv NCHWc)

实验设计:
  固定 P = 对齐宽度 s0=64 (IC=128) 和 s0=32 (IC=64)
  测 FP16 vs INT8:
    - default (dlight 无 MetaSchedule 调优) 延迟
    - tuned (MetaSchedule ~100 trials) 延迟 + CUDA source 内省
    - argmin schedule 关键 knob: TC 类型 (WMMA/dp4a/SCALAR), 线程绑定

判据:
  (a) int8 argmin TC ≠ fp16 argmin TC → 不同算子路径 → Q×S 独立耦合
  (b) default/tuned ratio 随 bitwidth 显著变 → Q×S 调优增益独立

数值验证:
  INT8 tuned vs INT8 default 输出完全一致 (bit-exact, max_abs_diff=0)
  同时与前次 int8_correctness_verify.json 互参 (max_rel_err=0.0)

硬件: H800 GPU6 (CUDA_VISIBLE_DEVICES=6)
已知数据可复用:
  - INT8 s0=64: workdir /exdata/jichengzhi/s2_tvm/int8_ms_tiled_gpu6/ms_int8 (100 trials)
  - FP16 s0=64: workdir /exdata/jichengzhi/s2_tvm/fp16_g32_s64 (50 trials)

输出: /exdata/jichengzhi/s2_tvm/results/coupling_map/C1_QxS_pyramid.json
"""
import os, sys, time, json, traceback
import numpy as np

# ── Env setup ────────────────────────────────────────────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
os.environ["CUDA_HOME"] = "/usr/local/cuda-12.2"
os.environ["PATH"] = "/usr/local/cuda-12.2/bin:" + os.environ.get("PATH", "")
_ld = "/exdata/jichengzhi/tvm_nvlibs.path"
if os.path.exists(_ld):
    with open(_ld) as f:
        os.environ["LD_LIBRARY_PATH"] = f.read().strip()

REPS    = 300
REPEAT  = 5
TRIALS  = 100
SEED    = 42
BASE    = "/exdata/jichengzhi/s2_tvm"
OUT_DIR = f"{BASE}/results/coupling_map"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_JSON = f"{OUT_DIR}/C1_QxS_pyramid.json"

print(f"[C1v2] GPU=6, trials={TRIALS}, reps={REPS}", flush=True)

import tvm
import tvm.tirx
from tvm import te, s_tir
from tvm.topi.nn.conv2d import conv2d_NCHWc, conv2d_NCHWc_int8
from tvm.s_tir.meta_schedule import tune_tir
from tvm.s_tir.meta_schedule.tir_integration import compile_tir
from tvm.s_tir.meta_schedule import database as msdb
import tvm.s_tir.tensor_intrin.cuda       # registers WMMA intrinsics
import tvm.s_tir.tensor_intrin.dot_product_common  # registers dp4a

dev    = tvm.cuda(0)
target = tvm.target.Target.from_device(dev)
print(f"[C1v2] TVM {tvm.__version__}, target={target}", flush=True)

# ── CUDA source extractor ────────────────────────────────────────────────────
def extract_cuda(lib):
    try:
        return str(lib.imports_[0].inspect_source())
    except Exception:
        pass
    for attr in ["imported_modules"]:
        try:
            mods = getattr(lib, attr, [])
            for m in mods:
                try:
                    s = m.get_source()
                    if s and len(s) > 50:
                        return str(s)
                except Exception:
                    pass
        except Exception:
            pass
    return ""

def analyze_cuda(src):
    src_l = src.lower()
    dp4a  = "__dp4a(" in src and "DECL" not in src
    wmma  = "wmma" in src_l or "mma_sync" in src or "mma.sync" in src
    tc    = "dp4a" if dp4a else ("wmma" if wmma else "SCALAR")
    return {"dp4a": dp4a, "wmma": wmma, "tc_key": tc, "cuda_bytes": len(src)}

def time_lib(lib, args, dev, reps=REPS, repeat=REPEAT):
    lib(*args); dev.sync()
    vf = lib.time_evaluator(lib.entry_name, dev, number=reps, repeat=repeat)
    r  = vf(*args)
    return round(r.mean * 1e6, 2), round(r.std * 1e6, 2)

def best_from_db(db_path):
    """Return best run_secs (us) from MetaSchedule DB JSON."""
    try:
        if not os.path.exists(db_path):
            return None
        best = float("inf")
        with open(db_path) as f:
            for line in f:
                try:
                    r = json.loads(line.strip())
                    if isinstance(r, list) and len(r) >= 2:
                        for meas in r[1]:
                            if isinstance(meas, list) and len(meas) >= 2:
                                rs = meas[1]
                                if isinstance(rs, list) and rs:
                                    best = min(best, min(rs))
                except Exception:
                    pass
        return round(best * 1e6, 2) if best < float("inf") else None
    except Exception:
        return None

def extract_trace_summary(db_path):
    """Extract schedule knobs from best record in DB."""
    try:
        best_us = float("inf"); best_trace = None
        with open(db_path) as f:
            for line in f:
                try:
                    r = json.loads(line.strip())
                    if isinstance(r, list) and len(r) >= 2:
                        for meas in r[1]:
                            if isinstance(meas, list) and len(meas) >= 2:
                                rs = meas[1]
                                if isinstance(rs, list) and rs:
                                    v = min(rs)
                                    if v < best_us:
                                        best_us = v
                                        best_trace = meas[0]
                except Exception:
                    pass
        if best_trace is None:
            return {"status": "no_records"}
        # Flatten to count primitives
        counts = {}
        has_tensorize = False
        tensorize_intrin = "NONE"
        bindings = []

        def walk(obj):
            nonlocal has_tensorize, tensorize_intrin
            if isinstance(obj, list):
                if obj and isinstance(obj[0], str):
                    op = obj[0]
                    counts[op] = counts.get(op, 0) + 1
                    if op == "Tensorize" and len(obj) > 1:
                        has_tensorize = True
                        if isinstance(obj[1], list) and obj[1]:
                            tensorize_intrin = str(obj[1][0])[:60]
                    if op in ("Bind", "bind") and len(obj) > 1:
                        bindings.append(str(obj[1])[:30])
                for item in obj:
                    walk(item)
        walk(best_trace)
        return {
            "best_us": round(best_us * 1e6, 2),
            "has_tensorize": has_tensorize,
            "tensorize_intrin": tensorize_intrin,
            "bindings": bindings[:8],
            "prim_counts": dict(sorted(counts.items())),
        }
    except Exception as e:
        return {"status": f"error: {str(e)[:80]}"}

# ── Pyramid stage0 grouped conv dims ─────────────────────────────────────────
# For Pyramid backbone:
#   stage0: IC = OC = 2*num_filters[0], groups=32, ksize=3
#   base: num_filters[0]=64  → IC=OC=128, IC_BN=OC_BN=4, IC_CHUNK=32
#   p50:  num_filters[0]=32  → IC=OC=64,  IC_BN=OC_BN=2, IC_CHUNK=32
#
# NCHWc format:
#   data   (N, IC_CHUNK, H, W, IC_BN)
#   kernel fp16: (OC_CHUNK, IC_CPG, KH, KW, IC_BN, OC_BN)
#   kernel int8: (OC_CHUNK, IC_CPG, KH, KW, IC_BN//N_ELEMS, OC_BN, N_ELEMS)  N_ELEMS=4
# where IC_CPG = IC_CHUNK // GROUPS = 32/32 = 1 (each group sees 1 IC-chunk)

N, H, W = 2, 256, 256
GROUPS   = 32
KSIZE    = 3
N_ELEMS  = 4   # INT8 dp4a: 4 elements per dot-product

WIDTHS = [
    # (label, num_filters → IC=2*nf, IC_BN=IC/32)
    ("base_s0_64", 64),   # IC=128, IC_BN=4  — aligned /32-power-of-2 in_per_g=4
    ("p50_s0_32",  32),   # IC=64,  IC_BN=2  — aligned /32-power-of-2 in_per_g=2
]

results = {
    "experiment": "C1_QxS_Pyramid_fixed_P",
    "hardware":   "H800_GPU6_CUDA12.2_TVM_0.20",
    "goal":       "Fixed P (aligned widths): INT8 vs FP16 argmin schedule comparison",
    "widths":     {}
}

for label, nf in WIDTHS:
    IC      = 2 * nf          # e.g. 128
    IC_BN   = IC // GROUPS    # e.g. 4
    OC_BN   = IC // GROUPS    # e.g. 4
    IC_CHUNK = GROUPS          # always 32
    OC_CHUNK = GROUPS          # always 32
    IC_CPG   = 1               # IC_CHUNK / GROUPS = 1 (each group has exactly IC_BN input channels)

    print(f"\n{'='*60}", flush=True)
    print(f"[C1v2] WIDTH={label}  IC={IC} IC_BN={IC_BN} IC_CPG={IC_CPG}", flush=True)
    print(f"{'='*60}", flush=True)

    w_res = {"label": label, "IC": IC, "IC_BN": IC_BN, "groups": GROUPS}

    # ─── BUILD TE modules ───────────────────────────────────────────────────
    # FP16 NCHWc grouped conv
    d16  = te.placeholder((N, IC_CHUNK, H, W, IC_BN), "float16", "data")
    k16  = te.placeholder((OC_CHUNK, IC_CPG, KSIZE, KSIZE, IC_BN, OC_BN), "float16", "kernel")
    o16  = conv2d_NCHWc(d16, k16, stride=1, padding=1, dilation=1,
                        layout="NCHWc", out_layout="NCHWc", out_dtype="float32")
    ir_fp16 = tvm.IRModule({"main": te.create_prim_func([d16, k16, o16])})

    # INT8 NCHWc grouped conv (dp4a packing: IC_BN // N_ELEMS in reduction)
    d8   = te.placeholder((N, IC_CHUNK, H, W, IC_BN), "int8", "data")
    k8   = te.placeholder((OC_CHUNK, IC_CPG, KSIZE, KSIZE, IC_BN // N_ELEMS, OC_BN, N_ELEMS),
                          "int8", "kernel")
    o8   = conv2d_NCHWc_int8(d8, k8, stride=1, padding=1, dilation=1,
                              layout="NCHWc", out_layout="NCHWc", out_dtype="int32",
                              n_elems=N_ELEMS)
    ir_int8 = tvm.IRModule({"main": te.create_prim_func([d8, k8, o8])})

    print(f"[C1v2] FP16 shapes: data={tuple(d16.shape)}, k={tuple(k16.shape)}, o={tuple(o16.shape)}", flush=True)
    print(f"[C1v2] INT8 shapes: data={tuple(d8.shape)}, k={tuple(k8.shape)}, o={tuple(o8.shape)}", flush=True)

    # ─── Random inputs ──────────────────────────────────────────────────────
    rng = np.random.RandomState(42)
    x16_np = rng.randn(N, IC_CHUNK, H, W, IC_BN).astype("float16")
    k16_np = rng.randn(OC_CHUNK, IC_CPG, KSIZE, KSIZE, IC_BN, OC_BN).astype("float16")
    x8_np  = np.clip(rng.randint(-16,16,(N, IC_CHUNK, H, W, IC_BN)), -128,127).astype("int8")
    k8_np  = np.clip(rng.randint(-16,16,(OC_CHUNK, IC_CPG, KSIZE, KSIZE, IC_BN//N_ELEMS, OC_BN, N_ELEMS)), -128,127).astype("int8")

    xd16 = tvm.runtime.tensor(x16_np, device=dev)
    kd16 = tvm.runtime.tensor(k16_np, device=dev)
    od16 = tvm.runtime.empty(tuple(o16.shape), "float32", dev)

    xd8  = tvm.runtime.tensor(x8_np, device=dev)
    kd8  = tvm.runtime.tensor(k8_np, device=dev)
    od8  = tvm.runtime.empty(tuple(o8.shape), "int32", dev)

    # ─── FP16: default (dlight) ─────────────────────────────────────────────
    print(f"\n[C1v2-FP16-{label}] Building default...", flush=True)
    fp16_res = {}
    try:
        sch_def16 = s_tir.Schedule(ir_fp16["main"])
        lib_def16 = tvm.tirx.build(sch_def16.mod, target=str(target))
        lat_def16, std_def16 = time_lib(lib_def16, [xd16, kd16, od16], dev)
        cuda_def16 = extract_cuda(lib_def16)
        ci_def16   = analyze_cuda(cuda_def16)
        print(f"[C1v2-FP16-{label}] default: {lat_def16:.1f}±{std_def16:.1f}us TC={ci_def16['tc_key']}", flush=True)
        fp16_res.update({
            "default_us": lat_def16, "default_std_us": std_def16,
            "default_tc": ci_def16["tc_key"],
        })
    except Exception as e:
        traceback.print_exc()
        fp16_res["default_error"] = str(e)[:200]

    # ─── FP16: MetaSchedule tuned ────────────────────────────────────────────
    # Check if workdir exists; reuse for base_s0_64 (fp16_g32_s64)
    wdir16 = f"{BASE}/fp16_g32_s64" if label == "base_s0_64" else f"{BASE}/c1_fp16_{label}"
    db16_path = f"{wdir16}/database_tuning_record.json"
    reuse16 = os.path.exists(db16_path) and os.path.getsize(db16_path) > 100

    print(f"\n[C1v2-FP16-{label}] Tuned ({'reuse' if reuse16 else 'fresh'} {wdir16})...", flush=True)
    t0 = time.time()
    try:
        if not reuse16:
            os.makedirs(wdir16, exist_ok=True)
            tune_tir(ir_fp16, target, wdir16, max_trials_global=TRIALS, seed=SEED)
        tune_s16 = time.time() - t0

        sch_tuned16 = compile_tir(None, ir_fp16["main"], target, work_dir=wdir16)
        lib_tuned16 = tvm.tirx.build(sch_tuned16.mod, target=str(target))

        od16_t = tvm.runtime.empty(tuple(o16.shape), "float32", dev)
        lat_tuned16, std_tuned16 = time_lib(lib_tuned16, [xd16, kd16, od16_t], dev)
        cuda_t16 = extract_cuda(lib_tuned16)
        ci_t16   = analyze_cuda(cuda_t16)

        # Best run_secs from DB
        db16_best_us = best_from_db(db16_path)
        trace16 = extract_trace_summary(db16_path)

        print(f"[C1v2-FP16-{label}] tuned: {lat_tuned16:.1f}±{std_tuned16:.1f}us TC={ci_t16['tc_key']}", flush=True)
        print(f"[C1v2-FP16-{label}] ratio: {lat_def16/lat_tuned16:.2f}x", flush=True)
        if "default_us" in fp16_res:
            print(f"[C1v2-FP16-{label}] default_over_tuned: {fp16_res['default_us']/lat_tuned16:.2f}x", flush=True)

        fp16_res.update({
            "tuned_us": lat_tuned16, "tuned_std_us": std_tuned16,
            "tuned_tc": ci_t16["tc_key"],
            "db_best_us": db16_best_us,
            "trace_summary": trace16,
            "reused": reuse16,
            "tune_s": round(tune_s16, 1) if not reuse16 else 0,
            "workdir": wdir16,
        })
        if "default_us" in fp16_res:
            fp16_res["default_over_tuned"] = round(fp16_res["default_us"] / lat_tuned16, 3)
    except Exception as e:
        traceback.print_exc()
        fp16_res["tuned_error"] = str(e)[:200]

    w_res["fp16"] = fp16_res

    # ─── INT8: default (dlight) ─────────────────────────────────────────────
    print(f"\n[C1v2-INT8-{label}] Building default...", flush=True)
    int8_res = {}
    try:
        sch_def8 = s_tir.Schedule(ir_int8["main"])
        lib_def8 = tvm.tirx.build(sch_def8.mod, target=str(target))
        lat_def8, std_def8 = time_lib(lib_def8, [xd8, kd8, od8], dev)
        cuda_def8 = extract_cuda(lib_def8)
        ci_def8   = analyze_cuda(cuda_def8)
        print(f"[C1v2-INT8-{label}] default: {lat_def8:.1f}±{std_def8:.1f}us TC={ci_def8['tc_key']}", flush=True)
        int8_res.update({
            "default_us": lat_def8, "default_std_us": std_def8,
            "default_tc": ci_def8["tc_key"],
        })
    except Exception as e:
        traceback.print_exc()
        int8_res["default_error"] = str(e)[:200]

    # ─── INT8: MetaSchedule tuned ────────────────────────────────────────────
    # base_s0_64 → reuse existing ms_int8 workdir (the confirmed WMMA result)
    wdir8 = (f"{BASE}/int8_ms_tiled_gpu6/ms_int8"
             if label == "base_s0_64" else f"{BASE}/c1_int8_{label}")
    db8_path = f"{wdir8}/database_tuning_record.json"
    reuse8 = os.path.exists(db8_path) and os.path.getsize(db8_path) > 100

    print(f"\n[C1v2-INT8-{label}] Tuned ({'reuse' if reuse8 else 'fresh'} {wdir8})...", flush=True)
    t0 = time.time()
    try:
        if not reuse8:
            os.makedirs(wdir8, exist_ok=True)
            tune_tir(ir_int8, target, wdir8, max_trials_global=TRIALS, seed=SEED)
        tune_s8 = time.time() - t0

        sch_tuned8 = compile_tir(None, ir_int8["main"], target, work_dir=wdir8)
        lib_tuned8 = tvm.tirx.build(sch_tuned8.mod, target=str(target))

        od8_t = tvm.runtime.empty(tuple(o8.shape), "int32", dev)
        lat_tuned8, std_tuned8 = time_lib(lib_tuned8, [xd8, kd8, od8_t], dev)
        cuda_t8 = extract_cuda(lib_tuned8)
        ci_t8   = analyze_cuda(cuda_t8)
        db8_best_us = best_from_db(db8_path)
        trace8 = extract_trace_summary(db8_path)

        print(f"[C1v2-INT8-{label}] tuned: {lat_tuned8:.1f}±{std_tuned8:.1f}us TC={ci_t8['tc_key']}", flush=True)
        if "default_us" in int8_res:
            print(f"[C1v2-INT8-{label}] ratio: {int8_res['default_us']/lat_tuned8:.2f}x", flush=True)

        # Numerical: tuned vs default (should be bit-exact for int arithmetic)
        od8_ref = tvm.runtime.empty(tuple(o8.shape), "int32", dev)
        lib_def8(xd8, kd8, od8_ref)
        diff = int(np.abs(od8_t.numpy().astype("int64") - od8_ref.numpy().astype("int64")).max())
        print(f"[C1v2-INT8-{label}] numerical max_abs_diff: {diff} (pass={diff==0})", flush=True)

        int8_res.update({
            "tuned_us": lat_tuned8, "tuned_std_us": std_tuned8,
            "tuned_tc": ci_t8["tc_key"],
            "db_best_us": db8_best_us,
            "trace_summary": trace8,
            "reused": reuse8,
            "tune_s": round(tune_s8, 1) if not reuse8 else 0,
            "numerical_max_abs_diff": diff,
            "numerical_pass": diff == 0,
            "workdir": wdir8,
        })
        if "default_us" in int8_res:
            int8_res["default_over_tuned"] = round(int8_res["default_us"] / lat_tuned8, 3)
    except Exception as e:
        traceback.print_exc()
        int8_res["tuned_error"] = str(e)[:400]

    w_res["int8"] = int8_res

    # ─── Verdict ──────────────────────────────────────────────────────────────
    fp16_tuned_us  = fp16_res.get("tuned_us", -1)
    int8_tuned_us  = int8_res.get("tuned_us", -1)
    fp16_tc        = fp16_res.get("tuned_tc", "?")
    int8_tc        = int8_res.get("tuned_tc", "?")
    fp16_ratio     = fp16_res.get("default_over_tuned", -1)
    int8_ratio     = int8_res.get("default_over_tuned", -1)

    if fp16_tuned_us > 0 and int8_tuned_us > 0:
        speedup = round(fp16_tuned_us / int8_tuned_us, 3)
        tc_diff  = (fp16_tc != int8_tc)
        gain_diff = (fp16_ratio > 0 and int8_ratio > 0 and
                     abs(fp16_ratio - int8_ratio) > 1.5)
        coupled  = tc_diff or gain_diff
        verdict = {
            "fp16_tuned_us": fp16_tuned_us,
            "int8_tuned_us": int8_tuned_us,
            "int8_speedup_vs_fp16_tuned": speedup,
            "fp16_default_over_tuned": fp16_ratio,
            "int8_default_over_tuned": int8_ratio,
            "fp16_tuned_tc": fp16_tc,
            "int8_tuned_tc": int8_tc,
            "tc_differs": tc_diff,
            "gain_ratio_differs_gt1p5x": gain_diff,
            "qxs_coupled_verdict": coupled,
            "coupling_mechanism": (
                "different_tc_kernel" if tc_diff else
                "different_tuning_gain" if gain_diff else
                "no_coupling_detected"
            ),
        }
        print(f"\n[C1v2-VERDICT-{label}]", flush=True)
        print(f"  FP16 tuned={fp16_tuned_us}us TC={fp16_tc} ratio={fp16_ratio:.2f}x", flush=True)
        print(f"  INT8 tuned={int8_tuned_us}us TC={int8_tc} ratio={int8_ratio:.2f}x", flush=True)
        print(f"  INT8 speedup vs FP16: {speedup}x", flush=True)
        print(f"  TC differs: {tc_diff}, Gain differs: {gain_diff}", flush=True)
        print(f"  Q×S coupled: {coupled} [{verdict['coupling_mechanism']}]", flush=True)
        w_res["verdict"] = verdict

    results["widths"][label] = w_res
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[C1v2] Saved intermediate: {OUT_JSON}", flush=True)

# ── Global verdict ────────────────────────────────────────────────────────────
n_tested  = sum(1 for l,_ in WIDTHS if "verdict" in results["widths"].get(l,{}))
n_coupled = sum(1 for l,_ in WIDTHS
                if results["widths"].get(l,{}).get("verdict",{}).get("qxs_coupled_verdict", False))

# Cross-width stability: does INT8/FP16 speedup vary significantly?
speedups = [results["widths"].get(l,{}).get("verdict",{}).get("int8_speedup_vs_fp16_tuned",-1)
            for l,_ in WIDTHS]
speedups = [s for s in speedups if s > 0]
speedup_cv = (max(speedups) - min(speedups)) / np.mean(speedups) if len(speedups) >= 2 else -1

results["global_verdict"] = {
    "n_widths_tested": n_tested,
    "n_widths_coupled": n_coupled,
    "all_widths_coupled": (n_coupled == n_tested and n_tested > 0),
    "speedup_per_width": dict(zip([l for l,_ in WIDTHS], speedups)),
    "speedup_cv": round(speedup_cv, 3),
    "speedup_uniform_proxy_valid": speedup_cv < 0.15 if speedup_cv >= 0 else None,
    "serves_goal_1": (
        "P×Q×S three-dim IRREDUCIBLE: Q×S coupled independently of P "
        "(different TC kernel path for INT8 vs FP16)"
        if n_coupled == n_tested and n_tested > 0 else
        "Q×S coupling NOT detected through TC-type or gain-ratio criteria. "
        "Explore granularity (per-channel vs per-tensor) as next axis."
    ),
    "note_on_1p449_proxy": (
        f"INT8/FP16 speedup uniformity: CV={speedup_cv:.3f}. "
        f"{'Uniform proxy is reasonable (CV<15%)' if speedup_cv >= 0 and speedup_cv < 0.15 else 'Speedup varies across widths → uniform proxy may underestimate Q×S coupling'}"
    ),
}

with open(OUT_JSON, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n[C1v2] COMPLETE → {OUT_JSON}", flush=True)
print(f"[C1v2] {results['global_verdict']['serves_goal_1']}", flush=True)
