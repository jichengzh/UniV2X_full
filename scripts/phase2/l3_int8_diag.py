"""L3 STEP 3: INT8 diagnostic gate + multi-width screen for CoDriving.

Research question: Does standard-conv CoDriving (groups=1) become coupled under INT8?
Mechanism: INT8 WMMA requires K÷32 where K=Cin×Ksize² for im2col.
For 3×3 conv: K=Cin×9. So K÷32 ⟺ Cin×9÷32.

Results expected:
  Cin=16 (p75): K=144, 144÷32=4.5 ✗ → TVM may not use WMMA int8
  Cin=32 (p50): K=288, 288÷32=9 ✓ → real WMMA int8
  Cin=48 (p25): K=432, 432÷32=13.5 ✗ → TVM may not use WMMA int8
  Cin=64 (base): K=576, 576÷32=18 ✓ → real WMMA int8

Under fp16, K÷16: ALL aligned (144/16=9 ✓, 288/16=18 ✓, 432/16=27 ✓, 576/16=36 ✓)
→ fp16: monotone by size (smaller=faster), NO rank-flip
→ int8: p50 aligned, p25 NOT aligned → alignment creates coupling that fp16 doesn't have

This script:
1. Builds int8 3×3 conv DIRECTLY in TVM relax IR (not via QDQ-ONNX → avoids FP32+QDQ trap)
2. Tunes each config
3. Dumps generated CUDA and greps for dp4a/__dp4a/mma*.sync signatures
4. Reports alignment vs timing correlation

Usage: python l3_int8_diag.py <out_csv> [gpu=5] [trials=300]
"""
import sys, os, re, time, traceback, json, subprocess, tempfile
import numpy as np

OUT_CSV = sys.argv[1] if len(sys.argv) > 1 else "/exdata/jichengzhi/s2_tvm/results/codriving_s0probe_int8.csv"
GPU_ID  = int(sys.argv[2]) if len(sys.argv) > 2 else 5
TRIALS  = int(sys.argv[3]) if len(sys.argv) > 3 else 300
REPS    = 500
BATCH   = 2
KSIZE   = 3

os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

# Configurations: (cin, cout, h, w, label, stage0_k_div32)
# Using stage0 spatial dims: after stride-2 from 256×512 input → 128×256
CONFIGS = [
    (16, 16, 128, 256, "p75_s0", False),   # K=144÷32=4.5 ✗
    (32, 32, 128, 256, "p50_s0", True),    # K=288÷32=9 ✓
    (48, 48, 128, 256, "p25_s0", False),   # K=432÷32=13.5 ✗
    (64, 64, 128, 256, "base_s0", True),   # K=576÷32=18 ✓
    (96, 96, 64, 128, "p25_s1", True),     # stage1 of p25, K=864÷32=27 ✓
    (128, 128, 64, 128, "base_s1", True),  # stage1 of base, K=1152÷32=36 ✓
]

TC_KEYS = (b"dp4a", b"__dp4a", b"mma.sync.aligned.m8n8k32.s8",
           b"tvm_mma_sync", b"wmma::fragment", b"ptx_mma_s8")

import tvm
from tvm import relax
import tvm.s_tir.dlight as dl
import tvm.s_tir.tensor_intrin.cuda  # noqa
from tvm.s_tir.meta_schedule import relax_integration as ri


def build_int8_conv(cin, cout, h, w, batch=2):
    """Build a single 3×3 int8 conv as TVM relax module.
    Direct relax IR construction — NOT via QDQ-ONNX — so inputs are truly int8.
    """
    bb = relax.BlockBuilder()
    pad = 1  # same padding
    x = relax.Var("x", relax.TensorStructInfo((batch, cin, h, w), "int8"))
    wt = relax.Var("wt", relax.TensorStructInfo((cout, cin, KSIZE, KSIZE), "int8"))
    with bb.function("main", [x, wt]):
        with bb.dataflow():
            y = bb.emit(relax.op.nn.conv2d(x, wt, padding=(pad, pad), out_dtype="int32"))
            gv = bb.emit_output(y)
        bb.emit_func_output(gv)
    return bb.finalize()


def get_cuda_source(mod_ex):
    """Attempt to dump CUDA source from compiled module."""
    try:
        src = mod_ex.get_source("cuda")
        return src.encode() if isinstance(src, str) else src
    except Exception:
        pass
    try:
        src = mod_ex.get_source()
        return src.encode() if isinstance(src, str) else src
    except Exception:
        pass
    # Try writing to temp file and reading back
    try:
        with tempfile.NamedTemporaryFile(suffix=".cu", delete=False) as f:
            tmp = f.name
        mod_ex.export_library(tmp + ".so")
        # Check generated files
        for ext in [".cu", ".ptx"]:
            path = tmp + ext
            if os.path.exists(path):
                with open(path, "rb") as rf:
                    return rf.read()
    except Exception:
        pass
    return b""


def check_tc(cuda_bytes):
    """Returns (has_int8_tc: bool, which_key: str)."""
    for k in TC_KEYS:
        if k in cuda_bytes:
            return True, k.decode("utf-8", errors="replace")
    return False, ""


def time_vm(vm, args_tvm, dev, reps):
    vm["main"](*args_tvm); dev.sync()
    vf = vm.time_evaluator("main", dev, number=reps, repeat=5)
    r = vf(*args_tvm)
    return r.mean * 1e6, min(r.results) * 1e6


dev = tvm.cuda(0)
tgt = tvm.target.Target.from_device(dev)

results = []

for cin, cout, h, w, label, k_div32_ok in CONFIGS:
    k = cin * KSIZE * KSIZE
    k_div32 = k % 32 == 0
    print(f"\n[L3-INT8-DIAG] === {label}: Cin={cin} K={k} K÷32={'OK' if k_div32 else 'FAIL'} ===", flush=True)

    mod = build_int8_conv(cin, cout, h, w, BATCH)
    rng = np.random.RandomState(42)
    x_data  = rng.randint(-64, 64, (BATCH, cin, h, w)).astype("int8")
    wt_data = rng.randint(-64, 64, (cout, cin, KSIZE, KSIZE)).astype("int8")

    # Tune
    workdir = f"/exdata/jichengzhi/s2_tvm/ms_work_l3_int8_{label}_v2"
    # Use fresh workdir (v2 suffix avoids poisoning from old 200-trial run)
    os.makedirs(workdir, exist_ok=True)

    # Prepare for MetaSchedule
    seq = tvm.transform.Sequential([
        relax.transform.LegalizeOps(), relax.transform.AnnotateTIROpPattern(),
        relax.transform.FuseOps(), relax.transform.FuseTIR()])
    with tgt, tvm.transform.PassContext(opt_level=3):
        modt = seq(mod)

    t0 = time.time()
    lat_us = -1.0
    has_tc = False
    tc_key = ""

    try:
        ri.tune_relax(mod=modt, params={}, target=tgt, work_dir=workdir,
                      max_trials_global=TRIALS, seed=42)
        tune_s = time.time() - t0
        print(f"[L3-INT8-DIAG] {label}: tune done in {tune_s:.0f}s", flush=True)

        # Apply schedule
        sched = relax.transform.MetaScheduleApplyDatabase(work_dir=workdir)(modt)
        sched = dl.ApplyDefaultSchedule(
            dl.gpu.Matmul(), dl.gpu.GEMV(), dl.gpu.Reduction(),
            dl.gpu.GeneralReduction(), dl.gpu.Fallback())(sched)
        ex_tuned = tvm.compile(sched, target=tgt)

        # Diagnostic gate: check CUDA for dp4a/WMMA signatures
        cuda_bytes = get_cuda_source(ex_tuned)
        if cuda_bytes:
            has_tc, tc_key = check_tc(cuda_bytes)
            # Also dump CUDA for manual inspection
            diag_path = f"/exdata/jichengzhi/s2_tvm/results/cod_int8_diag_{label}.cu"
            with open(diag_path, "wb") as f:
                f.write(cuda_bytes[:50000])  # first 50k bytes
            print(f"[L3-INT8-DIAG] {label}: CUDA dumped to {diag_path} ({len(cuda_bytes)} bytes)", flush=True)
            print(f"[L3-INT8-DIAG] {label}: has_tc={has_tc} tc_key={tc_key!r}", flush=True)
        else:
            print(f"[L3-INT8-DIAG] {label}: CUDA dump FAILED (no source)", flush=True)

        vm_tuned = relax.VirtualMachine(ex_tuned, dev)
        x_tvm  = tvm.nd.array(x_data, dev)
        wt_tvm = tvm.nd.array(wt_data, dev)
        lat_us, lat_mn = time_vm(vm_tuned, [x_tvm, wt_tvm], dev, REPS)
        print(f"[L3-INT8-DIAG] {label}: tuned lat={lat_us:.2f}µs (min={lat_mn:.2f}µs) has_tc={has_tc}", flush=True)

    except Exception:
        traceback.print_exc()
        print(f"[L3-INT8-DIAG] {label}: FAILED", flush=True)

    results.append({
        "label": label, "cin": cin, "cout": cout, "k": k,
        "k_div32": int(k_div32), "k_mod32": k % 32,
        "lat_us": lat_us, "has_tc": int(has_tc), "tc_key": tc_key,
        "trials": TRIALS, "note": "direct_relax_int8_not_qdq"
    })

# Write CSV
print("\n[L3-INT8-DIAG] === SUMMARY ===", flush=True)
header = "label,cin,cout,k,k_div32,k_mod32,lat_us,has_tc,tc_key,trials,note"
rows = []
for r in results:
    row = f"{r['label']},{r['cin']},{r['cout']},{r['k']},{r['k_div32']},{r['k_mod32']},{r['lat_us']:.2f},{r['has_tc']},{r['tc_key']},{r['trials']},{r['note']}"
    rows.append(row)
    print(f"  {r['label']}: Cin={r['cin']} K={r['k']} K÷32_ok={r['k_div32']} lat={r['lat_us']:.1f}µs has_tc={r['has_tc']}", flush=True)

with open(OUT_CSV, "w") as f:
    f.write(header + "\n")
    f.write("\n".join(rows) + "\n")
print(f"\n[L3-INT8-DIAG] Written to {OUT_CSV}", flush=True)

# Key verdict on coupling
aligned_lats = [r["lat_us"] for r in results if r["k_div32"] == 1 and r["lat_us"] > 0]
misaligned_lats = [r["lat_us"] for r in results if r["k_div32"] == 0 and r["lat_us"] > 0]
# Compare p25_s0 vs base_s0 (same stage, different alignment)
p25_lat = next((r["lat_us"] for r in results if r["label"] == "p25_s0"), None)
base_lat = next((r["lat_us"] for r in results if r["label"] == "base_s0"), None)
if p25_lat and base_lat and p25_lat > 0 and base_lat > 0:
    ratio = p25_lat / base_lat
    if ratio > 1.0:
        verdict = f"RANK-FLIP: p25_s0({p25_lat:.1f}µs) > base_s0({base_lat:.1f}µs) by {ratio:.2f}x despite p25 having 0.5625x flops → INT8 alignment creates coupling"
    else:
        verdict = f"NO-FLIP: p25_s0({p25_lat:.1f}µs) < base_s0({base_lat:.1f}µs) → INT8 alignment does NOT create rank-flip at stage0 level"
    print(f"\n[L3-INT8-DIAG] VERDICT: {verdict}", flush=True)

print("[L3-INT8-DIAG] DONE", flush=True)
