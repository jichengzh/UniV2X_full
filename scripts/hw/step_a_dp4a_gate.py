"""STEP A — dp4a diagnostic gate for Pyramid backbone grouped conv.

Builds ONE grouped conv directly in TVM relax with int8 → int32 dtype
(BYPASSES QDQ-ONNX entirely — no fake FP32+QDQ overhead).

Target: s0=64 case: conv2 with groups=32, in_per_g=4, weight=(128, 4, 3, 3)
These are the REAL dimensions from base_backbone.onnx layer0 grouped conv.

STEP A question: Does MetaSchedule generate real __dp4a in the CUDA kernel
for this grouped conv when in_per_g=4 (÷4-aligned)?

Expected outputs:
- /exdata/jichengzhi/s2_tvm/step_a_dp4a_gate/tir_dump.txt  — scheduled TIR
- /exdata/jichengzhi/s2_tvm/step_a_dp4a_gate/cuda_dump.txt — generated CUDA
- /exdata/jichengzhi/s2_tvm/step_a_dp4a_gate/result.json   — verdict + latency

Usage (H800, idle GPU):
  CUDA_VISIBLE_DEVICES=5 LD_LIBRARY_PATH=$(cat /exdata/jichengzhi/tvm_nvlibs.path) \
  PATH=/usr/local/cuda-12.2/bin:$PATH \
  /exdata/jichengzhi/tvm310/bin/python3 step_a_dp4a_gate.py [--width s0=48|s0=64] [--trials N]
"""
from __future__ import annotations
import argparse, json, os, sys, time, traceback
import numpy as np

# ── CLI args ─────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--width", default="s0=64", choices=["s0=64", "s0=48"],
                help="s0=64 → in_per_g=4 (dp4a aligned); s0=48 → in_per_g=3 (misaligned)")
ap.add_argument("--trials", type=int, default=100,
                help="MetaSchedule trials (100 for gate, 500+ for timing)")
ap.add_argument("--reps", type=int, default=200)
ap.add_argument("--gpu", type=int, default=0,
                help="GPU index (CUDA_VISIBLE_DEVICES overrides)")
ap.add_argument("--workdir", default="/exdata/jichengzhi/s2_tvm/step_a_dp4a_gate")
args = ap.parse_args()

# Grouped conv dimensions for each s0 value:
#   Base backbone conv2 at stage0:
#   - s0=64: out_channels=128, groups=32, in_per_g=128/32=4 → weight (128,4,3,3)
#   - s0=48: out_channels=96,  groups=32, in_per_g=96/32=3  → weight (96,3,3,3)
if args.width == "s0=64":
    S0 = 64
    COUT = 128   # 2 * s0
    GROUPS = 32
    # in_per_g = COUT // GROUPS = 4 → ÷4 aligned, dp4a possible
else:  # s0=48
    S0 = 48
    COUT = 96    # 2 * s0
    GROUPS = 32
    # in_per_g = COUT // GROUPS = 3 → NOT ÷4 aligned, dp4a impossible

CIN = COUT      # depthwise-like: in_channels = out_channels for conv2
IN_PER_G = CIN // GROUPS
KSIZE = 3
H, W = 256, 256  # backbone input spatial
TRIALS = args.trials
REPS = args.reps
WORK_DIR = os.path.join(args.workdir, args.width.replace("=", "_"))
RESULT_JSON = os.path.join(WORK_DIR, "result.json")

os.makedirs(WORK_DIR, exist_ok=True)
os.environ["PATH"] = "/usr/local/cuda-12.2/bin:" + os.environ.get("PATH", "")

result = {
    "width": args.width,
    "s0": S0, "cout": COUT, "groups": GROUPS, "in_per_g": IN_PER_G,
    "ksize": KSIZE, "H": H, "W": W,
    "trials": TRIALS,
    "dp4a_in_tir": False,
    "dp4a_in_cuda": False,
    "wmma_in_tir": False,
    "wmma_in_cuda": False,
    "intrinsic_type": "UNKNOWN",
    "lat_default_us": None,
    "lat_tuned_us": None,
    "status": "STARTED",
}


def save():
    with open(RESULT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[GATE] result saved → {RESULT_JSON}", flush=True)


print("=" * 72, flush=True)
print(f"[GATE] STEP A: dp4a diagnostic gate", flush=True)
print(f"[GATE] width={args.width} Cin={CIN} Cout={COUT} groups={GROUPS} "
      f"in_per_g={IN_PER_G}", flush=True)
print(f"[GATE] dp4a feasible (in_per_g % 4 == 0): {IN_PER_G % 4 == 0}", flush=True)
print(f"[GATE] work_dir={WORK_DIR} trials={TRIALS}", flush=True)
print("=" * 72, flush=True)

try:
    import tvm
    from tvm import relax
    from tvm.s_tir.meta_schedule import relax_integration as ri
    import tvm.s_tir.tensor_intrin.cuda   # registers WMMA/PTX intrins
    import tvm.s_tir.tensor_intrin.dot_product_common  # registers dp4a intrin

    print(f"[GATE] TVM {tvm.__version__}", flush=True)

    dev = tvm.cuda(args.gpu)
    target = tvm.target.Target.from_device(dev)
    print(f"[GATE] target={target}", flush=True)

    # ── STEP A.1: Build relax IR directly with int8 dtype ─────────────────────
    # This is the KEY difference from QDQ-ONNX approach:
    # We give TVM int8 tensors directly → it generates REAL int8 compute kernels
    print(f"\n[GATE] A.1: building relax IR with int8 → int32 compute", flush=True)
    bb = relax.BlockBuilder()
    # Input: (batch=2, in_channels=CIN, H, W) in int8
    x = relax.Var("x", relax.TensorStructInfo((2, CIN, H, W), "int8"))
    # Weight: (out_channels=COUT, in_channels/groups=IN_PER_G, kH, kW) in int8
    wt = relax.Var("wt", relax.TensorStructInfo((COUT, IN_PER_G, KSIZE, KSIZE), "int8"))
    pad = KSIZE // 2
    with bb.function("main", [x, wt]):
        with bb.dataflow():
            # grouped conv2d with int32 accumulation
            y = bb.emit(relax.op.nn.conv2d(
                x, wt,
                groups=GROUPS,
                padding=(pad, pad),
                out_dtype="int32",
            ))
            gv = bb.emit_output(y)
        bb.emit_func_output(gv)
    mod = bb.finalize()
    print(f"[GATE] A.1 relax IR built: Cin={CIN}, Cout={COUT}, groups={GROUPS}, "
          f"ksize={KSIZE}", flush=True)

    # ── STEP A.2: Legalize ─────────────────────────────────────────────────────
    print(f"\n[GATE] A.2: legalizing ops", flush=True)
    seq = tvm.transform.Sequential([
        relax.transform.LegalizeOps(),
        relax.transform.AnnotateTIROpPattern(),
        relax.transform.FuseOps(),
        relax.transform.FuseTIR(),
    ])
    with target, tvm.transform.PassContext(opt_level=3):
        mod_legal = seq(mod)
    print(f"[GATE] A.2 legalization done", flush=True)

    # ── STEP A.3: Default schedule baseline ───────────────────────────────────
    print(f"\n[GATE] A.3: benchmarking with DEFAULT schedule (dlight)", flush=True)
    try:
        from tvm.s_tir import dlight
        with target, tvm.transform.PassContext(opt_level=3):
            mod_default = dlight.ApplyDefaultSchedule(
                dlight.gpu.Matmul(),
                dlight.gpu.GEMV(),
                dlight.gpu.Reduction(),
                dlight.gpu.GeneralReduction(),
                dlight.gpu.Fallback(),
            )(mod_legal)
        ex_def = relax.build(mod_default, target=target)
        vm_def = relax.VirtualMachine(ex_def, dev)
        x_data = tvm.runtime.ndarray.array(np.random.randint(-128, 127, (2, CIN, H, W), dtype=np.int8), dev)
        wt_data = tvm.runtime.ndarray.array(np.random.randint(-128, 127, (COUT, IN_PER_G, KSIZE, KSIZE), dtype=np.int8), dev)
        vm_def["main"](x_data, wt_data)
        dev.sync()
        timer = vm_def.time_evaluator("main", dev, number=REPS, repeat=5)
        t = timer(x_data, wt_data)
        lat_def = t.mean * 1e6
        result["lat_default_us"] = round(lat_def, 2)
        print(f"[GATE] A.3 DEFAULT lat: {lat_def:.1f} us", flush=True)
    except Exception as e:
        print(f"[GATE] A.3 default schedule FAILED: {e}", flush=True)
        traceback.print_exc()

    # ── STEP A.4: MetaSchedule tuning ─────────────────────────────────────────
    print(f"\n[GATE] A.4: MetaSchedule tuning ({TRIALS} trials)", flush=True)
    t0_tune = time.time()
    db = ri.tune_relax(
        mod=mod_legal, params={}, target=target,
        work_dir=WORK_DIR,
        max_trials_global=TRIALS,
        seed=42,
    )
    tune_s = time.time() - t0_tune
    print(f"[GATE] A.4 tuning done in {tune_s:.0f}s", flush=True)

    # ── STEP A.5: Apply best schedule + dump TIR ─────────────────────────────
    print(f"\n[GATE] A.5: applying best schedule + dumping TIR", flush=True)
    with target, tvm.transform.PassContext(opt_level=3):
        mod_tuned = relax.transform.MetaScheduleApplyDatabase(work_dir=WORK_DIR)(mod_legal)

    tir_txt = mod_tuned.script()
    tir_path = os.path.join(WORK_DIR, "tir_dump.txt")
    with open(tir_path, "w") as f:
        f.write(tir_txt)
    print(f"[GATE] A.5 TIR dumped: {tir_path} ({len(tir_txt)} chars)", flush=True)

    # ── STEP A.6: Check TIR for dp4a ──────────────────────────────────────────
    tir_lower = tir_txt.lower()
    dp4a_in_tir = "dp4a" in tir_lower or "__dp4a" in tir_lower
    wmma_in_tir = "tvm_mma_sync" in tir_txt or "tvm_load_matrix_sync" in tir_txt
    result["dp4a_in_tir"] = dp4a_in_tir
    result["wmma_in_tir"] = wmma_in_tir
    print(f"[GATE] A.6 TIR analysis:", flush=True)
    print(f"  dp4a in TIR: {dp4a_in_tir}", flush=True)
    print(f"  WMMA in TIR: {wmma_in_tir}", flush=True)
    if dp4a_in_tir:
        # Find the dp4a line for evidence
        for line in tir_txt.split("\n"):
            if "dp4a" in line.lower():
                print(f"  TIR dp4a line: {line.strip()[:120]}", flush=True)
                break

    # ── STEP A.7: Build executable + dump CUDA ────────────────────────────────
    print(f"\n[GATE] A.7: building + extracting CUDA source", flush=True)
    ex_tuned = relax.build(mod_tuned, target=target)
    # Try to get CUDA source from the compiled module
    cuda_src = None
    try:
        # Method 1: imported_modules (CUDA ptx/cubin)
        imported = ex_tuned.mod.imported_modules
        for i, m in enumerate(imported):
            try:
                src = m.get_source()
                if src and ("__global__" in src or "cuda" in src.lower()):
                    cuda_src = src
                    print(f"[GATE] A.7 CUDA source from imported_modules[{i}]: "
                          f"{len(src)} chars", flush=True)
                    break
            except Exception:
                pass
    except Exception:
        pass

    if cuda_src is None:
        try:
            # Method 2: DSO source export
            cuda_src = ex_tuned.mod.get_source("cuda")
            print(f"[GATE] A.7 CUDA source from get_source('cuda'): "
                  f"{len(cuda_src)} chars", flush=True)
        except Exception:
            pass

    if cuda_src is None:
        # Method 3: export library and read
        try:
            lib_path = os.path.join(WORK_DIR, "tuned_lib.so")
            ex_tuned.export_library(lib_path)
            print(f"[GATE] A.7 library exported to {lib_path}", flush=True)
            # Try reading any .cu or .ptx companion files
            for f in os.listdir(WORK_DIR):
                if f.endswith(".cu") or f.endswith(".ptx"):
                    with open(os.path.join(WORK_DIR, f)) as fh:
                        cuda_src = fh.read()
                    print(f"[GATE] A.7 found CUDA source file: {f}", flush=True)
                    break
        except Exception as e:
            print(f"[GATE] A.7 export failed: {e}", flush=True)

    if cuda_src:
        cuda_path = os.path.join(WORK_DIR, "cuda_dump.txt")
        with open(cuda_path, "w") as f:
            f.write(cuda_src)
        print(f"[GATE] A.7 CUDA source dumped: {cuda_path}", flush=True)

        # Critical check: grep for dp4a
        cuda_lower = cuda_src.lower()
        dp4a_in_cuda = "__dp4a" in cuda_lower or "dp4a" in cuda_lower
        wmma_in_cuda = "wmma" in cuda_lower or "mma.sync" in cuda_lower
        result["dp4a_in_cuda"] = dp4a_in_cuda
        result["wmma_in_cuda"] = wmma_in_cuda

        if dp4a_in_cuda:
            result["intrinsic_type"] = "DP4A_REAL"
            # Extract surrounding context for evidence
            for line in cuda_src.split("\n"):
                if "__dp4a" in line.lower() or ("dp4a" in line.lower() and "__" in line):
                    print(f"  ★ REAL dp4a line: {line.strip()[:120]}", flush=True)
                    break
        elif wmma_in_cuda:
            result["intrinsic_type"] = "WMMA"
            for line in cuda_src.split("\n"):
                if "wmma" in line.lower():
                    print(f"  ★ WMMA line: {line.strip()[:120]}", flush=True)
                    break
        else:
            result["intrinsic_type"] = "SCALAR_FP32_or_FAKE"
            print(f"  ★ WARNING: no dp4a or WMMA found — checking if FP32", flush=True)
            for line in cuda_src.split("\n"):
                if "float" in line.lower() and "__global__" not in line:
                    print(f"  FP32 line: {line.strip()[:120]}", flush=True)
                    break

        print(f"\n[GATE] A.7 CUDA analysis:", flush=True)
        print(f"  dp4a in CUDA: {dp4a_in_cuda}", flush=True)
        print(f"  WMMA in CUDA: {wmma_in_cuda}", flush=True)
        print(f"  intrinsic_type: {result['intrinsic_type']}", flush=True)
    else:
        result["intrinsic_type"] = "CUDA_DUMP_FAILED"
        print(f"[GATE] A.7 WARNING: could not dump CUDA source", flush=True)

    # ── STEP A.8: Benchmark tuned ─────────────────────────────────────────────
    print(f"\n[GATE] A.8: benchmarking tuned kernel", flush=True)
    try:
        vm_tuned = relax.VirtualMachine(ex_tuned, dev)
        x_data = tvm.runtime.ndarray.array(np.random.randint(-128, 127, (2, CIN, H, W), dtype=np.int8), dev)
        wt_data = tvm.runtime.ndarray.array(np.random.randint(-128, 127, (COUT, IN_PER_G, KSIZE, KSIZE), dtype=np.int8), dev)
        # Warmup
        vm_tuned["main"](x_data, wt_data)
        dev.sync()
        timer = vm_tuned.time_evaluator("main", dev, number=REPS, repeat=5)
        t = timer(x_data, wt_data)
        lat_tuned = t.mean * 1e6
        lat_min = min(t.results) * 1e6
        result["lat_tuned_us"] = round(lat_tuned, 2)
        result["lat_tuned_min_us"] = round(lat_min, 2)
        print(f"[GATE] A.8 TUNED lat: {lat_tuned:.1f} us (min={lat_min:.1f} us)", flush=True)

        if result["lat_default_us"] is not None:
            ratio = result["lat_default_us"] / lat_tuned
            result["tuned_vs_default_ratio"] = round(ratio, 3)
            print(f"[GATE] A.8 tuned_vs_default: {ratio:.2f}x", flush=True)
    except Exception as e:
        print(f"[GATE] A.8 tuned benchmark FAILED: {e}", flush=True)
        traceback.print_exc()

    result["status"] = "DONE"

except Exception as e:
    traceback.print_exc()
    result["status"] = f"FAILED: {e}"

# ── VERDICT ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 72, flush=True)
print("[GATE] ★ STEP A VERDICT:", flush=True)
dp4a_ok = result.get("dp4a_in_cuda") or result.get("dp4a_in_tir")
wmma_ok = result.get("wmma_in_cuda") or result.get("wmma_in_tir")
print(f"  width={args.width} in_per_g={IN_PER_G} (÷4-aligned: {IN_PER_G % 4 == 0})", flush=True)
print(f"  dp4a in TIR: {result.get('dp4a_in_tir')}", flush=True)
print(f"  dp4a in CUDA: {result.get('dp4a_in_cuda')}", flush=True)
print(f"  WMMA in CUDA: {result.get('wmma_in_cuda')}", flush=True)
print(f"  intrinsic_type: {result.get('intrinsic_type')}", flush=True)
print(f"  lat_default_us: {result.get('lat_default_us')}", flush=True)
print(f"  lat_tuned_us: {result.get('lat_tuned_us')}", flush=True)
if dp4a_ok:
    print(f"  RESULT: ✓ REAL dp4a INT8 confirmed → STEP B can proceed", flush=True)
elif wmma_ok:
    print(f"  RESULT: ✓ WMMA INT8 confirmed → real int8 TC (different intrinsic than dp4a)", flush=True)
else:
    print(f"  RESULT: ✗ NO real int8 compute found → DO NOT fall back to FP32 proxy", flush=True)
print("=" * 72, flush=True)

save()
print(f"[GATE] complete. Result: {RESULT_JSON}", flush=True)
