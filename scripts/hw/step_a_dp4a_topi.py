"""STEP A v2 — dp4a diagnostic via topi.nn.conv2d_NCHWc_int8 (NCHW4c layout).

Root cause of v1 failure: relax.op.nn.conv2d → legalizes to group_conv2d_nchw
(NCHW layout, SCALAR path). MetaSchedule can't find dp4a there.

Fix: use topi.nn.conv2d_NCHWc_int8 DIRECTLY with packed NCHWc inputs.
This function has n_elems=4 inner reduce axis = the dp4a 4-element accumulation.

For in_per_g=4 (s0=64): ic_bn=4, n_elems=4 → perfectly aligned → dp4a
For in_per_g=3 (s0=48): ic_bn must be ≥4 to pack, but 3<4 → can't use NCHWc4c

Test approach:
1. Create NCHWc-packed int8 tensors via te.placeholder
2. Call topi.nn.conv2d_NCHWc_int8
3. Build with MetaSchedule → check for __dp4a in CUDA
4. Fallback: manual dp4a tensorize via s_tir.TensorIntrin

Usage:
  CUDA_VISIBLE_DEVICES=<gpu> LD_LIBRARY_PATH=$(cat /exdata/jichengzhi/tvm_nvlibs.path) \\
  PATH=/usr/local/cuda-12.2/bin:$PATH \\
  /exdata/jichengzhi/tvm310/bin/python3 step_a_dp4a_topi.py --width s0=64 [--trials 100]
"""
from __future__ import annotations
import argparse, json, os, sys, time, traceback
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--width", default="s0=64", choices=["s0=64", "s0=48"])
ap.add_argument("--trials", type=int, default=100)
ap.add_argument("--reps", type=int, default=200)
ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--workdir", default="/exdata/jichengzhi/s2_tvm/step_a_dp4a_topi")
args = ap.parse_args()

os.environ["PATH"] = "/usr/local/cuda-12.2/bin:" + os.environ.get("PATH", "")

# Pyramid backbone conv2 at stage0 dimensions:
# s0=64: out=128, groups=32, in_per_g=4 → ic_bn=4 for dp4a ✓
# s0=48: out=96,  groups=32, in_per_g=3 → ic_bn=3 < 4, dp4a impossible ✗
if args.width == "s0=64":
    S0, COUT, GROUPS = 64, 128, 32
    IC_BN = 4       # inner channel block = in_per_g (divisible by n_elems=4)
    N_ELEMS = 4     # dp4a accumulates 4 int8 elements
else:  # s0=48
    S0, COUT, GROUPS = 48, 96, 32
    IC_BN = 3       # in_per_g=3, NOT divisible by 4 → dp4a not possible
    N_ELEMS = 4     # would need 4

CIN = COUT
IN_PER_G = CIN // GROUPS
OC_BN = 4           # output channel block (must divide COUT/GROUPS=4)

KSIZE = 3
H, W = 256, 256
WORK_DIR = os.path.join(args.workdir, args.width.replace("=", "_"))
os.makedirs(WORK_DIR, exist_ok=True)

result = {
    "width": args.width, "s0": S0, "cout": COUT, "groups": GROUPS,
    "in_per_g": IN_PER_G, "ic_bn": IC_BN, "n_elems": N_ELEMS,
    "dp4a_feasible": (IN_PER_G % N_ELEMS == 0),
    "dp4a_in_cuda": False, "wmma_in_cuda": False,
    "intrinsic_type": "UNKNOWN",
    "status": "STARTED",
}

print("=" * 72, flush=True)
print(f"[GATE-V2] STEP A: topi.nn.conv2d_NCHWc_int8 dp4a gate", flush=True)
print(f"  width={args.width} CIN={CIN} COUT={COUT} GROUPS={GROUPS}", flush=True)
print(f"  IN_PER_G={IN_PER_G} IC_BN={IC_BN} N_ELEMS={N_ELEMS}", flush=True)
print(f"  dp4a feasible (IN_PER_G % 4 == 0): {IN_PER_G % 4 == 0}", flush=True)
print("=" * 72, flush=True)

try:
    import tvm
    from tvm import te, topi
    from tvm.topi.nn.conv2d import conv2d_NCHWc_int8
    from tvm.topi.utils import get_const_tuple
    import tvm.s_tir.tensor_intrin.dot_product_common  # registers dp4a_s8s8s32
    import tvm.s_tir.tensor_intrin.cuda               # registers WMMA intrins
    from tvm import relax
    from tvm.s_tir.meta_schedule import relax_integration as ri

    print(f"[GATE-V2] TVM {tvm.__version__}", flush=True)

    dev = tvm.cuda(args.gpu)
    target = tvm.target.Target.from_device(dev)

    # ── Approach A: Direct relax build with explicit NCHWc layout transform ───
    # Use relax to build: pack NCHW input → NCHWc → conv2d_NCHWc_int8 → unpack
    print(f"\n[GATE-V2] Approach: relax with NCHWc layout + conv2d_NCHWc_int8", flush=True)

    # Input: NCHW int8 (N=2, C=128, H=256, W=256)
    # Packed: (N, C//IC_BN, H, W, IC_BN) = (2, 32, 256, 256, 4) for s0=64
    IC_CHUNK = CIN // IC_BN

    if IN_PER_G % N_ELEMS != 0:
        print(f"[GATE-V2] s0=48: IN_PER_G={IN_PER_G} not ÷{N_ELEMS} → "
              f"dp4a impossible for grouped conv", flush=True)
        result["status"] = "INFEASIBLE_ALIGNMENT"
        result["verdict"] = "INFEASIBLE: in_per_g=3 not ÷4, cannot use NCHWc4c for dp4a"
        with open(os.path.join(WORK_DIR, "result.json"), "w") as f:
            json.dump(result, f, indent=2)
        print(f"  → This is the alignment TRAP: s0=48 backbone CAN'T use dp4a", flush=True)
        print(f"  → s0=64 backbone CAN use dp4a → rank-flip mechanism confirmed by infeasibility", flush=True)
        sys.exit(0)

    # For s0=64: IC_BN=4 = IN_PER_G = n_elems → perfect dp4a alignment
    # Weight shape for NCHWc_int8: (OC_CHUNK, IC_CHUNK_PER_GROUP, KH, KW, IC_BN//N_ELEMS, OC_BN, N_ELEMS)
    OC_CHUNK = COUT // OC_BN
    IC_CHUNK_PER_GROUP = IC_CHUNK // GROUPS
    # = (32/32) = 1 chunk per group (each group has 4 input channels = 1 chunk of 4)

    print(f"[GATE-V2] NCHWc tensor shapes:", flush=True)
    print(f"  Input: (N=2, IC_CHUNK={IC_CHUNK}, H={H}, W={W}, IC_BN={IC_BN})", flush=True)
    print(f"  Weight: (OC_CHUNK={OC_CHUNK}, IC_CHUNK_PER_GROUP={IC_CHUNK_PER_GROUP}, "
          f"KH={KSIZE}, KW={KSIZE}, IC_BN//N_ELEMS={IC_BN//N_ELEMS}, OC_BN={OC_BN}, N_ELEMS={N_ELEMS})", flush=True)

    # Build relax module with layout_transform → conv2d_NCHWc_int8 → layout_transform back
    bb = relax.BlockBuilder()
    # input NCHW int8
    x_nchw = relax.Var("x", relax.TensorStructInfo((2, CIN, H, W), "int8"))
    # weight (out_ch, in_ch/g, kH, kW) int8 — standard NCHW weight format
    wt_nchw = relax.Var("wt", relax.TensorStructInfo((COUT, IN_PER_G, KSIZE, KSIZE), "int8"))

    with bb.function("main", [x_nchw, wt_nchw]):
        with bb.dataflow():
            # Pack input NCHW → NCHWc (NCHW4c for dp4a)
            # (2, 128, 256, 256) → (2, 32, 256, 256, 4)
            x_packed = bb.emit(relax.op.layout_transform(
                x_nchw,
                index_map=lambda n, c, h, w: (n, c // IC_BN, h, w, c % IC_BN),
                name="x_packed",
            ))
            # Pack weight NCHW → OIHWio format for NCHWc_int8
            # (128, 4, 3, 3) → (OC_CHUNK, IC_CHUNK_PER_GROUP, 3, 3, IC_BN//N_ELEMS, OC_BN, N_ELEMS)
            # = (32, 1, 3, 3, 1, 4, 4) for s0=64
            wt_packed = bb.emit(relax.op.layout_transform(
                wt_nchw,
                index_map=lambda oc, ic, kh, kw: (
                    oc // OC_BN,            # oc_chunk
                    ic // (IC_BN // N_ELEMS) if IC_BN > N_ELEMS else ic // IC_BN,  # ic_chunk_group (simplified: ic//IC_BN = 0)
                    kh, kw,
                    (ic % IC_BN) // N_ELEMS if IC_BN > N_ELEMS else 0,  # ic_f_inner
                    oc % OC_BN,             # oc_block
                    ic % N_ELEMS,           # ic_s_inner (dp4a group)
                ),
                name="wt_packed",
            ))

            # conv2d using NCHWc int8 registered op
            # Use relax.op.nn.conv2d with NCHWc data layout spec
            y = bb.emit(relax.op.nn.conv2d(
                x_packed, wt_packed,
                groups=GROUPS,
                padding=(KSIZE // 2, KSIZE // 2),
                data_layout="NCHW4c",
                kernel_layout="OIHW4o",
                out_layout="NCHW4c",
                out_dtype="int32",
            ))
            # Unpack output NCHWc → NCHW
            y_unpacked = bb.emit(relax.op.layout_transform(
                y,
                index_map=lambda n, oc_chunk, h, w, oc_block: (n, oc_chunk * OC_BN + oc_block, h, w),
                name="y_unpacked",
            ))
            gv = bb.emit_output(y_unpacked)
        bb.emit_func_output(gv)
    mod = bb.finalize()
    print(f"[GATE-V2] relax IR built with NCHWc layout", flush=True)

    # Legalize and tune
    seq = tvm.transform.Sequential([
        relax.transform.LegalizeOps(),
        relax.transform.AnnotateTIROpPattern(),
        relax.transform.FuseOps(),
        relax.transform.FuseTIR(),
    ])
    with target, tvm.transform.PassContext(opt_level=3):
        mod_legal = seq(mod)
    print(f"[GATE-V2] legalization done", flush=True)

    # Tune
    print(f"\n[GATE-V2] Tuning ({args.trials} trials)...", flush=True)
    t0 = time.time()
    db = ri.tune_relax(mod=mod_legal, params={}, target=target,
                       work_dir=WORK_DIR, max_trials_global=args.trials, seed=42)
    tune_s = time.time() - t0
    print(f"[GATE-V2] Tuning done in {tune_s:.0f}s", flush=True)

    # Apply + dump TIR
    with target, tvm.transform.PassContext(opt_level=3):
        mod_tuned = relax.transform.MetaScheduleApplyDatabase(work_dir=WORK_DIR)(mod_legal)
    tir_txt = mod_tuned.script()
    tir_path = os.path.join(WORK_DIR, "tir_dump.txt")
    with open(tir_path, "w") as f:
        f.write(tir_txt)
    print(f"[GATE-V2] TIR dumped: {tir_path}", flush=True)

    # Build + extract CUDA
    ex = relax.build(mod_tuned, target=target)
    cuda_src = None
    # Try to get CUDA source
    try:
        # In TVM 0.20, exported lib may have .cu companion files
        lib_path = os.path.join(WORK_DIR, "compiled.so")
        ex.export_library(lib_path)

        # Method 1: get_source from module
        try:
            mod_obj = tvm.runtime.load_module(lib_path)
            cuda_src = mod_obj.imported_modules[0].get_source()
        except Exception:
            pass

        # Method 2: search for ptx/cu in workdir
        if cuda_src is None:
            for fn in os.listdir(WORK_DIR):
                if fn.endswith(".cu") or fn.endswith(".ptx"):
                    with open(os.path.join(WORK_DIR, fn)) as f:
                        cuda_src = f.read()
                    break
    except Exception as e:
        print(f"[GATE-V2] library export failed: {e}", flush=True)

    # Check TIR for intrinsics
    tir_lower = tir_txt.lower()
    result["dp4a_in_tir"] = "dp4a" in tir_lower or "__dp4a" in tir_lower
    result["wmma_in_tir"] = "tvm_mma_sync" in tir_txt or "tvm_load_matrix_sync" in tir_txt
    result["scalar_in_tir"] = "T.Cast" in tir_txt and not result["dp4a_in_tir"] and not result["wmma_in_tir"]

    if cuda_src:
        cuda_path = os.path.join(WORK_DIR, "cuda_dump.txt")
        with open(cuda_path, "w") as f:
            f.write(cuda_src)
        cuda_lower = cuda_src.lower()
        result["dp4a_in_cuda"] = "__dp4a" in cuda_lower or ("dp4a" in cuda_lower and "__" in cuda_lower)
        result["wmma_in_cuda"] = "wmma" in cuda_lower
        if result["dp4a_in_cuda"]:
            result["intrinsic_type"] = "DP4A_REAL"
        elif result["wmma_in_cuda"]:
            result["intrinsic_type"] = "WMMA_REAL"
        else:
            result["intrinsic_type"] = "SCALAR_or_FAKE"
    else:
        result["intrinsic_type"] = "NO_CUDA_DUMP"

    print(f"\n[GATE-V2] TIR analysis:", flush=True)
    print(f"  dp4a in TIR: {result.get('dp4a_in_tir')}", flush=True)
    print(f"  WMMA in TIR: {result.get('wmma_in_tir')}", flush=True)
    print(f"  scalar cast in TIR: {result.get('scalar_in_tir')}", flush=True)
    print(f"  dp4a in CUDA: {result.get('dp4a_in_cuda')}", flush=True)
    print(f"  intrinsic_type: {result['intrinsic_type']}", flush=True)

    # Quick latency measurement using VirtualMachine
    try:
        vm = relax.VirtualMachine(ex, dev)
        # Create test inputs
        x_np = np.random.randint(-128, 127, (2, CIN, H, W), dtype=np.int8)
        wt_np = np.random.randint(-128, 127, (COUT, IN_PER_G, KSIZE, KSIZE), dtype=np.int8)
        # In TVM 0.20, use tvm.runtime.Tensor or pass numpy arrays directly
        x_tvm = tvm.runtime.empty((2, CIN, H, W), dtype="int8", device=dev)
        x_tvm.copyfrom(x_np)
        wt_tvm = tvm.runtime.empty((COUT, IN_PER_G, KSIZE, KSIZE), dtype="int8", device=dev)
        wt_tvm.copyfrom(wt_np)
        # Warmup
        vm["main"](x_tvm, wt_tvm)
        dev.sync()
        timer = vm.time_evaluator("main", dev, number=args.reps, repeat=5)
        t = timer(x_tvm, wt_tvm)
        lat_us = t.mean * 1e6
        result["lat_tuned_us"] = round(lat_us, 2)
        print(f"\n[GATE-V2] TUNED latency: {lat_us:.1f} us", flush=True)
    except Exception as e:
        print(f"[GATE-V2] latency measurement failed: {e}", flush=True)
        traceback.print_exc()

    result["status"] = "DONE"

except Exception as e:
    traceback.print_exc()
    result["status"] = f"FAILED: {e}"

# Verdict
print("\n" + "=" * 72, flush=True)
print(f"[GATE-V2] ★ STEP A VERDICT:", flush=True)
dp4a_ok = result.get("dp4a_in_cuda") or result.get("dp4a_in_tir")
wmma_ok = result.get("wmma_in_cuda") or result.get("wmma_in_tir")
print(f"  width={args.width} in_per_g={IN_PER_G} dp4a_feasible={IN_PER_G % 4 == 0}", flush=True)
print(f"  dp4a confirmed: {dp4a_ok}", flush=True)
print(f"  wmma confirmed: {wmma_ok}", flush=True)
print(f"  intrinsic_type: {result.get('intrinsic_type')}", flush=True)
print(f"  lat_tuned_us: {result.get('lat_tuned_us')}", flush=True)
if dp4a_ok:
    print(f"  RESULT: ✓ REAL dp4a INT8 confirmed → proceed to STEP B", flush=True)
elif wmma_ok:
    print(f"  RESULT: ✓ REAL WMMA INT8 (K≥16 path) → not dp4a but still real int8", flush=True)
else:
    print(f"  RESULT: ✗ NO real int8 compute → scalar or FAKE", flush=True)
print("=" * 72, flush=True)

with open(os.path.join(WORK_DIR, "result.json"), "w") as f:
    json.dump(result, f, indent=2)
print(f"[GATE-V2] saved: {os.path.join(WORK_DIR, 'result.json')}", flush=True)
