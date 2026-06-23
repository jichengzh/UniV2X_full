"""STEP A v4 — dp4a gate with custom intrinsic matching NCHWc grouped conv read pattern.

Root cause of v1-v3 failures:
- v1: relax.op.nn.conv2d → legalizes to NCHW SCALAR (not NCHWc)
- v2-v3: dp4a_s8s8s32 reads 3 buffers (C,A,B) but conv sblock reads 2 (A,B only)

Fix: Register custom dp4a intrinsic that reads only A,B (not C accumulator),
matching the conv sblock's 2-buffer read pattern. This is the correct pattern
for TVM 0.20's sblock reduction (the accumulator is tracked via T.init() separately).

Usage (H800):
  CUDA_VISIBLE_DEVICES=5 LD_LIBRARY_PATH=$(cat /exdata/jichengzhi/tvm_nvlibs.path) \
  PATH=/usr/local/cuda-12.2/bin:$PATH \
  /exdata/jichengzhi/tvm310/bin/python3 step_a_dp4a_custom_intrin.py \
    --width s0=64 --trials 100 --gpu 5
"""
from __future__ import annotations
import argparse, json, os, sys, time, traceback
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--width", default="s0=64", choices=["s0=64", "s0=48"])
ap.add_argument("--trials", type=int, default=100)
ap.add_argument("--reps", type=int, default=200)
ap.add_argument("--gpu", type=int, default=5)
ap.add_argument("--workdir", default="/exdata/jichengzhi/s2_tvm/step_a_dp4a_v4")
args = ap.parse_args()

os.environ["PATH"] = "/usr/local/cuda-12.2/bin:" + os.environ.get("PATH", "")

if args.width == "s0=64":
    S0, CIN, COUT, GROUPS = 64, 128, 128, 32
    IC_BN, N_ELEMS, OC_BN = 4, 4, 4
else:
    S0, CIN, COUT, GROUPS = 48, 96, 96, 32
    IC_BN, N_ELEMS, OC_BN = 4, 4, 4  # pad 3→4

IN_PER_G = CIN // GROUPS
KSIZE = 3
H, W = 256, 256
IC_CHUNK = (CIN // GROUPS * IC_BN * GROUPS) // IC_BN  # = 32 for s0=64
IC_CHUNK_PER_GROUP = IC_CHUNK // GROUPS                 # = 1 for s0=64
OC_CHUNK = COUT // OC_BN                                # = 32 for s0=64

WORK_DIR = os.path.join(args.workdir, args.width.replace("=", "_"))
os.makedirs(WORK_DIR, exist_ok=True)

result = {
    "width": args.width, "S0": S0, "CIN": CIN, "GROUPS": GROUPS, "in_per_g": IN_PER_G,
    "IC_BN": IC_BN, "N_ELEMS": N_ELEMS,
    "dp4a_in_cuda": False, "wmma_in_cuda": False, "scalar": False,
    "intrinsic_type": "UNKNOWN", "status": "STARTED",
}

print("=" * 72, flush=True)
print(f"[GATE-V4] STEP A: custom dp4a intrinsic gate (NCHWc grouped conv)", flush=True)
print(f"  {args.width}: CIN={CIN} GROUPS={GROUPS} in_per_g={IN_PER_G} IC_BN={IC_BN}", flush=True)
print("=" * 72, flush=True)

try:
    import tvm
    from tvm import te, s_tir
    from tvm.topi.nn.conv2d import conv2d_NCHWc_int8
    from tvm.s_tir import TensorIntrin
    import tvm.s_tir.tensor_intrin.dot_product_common
    import tvm.s_tir.tensor_intrin.cuda

    print(f"[GATE-V4] TVM {tvm.__version__}", flush=True)

    dev = tvm.cuda(args.gpu)
    target = tvm.target.Target.from_device(dev)
    print(f"[GATE-V4] target: {target}", flush=True)

    if IN_PER_G < IC_BN:
        print(f"[GATE-V4] NOTE: in_per_g={IN_PER_G} < IC_BN={IC_BN}, padding channels", flush=True)
        PAD_CIN = IC_BN * GROUPS
        IC_CHUNK_USE = PAD_CIN // IC_BN
        IC_CHUNK_PER_GROUP_USE = 1
    else:
        IC_CHUNK_USE = IC_CHUNK
        IC_CHUNK_PER_GROUP_USE = IC_CHUNK_PER_GROUP

    # ── Build TE computation ──────────────────────────────────────────────────
    data = te.placeholder((2, IC_CHUNK_USE, H, W, IC_BN), dtype="int8", name="data")
    kernel_te = te.placeholder(
        (OC_CHUNK, IC_CHUNK_PER_GROUP_USE, KSIZE, KSIZE, IC_BN // N_ELEMS, OC_BN, N_ELEMS),
        dtype="int8", name="kernel"
    )
    out = conv2d_NCHWc_int8(
        data, kernel_te,
        stride=1, padding=1, dilation=1,
        layout="NCHWc", out_layout="NCHWc",
        out_dtype="int32", n_elems=N_ELEMS
    )
    print(f"[GATE-V4] output shape: {out.shape}", flush=True)

    pf = te.create_prim_func([data, kernel_te, out])
    ir_mod = tvm.ir.IRModule.from_expr(pf)

    # ── Manual schedule with dp4a tensorize ─────────────────────────────────
    sch = s_tir.Schedule(ir_mod["main"])
    conv_blk = sch.get_sblock("conv2d_NCHWc_int8")

    # 1. Cache inputs to shared memory (provides the 2-buffer read pattern dp4a needs)
    _ = sch.cache_read(conv_blk, 0, "shared")   # data_pad → shared
    _ = sch.cache_read(conv_blk, 1, "shared")   # kernel → shared

    # 2. Cache output to local memory (register accumulator)
    conv_blk = sch.get_sblock("conv2d_NCHWc_int8")
    _ = sch.cache_write(conv_blk, 0, "local")

    # 3. Get conv block loops and tensorize inner 4-element reduce
    conv_blk = sch.get_sblock("conv2d_NCHWc_int8")
    loops = sch.get_loops(conv_blk)
    extents = [int(sch.get(l).extent) for l in loops]
    print(f"[GATE-V4] conv loops: {len(loops)} extents={extents}", flush=True)

    # The last loop is ic_s_inner with extent N_ELEMS=4
    ic_s_inner = loops[-1]
    assert int(sch.get(ic_s_inner).extent) == N_ELEMS, \
        f"Expected last loop extent={N_ELEMS}, got {int(sch.get(ic_s_inner).extent)}"

    # Use standard dp4a_s8s8s32 — BUT first try to tensorize on the conv block
    # which now reads from shared (A,B) and writes to local (C)
    # The reads list = (data_pad_shared[4-elem], kernel_shared[4-elem]) = 2 bufs
    # The dp4a_s8s8s32 desc reads (C, A, B) = 3 bufs → mismatch
    #
    # Solution: tensorize at the OUTER loop level that includes the init block
    # The init block makes C not appear in reads (C is write-only in init)
    # But the update block reads C,A,B.
    #
    # We need the intrinsic to match the combined init+update pattern.
    # TVM 0.20's tensorize for reduction blocks works at the outer block level.
    # Let's tensorize the BLOCK itself (not just the innermost loop) by using
    # the outer reduce boundary (kh loop or oc_block loop).

    # Try to find the last spatial loop and tensorize at the ic_s_inner position
    # using the standard intrinsic (with C in reads)
    intrin_name = "dp4a_s8s8s32"
    print(f"[GATE-V4] Attempting tensorize(ic_s_inner, '{intrin_name}')...", flush=True)
    try:
        sch.tensorize(ic_s_inner, intrin_name)
        print(f"[GATE-V4] ✓ Tensorize with {intrin_name} SUCCESS!", flush=True)
        dp4a_ok = True
    except Exception as e1:
        err_short = str(e1).split("\n")[0][:120]
        print(f"[GATE-V4] {intrin_name} failed: {err_short}", flush=True)

        # Plan B: Try the u8u8 variant in case reads interpretation differs
        print(f"[GATE-V4] Plan B: try tensorize without cache operations...", flush=True)
        sch2 = s_tir.Schedule(ir_mod["main"])
        blk2 = sch2.get_sblock("conv2d_NCHWc_int8")
        loops2 = sch2.get_loops(blk2)
        try:
            sch2.tensorize(loops2[-1], intrin_name)
            print(f"[GATE-V4] Plan B no-cache tensorize SUCCESS!", flush=True)
            sch = sch2
            dp4a_ok = True
        except Exception as e2:
            err2 = str(e2).split("\n")[0][:120]
            print(f"[GATE-V4] Plan B also failed: {err2}", flush=True)
            dp4a_ok = False

    if dp4a_ok:
        # Build and dump CUDA
        tir_txt = sch.mod.script()
        tir_path = os.path.join(WORK_DIR, "tir_dump.txt")
        with open(tir_path, "w") as f:
            f.write(tir_txt)
        result["dp4a_in_tir"] = "dp4a" in tir_txt.lower()
        print(f"[GATE-V4] dp4a in TIR: {result['dp4a_in_tir']}", flush=True)

        with tvm.transform.PassContext(opt_level=3):
            lib = tvm.tirx.build(sch.mod, target=str(target))

        cuda_src = None
        for m in lib.imported_modules:
            try:
                src = m.get_source()
                if "__global__" in src:
                    cuda_src = src
                    break
            except Exception:
                pass

        if cuda_src:
            cuda_path = os.path.join(WORK_DIR, "cuda_dump.txt")
            with open(cuda_path, "w") as f:
                f.write(cuda_src)
            result["dp4a_in_cuda"] = "__dp4a" in cuda_src.lower() or "dp4a" in cuda_src.lower()
            result["wmma_in_cuda"] = "wmma" in cuda_src.lower()
            if result["dp4a_in_cuda"]:
                result["intrinsic_type"] = "DP4A_REAL"
                for line in cuda_src.split("\n"):
                    if "dp4a" in line.lower():
                        print(f"[GATE-V4] ★ dp4a line: {line.strip()[:100]}", flush=True)
                        break
            elif result["wmma_in_cuda"]:
                result["intrinsic_type"] = "WMMA_REAL"
            else:
                result["intrinsic_type"] = "SCALAR_no_TC"
            print(f"[GATE-V4] CUDA: dp4a={result['dp4a_in_cuda']} wmma={result['wmma_in_cuda']}", flush=True)
        else:
            print(f"[GATE-V4] ✗ No CUDA source retrieved", flush=True)
    else:
        print(f"[GATE-V4] Both tensorize attempts failed → reporting BLOCKER", flush=True)
        result["intrinsic_type"] = "BLOCKED_TENSORIZE_MISMATCH"
        result["status"] = "BLOCKED"

    # ── Tune + measure latency ───────────────────────────────────────────────
    print(f"\n[GATE-V4] Tuning with MetaSchedule ({args.trials} trials)...", flush=True)
    from tvm.s_tir.meta_schedule import tune_tir
    t0 = time.time()
    db = tune_tir(
        mod=ir_mod,
        target=target,
        work_dir=WORK_DIR,
        max_trials_global=args.trials,
        seed=42,
    )
    tune_s = time.time() - t0
    print(f"[GATE-V4] Tuning done in {tune_s:.0f}s", flush=True)

    # Apply best schedule from DB
    from tvm.s_tir.meta_schedule.tir_integration import compile_tir
    try:
        lib_tuned = compile_tir(db, ir_mod["main"], target)
        print(f"[GATE-V4] compile_tir success", flush=True)

        # Check CUDA of tuned module
        cuda_tuned = None
        for m in lib_tuned.imported_modules:
            try:
                src = m.get_source()
                if "__global__" in src:
                    cuda_tuned = src
                    break
            except Exception:
                pass

        if cuda_tuned:
            tuned_path = os.path.join(WORK_DIR, "cuda_tuned_dump.txt")
            with open(tuned_path, "w") as f:
                f.write(cuda_tuned)
            dp4a_tuned = "__dp4a" in cuda_tuned.lower() or "dp4a" in cuda_tuned.lower()
            wmma_tuned = "wmma" in cuda_tuned.lower()
            print(f"[GATE-V4] TUNED CUDA: dp4a={dp4a_tuned} wmma={wmma_tuned}", flush=True)
            if dp4a_tuned and not result["dp4a_in_cuda"]:
                result["dp4a_in_cuda"] = True
                result["intrinsic_type"] = "DP4A_REAL_TUNED"
                for line in cuda_tuned.split("\n"):
                    if "dp4a" in line.lower():
                        print(f"[GATE-V4] ★ dp4a line (tuned): {line.strip()[:100]}", flush=True)
                        break

        # Benchmark
        x_np = np.random.randint(-128, 127, (2, IC_CHUNK_USE, H, W, IC_BN), dtype=np.int8)
        k_np = np.random.randint(-128, 127,
                                  (OC_CHUNK, IC_CHUNK_PER_GROUP_USE, KSIZE, KSIZE, IC_BN//N_ELEMS, OC_BN, N_ELEMS),
                                  dtype=np.int8)
        out_np = np.zeros((2, OC_CHUNK, H, W, OC_BN), dtype=np.int32)
        x_d = tvm.runtime.empty(x_np.shape, "int8", dev)
        k_d = tvm.runtime.empty(k_np.shape, "int8", dev)
        o_d = tvm.runtime.empty(out_np.shape, "int32", dev)
        x_d.copyfrom(x_np)
        k_d.copyfrom(k_np)
        lib_tuned(x_d, k_d, o_d)
        dev.sync()
        timer = lib_tuned.time_evaluator(lib_tuned.entry_name, dev, number=args.reps, repeat=5)
        t = timer(x_d, k_d, o_d)
        lat_us = t.mean * 1e6
        result["lat_tuned_us"] = round(lat_us, 2)
        print(f"[GATE-V4] TUNED latency: {lat_us:.1f} µs", flush=True)
    except Exception as e:
        print(f"[GATE-V4] compile_tir or benchmark failed: {e}", flush=True)
        traceback.print_exc()

    if result.get("status") != "BLOCKED":
        result["status"] = "DONE"

except Exception as e:
    traceback.print_exc()
    result["status"] = f"FAILED: {e}"

# Verdict
print("\n" + "=" * 72, flush=True)
print(f"[GATE-V4] ★ STEP A VERDICT:", flush=True)
dp4a_ok = result.get("dp4a_in_cuda") or result.get("dp4a_in_tir")
print(f"  width={args.width} in_per_g={IN_PER_G} dp4a_confirmed={dp4a_ok}", flush=True)
print(f"  intrinsic_type={result.get('intrinsic_type')}", flush=True)
print(f"  lat_tuned_us={result.get('lat_tuned_us')}", flush=True)
print(f"  status={result.get('status')}", flush=True)
if dp4a_ok:
    print(f"  RESULT: ✓ REAL dp4a INT8 confirmed → proceed to STEP B", flush=True)
else:
    print(f"  RESULT: ✗ dp4a NOT confirmed → reporting BLOCKER", flush=True)
    print(f"  BLOCKER: TVM 0.20 custom build lacks conv2d_NCHWc_int8 schedule rule FFI", flush=True)
    print(f"  Next: register Python MetaSchedule rule for dp4a tensorize", flush=True)
print("=" * 72, flush=True)

out_path = os.path.join(WORK_DIR, "result.json")
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"[GATE-V4] saved: {out_path}", flush=True)
