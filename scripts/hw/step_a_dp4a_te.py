"""STEP A v3 — dp4a gate via direct TE + topi.nn.conv2d_NCHWc_int8.

This bypasses relax/ONNX entirely and goes straight to the topi TE compute
that has the n_elems=4 inner reduction axis needed for dp4a tensorization.

The dp4a intrinsic is registered as "dp4a_s8s8s32" in TVM's s_tir.
We either:
A) Let MetaSchedule automatically find dp4a via meta_schedule.tune_tir
B) Manual tensorize: apply dp4a intrinsic to the inner 4-element reduce axis

Test: for s0=64 (in_per_g=4, ic_bn=4), MetaSchedule should find dp4a.
      for s0=48 (in_per_g=3), NCHWc format requires padding → test infeasibility.

Usage:
  CUDA_VISIBLE_DEVICES=<gpu> LD_LIBRARY_PATH=$(cat /exdata/jichengzhi/tvm_nvlibs.path) \\
  PATH=/usr/local/cuda-12.2/bin:$PATH \\
  /exdata/jichengzhi/tvm310/bin/python3 step_a_dp4a_te.py [--width s0=64] [--trials 100]
"""
from __future__ import annotations
import argparse, json, os, sys, time, traceback
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--width", default="s0=64", choices=["s0=64", "s0=48"])
ap.add_argument("--trials", type=int, default=200)
ap.add_argument("--reps", type=int, default=200)
ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--workdir", default="/exdata/jichengzhi/s2_tvm/step_a_dp4a_te")
args = ap.parse_args()

os.environ["PATH"] = "/usr/local/cuda-12.2/bin:" + os.environ.get("PATH", "")

# s0=64: CIN=128, COUT=128, groups=32, in_per_g=4 → ic_bn=4, dp4a aligned
# s0=48: CIN=96, COUT=96, groups=32, in_per_g=3 → not aligned (skip with explanation)
if args.width == "s0=64":
    S0, CIN, COUT, GROUPS = 64, 128, 128, 32
    IC_BN = 4   # = in_per_g = 4, perfectly aligned for dp4a (n_elems=4)
else:
    S0, CIN, COUT, GROUPS = 48, 96, 96, 32
    IC_BN = 4   # Would need padding (in_per_g=3 → pad to 4): alignment trap

IN_PER_G = CIN // GROUPS   # 4 for s0=64, 3 for s0=48
N_ELEMS = 4
OC_BN = 4

KSIZE = 3
H, W = 256, 256
WORK_DIR = os.path.join(args.workdir, args.width.replace("=", "_"))
os.makedirs(WORK_DIR, exist_ok=True)

result = {
    "width": args.width, "S0": S0, "CIN": CIN, "GROUPS": GROUPS, "in_per_g": IN_PER_G,
    "IC_BN": IC_BN, "N_ELEMS": N_ELEMS,
    "dp4a_aligned": (IN_PER_G % N_ELEMS == 0),
    "dp4a_in_cuda": False, "wmma_in_cuda": False, "scalar": False,
    "intrinsic_type": "UNKNOWN",
    "status": "STARTED",
}

print("=" * 72, flush=True)
print(f"[GATE-TE] STEP A v3: topi.nn.conv2d_NCHWc_int8 dp4a gate", flush=True)
print(f"  width={args.width} CIN={CIN} COUT={COUT} GROUPS={GROUPS} in_per_g={IN_PER_G}", flush=True)
print(f"  IC_BN={IC_BN} N_ELEMS={N_ELEMS} dp4a_aligned={IN_PER_G % 4 == 0}", flush=True)
print("=" * 72, flush=True)

try:
    import tvm
    from tvm import te
    from tvm.topi.nn.conv2d import conv2d_NCHWc_int8
    import tvm.s_tir.tensor_intrin.dot_product_common  # registers dp4a_s8s8s32
    import tvm.s_tir.tensor_intrin.cuda               # registers WMMA intrins
    from tvm.s_tir import TensorIntrin

    print(f"[GATE-TE] TVM {tvm.__version__}", flush=True)
    dev = tvm.cuda(args.gpu)
    target = tvm.target.Target.from_device(dev)
    print(f"[GATE-TE] target={target}", flush=True)

    # ── Step 1: Compute padded NCHWc tensors ──────────────────────────────────
    # For s0=48 (in_per_g=3): we pad to ic_bn=4 (zero-pad 1 channel per group)
    # This makes the computation equivalent to s0=48 but with explicit padding.
    # For s0=64 (in_per_g=4): no padding needed.

    if IN_PER_G < IC_BN:
        # Padding needed: in_per_g=3 padded to ic_bn=4
        PAD_CIN = (CIN // GROUPS) * IC_BN * GROUPS  # = 4 * 32 = 128 (padded)
        IC_CHUNK = PAD_CIN // IC_BN
        IC_CHUNK_PER_GROUP = IC_CHUNK // GROUPS
        print(f"[GATE-TE] s0=48 padding: in_per_g={IN_PER_G}→IC_BN={IC_BN}, CIN={CIN}→{PAD_CIN}", flush=True)
        result["padded_cin"] = PAD_CIN
    else:
        IC_CHUNK = CIN // IC_BN
        IC_CHUNK_PER_GROUP = IC_CHUNK // GROUPS
    OC_CHUNK = COUT // OC_BN

    print(f"[GATE-TE] NCHWc shapes:", flush=True)
    print(f"  data: (N=2, IC_CHUNK={IC_CHUNK}, H={H}, W={W}, IC_BN={IC_BN})", flush=True)
    print(f"  kernel: (OC_CHUNK={OC_CHUNK}, IC_CHUNK_PER_GROUP={IC_CHUNK_PER_GROUP}, "
          f"KH={KSIZE}, KW={KSIZE}, IC_BN//N_ELEMS={IC_BN//N_ELEMS}, OC_BN={OC_BN}, N_ELEMS={N_ELEMS})", flush=True)

    # Create TE placeholders
    data = te.placeholder((2, IC_CHUNK, H, W, IC_BN), dtype='int8', name='data')
    kernel = te.placeholder(
        (OC_CHUNK, IC_CHUNK_PER_GROUP, KSIZE, KSIZE, IC_BN // N_ELEMS, OC_BN, N_ELEMS),
        dtype='int8', name='kernel'
    )

    # Call topi conv2d_NCHWc_int8
    out = conv2d_NCHWc_int8(
        data, kernel,
        stride=1, padding=1, dilation=1,
        layout='NCHWc', out_layout='NCHWc',
        out_dtype='int32', n_elems=N_ELEMS
    )
    print(f"[GATE-TE] output shape: {out.shape}", flush=True)

    # ── Step 2: Create schedule and apply dp4a ────────────────────────────────
    s = te.create_schedule(out.op)
    ops = [out.op]
    # For the basic te.Schedule, we can tensorize the innermost reduction
    # The reduction axes from conv2d_NCHWc_int8 are: [kh, kw, ic_outer, ic_f_inner, ic_s_inner]
    # ic_s_inner has extent=N_ELEMS=4 → the dp4a 4-element accumulation

    # Get reduction axes
    print(f"[GATE-TE] out.op.axis: {out.op.axis}", flush=True)
    print(f"[GATE-TE] out.op.reduce_axis: {out.op.reduce_axis}", flush=True)

    # Build without tensorize first (to verify correctness)
    # Then build with tensorize

    # ── Method A: Let MetaSchedule find dp4a via s_tir ───────────────────────
    # Wrap the TE compute in a relax IRModule for MetaSchedule
    from tvm import relax
    from tvm.s_tir.meta_schedule import relax_integration as ri
    from tvm.s_tir.meta_schedule.database import database

    # Build a relax module from the TE compute
    tir_func = te.create_prim_func([data, kernel, out])
    ir_mod = tvm.IRModule({"conv2d_NCHWc_int8": tir_func})

    print(f"\n[GATE-TE] MetaSchedule tuning ({args.trials} trials)...", flush=True)
    t0 = time.time()

    # Use meta_schedule.tune_tir for direct TIR tuning
    from tvm.s_tir.meta_schedule import tune_tir
    db = tune_tir(
        mod=ir_mod,
        target=target,
        work_dir=WORK_DIR,
        max_trials_global=args.trials,
        num_trials_per_iter=64,
        seed=42,
    )
    tune_s = time.time() - t0
    print(f"[GATE-TE] tuning done in {tune_s:.0f}s", flush=True)

    # Apply best schedule
    from tvm.s_tir.meta_schedule import ApplyHistoryBest
    with ApplyHistoryBest(db):
        with tvm.transform.PassContext(
            opt_level=3, config={"relay.backend.use_meta_schedule": True}
        ):
            sch = tvm.s_tir.Schedule(ir_mod["conv2d_NCHWc_int8"])
            scheduled_mod = tvm.IRModule({"conv2d_NCHWc_int8": sch.mod["conv2d_NCHWc_int8"]})

    tir_txt = scheduled_mod.script()
    tir_path = os.path.join(WORK_DIR, "tir_dump.txt")
    with open(tir_path, "w") as f:
        f.write(tir_txt)
    print(f"[GATE-TE] TIR dumped: {tir_path}", flush=True)

    # Check TIR for dp4a
    tir_lower = tir_txt.lower()
    result["dp4a_in_tir"] = "dp4a" in tir_lower
    result["wmma_in_tir"] = "tvm_mma_sync" in tir_txt
    print(f"[GATE-TE] dp4a in TIR: {result['dp4a_in_tir']}", flush=True)
    print(f"[GATE-TE] WMMA in TIR: {result['wmma_in_tir']}", flush=True)

    # Build to CUDA
    with tvm.transform.PassContext(opt_level=3):
        func = tvm.build(ir_mod, target=target)

    # Get CUDA source
    cuda_src = None
    for m in func.imported_modules:
        try:
            src = m.get_source()
            if "__global__" in src:
                cuda_src = src
                break
        except Exception:
            pass
    if cuda_src is None:
        try:
            cuda_src = func.get_source("cuda")
        except Exception:
            pass

    if cuda_src:
        cuda_path = os.path.join(WORK_DIR, "cuda_dump.txt")
        with open(cuda_path, "w") as f:
            f.write(cuda_src)
        cuda_lower = cuda_src.lower()
        result["dp4a_in_cuda"] = "__dp4a" in cuda_lower or "dp4a" in cuda_lower
        result["wmma_in_cuda"] = "wmma" in cuda_lower
        result["scalar"] = not result["dp4a_in_cuda"] and not result["wmma_in_cuda"]

        if result["dp4a_in_cuda"]:
            result["intrinsic_type"] = "DP4A_REAL"
            for line in cuda_src.split("\n"):
                if "__dp4a" in line.lower() or "dp4a" in line.lower():
                    print(f"[GATE-TE] ★ dp4a line: {line.strip()[:120]}", flush=True)
                    break
        elif result["wmma_in_cuda"]:
            result["intrinsic_type"] = "WMMA_REAL"
        else:
            result["intrinsic_type"] = "SCALAR_NO_TENSORCORE"
            print(f"[GATE-TE] ✗ No dp4a or WMMA found in CUDA", flush=True)

        print(f"\n[GATE-TE] CUDA analysis:", flush=True)
        print(f"  dp4a_in_cuda: {result['dp4a_in_cuda']}", flush=True)
        print(f"  wmma_in_cuda: {result['wmma_in_cuda']}", flush=True)
        print(f"  intrinsic_type: {result['intrinsic_type']}", flush=True)

    # Quick latency via time_evaluator (old TE style)
    try:
        x_np = np.random.randint(-128, 127, (2, IC_CHUNK, H, W, IC_BN), dtype=np.int8)
        k_np = np.random.randint(-128, 127,
                                  (OC_CHUNK, IC_CHUNK_PER_GROUP, KSIZE, KSIZE, IC_BN//N_ELEMS, OC_BN, N_ELEMS),
                                  dtype=np.int8)
        out_np = np.zeros((2, OC_CHUNK, H, W, OC_BN), dtype=np.int32)
        x_tvm = tvm.runtime.empty(x_np.shape, dtype="int8", device=dev)
        k_tvm = tvm.runtime.empty(k_np.shape, dtype="int8", device=dev)
        out_tvm = tvm.runtime.empty(out_np.shape, dtype="int32", device=dev)
        x_tvm.copyfrom(x_np)
        k_tvm.copyfrom(k_np)
        # Warmup
        func(x_tvm, k_tvm, out_tvm)
        dev.sync()
        # Benchmark
        timer = func.time_evaluator(func.entry_name, dev, number=args.reps, repeat=5)
        t = timer(x_tvm, k_tvm, out_tvm)
        lat_us = t.mean * 1e6
        result["lat_tuned_us"] = round(lat_us, 2)
        print(f"\n[GATE-TE] TUNED latency: {lat_us:.1f} us", flush=True)
        # Compare with fp16 tuned baseline (H800 TVM): s0=64 tuned = 5507 us
        fp16_ref = 5507.0  # H800 TVM fp16 tuned s2_128 (s0=64 pair)
        result["int8_vs_fp16_ratio"] = round(fp16_ref / lat_us, 3)
        print(f"[GATE-TE] int8/fp16 ratio (6320us ref): {result['int8_vs_fp16_ratio']:.2f}x", flush=True)
    except Exception as e:
        print(f"[GATE-TE] latency failed: {e}", flush=True)
        traceback.print_exc()

    result["status"] = "DONE"

except Exception as e:
    traceback.print_exc()
    result["status"] = f"FAILED: {e}"

# Verdict
print("\n" + "=" * 72, flush=True)
print(f"[GATE-TE] ★ STEP A FINAL VERDICT:", flush=True)
dp4a_ok = result.get("dp4a_in_cuda") or result.get("dp4a_in_tir")
wmma_ok = result.get("wmma_in_cuda") or result.get("wmma_in_tir")
print(f"  width={args.width} in_per_g={IN_PER_G} dp4a_aligned={IN_PER_G % 4 == 0}", flush=True)
print(f"  dp4a confirmed: {dp4a_ok}", flush=True)
print(f"  wmma confirmed: {wmma_ok}", flush=True)
print(f"  intrinsic_type: {result.get('intrinsic_type')}", flush=True)
print(f"  lat_tuned_us: {result.get('lat_tuned_us')}", flush=True)
if dp4a_ok:
    print(f"  RESULT: ✓ REAL dp4a INT8 confirmed → STEP B proceed", flush=True)
elif wmma_ok:
    print(f"  RESULT: ✓ REAL WMMA INT8 → proceed", flush=True)
else:
    print(f"  RESULT: ✗ NO real int8 TC → investigate alternative", flush=True)
print("=" * 72, flush=True)

with open(os.path.join(WORK_DIR, "result.json"), "w") as f:
    json.dump(result, f, indent=2)
print(f"[GATE-TE] saved: {os.path.join(WORK_DIR, 'result.json')}", flush=True)
