"""S2.2 step-2 DEMO: the INT8 alignment cliff, discovered by the schedule search.

MMA_i8i8i32 is 16x16xK=32 -> the conv contraction dim (Cin*kh*kw) must be a
multiple of 32 to map onto INT8 tensor cores. We build a synthetic int8 1x1
conv (the bottleneck shape) and let MetaSchedule try to tensorize:

  aligned   Cin=64  (64 % 32 == 0)  -> expect i8 MMA engages
  misaligned Cin=48 (48 % 32 == 16) -> expect NO i8 MMA (TC tiling rejects K=48)
  mitigated  Cin=48 padded to 64    -> expect i8 MMA re-engages (FAST/ALT pad)

This isolates the cliff to the INT8 k_dim=32 constraint (fp16 16x16x16 does not
trip it -- see s2_2_fp16_couple.py control). It mirrors why TRT-auto must fall
back on misaligned widths, and shows TVM can co-configure layout/pad to recover.

Usage: python s2_2_int8_cliff.py <Cin> <Cout> <H> <W> <work_dir> <label> [trials] [pad_to]
"""
from __future__ import annotations
import sys, os, traceback
import numpy as np

CIN = int(sys.argv[1]); COUT = int(sys.argv[2]); H = int(sys.argv[3]); W = int(sys.argv[4])
WORK_DIR = sys.argv[5]; LABEL = sys.argv[6]
TRIALS = int(sys.argv[7]) if len(sys.argv) > 7 else 64
PAD_TO = int(sys.argv[8]) if len(sys.argv) > 8 else 0  # 0 = no pad

TC_KEYS = ("mma_i8", "mma_sync", "wmma", "tensor_core", "ldmatrix", "m16n16")


def build_int8_conv(cin, cout, h, w):
    """1x1 int8 conv as a relax module via BlockBuilder (no source introspection)."""
    import tvm
    from tvm import relax
    bb = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorStructInfo((2, cin, h, w), "int8"))
    wt = relax.Var("wt", relax.TensorStructInfo((cout, cin, 1, 1), "int8"))
    with bb.function("main", [x, wt]):
        with bb.dataflow():
            y = bb.emit(relax.op.nn.conv2d(x, wt, out_dtype="int32"))
            gv = bb.emit_output(y)
        bb.emit_func_output(gv)
    return bb.finalize()


def trace_uses_tc(rec):
    try:
        s = str(rec.trace).lower()
    except Exception:
        return False
    return any(k in s for k in TC_KEYS)


def main():
    import tvm
    from tvm import relax
    from tvm.s_tir.meta_schedule import relax_integration as ri
    cin_eff = PAD_TO if PAD_TO else CIN
    print(f"[{LABEL}] int8 conv Cin={CIN}(eff {cin_eff}) Cout={COUT} {H}x{W} k32_align={cin_eff%32==0} trials={TRIALS}", flush=True)

    dev = tvm.cuda(0)
    target = tvm.target.Target.from_device(dev)
    mod = build_int8_conv(cin_eff, COUT, H, W)
    print(f"[{LABEL}] relax module built", flush=True)

    seq = tvm.transform.Sequential([
        relax.transform.LegalizeOps(),
        relax.transform.AnnotateTIROpPattern(),
        relax.transform.FuseOps(),
        relax.transform.FuseTIR(),
    ])
    with target, tvm.transform.PassContext(opt_level=3):
        mod = seq(mod)
    tasks = ri.extract_tasks(mod, target, params={})
    print(f"[{LABEL}] extract_tasks -> {len(tasks)} tasks", flush=True)

    os.makedirs(WORK_DIR, exist_ok=True)
    db = ri.tune_relax(mod=mod, params={}, target=target, work_dir=WORK_DIR,
                       max_trials_global=TRIALS)
    print(f"[{LABEL}] tune COMPLETED", flush=True)
    recs = db.get_all_tuning_records()
    tc_hits = sum(1 for r in recs if trace_uses_tc(r))
    print(f"[{LABEL}] records={len(recs)} tc_traces={tc_hits}", flush=True)

    ex = ri.compile_relax(db, mod, target, params={})
    vm = relax.VirtualMachine(ex, dev)
    from tvm.runtime import tensor as _tensor
    rng = np.random.RandomState(0)
    x = _tensor(rng.randint(-8, 8, (2, cin_eff, H, W)).astype("int8"), device=dev)
    wt = _tensor(rng.randint(-8, 8, (COUT, cin_eff, 1, 1)).astype("int8"), device=dev)
    import time
    for _ in range(3):
        vm["main"](x, wt); dev.sync()
    N = 100
    t0 = time.time()
    for _ in range(N):
        vm["main"](x, wt)
    dev.sync()
    us = (time.time() - t0) / N * 1e6
    print(f"[{LABEL}] PROVISIONAL_LAT_us {us:.2f} (noisy GPU)", flush=True)
    print(f"[{LABEL}] RESULT label={LABEL} Cin={CIN} eff={cin_eff} k32={cin_eff%32==0} tc_traces={tc_hits} lat_us={us:.2f}", flush=True)
    print(f"[{LABEL}] DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        traceback.print_exc(); print(f"[{LABEL}] EXCEPTION", repr(e)); sys.exit(3)
