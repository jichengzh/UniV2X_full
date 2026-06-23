"""S2.2 step-1 CONTROL: fp16 tensor-core engagement on aligned(p50) vs
misaligned(trap25) backbone.

Hypothesis (from MMA shape audit): fp16 MMA is 16x16x16, and BOTH p50
(32_64_128) and trap25 (48_96_192) channels are divisible by 16, so fp16
should engage tensor-core on BOTH with NO alignment cliff. This is the
control that isolates the cliff to INT8 (k_dim=32) -- handled separately.

Reports, per conv task: whether the best tuned schedule used a tensor-core
(mma/wmma) tensorize, plus the tuned module wall latency. TC engagement is
robust to foreign GPU load; absolute latency is provisional (noisy GPU).

Usage: python s2_2_fp16_couple.py <backbone.onnx> <work_dir> <label> [trials]
"""
from __future__ import annotations
import sys, os, traceback
import numpy as np

ONNX_PATH = sys.argv[1]
WORK_DIR = sys.argv[2]
LABEL = sys.argv[3] if len(sys.argv) > 3 else "model"
TRIALS = int(sys.argv[4]) if len(sys.argv) > 4 else 300

TC_KEYS = ("mma", "wmma", "tensor_core", "m16n16", "ldmatrix")


def read_inputs(m):
    init = {i.name for i in m.graph.initializer}
    return {i.name: tuple(d.dim_value for d in i.type.tensor_type.shape.dim)
            for i in m.graph.input if i.name not in init}


def trace_uses_tc(rec):
    try:
        s = str(rec.trace).lower()
    except Exception:
        return False
    return any(k in s for k in TC_KEYS)


def main():
    import onnx, tvm
    from tvm import relax
    from tvm.relax.frontend.onnx import from_onnx
    from tvm.s_tir.meta_schedule import relax_integration as ri
    print(f"[{LABEL}] TVM", tvm.__version__, "onnx:", ONNX_PATH, "trials:", TRIALS, flush=True)

    dev = tvm.cuda(0)
    target = tvm.target.Target.from_device(dev)
    m = onnx.load(ONNX_PATH)
    SHAPES = read_inputs(m)
    print(f"[{LABEL}] INPUTS", SHAPES, flush=True)
    mod = from_onnx(m, shape_dict=SHAPES, keep_params_in_input=False)

    # fp16 conversion (before legalize, at relax-op level)
    mod = relax.transform.ToMixedPrecision(out_dtype="float16")(mod)
    print(f"[{LABEL}] ToMixedPrecision OK", flush=True)

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
    ex = ri.compile_relax(db, mod, target, params={})
    vm = relax.VirtualMachine(ex, dev)

    # per-task TC engagement from best records
    recs = db.get_all_tuning_records()
    # group best (lowest run_secs) per workload
    from collections import defaultdict
    best = {}
    for r in recs:
        try:
            wl = r.workload
            key = id(wl) if wl is not None else 0
            rs = r.run_secs
            cost = float(min(x for x in rs)) if rs else float("inf")
        except Exception:
            key, cost = id(r), float("inf")
        if key not in best or cost < best[key][0]:
            best[key] = (cost, trace_uses_tc(r))
    n_tc = sum(1 for _, tc in best.values() if tc)
    print(f"[{LABEL}] workloads={len(best)} best_with_TC={n_tc} total_records={len(recs)}", flush=True)

    # run + time (provisional, noisy GPU)
    rng = np.random.RandomState(0)
    from tvm.runtime import tensor as _tensor
    feeds = {k: rng.rand(*v).astype("float32") for k, v in SHAPES.items()}
    args = [_tensor(feeds[k], device=dev) for k in SHAPES]
    out = vm["main"](*args)
    outs = [out] if hasattr(out, "shape") else list(out)
    print(f"[{LABEL}] RUN_OK n_out", len(outs), flush=True)
    # crude timing
    import time
    for _ in range(3):
        vm["main"](*args); dev.sync()
    N = 50
    t0 = time.time()
    for _ in range(N):
        vm["main"](*args)
    dev.sync()
    ms = (time.time() - t0) / N * 1000
    print(f"[{LABEL}] PROVISIONAL_LAT_ms {ms:.3f} (noisy GPU, not final)", flush=True)
    print(f"[{LABEL}] RESULT label={LABEL} tasks={len(tasks)} tc_workloads={n_tc}/{len(best)} lat_ms={ms:.3f}", flush=True)
    print(f"[{LABEL}] DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        traceback.print_exc(); print(f"[{LABEL}] EXCEPTION", repr(e)); sys.exit(3)
