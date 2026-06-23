"""S2.2 toolchain go/no-go probe (single-agent gate, before any fan-out).

Answers empirically, on the Pyramid conv backbone subgraph:
  Q1. Does TVM 0.20 relax + MetaSchedule run end-to-end?
      (relax_integration.tune_relax + compile_relax, tiny trial budget, fp16)
  Q2. Can we express INT8 conv + tensorize (MMA) in relax 0.20?
      (relay/QNN removed -> this is THE risk. We probe a single int8 conv
       through relax and check it builds + a tensor-core intrin is selectable.)
  Q3. After a tiny tune, is a tensor-core (wmma/mma) schedule actually picked
      for the ALIGNED backbone vs the MISALIGNED one? (first coupling signal)

This is a TOOLCHAIN validation, not a latency benchmark -- foreign GPU load is
acceptable here; final latency numbers wait for a clean-GPU window.

Usage: python s2_2_probe.py <backbone.onnx> <work_dir> [max_trials]
"""
from __future__ import annotations
import sys, os, traceback, json
import numpy as np

ONNX_PATH = sys.argv[1] if len(sys.argv) > 1 else "/exdata/jichengzhi/s2_tvm/models/p50_backbone.onnx"
WORK_DIR = sys.argv[2] if len(sys.argv) > 2 else "/exdata/jichengzhi/s2_tvm/ms_work/probe"
MAX_TRIALS = int(sys.argv[3]) if len(sys.argv) > 3 else 32
ARCH = "sm_90"  # H800 = Hopper


def read_inputs(onnx_model):
    init = {i.name for i in onnx_model.graph.initializer}
    return {i.name: tuple(d.dim_value for d in i.type.tensor_type.shape.dim)
            for i in onnx_model.graph.input if i.name not in init}


def banner(q):
    print(f"\n========== {q} ==========", flush=True)


def main():
    import tvm
    from tvm import relax
    from tvm.relax.frontend.onnx import from_onnx
    print("TVM", tvm.__version__, "| onnx:", ONNX_PATH, "| arch:", ARCH, "| trials:", MAX_TRIALS, flush=True)
    dev = tvm.cuda(0)
    print("CUDA device exist:", dev.exist, flush=True)
    # from_device populates max_threads_per_block / shared mem / arch (MetaSchedule needs these)
    target = tvm.target.Target.from_device(dev)
    print("target:", target, flush=True)

    import onnx
    onnx_model = onnx.load(ONNX_PATH)
    SHAPES = read_inputs(onnx_model)
    print("INPUTS", SHAPES, flush=True)
    mod = from_onnx(onnx_model, shape_dict=SHAPES, keep_params_in_input=False)
    print("[relax] IMPORT_OK", flush=True)

    # ---- Q1: MetaSchedule API surface (relocated to tvm.s_tir.meta_schedule) ----
    banner("Q1 MetaSchedule API")
    from tvm.s_tir.meta_schedule import relax_integration as ri
    print("relax_integration has tune_relax:", hasattr(ri, "tune_relax"), flush=True)
    print("relax_integration has compile_relax:", hasattr(ri, "compile_relax"), flush=True)

    # ---- Q2: INT8 conv + tensorize expressibility ----
    banner("Q2 INT8 conv + tensorize in relax 0.20")
    try:
        from tvm.s_tir.tensor_intrin import cuda as tc_cuda
        tc = [x for x in dir(tc_cuda) if x.endswith("_INTRIN")]
        i8 = [x for x in tc if "i8" in x or "INT8" in x.upper()]
        print("INT8 tensor-core intrins:", i8, flush=True)
        print("VERDICT_Q2", "INT8_TENSORIZE_OK" if i8 else "INT8_TENSORIZE_MISSING", flush=True)
    except Exception as e:
        print("Q2 probe error:", repr(e), flush=True)

    # ---- Q3: tiny tune, inspect chosen schedule ----
    banner("Q3 tiny tune + schedule inspect (fp32-domain smoke)")
    os.makedirs(WORK_DIR, exist_ok=True)
    # lower relax ops -> TIR PrimFuncs so MetaSchedule can extract conv tasks
    seq = tvm.transform.Sequential([
        relax.transform.LegalizeOps(),
        relax.transform.AnnotateTIROpPattern(),
        relax.transform.FuseOps(),
        relax.transform.FuseTIR(),
    ])
    with target, tvm.transform.PassContext(opt_level=3):
        mod = seq(mod)
    try:
        tasks = ri.extract_tasks(mod, target, params={})
        print(f"[ms] extract_tasks -> {len(tasks)} tasks", flush=True)
    except Exception as e:
        print("[ms] extract_tasks err:", repr(e)[:120], flush=True)
    try:
        db = ri.tune_relax(
            mod=mod,
            params={},
            target=target,
            work_dir=WORK_DIR,
            max_trials_global=MAX_TRIALS,
        )
        print("[ms] tune_relax COMPLETED, db:", type(db).__name__, flush=True)
        ex = ri.compile_relax(db, mod, target, params={})
        print("[ms] compile_relax COMPLETED", flush=True)
        vm = relax.VirtualMachine(ex, dev)
        rng = np.random.RandomState(0)
        from tvm.runtime import tensor as _tensor
        feeds = {k: rng.rand(*v).astype("float32") for k, v in SHAPES.items()}
        args = [_tensor(feeds[k], device=dev) for k in SHAPES]
        out = vm["main"](*args)
        outs = [out] if hasattr(out, "shape") else list(out)
        print("[ms] RUN_OK n_out", len(outs), flush=True)
        # scan tuning records for tensor-core usage
        recs = db.get_all_tuning_records()
        tc_hits = 0
        for r in recs:
            try:
                trace_str = str(r.trace)
            except Exception:
                trace_str = ""
            if any(k in trace_str for k in ("wmma", "mma_", "tensor_core", "WMMA")):
                tc_hits += 1
        print(f"[ms] tuning_records={len(recs)} tensor_core_traces={tc_hits}", flush=True)
        print("VERDICT_Q3", "TUNE_OK_TC" if tc_hits > 0 else "TUNE_OK_NOTC", flush=True)
    except Exception as e:
        traceback.print_exc()
        print("VERDICT_Q3 TUNE_FAIL", repr(e), flush=True)

    print("\nPROBE_DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        traceback.print_exc(); print("PROBE_EXCEPTION", repr(e)); sys.exit(3)
