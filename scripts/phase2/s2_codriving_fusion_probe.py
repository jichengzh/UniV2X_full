"""P3-(b) probe: can the CoDriving where2comm fusion (collab ONNX) import into TVM relax?
This is the 39%-of-pipeline headroom (the CoDriving 'neck' equivalent, analog of Pyramid's
attention neck which was import-blocked). Reports op inventory (esp. grid_sample/scatter),
attempts from_onnx import, and if it imports, builds + numerically aligns vs the PyTorch
reference npz.

Usage: python s2_codriving_fusion_probe.py <collab_onnx> [ref_npz] [target=llvm]
"""
from __future__ import annotations
import sys, traceback, collections
import numpy as np

ONNX = sys.argv[1]
REF = sys.argv[2] if len(sys.argv) > 2 else ""
TARGET = sys.argv[3] if len(sys.argv) > 3 else "llvm"
ATOL = 1e-3


def read_inputs(m):
    init = {i.name for i in m.graph.initializer}
    sh = {}
    for i in m.graph.input:
        if i.name in init:
            continue
        sh[i.name] = tuple(d.dim_value if d.dim_value > 0 else 2
                           for d in i.type.tensor_type.shape.dim)
    return sh


def main():
    import onnx, tvm
    from tvm import relax
    from tvm.relax.frontend.onnx import from_onnx
    m = onnx.load(ONNX)
    ops = collections.Counter(n.op_type for n in m.graph.node)
    print("TOTAL_NODES", sum(ops.values()))
    for k, v in ops.most_common():
        flag = "  <-- HARD" if k in ("GridSample", "ScatterND", "ScatterElements", "NonZero", "GatherND") else ""
        print(f"  {k:26s} {v}{flag}")
    SH = read_inputs(m)
    print("INPUTS", SH)
    print("OUTPUTS", [o.name for o in m.graph.output])

    print("[relax] from_onnx ...", flush=True)
    try:
        mod = from_onnx(m, shape_dict=SH, keep_params_in_input=False)
        print("[relax] IMPORT_OK", flush=True)
    except Exception:
        traceback.print_exc()
        print("VERDICT IMPORT_BLOCKED", flush=True)
        return

    dev = tvm.cuda(0) if TARGET == "cuda" else tvm.cpu(0)
    try:
        with tvm.transform.PassContext(opt_level=3):
            ex = relax.build(mod, target=TARGET)
        vm = relax.VirtualMachine(ex, dev)
        print("[relax] BUILD_OK target=", TARGET, flush=True)
    except Exception:
        traceback.print_exc()
        print("VERDICT BUILD_BLOCKED", flush=True)
        return

    rng = np.random.RandomState(0)
    feeds = {k: rng.rand(*v).astype("float32") for k, v in SH.items()}
    from tvm.runtime import tensor as _t
    out = vm["main"](*[_t(feeds[k], device=dev) for k in SH])
    outs = [out] if hasattr(out, "shape") else list(out)
    print("[relax] RUN_OK n_out", len(outs))

    if REF:
        ref = np.load(REF)
        refkeys = list(ref.keys())
        print("REF keys", refkeys)
        worst = 0.0
        for i, o in enumerate(outs):
            on = o.numpy() if hasattr(o, "numpy") else np.from_dlpack(o)
            if i < len(refkeys):
                r = ref[refkeys[i]]
                if tuple(on.shape) == tuple(r.shape):
                    md = float(np.max(np.abs(on - r)))
                    print(f"  out[{i}] shape{list(on.shape)} maxdiff={md:.3e}")
                    worst = max(worst, md)
                else:
                    print(f"  out[{i}] shape{list(on.shape)} vs ref{list(r.shape)} MISMATCH")
                    worst = float("inf")
        print(f"WORST_MAXDIFF {worst:.3e}")
        print("VERDICT", "PASS_FULL" if worst < ATOL else "FAIL_NUM")
    else:
        print("VERDICT IMPORT_BUILD_OK (no ref provided)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        traceback.print_exc(); print("VERDICT EXCEPTION", repr(e)); sys.exit(3)
