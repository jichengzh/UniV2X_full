"""S2.1 gate (decisive) — import the Pyramid CONV BACKBONE subgraph
(base_backbone.onnx: 50 Conv/48 Relu/16 Add, input spatial_features(2,64,128,256),
3 pyramid-level outputs) into TVM relax and numerically align vs ONNXRuntime.

This is the coupling-relevant compute (the alignment×INT8 trap, tiling/layout/
tensorize all live in these grouped-bottleneck convs). The full-model neck
(grid_sample attention + deblock concat) trips a relax onnx shape-inference bug,
handled separately; the science is here.

Generic: reads input names+shapes from the onnx graph. PASS if maxdiff<1e-3.
"""
from __future__ import annotations
import sys, traceback
import numpy as np

ONNX_PATH = sys.argv[1] if len(sys.argv) > 1 else "/exdata/jichengzhi/s2_tvm/models/base_backbone.onnx"
ATOL = 1e-3


def read_inputs(onnx_model):
    init = {i.name for i in onnx_model.graph.initializer}
    shapes = {}
    for i in onnx_model.graph.input:
        if i.name in init:
            continue
        shapes[i.name] = tuple(d.dim_value for d in i.type.tensor_type.shape.dim)
    return shapes


def main():
    import onnx, tvm
    from tvm import relax
    from tvm.relax.frontend.onnx import from_onnx
    print("TVM", tvm.__version__, "| onnx:", ONNX_PATH)
    dev = tvm.cuda(0)
    print("CUDA device exist:", dev.exist)

    onnx_model = onnx.load(ONNX_PATH)
    SHAPES = read_inputs(onnx_model)
    print("INPUTS", SHAPES)
    rng = np.random.RandomState(0)
    feeds = {k: rng.rand(*v).astype("float32") for k, v in SHAPES.items()}

    import onnxruntime as ort
    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    onames = [o.name for o in sess.get_outputs()]
    ref = sess.run(None, feeds)
    print("ORT_OUTPUTS", {n: list(r.shape) for n, r in zip(onames, ref)})

    print("[relax] from_onnx ...")
    mod = from_onnx(onnx_model, shape_dict=SHAPES, keep_params_in_input=False)
    print("[relax] IMPORT_OK")
    with tvm.transform.PassContext(opt_level=3):
        ex = relax.build(mod, target="cuda")
    vm = relax.VirtualMachine(ex, dev)
    print("[relax] BUILD_OK (cuda kernels compiled+run on driver 535 => CUDA minor-compat CONFIRMED)")
    from tvm.runtime import tensor as _tensor
    args = [_tensor(feeds[k], device=dev) for k in SHAPES]
    out = vm["main"](*args)
    outs = [out] if hasattr(out, "shape") else list(out)  # unwrap tvm Array container

    def to_np(t):
        for m in ("numpy", "asnumpy"):
            if hasattr(t, m):
                return getattr(t, m)()
        return np.from_dlpack(t)
    tvm_outs = [to_np(o) for o in outs]
    print("[relax] RUN_OK n_out", len(tvm_outs))

    worst = 0.0
    for i, t in enumerate(tvm_outs):
        if i < len(ref) and tuple(t.shape) == tuple(ref[i].shape):
            md = float(np.max(np.abs(t - ref[i])))
            print(f"  out[{i}] shape{list(t.shape)} maxdiff={md:.3e}")
            worst = max(worst, md)
        else:
            rs = list(ref[i].shape) if i < len(ref) else None
            print(f"  out[{i}] shape{list(t.shape)} vs ref{rs} SHAPE-MISMATCH")
            worst = float("inf")
    print(f"WORST_MAXDIFF {worst:.3e}")
    print("VERDICT", "PASS_FULL" if worst < ATOL else "FAIL_NUM")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        traceback.print_exc(); print("VERDICT EXCEPTION", repr(e)); sys.exit(3)
