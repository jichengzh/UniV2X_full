"""S2.1 gate (CoDriving) — import a CoDriving subgraph ONNX into TVM relax and
numerically align vs ONNXRuntime. Mirrors the Pyramid s2_1_backbone_align.py but
handles dynamic batch (CoDriving ONNX exports batch as a symbolic dim => dim_value=0):
any 0-extent dim is replaced with the CLI batch (default 2 = 2-agent collab).

Usage: python s2_1_codriving_align.py <onnx> [batch=2]
PASS if maxdiff<1e-3.
"""
from __future__ import annotations
import sys, traceback
import numpy as np

ONNX_PATH = sys.argv[1] if len(sys.argv) > 1 else "/exdata/jichengzhi/V2Xverse_pyramid/output/codriving_pilot/base/codriving_core_dair_fp32.onnx"
BATCH = int(sys.argv[2]) if len(sys.argv) > 2 else 2
TARGET = sys.argv[3] if len(sys.argv) > 3 else "llvm"  # llvm CPU = robust numerical gate (GPU dlight trips on 384->1 head)
ATOL = 1e-3


def read_inputs(onnx_model):
    init = {i.name for i in onnx_model.graph.initializer}
    shapes = {}
    for i in onnx_model.graph.input:
        if i.name in init:
            continue
        dims = []
        for d in i.type.tensor_type.shape.dim:
            v = d.dim_value
            dims.append(v if v > 0 else BATCH)  # symbolic/0 batch -> BATCH
        shapes[i.name] = tuple(dims)
    return shapes


def main():
    import onnx, tvm
    from tvm import relax
    from tvm.relax.frontend.onnx import from_onnx
    print("TVM", tvm.__version__, "| onnx:", ONNX_PATH, "| batch:", BATCH, "| target:", TARGET)
    dev = tvm.cuda(0) if TARGET == "cuda" else tvm.cpu(0)

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
        ex = relax.build(mod, target=TARGET)
    vm = relax.VirtualMachine(ex, dev)
    print("[relax] BUILD_OK target=", TARGET)
    from tvm.runtime import tensor as _tensor
    args = [_tensor(feeds[k], device=dev) for k in SHAPES]
    out = vm["main"](*args)
    outs = [out] if hasattr(out, "shape") else list(out)

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
            print(f"  out[{i}] {onames[i] if i < len(onames) else ''} shape{list(t.shape)} maxdiff={md:.3e}")
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
