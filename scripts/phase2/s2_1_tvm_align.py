"""S2.1 gate (relax-only; TVM 0.20 Unity dropped relay) — import Pyramid
backbone (base_ctfix.onnx: ConvTranspose output_shape patched to fix relax's
deblock shape-inference) and numerically align vs ONNXRuntime (fp32).

base.onnx has GridSample/IsNaN/Equal/Einsum/Softmax (pyramid attention warp) +
3-scale ConvTranspose deblock concat. relax ingested everything except the
ConvTranspose shape; the output_shape patch addresses that.

Verdict: PASS_FULL if maxdiff<1e-3 on all outputs vs ORT.
"""
from __future__ import annotations
import sys, traceback
import numpy as np

ONNX_PATH = sys.argv[1] if len(sys.argv) > 1 else "/exdata/jichengzhi/s2_tvm/models/base_ctfix.onnx"
SHAPES = {"spatial_features": (2, 64, 128, 256), "t_ego": (2, 2, 3)}
ATOL = 1e-3


def ort_reference(feeds):
    import onnxruntime as ort
    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    names = [o.name for o in sess.get_outputs()]
    return dict(zip(names, sess.run(None, feeds)))


def main():
    import onnx, tvm
    from tvm import relax
    from tvm.relax.frontend.onnx import from_onnx
    print("TVM", tvm.__version__, "| onnx:", ONNX_PATH)
    dev = tvm.cuda(0)
    print("CUDA device exist:", dev.exist)
    rng = np.random.RandomState(0)
    feeds = {k: rng.rand(*v).astype("float32") for k, v in SHAPES.items()}
    ref = ort_reference(feeds)
    print("ORT_OUTPUTS", {k: list(v.shape) for k, v in ref.items()})

    onnx_model = onnx.load(ONNX_PATH)
    print("[relax] from_onnx ...")
    mod = from_onnx(onnx_model, shape_dict=SHAPES, keep_params_in_input=False)
    print("[relax] IMPORT_OK")
    with tvm.transform.PassContext(opt_level=3):
        ex = relax.build(mod, target="cuda")
    vm = relax.VirtualMachine(ex, dev)
    print("[relax] BUILD_OK (cuda kernels compiled+loaded on driver 535 => CUDA confirmed)")
    args = [tvm.nd.array(feeds[k], dev) for k in SHAPES]
    out = vm["main"](*args)
    outs = out if isinstance(out, (list, tuple)) else [out]
    tvm_outs = [o.numpy() for o in outs]
    print("[relax] RUN_OK n_out", len(tvm_outs), "shapes", [list(o.shape) for o in tvm_outs])

    ref_list = list(ref.values())
    worst = 0.0
    for i, t in enumerate(tvm_outs):
        if i < len(ref_list) and tuple(t.shape) == tuple(ref_list[i].shape):
            md = float(np.max(np.abs(t - ref_list[i])))
            print(f"  out[{i}] shape{list(t.shape)} maxdiff={md:.3e}")
            worst = max(worst, md)
        else:
            rs = list(ref_list[i].shape) if i < len(ref_list) else None
            print(f"  out[{i}] shape{list(t.shape)} vs ref{rs} SHAPE-MISMATCH")
            worst = float("inf")
    print(f"WORST_MAXDIFF {worst:.3e}")
    print("VERDICT", "PASS_FULL" if worst < ATOL else "FAIL_NUM")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        traceback.print_exc(); print("VERDICT EXCEPTION", repr(e)); sys.exit(3)
