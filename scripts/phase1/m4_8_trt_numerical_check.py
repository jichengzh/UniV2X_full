"""Phase A.3 — Numerical comparison: PyTorch vs TRT (FP16 + INT8) sub-module.

Forwards the *same* set of OPV2V calibration features through:
  (a) PyTorch FP32 sub-module (truth)
  (b) TRT FP16 engine
  (c) TRT INT8 engine

Reports max absolute / relative output drift to verify the engines are
numerically faithful to the ckpt — separating "is the engine correct?"
from "does the post-process / multi-agent path work?".
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.export_onnx_pyramid import build_pyramid_from_ckpt, PyramidSubnet  # noqa: E402

CKPT = "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_m1_base_2023_08_14_04_28_12/net_epoch_bestval_at23.pth"
HYPES = "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_m1_base_2023_08_14_04_28_12/config.yaml"
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def load_engine(path):
    runtime = trt.Runtime(TRT_LOGGER)
    with open(path, "rb") as f:
        return runtime.deserialize_cuda_engine(f.read())


def trt_run(engine, x: torch.Tensor):
    ctx = engine.create_execution_context()
    in_name = next(engine.get_tensor_name(i) for i in range(engine.num_io_tensors)
                   if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT)
    out_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)
                 if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT]
    ctx.set_input_shape(in_name, tuple(x.shape))
    bufs = {in_name: x.contiguous()}
    for n in [in_name, *out_names]:
        if n != in_name:
            shape = tuple(ctx.get_tensor_shape(n))
            bufs[n] = torch.empty(shape, dtype=torch.float32, device="cuda")
        ctx.set_tensor_address(n, int(bufs[n].data_ptr()))
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        ctx.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    return tuple(bufs[n] for n in out_names)


def diff_table(name_a, name_b, A, B):
    assert A.shape == B.shape
    A = A.cpu().numpy()
    B = B.cpu().numpy()
    d = np.abs(A - B)
    rel = d / max(np.abs(A).max(), 1e-6)
    print(f"  vs {name_b}:  max|Δ|={d.max():.4f}  mean|Δ|={d.mean():.5f}  "
          f"rel_max={rel.max():.2%}  |a|max={np.abs(A).max():.3f}")


def main():
    print("[1] build PyTorch reference")
    full = build_pyramid_from_ckpt(HYPES, CKPT, device="cuda")
    pt = PyramidSubnet(full).cuda().eval()

    print("[2] load TRT engines")
    fp16 = load_engine(str(REPO_ROOT / "models/pyramid_m1_subnet_fp16.engine"))
    int8 = load_engine(str(REPO_ROOT / "models/pyramid_m1_subnet_int8.engine"))
    int8e_path = REPO_ROOT / "models/pyramid_m1_subnet_int8_entropy.engine"
    int8e = load_engine(str(int8e_path)) if int8e_path.exists() else None

    print("[3] load 5 calib samples for input")
    arr = np.load(REPO_ROOT / "calibration/pyramid_calib.npy")[:5]
    print(f"  shape={arr.shape}  range=[{arr.min():.3f}, {arr.max():.3f}]")

    for k in range(5):
        x = torch.from_numpy(arr[k:k+1]).cuda()
        with torch.inference_mode():
            cls_pt, reg_pt, dir_pt = pt(x)
        cls_16, reg_16, dir_16 = trt_run(fp16, x)
        cls_8, reg_8, dir_8 = trt_run(int8, x)
        if int8e is not None:
            cls_e, reg_e, dir_e = trt_run(int8e, x)

        print(f"\n=== sample {k} (|x|max={x.abs().max():.3f}) ===")
        print(" cls  TRT_FP16:");      diff_table("pt", "fp16", cls_pt, cls_16)
        print(" cls  TRT_INT8_minmax:"); diff_table("pt", "int8mm", cls_pt, cls_8)
        if int8e is not None:
            print(" cls  TRT_INT8_entropy:"); diff_table("pt", "int8en", cls_pt, cls_e)
        print(" reg  TRT_FP16:");      diff_table("pt", "fp16", reg_pt, reg_16)
        print(" reg  TRT_INT8_minmax:"); diff_table("pt", "int8mm", reg_pt, reg_8)
        if int8e is not None:
            print(" reg  TRT_INT8_entropy:"); diff_table("pt", "int8en", reg_pt, reg_e)


if __name__ == "__main__":
    main()
