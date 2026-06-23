"""Build TRT FP16 and INT8 engines for CoDriving p50 collab ONNX.

ONNX: codriving_collab_p50_fp32.onnx
  Inputs:  spatial_features (2,64,256,512), pairwise_t_matrix (1,2,2,4,4)
  Outputs: cls_preds (1,1,128,256), reg_preds (1,8,128,256)

INT8 calibration: reuses base collab calibration data (same input shapes).

Usage:
    CUDA_VISIBLE_DEVICES=3 python scripts/phase1/build_p50_collab_engines.py --mode fp16
    CUDA_VISIBLE_DEVICES=3 python scripts/phase1/build_p50_collab_engines.py --mode int8
    CUDA_VISIBLE_DEVICES=3 python scripts/phase1/build_p50_collab_engines.py --mode both
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import tensorrt as trt

REPO_ROOT = Path(__file__).resolve().parents[2]

# Paths
ONNX_PATH = REPO_ROOT / "output/codriving_pilot/collab_export/codriving_collab_p50_fp32.onnx"
ENGINE_DIR = REPO_ROOT / "output/codriving_pilot/collab_engines"
ENGINE_FP16 = ENGINE_DIR / "codriving_collab_p50_fp16.engine"
ENGINE_INT8 = ENGINE_DIR / "codriving_collab_p50_int8.engine"
# Reuse base calibration data (same input shapes)
CALIB_DATA = ENGINE_DIR / "collab_calib_data.npz"
CALIB_CACHE = ENGINE_DIR / "collab_p50_int8_calib_cache.bin"
LOG_DIR = ENGINE_DIR / "logs"

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

SPATIAL_SHAPE = (2, 64, 256, 512)
TMAT_SHAPE = (1, 2, 2, 4, 4)


class CollabInt8Calibrator(trt.IInt8MinMaxCalibrator):
    """Calibrator feeding (spatial_features, pairwise_t_matrix) pairs."""

    def __init__(self, calib_data_path: str, cache_file: str = None):
        super().__init__()
        self.cache_file = cache_file
        data = np.load(calib_data_path)
        self.spatial_data = data["spatial_features"]   # (N, 2, 64, 256, 512)
        self.tmat_data = data["pairwise_t_matrix"]     # (N, 1, 2, 2, 4, 4)
        self.n_batches = len(self.spatial_data)
        self.current_idx = 0
        print(f"  [calib] {self.n_batches} samples from {calib_data_path}")

        self._spatial_buf = torch.zeros(SPATIAL_SHAPE, dtype=torch.float32,
                                        device="cuda").contiguous()
        self._tmat_buf = torch.zeros(TMAT_SHAPE, dtype=torch.float32,
                                     device="cuda").contiguous()

    def get_batch_size(self):
        return 1

    def get_batch(self, names):
        if self.current_idx >= self.n_batches:
            return None
        if self.current_idx % 25 == 0:
            print(f"  [calib] batch {self.current_idx+1}/{self.n_batches}")
        spatial = torch.from_numpy(self.spatial_data[self.current_idx]).float()
        tmat = torch.from_numpy(self.tmat_data[self.current_idx]).float()
        self._spatial_buf.copy_(spatial)
        self._tmat_buf.copy_(tmat)
        self.current_idx += 1
        return [int(self._spatial_buf.data_ptr()), int(self._tmat_buf.data_ptr())]

    def read_calibration_cache(self):
        if self.cache_file and Path(self.cache_file).exists():
            print(f"  [calib] reading cache: {self.cache_file}")
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        if self.cache_file:
            with open(self.cache_file, "wb") as f:
                f.write(cache)
            print(f"  [calib] wrote cache: {self.cache_file}")


def build_engine(mode: str, workspace_mb: int = 4096):
    """Build TRT engine. mode: 'fp16' or 'int8'."""
    assert mode in ("fp16", "int8")
    output_path = ENGINE_FP16 if mode == "fp16" else ENGINE_INT8
    ENGINE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n[TRT BUILD] mode={mode.upper()} -> {output_path}")
    t0 = time.time()

    builder = trt.Builder(TRT_LOGGER)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(str(ONNX_PATH), "rb") as f:
        onnx_bytes = f.read()
    if not parser.parse(onnx_bytes):
        for i in range(parser.num_errors):
            print(f"  ONNX parse error: {parser.get_error(i).desc()}")
        raise RuntimeError("ONNX parse failed")
    print(f"  ONNX parsed: {ONNX_PATH} ({len(onnx_bytes)/1e6:.1f} MB)")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb * 1024 * 1024)

    if mode == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
        print("  FP16 flag set")
    elif mode == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16)
        assert CALIB_DATA.exists(), f"Calibration data not found: {CALIB_DATA}"
        calibrator = CollabInt8Calibrator(
            calib_data_path=str(CALIB_DATA),
            cache_file=str(CALIB_CACHE),
        )
        config.int8_calibrator = calibrator
        print("  INT8 flag set + calibrator attached")

    print(f"  Building engine (this may take 1-5 minutes)...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TRT engine build failed (serialized is None)")

    with open(str(output_path), "wb") as f:
        f.write(serialized)

    elapsed = time.time() - t0
    size_mb = output_path.stat().st_size / 1e6
    print(f"  Engine saved: {output_path}  ({size_mb:.1f} MB, {elapsed:.0f}s)")
    return str(output_path)


def quick_sanity(engine_path: str):
    """Load engine and run one forward pass to verify output shapes."""
    print(f"\n[sanity] Loading engine: {engine_path}")
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    spatial = torch.randn(SPATIAL_SHAPE, device="cuda", dtype=torch.float32)
    t_mat = torch.zeros(TMAT_SHAPE, device="cuda", dtype=torch.float32)
    t_mat[0, :, :, 0, 0] = 1.0; t_mat[0, :, :, 1, 1] = 1.0
    t_mat[0, :, :, 2, 2] = 1.0; t_mat[0, :, :, 3, 3] = 1.0

    cls_out = torch.zeros(1, 1, 128, 256, device="cuda", dtype=torch.float32)
    reg_out = torch.zeros(1, 8, 128, 256, device="cuda", dtype=torch.float32)

    context.set_tensor_address("spatial_features", spatial.data_ptr())
    context.set_tensor_address("pairwise_t_matrix", t_mat.data_ptr())
    context.set_tensor_address("cls_preds", cls_out.data_ptr())
    context.set_tensor_address("reg_preds", reg_out.data_ptr())

    stream = torch.cuda.current_stream().cuda_stream
    ok = context.execute_async_v3(stream)
    torch.cuda.synchronize()

    print(f"  execute_async_v3 returned: {ok}")
    print(f"  cls_preds: {tuple(cls_out.shape)}, reg_preds: {tuple(reg_out.shape)}")
    print(f"  cls non-zero: {(cls_out != 0).sum().item()}, reg non-zero: {(reg_out != 0).sum().item()}")
    return ok


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["fp16", "int8", "both"], default="both")
    p.add_argument("--workspace_mb", type=int, default=4096)
    p.add_argument("--skip_sanity", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    modes = ["fp16", "int8"] if args.mode == "both" else [args.mode]

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"TRT: {trt.__version__}")
    print(f"ONNX: {ONNX_PATH}  exists={ONNX_PATH.exists()}")

    for mode in modes:
        engine_path = build_engine(mode, workspace_mb=args.workspace_mb)
        if not args.skip_sanity:
            quick_sanity(engine_path)

    print("\n=== BUILD COMPLETE ===")
    for mode in modes:
        epath = ENGINE_FP16 if mode == "fp16" else ENGINE_INT8
        if epath.exists():
            print(f"  {mode.upper()}: {epath}  ({epath.stat().st_size/1e6:.1f} MB)")
