"""Build TRT FP16 and INT8 engines for CoDriving collab ONNX.

ONNX: codriving_collab_base_fp32.onnx
  Inputs:  spatial_features (2,64,256,512), pairwise_t_matrix (1,2,2,4,4)
  Outputs: cls_preds (1,1,128,256), reg_preds (1,8,128,256)

INT8 calibration: collect real (spatial_features, pairwise_t_matrix) pairs
from DAIR val dataset via V2Xverse loader.

Usage:
    CUDA_VISIBLE_DEVICES=7 python scripts/phase1/build_codriving_collab_engines.py \
        --mode fp16    # build FP16 engine
    CUDA_VISIBLE_DEVICES=7 python scripts/phase1/build_codriving_collab_engines.py \
        --mode int8    # build INT8 engine (requires calibration data collection)
    CUDA_VISIBLE_DEVICES=7 python scripts/phase1/build_codriving_collab_engines.py \
        --mode both    # build both
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import re
import yaml
from pathlib import Path

import numpy as np
import torch
import tensorrt as trt

REPO_ROOT = Path(__file__).resolve().parents[2]
V2XVERSE_ROOT = Path("/home/jichengzhi/V2Xverse")

# Paths
ONNX_PATH = REPO_ROOT / "output/codriving_pilot/collab_export/codriving_collab_base_fp32.onnx"
CKPT_PATH = REPO_ROOT / "output/codriving_pilot/collab_export/net_epoch_bestval_at11.pth"
HYPES_PATH = REPO_ROOT / "output/codriving_pilot/collab_export/dair_centerpoint_codriving_4090.yaml"
ENGINE_DIR = REPO_ROOT / "output/codriving_pilot/collab_engines"
ENGINE_FP16 = ENGINE_DIR / "codriving_collab_base_fp16.engine"
ENGINE_INT8 = ENGINE_DIR / "codriving_collab_base_int8.engine"
CALIB_CACHE = ENGINE_DIR / "collab_int8_calib_cache.bin"
CALIB_DATA = ENGINE_DIR / "collab_calib_data.npz"
LOG_DIR = ENGINE_DIR / "logs"

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

SPATIAL_SHAPE = (2, 64, 256, 512)
TMAT_SHAPE = (1, 2, 2, 4, 4)


# ---------------------------------------------------------------------------
# INT8 Calibrator
# ---------------------------------------------------------------------------

class CollabInt8Calibrator(trt.IInt8MinMaxCalibrator):
    """Calibrator feeding (spatial_features, pairwise_t_matrix) pairs."""

    def __init__(self, calib_data_path: str, batch_size: int = 1,
                 cache_file: str = None):
        super().__init__()
        self.cache_file = cache_file
        data = np.load(calib_data_path)
        self.spatial_data = data["spatial_features"]   # (N, 2, 64, 256, 512)
        self.tmat_data = data["pairwise_t_matrix"]     # (N, 1, 2, 2, 4, 4)
        self.n_batches = len(self.spatial_data)
        self.current_idx = 0
        print(f"  [calib] {self.n_batches} calibration samples loaded from {calib_data_path}")

        # Pre-allocate GPU buffers
        import ctypes
        self._spatial_buf = torch.zeros(SPATIAL_SHAPE, dtype=torch.float32, device="cuda").contiguous()
        self._tmat_buf = torch.zeros(TMAT_SHAPE, dtype=torch.float32, device="cuda").contiguous()

    def get_batch_size(self):
        return 1

    def get_batch(self, names):
        if self.current_idx >= self.n_batches:
            return None
        print(f"  [calib] batch {self.current_idx+1}/{self.n_batches}")
        spatial = torch.from_numpy(self.spatial_data[self.current_idx]).float()
        tmat = torch.from_numpy(self.tmat_data[self.current_idx]).float()
        self._spatial_buf.copy_(spatial)
        self._tmat_buf.copy_(tmat)
        self.current_idx += 1
        # Return pointers in input name order
        # TRT calls with names in ONNX input order: spatial_features, pairwise_t_matrix
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


# ---------------------------------------------------------------------------
# Collect calibration data from DAIR val dataset
# ---------------------------------------------------------------------------

def collect_calib_data(n_samples: int = 150, output_path: str = None):
    """Collect (spatial_features, pairwise_t_matrix) from DAIR val.

    Uses V2Xverse's DAIR dataset loader with the 4090 config.
    """
    if output_path and Path(output_path).exists():
        print(f"  [calib-collect] data already exists: {output_path}")
        return

    print(f"  [calib-collect] collecting {n_samples} calibration samples from DAIR val...")
    sys.path.insert(0, str(V2XVERSE_ROOT))
    os.chdir(V2XVERSE_ROOT)

    from opencood.tools import train_utils
    from opencood.data_utils.datasets import build_dataset
    from torch.utils.data import DataLoader

    # Load hypes with yaml.Loader to handle numpy tags
    loader_cls = yaml.Loader
    loader_cls.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
         [-+]?(?:[0-9][0-9_]*)\\.([0-9_]*)(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))

    with open(str(HYPES_PATH), "r") as f:
        hypes = yaml.load(f, Loader=loader_cls)

    hypes["validate_dir"] = hypes["test_dir"]  # use val split

    model = train_utils.create_model(hypes)
    _, model = train_utils.load_saved_model(
        str(REPO_ROOT / "output/codriving_pilot/collab_export"), model)
    model = model.cuda().eval()

    dataset = build_dataset(hypes, visualize=False, train=False)
    loader = DataLoader(dataset, batch_size=1, num_workers=2,
                        collate_fn=dataset.collate_batch_test, shuffle=False)

    spatial_list = []
    tmat_list = []
    n_collected = 0

    with torch.no_grad():
        for batch_data in loader:
            if batch_data is None:
                continue
            if n_collected >= n_samples:
                break
            batch_data = train_utils.to_device(batch_data, "cuda")
            ego = batch_data["ego"]
            record_len = ego["record_len"]

            # Only use 2-agent samples (same as collab engine scenario)
            if record_len[0].item() != 2:
                continue

            # Run VFE + scatter to get spatial_features
            # center_point_codriving forward up to scatter
            voxel_batch = {
                "voxel_features": ego["processed_lidar"]["voxel_features"],
                "voxel_coords": ego["processed_lidar"]["voxel_coords"],
                "voxel_num_points": ego["processed_lidar"]["voxel_num_points"],
                "record_len": ego["record_len"],
            }
            voxel_batch = model.pillar_vfe(voxel_batch)
            voxel_batch = model.scatter(voxel_batch)
            spatial_features = voxel_batch["spatial_features"]
            # spatial_features: (2, 64, 256, 512)

            pairwise_t_matrix = ego["pairwise_t_matrix"]   # (1, 2, 2, 4, 4)

            spatial_list.append(spatial_features.cpu().float().numpy())
            tmat_list.append(pairwise_t_matrix.cpu().float().numpy())
            n_collected += 1
            if n_collected % 25 == 0:
                print(f"  [calib-collect] {n_collected}/{n_samples}")

    print(f"  [calib-collect] collected {n_collected} 2-agent samples")
    np.savez(output_path,
             spatial_features=np.stack(spatial_list),
             pairwise_t_matrix=np.stack(tmat_list))
    print(f"  [calib-collect] saved: {output_path}")


# ---------------------------------------------------------------------------
# Build TRT engine
# ---------------------------------------------------------------------------

def build_engine(mode: str, workspace_mb: int = 4096):
    """Build TRT engine from ONNX. mode: 'fp16' or 'int8'."""
    assert mode in ("fp16", "int8")
    output_path = ENGINE_FP16 if mode == "fp16" else ENGINE_INT8
    ENGINE_DIR.mkdir(parents=True, exist_ok=True)

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
        config.set_flag(trt.BuilderFlag.FP16)  # fallback layers can use FP16
        if not Path(str(CALIB_DATA)).exists():
            collect_calib_data(n_samples=150, output_path=str(CALIB_DATA))
        calibrator = CollabInt8Calibrator(
            calib_data_path=str(CALIB_DATA),
            cache_file=str(CALIB_CACHE),
        )
        config.int8_calibrator = calibrator
        print("  INT8 flag set + calibrator attached")

    print(f"  Building engine (may take several minutes)...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TRT engine build failed")

    with open(str(output_path), "wb") as f:
        f.write(serialized)

    elapsed = time.time() - t0
    size_mb = output_path.stat().st_size / 1e6
    print(f"  Engine saved: {output_path}  ({size_mb:.1f} MB, {elapsed:.0f}s)")
    return str(output_path)


# ---------------------------------------------------------------------------
# Quick sanity: run engine forward once and check output shapes
# ---------------------------------------------------------------------------

def sanity_check(engine_path: str, mode: str):
    print(f"\n[SANITY] {mode.upper()} engine: {engine_path}")
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    ctx = engine.create_execution_context()

    spatial = torch.zeros(SPATIAL_SHAPE, dtype=torch.float32, device="cuda")
    tmat = torch.zeros(TMAT_SHAPE, dtype=torch.float32, device="cuda")
    tmat[0, :, :, 0, 0] = 1.0
    tmat[0, :, :, 1, 1] = 1.0
    tmat[0, :, :, 2, 2] = 1.0
    tmat[0, :, :, 3, 3] = 1.0

    # Find tensor names
    input_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)
                   if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT]
    output_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)
                    if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT]

    # Set input shapes
    for n in input_names:
        if "spatial" in n:
            ctx.set_input_shape(n, SPATIAL_SHAPE)
        elif "pairwise" in n or "t_matrix" in n:
            ctx.set_input_shape(n, TMAT_SHAPE)

    # Alloc output buffers
    bufs = {}
    for n in input_names:
        if "spatial" in n:
            bufs[n] = spatial
        else:
            bufs[n] = tmat
    for n in output_names:
        shape = tuple(ctx.get_tensor_shape(n))
        bufs[n] = torch.empty(shape, dtype=torch.float32, device="cuda")

    for n, buf in bufs.items():
        ctx.set_tensor_address(n, int(buf.data_ptr()))

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        ctx.execute_async_v3(stream.cuda_stream)
    stream.synchronize()

    for n in output_names:
        print(f"  output {n}: {tuple(bufs[n].shape)}  mean={bufs[n].mean():.4f}")
    print(f"  SANITY PASS")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["fp16", "int8", "both"], default="both")
    p.add_argument("--workspace-mb", type=int, default=4096)
    p.add_argument("--n-calib", type=int, default=150,
                   help="Number of calibration samples for INT8")
    p.add_argument("--skip-sanity", action="store_true")
    args = p.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ENGINE_DIR.mkdir(parents=True, exist_ok=True)

    modes = ["fp16", "int8"] if args.mode == "both" else [args.mode]

    for mode in modes:
        # Skip if already built
        target = ENGINE_FP16 if mode == "fp16" else ENGINE_INT8
        if target.exists() and target.stat().st_size > 1e6:
            print(f"[SKIP] {mode.upper()} engine already exists: {target}")
        else:
            build_engine(mode, workspace_mb=args.workspace_mb)

        if not args.skip_sanity:
            sanity_check(str(target), mode)

    print(f"\n[DONE] Engines:")
    for path in [ENGINE_FP16, ENGINE_INT8]:
        if path.exists():
            print(f"  {path}  ({path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
