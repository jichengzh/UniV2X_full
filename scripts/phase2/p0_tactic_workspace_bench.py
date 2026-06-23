"""Part B-4090.2 — TRT tactic_sources + workspace bench on 4090.

Build engine with 4 (tactic, workspace) combos for T_baseline only:
    1. default (CUBLAS + CUBLAS_LT + EDGE_MASK + JIT) × 4GB
    2. default × 8GB
    3. default + CUDNN × 4GB
    4. default + CUDNN × 8GB

× {FP16, INT8} precision = 8 engines + bench points.

Output: data/tactic_workspace_bench.parquet
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = REPO_ROOT / "models/p0_random_cache"
OUT_DIR = REPO_ROOT / "models/tactic_ws_cache"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


# Try to read available tactic source flags
DEFAULT_TACTICS = (
    1 << int(trt.TacticSource.CUBLAS)
    | 1 << int(trt.TacticSource.CUBLAS_LT)
    | 1 << int(trt.TacticSource.EDGE_MASK_CONVOLUTIONS)
    | 1 << int(trt.TacticSource.JIT_CONVOLUTIONS)
)
WITH_CUDNN = DEFAULT_TACTICS | (1 << int(trt.TacticSource.CUDNN))

CONFIGS = [
    ("default_4GB",   DEFAULT_TACTICS, 4 * 1024**3),
    ("default_8GB",   DEFAULT_TACTICS, 8 * 1024**3),
    ("cudnn_4GB",     WITH_CUDNN,      4 * 1024**3),
    ("cudnn_8GB",     WITH_CUDNN,      8 * 1024**3),
]


class MinMaxCalib(trt.IInt8MinMaxCalibrator):
    """Minimal int8 calibrator using random tensor for 1 batch (faster than full calib)."""
    def __init__(self, cache_path, shapes_by_name):
        super().__init__()
        self.cache_path = Path(cache_path)
        self.shapes = shapes_by_name
        self.bufs = {n: torch.randn(*s, device="cuda").contiguous()
                     for n, s in shapes_by_name.items()}
        self.served = False

    def get_batch_size(self):
        return 1

    def get_batch(self, names):
        if self.served:
            return None
        self.served = True
        return [int(self.bufs[n].data_ptr()) for n in names]

    def read_calibration_cache(self):
        if self.cache_path.exists():
            return self.cache_path.read_bytes()
        return None

    def write_calibration_cache(self, blob):
        self.cache_path.write_bytes(bytes(blob))


def build_engine(onnx_path, precision, tactic_flag, workspace_bytes, engine_path, calib_cache=None):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX parse failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    config.set_tactic_sources(tactic_flag)

    # Default opt profile for dynamic shapes — use shape from input
    profile = builder.create_optimization_profile()
    spatial_shape = (2, 64, 128, 256)
    tego_shape = (2, 2, 3)
    profile.set_shape("spatial_features", spatial_shape, spatial_shape, spatial_shape)
    profile.set_shape("t_ego", tego_shape, tego_shape, tego_shape)
    config.add_optimization_profile(profile)

    calibrator = None
    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16)
        calibrator = MinMaxCalib(calib_cache, {
            "spatial_features": spatial_shape,
            "t_ego": tego_shape,
        })
        config.int8_calibrator = calibrator
        config.set_calibration_profile(profile)

    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    build_secs = time.time() - t0
    if serialized is None:
        raise RuntimeError("build failed")
    Path(engine_path).write_bytes(bytes(serialized))
    del calibrator, builder, network, parser, config
    gc.collect()
    torch.cuda.empty_cache()
    return build_secs


def bench(engine_path, n_warmup=100, n_measure=200):
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
    context = engine.create_execution_context()
    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    input_names = [n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]

    shapes_by_name = {"spatial_features": (2, 64, 128, 256), "t_ego": (2, 2, 3)}
    for n in input_names:
        context.set_input_shape(n, shapes_by_name[n])
    bufs = {}
    for name in names:
        dtype = engine.get_tensor_dtype(name)
        shape = tuple(context.get_tensor_shape(name))
        torch_dtype = {trt.float32: torch.float32, trt.float16: torch.float16,
                        trt.int32: torch.int32, trt.int8: torch.int8,
                        trt.int64: torch.int64}.get(dtype, torch.float32)
        bufs[name] = torch.empty(shape, dtype=torch_dtype, device="cuda")
        context.set_tensor_address(name, int(bufs[name].data_ptr()))
    torch.manual_seed(42)
    for n in input_names:
        bufs[n].copy_(torch.randn_like(bufs[n].float()).to(bufs[n].dtype))

    stream = torch.cuda.Stream()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = np.empty(n_measure, dtype=np.float64)
    with torch.cuda.stream(stream):
        for _ in range(n_warmup):
            context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        for k in range(n_measure):
            start.record(stream)
            context.execute_async_v3(stream.cuda_stream)
            end.record(stream)
            end.synchronize()
            times[k] = start.elapsed_time(end)
    return {"p50_ms": float(np.percentile(times, 50)),
            "p99_ms": float(np.percentile(times, 99)),
            "mean_ms": float(times.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="data/tactic_workspace_bench.parquet")
    args = ap.parse_args()

    # Only T_baseline (per plan2 B-4090.2.2)
    s0, s1, s2 = 64, 128, 256
    sig = f"{s0:03d}_{s1:03d}_{s2:03d}"
    onnx_path = CACHE_ROOT / f"onnx_{sig}.onnx"

    rows = []
    for prec in ("fp16", "int8"):
        for cfg_label, tactic, ws in CONFIGS:
            engine_path = OUT_DIR / f"engine_{sig}_{prec}_{cfg_label}.engine"
            calib_cache = OUT_DIR / f"calib_{sig}_{prec}_{cfg_label}.cache"
            print(f"[{prec} {cfg_label}] building (ws={ws//1024**3}GB) ...")
            try:
                t0 = time.time()
                build_secs = build_engine(onnx_path, prec, tactic, ws,
                                           engine_path, calib_cache)
                stats = bench(engine_path)
                elapsed = time.time() - t0
            except Exception as e:
                print(f"  FAILED: {e}")
                continue
            print(f"  build={build_secs:.0f}s bench p50={stats['p50_ms']:.3f} p99={stats['p99_ms']:.3f} ({elapsed:.0f}s)")
            rows.append({
                "precision": prec, "config": cfg_label,
                "tactic_flag": int(tactic),
                "workspace_gb": ws // 1024**3,
                "build_secs": build_secs,
                "lat_p50_ms": stats["p50_ms"],
                "lat_p99_ms": stats["p99_ms"],
                "lat_mean_ms": stats["mean_ms"],
            })

    import pandas as pd
    df = pd.DataFrame(rows)
    out = REPO_ROOT / args.output
    df.to_parquet(out)
    df.to_csv(out.with_suffix(".csv"), index=False)
    print(f"\nwrote {len(df)} rows -> {out}")
    print(df[["precision","config","lat_p50_ms","lat_p99_ms","build_secs"]].to_string(index=False))


if __name__ == "__main__":
    main()
