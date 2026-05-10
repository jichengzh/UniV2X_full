"""Phase A.2/A.3 — Build TRT engine (FP16 / INT8) from ONNX and benchmark.

This is the *real* acceleration path that M4.6 was avoiding:
  * Phase A.2: --precision fp16 (no calibration needed)
  * Phase A.3: --precision int8 --calib-cache <bin>  (post-training quantization)

Output engine + JSON report with mean/p50/p95/p99 latency in ms (model body only).

Usage
-----
Phase A.2 (FP16):
    python scripts/phase1/m4_8_trt_build_bench.py \
        --onnx models/pyramid_m1_subnet_fp32.onnx \
        --precision fp16 \
        --engine models/pyramid_m1_subnet_fp16.engine \
        --report results/m4_8_trt_fp16.json

Phase A.3 (INT8):
    python scripts/phase1/m4_8_trt_build_bench.py \
        --onnx models/pyramid_m1_subnet_fp32.onnx \
        --precision int8 \
        --calib-data calibration/pyramid_calib.npy \
        --engine models/pyramid_m1_subnet_int8.engine \
        --report results/m4_8_trt_int8.json
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
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


# ---------------------------------------------------------------------------
# INT8 calibrator
# ---------------------------------------------------------------------------

def _make_calibrator_class(base_cls):
    """Build a calibrator subclass on top of the chosen TRT base class."""

    class _NumpyCalib(base_cls):  # type: ignore[misc, valid-type]
        def __init__(self, calib_npy: str, batch_size: int = 1, cache_path: str | None = None):
            super().__init__()
            self._init_data(calib_npy, batch_size, cache_path)

        def _init_data(self, calib_npy, batch_size, cache_path):
            data = np.load(calib_npy).astype(np.float32)
            assert data.ndim == 4, f"calib data must be (N,C,H,W), got {data.shape}"
            self.data = data
            self.batch_size = batch_size
            self.idx = 0
            self.n = data.shape[0]
            self.device_input = torch.zeros(
                (batch_size, *data.shape[1:]), dtype=torch.float32, device="cuda"
            )
            self.cache_path = cache_path
            print(f"[calib:{base_cls.__name__}] {self.n} samples shape={data.shape}")

        def get_batch_size(self):
            return self.batch_size

        def get_batch(self, names):
            if self.idx + self.batch_size > self.n:
                return None
            batch = self.data[self.idx : self.idx + self.batch_size]
            self.idx += self.batch_size
            self.device_input.copy_(torch.from_numpy(batch))
            return [int(self.device_input.data_ptr())]

        def read_calibration_cache(self):
            if self.cache_path and Path(self.cache_path).exists():
                with open(self.cache_path, "rb") as f:
                    return f.read()
            return None

        def write_calibration_cache(self, cache):
            if self.cache_path:
                with open(self.cache_path, "wb") as f:
                    f.write(cache)

    _NumpyCalib.__name__ = f"NumpyCalibrator_{base_cls.__name__}"
    return _NumpyCalib


NumpyCalibratorMinMax = _make_calibrator_class(trt.IInt8MinMaxCalibrator)
NumpyCalibratorEntropy = _make_calibrator_class(trt.IInt8EntropyCalibrator2)


class NumpyCalibrator(trt.IInt8MinMaxCalibrator):
    """[DEPRECATED — kept for compat] Min-max INT8 calibrator.

    Prefer NumpyCalibratorMinMax / NumpyCalibratorEntropy from
    _make_calibrator_class() for explicit calibrator selection.
    """

    def __init__(self, calib_npy: str, batch_size: int = 1, cache_path: str | None = None):
        super().__init__()
        self.data = np.load(calib_npy).astype(np.float32)
        assert self.data.ndim == 4, f"calib data must be (N,C,H,W), got {self.data.shape}"
        self.batch_size = batch_size
        self.idx = 0
        self.n = self.data.shape[0]
        # Pre-allocate device buffer
        self.device_input = torch.zeros(
            (batch_size, *self.data.shape[1:]), dtype=torch.float32, device="cuda"
        )
        self.cache_path = cache_path
        print(f"[calib] loaded {self.n} samples shape={self.data.shape}")

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.idx + self.batch_size > self.n:
            return None
        batch = self.data[self.idx : self.idx + self.batch_size]
        self.idx += self.batch_size
        self.device_input.copy_(torch.from_numpy(batch))
        return [int(self.device_input.data_ptr())]

    def read_calibration_cache(self):
        if self.cache_path and Path(self.cache_path).exists():
            with open(self.cache_path, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        if self.cache_path:
            with open(self.cache_path, "wb") as f:
                f.write(cache)


# ---------------------------------------------------------------------------
# Build engine
# ---------------------------------------------------------------------------

def build_engine(
    onnx_path: str,
    precision: str,
    engine_path: str,
    workspace_mb: int = 4096,
    calib_npy: str | None = None,
    calib_cache: str | None = None,
    calibrator_kind: str = "minmax",
):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(0)  # explicit batch (default in TRT 10)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    print(f"[build] parsing ONNX: {onnx_path}")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  parse error: {parser.get_error(i)}")
            raise RuntimeError("ONNX parse failed")

    print(f"[build] network: inputs={network.num_inputs}  outputs={network.num_outputs}")
    for i in range(network.num_inputs):
        t = network.get_input(i)
        print(f"  input[{i}] {t.name} shape={t.shape} dtype={t.dtype}")
    for i in range(network.num_outputs):
        t = network.get_output(i)
        print(f"  output[{i}] {t.name} shape={t.shape} dtype={t.dtype}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb << 20)

    calibrator = None
    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
        print("[build] precision=FP16")
    elif precision == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16)  # mixed: INT8 + FP16 fallback
        if calib_npy is None:
            raise ValueError("INT8 build requires --calib-data")
        cls_map = {"minmax": NumpyCalibratorMinMax, "entropy": NumpyCalibratorEntropy}
        calib_cls = cls_map[calibrator_kind]
        calibrator = calib_cls(calib_npy, batch_size=1, cache_path=calib_cache)
        config.int8_calibrator = calibrator
        print(f"[build] precision=INT8 (FP16 fallback) calibrator={calibrator_kind}")
    elif precision == "fp32":
        print("[build] precision=FP32")
    else:
        raise ValueError(precision)

    print(f"[build] building engine (workspace={workspace_mb}MB) ...")
    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TRT build failed")
    build_secs = time.time() - t0
    blob = bytes(serialized)
    print(f"[build] done in {build_secs:.1f}s, engine size {len(blob) / 1e6:.2f} MB")

    Path(engine_path).parent.mkdir(parents=True, exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(blob)
    print(f"[build] saved engine -> {engine_path}")

    # Free calibrator (frees device memory)
    del calibrator
    gc.collect()
    torch.cuda.empty_cache()

    return engine_path, build_secs


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def benchmark_engine(
    engine_path: str,
    input_shape: tuple[int, int, int, int] = (1, 64, 256, 256),
    n_warmup: int = 200,
    n_measure: int = 200,
):
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    # Bind tensors
    n_io = engine.num_io_tensors
    tensor_names = [engine.get_tensor_name(i) for i in range(n_io)]
    input_name = next(n for n in tensor_names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
    output_names = [n for n in tensor_names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]

    context.set_input_shape(input_name, input_shape)

    # Allocate buffers
    bufs: dict[str, torch.Tensor] = {}
    for name in tensor_names:
        shape = tuple(context.get_tensor_shape(name))
        dtype = engine.get_tensor_dtype(name)
        torch_dtype = {
            trt.float32: torch.float32, trt.float16: torch.float16,
            trt.int32: torch.int32, trt.int8: torch.int8,
            trt.int64: torch.int64,
        }.get(dtype, torch.float32)
        bufs[name] = torch.empty(shape, dtype=torch_dtype, device="cuda")
        context.set_tensor_address(name, int(bufs[name].data_ptr()))

    # Fill input with deterministic data
    torch.manual_seed(42)
    bufs[input_name].copy_(torch.randn_like(bufs[input_name].float()).to(bufs[input_name].dtype))

    stream = torch.cuda.Stream()
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    print(f"[bench] warmup {n_warmup} iters ...")
    with torch.cuda.stream(stream):
        for _ in range(n_warmup):
            context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()

    print(f"[bench] measure {n_measure} iters ...")
    times_ms = np.empty(n_measure, dtype=np.float64)
    with torch.cuda.stream(stream):
        for k in range(n_measure):
            start_evt.record(stream)
            context.execute_async_v3(stream.cuda_stream)
            end_evt.record(stream)
            end_evt.synchronize()
            times_ms[k] = start_evt.elapsed_time(end_evt)

    stats = {
        "n_warmup": n_warmup,
        "n_measure": n_measure,
        "mean_ms": float(times_ms.mean()),
        "std_ms": float(times_ms.std()),
        "p50_ms": float(np.percentile(times_ms, 50)),
        "p95_ms": float(np.percentile(times_ms, 95)),
        "p99_ms": float(np.percentile(times_ms, 99)),
        "min_ms": float(times_ms.min()),
        "max_ms": float(times_ms.max()),
        "engine_size_mb": Path(engine_path).stat().st_size / 1e6,
        "input_shape": list(input_shape),
        "input_name": input_name,
        "output_names": output_names,
    }
    print(f"[bench] mean={stats['mean_ms']:.3f}ms  p50={stats['p50_ms']:.3f}  "
          f"p95={stats['p95_ms']:.3f}  p99={stats['p99_ms']:.3f}  std={stats['std_ms']:.3f}")
    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True)
    p.add_argument("--precision", choices=["fp32", "fp16", "int8"], required=True)
    p.add_argument("--engine", required=True)
    p.add_argument("--report", required=True)
    p.add_argument("--workspace-mb", type=int, default=4096)
    p.add_argument("--calib-data", default=None, help="numpy (N,C,H,W) for INT8")
    p.add_argument("--calib-cache", default=None, help="cache file for calibrator")
    p.add_argument("--calibrator", choices=["minmax", "entropy"], default="minmax",
                   help="INT8 calibrator (minmax=IInt8MinMaxCalibrator, entropy=IInt8EntropyCalibrator2)")
    p.add_argument("--n-warmup", type=int, default=200)
    p.add_argument("--n-measure", type=int, default=200)
    p.add_argument("--input-shape", default="1,64,256,256")
    p.add_argument("--skip-build", action="store_true", help="reuse existing engine")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.skip_build:
        engine_path, build_secs = build_engine(
            args.onnx, args.precision, args.engine,
            workspace_mb=args.workspace_mb,
            calib_npy=args.calib_data, calib_cache=args.calib_cache,
            calibrator_kind=args.calibrator,
        )
    else:
        engine_path = args.engine
        build_secs = None

    shape = tuple(int(x) for x in args.input_shape.split(","))
    stats = benchmark_engine(engine_path, input_shape=shape,
                             n_warmup=args.n_warmup, n_measure=args.n_measure)
    stats["precision"] = args.precision
    stats["onnx"] = args.onnx
    stats["engine"] = args.engine
    stats["build_secs"] = build_secs

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n  report -> {args.report}")


if __name__ == "__main__":
    main()
