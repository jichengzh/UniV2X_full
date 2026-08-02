#!/usr/bin/env python3
"""TRT baseline profiler for Phase 2.A multi-framework comparison (Pyramid-LiDAR).

Builds a TensorRT engine from the SAME ONNX our TVM path uses (pyramid_backbone
subnet, input spatial_features [2,64,128,256], input_hw=128x256) and profiles
latency + energy with the SAME caliber as framework/measure_config.py:
  - latency: p50 over warmup=20 + measure=300 x repeat=5, CUDA-event timed.
  - energy : energy_j = watt_avg * lat_p50_ms / 1000  (total power x latency,
             NVML-sampled over a >=5s active window). Matches measure_config.

TRT is a COMPARISON BASELINE (the opponent), NOT our backend. Our backend is TVM.

Usage:
  trt_profile_v1.py --onnx PATH --precision {fp32,fp16,int8} --gpu N [--calib-dir DIR]
                    [--warmup 20 --iters 300 --repeat 5 --energy-secs 5]
                    [--out result.json]

INT8 uses IInt8EntropyCalibrator2 reading real DAIR backbone-input activations
(.npy of shape [2,64,128,256]) from --calib-dir (no-peek: calib set != eval set).
"""
import argparse
import gc
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def builder_flags_for_precision(precision):
    if precision == "fp32":
        return []
    if precision == "fp16":
        return ["fp16"]
    if precision == "int8":
        return ["int8", "fp16_fallback"]
    raise ValueError(precision)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# INT8 calibrator: reads pre-dumped real activations from a directory of .npy
# ---------------------------------------------------------------------------
class NpyEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, calib_dir, input_name, cache_file):
        super().__init__()
        import pycuda.driver as cuda
        self.cuda = cuda
        self.input_name = input_name
        self.cache_file = cache_file
        self.files = sorted(
            os.path.join(calib_dir, f) for f in os.listdir(calib_dir) if f.endswith(".npy")
        )
        if not self.files:
            raise RuntimeError(f"no .npy calibration tensors in {calib_dir}")
        self.idx = 0
        arr = np.load(self.files[0]).astype(np.float32)
        self.shape = arr.shape
        self.nbytes = int(arr.nbytes)
        self.d_in = cuda.mem_alloc(self.nbytes)

    def get_batch_size(self):
        return int(self.shape[0])

    def get_batch(self, names):
        if self.idx >= len(self.files):
            return None
        arr = np.ascontiguousarray(np.load(self.files[self.idx]).astype(np.float32))
        self.cuda.memcpy_htod(self.d_in, arr)
        self.idx += 1
        return [int(self.d_in)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)


def build_engine(
    onnx_path,
    precision,
    calib_dir,
    workspace_gb=4,
    artifact_dir=None,
    builder_optimization_level=3,
    calibration_dataset="dair-v2x-c",
):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errs = [parser.get_error(i) for i in range(parser.num_errors)]
            raise RuntimeError(f"ONNX parse failed: {errs}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    config.builder_optimization_level = int(builder_optimization_level)
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    input_name = network.get_input(0).name
    input_shape = tuple(network.get_input(0).shape)

    if precision == "fp32":
        pass
    elif precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        if not builder.platform_has_fast_int8:
            raise RuntimeError("platform lacks fast int8")
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16)  # allow int8+fp16 fallback (TRT default practice)
        if not calib_dir:
            raise RuntimeError("int8 requires --calib-dir with real activations")
        cache = os.path.join(os.path.dirname(onnx_path), "trt_int8_calib.cache")
        config.int8_calibrator = NpyEntropyCalibrator(calib_dir, input_name, cache)
    else:
        raise ValueError(precision)

    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    build_s = time.time() - t0
    if serialized is None:
        raise RuntimeError("engine build returned None")
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(serialized)
    if artifact_dir is not None:
        artifact_dir = Path(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "compiled.engine").write_bytes(bytes(serialized))
        inspector = engine.create_engine_inspector()
        (artifact_dir / "engine_inspector.json").write_text(
            inspector.get_engine_information(trt.LayerInformationFormat.JSON),
            encoding="utf-8",
        )
        calibration_manifest_sha256 = None
        if precision == "int8":
            calibration_files = []
            for source in sorted(Path(calib_dir).glob("*.npy")):
                calibration_files.append(
                    {
                        "name": source.name,
                        "size_bytes": source.stat().st_size,
                        "sha256": sha256_file(source),
                    }
                )
            calibration_manifest = artifact_dir / "calibration_manifest.json"
            calibration_manifest.write_text(
                json.dumps(
                    {
                        "schema_version": "stage2_formal_calibration_manifest_v3",
                        "dataset": str(calibration_dataset),
                        "split_role": "calibration_only",
                        "sample_count": len(calibration_files),
                        "input_name": input_name,
                        "input_shape": list(input_shape),
                        "source_onnx_sha256": sha256_file(onnx_path),
                        "files": calibration_files,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            calibration_manifest_sha256 = sha256_file(calibration_manifest)
        (artifact_dir / "engine_build_config.json").write_text(
            json.dumps(
                {
                    "onnx_path": str(Path(onnx_path).resolve()),
                    "precision": precision,
                    "workspace_gb": workspace_gb,
                    "builder_optimization_level": int(builder_optimization_level),
                    "calibration_dataset": str(calibration_dataset),
                    "calib_dir": str(Path(calib_dir).resolve()) if calib_dir else None,
                    "trt_version": trt.__version__,
                    "builder_flags": builder_flags_for_precision(precision),
                    "source_onnx_sha256": sha256_file(onnx_path),
                    "calibration_manifest_sha256": calibration_manifest_sha256,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        if precision == "int8" and Path(cache).is_file():
            (artifact_dir / "calibration.cache").write_bytes(Path(cache).read_bytes())
    return engine, input_name, input_shape, int(serialized.nbytes), build_s


def profile(engine, input_name, input_shape, nvml_gpu, warmup, iters, repeat, energy_secs):
    import pycuda.driver as cuda
    import pynvml

    context = engine.create_execution_context()
    # allocate device buffers for all IO tensors
    bufs, host_out = {}, {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = tuple(context.get_tensor_shape(name))
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        d = cuda.mem_alloc(nbytes)
        bufs[name] = d
        context.set_tensor_address(name, int(d))
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            h = np.ascontiguousarray(np.random.randn(*shape).astype(dtype))
            cuda.memcpy_htod(d, h)
        else:
            host_out[name] = np.empty(shape, dtype=dtype)
    stream = cuda.Stream()

    def run_once():
        context.execute_async_v3(stream.handle)

    # warmup
    for _ in range(warmup):
        run_once()
    stream.synchronize()

    for name, host in host_out.items():
        cuda.memcpy_dtoh_async(host, bufs[name], stream)
    stream.synchronize()
    numerical_finite = all(bool(np.isfinite(host).all()) for host in host_out.values())

    # latency: CUDA-event timed, iters x repeat samples, p50
    start_ev, end_ev = cuda.Event(), cuda.Event()
    samples_ms = []
    for _ in range(repeat):
        for _ in range(iters):
            start_ev.record(stream)
            run_once()
            end_ev.record(stream)
            end_ev.synchronize()
            samples_ms.append(start_ev.time_till(end_ev))
    lat_p50 = statistics.median(samples_ms)
    lat_mean = statistics.mean(samples_ms)
    ordered_samples = sorted(samples_ms)
    lat_p90 = ordered_samples[int(0.90 * len(samples_ms)) - 1]
    lat_p99 = ordered_samples[int(0.99 * len(samples_ms)) - 1]

    # energy: sample NVML power over >= energy_secs active window.
    # nvml_gpu is the ABSOLUTE index (matches the pycuda-selected device).
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(nvml_gpu)
    watts, t_end = [], time.time() + energy_secs
    n_active = 0
    while time.time() < t_end or n_active < iters:
        run_once()
        n_active += 1
        if n_active % 10 == 0:
            stream.synchronize()
            watts.append(pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0)
    stream.synchronize()
    watt_avg = statistics.mean(watts) if watts else float("nan")
    pynvml.nvmlShutdown()
    energy_j = watt_avg * lat_p50 / 1000.0

    return {
        "lat_p50_ms": lat_p50, "lat_mean_ms": lat_mean,
        "lat_p90_ms": lat_p90, "lat_p99_ms": lat_p99,
        "n_lat_samples": len(samples_ms),
        "watt_avg": watt_avg, "energy_j": energy_j, "n_watt_samples": len(watts),
        "warmup": warmup, "iters": iters, "repeat": repeat, "energy_secs": energy_secs,
        "numerical_finite": numerical_finite,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--precision", choices=["fp32", "fp16", "int8"], required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--calib-dir", default=None)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--energy-secs", type=float, default=5.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--artifact-dir", default=None)
    ap.add_argument(
        "--builder-optimization-level",
        type=int,
        choices=range(0, 6),
        default=3,
    )
    ap.add_argument("--calibration-dataset", default="dair-v2x-c")
    args = ap.parse_args()

    # Select the target device explicitly via pycuda so CUDA and NVML agree on the
    # absolute index (avoids CUDA_VISIBLE_DEVICES vs NVML index mismatch).
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    import pycuda.driver as cuda
    cuda.init()
    ctx = cuda.Device(args.gpu).make_context()
    engine = None
    try:
        engine, in_name, in_shape, engine_bytes, build_s = build_engine(
            args.onnx,
            args.precision,
            args.calib_dir,
            artifact_dir=args.artifact_dir,
            builder_optimization_level=args.builder_optimization_level,
            calibration_dataset=args.calibration_dataset,
        )
        prof = profile(engine, in_name, in_shape, args.gpu,
                       args.warmup, args.iters, args.repeat, args.energy_secs)
    finally:
        if engine is not None:
            del engine
        gc.collect()
        ctx.pop()

    rec = {
        "framework": "tensorrt", "trt_version": trt.__version__,
        "precision": args.precision, "onnx": os.path.basename(args.onnx),
        "input_name": in_name, "input_shape": list(in_shape),
        "engine_bytes": engine_bytes, "build_s": round(build_s, 2),
        "build_success": True, "gpu_abs": args.gpu,
        "builder_optimization_level": args.builder_optimization_level,
        "caliber": "backbone-subnet, input_hw=128x256, batch=2; "
                   "latency p50 warmup20/iters300/repeat5 CUDA-event; "
                   "energy watt_avg*lat_p50/1000 NVML >=5s (matches measure_config)",
        **prof,
    }
    if args.artifact_dir:
        artifact_dir = Path(args.artifact_dir)
        rec["artifact_sha256"] = {
            "source_onnx": sha256_file(args.onnx),
            "compiled_engine": sha256_file(artifact_dir / "compiled.engine"),
            "engine_build_config": sha256_file(artifact_dir / "engine_build_config.json"),
            "engine_inspector": sha256_file(artifact_dir / "engine_inspector.json"),
        }
        calibration_manifest = artifact_dir / "calibration_manifest.json"
        if calibration_manifest.is_file():
            rec["artifact_sha256"]["calibration_manifest"] = sha256_file(calibration_manifest)
        calibration_cache = artifact_dir / "calibration.cache"
        if calibration_cache.is_file():
            rec["artifact_sha256"]["calibration_cache"] = sha256_file(calibration_cache)
    print(json.dumps(rec, indent=1))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rec, f, indent=1)
        print(f"[out] {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
