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
    """Build a calibrator subclass on top of the chosen TRT base class.

    Now supports multi-input engines: pass calib_npy as a dict
    {input_name: npy_path} for engines with >1 input (Phase A.5 collab).
    Single-input legacy: calib_npy is a str path → uses 'auto' name detection.
    """

    class _NumpyCalib(base_cls):  # type: ignore[misc, valid-type]
        def __init__(self, calib_npy, batch_size: int = 1, cache_path: str | None = None):
            super().__init__()
            self._init_data(calib_npy, batch_size, cache_path)

        def _init_data(self, calib_npy, batch_size, cache_path):
            # Normalize: dict {name: path} or single str path
            if isinstance(calib_npy, dict):
                self.npy_map = calib_npy
            else:
                # Single-input legacy
                self.npy_map = {"_auto": calib_npy}

            self.arrays = {}
            self.devs = {}
            n_total = None
            for k, p in self.npy_map.items():
                # Preserve native dtype (int32 for index inputs like voxel_coords)
                a = np.load(p)
                if a.dtype not in (np.int32, np.int64):
                    a = a.astype(np.float32)
                assert a.ndim >= 2
                self.arrays[k] = a
                torch_dtype = {
                    np.float32: torch.float32, np.int32: torch.int32,
                    np.int64: torch.int64,
                }.get(a.dtype.type, torch.float32)
                self.devs[k] = torch.zeros(
                    (batch_size, *a.shape[1:]), dtype=torch_dtype, device="cuda"
                )
                if n_total is None:
                    n_total = a.shape[0]
                else:
                    assert a.shape[0] == n_total, "calib datasets must have same N"
            self.batch_size = batch_size
            self.idx = 0
            self.n = n_total
            self.cache_path = cache_path
            print(f"[calib:{base_cls.__name__}] {self.n} samples, "
                  f"inputs={list(self.arrays.keys())}")

        def get_batch_size(self):
            return self.batch_size

        def get_batch(self, names):
            if self.idx + self.batch_size > self.n:
                return None
            batch_addrs = []
            # If single-input legacy, ignore names and just feed the one array
            if "_auto" in self.arrays:
                arr = self.arrays["_auto"]
                self.devs["_auto"].copy_(
                    torch.from_numpy(arr[self.idx : self.idx + self.batch_size])
                )
                batch_addrs = [int(self.devs["_auto"].data_ptr())]
            else:
                for nm in names:
                    if nm not in self.arrays:
                        raise RuntimeError(
                            f"calibrator: no data for input '{nm}', "
                            f"have {list(self.arrays.keys())}"
                        )
                    a = self.arrays[nm]
                    self.devs[nm].copy_(
                        torch.from_numpy(a[self.idx : self.idx + self.batch_size])
                    )
                    batch_addrs.append(int(self.devs[nm].data_ptr()))
            self.idx += self.batch_size
            return batch_addrs

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

    # --tactic D 维度: set TRT tactic sources (TRT 10)
    tactic_mode = globals().get("_TACTIC_MODE")
    if tactic_mode:
        DEFAULT_TC = (
            1 << int(trt.TacticSource.CUBLAS)
            | 1 << int(trt.TacticSource.CUBLAS_LT)
            | 1 << int(trt.TacticSource.EDGE_MASK_CONVOLUTIONS)
            | 1 << int(trt.TacticSource.JIT_CONVOLUTIONS)
        )
        WITH_CUDNN = DEFAULT_TC | (1 << int(trt.TacticSource.CUDNN))
        EDGE_ONLY = (
            1 << int(trt.TacticSource.EDGE_MASK_CONVOLUTIONS)
            | 1 << int(trt.TacticSource.JIT_CONVOLUTIONS)
        )
        CUBLAS_LT_ONLY = 1 << int(trt.TacticSource.CUBLAS_LT)
        ALL_ENABLED = WITH_CUDNN  # 全部启用 = default + CUDNN
        tactic_map = {
            "default": DEFAULT_TC,
            "with_cudnn": WITH_CUDNN,
            "edge_only": EDGE_ONLY,
            "cublas_lt": CUBLAS_LT_ONLY,
            "all_enabled": ALL_ENABLED,
        }
        if tactic_mode not in tactic_map:
            raise ValueError(f"unknown tactic {tactic_mode}")
        config.set_tactic_sources(tactic_map[tactic_mode])
        print(f"[build] tactic_sources = {tactic_mode}")

    # --builder-opt-level D 子维度: TRT 10 builder_optimization_level
    bl = globals().get("_BUILDER_OPT_LEVEL")
    if bl is not None:
        config.builder_optimization_level = bl
        print(f"[build] builder_optimization_level = {bl}")

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
    elif precision == "mixed":
        # Mixed precision: globally FP16, but layers matching --mixed-int8-pattern
        # are forced INT8. Used for #3 (per-module q_bits dimension).
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.INT8)
        if calib_npy is None:
            raise ValueError("Mixed precision needs INT8 calibrator for INT8 layers")
        cls_map = {"minmax": NumpyCalibratorMinMax, "entropy": NumpyCalibratorEntropy}
        calibrator = cls_map[calibrator_kind](calib_npy, batch_size=1, cache_path=calib_cache)
        config.int8_calibrator = calibrator
        print("[build] precision=MIXED FP16+INT8 (per-layer constraint via OBEY_PRECISION_CONSTRAINTS)")
        # Caller will set per-layer precision via network.get_layer().set_precision()
    elif precision == "fp16_sparse":
        # NVIDIA 2:4 sparsity + FP16 (#4). Requires weights pre-sparsified to 2:4.
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)
        print("[build] precision=FP16 + SPARSE_WEIGHTS (2:4 sparsity if weights pre-sparsified)")
    elif precision == "int8_sparse":
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)
        if calib_npy is None:
            raise ValueError("int8_sparse needs INT8 calibrator")
        cls_map = {"minmax": NumpyCalibratorMinMax, "entropy": NumpyCalibratorEntropy}
        calibrator = cls_map[calibrator_kind](calib_npy, batch_size=1, cache_path=calib_cache)
        config.int8_calibrator = calibrator
        print("[build] precision=INT8 + SPARSE_WEIGHTS (NVIDIA SparsityINT8)")
    else:
        raise ValueError(precision)

    # Optional: per-layer mixed precision constraint (#3)
    # Pass --mixed-int8-substr "Conv;BatchNorm" to force INT8 for layers whose name
    # contains any substring; FP16 elsewhere.
    if precision == "mixed" and getattr(config, "_mixed_int8_substr", None):
        # Will be set by caller after network is parsed
        pass

    # Apply per-layer mixed precision (#3) BEFORE build
    # Skip non-quantizable layer kinds (Shape/Identity/Constant/Cast/Slice/Concatenation
    # that pass through Int64/Bool tensors). Setting INT8/FP16 on these makes TRT bail.
    QUANTIZABLE_KINDS = {
        trt.LayerType.CONVOLUTION,
        trt.LayerType.DECONVOLUTION,
        trt.LayerType.MATRIX_MULTIPLY,
        trt.LayerType.ELEMENTWISE,
        trt.LayerType.ACTIVATION,
        trt.LayerType.POOLING,
        trt.LayerType.SCALE,
        trt.LayerType.SOFTMAX,
        trt.LayerType.UNARY,
        trt.LayerType.REDUCE,
        trt.LayerType.NORMALIZATION,
        # Note: GRID_SAMPLE (warp_affine) is NOT INT8-quantizable under OBEY_PRECISION_CONSTRAINTS
        # Note: SHAPE / IDENTITY / CONSTANT / CAST / SLICE / CONCATENATION / GATHER pass through
    }
    def _set_layer_precision(substrs, match_int8: bool):
        n_int8 = n_fp16 = n_skip = 0
        for li in range(network.num_layers):
            layer = network.get_layer(li)
            if layer.type not in QUANTIZABLE_KINDS:
                n_skip += 1
                continue
            hit = any(s and s.lower() in layer.name.lower() for s in substrs)
            if hit == match_int8:  # match → INT8 (when match_int8=True), else FP16
                layer.precision = trt.int8
                n_int8 += 1
            else:
                layer.precision = trt.float16
                n_fp16 += 1
        # PREFER (not OBEY) so TRT can fall back to FP16/FP32 for layers that
        # don't have INT8 kernels (e.g. GridSample, Equal, Myelin foreign nodes)
        config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
        return n_int8, n_fp16, n_skip

    if precision == "mixed" and globals().get("_MIXED_INT8_SUBSTRS"):
        substrs = globals()["_MIXED_INT8_SUBSTRS"]
        n_int8, n_fp16, n_skip = _set_layer_precision(substrs, match_int8=True)
        print(f"[build] mixed (INT8-match): {n_int8} INT8 / {n_fp16} FP16 / {n_skip} unconstrained "
              f"(substrs={substrs})")
    elif precision == "mixed" and globals().get("_MIXED_FP16_SUBSTRS"):
        substrs = globals()["_MIXED_FP16_SUBSTRS"]
        n_int8, n_fp16, n_skip = _set_layer_precision(substrs, match_int8=False)
        print(f"[build] mixed (FP16-match): {n_fp16} FP16 / {n_int8} INT8 / {n_skip} unconstrained "
              f"(substrs={substrs})")

    # --w-only: 对 INT8 build 强制 Conv 输出 FP16 (activation 在层间保持 FP16, weight 仍 INT8)
    # 实现机制: layer.set_output_type(0, trt.float16) 让 TRT 在 Conv 出口插入 dequant
    if precision in ("int8", "int8_sparse") and globals().get("_W_ONLY", False):
        n_conv_fp16 = 0
        for li in range(network.num_layers):
            layer = network.get_layer(li)
            if layer.type == trt.LayerType.CONVOLUTION:
                layer.set_output_type(0, trt.float16)
                n_conv_fp16 += 1
        config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
        print(f"[build] W-only: forced {n_conv_fp16} Conv output_type → FP16 "
              f"(weight INT8 + activation FP16 between layers)")

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
    input_shape=(1, 64, 256, 256),
    extra_input_shapes: dict[str, tuple] | None = None,
    n_warmup: int = 200,
    n_measure: int = 200,
):
    """input_shape: shape for the *first* input. extra_input_shapes: optional dict
    {name: shape} for additional inputs (Phase A.5 collab engine has 2 inputs)."""
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    # Bind tensors
    n_io = engine.num_io_tensors
    tensor_names = [engine.get_tensor_name(i) for i in range(n_io)]
    input_names = [n for n in tensor_names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
    output_names = [n for n in tensor_names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]

    # Set shapes: first input gets input_shape, rest from extra_input_shapes
    context.set_input_shape(input_names[0], input_shape)
    extra = extra_input_shapes or {}
    for nm in input_names[1:]:
        if nm not in extra:
            raise ValueError(f"engine has multi-input ({input_names}); "
                             f"missing shape for '{nm}' in extra_input_shapes")
        context.set_input_shape(nm, extra[nm])

    input_name = input_names[0]  # for backwards-compat (first input)

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

    # Fill all inputs with deterministic random data (multi-input safe)
    torch.manual_seed(42)
    for nm in input_names:
        bufs[nm].copy_(torch.randn_like(bufs[nm].float()).to(bufs[nm].dtype))

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
    p.add_argument("--precision",
                   choices=["fp32", "fp16", "int8", "mixed", "fp16_sparse", "int8_sparse"],
                   required=True,
                   help="mixed: per-layer (caller specifies via --mixed-int8-substr); "
                        "fp16_sparse/int8_sparse: NVIDIA 2:4 sparsity")
    p.add_argument("--mixed-int8-substr", default="",
                   help="For --precision mixed: comma-separated substrings — layers "
                        "whose name contains any get INT8 precision; rest stay FP16")
    p.add_argument("--mixed-fp16-substr", default="",
                   help="For --precision mixed: layers matching these substrs stay FP16, "
                        "rest become INT8 (use this when heads should stay FP16)")
    p.add_argument("--engine", required=True)
    p.add_argument("--report", required=True)
    p.add_argument("--workspace-mb", type=int, default=4096)
    p.add_argument("--calib-data", default=None, help="numpy (N,C,H,W) for INT8")
    p.add_argument("--calib-multi", action="append", default=[],
                   help="multi-input calib spec 'input_name:path.npy' (repeatable). "
                        "Use --calib-multi for engines with >1 input (Phase A.5).")
    p.add_argument("--calib-cache", default=None, help="cache file for calibrator")
    p.add_argument("--calibrator", choices=["minmax", "entropy"], default="minmax",
                   help="INT8 calibrator (minmax=IInt8MinMaxCalibrator, entropy=IInt8EntropyCalibrator2)")
    p.add_argument("--w-only", action="store_true",
                   help="W-only mode: weights INT8 (per-channel default), activations FP16. "
                        "Forces Conv layer output_type to FP16 — TRT keeps INT8 weights from "
                        "calibrator but dequantizes activations between layers.")
    p.add_argument("--tactic",
                   choices=["default", "with_cudnn", "edge_only", "cublas_lt", "all_enabled"],
                   default=None,
                   help="TRT tactic_sources mode (D-dim D2-D4 sweep). Omit = TRT defaults.")
    p.add_argument("--builder-opt-level", type=int, default=None,
                   help="TRT builder_optimization_level ∈ [0, 5]. Omit = TRT default (3). "
                        "0 = least tactic search (fastest build, noisiest lat); "
                        "5 = full tactic search (slowest build, most deterministic lat). "
                        "D-dim 3rd sub-dim — paper §C 噪声地板控制.")
    p.add_argument("--n-warmup", type=int, default=200)
    p.add_argument("--n-measure", type=int, default=200)
    p.add_argument("--input-shape", default="1,64,256,256")
    p.add_argument("--extra-input-shape", action="append", default=[],
                   help="additional input shape, format 'name:1,2,3,4' (repeatable)")
    p.add_argument("--skip-build", action="store_true", help="reuse existing engine")
    p.add_argument("--skip-bench", action="store_true",
                   help="build engine only, skip benchmark_engine (avoids random-data "
                        "OOB on engines with index-typed inputs, e.g. e2e voxel_coords)")
    return p.parse_args()


def main():
    args = parse_args()
    # Stash mixed precision substrings as module-global so build_engine sees them
    if args.mixed_int8_substr:
        globals()["_MIXED_INT8_SUBSTRS"] = [s.strip() for s in args.mixed_int8_substr.split(",")]
    if args.mixed_fp16_substr:
        globals()["_MIXED_FP16_SUBSTRS"] = [s.strip() for s in args.mixed_fp16_substr.split(",")]
    if args.w_only:
        globals()["_W_ONLY"] = True
    if args.tactic:
        globals()["_TACTIC_MODE"] = args.tactic
    if args.builder_opt_level is not None:
        if not 0 <= args.builder_opt_level <= 5:
            raise ValueError(f"--builder-opt-level must be in [0,5], got {args.builder_opt_level}")
        globals()["_BUILDER_OPT_LEVEL"] = args.builder_opt_level
    if not args.skip_build:
        # Support multi-input calibration data via --calib-multi name:path
        calib_arg = args.calib_data
        if args.calib_multi:
            calib_arg = {}
            for spec in args.calib_multi:
                nm, p = spec.split(":", 1)
                calib_arg[nm.strip()] = p.strip()
        engine_path, build_secs = build_engine(
            args.onnx, args.precision, args.engine,
            workspace_mb=args.workspace_mb,
            calib_npy=calib_arg, calib_cache=args.calib_cache,
            calibrator_kind=args.calibrator,
        )
    else:
        engine_path = args.engine
        build_secs = None

    if args.skip_bench:
        stats = {
            "precision": args.precision,
            "onnx": args.onnx,
            "engine": args.engine,
            "build_secs": build_secs,
            "engine_size_mb": Path(args.engine).stat().st_size / 1e6
            if Path(args.engine).exists() else None,
            "skipped_bench": True,
        }
    else:
        shape = tuple(int(x) for x in args.input_shape.split(","))
        extra: dict[str, tuple] = {}
        for spec in args.extra_input_shape:
            name, dims = spec.split(":", 1)
            extra[name.strip()] = tuple(int(x) for x in dims.split(","))
        stats = benchmark_engine(engine_path, input_shape=shape,
                                 extra_input_shapes=extra,
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
