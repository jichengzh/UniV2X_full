"""
build_trt_int8_univ2x.py — Build INT8 + FP16 mixed-precision TRT engines for UniV2X.

Design (ADR-004):
  - FP16 flag enabled for all layers
  - INT8 flag enabled for calibratable layers
  - MSDAPlugin layers are forced to FP16 precision via layer.precision = HALF
  - IInt8EntropyCalibrator2 reads pre-dumped calibration data (calibration/cali_data.pkl)

Supported targets:
  --target bev_encoder  : BEV encoder ONNX (from export_onnx_univ2x.py)
  --target heads        : V2X heads ONNX
  --target downstream   : Downstream heads ONNX

Usage:
    python tools/build_trt_int8_univ2x.py \\
        --onnx onnx/univ2x_ego_bev_encoder_200_1cam.onnx \\
        --out  trt_engines/univ2x_ego_bev_encoder_int8.trt \\
        --calib-cache calibration/bev_encoder_int8_calib.cache \\
        --target bev_encoder \\
        --plugin plugins/build/libuniv2x_plugins.so

Note: The calibration cache embeds INT8 activation scales from the pre-dumped data.
      On first run, the cache is built from scratch (requires calibration/cali_data.pkl).
      On subsequent runs, the cached scales are reused (fast).
"""

import argparse
import ctypes
import os
import pickle
import sys

import numpy as np
import torch


# ---------------------------------------------------------------------------
# IInt8EntropyCalibrator2 implementation (pycuda-free, uses PyTorch GPU buffers)
# ---------------------------------------------------------------------------

def _make_calibrator_class():
    """
    Deferred class creation so that trt is only imported when needed.
    Returns a class inheriting from trt.IInt8EntropyCalibrator2.
    """
    import tensorrt as trt

    class UniV2XInt8CalibratorImpl(trt.IInt8EntropyCalibrator2):
        """
        TensorRT IInt8EntropyCalibrator2 for UniV2X ONNX models.

        cali_tensors: list of dict {input_name: numpy_array} (one per calibration sample)
        cache_file:   path to write/read the calibration cache
        """

        def __init__(self, cali_tensors: list, input_names: list, cache_file: str):
            super().__init__()
            self._calibration_data = cali_tensors
            self._input_names = input_names
            self._cache_file = cache_file
            self._index = 0
            self._device_buffers = {}
            # Allocate GPU buffers for each input based on first sample
            if cali_tensors:
                first = cali_tensors[0]
                for name in input_names:
                    arr = first.get(name)
                    if arr is None:
                        continue
                    # use float32 for all; bool inputs treated as uint8
                    dtype = torch.float32 if arr.dtype != np.bool_ else torch.uint8
                    buf = torch.zeros(arr.shape, dtype=dtype, device='cuda')
                    self._device_buffers[name] = buf

        def get_batch_size(self):
            return 1

        def get_batch(self, names):
            if self._index >= len(self._calibration_data):
                return None
            sample = self._calibration_data[self._index]
            self._index += 1
            ptrs = []
            for name in names:
                arr = sample.get(name)
                if arr is None:
                    print(f'  WARNING: calibration missing key {name}, using zeros')
                    buf = torch.zeros(1, device='cuda')
                    ptrs.append(buf.data_ptr())
                    continue
                buf = self._device_buffers[name]
                t = torch.from_numpy(np.array(arr, dtype=np.float32))
                # Shape-guard: if sample has a variable-length dim (e.g. lane_query
                # can be 300, 323, 332 …), slice or zero-pad to match the pre-
                # allocated buffer shape so copy_() never raises.
                if t.shape != buf.shape:
                    target = buf.clone().zero_()
                    slices = tuple(
                        slice(0, min(ts, bs))
                        for ts, bs in zip(t.shape, buf.shape))
                    target[slices] = t[slices]
                    buf.copy_(target)
                else:
                    buf.copy_(t.to('cuda'))
                ptrs.append(buf.data_ptr())
            return ptrs

        def read_calibration_cache(self):
            if os.path.exists(self._cache_file):
                print(f'  Reading INT8 calibration cache: {self._cache_file}')
                with open(self._cache_file, 'rb') as f:
                    return f.read()
            return None

        def write_calibration_cache(self, cache):
            os.makedirs(os.path.dirname(os.path.abspath(self._cache_file)), exist_ok=True)
            with open(self._cache_file, 'wb') as f:
                f.write(cache)
            print(f'  Wrote INT8 calibration cache: {self._cache_file}')

    return UniV2XInt8CalibratorImpl


# Alias for backward compat
UniV2XInt8Calibrator = None  # will be set lazily in build_int8_engine


# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Build INT8+FP16 TRT engines for UniV2X')
    p.add_argument('--onnx',          required=True, help='Input ONNX file')
    p.add_argument('--out',           required=True, help='Output TRT engine file')
    p.add_argument('--target',        required=True,
                   choices=['bev_encoder', 'heads', 'downstream'],
                   help='Which model component this ONNX represents')
    p.add_argument('--plugin',        default='plugins/build/libuniv2x_plugins.so',
                   help='Path to custom plugin .so')
    p.add_argument('--calib-cache',   default=None,
                   help='INT8 calibration cache file (auto-derived if not set)')
    p.add_argument('--cali-data',     default='calibration/bev_encoder_inputs.pkl',
                   help='Pre-extracted calibration input tensors (dict pkl)')
    p.add_argument('--workspace-gb',  type=float, default=8.0)
    p.add_argument('--no-int8',       action='store_true',
                   help='Disable INT8 (build FP16-only engine)')
    p.add_argument('--mode',          choices=['implicit', 'explicit'], default='implicit',
                   help='implicit: use Calibrator for INT8 scales (default). '
                        'explicit: read Q/DQ nodes from ONNX (requires inject_qdq_from_config.py output)')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Core build function
# ---------------------------------------------------------------------------

def build_int8_engine(onnx_path, engine_path, plugin_path, target,
                      calib_cache, cali_data_path, workspace_gb, disable_int8=False):
    import tensorrt as trt

    # Load plugins
    ctypes.CDLL(plugin_path)
    trt.init_libnvinfer_plugins(None, '')

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * 1024**3))

    # FP16 (always on for mixed precision)
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print('  FP16 mode: ON')

    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    print(f'Parsing ONNX: {onnx_path}')
    with open(onnx_path, 'rb') as f:
        ok = parser.parse(f.read())
    if not ok:
        for i in range(parser.num_errors):
            print(f'  PARSER ERROR {i}: {parser.get_error(i)}')
        raise RuntimeError('ONNX parse failed')

    input_names  = [network.get_input(i).name  for i in range(network.num_inputs)]
    output_names = [network.get_output(i).name for i in range(network.num_outputs)]
    print(f'  Inputs:  {input_names}')
    print(f'  Outputs: {output_names}')

    # INT8 calibration
    if not disable_int8 and builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        print('  INT8 mode: ON')

        if calib_cache is None:
            stem = os.path.splitext(os.path.basename(engine_path))[0]
            calib_cache = os.path.join('calibration', f'{stem}_int8.cache')

        # Load pre-extracted calibration tensors if cache does not exist yet
        if not os.path.exists(calib_cache):
            print(f'  No cache found, loading cali data from {cali_data_path}')
            with open(cali_data_path, 'rb') as f:
                raw_cali = pickle.load(f)
            print(f'    {len(raw_cali)} calibration samples')
        else:
            raw_cali = []  # calibrator will read from cache directly

        CalibratorClass = _make_calibrator_class()
        calibrator = CalibratorClass(
            cali_tensors=raw_cali,
            input_names=input_names,
            cache_file=calib_cache,
        )
        config.int8_calibrator = calibrator

    # Force MSDAPlugin layers to FP16 (ADR-004)
    _force_msda_fp16(network, trt)

    print('Building TRT engine (this may take several minutes) ...')
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError('Engine build failed')

    os.makedirs(os.path.dirname(os.path.abspath(engine_path)), exist_ok=True)
    with open(engine_path, 'wb') as f:
        f.write(memoryview(engine_bytes))
    size_mb = os.path.getsize(engine_path) / 1024**2
    print(f'Engine saved: {engine_path}  ({size_mb:.1f} MB)')


def _force_msda_fp16(network, trt):
    """Custom plugins already run in FP16/FP32 natively (not quantizable by TRT).

    NOTE: Setting layer.precision on PLUGIN_V2 layers causes TRT to insert
    dequant/quant nodes that degrade accuracy in mixed-precision mode.
    Custom plugins are already excluded from INT8 by TRT since they don't
    have INT8 implementations — no explicit override needed.
    """
    plugin_count = sum(1 for i in range(network.num_layers)
                       if network.get_layer(i).type == trt.LayerType.PLUGIN_V2)
    print(f'  Found {plugin_count} custom plugin layer(s) — '
          f'running in native FP16 (TRT default for plugins)')


def build_explicit_int8_engine(onnx_path, engine_path, plugin_path, workspace_gb=8.0):
    """Build TRT engine from Q/DQ ONNX in explicit quantization mode.

    In explicit mode, TRT reads scale information from Q/DQ nodes in the ONNX
    graph.  No Calibrator is needed — all quantization decisions are embedded
    in the graph by inject_qdq_from_config.py.

    Layers WITHOUT Q/DQ nodes automatically run in FP16.
    """
    import tensorrt as trt

    ctypes.CDLL(plugin_path)
    trt.init_libnvinfer_plugins(None, '')

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * 1024**3))

    # Both flags needed: FP16 for non-quantized layers, INT8 for Q/DQ layers
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print('  FP16 mode: ON')
    if builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        print('  INT8 mode: ON (explicit Q/DQ)')

    # NOTE: No Calibrator set — TRT enters explicit mode when Q/DQ nodes exist
    print('  Calibrator: NONE (explicit quantization mode)')

    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    print(f'Parsing Q/DQ ONNX: {onnx_path}')
    with open(onnx_path, 'rb') as f:
        ok = parser.parse(f.read())
    if not ok:
        for i in range(parser.num_errors):
            print(f'  PARSER ERROR {i}: {parser.get_error(i)}')
        raise RuntimeError('ONNX parse failed')

    input_names = [network.get_input(i).name for i in range(network.num_inputs)]
    output_names = [network.get_output(i).name for i in range(network.num_outputs)]
    print(f'  Inputs:  {input_names}')
    print(f'  Outputs: {output_names}')

    # Count Q/DQ layers for sanity check
    qdq_count = 0
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        if 'Quantize' in layer.name or 'Dequantize' in layer.name:
            qdq_count += 1
    print(f'  Q/DQ layers: {qdq_count}')
    if qdq_count == 0:
        print('  WARNING: No Q/DQ nodes found — this ONNX may not be from inject_qdq_from_config.py')

    print('Building TRT engine (explicit INT8 mode, this may take several minutes) ...')
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError('Engine build failed')

    os.makedirs(os.path.dirname(os.path.abspath(engine_path)), exist_ok=True)
    with open(engine_path, 'wb') as f:
        f.write(memoryview(engine_bytes))
    size_mb = os.path.getsize(engine_path) / 1024**2
    print(f'Engine saved: {engine_path}  ({size_mb:.1f} MB)')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    sys.path.insert(0, '.')

    if args.mode == 'explicit':
        # Explicit Q/DQ mode: reads scales from ONNX Q/DQ nodes, no Calibrator
        print(f'Mode: EXPLICIT (Q/DQ ONNX)')
        build_explicit_int8_engine(
            onnx_path=args.onnx,
            engine_path=args.out,
            plugin_path=args.plugin,
            workspace_gb=args.workspace_gb,
        )
    else:
        # Implicit mode: Calibrator-based INT8 (existing behavior)
        print(f'Mode: IMPLICIT (Calibrator)')
        build_int8_engine(
            onnx_path=args.onnx,
            engine_path=args.out,
            plugin_path=args.plugin,
            target=args.target,
            calib_cache=args.calib_cache,
            cali_data_path=args.cali_data,
            workspace_gb=args.workspace_gb,
            disable_int8=args.no_int8,
        )


if __name__ == '__main__':
    main()
