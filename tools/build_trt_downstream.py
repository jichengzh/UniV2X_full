"""Build TRT engines for downstream heads (Stages D+E+F).

Usage
-----
# Step A — random weights, validate graph structure
python tools/build_trt_downstream.py \
    --onnx onnx/univ2x_ego_downstream_50_rand.onnx \
    --out  trt_engines/univ2x_ego_downstream_50_rand.trt

# Step B — real checkpoint, full accuracy check
python tools/build_trt_downstream.py \
    --onnx onnx/univ2x_ego_downstream.onnx \
    --out  trt_engines/univ2x_ego_downstream.trt
"""
import argparse
import ctypes
import os
import sys


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--onnx', required=True, help='Input ONNX file')
    p.add_argument('--out', required=True, help='Output TRT engine file')
    p.add_argument('--plugin', default='plugins/build/libuniv2x_plugins.so',
                   help='Path to custom plugin .so')
    p.add_argument('--workspace-gb', type=float, default=8.0,
                   help='Max workspace in GB (default 8)')
    p.add_argument('--fp16', action='store_true', help='Enable FP16 mode')
    return p.parse_args()


def build(onnx_path, engine_path, plugin_path, workspace_gb, fp16):
    import tensorrt as trt

    # Load custom plugins (MSDAPlugin, RotatePlugin, InversePlugin)
    ctypes.CDLL(plugin_path)
    trt.init_libnvinfer_plugins(None, '')

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, int(workspace_gb * 1024**3))

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print('FP16 mode enabled')

    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    print(f'Parsing {onnx_path} ...')
    with open(onnx_path, 'rb') as f:
        ok = parser.parse(f.read())
    if not ok:
        for i in range(parser.num_errors):
            print(f'  PARSER ERROR {i}: {parser.get_error(i)}')
        raise RuntimeError('ONNX parse failed')
    print(f'  Inputs:  {[network.get_input(i).name for i in range(network.num_inputs)]}')
    print(f'  Outputs: {[network.get_output(i).name for i in range(network.num_outputs)]}')

    print('Building TRT engine (this may take several minutes) ...')
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError('Engine build failed')

    os.makedirs(os.path.dirname(os.path.abspath(engine_path)), exist_ok=True)
    with open(engine_path, 'wb') as f:
        f.write(memoryview(engine_bytes))
    size_mb = os.path.getsize(engine_path) / 1024**2
    print(f'Engine saved: {engine_path}  ({size_mb:.1f} MB)')


def main():
    args = parse_args()
    build(args.onnx, args.out, args.plugin, args.workspace_gb, args.fp16)


if __name__ == '__main__':
    main()
