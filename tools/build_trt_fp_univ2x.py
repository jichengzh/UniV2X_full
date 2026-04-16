"""
build_trt_fp_univ2x.py — 最简 FP32/FP16 TRT engine 构建器（用于剪枝模型集成验证）

区别于 build_trt_int8_univ2x.py：不做 INT8 校准，只构建 FP32 或 FP16 engine。
用途：Phase C.0.4 验证剪枝后模型能走通 ONNX → TRT engine 全链路。

用法：
    python tools/build_trt_fp_univ2x.py \\
        --onnx onnx/univ2x_ego_bev_pruned_60_50.onnx \\
        --out  trt_engines/univ2x_ego_bev_pruned_60_fp32.trt \\
        --plugin plugins/build/libuniv2x_plugins.so \\
        --precision fp32
"""
from __future__ import annotations

import argparse
import ctypes
import os
import sys
import time


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build FP32/FP16 TRT engine (no quantization)")
    p.add_argument("--onnx", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--plugin", default="plugins/build/libuniv2x_plugins.so")
    p.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    p.add_argument("--workspace-mb", type=int, default=4096,
                   help="TRT workspace memory in MB (default 4096)")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    import tensorrt as trt

    # Load UniV2X custom plugin (MSDAPlugin, RotatePlugin, etc.)
    plugin_path = os.path.abspath(args.plugin)
    if os.path.exists(plugin_path):
        ctypes.CDLL(plugin_path, mode=ctypes.RTLD_GLOBAL)
        print(f"[plugin] loaded {plugin_path}")
    else:
        print(f"[plugin] WARN: {plugin_path} not found — MSDAPlugin will fail")

    # TRT logger
    log_severity = trt.Logger.VERBOSE if args.verbose else trt.Logger.WARNING
    logger = trt.Logger(log_severity)
    trt.init_libnvinfer_plugins(logger, "")

    # Build TRT engine
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    with open(args.onnx, "rb") as f:
        ok = parser.parse(f.read())
    if not ok:
        for i in range(parser.num_errors):
            print(f"[onnx] ERROR {i}: {parser.get_error(i)}")
        return 1

    print(f"[onnx] parsed {args.onnx}")
    print(f"[onnx] inputs: {[network.get_input(i).name for i in range(network.num_inputs)]}")
    print(f"[onnx] outputs: {[network.get_output(i).name for i in range(network.num_outputs)]}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, args.workspace_mb * (1 << 20)
    )
    if args.precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
        print("[precision] FP16")
    else:
        print("[precision] FP32")

    # Build
    print(f"[build] building engine (workspace={args.workspace_mb} MB) …")
    t0 = time.time()
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        print("[build] FAILED — engine is None")
        return 2
    dt = time.time() - t0
    engine_raw = bytes(engine_bytes)
    print(f"[build] OK in {dt:.1f}s, engine size: {len(engine_raw)/1e6:.2f} MB")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as f:
        f.write(engine_raw)
    print(f"[out] engine saved -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
