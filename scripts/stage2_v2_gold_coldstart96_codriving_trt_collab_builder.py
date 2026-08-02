#!/usr/bin/env python3
"""Build TensorRT collab engines for v2 CoDriving gold AP evaluation."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SPATIAL_SHAPE = (2, 64, 256, 512)
TMAT_SHAPE = (1, 2, 2, 4, 4)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def engine_path(out_root: Path, width: str, mode: str) -> Path:
    return out_root / width / f"collab_{width}_{mode}.engine"


def build_plan(
    onnx: Path,
    out_root: Path,
    width: str,
    modes: Iterable[str],
    calib_data: Path | None,
    workspace_mb: int,
) -> list[dict[str, Any]]:
    jobs = []
    for mode in modes:
        if mode not in {"fp16", "int8"}:
            raise ValueError(f"unsupported mode: {mode}")
        jobs.append(
            {
                "mode": mode,
                "width": width,
                "onnx": str(onnx),
                "engine": str(engine_path(out_root, width, mode)),
                "calib_data": str(calib_data) if mode == "int8" and calib_data is not None else None,
                "workspace_mb": int(workspace_mb),
            }
        )
    return jobs


def _build_engine(job: dict[str, Any], skip_sanity: bool = False) -> dict[str, Any]:
    import numpy as np
    import tensorrt as trt
    import torch

    logger = trt.Logger(trt.Logger.INFO)
    mode = job["mode"]
    onnx = Path(job["onnx"])
    output = Path(job["engine"])
    output.parent.mkdir(parents=True, exist_ok=True)
    if not onnx.is_file():
        raise FileNotFoundError(onnx)

    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx.read_bytes()):
        errors = [parser.get_error(i).desc() for i in range(parser.num_errors)]
        raise RuntimeError(f"ONNX parse failed: {errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(job["workspace_mb"]) * 1024 * 1024)
    if mode == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif mode == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16)
        calib_data = job.get("calib_data")
        if not calib_data:
            raise ValueError("INT8 engine build requires --calib-data")
        calib_path = Path(calib_data)
        if not calib_path.is_file():
            raise FileNotFoundError(calib_path)
        config.int8_calibrator = CollabInt8Calibrator(np.load(str(calib_path)), output.with_suffix(".calib.cache"))

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT build_serialized_network returned None")
    output.write_bytes(serialized)
    result = {
        "mode": mode,
        "width": job["width"],
        "onnx": str(onnx),
        "engine": str(output),
        "engine_size_mb": output.stat().st_size / 1e6,
        "status": "success",
    }
    if not skip_sanity:
        result["sanity"] = sanity_check(str(output), logger=logger, torch=torch, trt=trt)
    return result


class CollabInt8Calibrator:
    def __new__(cls, data, cache_file: Path):
        import tensorrt as trt
        import torch

        class _Calibrator(trt.IInt8MinMaxCalibrator):
            def __init__(self):
                super().__init__()
                self.spatial_data = data["spatial_features"]
                self.tmat_data = data["pairwise_t_matrix"]
                self.cache_file = cache_file
                self.idx = 0
                self.spatial_buf = torch.empty(SPATIAL_SHAPE, dtype=torch.float32, device="cuda").contiguous()
                self.tmat_buf = torch.empty(TMAT_SHAPE, dtype=torch.float32, device="cuda").contiguous()

            def get_batch_size(self):
                return 1

            def get_batch(self, names):
                if self.idx >= len(self.spatial_data):
                    return None
                self.spatial_buf.copy_(torch.from_numpy(self.spatial_data[self.idx]).float())
                self.tmat_buf.copy_(torch.from_numpy(self.tmat_data[self.idx]).float())
                self.idx += 1
                return [int(self.spatial_buf.data_ptr()), int(self.tmat_buf.data_ptr())]

            def read_calibration_cache(self):
                return self.cache_file.read_bytes() if self.cache_file.is_file() else None

            def write_calibration_cache(self, cache):
                self.cache_file.write_bytes(cache)

        return _Calibrator()


def sanity_check(engine: str, *, logger, torch, trt) -> dict[str, Any]:
    runtime = trt.Runtime(logger)
    trt_engine = runtime.deserialize_cuda_engine(Path(engine).read_bytes())
    if trt_engine is None:
        raise RuntimeError(f"failed to deserialize engine: {engine}")
    ctx = trt_engine.create_execution_context()
    if ctx is None:
        raise RuntimeError(f"failed to create execution context: {engine}")

    input_names = [
        trt_engine.get_tensor_name(i)
        for i in range(trt_engine.num_io_tensors)
        if trt_engine.get_tensor_mode(trt_engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT
    ]
    output_names = [
        trt_engine.get_tensor_name(i)
        for i in range(trt_engine.num_io_tensors)
        if trt_engine.get_tensor_mode(trt_engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT
    ]
    spatial_name = next(name for name in input_names if "spatial" in name.lower())
    tmat_name = next(name for name in input_names if "pairwise" in name.lower() or "t_matrix" in name.lower())
    ctx.set_input_shape(spatial_name, SPATIAL_SHAPE)
    ctx.set_input_shape(tmat_name, TMAT_SHAPE)
    buffers = {
        spatial_name: torch.zeros(SPATIAL_SHAPE, dtype=torch.float32, device="cuda").contiguous(),
        tmat_name: torch.zeros(TMAT_SHAPE, dtype=torch.float32, device="cuda").contiguous(),
    }
    buffers[tmat_name][0, :, :, 0, 0] = 1.0
    buffers[tmat_name][0, :, :, 1, 1] = 1.0
    buffers[tmat_name][0, :, :, 2, 2] = 1.0
    buffers[tmat_name][0, :, :, 3, 3] = 1.0
    for name in output_names:
        buffers[name] = torch.empty(tuple(ctx.get_tensor_shape(name)), dtype=torch.float32, device="cuda")
    for name, tensor in buffers.items():
        ctx.set_tensor_address(name, int(tensor.data_ptr()))
    stream = torch.cuda.current_stream().cuda_stream
    ok = bool(ctx.execute_async_v3(stream))
    torch.cuda.synchronize()
    return {
        "execute_async_v3": ok,
        "inputs": input_names,
        "outputs": {name: list(buffers[name].shape) for name in output_names},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--width", required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--mode", choices=["fp16", "int8", "both"], default="both")
    parser.add_argument("--calib-data", type=Path, default=None)
    parser.add_argument("--workspace-mb", type=int, default=4096)
    parser.add_argument("--skip-sanity", action="store_true")
    parser.add_argument("--summary", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    modes = ["fp16", "int8"] if args.mode == "both" else [args.mode]
    jobs = build_plan(args.onnx, args.out_root, args.width, modes, args.calib_data, args.workspace_mb)
    results = []
    for job in jobs:
        results.append(_build_engine(job, skip_sanity=args.skip_sanity))
    payload = {
        "schema": "v2_gold_coldstart_96_codriving_trt_collab_builder_v1",
        "created_at_utc": utc_now(),
        "results": results,
    }
    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
