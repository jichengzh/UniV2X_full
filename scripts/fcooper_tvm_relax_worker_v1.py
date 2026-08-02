#!/usr/bin/env python3
"""Execute one compiled F-Cooper Relax VM through persistent mmap buffers."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ALLOWED_RUNTIME_ROLES = {"graph_input", "weight_input", "bias_input", "graph_output"}
DYNAMIC_QUANT_KEYS = ("dynamic", "dynamic_per_chunk", "per_chunk", "allow_fallback")
GRAPH_INPUT_NAME = "spatial_features"
ARTIFACT_BATCH = 5


def parse_shape(value: str) -> tuple[int, ...]:
    shape = tuple(int(part) for part in value.split(","))
    if not shape or any(item <= 0 for item in shape):
        raise argparse.ArgumentTypeError("shape must contain positive dimensions")
    return shape


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def single_output(value: Any) -> np.ndarray:
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    if hasattr(value, "asnumpy"):
        return np.asarray(value.asnumpy())
    try:
        outputs = [value[index] for index in range(len(value))]
    except TypeError:
        return np.asarray(value)
    if len(outputs) != 1:
        raise ValueError(f"Relax VM returned {len(outputs)} outputs; expected one")
    return single_output(outputs[0])


def quant_params(payload: Mapping[str, Any], name: str) -> tuple[float, int]:
    params = payload.get("params", payload)
    if not isinstance(params, Mapping) or not isinstance(params.get(name), Mapping):
        raise ValueError(f"missing quant params for {name}")
    item = params[name]
    source = str(item.get("source") or "").lower().replace("-", "_")
    if any(token in source for token in ("dynamic", "per_chunk", "fallback")) or any(
        bool(item.get(key)) for key in DYNAMIC_QUANT_KEYS
    ):
        raise ValueError(f"dynamic quantization is forbidden for {name}")
    scale = float(item["scale"])
    zero_point = int(item["zero_point"])
    if not math.isfinite(scale) or scale <= 0 or not 0 <= zero_point <= 255:
        raise ValueError(f"invalid quant params for {name}")
    return scale, zero_point


def validate_runtime_tensor(payload: Mapping[str, Any]) -> dict[str, Any]:
    name = str(payload.get("arg_name") or "")
    role = str(payload.get("role") or "")
    if not name:
        raise ValueError("runtime tensor is missing arg_name")
    if role not in ALLOWED_RUNTIME_ROLES:
        raise ValueError(f"unsupported runtime role for {name}: {role}")
    try:
        shape = tuple(int(value) for value in payload["shape"])
        dtype = np.dtype(str(payload["dtype"])).name
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid runtime tensor contract for {name}") from error
    if not shape or any(value <= 0 for value in shape):
        raise ValueError(f"runtime tensor {name} requires positive static dimensions")
    return {
        **dict(payload),
        "arg_name": name,
        "role": role,
        "shape": shape,
        "dtype": dtype,
    }


def make_tvm_array(tvm: Any, device: Any, value: np.ndarray) -> Any:
    constructor = getattr(tvm.runtime, "tensor", None)
    if constructor is not None:
        return constructor(value, device=device)
    return tvm.nd.array(value, device)


def load_compiled_module(tvm: Any, artifact: Path) -> Any:
    artifact = Path(artifact)
    if artifact.suffix == ".vmexec":
        loader = tvm.get_global_func("ffi.Module.load_from_file.so")
        return loader(str(artifact), "so")
    return tvm.runtime.load_module(str(artifact))


def runtime_weight_archive_key(
    item: Mapping[str, Any],
    *,
    declared: Mapping[str, Any],
    archive_keys: set[str],
) -> str:
    name = str(item["arg_name"])
    explicit = declared.get(name)
    if explicit is not None:
        key = str(explicit)
        if key not in archive_keys:
            raise ValueError(f"runtime weight archive missing declared key for {name}: {key}")
        return key
    for candidate in (name, item.get("initializer_name")):
        if candidate is not None and str(candidate) in archive_keys:
            return str(candidate)
    raise ValueError(f"runtime weight archive missing binding for {name}")


class Worker:
    def __init__(self, args: argparse.Namespace) -> None:
        import tvm
        from tvm import relax

        self.args = args
        self.tvm = tvm
        self.device = tvm.cuda(args.gpu_id)
        if hasattr(self.device, "exist") and not self.device.exist:
            raise RuntimeError(f"TVM CUDA device {args.gpu_id} is unavailable")
        library = load_compiled_module(tvm, args.artifact)
        self.main = relax.VirtualMachine(library, self.device)["main"]
        self.source = np.memmap(
            args.input_buffer,
            mode="r+",
            dtype=np.dtype(args.input_dtype),
            shape=args.input_shape,
        )
        self.target = np.memmap(
            args.output_buffer,
            mode="r+",
            dtype=np.dtype(args.output_dtype),
            shape=args.output_shape,
        )
        self.runtime_inputs: tuple[tuple[str, np.ndarray | None], ...] = ()
        self.input_quant: tuple[float, int] | None = None
        self.output_quant: tuple[float, int] | None = None
        if args.mode == "int8":
            self._load_int8_contract()

    def _load_int8_contract(self) -> None:
        result = load_json(self.args.route_result)
        plan = result.get("runtime_arg_plan")
        if not isinstance(plan, list):
            raise ValueError("INT8 worker requires runtime_arg_plan")
        tensors = [validate_runtime_tensor(item) for item in plan]
        graph_inputs = [item for item in tensors if item["role"] == "graph_input"]
        graph_outputs = [item for item in tensors if item["role"] == "graph_output"]
        if len(graph_inputs) != 1 or len(graph_outputs) != 1:
            raise ValueError("INT8 worker requires one graph input and output")
        graph_input_tensor = graph_inputs[0]
        graph_output_tensor = graph_outputs[0]
        graph_input = graph_input_tensor["arg_name"]
        graph_output = graph_output_tensor["arg_name"]
        if graph_input != GRAPH_INPUT_NAME:
            raise ValueError(f"graph input must be {GRAPH_INPUT_NAME}, got {graph_input}")
        for tensor in (graph_input_tensor, graph_output_tensor):
            if tensor["dtype"] != "uint8":
                raise ValueError("INT8 graph input and output must be uint8")
            if len(tensor["shape"]) != 4 or tensor["shape"][0] != ARTIFACT_BATCH:
                raise ValueError("INT8 graph boundary must be rank-4 with batch 5")
        if graph_input_tensor["shape"] != self.args.input_shape:
            raise ValueError("worker input shape disagrees with runtime_arg_plan")
        if graph_output_tensor["shape"] != self.args.output_shape:
            raise ValueError("worker output shape disagrees with runtime_arg_plan")
        quant = load_json(self.args.quant_contract)
        self.input_quant = quant_params(quant, graph_input)
        self.output_quant = quant_params(quant, graph_output)
        declared = result.get("runtime_weight_archive_keys") or {}
        runtime_inputs: list[tuple[str, np.ndarray | None]] = []
        with np.load(self.args.runtime_weights, allow_pickle=False) as archive:
            archive_keys = set(archive.files)
            for item in tensors:
                role = item["role"]
                if role == "graph_output":
                    continue
                if role == "graph_input":
                    runtime_inputs.append((role, None))
                    continue
                name = str(item["arg_name"])
                key = runtime_weight_archive_key(
                    item,
                    declared=declared,
                    archive_keys=archive_keys,
                )
                value = np.ascontiguousarray(archive[key])
                expected_shape = tuple(int(v) for v in item["shape"])
                expected_dtype = np.dtype(str(item["dtype"]))
                if tuple(value.shape) != expected_shape or value.dtype != expected_dtype:
                    raise ValueError(f"runtime weight contract mismatch for {name}")
                runtime_inputs.append((role, value))
        self.runtime_inputs = tuple(runtime_inputs)

    def execute(self) -> None:
        host_input = np.ascontiguousarray(self.source)
        if self.args.mode == "int8":
            assert self.input_quant is not None and self.output_quant is not None
            scale, zero_point = self.input_quant
            graph_input = np.clip(
                np.rint(host_input.astype(np.float32) / scale) + zero_point,
                0,
                255,
            ).astype(np.uint8)
            inputs = [
                graph_input if role == "graph_input" else value
                for role, value in self.runtime_inputs
            ]
        else:
            inputs = [host_input]
        returned = self.main(
            *(make_tvm_array(self.tvm, self.device, value) for value in inputs)
        )
        self.device.sync()
        output = np.ascontiguousarray(single_output(returned))
        if self.args.mode == "int8":
            if output.dtype != np.uint8:
                raise ValueError(f"INT8 graph output dtype drift: {output.dtype}")
            scale, zero_point = self.output_quant
            output = (output.astype(np.float32) - zero_point) * scale
        if tuple(output.shape) != self.args.output_shape:
            raise ValueError(
                f"output shape drift: {tuple(output.shape)} != {self.args.output_shape}"
            )
        if output.dtype != np.dtype(self.args.output_dtype):
            raise ValueError(
                f"output dtype drift: {output.dtype} != {self.args.output_dtype}"
            )
        self.target[:] = output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("fp16", "fp32", "int8"), required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--input-shape", type=parse_shape, required=True)
    parser.add_argument("--output-shape", type=parse_shape, required=True)
    parser.add_argument("--input-dtype", required=True)
    parser.add_argument("--output-dtype", required=True)
    parser.add_argument("--input-buffer", type=Path, required=True)
    parser.add_argument("--output-buffer", type=Path, required=True)
    parser.add_argument("--route-result", type=Path)
    parser.add_argument("--runtime-weights", type=Path)
    parser.add_argument("--quant-contract", type=Path)
    parser.add_argument("--gpu-id", type=int, default=0)
    args = parser.parse_args()
    if args.mode == "int8" and not all(
        (args.route_result, args.runtime_weights, args.quant_contract)
    ):
        parser.error("INT8 mode requires route result, runtime weights, and quant contract")
    return args


def emit(payload: Mapping[str, Any]) -> None:
    print(json.dumps(dict(payload), sort_keys=True), flush=True)


def main() -> int:
    try:
        worker = Worker(parse_args())
        emit({"status": "ready"})
        for command in sys.stdin:
            action = command.strip()
            if action == "close":
                emit({"status": "closed"})
                return 0
            if action != "run":
                raise ValueError(f"unsupported worker command: {action}")
            worker.execute()
            emit({"status": "success"})
        return 0
    except Exception as error:
        emit({"status": "failure", "error": f"{type(error).__name__}: {error}"})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
