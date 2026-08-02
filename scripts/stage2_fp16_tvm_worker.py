#!/usr/bin/env python3
"""Run one FP16 Relax VM backbone/subnet module from a JSON worker request."""

from __future__ import annotations

import argparse
import json
import re
import traceback
from pathlib import Path
from typing import Any

import numpy as np


FP16_TVM_WORKER_RESPONSE_SCHEMA = "stage2_fp16_tvm_worker_response_v1"
SUPPORTED_ACTIVATION_DTYPES = ("float16", "float32")


def sanitize_name(name: str, fallback: str) -> str:
    text = re.sub(r"[^0-9A-Za-z_]", "_", name or fallback)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text or text[0].isdigit():
        text = f"{fallback}_{text}" if text else fallback
    return text[:160]


def output_filename_for_arg(arg_name: str) -> str:
    return f"out_{sanitize_name(arg_name, 'tensor')}.npy"


def build_output_record(*, arg_name: str, output_path: Path, array: np.ndarray) -> dict[str, Any]:
    return {
        "arg_name": str(arg_name),
        "path": str(output_path),
        "shape": [int(dim) for dim in array.shape],
        "dtype": str(array.dtype),
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _require_path(request: dict[str, Any], key: str) -> Path:
    value = request.get(key)
    if not value:
        raise ValueError(f"worker request missing path field: {key}")
    return Path(str(value))


def resolve_activation_dtype(request: dict[str, Any]) -> str:
    dtype = str(request.get("activation_dtype", "float16")).strip().lower()
    if dtype not in SUPPORTED_ACTIVATION_DTYPES:
        raise ValueError(
            f"unsupported activation_dtype {dtype!r}; expected one of {SUPPORTED_ACTIVATION_DTYPES}"
        )
    return dtype


def validate_request(request: dict[str, Any]) -> None:
    artifact_path = _require_path(request, "artifact_path")
    activation_path = _require_path(request, "activation_npy_path")
    output_dir = _require_path(request, "output_dir")
    if not artifact_path.exists():
        raise FileNotFoundError(f"missing artifact_path: {artifact_path}")
    if not activation_path.exists():
        raise FileNotFoundError(f"missing activation_npy_path: {activation_path}")
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"output_dir is not a directory: {output_dir}")
    resolve_activation_dtype(request)


def make_tvm_array(tvm: Any, value: np.ndarray, dev: Any) -> Any:
    tensor_ctor = getattr(tvm.runtime, "tensor", None)
    if tensor_ctor is not None:
        return tensor_ctor(value, device=dev)
    return tvm.nd.array(value, dev)


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "numpy"):
        return value.numpy()
    return value.asnumpy()


def _iter_outputs(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple)):
        return list(value)
    try:
        return [value[idx] for idx in range(len(value))]
    except TypeError:
        return [value]


def _assert_expected_outputs(
    outputs: list[np.ndarray],
    output_names: list[str],
    expected_shapes: dict[str, Any],
) -> None:
    for idx, array in enumerate(outputs):
        name = output_names[idx] if idx < len(output_names) else f"output{idx}"
        expected = expected_shapes.get(name) or expected_shapes.get(str(idx))
        if expected is None:
            continue
        expected_shape = tuple(int(dim) for dim in expected)
        if tuple(array.shape) != expected_shape:
            raise ValueError(f"output shape mismatch for {name}: {array.shape} != {expected_shape}")


def _load_context(request: dict[str, Any], _cache: dict[str, Any] = {}) -> dict[str, Any]:
    """Load and cache a Relax VM executable for CLI or persistent server mode."""
    import tvm
    from tvm import relax

    gpu = int(request.get("gpu", 0))
    artifact_path = str(_require_path(request, "artifact_path"))
    key = (gpu, artifact_path)
    ctx = _cache.get("ctx")
    if ctx is None or ctx["key"] != key:
        dev = tvm.cuda(gpu)
        lib = tvm.runtime.load_module(artifact_path)
        vm = relax.VirtualMachine(lib, dev)
        ctx = {"key": key, "tvm": tvm, "dev": dev, "vm": vm}
        _cache["ctx"] = ctx
    return ctx


def _execute(ctx: dict[str, Any], request: dict[str, Any], request_json: Path) -> dict[str, Any]:
    validate_request(request)
    tvm = ctx["tvm"]
    dev = ctx["dev"]
    vm = ctx["vm"]

    output_dir = _require_path(request, "output_dir")
    output_dir.mkdir(parents=True, exist_ok=True)

    activation_dtype = resolve_activation_dtype(request)
    activation = np.load(_require_path(request, "activation_npy_path"))
    activation = np.ascontiguousarray(activation.astype(np.dtype(activation_dtype)))
    result = vm["main"](make_tvm_array(tvm, activation, dev))
    dev.sync()

    output_names = [str(item) for item in request.get("output_names", ["output0", "output1", "output2"])]
    arrays = [np.asarray(_as_numpy(item)) for item in _iter_outputs(result)]
    _assert_expected_outputs(arrays, output_names, dict(request.get("expected_output_shapes") or {}))

    output_records: list[dict[str, Any]] = []
    for idx, array in enumerate(arrays):
        name = output_names[idx] if idx < len(output_names) else f"output{idx}"
        output_path = output_dir / output_filename_for_arg(name)
        np.save(output_path, array)
        output_records.append(build_output_record(arg_name=name, output_path=output_path, array=array))

    response = {
        "schema": FP16_TVM_WORKER_RESPONSE_SCHEMA,
        "status": "success",
        "label": str(request.get("label", "")),
        "run_id": str(request.get("run_id", "")),
        "request_json": str(request_json),
        "artifact_path": str(_require_path(request, "artifact_path")),
        "activation_npy_path": str(_require_path(request, "activation_npy_path")),
        "activation_dtype": activation_dtype,
        "outputs": output_records,
    }
    _write_json(output_dir / "fp16_tvm_worker_response.json", response)
    return response


def _failure_output_path(request_json: Path) -> Path:
    try:
        request = _read_json(request_json)
        if request.get("output_dir"):
            return Path(str(request["output_dir"])) / "fp16_tvm_worker_response.json"
    except Exception:
        pass
    return request_json.parent / "fp16_tvm_worker_response.json"


def run_worker(request_json: Path) -> dict[str, Any]:
    request = _read_json(request_json)
    validate_request(request)
    ctx = _load_context(request)
    return _execute(ctx, request, request_json)


def run_server() -> int:
    import sys

    sys.stdout.write("READY\n")
    sys.stdout.flush()
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        if line == "STOP":
            break
        request_json = Path(line)
        try:
            request = _read_json(request_json)
            validate_request(request)
            ctx = _load_context(request)
            _execute(ctx, request, request_json)
            sys.stdout.write(f"DONE {_failure_output_path(request_json)}\n")
        except Exception as exc:  # noqa: BLE001 - report and keep server alive
            failure = {
                "schema": FP16_TVM_WORKER_RESPONSE_SCHEMA,
                "status": "failed",
                "failure_reason": str(exc),
                "traceback": traceback.format_exc(),
                "request_json": str(request_json),
            }
            try:
                _write_json(_failure_output_path(request_json), failure)
            except Exception:
                pass
            sys.stdout.write(f"FAIL {_failure_output_path(request_json)}\n")
        sys.stdout.flush()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request-json", type=Path, default=None)
    parser.add_argument(
        "--server",
        action="store_true",
        help="persistent server mode: read request paths from stdin",
    )
    args = parser.parse_args(argv)

    if args.server:
        return run_server()

    if args.request_json is None:
        parser.error("--request-json is required unless --server is set")

    try:
        run_worker(args.request_json)
        return 0
    except Exception as exc:
        failure = {
            "schema": FP16_TVM_WORKER_RESPONSE_SCHEMA,
            "status": "failed",
            "failure_reason": str(exc),
            "traceback": traceback.format_exc(),
            "request_json": str(args.request_json),
        }
        _write_json(_failure_output_path(args.request_json), failure)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
