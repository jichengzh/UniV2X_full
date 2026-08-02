#!/usr/bin/env python3
"""Run one native INT8 TVM backbone module from a JSON worker request."""

from __future__ import annotations

import argparse
import json
import re
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from framework.stage2.native_int8_full_onnx import (
    TVM_WORKER_RESPONSE_SCHEMA,
    validate_tvm_worker_response,
)


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


def make_tvm_array(tvm: Any, value: np.ndarray, dev: Any) -> Any:
    tensor_ctor = getattr(tvm.runtime, "tensor", None)
    if tensor_ctor is not None:
        return tensor_ctor(value, device=dev)
    return tvm.nd.array(value, dev)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _require_path(request: dict[str, Any], key: str) -> Path:
    value = request.get(key)
    if not value:
        raise ValueError(f"worker request missing path field: {key}")
    return Path(str(value))


def _runtime_arg_plan(request: dict[str, Any]) -> list[dict[str, Any]]:
    plan = request.get("runtime_arg_plan")
    if not isinstance(plan, list) or not plan:
        raise ValueError("worker request runtime_arg_plan must be a non-empty list")
    return [dict(item) for item in plan]


def _execution_abi(request: dict[str, Any]) -> str | None:
    value = request.get("execution_abi")
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text not in {"direct_packed", "relax_vm_return"}:
        raise ValueError(f"unsupported execution_abi: {text}")
    return text


def _assert_array_contract(array: np.ndarray, item: dict[str, Any]) -> np.ndarray:
    expected_shape = tuple(int(dim) for dim in item["shape"])
    expected_dtype = np.dtype(str(item["dtype"]))
    if tuple(array.shape) != expected_shape:
        raise ValueError(f"array shape mismatch for {item['arg_name']}: {array.shape} != {expected_shape}")
    if array.dtype != expected_dtype:
        raise ValueError(f"array dtype mismatch for {item['arg_name']}: {array.dtype} != {expected_dtype}")
    return np.ascontiguousarray(array)


def _weight_archive_key(arg_name: str, request: dict[str, Any]) -> str:
    archive_keys = request.get("runtime_weight_archive_keys") or {}
    if isinstance(archive_keys, dict) and arg_name in archive_keys:
        return str(archive_keys[arg_name])
    return sanitize_name(arg_name, "weight")


def _as_numpy(tvm_array: Any) -> np.ndarray:
    if hasattr(tvm_array, "numpy"):
        return tvm_array.numpy()
    return tvm_array.asnumpy()


def _iter_outputs(value: Any) -> list[Any]:
    try:
        from tvm.runtime.container import ADT

        if isinstance(value, ADT):
            return [value[idx] for idx in range(len(value))]
    except Exception:
        pass
    if isinstance(value, (list, tuple)):
        return list(value)
    if not (hasattr(value, "numpy") or hasattr(value, "asnumpy")) and hasattr(value, "__len__") and hasattr(value, "__getitem__"):
        try:
            return [value[idx] for idx in range(len(value))]
        except Exception:
            pass
    return [value]


def _main_from_library(lib: Any) -> Any | None:
    try:
        return lib["main"]
    except Exception:
        return None


def _build_relax_vm(tvm: Any, lib: Any, dev: Any) -> Any:
    relax = getattr(tvm, "relax", None)
    if relax is None:
        from tvm import relax as imported_relax

        relax = imported_relax
    return relax.VirtualMachine(lib, dev)


def _load_context(request: dict[str, Any], _cache: dict[str, Any] = {}) -> dict[str, Any]:
    """Load (and cache) the TVM runtime context for a request.

    The cache key is (gpu, artifact_path, weights_path). In per-sample CLI mode
    each process loads exactly once (fresh _cache). In persistent --server mode
    the same process reuses the cached tvm import / cuda device / loaded module /
    weights across many frames, eliminating the dominant per-frame overhead
    (python startup + import tvm + cuda init + load_module + np.load).
    """
    import tvm

    gpu = int(request.get("gpu", 0))
    art = str(_require_path(request, "artifact_path"))
    wtp = str(_require_path(request, "runtime_weight_archive_path"))
    execution_abi = _execution_abi(request)
    key = (gpu, art, wtp, execution_abi)
    ctx = _cache.get("ctx")
    if ctx is None or ctx["key"] != key:
        dev = tvm.cuda(gpu)
        lib = tvm.runtime.load_module(art)
        weights = np.load(wtp)
        main_func = _main_from_library(lib)
        if execution_abi == "direct_packed":
            if main_func is None:
                raise ValueError("execution_abi=direct_packed requires artifact exporting packed function main")
            ctx = {
                "key": key,
                "tvm": tvm,
                "dev": dev,
                "weights": weights,
                "execution_abi": "direct_packed",
                "main": main_func,
            }
        elif execution_abi == "relax_vm_return" or main_func is None:
            vm = _build_relax_vm(tvm, lib, dev)
            ctx = {
                "key": key,
                "tvm": tvm,
                "dev": dev,
                "weights": weights,
                "execution_abi": "relax_vm_return",
                "vm": vm,
            }
        else:
            ctx = {
                "key": key,
                "tvm": tvm,
                "dev": dev,
                "weights": weights,
                "execution_abi": "direct_packed",
                "main": main_func,
            }
        _cache["ctx"] = ctx
    return ctx


def _execute(ctx: dict[str, Any], request: dict[str, Any], request_json: Path) -> dict[str, Any]:
    tvm = ctx["tvm"]
    dev = ctx["dev"]
    weights = ctx["weights"]
    execution_abi = str(ctx["execution_abi"])

    output_dir = _require_path(request, "output_dir")
    output_dir.mkdir(parents=True, exist_ok=True)

    activation = np.load(_require_path(request, "activation_npy_path"))
    plan = _runtime_arg_plan(request)

    call_args: list[Any] = []
    output_plan: list[dict[str, Any]] = []
    direct_output_items: list[tuple[dict[str, Any], Any]] = []
    for item in plan:
        role = str(item.get("role"))
        if role == "graph_input":
            value = _assert_array_contract(activation, item)
            call_args.append(make_tvm_array(tvm, value, dev))
        elif role in {"weight_input", "bias_input"}:
            key = _weight_archive_key(str(item["arg_name"]), request)
            if key not in weights:
                raise ValueError(f"runtime initializer archive missing key for {item['arg_name']}: {key}")
            value = _assert_array_contract(weights[key], item)
            call_args.append(make_tvm_array(tvm, value, dev))
        elif role == "graph_output":
            output_plan.append(item)
            if execution_abi == "direct_packed":
                out = np.zeros(tuple(int(dim) for dim in item["shape"]), dtype=str(item["dtype"]))
                tvm_out = make_tvm_array(tvm, out, dev)
                call_args.append(tvm_out)
                direct_output_items.append((item, tvm_out))
        else:
            raise ValueError(f"unsupported runtime arg role: {role}")

    if execution_abi == "direct_packed":
        ctx["main"](*call_args)
        result_arrays = [
            _assert_array_contract(np.asarray(_as_numpy(tvm_out)), item)
            for item, tvm_out in direct_output_items
        ]
    elif execution_abi == "relax_vm_return":
        returned = ctx["vm"]["main"](*call_args)
        raw_outputs = _iter_outputs(returned)
        if len(raw_outputs) != len(output_plan):
            raise ValueError(
                f"relax_vm_return produced {len(raw_outputs)} outputs, expected {len(output_plan)}"
            )
        result_arrays = [
            _assert_array_contract(np.asarray(_as_numpy(raw_outputs[idx])), item)
            for idx, item in enumerate(output_plan)
        ]
    else:
        raise ValueError(f"unsupported execution abi in context: {execution_abi}")
    dev.sync()

    output_records: list[dict[str, Any]] = []
    for item, array in zip(output_plan, result_arrays, strict=True):
        arg_name = str(item["arg_name"])
        output_path = output_dir / output_filename_for_arg(arg_name)
        np.save(output_path, array)
        output_records.append(build_output_record(arg_name=arg_name, output_path=output_path, array=array))

    response = {
        "schema": TVM_WORKER_RESPONSE_SCHEMA,
        "status": "success",
        "label": str(request.get("label", "")),
        "run_id": str(request.get("run_id", "")),
        "request_json": str(request_json),
        "outputs": output_records,
    }
    validate_tvm_worker_response(
        response,
        expected_output_shapes=dict(request.get("expected_output_shapes") or {}),
    )
    _write_json(output_dir / "native_int8_worker_response.json", response)
    return response


def _failure_output_path(request_json: Path) -> Path:
    try:
        request = _read_json(request_json)
        if request.get("output_dir"):
            return Path(str(request["output_dir"])) / "native_int8_worker_response.json"
    except Exception:
        pass
    return request_json.parent / "native_int8_worker_response.json"


def run_worker(request_json: Path) -> dict[str, Any]:
    """Per-sample CLI path: load (fresh process) and execute one request."""
    request = _read_json(request_json)
    ctx = _load_context(request)
    return _execute(ctx, request, request_json)


def run_server() -> int:
    """Persistent server path: load TVM context once, then loop over request
    paths fed on stdin (one per line), reusing the cached module/weights for
    every frame. Emits 'DONE <response_path>' or 'FAIL <response_path>' per
    request, and 'READY' once on startup. Exits on 'STOP' or EOF.
    """
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
            ctx = _load_context(request)
            _execute(ctx, request, request_json)
            sys.stdout.write(f"DONE {_failure_output_path(request_json)}\n")
        except Exception as exc:  # noqa: BLE001 - report and keep serving
            failure = {
                "schema": TVM_WORKER_RESPONSE_SCHEMA,
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
    parser.add_argument("--server", action="store_true",
                        help="persistent server mode: read request paths from stdin")
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
            "schema": TVM_WORKER_RESPONSE_SCHEMA,
            "status": "failed",
            "failure_reason": str(exc),
            "traceback": traceback.format_exc(),
            "request_json": str(args.request_json),
        }
        _write_json(_failure_output_path(args.request_json), failure)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
