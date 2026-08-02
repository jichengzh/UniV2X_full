#!/usr/bin/env python3
"""Build a provenance-bound streaming INT8 contract for prepared F-Cooper ONNX."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import traceback
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


SCHEMA = "fcooper_tvm_int8_quant_contract_v1"
CALIBRATION_SCHEMA = "fcooper_calibration_manifest_v1"
EXPECTED_SAMPLE_COUNT = 16
EXPECTED_SAMPLE_SHAPE = (5, 64, 512, 512)
OBSERVED_OP_TYPES = frozenset({"Conv", "Relu", "DepthToSpace", "Concat"})
METHODS = ("absmax", "percentile_99_99")
UINT8_ZERO_POINT = 128
SYMMETRIC_QMAX = 127.0
MIN_FLOAT32_SCALE = float(np.finfo(np.float32).tiny)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def bounded_chunks(values: Sequence[str], size: int) -> list[tuple[str, ...]]:
    if size <= 0:
        raise ValueError("max_outputs_per_run must be positive")
    return [
        tuple(values[offset : offset + size])
        for offset in range(0, len(values), size)
    ]


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def observation_plan(model: Any) -> tuple[list[str], list[dict[str, Any]]]:
    initializer_names = {str(item.name) for item in model.graph.initializer}
    graph_inputs = [
        str(item.name)
        for item in model.graph.input
        if str(item.name) not in initializer_names
    ]
    if len(graph_inputs) != 1:
        raise ValueError(f"expected exactly one graph input, got {graph_inputs}")

    observed = list(graph_inputs)
    concat_groups: list[dict[str, Any]] = []
    for index, node in enumerate(model.graph.node):
        op_type = str(node.op_type)
        outputs = [str(name) for name in node.output if str(name)]
        if op_type in OBSERVED_OP_TYPES:
            observed.extend(outputs)
        if op_type == "Concat":
            members = _unique(
                [
                    *(str(name) for name in node.input if str(name)),
                    *outputs,
                ]
            )
            observed.extend(members)
            concat_groups.append(
                {
                    "node_name": str(node.name) or f"Concat_{index}",
                    "members": members,
                }
            )
    observed.extend(str(item.name) for item in model.graph.output)
    return _unique(observed), concat_groups


def _tensor_shape(value_info: Any) -> tuple[int, ...]:
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        raise ValueError(f"tensor {value_info.name} has no inferred shape")
    shape = []
    for dimension in tensor_type.shape.dim:
        if not dimension.HasField("dim_value") or int(dimension.dim_value) <= 0:
            raise ValueError(f"tensor {value_info.name} is not positive and static")
        shape.append(int(dimension.dim_value))
    return tuple(shape)


def _resolve_record_path(raw_path: Any, summary_path: Path) -> Path:
    path = Path(str(raw_path or ""))
    if not path.is_absolute():
        path = summary_path.parent / path
    return path.resolve()


def validate_calibration(
    *,
    calibration_summary: Path,
    calibration_dir: Path,
    expected_shape: Sequence[int] = EXPECTED_SAMPLE_SHAPE,
    expected_sample_count: int = EXPECTED_SAMPLE_COUNT,
) -> tuple[dict[str, Any], list[Path]]:
    summary_path = calibration_summary.resolve()
    directory = calibration_dir.resolve()
    if not summary_path.is_file():
        raise FileNotFoundError(f"calibration summary not found: {summary_path}")
    if not directory.is_dir():
        raise FileNotFoundError(f"calibration directory not found: {directory}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise ValueError("calibration summary must contain a JSON object")
    if summary.get("schema_version") != CALIBRATION_SCHEMA:
        raise ValueError(
            f"calibration summary schema must be {CALIBRATION_SCHEMA}"
        )
    records = summary.get("records")
    if not isinstance(records, list):
        raise ValueError("calibration summary records must be a list")
    if (
        summary.get("sample_count") != expected_sample_count
        or len(records) != expected_sample_count
    ):
        raise ValueError(
            f"calibration must contain exactly {expected_sample_count} samples"
        )

    disk_paths = sorted(path.resolve() for path in directory.glob("sample_*.npy"))
    if len(disk_paths) != expected_sample_count:
        raise ValueError(
            f"calibration directory must contain exactly {expected_sample_count} "
            f"sample_*.npy files, got {len(disk_paths)}"
        )

    required_shape = tuple(int(value) for value in expected_shape)
    record_paths: list[Path] = []
    sample_indices: list[int] = []
    for record_index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"calibration record {record_index} must be an object")
        record_path = _resolve_record_path(record.get("path"), summary_path)
        if record_path.parent != directory or not record_path.name.startswith("sample_"):
            raise ValueError(
                f"calibration record path is outside calibration directory: {record_path}"
            )
        if not record_path.is_file():
            raise FileNotFoundError(f"calibration record path not found: {record_path}")
        expected_sha = str(record.get("sha256") or "")
        actual_sha = sha256_file(record_path)
        if expected_sha != actual_sha:
            raise ValueError(
                f"calibration record SHA256 mismatch for {record_path}: "
                f"summary={expected_sha}, actual={actual_sha}"
            )
        summary_shape = tuple(record.get("shape") or ())
        if summary_shape != required_shape:
            raise ValueError(
                f"calibration record {record_path} expected shape "
                f"{list(required_shape)}, summary has {list(summary_shape)}"
            )
        array = np.load(record_path, mmap_mode="r", allow_pickle=False)
        if tuple(array.shape) != summary_shape:
            raise ValueError(
                f"calibration record shape mismatch for {record_path}: "
                f"summary={list(summary_shape)}, actual={list(array.shape)}"
            )
        if array.dtype != np.dtype(np.float32):
            raise ValueError(
                f"calibration record dtype must be float32 for {record_path}, "
                f"got {array.dtype}"
            )
        record_paths.append(record_path)
        sample_index = record.get("sample_index")
        if isinstance(sample_index, bool) or not isinstance(sample_index, int):
            raise ValueError(
                f"calibration record {record_index} sample_index must be an integer"
            )
        sample_indices.append(sample_index)

    if len(set(record_paths)) != expected_sample_count or set(record_paths) != set(
        disk_paths
    ):
        raise ValueError("calibration record paths do not exactly match sample files")
    if len(set(sample_indices)) != expected_sample_count:
        raise ValueError("calibration sample_index values must be unique")
    return summary, record_paths


def _chunked_absmax(value: np.ndarray, chunk_values: int) -> float:
    if chunk_values <= 0:
        raise ValueError("reduction_chunk_values must be positive")
    flat = np.asarray(value).reshape(-1)
    maximum = 0.0
    for offset in range(0, flat.size, chunk_values):
        chunk = flat[offset : offset + chunk_values]
        if not np.all(np.isfinite(chunk)):
            raise ValueError("observed tensor contains non-finite values")
        maximum = max(maximum, float(np.max(np.abs(chunk), initial=0.0)))
    return maximum


def _sample_values(value: np.ndarray, limit: int) -> np.ndarray:
    if limit <= 0:
        raise ValueError(
            "sample_values_per_tensor_per_sample must be positive"
        )
    flat = np.asarray(value).reshape(-1)
    if flat.size == 0:
        return np.empty((0,), dtype=np.float32)
    stride = max(1, math.ceil(flat.size / limit))
    sampled = np.asarray(flat[::stride][:limit], dtype=np.float32).copy()
    if not np.all(np.isfinite(sampled)):
        raise ValueError("observed tensor contains non-finite sampled values")
    return sampled


def _percentile_threshold(values: list[np.ndarray], percentile: float) -> float:
    if not 0.0 < percentile <= 100.0:
        raise ValueError("percentile must be in (0, 100]")
    if not values:
        raise ValueError("percentile calibration has no sampled values")
    merged = np.concatenate(values)
    return float(np.percentile(np.abs(merged), percentile))


def _quant_param(
    selected_absmax: float,
    *,
    observed_absmax: float,
    method: str,
) -> dict[str, Any]:
    if not math.isfinite(selected_absmax) or selected_absmax < 0.0:
        raise ValueError(f"invalid selected tensor range: {selected_absmax}")
    minimum_absmax = MIN_FLOAT32_SCALE * SYMMETRIC_QMAX
    effective_absmax = max(float(selected_absmax), minimum_absmax)
    return {
        "scale": effective_absmax / SYMMETRIC_QMAX,
        "zero_point": UINT8_ZERO_POINT,
        "absmax": effective_absmax,
        "observed_absmax": float(observed_absmax),
        "all_zero": selected_absmax == 0.0,
        "range_floored": selected_absmax < minimum_absmax,
        "minimum_float32_scale": MIN_FLOAT32_SCALE,
        "dtype": "uint8",
        "scheme": "static_symmetric_uint8_centered_128",
        "calibration_method": method,
    }


def _normalize_concat_scales(
    params: dict[str, dict[str, Any]],
    groups: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    parent = {name: name for group in groups for name in group["members"]}

    def find(name: str) -> str:
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for group in groups:
        members = group["members"]
        for member in members[1:]:
            union(members[0], member)

    component_maxima: dict[str, float] = {}
    for name in parent:
        root = find(name)
        component_maxima[root] = max(
            component_maxima.get(root, 0.0), float(params[name]["absmax"])
        )

    records = []
    for group in groups:
        members = group["members"]
        original = {name: float(params[name]["absmax"]) for name in members}
        common_absmax = component_maxima[find(members[0])]
        for name in members:
            params[name] = {
                **params[name],
                "scale": common_absmax / SYMMETRIC_QMAX,
                "absmax": common_absmax,
                "concat_normalized": True,
            }
        records.append(
            {
                "node_name": group["node_name"],
                "members": members,
                "original_absmax": original,
                "common_absmax": common_absmax,
                "normalization_applied": any(
                    value != common_absmax for value in original.values()
                ),
                "policy": "connected_concat_group_common_max_range",
            }
        )
    return records


def _instrument_model(
    model: Any, observed_names: Sequence[str], input_name: str
) -> Any:
    infos = {
        str(item.name): item
        for item in [
            *model.graph.input,
            *model.graph.value_info,
            *model.graph.output,
        ]
    }
    existing_outputs = {str(item.name) for item in model.graph.output}
    for name in observed_names:
        if name == input_name or name in existing_outputs:
            continue
        if name not in infos:
            raise ValueError(f"shape inference lacks observed tensor {name}")
        model.graph.output.append(copy.deepcopy(infos[name]))
    return model


def build_contract(
    *,
    onnx_path: Path,
    calibration_summary: Path,
    calibration_dir: Path | None = None,
    calibration_method: str = "absmax",
    percentile: float = 99.99,
    sample_values_per_tensor_per_sample: int = 32768,
    max_outputs_per_run: int = 1,
    reduction_chunk_values: int = 1_048_576,
    expected_shape: Sequence[int] = EXPECTED_SAMPLE_SHAPE,
    expected_sample_count: int = EXPECTED_SAMPLE_COUNT,
) -> dict[str, Any]:
    import onnx
    import onnxruntime as ort

    onnx_path = onnx_path.resolve()
    summary_path = calibration_summary.resolve()
    directory = (
        calibration_dir.resolve()
        if calibration_dir is not None
        else summary_path.parent
    )
    if not onnx_path.is_file():
        raise FileNotFoundError(f"prepared ONNX not found: {onnx_path}")
    if calibration_method not in METHODS:
        raise ValueError(f"calibration_method must be one of {METHODS}")
    if expected_sample_count != EXPECTED_SAMPLE_COUNT:
        raise ValueError("F-Cooper calibration contract requires exactly 16 samples")
    if max_outputs_per_run <= 0 or reduction_chunk_values <= 0:
        raise ValueError("streaming chunk bounds must be positive")
    if sample_values_per_tensor_per_sample <= 0:
        raise ValueError("percentile sample bound must be positive")
    if calibration_method == "percentile_99_99" and percentile != 99.99:
        raise ValueError("percentile_99_99 method requires percentile=99.99")

    summary, sample_paths = validate_calibration(
        calibration_summary=summary_path,
        calibration_dir=directory,
        expected_shape=expected_shape,
        expected_sample_count=expected_sample_count,
    )
    model = onnx.shape_inference.infer_shapes(onnx.load(str(onnx_path)))
    onnx.checker.check_model(model)
    observed_names, concat_groups = observation_plan(model)
    input_name = observed_names[0]
    input_info = next(
        item for item in model.graph.input if str(item.name) == input_name
    )
    model_input_shape = _tensor_shape(input_info)
    required_shape = tuple(int(value) for value in expected_shape)
    if model_input_shape != required_shape:
        raise ValueError(
            f"ONNX graph input expected shape {list(required_shape)}, "
            f"got {list(model_input_shape)}"
        )
    graph_outputs = [str(item.name) for item in model.graph.output]
    instrumented = _instrument_model(
        model, observed_names=observed_names, input_name=input_name
    )

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers()
        else ["CPUExecutionProvider"]
    )
    session = ort.InferenceSession(
        instrumented.SerializeToString(), providers=providers
    )
    output_names = [name for name in observed_names if name != input_name]
    output_chunks = bounded_chunks(output_names, max_outputs_per_run)
    maxima = {name: 0.0 for name in observed_names}
    samples: dict[str, list[np.ndarray]] | None = (
        {name: [] for name in observed_names}
        if calibration_method == "percentile_99_99"
        else None
    )

    for sample_path in sample_paths:
        sample = np.load(sample_path, mmap_mode="r", allow_pickle=False)
        maxima[input_name] = max(
            maxima[input_name], _chunked_absmax(sample, reduction_chunk_values)
        )
        if samples is not None:
            samples[input_name].append(
                _sample_values(sample, sample_values_per_tensor_per_sample)
            )
        for names in output_chunks:
            outputs = session.run(list(names), {input_name: sample})
            for name, value in zip(names, outputs):
                maxima[name] = max(
                    maxima[name],
                    _chunked_absmax(value, reduction_chunk_values),
                )
                if samples is not None:
                    samples[name].append(
                        _sample_values(
                            value, sample_values_per_tensor_per_sample
                        )
                    )
            del outputs
        del sample

    if samples is None:
        selected = dict(maxima)
    else:
        selected = {
            name: _percentile_threshold(values, percentile)
            for name, values in samples.items()
        }
    params = {
        name: _quant_param(
            selected[name],
            observed_absmax=maxima[name],
            method=calibration_method,
        )
        for name in observed_names
    }
    normalization = _normalize_concat_scales(params, concat_groups)
    missing = [name for name in observed_names if name not in params]
    method_parameters = (
        {}
        if calibration_method == "absmax"
        else {
            "percentile": percentile,
            "sample_values_per_tensor_per_sample": (
                sample_values_per_tensor_per_sample
            ),
        }
    )
    method_parameters.update(
        {
            "max_outputs_per_run": max_outputs_per_run,
            "reduction_chunk_values": reduction_chunk_values,
        }
    )
    return {
        "schema_version": SCHEMA,
        "status": "success",
        "quantization": {
            "dtype": "uint8",
            "zero_point": UINT8_ZERO_POINT,
            "qmax": int(SYMMETRIC_QMAX),
            "semantics": "static_symmetric_uint8_centered_128",
        },
        "onnx": {
            "path": str(onnx_path),
            "sha256": sha256_file(onnx_path),
            "graph_input": input_name,
            "graph_input_shape": list(model_input_shape),
            "graph_outputs": graph_outputs,
        },
        "calibration": {
            "directory": str(directory),
            "summary_path": str(summary_path),
            "summary_sha256": sha256_file(summary_path),
            "summary_schema": summary.get("schema_version"),
            "sample_count": len(sample_paths),
            "samples": [
                {
                    "path": str(path),
                    "sha256": str(record["sha256"]),
                    "shape": list(record["shape"]),
                }
                for path, record in zip(sample_paths, summary["records"])
            ],
            "method": calibration_method,
            "parameters": method_parameters,
        },
        "sample_count": len(sample_paths),
        "runtime": {
            "provider": session.get_providers()[0],
            "streaming": True,
            "sample_loading": "numpy_mmap_one_sample_at_a_time",
            "output_chunk_count": len(output_chunks),
        },
        "coverage": {
            "required_op_types": sorted(OBSERVED_OP_TYPES),
            "required": observed_names,
            "observed": list(params),
            "missing": missing,
            "required_count": len(observed_names),
            "observed_count": len(params),
        },
        "concat_scale_normalization": normalization,
        "params": params,
        "failure": None,
    }


def _failure_report(
    *,
    exc: Exception,
    onnx_path: Path,
    calibration_summary: Path,
    calibration_dir: Path | None,
    options: dict[str, Any],
) -> dict[str, Any]:
    onnx_path = onnx_path.resolve()
    summary_path = calibration_summary.resolve()
    directory = (
        calibration_dir.resolve()
        if calibration_dir is not None
        else summary_path.parent
    )
    method = str(options.get("calibration_method", "absmax"))
    expected_shape = list(options.get("expected_shape", EXPECTED_SAMPLE_SHAPE))
    expected_count = int(
        options.get("expected_sample_count", EXPECTED_SAMPLE_COUNT)
    )
    return {
        "schema_version": SCHEMA,
        "status": "failed",
        "onnx": {
            "path": str(onnx_path),
            "sha256": sha256_file(onnx_path) if onnx_path.is_file() else None,
        },
        "calibration": {
            "directory": str(directory),
            "summary_path": str(summary_path),
            "summary_sha256": (
                sha256_file(summary_path) if summary_path.is_file() else None
            ),
            "method": method,
            "expected_sample_count": expected_count,
            "expected_sample_shape": expected_shape,
            "parameters": {
                "percentile": options.get("percentile", 99.99),
                "sample_values_per_tensor_per_sample": options.get(
                    "sample_values_per_tensor_per_sample", 32768
                ),
                "max_outputs_per_run": options.get("max_outputs_per_run", 1),
                "reduction_chunk_values": options.get(
                    "reduction_chunk_values", 1_048_576
                ),
            },
        },
        "sample_count": 0,
        "coverage": {
            "required": [],
            "observed": [],
            "missing": [],
            "required_count": 0,
            "observed_count": 0,
        },
        "params": {},
        "concat_scale_normalization": [],
        "failure": {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        },
    }


def write_contract(
    *,
    output_path: Path,
    onnx_path: Path,
    calibration_summary: Path,
    calibration_dir: Path | None = None,
    **options: Any,
) -> dict[str, Any]:
    try:
        contract = build_contract(
            onnx_path=onnx_path,
            calibration_summary=calibration_summary,
            calibration_dir=calibration_dir,
            **options,
        )
    except Exception as exc:
        write_json(
            output_path,
            _failure_report(
                exc=exc,
                onnx_path=onnx_path,
                calibration_summary=calibration_summary,
                calibration_dir=calibration_dir,
                options=options,
            ),
        )
        raise
    write_json(output_path, contract)
    return contract


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--calibration-summary", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--calibration-method", choices=METHODS, default="absmax"
    )
    parser.add_argument("--percentile", type=float, default=99.99)
    parser.add_argument(
        "--sample-values-per-tensor-per-sample", type=int, default=32768
    )
    parser.add_argument("--max-outputs-per-run", type=int, default=1)
    parser.add_argument("--reduction-chunk-values", type=int, default=1_048_576)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        contract = write_contract(
            output_path=args.output_json,
            onnx_path=args.onnx,
            calibration_summary=args.calibration_summary,
            calibration_dir=args.calibration_dir,
            calibration_method=args.calibration_method,
            percentile=args.percentile,
            sample_values_per_tensor_per_sample=(
                args.sample_values_per_tensor_per_sample
            ),
            max_outputs_per_run=args.max_outputs_per_run,
            reduction_chunk_values=args.reduction_chunk_values,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "output": str(args.output_json),
                    "failure": f"{type(exc).__name__}: {exc}",
                },
                sort_keys=True,
            )
        )
        return 2
    print(
        json.dumps(
            {
                "status": "success",
                "output": str(args.output_json),
                "tensor_count": contract["coverage"]["observed_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
