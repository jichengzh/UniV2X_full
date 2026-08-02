#!/usr/bin/env python3
"""Build a provenance-bound static uint8 tensor quantization contract."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "stage3_tvm_int8_quant_contract_v3"
SAMPLE_VALUES_PER_TENSOR_PER_BATCH = 32768


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quant_param(absmax: float) -> dict[str, Any]:
    value = float(absmax)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"tensor absmax must be finite and positive, got {value}")
    return {"scale": value / 127.0, "zero_point": 128, "absmax": value,
            "scheme": "symmetric_absmax_uint8_centered_128",
            "source": "train16_onnxruntime_static_calibration"}


def percentile_clip_threshold(samples: np.ndarray, percentile: float = 99.99) -> float:
    values = np.abs(np.asarray(samples, dtype=np.float32).reshape(-1))
    if values.size == 0 or not 0.0 < percentile <= 100.0:
        raise ValueError("nonempty samples and percentile in (0, 100] are required")
    threshold = float(np.percentile(values, percentile))
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError(f"invalid percentile threshold: {threshold}")
    return threshold


def sampled_values(value: np.ndarray) -> np.ndarray:
    flat = np.asarray(value, dtype=np.float32).reshape(-1)
    stride = max(1, math.ceil(flat.size / SAMPLE_VALUES_PER_TENSOR_PER_BATCH))
    return flat[::stride][:SAMPLE_VALUES_PER_TENSOR_PER_BATCH].copy()


def observed_tensor_names(model: Any) -> list[str]:
    names = [str(item.name) for item in model.graph.input]
    for node in model.graph.node:
        if str(node.op_type) in {"Conv", "Add", "Relu", "Identity"}:
            names.extend(str(name) for name in node.output if str(name))
    names.extend(str(item.name) for item in model.graph.output)
    return list(dict.fromkeys(names))


def normalize_calibration_batch(spatial: np.ndarray, summary: dict[str, Any]) -> np.ndarray:
    array = np.asarray(spatial, dtype=np.float32)
    if array.ndim == 5 and array.shape[0] == 16:
        return array
    record_lens = summary.get("scene_record_lens")
    if array.ndim != 4 or not isinstance(record_lens, list) or len(record_lens) != 16:
        raise ValueError(f"expected train-16 spatial_features, got {array.shape}")
    if sum(int(value) for value in record_lens) != array.shape[0] or max(record_lens) > 2:
        raise ValueError("invalid flattened Pyramid scene_record_lens")
    batches = []
    offset = 0
    for raw_len in record_lens:
        record_len = int(raw_len)
        batch = np.zeros((2, *array.shape[1:]), dtype=np.float32)
        batch[:record_len] = array[offset:offset + record_len]
        batches.append(batch)
        offset += record_len
    return np.stack(batches, axis=0)


def build_contract(*, onnx_path: Path, calibration_npz: Path, calibration_summary: Path,
                   calibration_method: str = "absmax") -> dict[str, Any]:
    import onnx
    import onnxruntime as ort

    summary = json.loads(calibration_summary.read_text(encoding="utf-8"))
    with np.load(calibration_npz, allow_pickle=False) as payload:
        spatial = normalize_calibration_batch(payload["spatial_features"], summary)
    model = onnx.shape_inference.infer_shapes(onnx.load(str(onnx_path)))
    original_outputs = [item.name for item in model.graph.output]
    infos = {item.name: item for item in [*model.graph.input, *model.graph.value_info, *model.graph.output]}
    names = observed_tensor_names(model)
    initializer_names = {item.name for item in model.graph.initializer}
    graph_inputs = [item.name for item in model.graph.input if item.name not in initializer_names]
    if len(graph_inputs) != 1:
        raise ValueError(f"expected one graph input, got {graph_inputs}")
    input_name = graph_inputs[0]
    requested = [name for name in names if name != input_name]
    existing = set(original_outputs)
    for name in requested:
        if name not in infos:
            raise ValueError(f"shape inference lacks observed tensor {name}")
        if name not in existing:
            model.graph.output.append(copy.deepcopy(infos[name]))
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in ort.get_available_providers() else ["CPUExecutionProvider"]
    session = ort.InferenceSession(model.SerializeToString(), providers=providers)
    maxima = {name: 0.0 for name in names}
    maxima[input_name] = float(np.max(np.abs(spatial)))
    samples = {name: [] for name in names} if calibration_method == "percentile_99_99" else None
    if samples is not None:
        samples[input_name].append(sampled_values(spatial))
    for index in range(spatial.shape[0]):
        outputs = session.run(requested, {input_name: spatial[index]})
        for name, value in zip(requested, outputs):
            maxima[name] = max(maxima[name], float(np.max(np.abs(np.asarray(value, dtype=np.float32)))))
            if samples is not None:
                samples[name].append(sampled_values(value))
    thresholds = maxima if samples is None else {
        name: percentile_clip_threshold(np.concatenate(values)) for name, values in samples.items()
    }
    params = {name: quant_param(value) for name, value in thresholds.items()}
    for name, value in maxima.items():
        params[name].update({"observed_absmax": value, "calibration_method": calibration_method})
    return {"schema": SCHEMA, "quantization_semantics": "static_symmetric_uint8_centered_128",
            "sample_count": 16, "calibration_split": "train",
            "onnx_path": str(onnx_path.resolve()), "onnx_sha256": sha256_file(onnx_path),
            "calibration_npz": str(calibration_npz.resolve()), "calibration_npz_sha256": sha256_file(calibration_npz),
            "calibration_summary": str(calibration_summary.resolve()), "calibration_summary_sha256": sha256_file(calibration_summary),
            "calibration_summary_schema": summary.get("schema"), "runtime_provider": session.get_providers()[0],
            "calibration_method": calibration_method,
            "graph_input": input_name, "graph_outputs": original_outputs,
            "tensor_count": len(params), "params": params}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--calibration-npz", type=Path, required=True)
    parser.add_argument("--calibration-summary", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--calibration-method", choices=("absmax", "percentile_99_99"), default="absmax")
    args = parser.parse_args()
    for path in (args.onnx, args.calibration_npz, args.calibration_summary):
        if not path.is_file():
            parser.error(f"input file not found: {path}")
    return args


def main() -> int:
    args = parse_args()
    contract = build_contract(onnx_path=args.onnx, calibration_npz=args.calibration_npz,
                              calibration_summary=args.calibration_summary,
                              calibration_method=args.calibration_method)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "success", "output": str(args.output_json), "tensor_count": contract["tensor_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
