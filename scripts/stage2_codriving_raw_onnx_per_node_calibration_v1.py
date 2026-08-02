#!/usr/bin/env python3
"""Generate strict raw-ONNX per-Conv calibration for CoDriving mixed lowering."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from stage2_codriving_mixed_auto_lowering_v1 import collect_onnx_conv_records, sha256_file


SCHEMA = "codriving_raw_onnx_per_node_calibration_v1"
EXPECTED_SAMPLES = 16
EXPECTED_SHAPE = (16, 2, 64, 256, 512)
SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _exact_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an exact JSON integer")
    return value


def validate_source_summary(summary: Any) -> dict[str, Any]:
    if not isinstance(summary, dict):
        raise ValueError("calibration summary must be a mapping")
    if summary.get("schema") != "v2_gold_coldstart_96_codriving_calib_export_v1":
        raise ValueError("unexpected calibration summary schema")
    if summary.get("calibration_split") != "train":
        raise ValueError("calibration summary must use train split")
    if _exact_int(summary.get("collected_samples"), "collected_samples") != EXPECTED_SAMPLES:
        raise ValueError(f"collected_samples must equal {EXPECTED_SAMPLES}")
    shape = tuple(_exact_int(value, f"spatial_features[{index}]") for index, value in enumerate((summary.get("shapes") or {}).get("spatial_features") or ()))
    if shape != EXPECTED_SHAPE:
        raise ValueError(f"unexpected calibration spatial_features shape: {shape}")
    if not str(summary.get("split_source_file") or ""):
        raise ValueError("calibration summary lacks split_source_file")
    if SHA256_RE.fullmatch(str(summary.get("output_sha256") or "")) is None:
        raise ValueError("calibration summary lacks valid output_sha256")
    return dict(summary)


def build_manifest(
    *,
    records: list[dict[str, Any]],
    input_absmax: dict[str, float],
    weights: dict[str, Any],
    onnx_path: Path,
    onnx_sha256: str,
    calibration_source: Path,
    calibration_source_sha256: str,
    calibration_summary: Path,
    calibration_summary_sha256: str,
    calibration_split_source: Path,
    sample_count: int,
    runtime_provider: str | None = None,
) -> dict[str, Any]:
    import numpy as np

    nodes: dict[str, dict[str, Any]] = {}
    for record in records:
        node_id = str(record["node_id"])
        input_name = str(record["input_name"])
        weight_name = str(record["weight_name"])
        activation_max = float(input_absmax.get(input_name, 0.0))
        if not math.isfinite(activation_max) or activation_max <= 0.0:
            raise ValueError(f"invalid observed activation absmax for {node_id}: {activation_max}")
        if weight_name not in weights:
            raise ValueError(f"missing ONNX weight for {node_id}: {weight_name}")
        weight_max = float(np.max(np.abs(np.asarray(weights[weight_name], dtype="float32"))))
        if not math.isfinite(weight_max) or weight_max <= 0.0:
            raise ValueError(f"invalid weight absmax for {node_id}: {weight_max}")
        nodes[node_id] = {
            "node_index": int(record["node_index"]),
            "input_name": input_name,
            "weight_name": weight_name,
            "output_name": str(record["output_name"]),
            "input_shape": [int(value) for value in record["input_shape"]],
            "weight_shape": [int(value) for value in record["weight_shape"]],
            "output_shape": [int(value) for value in record["output_shape"]],
            "input_absmax": activation_max,
            "input_scale": activation_max / 127.0,
            "weight_absmax": weight_max,
            "weight_scale": weight_max / 127.0,
        }
    return {
        "schema": SCHEMA,
        "quantization_semantics": "symmetric_absmax_int8_per_raw_onnx_node",
        "calibration_split": "train",
        "sample_count": int(sample_count),
        "onnx": str(onnx_path),
        "onnx_sha256": onnx_sha256,
        "calibration_source": str(calibration_source),
        "calibration_source_sha256": calibration_source_sha256,
        "calibration_summary": str(calibration_summary),
        "calibration_summary_sha256": calibration_summary_sha256,
        "calibration_split_source": str(calibration_split_source),
        "runtime_provider": runtime_provider,
        "node_count": len(nodes),
        "nodes": nodes,
    }


def validate_manifest(manifest: Any, *, expected_node_ids: set[str]) -> dict[str, Any]:
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
        raise ValueError("unexpected per-node calibration schema")
    if manifest.get("calibration_split") != "train":
        raise ValueError("per-node calibration must use train split")
    if _exact_int(manifest.get("sample_count"), "sample_count") != EXPECTED_SAMPLES:
        raise ValueError("per-node calibration sample_count must equal 16")
    for field in ("onnx_sha256", "calibration_source_sha256", "calibration_summary_sha256"):
        if SHA256_RE.fullmatch(str(manifest.get(field) or "")) is None:
            raise ValueError(f"invalid {field}")
    nodes = manifest.get("nodes")
    if not isinstance(nodes, dict) or set(nodes) != set(expected_node_ids):
        raise ValueError("per-node calibration node coverage mismatch")
    if _exact_int(manifest.get("node_count"), "node_count") != len(nodes):
        raise ValueError("per-node calibration node_count mismatch")
    for node_id, record in nodes.items():
        if not all(str(record.get(field) or "") for field in ("input_name", "weight_name", "output_name")):
            raise ValueError(f"missing tensor binding for {node_id}")
        for field in ("input_scale", "weight_scale"):
            value = float(record.get(field) or 0.0)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"invalid {field} for {node_id}")
    return dict(manifest)


def _collect_input_absmax(onnx_path: Path, records: list[dict[str, Any]], spatial: Any) -> tuple[dict[str, float], str]:
    import numpy as np
    import onnx
    import onnxruntime as ort

    model = onnx.shape_inference.infer_shapes(onnx.load(str(onnx_path)))
    value_info = {value.name: value for value in [*model.graph.input, *model.graph.value_info, *model.graph.output]}
    tensor_names = list(dict.fromkeys(str(record["input_name"]) for record in records))
    graph_input_names = {item.name for item in model.graph.input}
    requested_names = [name for name in tensor_names if name not in graph_input_names]
    existing_outputs = {item.name for item in model.graph.output}
    for name in requested_names:
        if name not in value_info:
            raise ValueError(f"ONNX shape inference lacks Conv input tensor: {name}")
        if name not in existing_outputs:
            model.graph.output.append(copy.deepcopy(value_info[name]))
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in ort.get_available_providers() else ["CPUExecutionProvider"]
    session = ort.InferenceSession(model.SerializeToString(), providers=providers)
    provider = session.get_providers()[0]
    input_name = session.get_inputs()[0].name
    maxima = {name: 0.0 for name in tensor_names}
    if input_name in maxima:
        maxima[input_name] = float(np.max(np.abs(np.asarray(spatial, dtype="float32"))))
    for sample_index in range(spatial.shape[0]):
        outputs = session.run(requested_names, {input_name: np.asarray(spatial[sample_index], dtype="float32")})
        for name, value in zip(requested_names, outputs):
            maximum = float(np.max(np.abs(np.asarray(value, dtype="float32"))))
            maxima[name] = max(maxima[name], maximum)
        print(f"[calibration] sample={sample_index + 1}/{spatial.shape[0]}", flush=True)
    return maxima, provider


def run(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    import onnx
    from onnx import numpy_helper

    summary = validate_source_summary(json.loads(args.calib_summary.read_text(encoding="utf-8")))
    source_sha = sha256_file(args.calib_npz)
    if source_sha != summary["output_sha256"]:
        raise ValueError("calibration NPZ SHA256 does not match summary")
    if Path(str(summary["output"])).resolve() != args.calib_npz.resolve():
        raise ValueError("calibration summary output path does not match --calib-npz")
    with np.load(args.calib_npz, allow_pickle=False) as payload:
        spatial = np.asarray(payload["spatial_features"])
    if tuple(spatial.shape) != EXPECTED_SHAPE:
        raise ValueError(f"unexpected calibration NPZ shape: {spatial.shape}")

    records = collect_onnx_conv_records(args.onnx)
    model = onnx.load(str(args.onnx))
    weights = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    maxima, provider = _collect_input_absmax(args.onnx, records, spatial)
    manifest = build_manifest(
        records=records,
        input_absmax=maxima,
        weights=weights,
        onnx_path=args.onnx,
        onnx_sha256=sha256_file(args.onnx),
        calibration_source=args.calib_npz,
        calibration_source_sha256=source_sha,
        calibration_summary=args.calib_summary,
        calibration_summary_sha256=sha256_file(args.calib_summary),
        calibration_split_source=Path(summary["split_source_file"]),
        sample_count=int(spatial.shape[0]),
        runtime_provider=provider,
    )
    return validate_manifest(manifest, expected_node_ids={str(record["node_id"]) for record in records})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--calib-npz", type=Path, required=True)
    parser.add_argument("--calib-summary", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.onnx, args.calib_npz, args.calib_summary):
        if not path.is_file():
            parser.error(f"input file not found: {path}")
    return args


def main() -> int:
    args = parse_args()
    manifest = run(args)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out_json.with_suffix(args.out_json.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(args.out_json)
    print(json.dumps({"status": "success", "out_json": str(args.out_json), "node_count": manifest["node_count"], "provider": manifest["runtime_provider"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
