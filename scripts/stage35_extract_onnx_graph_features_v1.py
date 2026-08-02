#!/usr/bin/env python3
"""Extract neutral static ONNX graph descriptors for Gold groups."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


SCHEMA = "stage35_onnx_graph_features_v1"


def product(values: list[int]) -> int:
    return math.prod(value for value in values if value > 0)


def tensor_dims(value: Any) -> list[int]:
    return [int(dim.dim_value) for dim in value.type.tensor_type.shape.dim]


def attribute_ints(node: Any) -> dict[str, int | list[int]]:
    result: dict[str, int | list[int]] = {}
    for attribute in node.attribute:
        result[attribute.name] = list(attribute.ints) if attribute.ints else int(attribute.i)
    return result


def estimate_conv_macs(
    op_type: str, *, weight_dims: list[int], input_dims: list[int], output_dims: list[int]
) -> int:
    weight_elements = product(weight_dims)
    if op_type == "ConvTranspose":
        batch = max(1, input_dims[0]) if input_dims else 1
        spatial = product(input_dims[2:]) if len(input_dims) >= 4 else 1
    else:
        batch = max(1, output_dims[0]) if output_dims else 1
        spatial = product(output_dims[2:]) if len(output_dims) >= 4 else 1
    return batch * spatial * weight_elements


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_onnx_path(rows: list[dict[str, Any]]) -> Path:
    for row in rows:
        result_path = Path(str(row.get("performance_result_json") or ""))
        if not result_path.is_file():
            continue
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        onnx_path = Path(str(payload.get("onnx_path") or ""))
        if onnx_path.is_file():
            return onnx_path
    raise FileNotFoundError(f"no ONNX source for {rows[0]['group_id']}")


def graph_features(path: Path) -> dict[str, float | int | str | list[int]]:
    import onnx

    model = onnx.shape_inference.infer_shapes(onnx.load(path))
    graph = model.graph
    shapes = {
        value.name: tensor_dims(value)
        for value in [*graph.input, *graph.value_info, *graph.output]
    }
    initializers = {value.name: value for value in graph.initializer}
    op_counts: dict[str, int] = {}
    for node in graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1

    conv_macs = 0
    conv_output_elements = 0
    conv_parameter_elements = 0
    group_conv_count = 0
    depthwise_conv_count = 0
    stride2_conv_count = 0
    kernel1_conv_count = 0
    kernel3_conv_count = 0
    max_conv_channels = 0
    for node in graph.node:
        if node.op_type not in {"Conv", "ConvTranspose"} or len(node.input) < 2:
            continue
        weight = initializers.get(node.input[1])
        if weight is None:
            continue
        weight_dims = [int(value) for value in weight.dims]
        input_dims = shapes.get(node.input[0], [])
        output_dims = shapes.get(node.output[0], [])
        attrs = attribute_ints(node)
        group = int(attrs.get("group", 1))
        strides = attrs.get("strides", [1, 1])
        kernel = weight_dims[2:] if len(weight_dims) >= 4 else [1, 1]
        output_elements = product(output_dims)
        weight_elements = product(weight_dims)
        output_channels = output_dims[1] if len(output_dims) >= 2 else weight_dims[0]
        macs = estimate_conv_macs(
            node.op_type, weight_dims=weight_dims, input_dims=input_dims, output_dims=output_dims
        )
        conv_macs += macs
        conv_output_elements += output_elements
        conv_parameter_elements += weight_elements
        group_conv_count += int(group > 1)
        input_channels = weight_dims[1] * group if len(weight_dims) >= 2 else 0
        depthwise_conv_count += int(group > 1 and group == input_channels)
        stride2_conv_count += int(isinstance(strides, list) and any(value == 2 for value in strides))
        kernel1_conv_count += int(kernel == [1, 1])
        kernel3_conv_count += int(kernel == [3, 3])
        max_conv_channels = max(max_conv_channels, input_channels, output_channels)

    parameter_elements = sum(product([int(value) for value in item.dims]) for item in graph.initializer)
    input_dims = tensor_dims(graph.input[0])
    input_elements = product(input_dims)
    approximate_bytes = 4 * (input_elements + parameter_elements + conv_output_elements)
    return {
        "onnx_path": str(path),
        "onnx_sha256": sha256_file(path),
        "input_dims": input_dims,
        "input_elements": input_elements,
        "input_channels": input_dims[1] if len(input_dims) > 1 else 0,
        "input_height": input_dims[2] if len(input_dims) > 2 else 0,
        "input_width": input_dims[3] if len(input_dims) > 3 else 0,
        "node_count": len(graph.node),
        "conv_count": op_counts.get("Conv", 0),
        "conv_transpose_count": op_counts.get("ConvTranspose", 0),
        "relu_count": op_counts.get("Relu", 0),
        "add_count": op_counts.get("Add", 0),
        "concat_count": op_counts.get("Concat", 0),
        "group_conv_count": group_conv_count,
        "depthwise_conv_count": depthwise_conv_count,
        "stride2_conv_count": stride2_conv_count,
        "kernel1_conv_count": kernel1_conv_count,
        "kernel3_conv_count": kernel3_conv_count,
        "max_conv_channels": max_conv_channels,
        "parameter_elements": parameter_elements,
        "conv_parameter_elements": conv_parameter_elements,
        "conv_output_elements": conv_output_elements,
        "conv_macs": conv_macs,
        "conv_flops": 2 * conv_macs,
        "arithmetic_intensity_proxy": conv_macs / max(1, approximate_bytes),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads(args.gold_json.read_text(encoding="utf-8"))
    source_sha256 = sha256_file(args.gold_json)
    script_sha256 = sha256_file(Path(__file__))
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["group_id"]), []).append(row)
    output = []
    for group_id, group_rows in sorted(grouped.items()):
        onnx_path = resolve_onnx_path(group_rows)
        output.append({
            "schema": SCHEMA,
            "group_id": group_id,
            "model": group_rows[0]["model"],
            "width": group_rows[0]["width"],
            "source_gold_sha256": source_sha256,
            "extractor_script_sha256": script_sha256,
            **graph_features(onnx_path),
        })
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "success", "groups": len(output), "output": str(args.output_json)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
