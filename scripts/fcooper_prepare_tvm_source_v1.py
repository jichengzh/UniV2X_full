#!/usr/bin/env python3
"""Prepare a provenance-bound formal F-Cooper ONNX source for TVM."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import onnx
import onnxruntime as ort
from onnx import ModelProto, NodeProto, TensorProto, helper, numpy_helper


SOURCE_SCHEMA = "fcooper_source_export_v2"
REPORT_SCHEMA = "fcooper_tvm_source_preparation_v1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _validate_source(args: argparse.Namespace) -> dict[str, Any]:
    if not args.source_onnx.is_file():
        raise ValueError(f"source ONNX does not exist: {args.source_onnx}")
    if not args.source_export_report.is_file():
        raise ValueError(
            f"source export report does not exist: {args.source_export_report}"
        )

    actual_report_sha = sha256_file(args.source_export_report)
    expected_report_sha = str(args.source_export_report_sha256).lower()
    if actual_report_sha != expected_report_sha:
        raise ValueError(
            "source export report SHA256 mismatch: "
            f"expected {expected_report_sha}, got {actual_report_sha}"
        )

    source_report = json.loads(args.source_export_report.read_text())
    formal_contract = (
        source_report.get("schema_version") == SOURCE_SCHEMA
        and source_report.get("status") == "success"
        and source_report.get("source_kind") == "formal_recovered"
        and source_report.get("formal_measurement_eligible") is True
    )
    if not formal_contract:
        raise ValueError(
            "source export report does not describe a formal backend-neutral source"
        )

    actual_onnx_sha = sha256_file(args.source_onnx)
    if source_report.get("onnx_sha256") != actual_onnx_sha:
        raise ValueError(
            "source ONNX SHA256 mismatch: "
            f"report has {source_report.get('onnx_sha256')}, got {actual_onnx_sha}"
        )
    return source_report


def _attributes(node: NodeProto) -> dict[str, Any]:
    return {
        attribute.name: helper.get_attribute_value(attribute)
        for attribute in node.attribute
    }


def _initializer_map(model: ModelProto) -> dict[str, TensorProto]:
    return {value.name: value for value in model.graph.initializer}


def _single_consumer(nodes: Iterable[NodeProto], tensor_name: str) -> NodeProto | None:
    consumers = [node for node in nodes if tensor_name in node.input]
    return consumers[0] if len(consumers) == 1 else None


def _unsupported(node: NodeProto, reason: str) -> ValueError:
    return ValueError(
        f"unsupported ConvTranspose '{node.name or node.output[0]}': {reason}"
    )


def _decompose_convtranspose_bn(model: ModelProto) -> int:
    nodes = list(model.graph.node)
    initializers = _initializer_map(model)
    graph_outputs = {value.name for value in model.graph.output}
    replacements: dict[int, list[NodeProto]] = {}
    additions: list[TensorProto] = []

    for index, node in enumerate(nodes):
        if node.op_type != "ConvTranspose":
            continue
        if len(node.output) != 1:
            raise _unsupported(node, "multiple outputs are not supported")
        if node.output[0] in graph_outputs:
            raise _unsupported(node, "pre-BatchNormalization value is a graph output")
        bn = _single_consumer(nodes, node.output[0])
        if bn is None or bn.op_type != "BatchNormalization":
            raise _unsupported(
                node, "requires one following BatchNormalization consumer"
            )
        if not bn.input or bn.input[0] != node.output[0]:
            raise _unsupported(node, "must feed the data input of BatchNormalization")
        if len(bn.output) != 1 or len(bn.input) < 5:
            raise _unsupported(node, "following BatchNormalization is malformed")

        attrs = _attributes(node)
        if attrs.get("group", 1) != 1:
            raise _unsupported(node, "group must be 1")
        if attrs.get("auto_pad", b"NOTSET") not in (b"NOTSET", "NOTSET"):
            raise _unsupported(node, "auto_pad must be NOTSET")
        if "output_shape" in attrs:
            raise _unsupported(node, "output_shape is not supported")
        if any(attrs.get("output_padding", [0, 0])):
            raise _unsupported(node, "output_padding must be zero")
        if list(attrs.get("pads", [0, 0, 0, 0])) != [0, 0, 0, 0]:
            raise _unsupported(node, "pads must all be zero")
        if list(attrs.get("dilations", [1, 1])) != [1, 1]:
            raise _unsupported(node, "dilations must all be one")

        if len(node.input) < 2 or node.input[1] not in initializers:
            raise _unsupported(node, "weights must be a constant initializer")
        weights = numpy_helper.to_array(initializers[node.input[1]])
        if weights.ndim != 4:
            raise _unsupported(node, "only 2D rank-four weights are supported")
        kernel = list(attrs.get("kernel_shape", weights.shape[2:]))
        strides = list(attrs.get("strides", [1, 1]))
        if kernel != strides:
            raise _unsupported(node, "kernel_shape must equal strides")
        if len(kernel) != 2 or kernel[0] != kernel[1]:
            raise _unsupported(node, "DepthToSpace requires equal 2D strides")

        required_bn = list(bn.input[1:5])
        if any(name not in initializers for name in required_bn):
            raise _unsupported(
                node, "following BatchNormalization parameters must be initializers"
            )
        scale, beta, mean, variance = (
            numpy_helper.to_array(initializers[name]) for name in required_bn
        )
        input_channels, output_channels, _, _ = weights.shape
        if any(
            value.shape != (output_channels,) for value in (scale, beta, mean, variance)
        ):
            raise _unsupported(node, "BatchNormalization channel shape mismatch")

        if len(node.input) >= 3 and node.input[2]:
            if node.input[2] not in initializers:
                raise _unsupported(node, "bias must be a constant initializer")
            bias = numpy_helper.to_array(initializers[node.input[2]])
        else:
            bias = np.zeros(output_channels, dtype=weights.dtype)
        if bias.shape != (output_channels,):
            raise _unsupported(node, "bias channel shape mismatch")

        bn_attrs = _attributes(bn)
        if int(bn_attrs.get("training_mode", 0)) != 0:
            raise _unsupported(
                node, "following BatchNormalization must be in inference mode"
            )
        epsilon = float(bn_attrs.get("epsilon", 1e-5))
        alpha = scale / np.sqrt(variance + epsilon)
        folded_weights = weights * alpha.reshape(1, output_channels, 1, 1)
        conv_weights = folded_weights.transpose(2, 3, 1, 0).reshape(
            output_channels * kernel[0] * kernel[1], input_channels, 1, 1
        )
        folded_bias = (bias - mean) * alpha + beta
        conv_bias = np.tile(folded_bias, kernel[0] * kernel[1])

        prefix = node.name or f"convtranspose_{index}"
        weight_name = f"{prefix}.tvm_pre_shuffle.weight"
        bias_name = f"{prefix}.tvm_pre_shuffle.bias"
        conv_output = f"{node.output[0]}.tvm_pre_shuffle"
        additions.extend(
            [
                numpy_helper.from_array(
                    conv_weights.astype(weights.dtype), weight_name
                ),
                numpy_helper.from_array(conv_bias.astype(weights.dtype), bias_name),
            ]
        )
        replacements[index] = [
            helper.make_node(
                "Conv",
                [node.input[0], weight_name, bias_name],
                [conv_output],
                name=f"{prefix}.tvm_conv1x1",
                kernel_shape=[1, 1],
                strides=[1, 1],
                pads=[0, 0, 0, 0],
            ),
            helper.make_node(
                "DepthToSpace",
                [conv_output],
                list(bn.output),
                name=f"{prefix}.tvm_depth_to_space",
                blocksize=kernel[0],
                mode="DCR",
            ),
        ]
        replacements[nodes.index(bn)] = []

    if not replacements:
        return 0
    rewritten: list[NodeProto] = []
    for index, node in enumerate(nodes):
        rewritten.extend(replacements.get(index, [node]))
    del model.graph.node[:]
    model.graph.node.extend(rewritten)
    model.graph.initializer.extend(additions)
    return sum(1 for nodes_for_index in replacements.values() if nodes_for_index)


def _prune_unused_initializers(model: ModelProto) -> int:
    used_names = {name for node in model.graph.node for name in node.input}
    used_names.update(value.name for value in model.graph.input)
    used_names.update(value.name for value in model.graph.output)
    retained = [
        initializer
        for initializer in model.graph.initializer
        if initializer.name in used_names
    ]
    removed = len(model.graph.initializer) - len(retained)
    if removed:
        del model.graph.initializer[:]
        model.graph.initializer.extend(retained)
    return removed


def _ort_optimize(source: Path, destination: Path) -> None:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    options.optimized_model_filepath = str(destination)
    ort.InferenceSession(
        str(source), sess_options=options, providers=["CPUExecutionProvider"]
    )
    if not destination.is_file():
        raise RuntimeError("ONNX Runtime did not write the optimized model")


def _node_count(model: ModelProto) -> int:
    return len(model.graph.node)


def _op_counts(model: ModelProto) -> dict[str, int]:
    return dict(sorted(Counter(node.op_type for node in model.graph.node).items()))


def _sample_input(input_meta: ort.NodeArg, rng: np.random.Generator) -> np.ndarray:
    shape = [
        int(dimension) if isinstance(dimension, int) and dimension > 0 else 1
        for dimension in input_meta.shape
    ]
    dtypes = {
        "tensor(float)": np.float32,
        "tensor(double)": np.float64,
        "tensor(float16)": np.float16,
    }
    if input_meta.type not in dtypes:
        raise ValueError(
            f"numerical equivalence does not support input type {input_meta.type}"
        )
    return rng.normal(size=shape).astype(dtypes[input_meta.type])


def _check_equivalence(
    source: Path,
    prepared: Path,
    *,
    seed: int,
    samples: int,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    if samples <= 0:
        raise ValueError("--samples must be positive")
    source_session = ort.InferenceSession(
        str(source), providers=["CPUExecutionProvider"]
    )
    prepared_session = ort.InferenceSession(
        str(prepared), providers=["CPUExecutionProvider"]
    )
    source_inputs = source_session.get_inputs()
    prepared_inputs = prepared_session.get_inputs()
    if [(item.name, item.type) for item in source_inputs] != [
        (item.name, item.type) for item in prepared_inputs
    ]:
        raise ValueError("prepared model input contract differs from source")
    source_outputs = source_session.get_outputs()
    prepared_outputs = prepared_session.get_outputs()
    if [(item.name, item.type) for item in source_outputs] != [
        (item.name, item.type) for item in prepared_outputs
    ]:
        raise ValueError("prepared model output contract differs from source")

    rng = np.random.default_rng(seed)
    max_absolute_error = 0.0
    max_relative_error = 0.0
    for sample_index in range(samples):
        feeds = {item.name: _sample_input(item, rng) for item in source_inputs}
        expected = source_session.run(None, feeds)
        actual = prepared_session.run(None, feeds)
        if len(expected) != len(actual):
            raise ValueError("prepared model output count differs from source")
        for output_index, (left, right) in enumerate(zip(expected, actual)):
            if left.shape != right.shape:
                raise ValueError(
                    f"output {output_index} shape differs on sample {sample_index}"
                )
            difference = np.abs(left - right)
            if difference.size:
                max_absolute_error = max(max_absolute_error, float(difference.max()))
                denominator = np.maximum(np.abs(left), atol)
                max_relative_error = max(
                    max_relative_error, float((difference / denominator).max())
                )
            if not np.allclose(left, right, rtol=rtol, atol=atol, equal_nan=False):
                raise ValueError(
                    "numerical equivalence failed for "
                    f"sample {sample_index}, output {output_index}"
                )
    return {
        "status": "passed",
        "samples": samples,
        "seed": seed,
        "rtol": rtol,
        "atol": atol,
        "max_absolute_error": max_absolute_error,
        "max_relative_error": max_relative_error,
    }


def prepare_source(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    base_report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "status": "failed",
        "source_onnx_path": str(args.source_onnx.resolve()),
        "source_export_report_path": str(args.source_export_report.resolve()),
        "output_onnx_path": str(args.output_onnx.resolve()),
    }
    if args.source_onnx.is_file():
        base_report["input_sha256"] = sha256_file(args.source_onnx)
    if args.source_export_report.is_file():
        base_report["source_export_report_sha256"] = sha256_file(
            args.source_export_report
        )
    try:
        _validate_source(args)
        input_sha = sha256_file(args.source_onnx)
        source_model = onnx.load(str(args.source_onnx))
        onnx.checker.check_model(source_model)

        args.output_onnx.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".fcooper_tvm_prepare_", dir=args.output_onnx.parent
        ) as temp_dir:
            temp_root = Path(temp_dir)
            rewritten_path = temp_root / "rewritten.onnx"
            optimized_path = temp_root / "optimized.onnx"
            decomposed = (
                _decompose_convtranspose_bn(source_model)
                if args.decompose_convtranspose
                else 0
            )
            pruned_initializers = _prune_unused_initializers(source_model)
            onnx.checker.check_model(source_model)
            onnx.save(source_model, rewritten_path)
            _ort_optimize(rewritten_path, optimized_path)
            prepared_model = onnx.load(str(optimized_path))
            onnx.checker.check_model(prepared_model)
            equivalence = _check_equivalence(
                args.source_onnx,
                optimized_path,
                seed=args.seed,
                samples=args.samples,
                rtol=args.rtol,
                atol=args.atol,
            )
            result = {
                **base_report,
                "status": "success",
                "input_sha256": input_sha,
                "output_sha256": sha256_file(optimized_path),
                "source_export_report_sha256": sha256_file(args.source_export_report),
                "node_counts": {
                    "input": _node_count(onnx.load(str(args.source_onnx))),
                    "output": _node_count(prepared_model),
                },
                "op_counts": {
                    "input": _op_counts(onnx.load(str(args.source_onnx))),
                    "output": _op_counts(prepared_model),
                },
                "transformations": {
                    "ort_basic_simplification_and_constant_folding": True,
                    "convtranspose_bn_decomposition_enabled": bool(
                        args.decompose_convtranspose
                    ),
                    "convtranspose_bn_decomposed": decomposed,
                    "unused_initializers_pruned": pruned_initializers,
                },
                "numerical_equivalence": equivalence,
                "elapsed_seconds": time.perf_counter() - started,
            }
            os.replace(optimized_path, args.output_onnx)
        _write_report(args.report, result)
        return result
    except Exception as error:
        failure = {
            **base_report,
            "error": str(error),
            "error_type": type(error).__name__,
            "elapsed_seconds": time.perf_counter() - started,
        }
        _write_report(args.report, failure)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-onnx", type=Path, required=True)
    parser.add_argument("--source-export-report", type=Path, required=True)
    parser.add_argument("--source-export-report-sha256", required=True)
    parser.add_argument("--output-onnx", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--decompose-convtranspose", action="store_true")
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-5)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(prepare_source(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
