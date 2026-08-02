#!/usr/bin/env python3
"""Run the E1 2x2x2 INT8 scale-attribution matrix as an ONNX-QDQ pre-screen."""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import socket
import time
from pathlib import Path
from typing import Any

import numpy as np

import stage2_codriving_int8_math_oracle_v1 as oracle
import stage2_codriving_mixed_auto_lowering_v1 as lowering


SCHEMA = "codriving_int8_accuracy_matrix_v1"
SIGNED_INT8_MIN = -127
SIGNED_INT8_MAX = 127


def build_tvm_candidate_calibration(
    base_calibration: dict[str, Any],
    *,
    row: dict[str, Any],
    selected_node_ids: list[str],
    parent_sha256: str,
) -> dict[str, Any]:
    if row.get("weight_scale_granularity") != "per_tensor":
        raise ValueError("current TVM candidate manifest supports per-tensor weight scale only")
    if len(selected_node_ids) != 2:
        raise ValueError("E1 TVM candidate requires exactly two selected Conv IDs")
    candidate = copy.deepcopy(base_calibration)
    candidate.update(
        {
            "schema": "codriving_int8_accuracy_matrix_calibration_v1",
            "quantization_semantics": "symmetric_percentile_int8_per_raw_onnx_node",
            "parent_calibration_sha256": parent_sha256,
            "diagnostic_point_id": str(row["point_id"]),
            "trusted_for_final_frontier": False,
        }
    )
    nodes = candidate.get("nodes") or {}
    scales = [float(row["first_conv_input_scale"]), float(row["second_conv_input_scale"])]
    for node_id, scale in zip(selected_node_ids, scales):
        if node_id not in nodes:
            raise ValueError(f"base calibration lacks selected node: {node_id}")
        nodes[node_id] = {**nodes[node_id], "input_scale": scale}
    return candidate


def activation_scale(values: list[np.ndarray], *, strategy: str, percentile: float) -> float:
    if not values:
        raise ValueError("activation values must not be empty")
    absolute = np.concatenate([np.abs(np.asarray(value, dtype="float32")).reshape(-1) for value in values])
    if strategy == "absmax":
        bound = float(absolute.max(initial=0.0))
    elif strategy == "percentile":
        if not 0.0 < percentile <= 100.0:
            raise ValueError("percentile must be in (0, 100]")
        bound = float(np.percentile(absolute, percentile))
    else:
        raise ValueError(f"unsupported activation scale strategy: {strategy}")
    return max(bound, np.finfo("float32").tiny) / 127.0


def weight_scales(weight: Any, *, granularity: str) -> np.ndarray:
    value = np.asarray(weight, dtype="float32")
    if value.ndim != 4:
        raise ValueError("weight must be OIHW rank-4")
    if granularity == "per_tensor":
        bound = np.asarray(np.max(np.abs(value)), dtype="float32")
    elif granularity == "per_output_channel":
        bound = np.max(np.abs(value), axis=(1, 2, 3)).astype("float32")
    else:
        raise ValueError(f"unsupported weight granularity: {granularity}")
    return np.maximum(bound, np.finfo("float32").tiny) / np.float32(127.0)


def quantize_weight(weight: Any, scales: Any) -> np.ndarray:
    value = np.asarray(weight, dtype="float32")
    scale_array = np.asarray(scales, dtype="float32")
    divisor = scale_array if scale_array.ndim == 0 else scale_array.reshape(-1, 1, 1, 1)
    if scale_array.ndim not in (0, 1) or (scale_array.ndim == 1 and len(scale_array) != value.shape[0]):
        raise ValueError("weight scales must be scalar or one value per output channel")
    return np.clip(np.rint(value / divisor), -127.0, 127.0).astype("int8")


def dequantize_weight(weight: Any, scales: Any) -> np.ndarray:
    quantized = quantize_weight(weight, scales).astype("float32")
    scale_array = np.asarray(scales, dtype="float32")
    multiplier = scale_array if scale_array.ndim == 0 else scale_array.reshape(-1, 1, 1, 1)
    return quantized * multiplier


def quantize_dequantize_activation(values: Any, *, scale: float) -> tuple[np.ndarray, dict[str, Any]]:
    reference = np.asarray(values, dtype="float32")
    quantized, quant_stats = oracle.quantize_s8_reference(reference, scale=scale)
    dequantized = quantized.astype("float32") * np.float32(scale)
    difference = dequantized - reference
    denominator = float(np.linalg.norm(reference.reshape(-1)) * np.linalg.norm(dequantized.reshape(-1)))
    cosine = float(np.dot(reference.reshape(-1), dequantized.reshape(-1)) / denominator) if denominator else 1.0
    return dequantized, {
        **quant_stats,
        "mse": float(np.mean(np.square(difference), dtype="float64")),
        "max_abs_error": float(np.max(np.abs(difference), initial=0.0)),
        "cosine_similarity": max(-1.0, min(1.0, cosine)),
    }


def aggregate_multisample_output_errors(
    reference_outputs: list[list[np.ndarray]],
    candidate_outputs: list[list[np.ndarray]],
    *,
    max_mean_abs_error: float,
) -> tuple[list[dict[str, Any]], bool]:
    if len(reference_outputs) != len(candidate_outputs):
        raise ValueError("reference and candidate sample counts differ")
    if not reference_outputs:
        raise ValueError("at least one numerical sample is required")
    output_count = len(reference_outputs[0])
    if any(len(item) != output_count for item in [*reference_outputs, *candidate_outputs]):
        raise ValueError("reference and candidate output counts differ")
    reports = []
    for output_index in range(output_count):
        total_abs_error = 0.0
        total_elements = 0
        max_abs_error = 0.0
        per_sample_means = []
        shape = None
        for sample_index, (reference_sample, candidate_sample) in enumerate(
            zip(reference_outputs, candidate_outputs)
        ):
            reference = np.asarray(reference_sample[output_index], dtype="float32")
            candidate = np.asarray(candidate_sample[output_index], dtype="float32")
            if reference.shape != candidate.shape:
                raise ValueError(f"output shape mismatch at sample {sample_index}, output {output_index}")
            shape = list(reference.shape)
            difference = np.abs(reference - candidate)
            total_abs_error += float(np.sum(difference, dtype="float64"))
            total_elements += int(difference.size)
            max_abs_error = max(max_abs_error, float(difference.max(initial=0.0)))
            per_sample_means.append(float(np.mean(difference, dtype="float64")))
        reports.append(
            {
                "output": output_index,
                "shape": shape,
                "sample_count": len(reference_outputs),
                "max_abs_error": max_abs_error,
                "mean_abs_error": total_abs_error / total_elements if total_elements else 0.0,
                "worst_sample_mean_abs_error": max(per_sample_means, default=0.0),
                "per_sample_mean_abs_error": per_sample_means,
            }
        )
    passed = all(item["worst_sample_mean_abs_error"] <= max_mean_abs_error for item in reports)
    return reports, passed


def _load_calibration_samples(input_npz: Path, expected_shape: list[int]) -> list[np.ndarray]:
    with np.load(input_npz, allow_pickle=False) as payload:
        values = np.asarray(payload["spatial_features"], dtype="float32")
    expected = tuple(map(int, expected_shape))
    if values.shape == expected:
        return [values]
    if values.ndim == len(expected) + 1 and tuple(values.shape[1:]) == expected:
        return [np.asarray(value, dtype="float32") for value in values]
    raise ValueError(f"calibration shape {values.shape} does not contain samples shaped {expected}")


def _augment_output(model: Any, tensor_name: str, shape: list[int]) -> Any:
    import onnx
    from onnx import helper

    augmented = copy.deepcopy(model)
    if tensor_name not in {str(item.name) for item in augmented.graph.output}:
        augmented.graph.output.append(helper.make_tensor_value_info(tensor_name, onnx.TensorProto.FLOAT, shape))
    return augmented


def _run_onnx(model: Any, input_name: str, samples: list[np.ndarray], output_names: list[str]) -> list[list[np.ndarray]]:
    import onnxruntime as ort

    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return [
        [np.asarray(value, dtype="float32") for value in session.run(output_names, {input_name: sample})]
        for sample in samples
    ]


def _replace_initializer(model: Any, name: str, value: np.ndarray) -> None:
    from onnx import numpy_helper

    for index, initializer in enumerate(model.graph.initializer):
        if str(initializer.name) == name:
            model.graph.initializer[index].CopyFrom(numpy_helper.from_array(np.asarray(value, dtype="float32"), name=name))
            return
    raise ValueError(f"initializer not found: {name}")


def _qdq_model(
    base_model: Any,
    configs: list[dict[str, Any]],
) -> Any:
    import onnx
    from onnx import helper, numpy_helper

    model = copy.deepcopy(base_model)
    config_by_index = {int(config["node_index"]): config for config in configs}
    new_nodes = []
    new_initializers = []
    for node_index, node in enumerate(model.graph.node):
        config = config_by_index.get(node_index)
        if config is None:
            new_nodes.append(node)
            continue
        input_name = str(node.input[0])
        prefix = f"e1_rank{config['rank']}_node{node_index}"
        scale_name = f"{prefix}_activation_scale"
        zero_name = f"{prefix}_activation_zero"
        quantized_unclipped_name = f"{prefix}_activation_s8_unclipped"
        quantized_name = f"{prefix}_activation_s8"
        dequantized_name = f"{prefix}_activation_dq"
        clip_min_name = f"{prefix}_clip_min"
        clip_max_name = f"{prefix}_clip_max"
        new_initializers.extend(
            [
                numpy_helper.from_array(np.asarray(config["activation_scale"], dtype="float32"), name=scale_name),
                numpy_helper.from_array(np.asarray(0, dtype="int8"), name=zero_name),
                numpy_helper.from_array(np.asarray(SIGNED_INT8_MIN, dtype="int8"), name=clip_min_name),
                numpy_helper.from_array(np.asarray(SIGNED_INT8_MAX, dtype="int8"), name=clip_max_name),
            ]
        )
        new_nodes.extend(
            [
                helper.make_node(
                    "QuantizeLinear",
                    [input_name, scale_name, zero_name],
                    [quantized_unclipped_name],
                    name=f"{prefix}_Q",
                ),
                helper.make_node(
                    "Clip",
                    [quantized_unclipped_name, clip_min_name, clip_max_name],
                    [quantized_name],
                    name=f"{prefix}_ClipSigned127",
                ),
                helper.make_node("DequantizeLinear", [quantized_name, scale_name, zero_name], [dequantized_name], name=f"{prefix}_DQ"),
            ]
        )
        rewritten = copy.deepcopy(node)
        rewritten.input[0] = dequantized_name
        new_nodes.append(rewritten)
        _replace_initializer(model, str(node.input[1]), np.asarray(config["dequantized_weight"], dtype="float32"))
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    model.graph.initializer.extend(new_initializers)
    onnx.checker.check_model(model)
    return model


def _weight_metrics(weight: np.ndarray, dequantized: np.ndarray) -> dict[str, Any]:
    squared = np.square(np.asarray(weight, dtype="float32") - np.asarray(dequantized, dtype="float32"))
    channel_mse = np.mean(squared, axis=(1, 2, 3), dtype="float64")
    return {
        "mse": float(np.mean(squared, dtype="float64")),
        "per_output_channel_mse_mean": float(np.mean(channel_mse)),
        "per_output_channel_mse_max": float(np.max(channel_mse)),
    }


def run_matrix(
    *,
    onnx_path: Path,
    input_npz: Path,
    width: str,
    percentile: float,
    max_mean_abs_error: float,
) -> dict[str, Any]:
    import onnx
    from onnx import numpy_helper

    started = time.time()
    base_model = onnx.shape_inference.infer_shapes(onnx.load(str(onnx_path)))
    records = lowering.collect_onnx_conv_records(onnx_path)
    ranked = sorted(records, key=lambda record: (-int(record["macs"]), str(record["node_id"])))[:2]
    if len(ranked) != 2:
        raise ValueError("E1 matrix requires at least two Conv records")
    initializer_names = {str(item.name) for item in base_model.graph.initializer}
    runtime_inputs = [item for item in base_model.graph.input if str(item.name) not in initializer_names]
    if len(runtime_inputs) != 1:
        raise ValueError("E1 matrix currently requires one runtime input")
    input_name = str(runtime_inputs[0].name)
    input_shape = [int(dim.dim_value) for dim in runtime_inputs[0].type.tensor_type.shape.dim]
    samples = _load_calibration_samples(input_npz, input_shape)
    output_names = [str(item.name) for item in base_model.graph.output]
    raw_outputs = _run_onnx(base_model, input_name, samples, output_names)

    first_input_values = samples
    second_record = ranked[1]
    second_input_name = str(second_record["input_name"])
    raw_capture_model = _augment_output(base_model, second_input_name, list(map(int, second_record["input_shape"])))
    raw_second_values = [item[0] for item in _run_onnx(raw_capture_model, input_name, samples, [second_input_name])]
    initializer_map = {str(item.name): numpy_helper.to_array(item) for item in base_model.graph.initializer}
    raw_weights = {
        str(record["node_id"]): np.asarray(initializer_map[str(record["weight_name"])], dtype="float32")
        for record in ranked
    }

    rows = []
    for activation_strategy, granularity in itertools.product(
        ("absmax", "percentile"),
        ("per_tensor", "per_output_channel"),
    ):
        first_scale = activation_scale(first_input_values, strategy=activation_strategy, percentile=percentile)
        prepared_weights = {}
        for record in ranked:
            node_id = str(record["node_id"])
            scales = weight_scales(raw_weights[node_id], granularity=granularity)
            prepared_weights[node_id] = {
                "scales": scales,
                "dequantized": dequantize_weight(raw_weights[node_id], scales),
            }
        prefix_model = _qdq_model(
            base_model,
            [
                {
                    "rank": 1,
                    "node_index": ranked[0]["node_index"],
                    "activation_scale": first_scale,
                    "dequantized_weight": prepared_weights[str(ranked[0]["node_id"])]["dequantized"],
                }
            ],
        )
        prefix_capture_model = _augment_output(
            prefix_model,
            second_input_name,
            list(map(int, second_record["input_shape"])),
        )
        region_second_values = [
            item[0] for item in _run_onnx(prefix_capture_model, input_name, samples, [second_input_name])
        ]
        for calibration_source, second_values in (
            ("raw_fp32", raw_second_values),
            ("region_aware", region_second_values),
        ):
            second_scale = activation_scale(second_values, strategy=activation_strategy, percentile=percentile)
            configs = []
            for rank, record, scale in zip((1, 2), ranked, (first_scale, second_scale)):
                configs.append(
                    {
                        "rank": rank,
                        "node_index": record["node_index"],
                        "activation_scale": scale,
                        "dequantized_weight": prepared_weights[str(record["node_id"])]["dequantized"],
                    }
                )
            variant = _qdq_model(base_model, configs)
            candidate_outputs = _run_onnx(variant, input_name, samples, output_names)
            output_errors, numerical_pass = aggregate_multisample_output_errors(
                raw_outputs,
                candidate_outputs,
                max_mean_abs_error=max_mean_abs_error,
            )
            _, second_activation_metrics = quantize_dequantize_activation(second_values[0], scale=second_scale)
            rows.append(
                {
                    "point_id": f"a_{activation_strategy}__w_{granularity}__c_{calibration_source}",
                    "activation_scale_strategy": activation_strategy,
                    "activation_percentile": percentile if activation_strategy == "percentile" else None,
                    "weight_scale_granularity": granularity,
                    "calibration_source": calibration_source,
                    "first_conv_input_scale": first_scale,
                    "second_conv_input_scale": second_scale,
                    "second_conv_activation_metrics_sample0": second_activation_metrics,
                    "weight_metrics": {
                        str(record["node_id"]): _weight_metrics(
                            raw_weights[str(record["node_id"])],
                            prepared_weights[str(record["node_id"])]["dequantized"],
                        )
                        for record in ranked
                    },
                    "three_output_errors_vs_raw_fp32_onnx_all_calibration_samples": output_errors,
                    "numerical_gate_passed": numerical_pass,
                    "latency_ms_p50": None,
                    "trusted_for_final_frontier": False,
                }
            )
    rows.sort(key=lambda row: row["point_id"])
    return {
        "schema": SCHEMA,
        "status": "completed_reference_prescreen",
        "trusted_for_final_frontier": False,
        "scope": "E1_reference_prescreen_before_same_schedule_TVM_measurement",
        "host": socket.gethostname(),
        "width": width,
        "onnx": str(onnx_path),
        "onnx_sha256": lowering.sha256_file(onnx_path),
        "input_npz": str(input_npz),
        "input_npz_sha256": lowering.sha256_file(input_npz),
        "calibration_sample_count": len(samples),
        "selected_conv_records": ranked,
        "max_mean_abs_error": max_mean_abs_error,
        "point_count": len(rows),
        "passing_point_count": sum(bool(row["numerical_gate_passed"]) for row in rows),
        "rows": rows,
        "elapsed_s": time.time() - started,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--width", required=True)
    parser.add_argument("--percentile", type=float, default=99.99)
    parser.add_argument("--max-mean-abs-error", type=float, default=0.05)
    parser.add_argument("--base-per-node-calibration", type=Path, default=None)
    parser.add_argument("--tvm-candidate-point-id", default=None)
    parser.add_argument("--out-tvm-candidate-calibration", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.onnx, args.input_npz):
        if not path.is_file():
            parser.error(f"input file not found: {path}")
    candidate_args = (
        args.base_per_node_calibration,
        args.tvm_candidate_point_id,
        args.out_tvm_candidate_calibration,
    )
    if any(value is not None for value in candidate_args) and not all(value is not None for value in candidate_args):
        parser.error("TVM candidate emission requires base calibration, point ID, and output path")
    if args.base_per_node_calibration is not None and not args.base_per_node_calibration.is_file():
        parser.error(f"base calibration not found: {args.base_per_node_calibration}")
    return args


def main() -> int:
    args = parse_args()
    report = run_matrix(
        onnx_path=args.onnx,
        input_npz=args.input_npz,
        width=args.width,
        percentile=args.percentile,
        max_mean_abs_error=args.max_mean_abs_error,
    )
    if args.out_tvm_candidate_calibration is not None:
        matching_rows = [row for row in report["rows"] if row["point_id"] == args.tvm_candidate_point_id]
        if len(matching_rows) != 1:
            raise ValueError(f"TVM candidate point not found: {args.tvm_candidate_point_id}")
        if not matching_rows[0]["numerical_gate_passed"]:
            raise ValueError(f"TVM candidate point did not pass the all-sample numerical gate: {args.tvm_candidate_point_id}")
        base_calibration = json.loads(args.base_per_node_calibration.read_text(encoding="utf-8"))
        candidate = build_tvm_candidate_calibration(
            base_calibration,
            row=matching_rows[0],
            selected_node_ids=[str(item["node_id"]) for item in report["selected_conv_records"]],
            parent_sha256=lowering.sha256_file(args.base_per_node_calibration),
        )
        args.out_tvm_candidate_calibration.parent.mkdir(parents=True, exist_ok=True)
        args.out_tvm_candidate_calibration.write_text(
            json.dumps(candidate, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
