#!/usr/bin/env python3
"""Automatic source-Conv MetaSchedule closure for the first two CoDriving Conv nodes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import socket
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "codriving_source_conv_auto_tune_v1"
LOWERING_ORIGIN = "source_ir_automatic"
FORBIDDEN_SOURCE_TOKENS = (
    "x_col",
    "w_mat",
    "im2col",
    'name="matmul"',
    "@T.prim_func",
    "T.prim_func",
    "R.call_tir",
    "relax.call_tir",
)


@dataclass(frozen=True)
class ConvAnchor:
    anchor_id: str
    node_id: str
    input_shape: tuple[int, int, int, int]
    weight_shape: tuple[int, int, int, int]
    output_shape: tuple[int, int, int, int]
    strides: tuple[int, int]
    pads: tuple[int, int, int, int]
    group: int = 1


ANCHORS = (
    ConvAnchor(
        anchor_id="rank1_conv",
        node_id="onnx_conv_0000_/resnet/layer0/layer0.0/conv1/Conv",
        input_shape=(2, 64, 256, 512),
        weight_shape=(32, 64, 3, 3),
        output_shape=(2, 32, 128, 256),
        strides=(2, 2),
        pads=(1, 1, 1, 1),
    ),
    ConvAnchor(
        anchor_id="rank2_conv",
        node_id="onnx_conv_0002_/resnet/layer0/layer0.0/conv2/Conv",
        input_shape=(2, 32, 128, 256),
        weight_shape=(32, 32, 3, 3),
        output_shape=(2, 32, 128, 256),
        strides=(1, 1),
        pads=(1, 1, 1, 1),
    ),
)
ANCHOR_BY_ID = {anchor.anchor_id: anchor for anchor in ANCHORS}
REALIZATIONS = {
    "fp16_rank1": ("fp16_complete", "rank1_conv"),
    "fp16_rank2": ("fp16_complete", "rank2_conv"),
    "int8_core_rank1": ("int8_core", "rank1_conv"),
    "int8_core_rank2": ("int8_core", "rank2_conv"),
    "int8_complete_rank1": ("int8_complete", "rank1_conv"),
    "int8_complete_rank2": ("int8_complete", "rank2_conv"),
    "fp16_rank2_region": ("fp16_region", "rank2_region"),
    "int8_rank2_region": ("int8_region", "rank2_region"),
}


def expected_tensorcore_conv_count(realization: str) -> int:
    return 2 if REALIZATIONS[realization][1] == "rank2_region" else 1


def region_structural_signature(realization: str) -> dict[str, Any]:
    kind, anchor_ref = REALIZATIONS[realization]
    if anchor_ref != "rank2_region":
        raise ValueError(f"not a region realization: {realization}")
    first, second = ANCHORS
    quantization_ops = []
    if kind == "int8_region":
        quantization_ops = ["input_quantize", "conv1_dequantize", "requantize", "conv2_dequantize"]
    return {
        "anchors": [first.anchor_id, second.anchor_id],
        "region_input_shape": list(first.input_shape),
        "region_output_shape": list(second.output_shape),
        "semantic_ops": ["conv1", "bias1", "relu", "conv2", "bias2"],
        "quantization_ops": quantization_ops,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(root: Path) -> str:
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"tuning database contains no files: {root}")
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def validate_source_ir_text(text: str) -> None:
    if "R.nn.conv2d" not in text and "relax.nn.conv2d" not in text:
        raise ValueError("source IR lacks a high-level Conv operator")
    for token in FORBIDDEN_SOURCE_TOKENS:
        if token in text:
            raise ValueError(f"source IR contains explicit TE/im2col construct: {token}")


def classify_tir_products(text: str) -> dict[str, Any]:
    lower = text.lower()
    counts = {
        "tvm_mma_sync": text.count("tvm_mma_sync"),
        "mma_sync": lower.count("mma.sync"),
        "wmma": lower.count("wmma"),
        "ldmatrix": lower.count("ldmatrix"),
        "dp4a": lower.count("dp4a"),
    }
    tensorcore = counts["tvm_mma_sync"] > 0 or counts["mma_sync"] > 0
    if tensorcore:
        path_class = "TENSORCORE_MMA"
    elif counts["dp4a"] > 0:
        path_class = "DP4A"
    else:
        path_class = "SCALAR_OR_FALLBACK"
    conv_functions = []
    for chunk in re.split(r"(?=\s*@T\.prim_func)", text):
        name_match = re.search(r"\bdef\s+([A-Za-z0-9_]+)\s*\(", chunk)
        if name_match is None or "conv2d" not in chunk.lower():
            continue
        function_products = classify_tir_products_without_functions(chunk)
        conv_functions.append(
            {
                "name": name_match.group(1),
                "path_class": function_products["path_class"],
                "tensorcore_evidence": function_products["tensorcore_evidence"],
                "counts": function_products["counts"],
            }
        )
    return {
        "path_class": path_class,
        "tensorcore_evidence": tensorcore,
        "counts": counts,
        "tir_length": len(text),
        "conv_primfunc_count": len(conv_functions),
        "tensorcore_conv_count": sum(bool(item["tensorcore_evidence"]) for item in conv_functions),
        "conv_functions": conv_functions,
    }


def classify_tir_products_without_functions(text: str) -> dict[str, Any]:
    lower = text.lower()
    counts = {
        "tvm_mma_sync": text.count("tvm_mma_sync"),
        "mma_sync": lower.count("mma.sync"),
        "wmma": lower.count("wmma"),
        "ldmatrix": lower.count("ldmatrix"),
        "dp4a": lower.count("dp4a"),
    }
    tensorcore = counts["tvm_mma_sync"] > 0 or counts["mma_sync"] > 0
    path_class = "TENSORCORE_MMA" if tensorcore else "DP4A" if counts["dp4a"] else "SCALAR_OR_FALLBACK"
    return {"path_class": path_class, "tensorcore_evidence": tensorcore, "counts": counts}


def _sha_field_ok(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def automatic_candidate_eligible(result: dict[str, Any]) -> bool:
    products = result.get("tir_products") or {}
    expected_conv_count = int(result.get("expected_tensorcore_conv_count") or 1)
    observed_conv_count = int(products.get("tensorcore_conv_count", int(bool(products.get("tensorcore_evidence")))))
    return bool(
        result.get("lowering_origin") == LOWERING_ORIGIN
        and all(
            _sha_field_ok(result.get(field))
            for field in ("source_ir_sha256", "schedule_trace_sha256", "tuning_database_sha256")
        )
        and result.get("build_status") == "success"
        and result.get("numerical_status") == "pass"
        and bool(products.get("tensorcore_evidence"))
        and observed_conv_count >= expected_conv_count
    )


def search_training_eligible(result: dict[str, Any]) -> bool:
    return bool(
        result.get("action") == "measure"
        and result.get("calibration_kind") == "formal"
        and automatic_candidate_eligible(result)
    )


def load_runtime_support() -> tuple[Any, Any]:
    from stage2_h800_run_measurement_job import configure_tvm_env, wait_gpu_idle

    return configure_tvm_env, wait_gpu_idle


def _anchor_spec(anchor: ConvAnchor) -> dict[str, Any]:
    return {
        "input_shape": list(anchor.input_shape),
        "weight_shape": list(anchor.weight_shape),
        "output_shape": list(anchor.output_shape),
        "strides": list(anchor.strides),
        "pads": list(anchor.pads),
        "group": anchor.group,
    }


def _quantize_expr(bb: Any, relax: Any, value: Any, scale: float) -> Any:
    value_fp32 = bb.emit(relax.op.astype(value, "float32"))
    divided = bb.emit(relax.op.divide(value_fp32, relax.const(np.float32(scale), "float32")))
    rounded = bb.emit(relax.op.round(divided))
    clipped = bb.emit(relax.op.clip(rounded, -127.0, 127.0))
    return bb.emit(relax.op.astype(clipped, "int8"))


def _dequant_bias_fp32(
    bb: Any,
    relax: Any,
    accumulator: Any,
    bias: Any,
    accumulator_scale: float,
) -> Any:
    value = bb.emit(relax.op.astype(accumulator, "float32"))
    value = bb.emit(relax.op.multiply(value, relax.const(np.float32(accumulator_scale), "float32")))
    bias_fp32 = bb.emit(relax.op.astype(bias, "float32"))
    return bb.emit(relax.op.add(value, bias_fp32))


def build_source_relax_module(
    realization: str,
    *,
    scales: dict[str, dict[str, float]],
) -> Any:
    from tvm import relax

    kind, anchor_ref = REALIZATIONS[realization]
    bb = relax.BlockBuilder()
    if kind not in {"fp16_region", "int8_region"}:
        anchor = ANCHOR_BY_ID[anchor_ref]
        input_dtype = "int8" if kind == "int8_core" else "float16"
        weight_dtype = "float16" if kind == "fp16_complete" else "int8"
        x = relax.Var("x", relax.TensorStructInfo(anchor.input_shape, input_dtype))
        weight = relax.Var("weight", relax.TensorStructInfo(anchor.weight_shape, weight_dtype))
        inputs = [x, weight]
        bias = None
        if kind != "int8_core":
            bias = relax.Var("bias", relax.TensorStructInfo((1, anchor.output_shape[1], 1, 1), "float16"))
            inputs.append(bias)
        with bb.function("main", inputs):
            with bb.dataflow():
                conv_input = x
                if kind == "int8_complete":
                    conv_input = _quantize_expr(bb, relax, x, scales[anchor.anchor_id]["input_scale"])
                output_dtype = "float16" if kind == "fp16_complete" else "int32"
                conv = bb.emit(
                    relax.op.nn.conv2d(
                        conv_input,
                        weight,
                        strides=anchor.strides,
                        padding=anchor.pads,
                        groups=anchor.group,
                        out_dtype=output_dtype,
                    )
                )
                if kind == "int8_core":
                    output = conv
                elif kind == "fp16_complete":
                    output = bb.emit(relax.op.add(conv, bias))
                else:
                    output_fp32 = _dequant_bias_fp32(
                        bb,
                        relax,
                        conv,
                        bias,
                        scales[anchor.anchor_id]["input_scale"]
                        * scales[anchor.anchor_id]["weight_scale"],
                    )
                    output = bb.emit(relax.op.astype(output_fp32, "float16"))
                emitted = bb.emit_output(output)
            bb.emit_func_output(emitted)
        return bb.finalize()

    first, second = ANCHORS
    x = relax.Var("x", relax.TensorStructInfo(first.input_shape, "float16"))
    weight_dtype = "float16" if kind == "fp16_region" else "int8"
    weight1 = relax.Var("weight1", relax.TensorStructInfo(first.weight_shape, weight_dtype))
    bias1 = relax.Var("bias1", relax.TensorStructInfo((1, first.output_shape[1], 1, 1), "float16"))
    weight2 = relax.Var("weight2", relax.TensorStructInfo(second.weight_shape, weight_dtype))
    bias2 = relax.Var("bias2", relax.TensorStructInfo((1, second.output_shape[1], 1, 1), "float16"))
    with bb.function("main", [x, weight1, bias1, weight2, bias2]):
        with bb.dataflow():
            if kind == "fp16_region":
                conv1 = bb.emit(
                    relax.op.nn.conv2d(
                        x,
                        weight1,
                        strides=first.strides,
                        padding=first.pads,
                        groups=first.group,
                        out_dtype="float16",
                    )
                )
                conv1_bias = bb.emit(relax.op.add(conv1, bias1))
                relu = bb.emit(relax.op.nn.relu(conv1_bias))
                conv2 = bb.emit(
                    relax.op.nn.conv2d(
                        relu,
                        weight2,
                        strides=second.strides,
                        padding=second.pads,
                        groups=second.group,
                        out_dtype="float16",
                    )
                )
                output = bb.emit(relax.op.add(conv2, bias2))
            else:
                quantized = _quantize_expr(bb, relax, x, scales[first.anchor_id]["input_scale"])
                conv1 = bb.emit(
                    relax.op.nn.conv2d(
                        quantized,
                        weight1,
                        strides=first.strides,
                        padding=first.pads,
                        groups=first.group,
                        out_dtype="int32",
                    )
                )
                conv1_fp32 = _dequant_bias_fp32(
                    bb,
                    relax,
                    conv1,
                    bias1,
                    scales[first.anchor_id]["input_scale"] * scales[first.anchor_id]["weight_scale"],
                )
                requant = _quantize_expr(bb, relax, conv1_fp32, scales[second.anchor_id]["input_scale"])
                relu = bb.emit(relax.op.nn.relu(requant))
                conv2 = bb.emit(
                    relax.op.nn.conv2d(
                        relu,
                        weight2,
                        strides=second.strides,
                        padding=second.pads,
                        groups=second.group,
                        out_dtype="int32",
                    )
                )
                conv2_fp32 = _dequant_bias_fp32(
                    bb,
                    relax,
                    conv2,
                    bias2,
                    scales[second.anchor_id]["input_scale"] * scales[second.anchor_id]["weight_scale"],
                )
                output = bb.emit(relax.op.astype(conv2_fp32, "float16"))
            emitted = bb.emit_output(output)
        bb.emit_func_output(emitted)
    return bb.finalize()


def lower_source_module(mod: Any, target: Any) -> Any:
    import tvm
    from tvm import relax

    sequence = tvm.transform.Sequential(
        [
            relax.transform.LegalizeOps(),
            relax.transform.AnnotateTIROpPattern(),
            relax.transform.FuseOps(),
            relax.transform.FuseTIR(),
        ]
    )
    with target, tvm.transform.PassContext(opt_level=3):
        return sequence(mod)


def apply_database_and_fallback(mod: Any, target: Any, database_dir: Path) -> Any:
    import tvm
    from tvm import relax
    import tvm.s_tir.dlight as dl

    with target, tvm.transform.PassContext(opt_level=3):
        scheduled = relax.transform.MetaScheduleApplyDatabase(work_dir=str(database_dir))(mod)
        return dl.ApplyDefaultSchedule(
            dl.gpu.Matmul(),
            dl.gpu.GEMV(),
            dl.gpu.Reduction(),
            dl.gpu.GeneralReduction(),
            dl.gpu.Fallback(),
        )(scheduled)


def apply_default_fallback(mod: Any, target: Any) -> Any:
    import tvm
    import tvm.s_tir.dlight as dl

    with target, tvm.transform.PassContext(opt_level=3):
        return dl.ApplyDefaultSchedule(
            dl.gpu.Matmul(),
            dl.gpu.GEMV(),
            dl.gpu.Reduction(),
            dl.gpu.GeneralReduction(),
            dl.gpu.Fallback(),
        )(mod)


def _load_real_data(args: argparse.Namespace) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    import stage2_codriving_int8_math_oracle_v1 as oracle
    import stage2_codriving_mixed_auto_lowering_v1 as lowering

    calibration = json.loads(args.per_node_calibration.read_text(encoding="utf-8"))
    records = lowering.collect_onnx_conv_records(args.onnx)
    record_by_id = {str(record["node_id"]): record for record in records}
    onnx_sha = lowering.sha256_file(args.onnx)
    lowering.validate_per_node_calibration_for_lowering(
        calibration,
        records=records,
        onnx_sha256=onnx_sha,
        allow_diagnostic_accuracy_matrix=True,
    )
    selected_records = [record_by_id[anchor.node_id] for anchor in ANCHORS]
    captured, capture_evidence = oracle._capture_conv_inputs(args.onnx, args.input_npz, selected_records)
    specs = oracle._conv_runtime_specs(args.onnx, selected_records)
    scales = {}
    values = {}
    for anchor, record in zip(ANCHORS, selected_records):
        node_scales = lowering.resolve_node_scales(
            calibration,
            node_id=anchor.node_id,
            input_name=str(record["input_name"]),
            weight_name=str(record["weight_name"]),
            onnx_sha256=onnx_sha,
        )
        scales[anchor.anchor_id] = node_scales
        activation_fp16 = np.asarray(captured[str(record["input_name"])], dtype="float16")
        activation_s8, _ = oracle.quantize_s8_reference(
            activation_fp16.astype("float32"),
            scale=node_scales["input_scale"],
        )
        weight_s8, _ = oracle.quantize_s8_reference(
            specs[anchor.node_id]["weight"],
            scale=node_scales["weight_scale"],
        )
        values[anchor.anchor_id] = {
            "activation_fp16": activation_fp16,
            "activation_s8": activation_s8,
            "weight_fp16": np.asarray(specs[anchor.node_id]["weight"], dtype="float16"),
            "weight_s8": weight_s8,
            "bias_fp16": np.asarray(specs[anchor.node_id]["bias"], dtype="float16").reshape(
                1, anchor.output_shape[1], 1, 1
            ),
        }
    return scales, {
        "onnx_sha256": onnx_sha,
        "calibration_sha256": sha256_file(args.per_node_calibration),
        "capture_evidence": capture_evidence,
        "values": values,
    }


def _runtime_inputs(realization: str, values: dict[str, Any]) -> list[np.ndarray]:
    kind, anchor_ref = REALIZATIONS[realization]
    if kind not in {"fp16_region", "int8_region"}:
        value = values[anchor_ref]
        if kind == "fp16_complete":
            return [value["activation_fp16"], value["weight_fp16"], value["bias_fp16"]]
        if kind == "int8_core":
            return [value["activation_s8"], value["weight_s8"]]
        return [value["activation_fp16"], value["weight_s8"], value["bias_fp16"]]
    first = values["rank1_conv"]
    second = values["rank2_conv"]
    if kind == "fp16_region":
        return [
            first["activation_fp16"],
            first["weight_fp16"],
            first["bias_fp16"],
            second["weight_fp16"],
            second["bias_fp16"],
        ]
    return [
        first["activation_fp16"],
        first["weight_s8"],
        first["bias_fp16"],
        second["weight_s8"],
        second["bias_fp16"],
    ]


def _compile(mod: Any, target: Any) -> Any:
    import tvm

    with target, tvm.transform.PassContext(opt_level=3):
        try:
            return tvm.compile(mod, target=target)
        except Exception:
            return tvm.build(mod, target=target)


def _run_vm(executable: Any, dev: Any, inputs: list[np.ndarray]) -> tuple[Any, list[Any]]:
    import tvm
    from tvm import relax

    vm = relax.VirtualMachine(executable, dev)
    runtime_inputs = [tvm.runtime.tensor(value, device=dev) for value in inputs]
    output = vm["main"](*runtime_inputs)
    dev.sync()
    return vm, runtime_inputs, output


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    raise TypeError(f"unexpected VM output: {type(value)!r}")


def _numerical_report(reference: np.ndarray, candidate: np.ndarray, *, exact: bool) -> dict[str, Any]:
    if reference.shape != candidate.shape:
        return {"status": "fail", "reason": "shape_mismatch"}
    difference = np.abs(reference.astype("float64") - candidate.astype("float64"))
    max_error = float(difference.max(initial=0.0))
    mean_error = float(difference.mean()) if difference.size else 0.0
    passed = bool(np.array_equal(reference, candidate)) if exact else mean_error <= 0.05
    return {
        "status": "pass" if passed else "fail",
        "exact_required": exact,
        "max_abs_error": max_error,
        "mean_abs_error": mean_error,
        "shape": list(reference.shape),
    }


def _latency(vm: Any, runtime_inputs: list[Any], dev: Any, args: argparse.Namespace) -> dict[str, Any]:
    for _ in range(args.warmup):
        vm["main"](*runtime_inputs)
    dev.sync()
    evaluator = vm.time_evaluator("main", dev, number=args.number, repeat=args.repeat)
    repeats = [float(value) * 1000.0 for value in evaluator(*runtime_inputs).results]
    return {
        "p50_ms": float(statistics.median(repeats)),
        "p90_ms": float(np.percentile(np.asarray(repeats, dtype="float64"), 90.0)),
        "repeats_ms": repeats,
        "warmup": args.warmup,
        "number": args.number,
        "repeat": args.repeat,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import tvm
    from tvm.s_tir.meta_schedule import relax_integration as ri
    import tvm.s_tir.tensor_intrin.cuda  # noqa: F401
    configure_tvm_env, wait_gpu_idle = load_runtime_support()

    configure_tvm_env(str(args.gpu))
    dev = tvm.cuda(0)
    target = tvm.target.Target.from_device(dev)
    wait_gpu_idle(str(args.gpu), args.artifact_dir / "gpu_idle_gate", timeout_s=args.idle_timeout_s)
    scales, data = _load_real_data(args)
    source_mod = build_source_relax_module(args.realization, scales=scales)
    source_text = source_mod.script()
    validate_source_ir_text(source_text)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    source_path = args.artifact_dir / "source_relax.py"
    source_path.write_text(source_text, encoding="utf-8")
    lowered = lower_source_module(source_mod, target)
    lowered_path = args.artifact_dir / "lowered_source_tir.py"
    lowered_path.write_text(lowered.script(), encoding="utf-8")
    tasks = ri.extract_tasks(lowered, target, params={})
    args.database_dir.mkdir(parents=True, exist_ok=True)
    tune_seconds = None
    if args.action == "tune":
        tune_started = time.time()
        ri.tune_relax(
            mod=lowered,
            params={},
            target=target,
            work_dir=str(args.database_dir),
            max_trials_global=args.trials,
            seed=args.seed,
        )
        tune_seconds = time.time() - tune_started
    scheduled = apply_database_and_fallback(lowered, target, args.database_dir)
    scheduled_text = scheduled.script()
    scheduled_path = args.artifact_dir / "scheduled_tir.py"
    scheduled_path.write_text(scheduled_text, encoding="utf-8")
    tir_products = classify_tir_products(scheduled_text)

    runtime_inputs = _runtime_inputs(args.realization, data["values"])
    default_mod = apply_default_fallback(lowered, target)
    reference_executable = _compile(default_mod, target)
    candidate_executable = _compile(scheduled, target)
    _, _, reference_output = _run_vm(reference_executable, dev, runtime_inputs)
    candidate_vm, candidate_inputs, candidate_output = _run_vm(candidate_executable, dev, runtime_inputs)
    kind = REALIZATIONS[args.realization][0]
    numerical = _numerical_report(
        _to_numpy(reference_output),
        _to_numpy(candidate_output),
        exact=kind == "int8_core",
    )
    latency = _latency(candidate_vm, candidate_inputs, dev, args)
    database_sha = sha256_tree(args.database_dir)
    result = {
        "schema": SCHEMA,
        "status": "diagnostic_success",
        "trusted_for_final_frontier": False,
        "eligible_for_search_training": False,
        "lowering_origin": LOWERING_ORIGIN,
        "action": args.action,
        "realization": args.realization,
        "host": socket.gethostname(),
        "gpu": str(args.gpu),
        "tvm_version": tvm.__version__,
        "target": str(target),
        "source_ir": str(source_path),
        "source_ir_sha256": sha256_file(source_path),
        "lowered_source_tir": str(lowered_path),
        "lowered_source_tir_sha256": sha256_file(lowered_path),
        "schedule_trace": str(scheduled_path),
        "schedule_trace_sha256": sha256_file(scheduled_path),
        "tuning_database": str(args.database_dir),
        "tuning_database_sha256": database_sha,
        "onnx": str(args.onnx),
        "onnx_sha256": data["onnx_sha256"],
        "calibration": str(args.per_node_calibration),
        "calibration_sha256": data["calibration_sha256"],
        "calibration_kind": "diagnostic",
        "task_count": len(tasks),
        "trials": args.trials if args.action == "tune" else None,
        "seed": args.seed,
        "tune_seconds": tune_seconds,
        "tir_products": tir_products,
        "expected_tensorcore_conv_count": expected_tensorcore_conv_count(args.realization),
        "region_structural_signature": (
            region_structural_signature(args.realization)
            if REALIZATIONS[args.realization][1] == "rank2_region"
            else None
        ),
        "build_status": "success",
        "numerical_status": numerical["status"],
        "numerical_vs_default_source_graph": numerical,
        "latency": latency,
        "capture_evidence": data["capture_evidence"],
        "forbidden_source_tokens": list(FORBIDDEN_SOURCE_TOKENS),
        "not_used": [
            "explicit TE im2col",
            "x_col/w_mat PrimFunc construction",
            "MatmulInt8Tensorization applied to a handwritten matmul PrimFunc",
        ],
    }
    result["automatic_schedule_eligible"] = automatic_candidate_eligible(result)
    result["eligible_for_search_training"] = search_training_eligible(result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("tune", "measure"), required=True)
    parser.add_argument("--realization", choices=sorted(REALIZATIONS), required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--per-node-calibration", type=Path, required=True)
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--database-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--number", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--idle-timeout-s", type=int, default=600)
    args = parser.parse_args()
    for path in (args.onnx, args.per_node_calibration, args.input_npz):
        if not path.is_file():
            parser.error(f"input file not found: {path}")
    if args.action == "measure" and not args.database_dir.is_dir():
        parser.error(f"tuning database not found: {args.database_dir}")
    if args.trials <= 0 or args.number <= 0 or args.repeat <= 0:
        parser.error("trials, number, and repeat must be positive")
    return args


def main() -> int:
    args = parse_args()
    started = time.time()
    try:
        result = run(args)
        result = {**result, "elapsed_seconds": time.time() - started}
        exit_code = 0
    except Exception as exc:
        import traceback

        result = {
            "schema": SCHEMA,
            "status": "failed",
            "trusted_for_final_frontier": False,
            "eligible_for_search_training": False,
            "lowering_origin": LOWERING_ORIGIN,
            "action": args.action,
            "realization": args.realization,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.time() - started,
        }
        exit_code = 2
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
