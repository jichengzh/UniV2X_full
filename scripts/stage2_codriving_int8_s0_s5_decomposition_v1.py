#!/usr/bin/env python3
"""Measure the E2 S0-S5 rank-2 INT8 performance decomposition on H800."""

from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

import stage2_codriving_int8_math_oracle_v1 as oracle
import stage2_codriving_mixed_auto_lowering_v1 as lowering


SCHEMA = "codriving_int8_s0_s5_decomposition_v1"


def _sum_p50(items: list[dict[str, Any]]) -> float:
    return float(sum(float(item["latency_ms_p50"]) for item in items))


def summarize_micro_stages(profiles: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    s0 = _sum_p50(profiles["fp16"])
    s1 = _sum_p50(profiles["int8_core"])
    s2 = _sum_p50(profiles["int8_materialized"])
    bridge = _sum_p50(profiles.get("bridge", []))
    s3 = _sum_p50(profiles["int8_epilogue"]) + bridge
    quantize = _sum_p50(profiles["quantize"])
    return {
        "S0": {"realization": "two isolated FP16 TensorCore Conv kernels", "latency_ms_p50": s0},
        "S1": {"realization": "pre-im2col INT8 MMA cores", "latency_ms_p50": s1},
        "S2": {"realization": "INT8 MMA with on-the-fly im2col/layout formation", "latency_ms_p50": s2},
        "S3": {
            "realization": "S2 with fused bias/requant/dequant epilogues and retained INT8 bridge ops",
            "latency_ms_p50": s3,
            "bridge_latency_ms_p50": bridge,
        },
        "S4": {
            "realization": "S3 plus automatic-region activation quantize boundary",
            "latency_ms_p50": s3 + quantize,
            "quantize_latency_ms_p50": quantize,
        },
    }


def classify_first_reversal(stages: dict[str, dict[str, Any]]) -> str:
    baseline = float(stages["S0"]["latency_ms_p50"])
    labels = {
        "S1": "S1_core",
        "S2": "S2_im2col_layout",
        "S3": "S3_epilogue",
        "S4": "S4_activation_quantize",
    }
    for stage in ("S1", "S2", "S3", "S4"):
        if float(stages[stage]["latency_ms_p50"]) >= baseline:
            return labels[stage]
    return "no_micro_reversal_through_S4"


def validate_linked_full_results(
    fp16: dict[str, Any],
    int8: dict[str, Any],
    *,
    onnx_sha256: str,
    width: str,
    gpu: str,
    calibration_sha256: str,
    selected_node_ids: list[str],
) -> dict[str, Any]:
    for label, result in (("FP16", fp16), ("INT8", int8)):
        if result.get("onnx_sha256") != onnx_sha256:
            raise ValueError(f"{label} linked result ONNX mismatch")
        if result.get("width") != width:
            raise ValueError(f"{label} linked result width mismatch")
        if str(result.get("gpu")) != str(gpu):
            raise ValueError(f"{label} linked result GPU mismatch")
        if not bool((result.get("numerical_gate") or {}).get("passed")):
            raise ValueError(f"{label} linked result failed numerical gate")
    if fp16.get("status") != "success" or fp16.get("selected_int8_node_ids") != []:
        raise ValueError("FP16 linked result is not a clean full-graph baseline")
    if int8.get("status") != "diagnostic_only_accuracy_matrix_calibration":
        raise ValueError("INT8 linked result is not the expected diagnostic matrix realization")
    if int8.get("calibration_binding") != "diagnostic_accuracy_matrix_node_id_and_tensor_names":
        raise ValueError("INT8 linked result calibration binding mismatch")
    if (int8.get("calibration_evidence") or {}).get("manifest_sha256") != calibration_sha256:
        raise ValueError("INT8 linked result calibration SHA mismatch")
    if list(int8.get("selected_int8_node_ids") or []) != selected_node_ids:
        raise ValueError("INT8 linked result selected node IDs mismatch")
    if not bool(int8.get("retain_int8_regions")) or not bool(int8.get("fuse_int8_epilogue")):
        raise ValueError("INT8 linked result is not an automatic fused region")
    regions = ((int8.get("region_formation") or {}).get("regions") or [])
    if len(regions) != 1 or list(regions[0].get("node_ids") or []) != selected_node_ids:
        raise ValueError("INT8 linked result region topology mismatch")
    return {
        "onnx_sha256": onnx_sha256,
        "width": width,
        "gpu": str(gpu),
        "calibration_sha256": calibration_sha256,
        "selected_node_ids": selected_node_ids,
        "region_id": regions[0].get("region_id"),
    }


def make_int8_core_primfunc(stack: dict[str, Any], spec: dict[str, Any], name: str) -> Any:
    te = stack["te"]
    n, _, _, _ = map(int, spec["input_shape"])
    cout, cpg, kh, kw = map(int, spec["weight_shape"])
    _, _, oh, ow = map(int, spec["output_shape"])
    group = int(spec["group"])
    m = n * oh * ow
    k_total = cpg * kh * kw
    nper = cout // group
    nper_compute = max(128, nper)
    x_col = te.placeholder((group, m, k_total), "int8", name="x_col")
    w_mat = te.placeholder((group, k_total, nper_compute), "int8", name="w_mat")
    rk = te.reduce_axis((0, k_total), name="rk")
    output = te.compute(
        (group, m, nper_compute),
        lambda g, row, oc: te.sum(
            x_col[g, row, rk].astype("int32") * w_mat[g, rk, oc].astype("int32"),
            axis=rk,
        ),
        name="matmul",
    )
    return te.create_prim_func([x_col, w_mat, output]).with_attr("global_symbol", name)


def grouped_im2col(activation: np.ndarray, weight: np.ndarray, spec: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    activation = np.asarray(activation, dtype="int8")
    weight = np.asarray(weight, dtype="int8")
    n, _, _, _ = map(int, spec["input_shape"])
    cout, cpg, kh, kw = map(int, spec["weight_shape"])
    _, _, oh, ow = map(int, spec["output_shape"])
    group = int(spec["group"])
    sh, sw = map(int, spec["strides"])
    pt, pl, pb, pr = map(int, spec["pads"])
    padded = np.pad(activation, ((0, 0), (0, 0), (pt, pb), (pl, pr)), mode="constant")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (kh, kw), axis=(2, 3))
    windows = windows[:, :, : oh * sh : sh, : ow * sw : sw, :, :]
    m = n * oh * ow
    k_total = cpg * kh * kw
    nper = cout // group
    nper_compute = max(128, nper)
    x_col = np.empty((group, m, k_total), dtype="int8")
    w_mat = np.zeros((group, k_total, nper_compute), dtype="int8")
    for group_index in range(group):
        channel_start = group_index * cpg
        channel_stop = channel_start + cpg
        x_col[group_index] = windows[:, channel_start:channel_stop].transpose(0, 2, 3, 1, 4, 5).reshape(m, k_total)
        output_start = group_index * nper
        output_stop = output_start + nper
        w_mat[group_index, :, :nper] = weight[output_start:output_stop].reshape(nper, k_total).T
    return x_col, w_mat


def _profile_primfunc(
    stack: dict[str, Any],
    *,
    primfunc: Any,
    role: str,
    inputs: list[np.ndarray],
    output_shape: list[int],
    output_dtype: str,
    warmup: int,
    number: int,
    repeat: int,
) -> dict[str, Any]:
    tvm = stack["tvm"]
    scheduled, schedule_records = lowering.schedule_mixed_module(
        stack,
        tvm.IRModule({"main": primfunc}),
        {"main": role},
    )
    with stack["target"], tvm.transform.PassContext(opt_level=3):
        try:
            runtime_module = tvm.build(scheduled, target=stack["target"])
        except Exception:
            runtime_module = tvm.compile(scheduled, target=stack["target"])
    output = tvm.runtime.tensor(np.zeros(output_shape, dtype=output_dtype), device=stack["dev"])
    runtime_inputs = [tvm.runtime.tensor(value, device=stack["dev"]) for value in inputs]
    runtime_args = [*runtime_inputs, output]
    for _ in range(warmup):
        runtime_module["main"](*runtime_args)
    stack["dev"].sync()
    evaluator = runtime_module.time_evaluator("main", stack["dev"], number=number, repeat=repeat)
    repeats_ms = [float(value) * 1000.0 for value in evaluator(*runtime_args).results]
    return {
        "latency_ms_p50": float(statistics.median(repeats_ms)),
        "latency_ms_p90": float(np.percentile(np.asarray(repeats_ms, dtype="float64"), 90.0)),
        "latency_ms_repeats": repeats_ms,
        "schedule_records": schedule_records,
        "output_nbytes": int(np.prod(output_shape, dtype="int64")) * np.dtype(output_dtype).itemsize,
    }


def _gpu_snapshot(gpu: str) -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "-i",
        str(gpu),
        "--query-gpu=index,memory.used,utilization.gpu,power.draw,clocks.sm",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip().split(",")
    process_output = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(gpu),
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    ).stdout
    compute_pids = [int(line.strip()) for line in process_output.splitlines() if line.strip().isdigit()]
    return {
        "gpu": int(output[0]),
        "memory_used_mib": int(float(output[1])),
        "utilization_percent": int(float(output[2])),
        "power_w": float(output[3]),
        "sm_clock_mhz": int(float(output[4])),
        "compute_pids": compute_pids,
        "external_compute_pids": [pid for pid in compute_pids if pid != os.getpid()],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    calibration = json.loads(args.per_node_calibration.read_text(encoding="utf-8"))
    records = lowering.collect_onnx_conv_records(args.onnx)
    onnx_sha = lowering.sha256_file(args.onnx)
    lowering.validate_per_node_calibration_for_lowering(
        calibration,
        records=records,
        onnx_sha256=onnx_sha,
        allow_diagnostic_accuracy_matrix=True,
    )
    selected = sorted(records, key=lambda item: (-int(item["macs"]), str(item["node_id"])))[:2]
    selected_node_ids = [str(item["node_id"]) for item in selected]
    topology = lowering.form_int8_regions(
        lowering.collect_onnx_graph_nodes(args.onnx, records),
        set(selected_node_ids),
        graph_output_names=lowering.collect_onnx_graph_output_names(args.onnx),
    )
    if len(topology["regions"]) != 1 or topology["regions"][0]["node_ids"] != selected_node_ids:
        raise ValueError("top-2 Conv records do not form the required automatic rank-2 region")
    captured, capture_evidence = oracle._capture_conv_inputs(args.onnx, args.input_npz, selected)
    runtime_specs = oracle._conv_runtime_specs(args.onnx, selected)
    auto = lowering._load_auto_decomp()
    auto.cap.configure_tvm_env(str(args.gpu))
    stack = auto.cap.import_tvm_stack()
    before = _gpu_snapshot(args.gpu)
    if before["memory_used_mib"] >= args.max_idle_memory_mib or before["utilization_percent"] > args.max_idle_utilization:
        raise RuntimeError(f"GPU is not idle before E2 measurement: {before}")

    profiles: dict[str, list[dict[str, Any]]] = {
        "fp16": [],
        "int8_core": [],
        "int8_materialized": [],
        "int8_epilogue": [],
        "quantize": [],
        "bridge": [],
    }
    conv_evidence = []
    prepared = []
    for rank, record in enumerate(selected, start=1):
        node_id = str(record["node_id"])
        spec = runtime_specs[node_id]
        scales = lowering.resolve_node_scales(
            calibration,
            node_id=node_id,
            input_name=str(record["input_name"]),
            weight_name=str(record["weight_name"]),
            onnx_sha256=onnx_sha,
        )
        activation_fp16 = np.asarray(captured[str(record["input_name"])], dtype="float16")
        activation_s8, activation_stats = oracle.quantize_s8_reference(
            activation_fp16.astype("float32"),
            scale=float(scales["input_scale"]),
        )
        weight_s8, weight_stats = oracle.quantize_s8_reference(
            spec["weight"],
            scale=float(scales["weight_scale"]),
        )
        x_col, w_mat = grouped_im2col(activation_s8, weight_s8, spec)
        prepared.append((rank, record, spec, scales, activation_fp16, activation_s8, weight_s8, x_col, w_mat))
        conv_evidence.append(
            {
                "rank": rank,
                "node_id": node_id,
                "input_shape": spec["input_shape"],
                "weight_shape": spec["weight_shape"],
                "output_shape": spec["output_shape"],
                "group": spec["group"],
                "activation_quantization": activation_stats,
                "weight_quantization": weight_stats,
                "x_col_nbytes": int(x_col.nbytes),
                "w_mat_nbytes": int(w_mat.nbytes),
            }
        )

    for rank, record, spec, scales, activation_fp16, activation_s8, weight_s8, x_col, w_mat in prepared:
        output_shape = list(map(int, spec["output_shape"]))
        fp16_primfunc = auto.make_fp16_conv_direct_primfunc(stack, spec, "main")
        profiles["fp16"].append(
            {
                "rank": rank,
                **_profile_primfunc(
                    stack,
                    primfunc=fp16_primfunc,
                    role="fp16_conv",
                    inputs=[
                        activation_fp16,
                        np.asarray(spec["weight"], dtype="float16"),
                        np.asarray(spec["bias"], dtype="float16"),
                    ],
                    output_shape=output_shape,
                    output_dtype="float16",
                    warmup=args.warmup,
                    number=args.number,
                    repeat=args.repeat,
                ),
            }
        )
        nper_compute = max(128, output_shape[1] // int(spec["group"]))
        accumulator_shape = [int(spec["group"]), output_shape[0] * output_shape[2] * output_shape[3], nper_compute]
        profiles["int8_core"].append(
            {
                "rank": rank,
                **_profile_primfunc(
                    stack,
                    primfunc=make_int8_core_primfunc(stack, spec, "main"),
                    role="int8_accum",
                    inputs=[x_col, w_mat],
                    output_shape=accumulator_shape,
                    output_dtype="int32",
                    warmup=args.warmup,
                    number=args.number,
                    repeat=args.repeat,
                ),
            }
        )
        profiles["int8_materialized"].append(
            {
                "rank": rank,
                **_profile_primfunc(
                    stack,
                    primfunc=lowering.make_s8_conv_accum_primfunc(stack, spec, "main"),
                    role="int8_accum",
                    inputs=[activation_s8, weight_s8],
                    output_shape=accumulator_shape,
                    output_dtype="int32",
                    warmup=args.warmup,
                    number=args.number,
                    repeat=args.repeat,
                ),
            }
        )
        output_scale = None
        if rank == 1:
            next_record = selected[1]
            next_scales = lowering.resolve_node_scales(
                calibration,
                node_id=str(next_record["node_id"]),
                input_name=str(next_record["input_name"]),
                weight_name=str(next_record["weight_name"]),
                onnx_sha256=onnx_sha,
            )
            output_scale = float(next_scales["input_scale"])
        epilogue_dtype = "int8" if output_scale is not None else "float16"
        profiles["int8_epilogue"].append(
            {
                "rank": rank,
                "epilogue_output_scale": output_scale,
                **_profile_primfunc(
                    stack,
                    primfunc=lowering.make_s8_conv_fused_primfunc(
                        stack,
                        spec,
                        float(scales["input_scale"]) * float(scales["weight_scale"]),
                        "main",
                        requant_output_scale=output_scale,
                    ),
                    role="int8_fused",
                    inputs=[activation_s8, weight_s8, np.asarray(spec["bias"], dtype="float16")],
                    output_shape=output_shape,
                    output_dtype=epilogue_dtype,
                    warmup=args.warmup,
                    number=args.number,
                    repeat=args.repeat,
                ),
            }
        )
        if rank == 1:
            profiles["quantize"].append(
                {
                    "rank": rank,
                    **_profile_primfunc(
                        stack,
                        primfunc=lowering.make_quantize_s8_primfunc(
                            stack,
                            spec["input_shape"],
                            float(scales["input_scale"]),
                            "main",
                        ),
                        role="quantize",
                        inputs=[activation_fp16],
                        output_shape=list(map(int, spec["input_shape"])),
                        output_dtype="int8",
                        warmup=args.warmup,
                        number=args.number,
                        repeat=args.repeat,
                    ),
                }
            )

    bridge_ops = topology["regions"][0].get("bridges") or []
    for bridge_index, bridge in enumerate(bridge_ops, start=1):
        if bridge.get("op_types") != ["Relu"]:
            raise ValueError(f"unsupported retained INT8 bridge in E2: {bridge}")
        bridge_shape = list(map(int, runtime_specs[selected_node_ids[0]]["output_shape"]))
        profiles["bridge"].append(
            {
                "bridge_index": bridge_index,
                "op_types": bridge["op_types"],
                **_profile_primfunc(
                    stack,
                    primfunc=lowering.make_relu_s8_primfunc(stack, bridge_shape, "main"),
                    role="relu",
                    inputs=[np.zeros(bridge_shape, dtype="int8")],
                    output_shape=bridge_shape,
                    output_dtype="int8",
                    warmup=args.warmup,
                    number=args.number,
                    repeat=args.repeat,
                ),
            }
        )

    stages = summarize_micro_stages(profiles)
    fp16_full = json.loads(args.fp16_full_result.read_text(encoding="utf-8"))
    int8_full = json.loads(args.int8_full_result.read_text(encoding="utf-8"))
    linked_identity = validate_linked_full_results(
        fp16_full,
        int8_full,
        onnx_sha256=onnx_sha,
        width=args.width,
        gpu=str(args.gpu),
        calibration_sha256=lowering.sha256_file(args.per_node_calibration),
        selected_node_ids=selected_node_ids,
    )
    stages["S5"] = {
        "realization": "automatic rank-2 INT8 region in the full 27-Conv backbone",
        "latency_ms_p50": float(int8_full["latency_ms_p50"]),
        "fp16_full_graph_latency_ms_p50": float(fp16_full["latency_ms_p50"]),
        "relative_to_fp16_full_graph": float(int8_full["latency_ms_p50"]) / float(fp16_full["latency_ms_p50"]),
        "numerical_gate": int8_full["numerical_gate"],
        "source_artifacts": [str(args.fp16_full_result), str(args.int8_full_result)],
    }
    after = _gpu_snapshot(args.gpu)
    clean_gpu_gate = not after["external_compute_pids"]
    return {
        "schema": SCHEMA,
        "status": "diagnostic_success" if clean_gpu_gate else "invalid_gpu_contention",
        "trusted_for_final_frontier": False,
        "host": socket.gethostname(),
        "gpu": str(args.gpu),
        "width": args.width,
        "onnx": str(args.onnx),
        "onnx_sha256": onnx_sha,
        "calibration": str(args.per_node_calibration),
        "calibration_sha256": lowering.sha256_file(args.per_node_calibration),
        "capture_evidence": capture_evidence,
        "gpu_snapshots": {"before": before, "after": after},
        "clean_gpu_gate": clean_gpu_gate,
        "measurement": {"warmup": args.warmup, "number": args.number, "repeat": args.repeat},
        "conv_evidence": conv_evidence,
        "automatic_region_topology": topology,
        "linked_full_result_identity": linked_identity,
        "profiles": profiles,
        "stages": stages,
        "first_micro_reversal": classify_first_reversal(stages),
        "elapsed_s": time.time() - started,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--per-node-calibration", type=Path, required=True)
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--fp16-full-result", type=Path, required=True)
    parser.add_argument("--int8-full-result", type=Path, required=True)
    parser.add_argument("--width", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--number", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--max-idle-memory-mib", type=int, default=100)
    parser.add_argument("--max-idle-utilization", type=int, default=5)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()
    for path in (
        args.onnx,
        args.per_node_calibration,
        args.input_npz,
        args.fp16_full_result,
        args.int8_full_result,
    ):
        if not path.is_file():
            parser.error(f"input file not found: {path}")
    return args


def main() -> int:
    args = parse_args()
    report = run(args)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "diagnostic_success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
