#!/usr/bin/env python3
"""Generic ONNX Conv selection manifest for CoDriving mixed-precision lowering."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from pathlib import Path
from typing import Any


SCHEMA = "codriving_mixed_auto_lowering_selection_manifest_v1"
POLICY_FRACTIONS = {
    "top10_flops": 0.10,
    "top25_flops": 0.25,
    "top50_flops": 0.50,
    "top75_flops": 0.75,
    "all": 1.00,
}
PER_NODE_CALIBRATION_SCHEMA = "codriving_raw_onnx_per_node_calibration_v1"
PER_NODE_QUANTIZATION_SEMANTICS = "symmetric_absmax_int8_per_raw_onnx_node"
DIAGNOSTIC_ACCURACY_MATRIX_SCHEMA = "codriving_int8_accuracy_matrix_calibration_v1"
DIAGNOSTIC_ACCURACY_MATRIX_SEMANTICS = "symmetric_percentile_int8_per_raw_onnx_node"
EXPECTED_CALIBRATION_SAMPLES = 16
SHA256_RE = re.compile(r"[0-9a-f]{64}")


def conv_signature(input_shape: list[int], weight_shape: list[int]) -> str:
    return "input=" + "x".join(map(str, input_shape)) + "|weight=" + "x".join(map(str, weight_shape))


def quantize_weight_s8(value: Any, *, scale: float) -> Any:
    import numpy as np

    if not math.isfinite(float(scale)) or float(scale) <= 0.0:
        raise ValueError("weight scale must be finite and positive")
    return np.clip(np.rint(np.asarray(value, dtype="float32") / float(scale)), -127, 127).astype("int8")


def resolve_node_scales(
    calibration: dict[str, Any],
    *,
    node_id: str,
    input_name: str,
    weight_name: str,
    onnx_sha256: str,
) -> dict[str, float]:
    if calibration.get("schema") not in {PER_NODE_CALIBRATION_SCHEMA, DIAGNOSTIC_ACCURACY_MATRIX_SCHEMA}:
        raise ValueError("mixed lowering requires raw ONNX per-node calibration schema")
    if calibration.get("onnx_sha256") != onnx_sha256:
        raise ValueError("per-node calibration ONNX SHA256 mismatch")
    record = (calibration.get("nodes") or {}).get(node_id)
    if not isinstance(record, dict):
        raise ValueError(f"missing per-node calibration record: {node_id}")
    if record.get("input_name") != input_name:
        raise ValueError(f"per-node calibration input_name mismatch: {node_id}")
    if record.get("weight_name") != weight_name:
        raise ValueError(f"per-node calibration weight_name mismatch: {node_id}")
    scales = {field: float(record.get(field) or 0.0) for field in ("input_scale", "weight_scale")}
    if any(not math.isfinite(value) or value <= 0.0 for value in scales.values()):
        raise ValueError(f"invalid per-node scales: {node_id}")
    return scales


def validate_per_node_calibration_for_lowering(
    calibration: dict[str, Any],
    *,
    records: list[dict[str, Any]],
    onnx_sha256: str,
    allow_diagnostic_accuracy_matrix: bool = False,
) -> dict[str, Any]:
    schema = calibration.get("schema")
    if schema == DIAGNOSTIC_ACCURACY_MATRIX_SCHEMA and not allow_diagnostic_accuracy_matrix:
        raise ValueError("diagnostic accuracy-matrix calibration requires explicit opt-in")
    if schema not in {PER_NODE_CALIBRATION_SCHEMA, DIAGNOSTIC_ACCURACY_MATRIX_SCHEMA}:
        raise ValueError("mixed lowering requires raw ONNX per-node calibration schema")
    expected_semantics = (
        DIAGNOSTIC_ACCURACY_MATRIX_SEMANTICS
        if schema == DIAGNOSTIC_ACCURACY_MATRIX_SCHEMA
        else PER_NODE_QUANTIZATION_SEMANTICS
    )
    if calibration.get("quantization_semantics") != expected_semantics:
        raise ValueError("unexpected per-node calibration quantization semantics")
    if schema == DIAGNOSTIC_ACCURACY_MATRIX_SCHEMA:
        if SHA256_RE.fullmatch(str(calibration.get("parent_calibration_sha256") or "")) is None:
            raise ValueError("diagnostic accuracy-matrix calibration lacks parent_calibration_sha256")
    if calibration.get("calibration_split") != "train":
        raise ValueError("per-node calibration must use train split")
    sample_count = calibration.get("sample_count")
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count != EXPECTED_CALIBRATION_SAMPLES:
        raise ValueError(f"per-node calibration sample_count must equal {EXPECTED_CALIBRATION_SAMPLES}")
    if calibration.get("onnx_sha256") != onnx_sha256:
        raise ValueError("per-node calibration ONNX SHA256 mismatch")
    for field in ("calibration_source", "calibration_summary", "calibration_split_source"):
        if not str(calibration.get(field) or ""):
            raise ValueError(f"per-node calibration lacks {field}")
    for field in ("calibration_source_sha256", "calibration_summary_sha256"):
        if SHA256_RE.fullmatch(str(calibration.get(field) or "")) is None:
            raise ValueError(f"invalid per-node calibration {field}")
    for path_field, sha_field in (
        ("calibration_source", "calibration_source_sha256"),
        ("calibration_summary", "calibration_summary_sha256"),
    ):
        evidence_path = Path(str(calibration[path_field]))
        if not evidence_path.is_file():
            raise ValueError(f"per-node calibration evidence file not found: {path_field}")
        if sha256_file(evidence_path) != calibration[sha_field]:
            raise ValueError(f"per-node calibration {path_field} SHA256 mismatch")
    if not Path(str(calibration["calibration_split_source"])).is_file():
        raise ValueError("per-node calibration split source file not found")

    expected = {str(record["node_id"]): record for record in records}
    nodes = calibration.get("nodes")
    if not isinstance(nodes, dict) or set(nodes) != set(expected):
        raise ValueError("per-node calibration Conv coverage mismatch")
    node_count = calibration.get("node_count")
    if isinstance(node_count, bool) or not isinstance(node_count, int) or node_count != len(expected):
        raise ValueError("per-node calibration node_count mismatch")
    for node_id, record in expected.items():
        bound = nodes[node_id]
        if not isinstance(bound, dict):
            raise ValueError(f"invalid per-node calibration record: {node_id}")
        for field in ("input_name", "weight_name", "output_name"):
            if bound.get(field) != record.get(field):
                raise ValueError(f"per-node calibration {field} mismatch: {node_id}")
        resolve_node_scales(
            calibration,
            node_id=node_id,
            input_name=str(record["input_name"]),
            weight_name=str(record["weight_name"]),
            onnx_sha256=onnx_sha256,
        )
    return dict(calibration)


def validate_calibration_mode(
    *,
    int8_conv_limit: int,
    per_node_calibration: Path | None,
    calibration_ap_report: Path | None,
    allow_legacy_shape_calibration: bool,
) -> None:
    if int8_conv_limit <= 0:
        return
    if calibration_ap_report is not None or allow_legacy_shape_calibration:
        raise ValueError("legacy shape calibration is diagnostic history and is rejected by the formal lowering entry")
    if per_node_calibration is None:
        raise ValueError("--per-node-calibration is required when --int8-conv-limit > 0")


def result_exit_code(result: dict[str, Any]) -> int:
    status = result.get("status")
    return 0 if status in (None, "success") else 2


def compare_full_outputs(reference_outputs: list[Any], candidate_outputs: list[Any]) -> list[dict[str, Any]]:
    import numpy as np

    if len(reference_outputs) != len(candidate_outputs):
        raise ValueError(
            f"full graph output count mismatch: reference={len(reference_outputs)}, candidate={len(candidate_outputs)}"
        )
    numerical = []
    for index, (reference, candidate) in enumerate(zip(reference_outputs, candidate_outputs)):
        candidate_array = np.asarray(candidate, dtype="float32")
        reference_array = np.asarray(reference, dtype="float32")
        if reference_array.shape != candidate_array.shape:
            raise ValueError(f"full graph output shape mismatch at output {index}")
        diff = np.abs(reference_array - candidate_array)
        numerical.append(
            {
                "output": index,
                "shape": list(candidate_array.shape),
                "max_abs_err": float(diff.max()) if diff.size else 0.0,
                "mean_abs_err": float(diff.mean()) if diff.size else 0.0,
            }
        )
    return numerical


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_conv_ids(records: list[dict[str, Any]], policy: str) -> list[str]:
    if policy not in POLICY_FRACTIONS:
        raise ValueError(f"unsupported mixed policy: {policy}")
    node_ids = [str(record.get("node_id") or "") for record in records]
    if not node_ids or any(not node_id for node_id in node_ids):
        raise ValueError("Conv records require nonempty node_id")
    if len(set(node_ids)) != len(node_ids):
        raise ValueError("duplicate Conv node_id in selection records")
    ranked = sorted(
        records,
        key=lambda record: (-int(record.get("macs") or 0), str(record["node_id"])),
    )
    keep = max(1, int(math.ceil(len(ranked) * POLICY_FRACTIONS[policy])))
    return [str(record["node_id"]) for record in ranked[:keep]]


def form_int8_regions(
    nodes: list[dict[str, Any]],
    selected_node_ids: set[str],
    *,
    graph_output_names: set[str] | None = None,
) -> dict[str, Any]:
    selected_node_ids = set(selected_node_ids)
    graph_output_names = set(graph_output_names or set())
    node_by_index = {int(node["node_index"]): node for node in nodes}
    selected_order = [
        str(node["node_id"])
        for node in nodes
        if node.get("op_type") == "Conv" and str(node.get("node_id") or "") in selected_node_ids
    ]
    if set(selected_order) != selected_node_ids:
        missing = sorted(selected_node_ids - set(selected_order))
        raise ValueError(f"selected INT8 Conv IDs missing from graph: {missing}")
    consumers: dict[str, list[int]] = {}
    for node in nodes:
        for tensor_name in node.get("inputs") or []:
            consumers.setdefault(str(tensor_name), []).append(int(node["node_index"]))

    def trace_successor(node: dict[str, Any]) -> dict[str, Any]:
        outputs = [str(value) for value in node.get("outputs") or []]
        if len(outputs) != 1:
            return {"next_node_id": None, "bridges": [], "break_reason": "multiple_outputs"}
        tensor_name = outputs[0]
        bridges: list[dict[str, Any]] = []
        while True:
            if tensor_name in graph_output_names:
                return {"next_node_id": None, "bridges": bridges, "break_reason": "graph_output"}
            next_indices = consumers.get(tensor_name, [])
            if not next_indices:
                return {"next_node_id": None, "bridges": bridges, "break_reason": "graph_exit"}
            if len(next_indices) != 1:
                return {"next_node_id": None, "bridges": bridges, "break_reason": "fanout"}
            next_node = node_by_index[next_indices[0]]
            op_type = str(next_node.get("op_type") or "")
            if op_type == "Relu":
                bridge_outputs = [str(value) for value in next_node.get("outputs") or []]
                if len(bridge_outputs) != 1:
                    return {"next_node_id": None, "bridges": bridges, "break_reason": "Relu_multiple_outputs"}
                bridges.append({"node_index": int(next_node["node_index"]), "op_type": op_type})
                tensor_name = bridge_outputs[0]
                continue
            if op_type == "Conv":
                next_node_id = str(next_node.get("node_id") or "")
                if next_node_id in selected_node_ids:
                    return {"next_node_id": next_node_id, "bridges": bridges, "break_reason": None}
                return {"next_node_id": None, "bridges": bridges, "break_reason": "next_conv_not_selected"}
            return {"next_node_id": None, "bridges": bridges, "break_reason": f"unsupported_op:{op_type}"}

    selected_nodes = {
        str(node["node_id"]): node
        for node in nodes
        if node.get("op_type") == "Conv" and str(node.get("node_id") or "") in selected_node_ids
    }
    traces = {node_id: trace_successor(node) for node_id, node in selected_nodes.items()}
    incoming = {
        str(trace["next_node_id"])
        for trace in traces.values()
        if trace.get("next_node_id") is not None
    }
    starts = [node_id for node_id in selected_order if node_id not in incoming]
    regions: list[dict[str, Any]] = []
    assigned: set[str] = set()
    for start in starts:
        node_ids: list[str] = []
        bridges: list[dict[str, Any]] = []
        current = start
        while current not in assigned:
            assigned.add(current)
            node_ids.append(current)
            trace = traces[current]
            next_node_id = trace.get("next_node_id")
            if next_node_id is None:
                break
            bridges.append(
                {
                    "from_node_id": current,
                    "to_node_id": str(next_node_id),
                    "node_indices": [int(item["node_index"]) for item in trace["bridges"]],
                    "op_types": [str(item["op_type"]) for item in trace["bridges"]],
                }
            )
            current = str(next_node_id)
        region_id = f"int8_region_{len(regions):03d}"
        regions.append(
            {
                "region_id": region_id,
                "node_ids": node_ids,
                "entry_node_id": node_ids[0],
                "exit_node_id": node_ids[-1],
                "bridges": bridges,
                "quantize_boundary_count": 1,
                "dequantize_boundary_count": 1,
                "exit_break_reason": traces[node_ids[-1]]["break_reason"],
            }
        )
    if assigned != selected_node_ids:
        raise ValueError("INT8 region formation produced a cycle or unassigned selected Conv")
    node_annotations = {
        node_id: {
            "region_id": region["region_id"],
            "region_position": position,
            "region_length": len(region["node_ids"]),
            "is_region_entry": position == 0,
            "is_region_exit": position == len(region["node_ids"]) - 1,
        }
        for region in regions
        for position, node_id in enumerate(region["node_ids"])
    }
    return {
        "region_count": len(regions),
        "quantize_boundary_count": len(regions),
        "dequantize_boundary_count": len(regions),
        "per_conv_quantize_boundary_count": len(selected_node_ids),
        "per_conv_dequantize_boundary_count": len(selected_node_ids),
        "regions": regions,
        "node_annotations": node_annotations,
    }


def validate_supported_conv_attrs(attrs: dict[str, Any], *, node_id: str) -> None:
    auto_pad = attrs.get("auto_pad", "NOTSET")
    if isinstance(auto_pad, bytes):
        auto_pad = auto_pad.decode("ascii")
    if str(auto_pad) not in {"", "NOTSET"}:
        raise ValueError(f"unsupported Conv auto_pad at {node_id}: {auto_pad}")
    dilations = [int(value) for value in attrs.get("dilations", [1, 1])]
    if dilations != [1, 1]:
        raise ValueError(f"unsupported Conv dilations at {node_id}: {dilations}")
    strides = [int(value) for value in attrs.get("strides", [1, 1])]
    if len(strides) != 2 or any(value <= 0 for value in strides):
        raise ValueError(f"unsupported Conv strides at {node_id}: {strides}")
    pads = [int(value) for value in attrs.get("pads", [0, 0, 0, 0])]
    if len(pads) != 4 or any(value < 0 for value in pads):
        raise ValueError(f"unsupported Conv pads at {node_id}: {pads}")
    group = int(attrs.get("group", 1))
    if group <= 0:
        raise ValueError(f"unsupported Conv group at {node_id}: {group}")


def _tensor_shape(value_info: Any) -> list[int]:
    dims = []
    for dim in value_info.type.tensor_type.shape.dim:
        if not dim.HasField("dim_value"):
            raise ValueError(f"dynamic tensor dimension in {value_info.name}")
        dims.append(int(dim.dim_value))
    return dims


def collect_onnx_conv_records(onnx_path: Path) -> list[dict[str, Any]]:
    import onnx
    from onnx import helper

    model = onnx.shape_inference.infer_shapes(onnx.load(str(onnx_path)))
    shape_map: dict[str, list[int]] = {}
    for value in [*model.graph.input, *model.graph.value_info, *model.graph.output]:
        if value.type.HasField("tensor_type"):
            shape_map[value.name] = _tensor_shape(value)
    for initializer in model.graph.initializer:
        shape_map[initializer.name] = [int(dim) for dim in initializer.dims]

    records: list[dict[str, Any]] = []
    for node_index, node in enumerate(model.graph.node):
        if node.op_type != "Conv":
            continue
        input_shape = shape_map[node.input[0]]
        weight_shape = shape_map[node.input[1]]
        output_shape = shape_map[node.output[0]]
        attrs = {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}
        group = int(attrs.get("group", 1))
        n, _, _, _ = input_shape
        cout, cin_per_group, kh, kw = weight_shape
        _, out_channels, oh, ow = output_shape
        if out_channels != cout:
            raise ValueError(f"Conv output/weight channel mismatch at node {node_index}")
        macs = int(n * oh * ow * cout * cin_per_group * kh * kw)
        stable_name = str(node.name or f"Conv_{node_index}")
        node_id = f"onnx_conv_{node_index:04d}_{stable_name}"
        validate_supported_conv_attrs(attrs, node_id=node_id)
        records.append(
            {
                "node_id": node_id,
                "node_index": node_index,
                "node_name": stable_name,
                "input_name": str(node.input[0]),
                "weight_name": str(node.input[1]),
                "output_name": str(node.output[0]),
                "input_shape": input_shape,
                "weight_shape": weight_shape,
                "output_shape": output_shape,
                "group": group,
                "macs": macs,
            }
        )
    if not records:
        raise ValueError("ONNX graph contains no Conv nodes")
    return records


def collect_onnx_graph_nodes(onnx_path: Path, conv_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    import onnx

    model = onnx.load(str(onnx_path))
    conv_ids = {int(record["node_index"]): str(record["node_id"]) for record in conv_records}
    return [
        {
            "node_index": node_index,
            "op_type": str(node.op_type),
            "node_id": conv_ids.get(node_index),
            "inputs": [str(value) for value in node.input],
            "outputs": [str(value) for value in node.output],
        }
        for node_index, node in enumerate(model.graph.node)
    ]


def collect_onnx_graph_output_names(onnx_path: Path) -> set[str]:
    import onnx

    model = onnx.load(str(onnx_path))
    return {str(item.name) for item in model.graph.output}


def build_selection_manifest(onnx_path: Path, *, width: str, policy: str) -> dict[str, Any]:
    records = collect_onnx_conv_records(onnx_path)
    selected = set(select_conv_ids(records, policy))
    region_formation = form_int8_regions(
        collect_onnx_graph_nodes(onnx_path, records),
        selected,
        graph_output_names=collect_onnx_graph_output_names(onnx_path),
    )
    annotations = region_formation["node_annotations"]
    ranked = sorted(records, key=lambda record: (-int(record["macs"]), str(record["node_id"])))
    ranked_records = [
        {
            **record,
            "flops_rank": rank,
            "selected_int8": record["node_id"] in selected,
            **annotations.get(record["node_id"], {}),
        }
        for rank, record in enumerate(ranked, start=1)
    ]
    return {
        "schema": SCHEMA,
        "width": width,
        "mixed_policy_id": policy,
        "onnx": str(onnx_path),
        "onnx_sha256": sha256_file(onnx_path),
        "selection_rule": "stable_descending_macs_then_node_id",
        "total_conv_count": len(records),
        "selected_conv_count": len(selected),
        "selected_node_ids": [record["node_id"] for record in ranked_records if record["selected_int8"]],
        "region_formation": region_formation,
        "conv_records": ranked_records,
    }


def _load_auto_decomp() -> Any:
    import stage2_route_b_int8_auto_decomp as auto

    return auto


def make_cast_primfunc(stack: dict[str, Any], shape: list[int], src: str, dst: str, name: str) -> Any:
    te = stack["te"]
    x = te.placeholder(tuple(shape), src, name="x")
    out = te.compute(tuple(shape), lambda *idx: x[idx].astype(dst), name="cast_out")
    return te.create_prim_func([x, out]).with_attr("global_symbol", name)


def make_relu_fp16_primfunc(stack: dict[str, Any], shape: list[int], name: str) -> Any:
    te = stack["te"]
    x = te.placeholder(tuple(shape), "float16", name="x")
    out = te.compute(
        tuple(shape),
        lambda *idx: te.max(x[idx], te.const(0, "float16")),
        name="relu_out",
    )
    return te.create_prim_func([x, out]).with_attr("global_symbol", name)


def make_relu_s8_primfunc(stack: dict[str, Any], shape: list[int], name: str) -> Any:
    te = stack["te"]
    x = te.placeholder(tuple(shape), "int8", name="x")
    out = te.compute(
        tuple(shape),
        lambda *idx: te.max(x[idx], te.const(0, "int8")),
        name="relu_s8_out",
    )
    return te.create_prim_func([x, out]).with_attr("global_symbol", name)


def make_add_fp16_primfunc(stack: dict[str, Any], shape: list[int], name: str) -> Any:
    te = stack["te"]
    lhs = te.placeholder(tuple(shape), "float16", name="lhs")
    rhs = te.placeholder(tuple(shape), "float16", name="rhs")
    out = te.compute(tuple(shape), lambda *idx: lhs[idx] + rhs[idx], name="add_out")
    return te.create_prim_func([lhs, rhs, out]).with_attr("global_symbol", name)


def make_quantize_s8_primfunc(stack: dict[str, Any], shape: list[int], scale: float, name: str) -> Any:
    auto = _load_auto_decomp()
    te = stack["te"]
    x = te.placeholder(tuple(shape), "float16", name="x")

    def compute(*idx: Any) -> Any:
        scaled = x[idx].astype("float32") / te.const(float(scale), "float32")
        rounded = auto.route.round_expr(stack, scaled)
        upper = te.if_then_else(rounded > te.const(127.0, "float32"), te.const(127.0, "float32"), rounded)
        clipped = te.if_then_else(upper < te.const(-127.0, "float32"), te.const(-127.0, "float32"), upper)
        return clipped.astype("int8")

    out = te.compute(tuple(shape), compute, name="quantize_s8")
    return te.create_prim_func([x, out]).with_attr("global_symbol", name)


def make_s8_conv_accum_primfunc(stack: dict[str, Any], spec: dict[str, Any], name: str) -> Any:
    auto = _load_auto_decomp()
    te = stack["te"]
    n, cin, h, w = map(int, spec["input_shape"])
    cout, cpg, kh, kw = map(int, spec["weight_shape"])
    group = int(spec["group"])
    sh, sw = map(int, spec["strides"])
    pt, pl, _, _ = map(int, spec["pads"])
    oh, ow = auto.conv2d_out_hw(h, w, kh, kw, spec["strides"], spec["pads"])
    m = n * oh * ow
    k_total = cpg * kh * kw
    nper = cout // group
    nper_compute = 128 if nper < 128 else nper
    x = te.placeholder((n, cin, h, w), "int8", name="x")
    weight = te.placeholder((cout, cpg, kh, kw), "int8", name="weight")

    def xcol(g: Any, row: Any, kk: Any) -> Any:
        nn = row // (oh * ow)
        rem = row % (oh * ow)
        yy, xx = rem // ow, rem % ow
        ci = kk // (kh * kw)
        remk = kk % (kh * kw)
        ry, rx = remk // kw, remk % kw
        iy, ix = yy * sh + ry - pt, xx * sw + rx - pl
        return te.if_then_else((iy >= 0) & (iy < h) & (ix >= 0) & (ix < w), x[nn, g * cpg + ci, iy, ix], te.const(0, "int8"))

    x_col = te.compute((group, m, k_total), xcol, name="x_col")

    def wmat(g: Any, kk: Any, ocg: Any) -> Any:
        ci = kk // (kh * kw)
        remk = kk % (kh * kw)
        ry, rx = remk // kw, remk % kw
        valid = ocg < nper
        safe = te.if_then_else(valid, ocg, te.const(0, "int32"))
        return te.if_then_else(valid, weight[g * nper + safe, ci, ry, rx], te.const(0, "int8"))

    w_mat = te.compute((group, k_total, nper_compute), wmat, name="w_mat")
    rk = te.reduce_axis((0, k_total), name="rk")
    accum = te.compute((group, m, nper_compute), lambda g, r, o: te.sum(x_col[g, r, rk].astype("int32") * w_mat[g, rk, o].astype("int32"), axis=rk), name="matmul")
    return te.create_prim_func([x, weight, accum]).with_attr("global_symbol", name)


def make_dequant_bias_fp16_primfunc(stack: dict[str, Any], spec: dict[str, Any], scale: float, name: str) -> Any:
    auto = _load_auto_decomp()
    te = stack["te"]
    n, _, h, w = map(int, spec["input_shape"])
    cout, _, kh, kw = map(int, spec["weight_shape"])
    group = int(spec["group"])
    oh, ow = auto.conv2d_out_hw(h, w, kh, kw, spec["strides"], spec["pads"])
    m = n * oh * ow
    nper = cout // group
    nper_compute = 128 if nper < 128 else nper
    accum = te.placeholder((group, m, nper_compute), "int32", name="accum")
    bias = te.placeholder((cout,), "float16", name="bias")

    def compute(nn: Any, oc: Any, yy: Any, xx: Any) -> Any:
        g, ocg = oc // nper, oc % nper
        row = nn * (oh * ow) + yy * ow + xx
        return (accum[g, row, ocg].astype("float32") * te.const(float(scale), "float32") + bias[oc].astype("float32")).astype("float16")

    out = te.compute((n, cout, oh, ow), compute, name="dequant_bias")
    return te.create_prim_func([accum, bias, out]).with_attr("global_symbol", name)


def make_s8_conv_fused_primfunc(
    stack: dict[str, Any],
    spec: dict[str, Any],
    scale: float,
    name: str,
    *,
    requant_output_scale: float | None = None,
) -> Any:
    auto = _load_auto_decomp()
    te = stack["te"]
    n, cin, h, w = map(int, spec["input_shape"])
    cout, cpg, kh, kw = map(int, spec["weight_shape"])
    group = int(spec["group"])
    sh, sw = map(int, spec["strides"])
    pt, pl, _, _ = map(int, spec["pads"])
    oh, ow = auto.conv2d_out_hw(h, w, kh, kw, spec["strides"], spec["pads"])
    m, k_total, nper = n * oh * ow, cpg * kh * kw, cout // group
    nper_compute = 128 if nper < 128 else nper
    x = te.placeholder((n, cin, h, w), "int8", name="x")
    weight = te.placeholder((cout, cpg, kh, kw), "int8", name="weight")
    bias = te.placeholder((cout,), "float16", name="bias")

    def xcol(g: Any, row: Any, kk: Any) -> Any:
        nn, rem = row // (oh * ow), row % (oh * ow)
        yy, xx = rem // ow, rem % ow
        ci, remk = kk // (kh * kw), kk % (kh * kw)
        ry, rx = remk // kw, remk % kw
        iy, ix = yy * sh + ry - pt, xx * sw + rx - pl
        return te.if_then_else((iy >= 0) & (iy < h) & (ix >= 0) & (ix < w), x[nn, g * cpg + ci, iy, ix], te.const(0, "int8"))

    x_col = te.compute((group, m, k_total), xcol, name="x_col")

    def wmat(g: Any, kk: Any, ocg: Any) -> Any:
        ci, remk = kk // (kh * kw), kk % (kh * kw)
        ry, rx = remk // kw, remk % kw
        valid = ocg < nper
        safe = te.if_then_else(valid, ocg, te.const(0, "int32"))
        return te.if_then_else(valid, weight[g * nper + safe, ci, ry, rx], te.const(0, "int8"))

    w_mat = te.compute((group, k_total, nper_compute), wmat, name="w_mat")
    rk = te.reduce_axis((0, k_total), name="rk")
    matmul = te.compute((group, m, nper_compute), lambda g, r, o: te.sum(x_col[g, r, rk].astype("int32") * w_mat[g, rk, o].astype("int32"), axis=rk), name="matmul")

    def output(nn: Any, oc: Any, yy: Any, xx: Any) -> Any:
        g, ocg = oc // nper, oc % nper
        row = nn * (oh * ow) + yy * ow + xx
        value = matmul[g, row, ocg].astype("float32") * te.const(float(scale), "float32") + bias[oc].astype("float32")
        if requant_output_scale is None:
            return value.astype("float16")
        rounded = auto.route.round_expr(stack, value / te.const(float(requant_output_scale), "float32"))
        upper = te.if_then_else(rounded > te.const(127.0, "float32"), te.const(127.0, "float32"), rounded)
        clipped = te.if_then_else(upper < te.const(-127.0, "float32"), te.const(-127.0, "float32"), upper)
        return clipped.astype("int8")

    out_name = "requant_bias_out" if requant_output_scale is not None else "dequant_bias_out"
    out = te.compute((n, cout, oh, ow), output, name=out_name)
    return te.create_prim_func([x, weight, bias, out]).with_attr("global_symbol", name)


def build_fp16_blockbuilder_module(
    stack: dict[str, Any],
    onnx_path: Path,
    *,
    selected_node_ids: set[str] | None = None,
    calibration: dict[str, Any] | None = None,
    fuse_int8_epilogue: bool = False,
    retain_int8_regions: bool = False,
    allow_legacy_shape_calibration: bool = False,
) -> dict[str, Any]:
    import numpy as np
    from onnx import numpy_helper
    from tvm import relax

    auto = _load_auto_decomp()
    model, shape_map, initializers = auto.route.load_onnx_model(onnx_path)
    bb = relax.BlockBuilder()
    initializer_names = set(initializers)
    tensor_map: dict[str, Any] = {}
    inputs: list[Any] = []
    runtime_values: dict[str, Any] = {}
    block_roles: dict[str, str] = {}
    selected_node_ids = set(selected_node_ids or set())
    onnx_digest = sha256_file(onnx_path)
    calibration_binding = "none"
    input_specs: list[dict[str, Any]] = []
    int8_tensor_scales: dict[str, float] = {}
    conv_records = collect_onnx_conv_records(onnx_path)
    conv_record_by_id = {str(record["node_id"]): record for record in conv_records}
    region_formation = (
        form_int8_regions(
            collect_onnx_graph_nodes(onnx_path, conv_records),
            selected_node_ids,
            graph_output_names=collect_onnx_graph_output_names(onnx_path),
        )
        if selected_node_ids and retain_int8_regions
        else None
    )
    region_annotations = (region_formation or {}).get("node_annotations", {})
    regions_by_id = {
        str(region["region_id"]): region for region in (region_formation or {}).get("regions", [])
    }

    for item in model.graph.input:
        if item.name in initializer_names:
            continue
        shape = [int(value) for value in shape_map[item.name]]
        var_name = auto.route.sanitize_name(item.name, "input")
        var = relax.Var(var_name, auto.relax_tensor_info(relax, shape, "float32"))
        inputs.append(var)
        input_specs.append({"arg_name": var_name, "source_name": item.name, "shape": shape, "dtype": "float32"})
        tensor_map[item.name] = var

    with bb.function("main", inputs):
        with bb.dataflow():
            for input_spec in input_specs:
                name = f"mixed_cast_input_{input_spec['arg_name']}"
                gv = bb.add_func(
                    make_cast_primfunc(stack, input_spec["shape"], "float32", "float16", name),
                    name,
                )
                block_roles[name] = "cast"
                tensor_map[input_spec["source_name"]] = bb.emit(
                    relax.call_tir(
                        gv,
                        [tensor_map[input_spec["source_name"]]],
                        auto.relax_tensor_info(relax, input_spec["shape"], "float16"),
                    )
                )

            conv_index = 0
            relu_index = 0
            add_index = 0
            for node_index, node in enumerate(model.graph.node):
                op_name = str(node.name or f"{node.op_type}_{node_index}")
                if node.op_type == "Conv":
                    conv_index += 1
                    node_id = f"onnx_conv_{node_index:04d}_{op_name}"
                    attrs = auto.route.attr_dict(node)
                    input_shape = [int(value) for value in shape_map[node.input[0]]]
                    weight_shape = [int(value) for value in shape_map[node.input[1]]]
                    output_shape = [int(value) for value in shape_map[node.output[0]]]
                    raw_weight = np.asarray(numpy_helper.to_array(initializers[node.input[1]]), dtype="float32")
                    if len(node.input) >= 3 and node.input[2] in initializers:
                        bias_value = np.asarray(numpy_helper.to_array(initializers[node.input[2]]), dtype="float16")
                    else:
                        bias_value = np.zeros((weight_shape[0],), dtype="float16")
                    bias_name = f"fp16_bias_{conv_index}"
                    bias_var = relax.Var(bias_name, auto.relax_tensor_info(relax, list(bias_value.shape), "float16"))
                    spec = {
                        "input_shape": input_shape,
                        "output_shape": output_shape,
                        "weight_shape": weight_shape,
                        "group": int(attrs.get("group", 1) or 1),
                        "strides": [int(value) for value in attrs.get("strides", [1, 1])],
                        "pads": [int(value) for value in attrs.get("pads", [0, 0, 0, 0])],
                    }
                    if node_id in selected_node_ids:
                        if calibration is None:
                            raise ValueError("selected INT8 Conv requires calibration")
                        if calibration.get("schema") in {
                            PER_NODE_CALIBRATION_SCHEMA,
                            DIAGNOSTIC_ACCURACY_MATRIX_SCHEMA,
                        }:
                            scales = resolve_node_scales(
                                calibration,
                                node_id=node_id,
                                input_name=str(node.input[0]),
                                weight_name=str(node.input[1]),
                                onnx_sha256=onnx_digest,
                            )
                            calibration_binding = (
                                "diagnostic_accuracy_matrix_node_id_and_tensor_names"
                                if calibration.get("schema") == DIAGNOSTIC_ACCURACY_MATRIX_SCHEMA
                                else "onnx_node_id_and_tensor_names"
                            )
                        elif allow_legacy_shape_calibration:
                            signature = conv_signature(input_shape, weight_shape)
                            scales = calibration["scales_by_signature"].get(signature)
                            if scales is None:
                                raise ValueError(f"missing legacy calibration for selected Conv: {signature}")
                            calibration_binding = "legacy_shape_signature_diagnostic_only"
                        else:
                            raise ValueError("legacy shape calibration requires --allow-legacy-shape-calibration")
                        input_scale = float(scales["input_scale"])
                        weight_scale = float(scales["weight_scale"])
                        weight_name = f"int8_weight_{conv_index}"
                        weight_var = relax.Var(weight_name, auto.relax_tensor_info(relax, weight_shape, "int8"))
                        inputs.extend([weight_var, bias_var])
                        runtime_values[weight_name] = quantize_weight_s8(raw_weight, scale=weight_scale)
                        runtime_values[bias_name] = bias_value
                        annotation = region_annotations.get(node_id)
                        if retain_int8_regions and annotation is not None and not annotation["is_region_entry"]:
                            observed_scale = int8_tensor_scales.get(str(node.input[0]))
                            if observed_scale is None or not math.isclose(observed_scale, input_scale, rel_tol=1e-6):
                                raise ValueError(f"INT8 region input scale mismatch: {node_id}")
                            quantized = tensor_map[node.input[0]]
                        else:
                            quant_name = f"mixed_quant_s8_{conv_index}"
                            quant_gv = bb.add_func(make_quantize_s8_primfunc(stack, input_shape, input_scale, quant_name), quant_name)
                            block_roles[quant_name] = "quantize"
                            quantized = bb.emit(relax.call_tir(quant_gv, [tensor_map[node.input[0]]], auto.relax_tensor_info(relax, input_shape, "int8")))
                        if retain_int8_regions and annotation is not None:
                            region = regions_by_id[str(annotation["region_id"])]
                            if annotation["is_region_exit"]:
                                output_scale = None
                                output_dtype = "float16"
                            else:
                                next_node_id = str(region["node_ids"][int(annotation["region_position"]) + 1])
                                next_record = conv_record_by_id[next_node_id]
                                next_scales = resolve_node_scales(
                                    calibration,
                                    node_id=next_node_id,
                                    input_name=str(next_record["input_name"]),
                                    weight_name=str(next_record["weight_name"]),
                                    onnx_sha256=onnx_digest,
                                )
                                output_scale = float(next_scales["input_scale"])
                                output_dtype = "int8"
                            fused_name = f"mixed_int8_region_fused_{conv_index}"
                            fused_gv = bb.add_func(
                                make_s8_conv_fused_primfunc(
                                    stack,
                                    spec,
                                    input_scale * weight_scale,
                                    fused_name,
                                    requant_output_scale=output_scale,
                                ),
                                fused_name,
                            )
                            block_roles[fused_name] = "int8_fused"
                            tensor_map[node.output[0]] = bb.emit(
                                relax.call_tir(
                                    fused_gv,
                                    [quantized, weight_var, bias_var],
                                    auto.relax_tensor_info(relax, output_shape, output_dtype),
                                )
                            )
                            if output_scale is not None:
                                int8_tensor_scales[str(node.output[0])] = output_scale
                        elif fuse_int8_epilogue:
                            fused_name = f"mixed_int8_fused_{conv_index}"
                            fused_gv = bb.add_func(make_s8_conv_fused_primfunc(stack, spec, input_scale * weight_scale, fused_name), fused_name)
                            block_roles[fused_name] = "int8_fused"
                            tensor_map[node.output[0]] = bb.emit(relax.call_tir(fused_gv, [quantized, weight_var, bias_var], auto.relax_tensor_info(relax, output_shape, "float16")))
                        else:
                            accum_name = f"mixed_int8_accum_{conv_index}"
                            accum_gv = bb.add_func(make_s8_conv_accum_primfunc(stack, spec, accum_name), accum_name)
                            block_roles[accum_name] = "int8_accum"
                            n, _, h, w = input_shape
                            cout = weight_shape[0]
                            group = int(spec["group"])
                            oh, ow = auto.conv2d_out_hw(h, w, weight_shape[2], weight_shape[3], spec["strides"], spec["pads"])
                            accum_shape = [group, n * oh * ow, max(128, cout // group)]
                            accum = bb.emit(relax.call_tir(accum_gv, [quantized, weight_var], auto.relax_tensor_info(relax, accum_shape, "int32")))
                            dequant_name = f"mixed_dequant_bias_{conv_index}"
                            dequant_gv = bb.add_func(make_dequant_bias_fp16_primfunc(stack, spec, input_scale * weight_scale, dequant_name), dequant_name)
                            block_roles[dequant_name] = "dequant"
                            tensor_map[node.output[0]] = bb.emit(relax.call_tir(dequant_gv, [accum, bias_var], auto.relax_tensor_info(relax, output_shape, "float16")))
                    else:
                        weight_name = f"fp16_weight_{conv_index}"
                        weight_var = relax.Var(weight_name, auto.relax_tensor_info(relax, weight_shape, "float16"))
                        inputs.extend([weight_var, bias_var])
                        runtime_values[weight_name] = raw_weight.astype("float16")
                        runtime_values[bias_name] = bias_value
                        prim_name = auto.route.sanitize_name(f"mixed_fp16_conv_{conv_index}_{op_name}", f"mixed_fp16_conv_{conv_index}")
                        gv = bb.add_func(auto.make_fp16_conv_direct_primfunc(stack, spec, prim_name), prim_name)
                        block_roles[prim_name] = "fp16_conv"
                        tensor_map[node.output[0]] = bb.emit(relax.call_tir(gv, [tensor_map[node.input[0]], weight_var, bias_var], auto.relax_tensor_info(relax, output_shape, "float16")))
                    continue
                if node.op_type == "Relu":
                    relu_index += 1
                    shape = [int(value) for value in shape_map[node.output[0]]]
                    input_scale = int8_tensor_scales.get(str(node.input[0]))
                    dtype = "int8" if input_scale is not None else "float16"
                    prefix = "mixed_int8_relu" if dtype == "int8" else "mixed_fp16_relu"
                    prim_name = auto.route.sanitize_name(f"{prefix}_{relu_index}_{op_name}", f"{prefix}_{relu_index}")
                    relu_primfunc = (
                        make_relu_s8_primfunc(stack, shape, prim_name)
                        if dtype == "int8"
                        else make_relu_fp16_primfunc(stack, shape, prim_name)
                    )
                    gv = bb.add_func(relu_primfunc, prim_name)
                    block_roles[prim_name] = "relu"
                    tensor_map[node.output[0]] = bb.emit(
                        relax.call_tir(gv, [tensor_map[node.input[0]]], auto.relax_tensor_info(relax, shape, dtype))
                    )
                    if input_scale is not None:
                        int8_tensor_scales[str(node.output[0])] = input_scale
                    continue
                if node.op_type == "Add":
                    add_index += 1
                    shape = [int(value) for value in shape_map[node.output[0]]]
                    prim_name = auto.route.sanitize_name(f"mixed_fp16_add_{add_index}_{op_name}", f"mixed_fp16_add_{add_index}")
                    gv = bb.add_func(make_add_fp16_primfunc(stack, shape, prim_name), prim_name)
                    block_roles[prim_name] = "add"
                    tensor_map[node.output[0]] = bb.emit(
                        relax.call_tir(
                            gv,
                            [tensor_map[node.input[0]], tensor_map[node.input[1]]],
                            auto.relax_tensor_info(relax, shape, "float16"),
                        )
                    )
                    continue
                raise ValueError(f"unsupported ONNX op {node.op_type}: {op_name}")

            outputs = []
            output_shapes: dict[str, list[int]] = {}
            for output_index, item in enumerate(model.graph.output, start=1):
                shape = [int(value) for value in shape_map[item.name]]
                name = f"mixed_cast_output_{output_index}"
                gv = bb.add_func(make_cast_primfunc(stack, shape, "float16", "float32", name), name)
                block_roles[name] = "cast"
                outputs.append(
                    bb.emit(
                        relax.call_tir(
                            gv,
                            [tensor_map[item.name]],
                            auto.relax_tensor_info(relax, shape, "float32"),
                        )
                    )
                )
                output_shapes[item.name] = shape
            emitted = bb.emit_output(relax.Tuple(outputs) if len(outputs) > 1 else outputs[0])
        bb.emit_func_output(emitted)
    return {
        "mod": bb.finalize(),
        "inputs": inputs,
        "input_specs": input_specs,
        "runtime_values": runtime_values,
        "block_roles": block_roles,
        "output_shapes": output_shapes,
        "onnx_sha256": onnx_digest,
        "conv_count": conv_index,
        "calibration_binding": calibration_binding,
        "region_formation": region_formation,
    }


def schedule_mixed_module(stack: dict[str, Any], mod: Any, block_roles: dict[str, str]) -> tuple[Any, dict[str, Any]]:
    tvm = stack["tvm"]
    target = stack["target"]
    import tvm.s_tir.dlight as dl
    from tvm.s_tir.dlight.gpu.matmul import MatmulInt8Tensorization, MatmulTensorization

    out = tvm.IRModule(dict(mod.functions), attrs=mod.attrs)
    records: dict[str, Any] = {}
    for gv, func in list(out.functions_items()):
        if not hasattr(func, "buffer_map"):
            continue
        name = gv.name_hint
        role = block_roles.get(name, "unknown")
        if role == "fp16_conv":
            rules = [MatmulTensorization()]
        elif role in {"int8_accum", "int8_fused"}:
            rules = [MatmulInt8Tensorization()]
        else:
            rules = [dl.gpu.Fallback()]
        try:
            with target, tvm.transform.PassContext(opt_level=3):
                scheduled = dl.ApplyDefaultSchedule(*rules)(tvm.IRModule({gv: func}))
            out.update_func(gv, scheduled[gv])
            records[name] = {"role": role, "status": "scheduled", "counts": _load_auto_decomp().counts(scheduled[gv].script())}
        except Exception as exc:
            records[name] = {"role": role, "status": "failed", "error": repr(exc)}
            raise
    return out, records


def run_fp16_baseline(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    auto = _load_auto_decomp()

    auto.cap.configure_tvm_env(str(args.gpu))
    stack = auto.cap.import_tvm_stack()
    tvm = stack["tvm"]
    from tvm import relax
    selected_ids: set[str] = set()
    calibration = None
    calibration_evidence = None
    validate_calibration_mode(
        int8_conv_limit=int(args.int8_conv_limit),
        per_node_calibration=args.per_node_calibration,
        calibration_ap_report=args.calibration_ap_report,
        allow_legacy_shape_calibration=bool(args.allow_legacy_shape_calibration),
    )
    if args.int8_conv_limit > 0:
        manifest = build_selection_manifest(args.onnx, width=args.width, policy=args.mixed_policy)
        selected_ids = set(manifest["selected_node_ids"][: args.int8_conv_limit])
        calibration = json.loads(args.per_node_calibration.read_text(encoding="utf-8"))
        validate_per_node_calibration_for_lowering(
            calibration,
            records=manifest["conv_records"],
            onnx_sha256=manifest["onnx_sha256"],
            allow_diagnostic_accuracy_matrix=bool(args.allow_diagnostic_accuracy_matrix),
        )
        calibration_evidence = {
            "schema": calibration["schema"],
            "quantization_semantics": calibration["quantization_semantics"],
            "manifest": str(args.per_node_calibration),
            "manifest_sha256": sha256_file(args.per_node_calibration),
            "calibration_split": calibration["calibration_split"],
            "sample_count": calibration["sample_count"],
            "source": calibration["calibration_source"],
            "source_sha256": calibration["calibration_source_sha256"],
            "summary": calibration["calibration_summary"],
            "summary_sha256": calibration["calibration_summary_sha256"],
            "split_source": calibration["calibration_split_source"],
            "node_count": calibration["node_count"],
        }
    spec = build_fp16_blockbuilder_module(
        stack,
        args.onnx,
        selected_node_ids=selected_ids,
        calibration=calibration,
        fuse_int8_epilogue=bool(args.fuse_int8_epilogue),
        retain_int8_regions=bool(args.retain_int8_regions),
        allow_legacy_shape_calibration=bool(args.allow_legacy_shape_calibration),
    )
    scheduled, schedules = schedule_mixed_module(stack, spec["mod"], spec["block_roles"])
    with stack["target"], tvm.transform.PassContext(opt_level=3):
        executable = tvm.compile(scheduled, target=stack["target"])
    vm = relax.VirtualMachine(executable, stack["dev"])
    with np.load(args.input_npz, allow_pickle=False) as payload:
        spatial = np.asarray(payload["spatial_features"][0], dtype="float32")
    runtime_args = []
    for var in spec["inputs"]:
        name = str(var.name_hint)
        value = spec["runtime_values"].get(name, spatial)
        runtime_args.append(tvm.runtime.tensor(value, device=stack["dev"]))
    candidate_obj = vm["main"](*runtime_args)
    stack["dev"].sync()
    candidate_outputs = [auto.to_numpy(value).astype("float32") for value in auto.unpack_outputs(candidate_obj)]
    import onnxruntime as ort

    reference_outputs = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"]).run(
        None,
        {spec["input_specs"][0]["source_name"]: spatial},
    )
    numerical = compare_full_outputs(reference_outputs, candidate_outputs)
    for _ in range(args.warmup):
        vm["main"](*runtime_args)
    stack["dev"].sync()
    evaluator = vm.time_evaluator("main", stack["dev"], number=args.number, repeat=args.repeat)
    times = [float(value) * 1000.0 for value in evaluator(*runtime_args).results]
    numerical_passed = all(
        float(item["mean_abs_err"]) <= float(args.max_mean_abs_error)
        for item in numerical
    )
    diagnostic_calibration = bool(selected_ids) and spec["calibration_binding"] != "onnx_node_id_and_tensor_names"
    status = (
        "failed_numerical_gate"
        if not numerical_passed
        else "diagnostic_only_accuracy_matrix_calibration"
        if spec["calibration_binding"] == "diagnostic_accuracy_matrix_node_id_and_tensor_names"
        else "diagnostic_only_legacy_shape_calibration"
        if diagnostic_calibration
        else "success"
    )
    return {
        "schema": "codriving_mixed_auto_lowering_fp16_baseline_v1",
        "status": status,
        "trusted_for_final_frontier": status == "success",
        "width": args.width,
        "gpu": str(args.gpu),
        "measurement": {"warmup": args.warmup, "number": args.number, "repeat": args.repeat},
        "onnx": str(args.onnx),
        "onnx_sha256": spec["onnx_sha256"],
        "conv_count": spec["conv_count"],
        "selected_int8_node_ids": sorted(selected_ids),
        "fuse_int8_epilogue": bool(args.fuse_int8_epilogue),
        "retain_int8_regions": bool(args.retain_int8_regions),
        "region_formation": spec["region_formation"],
        "calibration_binding": spec["calibration_binding"],
        "calibration_evidence": calibration_evidence,
        "numerical_gate": {"max_mean_abs_error": args.max_mean_abs_error, "passed": numerical_passed},
        "latency_ms_repeats": times,
        "latency_ms_p50": sorted(times)[len(times) // 2],
        "schedule_records": schedules,
        "output_shapes": spec["output_shapes"],
        "numerical_vs_ort_fp32": numerical,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--width", required=True)
    parser.add_argument("--mixed-policy", choices=sorted(POLICY_FRACTIONS), default="top25_flops")
    parser.add_argument("--build-fp16-baseline", action="store_true")
    parser.add_argument("--input-npz", type=Path, default=None)
    parser.add_argument("--gpu", default="3")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--number", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--int8-conv-limit", type=int, default=0)
    parser.add_argument("--calibration-ap-report", type=Path, default=None)
    parser.add_argument("--per-node-calibration", type=Path, default=None)
    parser.add_argument("--fuse-int8-epilogue", action="store_true")
    parser.add_argument("--retain-int8-regions", action="store_true")
    parser.add_argument("--allow-legacy-shape-calibration", action="store_true")
    parser.add_argument("--allow-diagnostic-accuracy-matrix", action="store_true")
    parser.add_argument("--max-mean-abs-error", type=float, default=0.05)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()
    if not args.onnx.is_file():
        parser.error(f"ONNX file not found: {args.onnx}")
    if args.build_fp16_baseline and (args.input_npz is None or not args.input_npz.is_file()):
        parser.error("--input-npz is required for --build-fp16-baseline")
    return args


def main() -> int:
    args = parse_args()
    started = time.time()
    manifest = (
        run_fp16_baseline(args)
        if args.build_fp16_baseline
        else build_selection_manifest(args.onnx, width=args.width, policy=args.mixed_policy)
    )
    manifest["elapsed_s"] = time.time() - started
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return result_exit_code(manifest)


if __name__ == "__main__":
    raise SystemExit(main())
