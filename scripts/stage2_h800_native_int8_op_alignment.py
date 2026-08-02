#!/usr/bin/env python3
"""Op-level numeric alignment probe for the native INT8 TVM backbone route."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import sys
import traceback
from dataclasses import dataclass
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.stage2.native_int8_full_onnx import quantize_activation_uint8  # noqa: E402
from scripts.stage2_h800_native_int8_real_activation_bridge import (  # noqa: E402
    DEFAULT_HEAL_ROOT,
    DEFAULT_ROUTE_DIR,
    build_op_level_alignment_record,
    summarize_numpy_array,
)
from scripts.stage2_h800_true_fp16_ap_eval import best_checkpoint, dtype_counts, write_json  # noqa: E402


DEFAULT_UINT8_ZERO_POINT = 128


@dataclass(frozen=True)
class QuantTensor:
    values_uint8: np.ndarray
    scale: float
    zero_point: int
    tensor_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="s0_024")
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--heal-root", default=DEFAULT_HEAL_ROOT)
    parser.add_argument("--route-dir", default=str(DEFAULT_ROUTE_DIR))
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--eval-range", default="102.4,102.4")
    parser.add_argument("--num-spatial-convs", type=int, default=2)
    parser.add_argument(
        "--op-name-regex",
        action="append",
        default=[],
        help="Select Conv ops whose ONNX op_name matches this regex. Can be repeated.",
    )
    parser.add_argument(
        "--conv-indices",
        default="",
        help="Comma-separated Conv indices to probe, counted over Conv nodes only.",
    )
    parser.add_argument(
        "--trace-native-prefix",
        action="store_true",
        help="Simulate the native INT8 Conv/Relu/Add prefix from captured spatial_features and write uint8 trace stats.",
    )
    parser.add_argument(
        "--trace-max-ops",
        type=int,
        default=0,
        help="Maximum ONNX ops to simulate when --trace-native-prefix is set; 0 means no explicit limit.",
    )
    parser.add_argument(
        "--trace-stop-output",
        action="append",
        default=[],
        help="Stop native prefix simulation once this output tensor name is produced. Can be repeated.",
    )
    parser.add_argument(
        "--collect-reference-ranges",
        action="store_true",
        help="Write layer-prefix reference-range target metadata for calibrated scale-aware INT8 reruns.",
    )
    parser.add_argument(
        "--reference-range-stop-output",
        action="append",
        default=[],
        help="Stop reference-range target collection once this output tensor name is reached. Can be repeated.",
    )
    parser.add_argument(
        "--reference-range-out",
        default="tensor_reference_range_targets_v1.json",
        help="Output filename under --raw-dir for reference-range collection target metadata.",
    )
    parser.add_argument(
        "--execute-reference-range-capture",
        action="store_true",
        help="Load the PyTorch model, export module inventory, and capture reference ranges with hooks.",
    )
    parser.add_argument(
        "--module-inventory-out",
        default="pytorch_module_inventory_layer0_v1.json",
        help="Output filename under --raw-dir for PyTorch module inventory when executing reference range capture.",
    )
    parser.add_argument(
        "--reference-range-plan-out",
        default="tensor_reference_range_capture_plan_layer0_v1.json",
        help="Output filename under --raw-dir for hook capture plan when executing reference range capture.",
    )
    parser.add_argument(
        "--reference-ranges-out",
        default="tensor_reference_ranges_layer0_v1.json",
        help="Output filename under --raw-dir for captured reference ranges.",
    )
    parser.add_argument(
        "--reference-range-calibration-out",
        default="tensor_quant_params_calibration_v2.json",
        help="Output filename under --raw-dir for calibration derived from captured reference ranges.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def parse_int_list(value: str | None) -> list[int]:
    if not value:
        return []
    out: list[int] = []
    for part in str(value).split(","):
        text = part.strip()
        if not text:
            continue
        out.append(int(text))
    return out


def all_conv_records_from_op_records(records: list[dict[str, Any]]) -> list[tuple[int, dict[str, Any]]]:
    convs: list[tuple[int, dict[str, Any]]] = []
    conv_index = 0
    for item in records:
        if item.get("op_type") != "Conv":
            continue
        convs.append((conv_index, item))
        conv_index += 1
    return convs


def select_conv_records_from_op_records(
    records: list[dict[str, Any]],
    *,
    limit: int,
    op_name_regexes: list[str] | tuple[str, ...] | None = None,
    conv_indices: list[int] | tuple[int, ...] | None = None,
    spatial_only: bool = False,
) -> list[tuple[int, dict[str, Any]]]:
    convs = all_conv_records_from_op_records(records)
    index_filter = {int(item) for item in conv_indices or []}
    regexes = [re.compile(str(item)) for item in op_name_regexes or [] if str(item)]

    selected: list[tuple[int, dict[str, Any]]] = []
    for conv_index, item in convs:
        if index_filter and conv_index not in index_filter:
            continue
        if regexes and not any(pattern.search(str(item.get("op_name") or "")) for pattern in regexes):
            continue
        if spatial_only and str(item.get("input_name")) != "spatial_features":
            continue
        selected.append((conv_index, item))
        if int(limit) > 0 and len(selected) >= int(limit):
            break
    return selected


def prefix_reference_range_targets(
    op_records: list[dict[str, Any]],
    *,
    stop_output_names: list[str] | tuple[str, ...] | None = None,
    op_types: list[str] | tuple[str, ...] = ("Conv", "Relu", "Add"),
) -> list[dict[str, Any]]:
    stop_outputs = {str(item) for item in stop_output_names or []}
    selected_types = {str(item) for item in op_types}
    targets: list[dict[str, Any]] = []
    for op_index, op_record in enumerate(op_records):
        op_type = str(op_record.get("op_type") or "")
        output_name = str(op_record.get("output_name") or "")
        if op_type in selected_types and output_name:
            stop_matched = output_name in stop_outputs
            targets.append(
                {
                    "schema": "native_int8_reference_range_target_v1",
                    "op_index": int(op_index),
                    "op_type": op_type,
                    "op_name": str(op_record.get("op_name") or f"{op_type}_{op_index}"),
                    "output_name": output_name,
                    "input_name": op_record.get("input_name"),
                    "input_names": op_record.get("input_names"),
                    "requires_reference_range": True,
                    "stop_matched": bool(stop_matched),
                }
            )
            if stop_matched:
                break
        elif output_name in stop_outputs:
            break
    return targets


def build_reference_range_targets_payload(
    *,
    label: str,
    route_dir: Path,
    raw_dir: Path,
    op_records: list[dict[str, Any]],
    stop_output_names: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    targets = prefix_reference_range_targets(
        op_records,
        stop_output_names=list(stop_output_names),
    )
    missing_reason = (
        "reference_range_targets_recorded_without_pytorch_hooks"
        if targets
        else "no_reference_range_targets_matched"
    )
    return {
        "schema": "native_int8_reference_range_targets_v1",
        "status": "blocked",
        "label": str(label),
        "failure_reason": missing_reason,
        "target_count": len(targets),
        "stop_output_names": [str(item) for item in stop_output_names],
        "targets": targets,
        "route_dir": str(route_dir),
        "raw_artifact": str(raw_dir),
        "next_action": (
            "register PyTorch hooks for each target output and rerun to emit "
            "tensor_reference_ranges_layer0_v1.json"
        ),
        "full_network_claim": False,
        "ap_measured": False,
    }


def spatial_conv_records(route_dir: Path, limit: int) -> list[tuple[int, dict[str, Any]]]:
    records = load_json(route_dir / "onnx_op_records.json")
    return select_conv_records_from_op_records(records, limit=limit, spatial_only=True)


def weight_plan_by_conv_index(inventory: dict[str, Any]) -> dict[int, dict[str, Any]]:
    weights = [
        item
        for item in inventory.get("runtime_arg_plan", [])
        if item.get("role") == "weight_input"
    ]
    return {idx: item for idx, item in enumerate(weights)}


def module_suffix_candidates(op_name: str) -> list[str]:
    path = str(op_name).strip("/")
    for terminal_op in ("/Conv", "/Relu"):
        if path.endswith(terminal_op):
            path = path[: -len(terminal_op)]
            break
    dotted = path.replace("/", ".")
    candidates = [dotted]
    if dotted.startswith("pyramid_backbone."):
        candidates.append(dotted[len("pyramid_backbone.") :])
    if dotted.startswith("pb."):
        candidates.append(dotted[len("pb.") :])
    collapsed = re.sub(r"resnet\.layer(\d+)\.layer\1\.", r"resnet.layer\1.", dotted)
    if collapsed not in candidates:
        candidates.append(collapsed)
    collapsed_downsample = collapsed.replace(".downsample.downsample.", ".downsample.")
    if collapsed_downsample not in candidates:
        candidates.append(collapsed_downsample)
    for candidate in list(candidates):
        if candidate.startswith("pyramid_backbone."):
            stripped = candidate[len("pyramid_backbone.") :]
            if stripped not in candidates:
                candidates.append(stripped)
        if candidate.startswith("pb."):
            stripped = candidate[len("pb.") :]
            if stripped not in candidates:
                candidates.append(stripped)
        if candidate.startswith("resnet."):
            stripped = candidate[len("resnet.") :]
            if stripped not in candidates:
                candidates.append(stripped)
    return candidates


def resolve_module_name(module_names: list[str], op_name: str) -> str | None:
    candidates = module_suffix_candidates(op_name)
    for suffix in candidates:
        for name in module_names:
            if name.endswith(suffix):
                return name
    return None


def _conv_capture_module_name(module_names: list[str], op_name: str) -> str | None:
    module_name = resolve_module_name(module_names, op_name)
    if module_name is None:
        return None

    module_name_set = set(module_names)
    conv_to_bn_suffixes = (
        (".conv1", ".bn1"),
        (".conv2", ".bn2"),
        (".conv3", ".bn3"),
        (".downsample.0", ".downsample.1"),
    )
    for conv_suffix, bn_suffix in conv_to_bn_suffixes:
        if module_name.endswith(conv_suffix):
            bn_name = module_name[: -len(conv_suffix)] + bn_suffix
            if bn_name in module_name_set:
                return bn_name
    return module_name


def _relu_capture_source(
    *,
    module_names: list[str],
    op_name: str,
) -> dict[str, Any] | None:
    path = str(op_name).strip("/")
    if not path.endswith("/Relu"):
        return None
    path = path[: -len("/Relu")]
    parts = path.split("/")
    if not parts:
        return None
    relu_token = parts[-1]
    if relu_token == "relu":
        call_index = 0
    else:
        match = re.fullmatch(r"relu_(\d+)", relu_token)
        if not match:
            return None
        call_index = int(match.group(1))
    parts[-1] = "relu"
    module_op_name = "/" + "/".join(parts) + "/Relu"
    module_name = resolve_module_name(module_names, module_op_name)
    if module_name is None:
        return None
    return {
        "source_module_name": module_name,
        "source_call_index": call_index,
    }


def _add_capture_source(
    *,
    targets: list[dict[str, Any]],
    module_names: list[str],
    target_index: int,
) -> dict[str, Any] | None:
    op_name = str(targets[target_index].get("op_name") or "")
    path = op_name.strip("/")
    if not path.endswith("/Add"):
        return None
    block_prefix = path[: -len("/Add")]
    for later in targets[target_index + 1 :]:
        if str(later.get("op_type") or "") != "Relu":
            continue
        relu_op_name = str(later.get("op_name") or "")
        relu_path = relu_op_name.strip("/")
        if not relu_path.startswith(f"{block_prefix}/relu"):
            continue
        return _relu_capture_source(module_names=module_names, op_name=relu_op_name)
    return None


def reference_range_capture_plan(
    *,
    targets: list[dict[str, Any]],
    module_names: list[str],
) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for fallback_index, target in enumerate(targets):
        op_type = str(target.get("op_type") or "")
        op_name = str(target.get("op_name") or "")
        output_name = str(target.get("output_name") or "")
        op_index = int(target.get("op_index", fallback_index))

        item = {
            "schema": "native_int8_reference_range_capture_plan_item_v1",
            "op_index": op_index,
            "op_type": op_type,
            "op_name": op_name,
            "output_name": output_name,
            "input_name": target.get("input_name"),
            "input_names": target.get("input_names"),
            "source_capture": None,
            "source_module_name": None,
            "source_call_index": None,
            "capture_status": "blocked",
            "failure_reason": None,
        }

        if op_type == "Conv":
            module_name = _conv_capture_module_name(module_names, op_name)
            if module_name:
                item.update(
                    {
                        "source_capture": "module_forward_hook",
                        "source_module_name": module_name,
                        "capture_status": "hook_ready",
                    }
                )
            else:
                item["failure_reason"] = "missing_pytorch_module_for_reference_range_target"
        elif op_type == "Relu":
            relu_source = _relu_capture_source(module_names=module_names, op_name=op_name)
            if relu_source:
                item.update(
                    {
                        "source_capture": "module_forward_hook_call_index",
                        "source_module_name": relu_source["source_module_name"],
                        "source_call_index": relu_source["source_call_index"],
                        "capture_status": "hook_ready",
                    }
                )
            else:
                item["failure_reason"] = "missing_pytorch_relu_module_for_reference_range_target"
        elif op_type == "Add":
            add_source = _add_capture_source(
                targets=targets,
                module_names=module_names,
                target_index=fallback_index,
            )
            if add_source:
                item.update(
                    {
                        "source_capture": "module_forward_pre_hook_call_index",
                        "source_module_name": add_source["source_module_name"],
                        "source_call_index": add_source["source_call_index"],
                        "capture_status": "hook_ready",
                    }
                )
            else:
                item["failure_reason"] = "add_output_requires_following_relu_pre_hook_capture"
        else:
            item["failure_reason"] = "unsupported_reference_range_target_op_type"

        items.append(item)

    hook_ready = sum(1 for item in items if item["capture_status"] == "hook_ready")
    blocked = sum(1 for item in items if item["capture_status"] == "blocked")
    return {
        "schema": "native_int8_reference_range_capture_plan_v1",
        "summary": {
            "total": len(items),
            "hook_ready": hook_ready,
            "blocked": blocked,
        },
        "items": items,
    }


def minmax_align_to_reference(reference: np.ndarray, candidate: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    ref = np.asarray(reference, dtype=np.float32)
    cand = np.asarray(candidate, dtype=np.float32)
    cand_min = float(np.min(cand))
    cand_max = float(np.max(cand))
    ref_min = float(np.min(ref))
    ref_max = float(np.max(ref))
    if cand_max == cand_min:
        aligned = np.full(ref.shape, ref_min, dtype=np.float32)
        scale = 1.0
    else:
        scale = float((ref_max - ref_min) / (cand_max - cand_min)) if ref_max != ref_min else 1.0
        aligned = ((cand - cand_min) * scale + ref_min).astype(np.float32)
    return aligned, {
        "align_scheme": "candidate_minmax_to_reference_range",
        "candidate_min": cand_min,
        "candidate_max": cand_max,
        "reference_min": ref_min,
        "reference_max": ref_max,
        "scale": scale,
    }


def summarize_uint8_trace_tensor(array: np.ndarray, *, tensor_name: str) -> dict[str, Any]:
    arr = np.asarray(array, dtype=np.uint8)
    return {
        "tensor_name": str(tensor_name),
        "shape": [int(dim) for dim in arr.shape],
        "dtype": str(arr.dtype),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "zero_fraction": float(np.mean(arr == 0)),
        "max_fraction": float(np.mean(arr == 255)),
        "unique_count": int(np.unique(arr).size) if arr.size <= 1_000_000 else None,
    }


def _reference_range_name_and_bounds(item: dict[str, Any]) -> tuple[str, float, float]:
    source = item.get("reference") if isinstance(item.get("reference"), dict) else item
    tensor_name = str(source.get("tensor_name") or item.get("tensor_name") or "")
    if not tensor_name:
        raise ValueError(f"reference range is missing tensor_name: {item}")
    if "min" in source and "max" in source:
        minimum = source["min"]
        maximum = source["max"]
    elif "reference_min" in source and "reference_max" in source:
        minimum = source["reference_min"]
        maximum = source["reference_max"]
    else:
        raise ValueError(f"reference range is missing min/max for tensor {tensor_name}: {item}")
    return tensor_name, float(minimum), float(maximum)


def _scale_from_reference_range(
    *,
    minimum: float,
    maximum: float,
    min_scale: float,
) -> float:
    if float(min_scale) <= 0.0:
        raise ValueError("min_scale must be positive")
    bound = max(abs(float(minimum)), abs(float(maximum)))
    if bound <= 0.0:
        return float(min_scale)
    return max(float(bound) / 127.0, float(min_scale))


def build_tensor_quant_params_from_reference_ranges(
    *,
    label: str,
    reference_ranges: list[dict[str, Any]],
    graph_input_quant: dict[str, Any],
    route_dir: str,
    raw_artifact: str,
    min_scale: float = 1e-6,
    output_zero_point: int = DEFAULT_UINT8_ZERO_POINT,
) -> dict[str, Any]:
    input_tensor_name = str(graph_input_quant.get("tensor_name") or "spatial_features")
    input_scale = float(graph_input_quant.get("scale") or 0.0)
    if input_scale <= 0.0:
        raise ValueError(f"graph input scale must be positive: {graph_input_quant}")
    input_zero_point = int(graph_input_quant.get("zero_point", 0))

    params: dict[str, dict[str, Any]] = {
        input_tensor_name: {
            "scale": input_scale,
            "zero_point": input_zero_point,
            "source": "graph_input_quant",
        }
    }
    range_items: list[dict[str, Any]] = []
    for item in reference_ranges:
        tensor_name, minimum, maximum = _reference_range_name_and_bounds(item)
        scale = _scale_from_reference_range(
            minimum=minimum,
            maximum=maximum,
            min_scale=min_scale,
        )
        params[tensor_name] = {
            "scale": float(scale),
            "zero_point": int(output_zero_point),
            "source": "observed_reference_range",
            "reference_min": float(minimum),
            "reference_max": float(maximum),
            "range_absmax": float(max(abs(minimum), abs(maximum))),
            "quantization_scheme": "symmetric_uint8_zero_point_128_from_reference_range",
        }
        if "shape" in item:
            params[tensor_name]["shape"] = [int(dim) for dim in item.get("shape") or []]
        range_items.append(
            {
                "tensor_name": tensor_name,
                "min": float(minimum),
                "max": float(maximum),
                "scale": float(scale),
                "zero_point": int(output_zero_point),
            }
        )

    return {
        "schema": "native_int8_tensor_quant_params_calibration_v2",
        "status": "ready_for_scale_aware_simulator",
        "label": str(label),
        "route_dir": str(route_dir),
        "raw_artifact": str(raw_artifact),
        "policy": "observed_reference_absmax_symmetric_uint8_zp128",
        "range_count": len(range_items),
        "ranges": range_items,
        "params": params,
        "full_network_claim": False,
        "ap_measured": False,
    }


def build_reference_range_capture_payload(
    *,
    label: str,
    raw_dir: str | Path,
    route_dir: str | Path,
    plan: dict[str, Any],
    captured_tensors: dict[str, np.ndarray],
    sample_count: int,
    min_scale: float = 1e-6,
    output_zero_point: int = DEFAULT_UINT8_ZERO_POINT,
) -> dict[str, Any]:
    ranges: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for item in plan.get("items", []):
        output_name = str(item.get("output_name") or "")
        if str(item.get("capture_status") or "") != "hook_ready":
            continue
        if output_name not in captured_tensors:
            missing.append(
                {
                    "op_index": item.get("op_index"),
                    "op_type": item.get("op_type"),
                    "op_name": item.get("op_name"),
                    "output_name": output_name,
                    "failure_reason": "hook_ready_tensor_not_captured",
                }
            )
            continue
        array = np.asarray(captured_tensors[output_name], dtype=np.float32)
        minimum = float(np.min(array))
        maximum = float(np.max(array))
        scale = _scale_from_reference_range(
            minimum=minimum,
            maximum=maximum,
            min_scale=min_scale,
        )
        ranges.append(
            {
                "schema": "native_int8_reference_range_item_v1",
                "tensor_name": output_name,
                "op_index": int(item.get("op_index", len(ranges))),
                "op_type": str(item.get("op_type") or ""),
                "op_name": str(item.get("op_name") or ""),
                "shape": [int(dim) for dim in array.shape],
                "dtype": str(array.dtype),
                "min": minimum,
                "max": maximum,
                "mean": float(np.mean(array)),
                "std": float(np.std(array)),
                "recommended_uint8_scale": float(scale),
                "recommended_zero_point": int(output_zero_point),
                "sample_count": int(sample_count),
                "source_capture": item.get("source_capture"),
                "source_module_name": item.get("source_module_name"),
                "source_call_index": item.get("source_call_index"),
            }
        )
    status = "ready_for_calibration" if not missing else "blocked_missing_hook_ready_captures"
    return {
        "schema": "native_int8_reference_ranges_v1",
        "status": status,
        "label": str(label),
        "route_dir": str(route_dir),
        "raw_artifact": str(raw_dir),
        "range_count": len(ranges),
        "missing_count": len(missing),
        "reference_ranges": ranges,
        "missing": missing,
        "capture_plan_summary": plan.get("summary", {}),
        "full_network_claim": False,
        "ap_measured": False,
    }


def build_pytorch_module_inventory_payload(
    *,
    label: str,
    raw_dir: str | Path,
    route_dir: str | Path,
    module_map: dict[str, Any],
    scope: str,
) -> dict[str, Any]:
    module_names = sorted(str(name) for name in module_map.keys())
    modules: list[dict[str, Any]] = []
    for name in module_names:
        module = module_map[name]
        module_class = module.__class__
        modules.append(
            {
                "module_name": name,
                "class_name": module_class.__name__,
                "class_module": module_class.__module__,
            }
        )
    return {
        "schema": "native_int8_pytorch_module_inventory_v1",
        "status": "measured",
        "label": str(label),
        "scope": str(scope),
        "route_dir": str(route_dir),
        "raw_artifact": str(raw_dir),
        "module_count": len(modules),
        "module_names": module_names,
        "modules": modules,
        "full_network_claim": False,
        "ap_measured": False,
    }


def requantize_int32_scale_aware(
    *,
    accumulator: np.ndarray,
    input_scale: float,
    weight_scale: float,
    output_scale: float,
    output_zero_point: int = DEFAULT_UINT8_ZERO_POINT,
) -> np.ndarray:
    if float(output_scale) <= 0.0:
        raise ValueError("output_scale must be positive")
    effective_scale = float(input_scale) * float(weight_scale) / float(output_scale)
    quantized = np.rint(np.asarray(accumulator, dtype=np.float64) * effective_scale + int(output_zero_point))
    return np.clip(quantized, 0, 255).astype(np.uint8)


def scale_aware_add_uint8(
    *,
    lhs_uint8: np.ndarray,
    lhs_scale: float,
    lhs_zero_point: int,
    rhs_uint8: np.ndarray,
    rhs_scale: float,
    rhs_zero_point: int,
    output_scale: float,
    output_zero_point: int = DEFAULT_UINT8_ZERO_POINT,
) -> np.ndarray:
    if float(output_scale) <= 0.0:
        raise ValueError("output_scale must be positive")
    lhs = (np.asarray(lhs_uint8, dtype=np.float64) - int(lhs_zero_point)) * float(lhs_scale)
    rhs = (np.asarray(rhs_uint8, dtype=np.float64) - int(rhs_zero_point)) * float(rhs_scale)
    if lhs.shape != rhs.shape:
        raise ValueError(f"Add inputs have mismatched shapes: {lhs.shape} != {rhs.shape}")
    quantized = np.rint((lhs + rhs) / float(output_scale) + int(output_zero_point))
    return np.clip(quantized, 0, 255).astype(np.uint8)


def _tensor_quant_params(
    tensor_quant_params: dict[str, Any],
    tensor_name: str,
    *,
    default_scale: float | None = None,
    default_zero_point: int = DEFAULT_UINT8_ZERO_POINT,
) -> tuple[float, int]:
    params = tensor_quant_params.get(tensor_name) or {}
    scale = params.get("scale", default_scale)
    if scale is None:
        raise KeyError(f"missing quantization scale for tensor: {tensor_name}")
    if float(scale) <= 0.0:
        raise ValueError(f"quantization scale must be positive for tensor {tensor_name}: {scale}")
    zero_point = int(params.get("zero_point", default_zero_point))
    return float(scale), zero_point


def _conv2d_accumulator_int32(
    activation_uint8: np.ndarray,
    weight_int8: np.ndarray,
    op_record: dict[str, Any],
    *,
    input_zero_point: int,
    weight_zero_point: int = 0,
) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    activation = torch.from_numpy(np.asarray(activation_uint8)).to(device=device, dtype=torch.float32)
    activation = activation - float(input_zero_point)
    weight = torch.from_numpy(np.asarray(weight_int8)).to(device=device, dtype=torch.float32)
    if int(weight_zero_point) != 0:
        weight = weight - float(weight_zero_point)
    strides = [int(item) for item in op_record.get("strides", [1, 1])]
    pads = [int(item) for item in op_record.get("pads", [0, 0, 0, 0])]
    group = int(op_record.get("group") or 1)
    if pads[0] != pads[1] or pads[0] != pads[2] or pads[0] != pads[3]:
        activation = F.pad(activation, (pads[1], pads[3], pads[0], pads[2]))
        padding = 0
    else:
        padding = pads[0]
    conv = F.conv2d(activation, weight, bias=None, stride=tuple(strides), padding=padding, groups=group)
    return torch.round(conv).to(torch.int32).detach().cpu().numpy()


def simulate_scale_aware_int8_graph_prefix(
    *,
    graph_input: QuantTensor,
    op_records: list[dict[str, Any]],
    runtime_weights: Any,
    weights_by_index: dict[int, dict[str, Any]],
    tensor_quant_params: dict[str, Any],
    max_ops: int = 0,
    stop_output_names: list[str] | tuple[str, ...] | None = None,
) -> tuple[dict[str, QuantTensor], list[dict[str, Any]]]:
    tensor_map: dict[str, QuantTensor] = {
        graph_input.tensor_name: QuantTensor(
            values_uint8=np.asarray(graph_input.values_uint8, dtype=np.uint8),
            scale=float(graph_input.scale),
            zero_point=int(graph_input.zero_point),
            tensor_name=str(graph_input.tensor_name),
        )
    }
    trace: list[dict[str, Any]] = []
    stop_outputs = {str(item) for item in stop_output_names or []}
    conv_index = 0

    for op_index, op_record in enumerate(op_records):
        if int(max_ops) > 0 and op_index >= int(max_ops):
            break
        op_type = str(op_record.get("op_type") or "")
        op_name = str(op_record.get("op_name") or f"{op_type}_{op_index}")
        output_name = str(op_record.get("output_name") or "")
        if not output_name:
            raise ValueError(f"op record has no output_name: {op_name}")

        if op_type == "Conv":
            input_name = str(op_record.get("input_name") or "")
            if input_name not in tensor_map:
                raise KeyError(f"missing input tensor for {op_name}: {input_name}")
            input_tensor = tensor_map[input_name]
            weight_plan = weights_by_index.get(conv_index)
            if weight_plan is None:
                raise KeyError(f"missing weight plan for Conv index {conv_index}: {op_name}")
            weight_key = str(weight_plan["arg_name"])
            weight_quant = weight_plan.get("quantization") or {}
            weight_scale = float(weight_quant.get("scale") or 1.0)
            weight_zero_point = int(weight_quant.get("zero_point") or 0)
            output_scale, output_zero_point = _tensor_quant_params(
                tensor_quant_params,
                output_name,
                default_scale=input_tensor.scale * weight_scale,
            )
            accumulator = _conv2d_accumulator_int32(
                input_tensor.values_uint8,
                runtime_weights[weight_key],
                op_record,
                input_zero_point=input_tensor.zero_point,
                weight_zero_point=weight_zero_point,
            )
            out_uint8 = requantize_int32_scale_aware(
                accumulator=accumulator,
                input_scale=input_tensor.scale,
                weight_scale=weight_scale,
                output_scale=output_scale,
                output_zero_point=output_zero_point,
            )
            input_names = [input_name]
            conv_index_for_record: int | None = conv_index
            conv_index += 1
        elif op_type == "Relu":
            input_name = str(op_record.get("input_name") or "")
            if input_name not in tensor_map:
                raise KeyError(f"missing input tensor for {op_name}: {input_name}")
            input_tensor = tensor_map[input_name]
            output_scale, output_zero_point = _tensor_quant_params(
                tensor_quant_params,
                output_name,
                default_scale=input_tensor.scale,
                default_zero_point=input_tensor.zero_point,
            )
            real = (np.asarray(input_tensor.values_uint8, dtype=np.float64) - input_tensor.zero_point) * input_tensor.scale
            out_uint8 = np.clip(np.rint(np.maximum(real, 0.0) / output_scale + output_zero_point), 0, 255).astype(np.uint8)
            input_names = [input_name]
            conv_index_for_record = None
        elif op_type == "Add":
            input_names = [str(item) for item in op_record.get("input_names", [])]
            if len(input_names) != 2:
                raise ValueError(f"Add op requires exactly two input_names: {op_name}")
            missing = [name for name in input_names if name not in tensor_map]
            if missing:
                raise KeyError(f"missing Add inputs for {op_name}: {missing}")
            lhs = tensor_map[input_names[0]]
            rhs = tensor_map[input_names[1]]
            output_scale, output_zero_point = _tensor_quant_params(
                tensor_quant_params,
                output_name,
                default_scale=min(lhs.scale, rhs.scale),
            )
            out_uint8 = scale_aware_add_uint8(
                lhs_uint8=lhs.values_uint8,
                lhs_scale=lhs.scale,
                lhs_zero_point=lhs.zero_point,
                rhs_uint8=rhs.values_uint8,
                rhs_scale=rhs.scale,
                rhs_zero_point=rhs.zero_point,
                output_scale=output_scale,
                output_zero_point=output_zero_point,
            )
            conv_index_for_record = None
        elif op_type == "Identity":
            input_name = str(op_record.get("input_name") or "")
            if input_name not in tensor_map:
                raise KeyError(f"missing input tensor for {op_name}: {input_name}")
            input_tensor = tensor_map[input_name]
            output_scale, output_zero_point = _tensor_quant_params(
                tensor_quant_params,
                output_name,
                default_scale=input_tensor.scale,
                default_zero_point=input_tensor.zero_point,
            )
            out_uint8 = np.asarray(input_tensor.values_uint8, dtype=np.uint8).copy()
            input_names = [input_name]
            conv_index_for_record = None
        else:
            raise ValueError(f"unsupported op type in scale-aware prefix simulator: {op_type}")

        tensor_map[output_name] = QuantTensor(
            values_uint8=out_uint8,
            scale=float(output_scale),
            zero_point=int(output_zero_point),
            tensor_name=output_name,
        )
        trace.append(
            {
                "schema": "native_int8_scale_aware_prefix_trace_record_v1",
                "op_index": int(op_index),
                "conv_index": conv_index_for_record,
                "op_type": op_type,
                "op_name": op_name,
                "input_names": input_names,
                "output_name": output_name,
                "scale": float(output_scale),
                "zero_point": int(output_zero_point),
                "tensor": summarize_uint8_trace_tensor(out_uint8, tensor_name=output_name),
            }
        )
        if output_name in stop_outputs:
            break
    return tensor_map, trace


def simulate_native_int8_conv_requant(
    activation_uint8: np.ndarray,
    weight_int8: np.ndarray,
    op_record: dict[str, Any],
    *,
    centered_input_zero_point: int | None = None,
) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    activation = torch.from_numpy(np.asarray(activation_uint8)).to(device=device, dtype=torch.float32)
    if centered_input_zero_point is not None:
        activation = activation - float(centered_input_zero_point)
    weight = torch.from_numpy(np.asarray(weight_int8)).to(device=device, dtype=torch.float32)
    strides = [int(item) for item in op_record.get("strides", [1, 1])]
    pads = [int(item) for item in op_record.get("pads", [0, 0, 0, 0])]
    group = int(op_record.get("group") or 1)
    if pads[0] != pads[1] or pads[0] != pads[2] or pads[0] != pads[3]:
        activation = F.pad(activation, (pads[1], pads[3], pads[0], pads[2]))
        padding = 0
    else:
        padding = pads[0]
    conv = F.conv2d(activation, weight, bias=None, stride=tuple(strides), padding=padding, groups=group)
    requant = torch.clamp(torch.floor(conv / 256.0) + float(DEFAULT_UINT8_ZERO_POINT), 0.0, 255.0).to(torch.uint8)
    return requant.detach().cpu().numpy()


def simulate_native_int8_graph_prefix(
    *,
    activation_uint8: np.ndarray,
    op_records: list[dict[str, Any]],
    runtime_weights: Any,
    weights_by_index: dict[int, dict[str, Any]],
    max_ops: int = 0,
    stop_output_names: list[str] | tuple[str, ...] | None = None,
    uint8_zero_point: int = DEFAULT_UINT8_ZERO_POINT,
    graph_input_zero_point: int = 0,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    tensor_map: dict[str, np.ndarray] = {
        "spatial_features": np.asarray(activation_uint8, dtype=np.uint8),
    }
    tensor_zero_points: dict[str, int] = {
        "spatial_features": int(graph_input_zero_point),
    }
    trace: list[dict[str, Any]] = []
    stop_outputs = {str(item) for item in stop_output_names or []}
    conv_index = 0
    for op_index, op_record in enumerate(op_records):
        if int(max_ops) > 0 and op_index >= int(max_ops):
            break
        op_type = str(op_record.get("op_type") or "")
        op_name = str(op_record.get("op_name") or f"{op_type}_{op_index}")
        output_name = str(op_record.get("output_name") or "")
        if not output_name:
            raise ValueError(f"op record has no output_name: {op_name}")
        input_names: list[str]
        if op_type == "Conv":
            input_name = str(op_record.get("input_name") or "")
            if input_name not in tensor_map:
                raise KeyError(f"missing input tensor for {op_name}: {input_name}")
            weight_plan = weights_by_index.get(conv_index)
            if weight_plan is None:
                raise KeyError(f"missing weight plan for Conv index {conv_index}: {op_name}")
            weight_key = str(weight_plan["arg_name"])
            input_zero_point = int(tensor_zero_points.get(input_name, uint8_zero_point))
            out = simulate_native_int8_conv_requant(
                tensor_map[input_name],
                runtime_weights[weight_key],
                op_record,
                centered_input_zero_point=input_zero_point,
            )
            input_names = [input_name]
            conv_index_for_record: int | None = conv_index
            conv_index += 1
            tensor_zero_points[output_name] = int(uint8_zero_point)
        elif op_type == "Relu":
            input_name = str(op_record.get("input_name") or "")
            if input_name not in tensor_map:
                raise KeyError(f"missing input tensor for {op_name}: {input_name}")
            input_zero_point = int(tensor_zero_points.get(input_name, uint8_zero_point))
            out = np.maximum(
                np.asarray(tensor_map[input_name], dtype=np.uint8),
                np.uint8(input_zero_point),
            ).astype(np.uint8)
            input_names = [input_name]
            conv_index_for_record = None
            tensor_zero_points[output_name] = input_zero_point
        elif op_type == "Add":
            input_names = [str(item) for item in op_record.get("input_names", [])]
            if len(input_names) != 2:
                raise ValueError(f"Add op requires exactly two input_names: {op_name}")
            missing = [name for name in input_names if name not in tensor_map]
            if missing:
                raise KeyError(f"missing Add inputs for {op_name}: {missing}")
            lhs = np.asarray(tensor_map[input_names[0]], dtype=np.int32)
            rhs = np.asarray(tensor_map[input_names[1]], dtype=np.int32)
            if lhs.shape != rhs.shape:
                raise ValueError(f"Add inputs have mismatched shapes for {op_name}: {lhs.shape} != {rhs.shape}")
            lhs_zero_point = int(tensor_zero_points.get(input_names[0], uint8_zero_point))
            rhs_zero_point = int(tensor_zero_points.get(input_names[1], uint8_zero_point))
            output_zero_point = int(uint8_zero_point)
            out = np.clip(lhs + rhs - lhs_zero_point - rhs_zero_point + output_zero_point, 0, 255).astype(np.uint8)
            conv_index_for_record = None
            input_zero_point = None
            tensor_zero_points[output_name] = output_zero_point
        elif op_type == "Identity":
            input_name = str(op_record.get("input_name") or "")
            if input_name not in tensor_map:
                raise KeyError(f"missing input tensor for {op_name}: {input_name}")
            out = np.asarray(tensor_map[input_name], dtype=np.uint8).copy()
            input_names = [input_name]
            conv_index_for_record = None
            input_zero_point = int(tensor_zero_points.get(input_name, uint8_zero_point))
            tensor_zero_points[output_name] = input_zero_point
        else:
            raise ValueError(f"unsupported op type in native prefix simulator: {op_type}")
        tensor_map[output_name] = out
        trace.append(
            {
                "schema": "native_int8_prefix_trace_record_v1",
                "op_index": int(op_index),
                "conv_index": conv_index_for_record,
                "op_type": op_type,
                "op_name": op_name,
                "input_names": input_names,
                "output_name": output_name,
                "input_zero_point": input_zero_point,
                "output_zero_point": int(tensor_zero_points[output_name]),
                "tensor": summarize_uint8_trace_tensor(out, tensor_name=output_name),
            }
        )
        if output_name in stop_outputs:
            break
    return tensor_map, trace


def build_blocker(
    *,
    label: str,
    raw_dir: Path,
    reason: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema": "native_int8_op_level_numeric_alignment_blocker_v1",
        "status": "blocked",
        "label": str(label),
        "failure_reason": str(reason),
        "details": details or {},
        "raw_artifact": str(raw_dir),
        "full_network_claim": False,
        "ap_measured": False,
    }


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    if bool(getattr(args, "collect_reference_ranges", False)) and not bool(
        getattr(args, "execute_reference_range_capture", False)
    ):
        raw_dir = Path(args.raw_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)
        route_dir = Path(args.route_dir)
        op_records = load_json(route_dir / "onnx_op_records.json")
        payload = build_reference_range_targets_payload(
            label=str(args.label),
            route_dir=route_dir,
            raw_dir=raw_dir,
            op_records=op_records,
            stop_output_names=list(getattr(args, "reference_range_stop_output", []) or []),
        )
        output_name = str(getattr(args, "reference_range_out", "") or "tensor_reference_range_targets_v1.json")
        write_json(raw_dir / output_name, payload)
        write_json(raw_dir / "tensor_reference_range_collection_blocker.json", payload)
        return payload

    import torch
    from torch.utils.data import DataLoader

    heal_root = Path(args.heal_root)
    sys.path.insert(0, str(heal_root))
    os.chdir(heal_root)

    import opencood.hypes_yaml.yaml_utils as yaml_utils
    from opencood.data_utils.datasets import build_dataset
    from opencood.tools import train_utils
    from opencood.utils.common_utils import update_dict

    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    route_dir = Path(args.route_dir)
    inventory = load_json(route_dir / "tvm_operator_inventory.json")
    op_records = load_json(route_dir / "onnx_op_records.json")
    conv_indices = parse_int_list(getattr(args, "conv_indices", ""))
    op_name_regexes = list(getattr(args, "op_name_regex", []) or [])
    if conv_indices or op_name_regexes:
        op_items = select_conv_records_from_op_records(
            op_records,
            limit=int(args.num_spatial_convs),
            op_name_regexes=op_name_regexes,
            conv_indices=conv_indices,
        )
    else:
        op_items = select_conv_records_from_op_records(
            op_records,
            limit=int(args.num_spatial_convs),
            spatial_only=True,
        )
    if not op_items:
        blocker = build_blocker(
            label=str(args.label),
            raw_dir=raw_dir,
            reason="no_conv_records_matched_selection",
            details={
                "conv_indices": conv_indices,
                "op_name_regexes": op_name_regexes,
                "spatial_only": not (conv_indices or op_name_regexes),
            },
        )
        write_json(raw_dir / "native_int8_s0_024_op_level_numeric_alignment_blocker.json", blocker)
        return blocker
    weights_by_index = weight_plan_by_conv_index(inventory)
    runtime_weights = np.load(route_dir / "runtime_weights_int8.npz")

    ckpt_dir = Path(args.ckpt_dir)
    opt = argparse.Namespace(
        model_dir=str(ckpt_dir),
        fusion_method="intermediate",
        save_vis_interval=10**9,
        save_npy=False,
        range=args.eval_range,
        no_score=True,
        note="stage2_native_int8_op_alignment",
    )
    hypes = yaml_utils.load_yaml(None, opt)
    if "heter" in hypes:
        x_min, x_max = -eval(opt.range.split(",")[0]), eval(opt.range.split(",")[0])
        y_min, y_max = -eval(opt.range.split(",")[1]), eval(opt.range.split(",")[1])
        new_cav_range = [
            x_min,
            y_min,
            hypes["postprocess"]["anchor_args"]["cav_lidar_range"][2],
            x_max,
            y_max,
            hypes["postprocess"]["anchor_args"]["cav_lidar_range"][5],
        ]
        hypes = update_dict(
            hypes,
            {
                "cav_lidar_range": new_cav_range,
                "lidar_range": new_cav_range,
                "gt_range": new_cav_range,
            },
        )
        yaml_utils_lib = importlib.import_module("opencood.hypes_yaml.yaml_utils")
        parser_func = getattr(yaml_utils_lib, hypes["yaml_parser"])
        hypes = parser_func(hypes)
    hypes["validate_dir"] = hypes["test_dir"]
    if "box_align" in hypes.keys():
        hypes["box_align"]["val_result"] = hypes["box_align"]["test_result"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(int(args.gpu_id))
    model = train_utils.create_model(hypes)
    resume_epoch, model = train_utils.load_saved_model(str(ckpt_dir), model)
    model = model.to(device).eval()
    dataset = build_dataset(hypes, visualize=True, train=False)
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        collate_fn=dataset.collate_batch_test,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )

    module_map = dict(model.pyramid_backbone.named_modules())
    module_names = sorted(module_map)
    if bool(getattr(args, "collect_reference_ranges", False)) and bool(
        getattr(args, "execute_reference_range_capture", False)
    ):
        targets_payload = build_reference_range_targets_payload(
            label=str(args.label),
            route_dir=route_dir,
            raw_dir=raw_dir,
            op_records=op_records,
            stop_output_names=list(getattr(args, "reference_range_stop_output", []) or []),
        )
        write_json(raw_dir / str(getattr(args, "reference_range_out")), targets_payload)
        module_inventory = build_pytorch_module_inventory_payload(
            label=str(args.label),
            raw_dir=raw_dir,
            route_dir=route_dir,
            module_map=module_map,
            scope="model.pyramid_backbone",
        )
        write_json(raw_dir / str(getattr(args, "module_inventory_out")), module_inventory)
        capture_plan = reference_range_capture_plan(
            targets=list(targets_payload.get("targets", [])),
            module_names=module_names,
        )
        capture_plan.update(
            {
                "status": (
                    "ready_for_capture"
                    if int(capture_plan["summary"]["blocked"]) == 0
                    else "blocked"
                ),
                "failure_reason": (
                    None
                    if int(capture_plan["summary"]["blocked"]) == 0
                    else "capture_plan_has_blocked_targets"
                ),
                "label": str(args.label),
                "route_dir": str(route_dir),
                "raw_artifact": str(raw_dir),
                "target_source": str(raw_dir / str(getattr(args, "reference_range_out"))),
                "module_inventory_source": str(raw_dir / str(getattr(args, "module_inventory_out"))),
                "full_network_claim": False,
                "ap_measured": False,
            }
        )
        write_json(raw_dir / str(getattr(args, "reference_range_plan_out")), capture_plan)
        if int(capture_plan["summary"]["blocked"]) > 0:
            blocker = build_blocker(
                label=str(args.label),
                raw_dir=raw_dir,
                reason="reference_range_capture_plan_has_blocked_targets",
                details={
                    "capture_plan_path": str(raw_dir / str(getattr(args, "reference_range_plan_out"))),
                    "summary": capture_plan["summary"],
                    "blocked": [
                        item
                        for item in capture_plan.get("items", [])
                        if item.get("capture_status") == "blocked"
                    ],
                },
            )
            write_json(raw_dir / "tensor_reference_range_capture_blocker.json", blocker)
            return blocker

        captured_tensors: dict[str, np.ndarray] = {}
        captured_graph_inputs: dict[str, np.ndarray] = {}
        hook_counts: dict[tuple[str, str], int] = {}
        handles = []

        def tensor_to_numpy(tensor: Any) -> np.ndarray:
            return tensor.detach().to(torch.float32).cpu().numpy().copy()

        forward_direct: dict[str, list[dict[str, Any]]] = {}
        forward_call_index: dict[str, list[dict[str, Any]]] = {}
        pre_call_index: dict[str, list[dict[str, Any]]] = {}
        for item in capture_plan.get("items", []):
            if item.get("capture_status") != "hook_ready":
                continue
            module_name = str(item.get("source_module_name") or "")
            if module_name not in module_map:
                continue
            source_capture = str(item.get("source_capture") or "")
            if source_capture == "module_forward_hook":
                forward_direct.setdefault(module_name, []).append(item)
            elif source_capture == "module_forward_hook_call_index":
                forward_call_index.setdefault(module_name, []).append(item)
            elif source_capture == "module_forward_pre_hook_call_index":
                pre_call_index.setdefault(module_name, []).append(item)

        def make_forward_direct_hook(module_name: str, items: list[dict[str, Any]]):
            def _hook(_module: Any, inputs: tuple[Any, ...], output: Any) -> None:
                output_array = tensor_to_numpy(output)
                for item in items:
                    captured_tensors[str(item["output_name"])] = output_array
                    if str(item.get("input_name") or "") == "spatial_features" and inputs:
                        captured_graph_inputs["spatial_features"] = tensor_to_numpy(inputs[0])

            return _hook

        def make_forward_call_index_hook(module_name: str, items: list[dict[str, Any]]):
            def _hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
                key = (module_name, "module_forward_hook_call_index")
                call_index = int(hook_counts.get(key, 0))
                hook_counts[key] = call_index + 1
                output_array = tensor_to_numpy(output)
                for item in items:
                    if int(item.get("source_call_index", -1)) == call_index:
                        captured_tensors[str(item["output_name"])] = output_array

            return _hook

        def make_pre_call_index_hook(module_name: str, items: list[dict[str, Any]]):
            def _hook(_module: Any, inputs: tuple[Any, ...]) -> None:
                key = (module_name, "module_forward_pre_hook_call_index")
                call_index = int(hook_counts.get(key, 0))
                hook_counts[key] = call_index + 1
                if not inputs:
                    return None
                input_array = tensor_to_numpy(inputs[0])
                for item in items:
                    if int(item.get("source_call_index", -1)) == call_index:
                        captured_tensors[str(item["output_name"])] = input_array
                return None

            return _hook

        for module_name, items in forward_direct.items():
            handles.append(module_map[module_name].register_forward_hook(make_forward_direct_hook(module_name, items)))
        for module_name, items in forward_call_index.items():
            handles.append(module_map[module_name].register_forward_hook(make_forward_call_index_hook(module_name, items)))
        for module_name, items in pre_call_index.items():
            handles.append(module_map[module_name].register_forward_pre_hook(make_pre_call_index_hook(module_name, items)))

        try:
            batch_data = None
            for maybe_batch in loader:
                if maybe_batch is not None:
                    batch_data = maybe_batch
                    break
            if batch_data is None:
                raise RuntimeError("no non-empty batch found")
            with torch.no_grad():
                batch_data = train_utils.to_device(batch_data, device)
                model(batch_data["ego"])
        finally:
            for handle in handles:
                handle.remove()

        ranges_payload = build_reference_range_capture_payload(
            label=str(args.label),
            raw_dir=raw_dir,
            route_dir=route_dir,
            plan=capture_plan,
            captured_tensors=captured_tensors,
            sample_count=1,
        )
        write_json(raw_dir / str(getattr(args, "reference_ranges_out")), ranges_payload)
        if ranges_payload["status"] != "ready_for_calibration":
            blocker = build_blocker(
                label=str(args.label),
                raw_dir=raw_dir,
                reason=str(ranges_payload["status"]),
                details={
                    "reference_ranges_path": str(raw_dir / str(getattr(args, "reference_ranges_out"))),
                    "missing": ranges_payload.get("missing", []),
                },
            )
            write_json(raw_dir / "tensor_reference_range_capture_blocker.json", blocker)
            return blocker
        if "spatial_features" not in captured_graph_inputs:
            blocker = build_blocker(
                label=str(args.label),
                raw_dir=raw_dir,
                reason="reference_range_capture_missing_graph_input_quant",
                details={
                    "reference_ranges_path": str(raw_dir / str(getattr(args, "reference_ranges_out"))),
                },
            )
            write_json(raw_dir / "tensor_reference_range_capture_blocker.json", blocker)
            return blocker
        _, graph_input_quant = quantize_activation_uint8(
            captured_graph_inputs["spatial_features"],
            tensor_name="spatial_features",
        )
        write_json(raw_dir / "graph_input_quant_reference_range_capture_v1.json", graph_input_quant)
        calibration = build_tensor_quant_params_from_reference_ranges(
            label=str(args.label),
            reference_ranges=list(ranges_payload.get("reference_ranges", [])),
            graph_input_quant=graph_input_quant,
            route_dir=str(route_dir),
            raw_artifact=str(raw_dir),
        )
        write_json(raw_dir / str(getattr(args, "reference_range_calibration_out")), calibration)
        summary = {
            "schema": "native_int8_reference_range_capture_summary_v1",
            "status": "ready_for_scale_aware_simulator",
            "label": str(args.label),
            "range_count": int(ranges_payload["range_count"]),
            "capture_plan_path": str(raw_dir / str(getattr(args, "reference_range_plan_out"))),
            "reference_ranges_path": str(raw_dir / str(getattr(args, "reference_ranges_out"))),
            "calibration_path": str(raw_dir / str(getattr(args, "reference_range_calibration_out"))),
            "module_inventory_path": str(raw_dir / str(getattr(args, "module_inventory_out"))),
            "full_network_claim": False,
            "ap_measured": False,
        }
        write_json(raw_dir / "tensor_reference_range_capture_summary.json", summary)
        return summary

    hook_outputs: dict[str, Any] = {}
    hook_inputs: dict[str, Any] = {}
    handles = []
    missing_hooks: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    for conv_index, op_record in op_items:
        module_name = resolve_module_name(module_names, str(op_record["op_name"]))
        weight_plan = weights_by_index.get(conv_index)
        item = {
            "conv_index": conv_index,
            "op_record": op_record,
            "module_name": module_name,
            "weight_plan": weight_plan,
        }
        selected.append(item)
        if module_name is None or weight_plan is None:
            missing_hooks.append(item)
            continue

        def capture_hook(name: str):
            def _hook(_module: Any, inputs: tuple[Any, ...], output: Any) -> None:
                hook_inputs[name] = inputs[0].detach().to(torch.float32).cpu().numpy()
                hook_outputs[name] = output.detach().to(torch.float32).cpu().numpy()

            return _hook

        handles.append(module_map[module_name].register_forward_hook(capture_hook(module_name)))
    write_json(
        raw_dir / "op_level_selected_ops.json",
        {
            "items": selected,
            "module_suffix_candidates": {
                str(item["op_record"]["op_name"]): module_suffix_candidates(str(item["op_record"]["op_name"]))
                for item in selected
            },
        },
    )
    if missing_hooks:
        blocker = build_blocker(
            label=str(args.label),
            raw_dir=raw_dir,
            reason="missing_pytorch_module_or_weight_plan_for_selected_spatial_conv",
            details={
                "missing": missing_hooks,
                "module_inventory_sample": module_names[:200],
            },
        )
        write_json(raw_dir / "native_int8_s0_024_op_level_numeric_alignment_blocker.json", blocker)
        return blocker

    try:
        batch_data = None
        for maybe_batch in loader:
            if maybe_batch is not None:
                batch_data = maybe_batch
                break
        if batch_data is None:
            raise RuntimeError("no non-empty batch found")
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)
            output_dict: OrderedDict[str, Any] = OrderedDict()
            output_dict["ego"] = model(batch_data["ego"])
    finally:
        for handle in handles:
            handle.remove()

    records: list[dict[str, Any]] = []
    weight_records: list[dict[str, Any]] = []
    activation_records: list[dict[str, Any]] = []
    for item in selected:
        module_name = str(item["module_name"])
        op_record = item["op_record"]
        weight_plan = item["weight_plan"]
        reference = hook_outputs[module_name]
        spatial_activation = hook_inputs[module_name]
        activation_uint8, activation_quant = quantize_activation_uint8(
            spatial_activation,
            tensor_name=f"{module_name}.input_spatial_features",
        )
        activation_records.append(
            {
                "module_name": module_name,
                "op_name": op_record["op_name"],
                "input": summarize_numpy_array(spatial_activation, tensor_name="spatial_features"),
                "quantization": activation_quant,
            }
        )
        weight_key = str(weight_plan["arg_name"])
        weight_int8 = runtime_weights[weight_key]
        module_weight = module_map[module_name].weight.detach().to(torch.float32).cpu().numpy()
        weight_scale = float((weight_plan.get("quantization") or {}).get("scale") or 1.0)
        weight_dequant = weight_int8.astype(np.float32) * weight_scale
        aligned_weight, weight_align_details = minmax_align_to_reference(module_weight, weight_dequant)
        weight_records.append(
            build_op_level_alignment_record(
                tensor_name=f"{op_record['op_name']}:weight",
                reference=module_weight,
                candidate=aligned_weight,
                sample_index=0,
                candidate_kind="runtime_int8_weight_dequant_minmax_aligned",
                candidate_details={
                    "op_name": op_record["op_name"],
                    "module_name": module_name,
                    "conv_index": int(item["conv_index"]),
                    "weight_key": weight_key,
                    "weight_initializer_name": weight_plan.get("initializer_name"),
                    "weight_quantization": weight_plan.get("quantization"),
                    **weight_align_details,
                },
                pass_rel_rmse=0.10,
                pass_corrcoef=0.95,
            )
        )
        for candidate_kind, zero_point in (
            ("native_int8_formula_raw_u8_minmax_aligned", None),
            ("native_int8_formula_centered_input_minmax_aligned", int(activation_quant["zero_point"])),
        ):
            requant = simulate_native_int8_conv_requant(
                activation_uint8,
                weight_int8,
                op_record,
                centered_input_zero_point=zero_point,
            )
            aligned, align_details = minmax_align_to_reference(reference, requant)
            records.append(
                build_op_level_alignment_record(
                    tensor_name=str(op_record["output_name"]),
                    reference=reference,
                    candidate=aligned,
                    sample_index=0,
                    candidate_kind=candidate_kind,
                    candidate_details={
                        "op_name": op_record["op_name"],
                        "module_name": module_name,
                        "conv_index": int(item["conv_index"]),
                        "weight_key": weight_key,
                        "weight_initializer_name": weight_plan.get("initializer_name"),
                        "weight_quantization": weight_plan.get("quantization"),
                        "activation_quantization": activation_quant,
                        "requant_formula": "clip(floor(conv_int32 / 256) + 128, 0, 255)",
                        "centered_input_zero_point": zero_point,
                        **align_details,
                    },
                )
            )
    passed_records = [bool(record["passed"]) for record in records]
    passed_weight_records = [bool(record["passed"]) for record in weight_records]
    summary = {
        "schema": "native_int8_op_level_numeric_alignment_summary_v1",
        "status": "passed"
        if passed_records and all(passed_records) and passed_weight_records and all(passed_weight_records)
        else "blocked",
        "label": str(args.label),
        "processed_samples": 1,
        "selected_op_count": len(selected),
        "record_count": len(records),
        "records_passed": int(sum(1 for item in passed_records if item)),
        "records_failed": int(sum(1 for item in passed_records if not item)),
        "weight_record_count": len(weight_records),
        "weight_records_passed": int(sum(1 for item in passed_weight_records if item)),
        "weight_records_failed": int(sum(1 for item in passed_weight_records if not item)),
        "raw_artifact": str(raw_dir),
        "route_dir": str(route_dir),
        "resume_epoch": int(resume_epoch),
        "ckpt_path": str(best_checkpoint(ckpt_dir)),
        "model_dtype_counts": dtype_counts(model),
        "full_network_claim": False,
        "ap_measured": False,
        "hypothesis_tested": [
            "raw uint8 activation semantics with TVM fixed requant_u8",
            "zero-point centered activation semantics with the same fixed requant_u8",
        ],
    }
    write_json(raw_dir / "spatial_activation_alignment_inputs.json", {"items": activation_records})
    write_json(raw_dir / "op_level_alignment_records.json", {"items": records})
    write_json(raw_dir / "op_level_weight_alignment_records.json", {"items": weight_records})
    if bool(getattr(args, "trace_native_prefix", False)):
        trace_source = None
        for item in selected:
            if str(item["op_record"].get("input_name")) == "spatial_features" and item["module_name"] in hook_inputs:
                trace_source = item
                break
        if trace_source is None:
            trace_summary = {
                "schema": "native_int8_prefix_trace_summary_v1",
                "status": "blocked",
                "failure_reason": "trace_native_prefix_requires_selected_spatial_features_conv",
                "selected_op_count": len(selected),
                "raw_artifact": str(raw_dir),
                "full_network_claim": False,
                "ap_measured": False,
            }
            write_json(raw_dir / "native_prefix_trace_summary.json", trace_summary)
        else:
            source_module_name = str(trace_source["module_name"])
            trace_activation_uint8, trace_activation_quant = quantize_activation_uint8(
                hook_inputs[source_module_name],
                tensor_name="spatial_features",
            )
            _tensor_map, trace_records = simulate_native_int8_graph_prefix(
                activation_uint8=trace_activation_uint8,
                op_records=op_records,
                runtime_weights=runtime_weights,
                weights_by_index=weights_by_index,
                max_ops=int(getattr(args, "trace_max_ops", 0) or 0),
                stop_output_names=list(getattr(args, "trace_stop_output", []) or []),
            )
            first_heavily_saturated = next(
                (
                    record
                    for record in trace_records
                    if float(record["tensor"].get("max_fraction") or 0.0) >= 0.5
                ),
                None,
            )
            trace_summary = {
                "schema": "native_int8_prefix_trace_summary_v1",
                "status": "recorded",
                "record_count": len(trace_records),
                "first_heavily_saturated_output": first_heavily_saturated,
                "activation_quantization": trace_activation_quant,
                "trace_records_path": str(raw_dir / "native_prefix_trace_records.json"),
                "raw_artifact": str(raw_dir),
                "full_network_claim": False,
                "ap_measured": False,
            }
            write_json(raw_dir / "native_prefix_trace_records.json", {"items": trace_records})
            write_json(raw_dir / "native_prefix_trace_summary.json", trace_summary)
    write_json(raw_dir / "op_level_numeric_alignment_summary.json", summary)
    if summary["status"] != "passed":
        blocker = build_blocker(
            label=str(args.label),
            raw_dir=raw_dir,
            reason="first_spatial_conv_or_weight_alignment_failed",
            details={
                "summary_path": str(raw_dir / "op_level_numeric_alignment_summary.json"),
                "records_path": str(raw_dir / "op_level_alignment_records.json"),
                "weight_records_path": str(raw_dir / "op_level_weight_alignment_records.json"),
                "records_failed": summary["records_failed"],
                "weight_records_failed": summary["weight_records_failed"],
            },
        )
        write_json(raw_dir / "native_int8_s0_024_op_level_numeric_alignment_blocker.json", blocker)
    return summary


def main() -> int:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    try:
        report = run_probe(args)
        print(json.dumps(report, indent=2, sort_keys=True))
        success_schemas = {
            "native_int8_op_level_numeric_alignment_summary_v1",
            "native_int8_reference_range_targets_v1",
            "native_int8_reference_range_capture_summary_v1",
        }
        return 0 if report.get("schema") in success_schemas else 1
    except Exception as exc:
        raw_dir.mkdir(parents=True, exist_ok=True)
        blocker = build_blocker(
            label=str(args.label),
            raw_dir=raw_dir,
            reason=f"{type(exc).__name__}:{exc}",
            details={"traceback": traceback.format_exc()},
        )
        write_json(raw_dir / "native_int8_s0_024_op_level_numeric_alignment_blocker.json", blocker)
        print(json.dumps(blocker, indent=2, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
