#!/usr/bin/env python3
"""Audit whether native INT8 ONNX initializers match a HEAL AP checkpoint."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.stage2_h800_native_int8_op_alignment import (  # noqa: E402
    DEFAULT_HEAL_ROOT,
    DEFAULT_ROUTE_DIR,
    build_blocker,
    resolve_module_name,
)
from scripts.stage2_h800_native_int8_real_activation_bridge import (  # noqa: E402
    _corrcoef,
    summarize_numpy_array,
)
from scripts.stage2_h800_true_fp16_ap_eval import best_checkpoint, dtype_counts, write_json  # noqa: E402


def fuse_conv_bn_weight(
    *,
    conv_weight: np.ndarray,
    bn_weight: np.ndarray,
    bn_running_var: np.ndarray,
    bn_eps: float,
) -> np.ndarray:
    conv = np.asarray(conv_weight, dtype=np.float32)
    gamma = np.asarray(bn_weight, dtype=np.float32)
    running_var = np.asarray(bn_running_var, dtype=np.float32)
    scale = gamma / np.sqrt(running_var + float(bn_eps))
    return conv * scale.reshape((-1,) + (1,) * (conv.ndim - 1))


def batch_norm_name_for_conv_module(module_name: str) -> str | None:
    name = str(module_name)
    for idx in ("1", "2", "3"):
        suffix = f".conv{idx}"
        if name.endswith(suffix):
            return name[: -len(suffix)] + f".bn{idx}"
    if name.endswith(".downsample.0"):
        return name[: -len(".downsample.0")] + ".downsample.1"
    return None


def build_weight_alignment_record(
    *,
    op_name: str,
    initializer_name: str,
    module_name: str,
    initializer_weight: np.ndarray,
    module_weight: np.ndarray,
    pass_rel_rmse: float = 0.10,
    pass_corrcoef: float = 0.95,
) -> dict[str, Any]:
    init = np.asarray(initializer_weight, dtype=np.float32)
    module = np.asarray(module_weight, dtype=np.float32)
    if init.shape != module.shape:
        return {
            "schema": "native_int8_checkpoint_weight_alignment_record_v1",
            "op_name": str(op_name),
            "initializer_name": str(initializer_name),
            "module_name": str(module_name),
            "initializer": summarize_numpy_array(init, tensor_name="onnx_initializer"),
            "module_weight": summarize_numpy_array(module, tensor_name="checkpoint_module_weight"),
            "alignment_error": {
                "shape_mismatch": True,
                "mae": None,
                "rmse": None,
                "max_abs": None,
                "corrcoef": None,
                "reference_range": None,
                "rmse_over_reference_range": None,
            },
            "pass_criteria": {
                "pass_rel_rmse": float(pass_rel_rmse),
                "pass_corrcoef": float(pass_corrcoef),
            },
            "passed": False,
        }
    diff = init.astype(np.float64) - module.astype(np.float64)
    reference_range = float(np.max(module) - np.min(module)) if module.size else 0.0
    rmse = float(np.sqrt(np.mean(diff * diff))) if diff.size else float("inf")
    corrcoef = _corrcoef(init, module)
    passed = bool(reference_range > 0.0 and rmse <= max(1e-6, float(pass_rel_rmse) * reference_range))
    if corrcoef is not None:
        passed = passed and corrcoef >= float(pass_corrcoef)
    return {
        "schema": "native_int8_checkpoint_weight_alignment_record_v1",
        "op_name": str(op_name),
        "initializer_name": str(initializer_name),
        "module_name": str(module_name),
        "initializer": summarize_numpy_array(init, tensor_name="onnx_initializer"),
        "module_weight": summarize_numpy_array(module, tensor_name="checkpoint_module_weight"),
        "alignment_error": {
            "shape_mismatch": False,
            "mae": float(np.mean(np.abs(diff))) if diff.size else float("inf"),
            "rmse": rmse,
            "max_abs": float(np.max(np.abs(diff))) if diff.size else float("inf"),
            "corrcoef": corrcoef,
            "reference_range": reference_range,
            "rmse_over_reference_range": float(rmse / reference_range) if reference_range > 0.0 else None,
        },
        "pass_criteria": {
            "pass_rel_rmse": float(pass_rel_rmse),
            "pass_corrcoef": float(pass_corrcoef),
        },
        "passed": passed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="s0_024")
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--heal-root", default=DEFAULT_HEAL_ROOT)
    parser.add_argument("--route-dir", default=str(DEFAULT_ROUTE_DIR))
    parser.add_argument("--onnx-path", default=None)
    parser.add_argument("--eval-range", default="102.4,102.4")
    parser.add_argument("--max-records", type=int, default=8)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def conv_node_records(onnx_path: Path) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(onnx_path))
    initializers = {
        item.name: numpy_helper.to_array(item).astype(np.float32)
        for item in model.graph.initializer
    }
    records: list[dict[str, Any]] = []
    for index, node in enumerate(model.graph.node):
        if str(node.op_type) != "Conv":
            continue
        if len(node.input) < 2:
            continue
        records.append(
            {
                "conv_index": len(records),
                "node_index": index,
                "op_name": str(node.name or f"Conv_{index}"),
                "input_name": str(node.input[0]),
                "initializer_name": str(node.input[1]),
                "output_name": str(node.output[0]) if node.output else "",
            }
        )
    return records, initializers


def load_heal_model(args: argparse.Namespace) -> Any:
    heal_root = Path(args.heal_root)
    sys.path.insert(0, str(heal_root))
    os.chdir(heal_root)

    import opencood.hypes_yaml.yaml_utils as yaml_utils
    from opencood.tools import train_utils
    from opencood.utils.common_utils import update_dict

    ckpt_dir = Path(args.ckpt_dir)
    opt = argparse.Namespace(
        model_dir=str(ckpt_dir),
        fusion_method="intermediate",
        save_vis_interval=10**9,
        save_npy=False,
        range=args.eval_range,
        no_score=True,
        note="stage2_native_int8_checkpoint_weight_audit",
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
    model = train_utils.create_model(hypes)
    resume_epoch, model = train_utils.load_saved_model(str(ckpt_dir), model)
    return model.eval(), resume_epoch


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    route_dir = Path(args.route_dir)
    inventory = load_json(route_dir / "tvm_operator_inventory.json")
    onnx_path = Path(args.onnx_path or inventory["onnx_path"])
    if not onnx_path.exists():
        raise FileNotFoundError(f"missing ONNX path: {onnx_path}")

    model, resume_epoch = load_heal_model(args)
    module_map = dict(model.pyramid_backbone.named_modules())
    module_names = sorted(module_map)
    convs, initializers = conv_node_records(onnx_path)
    records: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for conv in convs:
        module_name = resolve_module_name(module_names, conv["op_name"])
        initializer_name = str(conv["initializer_name"])
        if module_name is None or initializer_name not in initializers:
            skipped.append(
                {
                    **conv,
                    "module_name": module_name,
                    "initializer_exists": initializer_name in initializers,
                }
            )
            continue
        module = module_map[module_name]
        weight = getattr(module, "weight", None)
        if weight is None:
            skipped.append({**conv, "module_name": module_name, "reason": "module_has_no_weight"})
            continue
        reference_weight = weight.detach().to("cpu").numpy()
        reference_kind = "checkpoint_conv_weight"
        bn_name = batch_norm_name_for_conv_module(module_name)
        if bn_name and bn_name in module_map:
            bn = module_map[bn_name]
            if all(hasattr(bn, attr) for attr in ("weight", "running_var", "eps")):
                reference_weight = fuse_conv_bn_weight(
                    conv_weight=reference_weight,
                    bn_weight=bn.weight.detach().to("cpu").numpy(),
                    bn_running_var=bn.running_var.detach().to("cpu").numpy(),
                    bn_eps=float(bn.eps),
                )
                reference_kind = "checkpoint_conv_bn_fused_weight"
        records.append(
            build_weight_alignment_record(
                op_name=str(conv["op_name"]),
                initializer_name=initializer_name,
                module_name=f"{module_name}:{reference_kind}",
                initializer_weight=initializers[initializer_name],
                module_weight=reference_weight,
            )
        )
        if len(records) >= int(args.max_records):
            break

    pass_flags = [bool(item["passed"]) for item in records]
    summary = {
        "schema": "native_int8_checkpoint_weight_alignment_summary_v1",
        "status": "passed" if pass_flags and all(pass_flags) else "blocked",
        "label": str(args.label),
        "onnx_path": str(onnx_path),
        "route_dir": str(route_dir),
        "ckpt_dir": str(args.ckpt_dir),
        "ckpt_path": str(best_checkpoint(Path(args.ckpt_dir))),
        "resume_epoch": int(resume_epoch),
        "records": len(records),
        "records_passed": int(sum(1 for item in pass_flags if item)),
        "records_failed": int(sum(1 for item in pass_flags if not item)),
        "skipped": len(skipped),
        "max_records": int(args.max_records),
        "model_dtype_counts": dtype_counts(model),
        "raw_artifact": str(raw_dir),
        "full_network_claim": False,
        "ap_measured": False,
    }
    write_json(raw_dir / "checkpoint_weight_alignment_records.json", {"items": records})
    write_json(raw_dir / "checkpoint_weight_alignment_skipped.json", {"items": skipped[:200]})
    write_json(raw_dir / "checkpoint_weight_alignment_summary.json", summary)
    if summary["status"] != "passed":
        blocker = build_blocker(
            label=str(args.label),
            raw_dir=raw_dir,
            reason="onnx_initializers_do_not_match_checkpoint_module_weights",
            details={
                "summary_path": str(raw_dir / "checkpoint_weight_alignment_summary.json"),
                "records_path": str(raw_dir / "checkpoint_weight_alignment_records.json"),
                "records_failed": summary["records_failed"],
            },
        )
        write_json(raw_dir / "native_int8_s0_024_checkpoint_weight_blocker.json", blocker)
    return summary


def main() -> int:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    try:
        report = run_audit(args)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        raw_dir.mkdir(parents=True, exist_ok=True)
        blocker = build_blocker(
            label=str(args.label),
            raw_dir=raw_dir,
            reason=f"{type(exc).__name__}:{exc}",
            details={"traceback": traceback.format_exc()},
        )
        write_json(raw_dir / "native_int8_s0_024_checkpoint_weight_blocker.json", blocker)
        print(json.dumps(blocker, indent=2, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
