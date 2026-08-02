#!/usr/bin/env python3
"""Export checkpoint-consistent PyramidFusion multiscale backbone ONNX."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.stage2_h800_native_int8_real_activation_bridge import DEFAULT_HEAL_ROOT  # noqa: E402
from scripts.stage2_h800_true_fp16_ap_eval import best_checkpoint, checkpoint_epoch, dtype_counts, write_json  # noqa: E402


def multiscale_output_names(num_levels: int) -> list[str]:
    return [f"pyramid_level{idx}" for idx in range(int(num_levels))]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PyramidMultiscaleBackbone(nn.Module):
    def __init__(self, full_model: Any) -> None:
        super().__init__()
        self.pyramid_backbone = full_model.pyramid_backbone

    def forward(self, spatial_features: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(self.pyramid_backbone.get_multiscale_feature(spatial_features))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="s0_024")
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--heal-root", default=DEFAULT_HEAL_ROOT)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--eval-range", default="102.4,102.4")
    parser.add_argument("--input-shape", default="1,64,256,256")
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def _load_best_checkpoint_state(
    model: Any,
    ckpt_dir: Path,
    checkpoint_path: Path | None = None,
) -> tuple[Any, Path, int]:
    ckpt_path = Path(checkpoint_path) if checkpoint_path is not None else best_checkpoint(ckpt_dir)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint state must be dict: {ckpt_path}")
    model.load_state_dict(state, strict=False)
    return model, ckpt_path, int(checkpoint_epoch(ckpt_path) or -1)


def load_heal_model(args: argparse.Namespace) -> tuple[Any, int, Path]:
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
        note="stage2_export_checkpoint_multiscale_onnx",
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
    explicit_checkpoint = Path(args.checkpoint_path).resolve() if args.checkpoint_path else None
    model, ckpt_path, resume_epoch = _load_best_checkpoint_state(
        model,
        ckpt_dir,
        checkpoint_path=explicit_checkpoint,
    )
    return model.eval(), int(resume_epoch), ckpt_path


def export_checkpoint_multiscale_onnx(args: argparse.Namespace) -> dict[str, Any]:
    import onnx

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(int(args.gpu_id))
    model, resume_epoch, ckpt_path = load_heal_model(args)
    model = model.to(device).eval()
    wrapper = PyramidMultiscaleBackbone(model).to(device).eval()
    shape = tuple(int(item.strip()) for item in str(args.input_shape).split(",") if item.strip())
    if len(shape) != 4:
        raise ValueError(f"input shape must be N,C,H,W: {args.input_shape}")
    dummy = torch.randn(*shape, device=device, dtype=torch.float32)
    with torch.no_grad():
        outputs = tuple(wrapper(dummy))
    output_names = multiscale_output_names(len(outputs))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy,),
            str(out_path),
            opset_version=int(args.opset),
            input_names=["spatial_features"],
            output_names=output_names,
            dynamic_axes=None,
            do_constant_folding=True,
            verbose=False,
        )
    model_proto = onnx.load(str(out_path))
    onnx.checker.check_model(model_proto)
    op_counts: dict[str, int] = {}
    for node in model_proto.graph.node:
        op_counts[str(node.op_type)] = op_counts.get(str(node.op_type), 0) + 1
    report = {
        "schema": "stage2_checkpoint_multiscale_onnx_export_report_v1",
        "status": "success",
        "label": str(args.label),
        "onnx_path": str(out_path),
        "onnx_digest": sha256_file(out_path),
        "ckpt_dir": str(args.ckpt_dir),
        "ckpt_path": str(ckpt_path),
        "resume_epoch": resume_epoch,
        "input_shape": [int(item) for item in shape],
        "output_names": output_names,
        "output_shapes": {
            name: [int(dim) for dim in tensor.shape]
            for name, tensor in zip(output_names, outputs)
        },
        "op_counts": dict(sorted(op_counts.items())),
        "model_dtype_counts": dtype_counts(model),
        "full_network_claim": False,
        "ap_measured": False,
    }
    write_json(Path(args.report_json), report)
    return report


def main() -> int:
    args = parse_args()
    try:
        report = export_checkpoint_multiscale_onnx(args)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        blocker = {
            "schema": "stage2_checkpoint_multiscale_onnx_export_blocker_v1",
            "status": "failed",
            "label": str(args.label),
            "failure_reason": f"{type(exc).__name__}:{exc}",
            "traceback": traceback.format_exc(),
            "onnx_path": str(args.out),
            "report_json": str(args.report_json),
            "full_network_claim": False,
            "ap_measured": False,
        }
        write_json(Path(args.report_json), blocker)
        print(json.dumps(blocker, indent=2, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
