#!/usr/bin/env python3
"""Export original60 Pyramid backbone ONNX artifacts from candidate_queue.jsonl."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
HEAL_ROOT = Path("/home/jichengzhi/heal_research/HEAL")
if str(HEAL_ROOT) not in sys.path:
    sys.path.insert(0, str(HEAL_ROOT))

from opencood.models.fuse_modules.pyramid_fuse import PyramidFusion  # noqa: E402


SCHEMA = "stage2_original60_onnx_export_row_v1"
INPUT_SHAPE = (2, 64, 128, 256)
OPSET = 17


class BackboneOnly(nn.Module):
    def __init__(self, pyramid_fusion: PyramidFusion):
        super().__init__()
        self.resnet = pyramid_fusion.resnet

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feats = self.resnet(x)
        return feats[0], feats[1], feats[2]


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    item = Path(path)
    if not item.exists():
        return []
    return [
        json.loads(line)
        for line in item.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: str | Path, row: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(row, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _safe(value: object) -> str:
    return str(value).replace("/", "-").replace(":", "-").replace(" ", "_")


def _label(candidate: dict[str, Any]) -> str:
    label = str(candidate.get("label") or "")
    if label:
        return _safe(label)
    candidate_id = str(candidate.get("candidate_id") or "unknown")
    return _safe(candidate_id.split(":")[-1])


def _width(candidate: dict[str, Any]) -> list[int]:
    width = candidate.get("width")
    if not isinstance(width, list) or len(width) != 3:
        raise ValueError(f"candidate has invalid width: {candidate.get('candidate_id')}")
    return [int(item) for item in width]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_backbone(num_filters: list[int]) -> BackboneOnly:
    cfg = {
        "anchor_number": 2,
        "layer_nums": [3, 5, 8],
        "layer_strides": [1, 2, 2],
        "num_filters": num_filters,
        "num_upsample_filter": [128, 128, 128],
        "resnext": True,
        "resnext_groups": 32,
        "width_per_group": 4,
        "upsample_strides": [1, 2, 4],
        "align_corners": False,
    }
    return BackboneOnly(PyramidFusion(cfg, input_channels=64))


def validate_grouped_conv_shapes(model: BackboneOnly, width: list[int]) -> None:
    wpg = 4
    groups = 32
    for stage, layer_name in enumerate(("layer0", "layer1", "layer2")):
        block = getattr(model.resnet, layer_name)[0]
        plane = width[stage]
        grouped_width = int(plane * wpg / 64) * groups
        if grouped_width <= 0:
            raise ValueError(f"{layer_name} grouped conv width is zero for plane={plane}")
        expected = [grouped_width, grouped_width // groups, 3, 3]
        actual = list(block.conv2.weight.shape)
        if actual != expected:
            raise ValueError(f"{layer_name}.0.conv2 shape {actual} != expected {expected}")


def export_one(candidate: dict[str, Any], *, out_dir: Path, force: bool) -> dict[str, Any]:
    import onnx

    label = _label(candidate)
    width = _width(candidate)
    out_path = out_dir / f"{label}_backbone.onnx"
    log_path = out_dir / f"{label}_backbone.export.json"
    row: dict[str, Any] = {
        "schema": SCHEMA,
        "candidate_id": candidate.get("candidate_id"),
        "label": label,
        "width": width,
        "onnx_path": str(out_path),
        "export_log_path": str(log_path),
        "status": "started",
    }
    try:
        if out_path.exists() and out_path.stat().st_size > 0 and not force:
            onnx_model = onnx.load(str(out_path))
            onnx.checker.check_model(onnx_model)
            row.update(
                {
                    "status": "cached",
                    "size_bytes": out_path.stat().st_size,
                    "sha256": sha256_file(out_path),
                    "opset": OPSET,
                }
            )
            write_json(log_path, row)
            return row

        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(42)
        model = build_backbone(width).eval()
        validate_grouped_conv_shapes(model, width)
        dummy = torch.randn(*INPUT_SHAPE)
        with torch.no_grad():
            outputs = model(dummy)
        expected_shapes = [
            [INPUT_SHAPE[0], width[0], 128, 256],
            [INPUT_SHAPE[0], width[1], 64, 128],
            [INPUT_SHAPE[0], width[2], 32, 64],
        ]
        actual_shapes = [list(item.shape) for item in outputs]
        if actual_shapes != expected_shapes:
            raise ValueError(f"output shapes {actual_shapes} != expected {expected_shapes}")
        with torch.no_grad():
            torch.onnx.export(
                model,
                (dummy,),
                str(out_path),
                opset_version=OPSET,
                input_names=["spatial_features"],
                output_names=[
                    "/resnet/layer0/layer0.2/relu_2/Relu_output_0",
                    "/resnet/layer1/layer1.4/relu_2/Relu_output_0",
                    "/resnet/layer2/layer2.7/relu_2/Relu_output_0",
                ],
                dynamic_axes=None,
                do_constant_folding=True,
                verbose=False,
            )
        onnx_model = onnx.load(str(out_path))
        onnx.checker.check_model(onnx_model)
        row.update(
            {
                "status": "succeeded",
                "size_bytes": out_path.stat().st_size,
                "sha256": sha256_file(out_path),
                "opset": OPSET,
                "input_shape": list(INPUT_SHAPE),
                "output_shapes": expected_shapes,
                "parameter_count": sum(param.numel() for param in model.parameters()),
            }
        )
    except Exception as exc:
        row.update(
            {
                "status": "failed",
                "failure_reason": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )
    write_json(log_path, row)
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-queue", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--manifest-out", required=True)
    parser.add_argument("--labels", default="")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    labels = {item.strip() for item in args.labels.split(",") if item.strip()}
    candidates = read_jsonl(args.candidate_queue)
    if labels:
        candidates = [row for row in candidates if _label(row) in labels]
    Path(args.manifest_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manifest_out).write_text("", encoding="utf-8")
    counts: dict[str, int] = {}
    for candidate in candidates:
        row = export_one(candidate, out_dir=out_dir, force=args.force)
        counts[row["status"]] = counts.get(row["status"], 0) + 1
        append_jsonl(args.manifest_out, row)
        print(json.dumps(row, ensure_ascii=False, sort_keys=True), flush=True)
    failed = counts.get("failed", 0)
    print(
        json.dumps(
            {
                "schema": "stage2_original60_onnx_export_summary_v1",
                "candidate_count": len(candidates),
                "counts": counts,
                "out_dir": str(out_dir),
                "manifest_out": args.manifest_out,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
