#!/usr/bin/env python3
"""Export CoDriving backbone-only ONNX files for v2 gold cold-start widths."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


for import_root in (
    "/data/jichengzhi_v2x/t2lib",
    "/exdata/jichengzhi/V2Xverse_pyramid",
    "/home/jichengzhi/V2Xverse",
):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

DEFAULT_OUT_ROOT = Path("/exdata/jichengzhi/codriving_onnx/qxs8_backboneonly_20260708")
DEFAULT_WIDTHS = ("24x64x128", "40x64x128", "48x64x128", "64x64x128")
BATCH = 2
IN_CH = 64
H = 256
W = 512


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_width(value: str) -> tuple[int, int, int]:
    parts = [int(item) for item in value.replace(",", "x").split("x") if item]
    if len(parts) != 3:
        raise ValueError(f"width must have 3 parts: {value}")
    return parts[0], parts[1], parts[2]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_export_kwargs(dynamic_batch: bool = False) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "input_names": ["spatial_features"],
        "output_names": ["backbone_output"],
        "opset_version": 17,
        "do_constant_folding": True,
    }
    if dynamic_batch:
        kwargs["dynamic_axes"] = {"spatial_features": {0: "batch"}}
    return kwargs


def export_one(width: str, out_root: Path, force: bool, dynamic_batch: bool = False) -> dict[str, Any]:
    import torch
    import onnx
    from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone

    s0, s1, s2 = parse_width(width)
    out_dir = out_root / width
    out_path = out_dir / "backbone_only.onnx"
    report_path = out_dir / "export_report.json"
    if out_path.is_file() and not force:
        return {"width": width, "status": "skipped_existing", "onnx": str(out_path)}

    class BackboneWrapper(torch.nn.Module):
        def __init__(self, backbone: torch.nn.Module) -> None:
            super().__init__()
            self.backbone = backbone

        def forward(self, spatial_features: torch.Tensor) -> Any:
            out = self.backbone({"spatial_features": spatial_features})
            if isinstance(out, dict):
                return out.get("spatial_features_2d", next(iter(out.values())))
            return out

    torch.manual_seed(20260708)
    cfg = {
        "layer_nums": [3, 4, 5],
        "layer_strides": [2, 2, 2],
        "num_filters": [s0, s1, s2],
        "upsample_strides": [1, 2, 4],
        "num_upsample_filter": [128, 128, 128],
        "inplanes": IN_CH,
    }
    model = BackboneWrapper(ResNetBEVBackbone(cfg, input_channels=IN_CH)).float().eval()
    dummy = torch.zeros(BATCH, IN_CH, H, W, dtype=torch.float32)
    out_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            out_path,
            **build_export_kwargs(dynamic_batch=dynamic_batch),
        )
    loaded = onnx.load(out_path)
    report = {
        "schema": "v2_gold_coldstart_96_codriving_backbone_onnx_export_v1",
        "created_at_utc": utc_now(),
        "width": width,
        "num_filters": [s0, s1, s2],
        "onnx": str(out_path),
        "input_shape": [BATCH, IN_CH, H, W],
        "dynamic_batch": dynamic_batch,
        "n_nodes": len(loaded.graph.node),
        "status": "success",
    }
    write_json(report_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--widths", nargs="*", default=list(DEFAULT_WIDTHS))
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dynamic-batch", action="store_true")
    args = parser.parse_args()

    results = [export_one(width, args.out_root, args.force, dynamic_batch=args.dynamic_batch) for width in args.widths]
    print(json.dumps({"created_at_utc": utc_now(), "results": results}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
