#!/usr/bin/env python3
"""Materialize scanner-derived F-Cooper widths and export the dense TRT scope."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from framework.stage5.genome_contract_v1 import canonical_group_id  # noqa: E402


WIDTH_SCHEMA = (
    "backbone.s0",
    "backbone.s1",
    "backbone.s2",
    "neck.deblock",
    "neck.output",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_width(text: str) -> tuple[int, int, int, int, int]:
    values = tuple(int(item) for item in text.split(","))
    if len(values) != 5 or any(value <= 0 for value in values):
        raise ValueError("--width must contain five positive comma-separated integers")
    return values  # type: ignore[return-value]


def apply_structure_to_hypes(
    source: Mapping[str, Any], width: Sequence[int]
) -> dict[str, Any]:
    if len(width) != 5:
        raise ValueError("F-Cooper materialization requires five scanner widths")
    s0, s1, s2, deblock, output = (int(value) for value in width)
    hypes = copy.deepcopy(dict(source))
    args = hypes["model"]["args"]
    modality = args["m1"]
    modality["backbone_args"]["num_filters"] = [s0, s1, s2]
    modality["backbone_args"]["num_upsample_filter"] = [deblock, deblock, deblock]
    modality["shrink_header"]["input_dim"] = 3 * deblock
    modality["shrink_header"]["dim"] = [output]
    args["in_head"] = output
    hypes["name"] = (
        f"{hypes['name']}_scanner_{s0}x{s1}x{s2}x{deblock}x{output}"
    )
    return hypes


def project_state_dict(
    source: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    projected: dict[str, torch.Tensor] = {}
    changed = []
    for name, target_tensor in target.items():
        if name not in source:
            raise ValueError(f"target state key is missing from source checkpoint: {name}")
        source_tensor = source[name].detach().cpu()
        if source_tensor.ndim != target_tensor.ndim:
            raise ValueError(f"state rank mismatch for {name}")
        if any(dst > src for dst, src in zip(target_tensor.shape, source_tensor.shape)):
            raise ValueError(
                f"target tensor grows source state for {name}: "
                f"{tuple(target_tensor.shape)} > {tuple(source_tensor.shape)}"
            )
        slices = tuple(slice(0, int(size)) for size in target_tensor.shape)
        value = source_tensor[slices].clone() if slices else source_tensor.clone()
        if value.shape != target_tensor.shape:
            raise ValueError(f"state projection shape mismatch for {name}")
        projected[name] = value
        if source_tensor.shape != target_tensor.shape:
            changed.append(
                {
                    "name": name,
                    "source_shape": list(source_tensor.shape),
                    "target_shape": list(target_tensor.shape),
                }
            )
    return projected, {
        "policy": "scanner_guided_prefix_channel_projection_v1",
        "target_tensor_count": len(projected),
        "changed_tensor_count": len(changed),
        "changed_tensors": changed,
    }


class FCooperDenseEncoder(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.backbone = model.backbone_m1
        self.shrinker = model.shrinker_m1

    def forward(self, spatial_features: torch.Tensor) -> torch.Tensor:
        features = self.backbone({"spatial_features": spatial_features})
        return self.shrinker(features["spatial_features_2d"])


def _load_hypes(config: Path) -> dict[str, Any]:
    from opencood.hypes_yaml import yaml_utils

    return yaml_utils.load_yaml(str(config), SimpleNamespace(model_dir=None))


def _create_model(hypes: Mapping[str, Any]) -> nn.Module:
    from opencood.tools import train_utils

    return train_utils.create_model(dict(hypes))


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    from opencood.hypes_yaml import yaml_utils

    width = parse_width(args.width)
    source_hypes = _load_hypes(args.source_config)
    target_hypes = apply_structure_to_hypes(source_hypes, width)
    target_model = _create_model(target_hypes)
    source_state = torch.load(args.source_checkpoint, map_location="cpu")
    projected, projection_audit = project_state_dict(
        source_state, target_model.state_dict()
    )
    target_model.load_state_dict(projected, strict=True)

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "net_epoch_bestval_at23.pth"
    config = output / "config.yaml"
    torch.save(projected, checkpoint)
    yaml_utils.save_yaml(target_hypes, str(config))

    dense = FCooperDenseEncoder(target_model).eval()
    sample = torch.randn(*args.input_shape, dtype=torch.float32)
    with torch.no_grad():
        output_tensor = dense(sample)
    onnx_path = output / f"fcooper_dense_{'x'.join(map(str, width))}.onnx"
    torch.onnx.export(
        dense,
        sample,
        str(onnx_path),
        input_names=["spatial_features"],
        output_names=["encoded_features"],
        opset_version=17,
        do_constant_folding=True,
    )
    import onnx

    graph = onnx.load(str(onnx_path))
    onnx.checker.check_model(graph)
    report = {
        "schema_version": "fcooper_source_materialization_v1",
        "status": "ready",
        "model": "fcooper",
        "group_id": canonical_group_id("fcooper", width, WIDTH_SCHEMA),
        "width": list(width),
        "width_schema": list(WIDTH_SCHEMA),
        "input_shape": list(args.input_shape),
        "output_shape": list(output_tensor.shape),
        "optimized_scope": "post_scatter_backbone_shrinker",
        "projection": projection_audit,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "config_path": str(config),
        "config_sha256": sha256_file(config),
        "onnx_path": str(onnx_path),
        "onnx_sha256": sha256_file(onnx_path),
        "source_checkpoint_sha256": sha256_file(args.source_checkpoint),
        "source_config_sha256": sha256_file(args.source_config),
    }
    report_path = output / "materialization_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (output / "source.ready").touch()
    return {**report, "materialization_report_sha256": sha256_file(report_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--width", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--input-shape", type=int, nargs=4, default=[5, 64, 512, 512]
    )
    return parser.parse_args()


def main() -> None:
    print(json.dumps(materialize(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
