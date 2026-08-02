#!/usr/bin/env python3
"""Prepare CoDriving AP jobs for v2 gold cold-start widths.

This script does not fabricate AP.  It creates the artifacts required to
measure AP for a target CoDriving width: a patched config, a prefix-sliced
warm-start checkpoint, and an optional model-load/forward sanity report.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


DEFAULT_REMOTE_REPO = Path("/exdata/jichengzhi/V2Xverse_pyramid")
DEFAULT_LOCAL_REPO = Path("/home/jichengzhi/V2X")
DEFAULT_OUT_ROOT = DEFAULT_REMOTE_REPO / "output/codriving_v2_gold_ap_20260709"
DEFAULT_BASE_CONFIG = DEFAULT_REMOTE_REPO / "opencood/hypes_yaml/dairv2x/lidar_only/dair_centerpoint_codriving.yaml"
DEFAULT_BASE_CKPT = (
    DEFAULT_REMOTE_REPO
    / "opencood/logs/dair_centerpoint_codriving_2026_06_15_21_19_14/net_epoch_bestval_at11.pth"
)
DEFAULT_WIDTHS = (
    "16x32x64",
    "24x32x96",
    "32x32x128",
    "32x64x128",
    "48x96x192",
    "56x112x224",
    "64x96x192",
    "64x128x256",
    "24x64x128",
    "40x64x128",
    "48x64x128",
    "64x64x128",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_width(value: str) -> tuple[int, int, int]:
    parts = [int(item.strip()) for item in value.replace(",", "x").split("x") if item.strip()]
    if len(parts) != 3:
        raise ValueError(f"width must have exactly 3 parts: {value!r}")
    return parts[0], parts[1], parts[2]


def width_str(width: tuple[int, int, int]) -> str:
    return f"{width[0]}x{width[1]}x{width[2]}"


def patch_codriving_config_text(text: str, width: tuple[int, int, int], ckpt_path: str | None = None) -> str:
    """Patch CoDriving BEV backbone filters while preserving the yaml structure."""
    w0, w1, w2 = width
    lines = text.splitlines()
    patched: list[str] = []
    in_backbone = False
    backbone_indent: int | None = None
    replaced_filters = False
    replaced_deblocks = False

    for line in lines:
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        if stripped.startswith("base_bev_backbone:"):
            in_backbone = True
            backbone_indent = indent
            patched.append(line)
            continue
        if in_backbone and backbone_indent is not None and indent <= backbone_indent and stripped:
            in_backbone = False
            backbone_indent = None

        if in_backbone and re.match(r"\s*num_filters:\s*&num_filters\s*\[[^\]]+\]", line):
            prefix = line[: indent]
            patched.append(f"{prefix}num_filters: &num_filters [{w0}, {w1}, {w2}]")
            replaced_filters = True
            continue
        if in_backbone and re.match(r"\s*num_upsample_filter:\s*\[[^\]]+\]", line):
            prefix = line[: indent]
            patched.append(f"{prefix}num_upsample_filter: [{w1}, {w1}, {w1}]")
            replaced_deblocks = True
            continue
        patched.append(line)

    if not replaced_filters:
        raise ValueError("failed to patch base_bev_backbone num_filters")
    if not replaced_deblocks:
        raise ValueError("failed to patch base_bev_backbone num_upsample_filter")

    header = [
        f"# v2_gold_coldstart_96 target width: {width_str(width)}",
        "# generated_by: scripts/stage2_v2_gold_coldstart96_codriving_ap_queue.py",
    ]
    if ckpt_path:
        header.append(f"# warmstart_ckpt: {ckpt_path}")
    return "\n".join(header + [""] + patched) + "\n"


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    raw = torch.load(str(path), map_location="cpu")
    if isinstance(raw, dict) and "model_state_dict" in raw:
        raw = raw["model_state_dict"]
    if not isinstance(raw, dict):
        raise TypeError(f"checkpoint is not a state dict: {path}")
    return {str(k).replace("module.", "", 1): v for k, v in raw.items() if torch.is_tensor(v)}


def slice_tensor_to_shape(source: torch.Tensor, target_shape: tuple[int, ...]) -> torch.Tensor:
    """Return source prefix-sliced to target_shape."""
    if source.ndim != len(target_shape):
        raise ValueError(f"rank mismatch: source={tuple(source.shape)} target={target_shape}")
    if any(dst > src for dst, src in zip(target_shape, source.shape)):
        raise ValueError(f"cannot prefix-slice {tuple(source.shape)} to larger shape {target_shape}")
    slices = tuple(slice(0, dim) for dim in target_shape)
    return source[slices].clone()


def _prepare_import_paths(repo_root: Path) -> None:
    repo = str(repo_root)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    for item in ("/exdata/jichengzhi/tp_lib", "/data/jichengzhi_v2x/t2lib"):
        if item not in sys.path:
            sys.path.append(item)


def _load_hypes(path: Path) -> dict[str, Any]:
    from opencood.hypes_yaml.yaml_utils import load_yaml

    return load_yaml(str(path))


def _build_codriving_model(repo_root: Path, config_path: Path, device: str = "cpu") -> torch.nn.Module:
    _prepare_import_paths(repo_root)
    os.chdir(str(repo_root))
    from opencood.models.center_point_codriving import centerpointcodriving

    hypes = _load_hypes(config_path)
    model = centerpointcodriving(hypes["model"]["args"])
    return model.to(device).eval()


def build_prefix_warmstart_ckpt(
    repo_root: Path,
    config_path: Path,
    base_ckpt: Path,
    out_ckpt: Path,
    device: str = "cpu",
) -> dict[str, Any]:
    target_model = _build_codriving_model(repo_root, config_path, device=device)
    target_sd = target_model.state_dict()
    base_sd = _load_state_dict(base_ckpt)

    warm_sd: dict[str, torch.Tensor] = {}
    copied_exact = 0
    copied_sliced = 0
    kept_init = 0
    missing: list[str] = []
    larger_than_source: list[str] = []

    for key, target_value in target_sd.items():
        source_value = base_sd.get(key)
        if source_value is None:
            warm_sd[key] = target_value.detach().cpu().clone()
            kept_init += 1
            missing.append(key)
            continue
        if tuple(source_value.shape) == tuple(target_value.shape):
            warm_sd[key] = source_value.detach().cpu().clone()
            copied_exact += 1
            continue
        try:
            warm_sd[key] = slice_tensor_to_shape(source_value.detach().cpu(), tuple(target_value.shape))
            copied_sliced += 1
        except ValueError:
            warm_sd[key] = target_value.detach().cpu().clone()
            kept_init += 1
            larger_than_source.append(key)

    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(warm_sd, str(out_ckpt))
    return {
        "base_ckpt": str(base_ckpt),
        "out_ckpt": str(out_ckpt),
        "target_keys": len(target_sd),
        "copied_exact": copied_exact,
        "copied_sliced": copied_sliced,
        "kept_init": kept_init,
        "missing_keys_from_base": missing[:20],
        "larger_than_source_keys": larger_than_source[:20],
    }


def stage_train_dir(config_path: Path, warmstart_ckpt: Path, model_dir: Path) -> dict[str, Any]:
    """Stage a CoDriving train.py --model_dir directory from a warm-start ckpt."""
    model_dir.mkdir(parents=True, exist_ok=True)
    config_dst = model_dir / "config.yaml"
    epoch0_dst = model_dir / "net_epoch_bestval_at0.pth"

    if config_path.resolve() != config_dst.resolve():
        shutil.copy2(config_path, config_dst)

    existing_bestval = sorted(model_dir.glob("net_epoch_bestval_at*.pth"))
    nonzero_bestval = [path for path in existing_bestval if path.name != epoch0_dst.name]
    if nonzero_bestval:
        return {
            "model_dir": str(model_dir),
            "config": str(config_dst),
            "warmstart_epoch0": str(epoch0_dst),
            "status": "existing_training_checkpoint",
            "staged_warmstart": False,
            "existing_bestval": [str(path) for path in nonzero_bestval],
        }

    shutil.copy2(warmstart_ckpt, epoch0_dst)
    return {
        "model_dir": str(model_dir),
        "config": str(config_dst),
        "warmstart_epoch0": str(epoch0_dst),
        "status": "staged",
        "staged_warmstart": True,
        "existing_bestval": [],
    }


def sanity_load_and_forward(repo_root: Path, config_path: Path, ckpt_path: Path, device: str = "cpu") -> dict[str, Any]:
    model = _build_codriving_model(repo_root, config_path, device=device)
    sd = _load_state_dict(ckpt_path)
    missing, unexpected = model.load_state_dict(sd, strict=False)

    from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone

    hypes = _load_hypes(config_path)
    backbone_cfg = hypes["model"]["args"]["base_bev_backbone"]
    backbone = ResNetBEVBackbone(backbone_cfg, input_channels=64).to(device).eval()
    backbone_sd = {key.replace("backbone.", "", 1): value for key, value in sd.items() if key.startswith("backbone.")}
    bb_missing, bb_unexpected = backbone.load_state_dict(backbone_sd, strict=False)
    x = torch.zeros(2, 64, 256, 512, device=device)
    with torch.inference_mode():
        out = backbone({"spatial_features": x})
    y = out.get("spatial_features_2d", next(iter(out.values()))) if isinstance(out, dict) else out
    return {
        "ckpt": str(ckpt_path),
        "missing": len(missing),
        "unexpected": len(unexpected),
        "missing_first": list(missing[:10]),
        "unexpected_first": list(unexpected[:10]),
        "backbone_missing": len(bb_missing),
        "backbone_unexpected": len(bb_unexpected),
        "backbone_output_shape": list(y.shape),
    }


def prepare_one(args: argparse.Namespace) -> dict[str, Any]:
    width = parse_width(args.width)
    label = width_str(width)
    out_dir = args.out_root / label
    ckpt_path = out_dir / "warmstart_prefix_base11.pth"
    config_path = out_dir / "config.yaml"
    report_path = out_dir / "prepare_report.json"

    config_text = patch_codriving_config_text(args.base_config.read_text(encoding="utf-8"), width, str(ckpt_path))
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(config_text, encoding="utf-8")

    warmstart_report = build_prefix_warmstart_ckpt(
        args.repo_root,
        config_path,
        args.base_ckpt,
        ckpt_path,
        device=args.device,
    )
    train_dir_report = stage_train_dir(config_path, ckpt_path, out_dir)
    sanity_report = None
    if not args.skip_sanity:
        sanity_report = sanity_load_and_forward(args.repo_root, config_path, ckpt_path, device=args.device)

    payload = {
        "schema": "v2_gold_coldstart_96_codriving_ap_prepare_v1",
        "created_at_utc": utc_now(),
        "width": label,
        "width_tuple": list(width),
        "config": str(config_path),
        "warmstart_ckpt": str(ckpt_path),
        "warmstart": warmstart_report,
        "train": train_dir_report,
        "sanity": sanity_report,
        "status": "success" if sanity_report is None or sanity_report["missing"] == 0 else "load_has_missing",
        "next_step": (
            "Run opencood/tools/train.py --hypes_yaml config.yaml --model_dir model_dir "
            "--fusion_method intermediate, then evaluate AP."
        ),
    }
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    rows = []
    for width_value in args.widths:
        width = parse_width(width_value)
        label = width_str(width)
        out_dir = args.out_root / label
        rows.append(
            {
                "width": label,
                "width_tuple": list(width),
                "config": str(out_dir / "config.yaml"),
                "warmstart_ckpt": str(out_dir / "warmstart_prefix_base11.pth"),
                "model_dir": str(out_dir),
                "warmstart_epoch0": str(out_dir / "net_epoch_bestval_at0.pth"),
                "prepare_report": str(out_dir / "prepare_report.json"),
                "status": "pending_prepare",
            }
        )
    payload = {
        "schema": "v2_gold_coldstart_96_codriving_ap_manifest_v1",
        "created_at_utc": utc_now(),
        "repo_root": str(args.repo_root),
        "base_config": str(args.base_config),
        "base_ckpt": str(args.base_ckpt),
        "out_root": str(args.out_root),
        "rows": rows,
    }
    args.out_root.mkdir(parents=True, exist_ok=True)
    path = args.out_root / "ap_prepare_manifest.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REMOTE_REPO)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--base-ckpt", type=Path, default=DEFAULT_BASE_CKPT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    sub = parser.add_subparsers(dest="cmd", required=True)

    one = sub.add_parser("prepare-one")
    one.add_argument("--width", required=True)
    one.add_argument("--device", default="cpu")
    one.add_argument("--skip-sanity", action="store_true")

    manifest = sub.add_parser("build-manifest")
    manifest.add_argument("--widths", nargs="*", default=list(DEFAULT_WIDTHS))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.cmd == "prepare-one":
        payload = prepare_one(args)
    elif args.cmd == "build-manifest":
        payload = build_manifest(args)
    else:
        raise ValueError(args.cmd)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
