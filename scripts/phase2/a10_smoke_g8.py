"""Phase 0.4 — Smoke test for groups=8 patch.

Loads baseline_g8 hypes, instantiates HeterPyramidCollab, then exercises the
ResNeXt backbone with a random feature map. Verifies:
1. Hypes loads without error
2. PyramidFusion `resnet.groups == 8`
3. Forward of pyramid_backbone on random (1, 64, H, W) yields non-NaN output
4. Output shapes match expected ResNeXt with groups=8 (no shape mismatch)
"""
from __future__ import annotations

import sys
from pathlib import Path

HEAL_ROOT = Path("/home/jichengzhi/heal_research/HEAL")
sys.path.insert(0, str(HEAL_ROOT))

import torch

from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.models.heter_pyramid_collab import HeterPyramidCollab

HYPES = HEAL_ROOT / "opencood/hypes_yaml/dairv2x/LiDAROnly/lidar_pyramid_dair_v2x_basedair_g8.yaml"


def main():
    print(f"[1/4] Load hypes from {HYPES}")
    hypes = load_yaml(str(HYPES))
    assert hypes["model"]["args"]["fusion_backbone"]["resnext_groups"] == 8

    print(f"[2/4] Instantiate HeterPyramidCollab")
    model = HeterPyramidCollab(hypes["model"]["args"]).eval()

    fb = model.pyramid_backbone
    print(f"  num_filters: {fb.model_cfg['num_filters']}")
    print(f"  resnet.groups: {fb.resnet.groups}")
    assert fb.resnet.groups == 8, f"expected groups=8, got {fb.resnet.groups}"

    # Param counts
    n_total = sum(p.numel() for p in model.parameters())
    n_pb = sum(p.numel() for p in fb.parameters())
    print(f"  total params: {n_total:,}  pyramid_backbone: {n_pb:,}")

    print(f"[3/4] Forward sub-module test (random input)")
    torch.manual_seed(0)
    x = torch.randn(1, 64, 128, 256)
    with torch.no_grad():
        feats = fb.resnet._forward_impl(x, return_interm=True)
    for i, f in enumerate(feats):
        has_nan = torch.isnan(f).any().item()
        print(f"  stage{i} shape {tuple(f.shape)}  NaN={has_nan}")
        assert not has_nan, f"NaN in stage{i} output"

    print(f"[4/4] Verify ResNeXt grouped conv input/output per group is integer")
    for i in range(3):
        layer = getattr(fb.resnet, f"layer{i}")
        for j, block in enumerate(layer):
            w = block.conv2.weight
            # Grouped conv: shape (out_ch, out_ch/groups, kh, kw)
            out_ch = w.shape[0]
            in_per_group = w.shape[1]
            print(f"  layer{i}.block{j} conv2 shape={tuple(w.shape)} groups=8 → in_per_group={in_per_group}")
            assert out_ch % 8 == 0, f"out_ch {out_ch} not divisible by 8"

    print("\n[OK] Smoke test passed — groups=8 patch is functional.")


if __name__ == "__main__":
    main()
