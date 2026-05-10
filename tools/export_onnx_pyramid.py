"""Phase A.1 — PyramidFusion sub-module ONNX export.

Exports the post-encoder model body for HEAL Pyramid_m1_base:

    spatial_features (1, 64, 256, 256)            <-- input
        |
        pyramid_backbone.get_multiscale_feature   (ResNeXt 3-stage)
        pyramid_backbone.decode_multiscale_feature (3 deblocks + concat)
        |
    fused_feature (1, 384, 256, 256)
        |
        shrink_conv  (3x3, 384 -> 256)
        |
        cls_head / reg_head / dir_head  (1x1 convs)
        |
    cls_preds (1, 2,  256, 256)
    reg_preds (1, 14, 256, 256)
    dir_preds (1, 4,  256, 256)

This is the largest contiguous compute chain in PyramidFusion (>95% of the
non-voxelize FLOPs) and is fully ONNX/TRT compatible (pure conv + BN + ReLU +
ConvT + concat) — no DCN, no sparse op, no dynamic shape.

The encoder_m1 (PointPillar VFE+scatter) and backbone_m1 (single ResNet stage)
are kept in PyTorch for now: VFE is sparse/dynamic and not the bottleneck;
backbone_m1 is small (3 BasicBlocks).

Output:
    models/pyramid_m1_subnet_fp32.onnx

Usage:
    python tools/export_onnx_pyramid.py \
        --ckpt /home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_m1_base_2023_08_14_04_28_12/net_epoch_bestval_at23.pth \
        --hypes /home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_m1_base_2023_08_14_04_28_12/config.yaml \
        --out models/pyramid_m1_subnet_fp32.onnx
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
HEAL_ROOT = Path("/home/jichengzhi/heal_research/HEAL")
if str(HEAL_ROOT) not in sys.path:
    sys.path.insert(0, str(HEAL_ROOT))

# HEAL uses yaml_utils.load_yaml (handles its custom tags)
from opencood.hypes_yaml.yaml_utils import load_yaml  # noqa: E402
from opencood.models.heter_pyramid_collab import HeterPyramidCollab  # noqa: E402


class PyramidSubnet(nn.Module):
    """Sub-module deployment wrapper: pyramid_backbone (single) + shrink + heads.

    Input
        spatial_features : (1, 64, 256, 256) float32
            Output of backbone_m1 (after PointPillar + ResNet stride-2 stage).

    Outputs
        cls_preds : (1, 2,  256, 256)
        reg_preds : (1, 14, 256, 256)
        dir_preds : (1, 4,  256, 256)
    """

    def __init__(self, full_model: HeterPyramidCollab):
        super().__init__()
        self.pyramid_backbone = full_model.pyramid_backbone
        self.shrink_conv = full_model.shrink_conv
        self.cls_head = full_model.cls_head
        self.reg_head = full_model.reg_head
        self.dir_head = full_model.dir_head

    def forward(self, spatial_features: torch.Tensor):
        # 1. Multi-scale features via ResNeXt 3-stage
        feats = self.pyramid_backbone.get_multiscale_feature(spatial_features)
        # 2. Deblocks + concat -> (1, 384, 256, 256)
        fused = self.pyramid_backbone.decode_multiscale_feature(feats)
        # 3. Shrink conv 384 -> 256
        out = self.shrink_conv(fused)
        # 4. Heads
        return self.cls_head(out), self.reg_head(out), self.dir_head(out)


def build_pyramid_from_ckpt(hypes_path: str, ckpt_path: str, device: str = "cuda") -> HeterPyramidCollab:
    hypes = load_yaml(hypes_path)
    model_args = hypes["model"]["args"]
    model = HeterPyramidCollab(model_args)
    raw = torch.load(ckpt_path, map_location="cpu")
    sd = raw.get("model_state_dict", raw)
    if "state_dict" in sd:
        sd = sd["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[ckpt] missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print(f"  missing[:5]: {missing[:5]}")
    if unexpected:
        print(f"  unexpected[:5]: {unexpected[:5]}")
    return model.to(device).eval()


def export(args):
    print(f"[1/4] Build model from ckpt: {args.ckpt}")
    full = build_pyramid_from_ckpt(args.hypes, args.ckpt, device="cuda")
    subnet = PyramidSubnet(full).cuda().eval()
    n_params = sum(p.numel() for p in subnet.parameters())
    print(f"  Subnet params: {n_params:,} ({n_params * 4 / 1e6:.2f} MB FP32)")

    print(f"[2/4] PyTorch forward sanity check")
    torch.manual_seed(42)
    shape = tuple(int(x) for x in args.input_shape.split(","))
    dummy = torch.randn(*shape, device="cuda")
    print(f"  input shape: {shape}")
    with torch.no_grad():
        cls_pt, reg_pt, dir_pt = subnet(dummy)
    print(f"  cls={tuple(cls_pt.shape)} reg={tuple(reg_pt.shape)} dir={tuple(dir_pt.shape)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[3/4] Export ONNX -> {out_path} (opset={args.opset})")
    with torch.no_grad():
        torch.onnx.export(
            subnet,
            (dummy,),
            str(out_path),
            opset_version=args.opset,
            input_names=["spatial_features"],
            output_names=["cls_preds", "reg_preds", "dir_preds"],
            dynamic_axes=None,
            do_constant_folding=True,
            verbose=False,
        )
    print(f"  ONNX size: {out_path.stat().st_size / 1e6:.2f} MB")

    print(f"[4/4] ONNX runtime sanity check (CPU provider)")
    import onnx
    import onnxruntime as ort

    m = onnx.load(str(out_path))
    onnx.checker.check_model(m)
    op_types = sorted({n.op_type for n in m.graph.node})
    print(f"  Op types ({len(op_types)}): {op_types}")

    # CPU ORT vs CUDA PyTorch: expect FP32 precision drift O(1e-3) absolute,
    # check both absolute and relative (vs PyTorch output magnitude).
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    np_in = dummy.cpu().numpy()
    cls_o, reg_o, dir_o = sess.run(None, {"spatial_features": np_in})

    rel_diffs = []
    for name, pt, ox in [
        ("cls", cls_pt.cpu().numpy(), cls_o),
        ("reg", reg_pt.cpu().numpy(), reg_o),
        ("dir", dir_pt.cpu().numpy(), dir_o),
    ]:
        d = np.abs(pt - ox)
        rel = d.max() / max(np.abs(pt).max(), 1e-6)
        rel_diffs.append(rel)
        print(f"  {name}: |pt|max={np.abs(pt).max():.3f}  max|Δ|={d.max():.3e}  "
              f"mean|Δ|={d.mean():.3e}  rel_max={rel:.2%}")

    max_rel = max(rel_diffs)
    if max_rel < 0.1 / 100:  # 0.1%
        print(f"  ✅ Sanity PASS (max rel diff {max_rel:.3%} < 0.1%)")
    elif max_rel < 1.0 / 100:  # 1%
        print(f"  ✅ Sanity PASS-MARGINAL (max rel {max_rel:.3%} < 1%; CPU-vs-CUDA FP32 drift)")
    else:
        print(f"  ❌ Sanity FAIL (max rel {max_rel:.3%} > 1%)")
        raise SystemExit(1)
    print(f"\n  ONNX saved: {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    default_ckpt = "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_m1_base_2023_08_14_04_28_12/net_epoch_bestval_at23.pth"
    default_hypes = "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_m1_base_2023_08_14_04_28_12/config.yaml"
    p.add_argument("--ckpt", default=default_ckpt)
    p.add_argument("--hypes", default=default_hypes)
    p.add_argument("--out", default=str(REPO_ROOT / "models/pyramid_m1_subnet_fp32.onnx"))
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--input-shape", default="1,64,256,256",
                   help="ONNX input shape (B,C,H,W). OPV2V: 1,64,256,256. DAIR: 1,64,128,256")
    return p.parse_args()


if __name__ == "__main__":
    export(parse_args())
