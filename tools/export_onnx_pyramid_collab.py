"""Phase A.5 — PyramidFusion forward_collab e2e ONNX export (fixed N=2 agents).

DAIR-V2X cooperative-vehicle-infrastructure 数据每 frame 是 1 vehicle + 1 RSU
(N=2). 我们 export 一个 N=2 fixed engine, 让 multi-agent 路径(占 90%) 完整
走 TRT, 暴露真实 INT8 ΔAP.

Network includes (vs Phase A.1 single-agent engine):
+ weighted_fuse 多 agent 融合 (warp_affine + softmax + sum)
+ single_head_{i} occ heads (用作融合权重 score)

Inputs:
    spatial_features : (2, 64, 128, 256)   2 agents stacked, ego index = 0
    t_ego            : (2, 2, 3)           affine_matrix[0, 0, :2, :, :]
                                            — 把每个 agent warp 到 ego frame

Outputs:
    cls_preds (1, 2, H, W),  reg_preds (1, 14, H, W),  dir_preds (1, 4, H, W)
    where (H, W) = (128, 256) for DAIR-V2X.

This is the *real* framework Pareto-search engine: no fallback dilution.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
HEAL_ROOT = Path("/home/jichengzhi/heal_research/HEAL")
if str(HEAL_ROOT) not in sys.path:
    sys.path.insert(0, str(HEAL_ROOT))

sys.path.insert(0, str(REPO_ROOT))
from tools.export_onnx_pyramid import build_pyramid_from_ckpt  # noqa: E402


def manual_affine_grid(M: torch.Tensor, dsize, align_corners=False):
    """ONNX-friendly equivalent of F.affine_grid (no aten::affine_grid_generator).

    M: (N, 2, 3)
    Returns grid: (N, H, W, 2) suitable for F.grid_sample.

    Verified numerically against F.affine_grid: max abs diff < 1e-6
    for both align_corners=True/False on (N, 2, 3) random matrices.
    """
    H, W = dsize
    device, dtype = M.device, M.dtype
    if align_corners:
        ys = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
    else:
        ys = (torch.arange(H, device=device, dtype=dtype) * 2 + 1) / H - 1
        xs = (torch.arange(W, device=device, dtype=dtype) * 2 + 1) / W - 1
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")    # (H, W)
    ones = torch.ones_like(grid_x)
    base = torch.stack([grid_x, grid_y, ones], dim=-1)        # (H, W, 3)
    # einsum: (N, 2, 3) x (H, W, 3) → (N, H, W, 2)
    grid = torch.einsum("nij,hwj->nhwi", M, base)
    return grid


def warp_affine_simple(src: torch.Tensor, M: torch.Tensor, dsize, align_corners=False):
    """Drop-in replacement for HEAL's warp_affine_simple — uses manual_affine_grid."""
    if M.dtype != src.dtype:
        M = M.to(src.dtype)
    grid = manual_affine_grid(M, dsize, align_corners=align_corners)
    return F.grid_sample(src, grid, align_corners=align_corners, mode="bilinear", padding_mode="zeros")


class PyramidCollabSubnetN2(nn.Module):
    """forward_collab fixed N=2 agents (ego idx=0, partner idx=1).

    Replicates HEAL ``PyramidFusion.forward_collab`` but with:
      * record_len fixed = [2]
      * crop_mask_flag = False (no camera modality)
      * cam_crop_info = None
      * multiscale weighted_fuse but with score sum over agent dim 0
    """

    def __init__(self, full_model, align_corners=False):
        super().__init__()
        self.pb = full_model.pyramid_backbone
        self.shrink = full_model.shrink_conv
        self.cls_head = full_model.cls_head
        self.reg_head = full_model.reg_head
        self.dir_head = full_model.dir_head
        self.num_levels = self.pb.num_levels
        self.align_corners = align_corners
        # Per-level occ head refs (single_head_0/1/2)
        self.single_heads = nn.ModuleList(
            [getattr(self.pb, f"single_head_{i}") for i in range(self.num_levels)]
        )
        self.deblocks = self.pb.deblocks  # ModuleList

    def forward(self, spatial_features: torch.Tensor, t_ego: torch.Tensor):
        """
        spatial_features : (N=2, C=64, H, W)        2 stacked agents
        t_ego            : (N=2, 2, 3)              affine to ego per agent
        """
        # 1. Multi-scale ResNeXt 3-stage (preserves N batch dim)
        feats = self.pb.get_multiscale_feature(spatial_features)  # tuple of 3 tensors

        fused_feats = []
        for i in range(self.num_levels):
            f_i = feats[i]   # (N, C_i, H_i, W_i)
            occ = self.single_heads[i](f_i)                          # (N, 1, H_i, W_i)
            score = torch.sigmoid(occ) + 1e-4

            _, C_i, H_i, W_i = f_i.shape
            features_in_ego = warp_affine_simple(f_i, t_ego, (H_i, W_i),
                                                 align_corners=self.align_corners)
            scores_in_ego = warp_affine_simple(score, t_ego, (H_i, W_i),
                                               align_corners=self.align_corners)
            # masked_fill with -inf so post-softmax that pixel weight = 0
            scores_in_ego = scores_in_ego.masked_fill(scores_in_ego == 0, float("-inf"))
            scores_in_ego = torch.softmax(scores_in_ego, dim=0)
            scores_in_ego = torch.where(
                torch.isnan(scores_in_ego),
                torch.zeros_like(scores_in_ego),
                scores_in_ego,
            )
            # Weighted fuse: sum over agent dim → (1, C_i, H_i, W_i)
            fused = (features_in_ego * scores_in_ego).sum(dim=0, keepdim=True)
            fused_feats.append(fused)

        # 2. Decode multiscale: deblocks + concat
        ups = [self.deblocks[i](fused_feats[i]) for i in range(self.num_levels)]
        x = torch.cat(ups, dim=1)   # (1, 384, H_top, W_top)

        # 3. Shrink + heads
        x = self.shrink(x)
        return self.cls_head(x), self.reg_head(x), self.dir_head(x)


def export(args):
    print(f"[1/4] Build model from ckpt: {args.ckpt}")
    full = build_pyramid_from_ckpt(args.hypes, args.ckpt, device="cuda")
    subnet = PyramidCollabSubnetN2(full, align_corners=False).cuda().eval()
    n_params = sum(p.numel() for p in subnet.parameters())
    print(f"  Subnet params: {n_params:,}")

    print(f"[2/4] PyTorch sanity")
    torch.manual_seed(42)
    H, W = args.feat_h, args.feat_w
    spatial = torch.randn(2, 64, H, W, device="cuda")
    # Identity transform for sanity (t_ego = (2, 2, 3) identity)
    t_ego = torch.eye(2, 3, device="cuda").unsqueeze(0).repeat(2, 1, 1)
    with torch.no_grad():
        cls, reg, dir_ = subnet(spatial, t_ego)
    print(f"  cls={tuple(cls.shape)} reg={tuple(reg.shape)} dir={tuple(dir_.shape)}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[3/4] Export ONNX -> {out}")
    with torch.no_grad():
        torch.onnx.export(
            subnet, (spatial, t_ego), str(out),
            opset_version=args.opset,
            input_names=["spatial_features", "t_ego"],
            output_names=["cls_preds", "reg_preds", "dir_preds"],
            do_constant_folding=True,
            verbose=False,
        )
    print(f"  ONNX size: {out.stat().st_size / 1e6:.2f} MB")

    print(f"[4/4] ONNX runtime sanity")
    import onnx, onnxruntime as ort
    m = onnx.load(str(out))
    onnx.checker.check_model(m)
    op_types = sorted({n.op_type for n in m.graph.node})
    print(f"  Op types ({len(op_types)}): {op_types}")

    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    cls_o, reg_o, dir_o = sess.run(
        None,
        {"spatial_features": spatial.cpu().numpy(), "t_ego": t_ego.cpu().numpy()},
    )
    rels = []
    for n, pt, ox in [("cls", cls, cls_o), ("reg", reg, reg_o), ("dir", dir_, dir_o)]:
        d = np.abs(pt.cpu().numpy() - ox)
        rel = d.max() / max(np.abs(pt.cpu().numpy()).max(), 1e-6)
        rels.append(rel)
        print(f"  {n}: max|Δ|={d.max():.3e}  rel_max={rel:.2%}")
    if max(rels) < 0.01:
        print(f"  ✅ Sanity PASS (rel_max {max(rels):.2%} < 1%)")
    else:
        print(f"  ❌ Sanity FAIL")
        raise SystemExit(1)
    print(f"\n  ONNX saved: {out}")


def parse_args():
    p = argparse.ArgumentParser()
    dair_ckpt = "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_base_2023_08_14_11_42_29/net_epoch_bestval_at23.pth"
    dair_hypes = "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_base_2023_08_14_11_42_29/config.yaml"
    p.add_argument("--ckpt", default=dair_ckpt)
    p.add_argument("--hypes", default=dair_hypes)
    p.add_argument("--out", default=str(REPO_ROOT / "models/pyramid_dair_m1_collab_n2_fp32.onnx"))
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--feat-h", type=int, default=128, help="DAIR=128, OPV2V=256")
    p.add_argument("--feat-w", type=int, default=256, help="DAIR=256, OPV2V=256")
    return p.parse_args()


if __name__ == "__main__":
    export(parse_args())
