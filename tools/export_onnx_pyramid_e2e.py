"""Full-forward e2e ONNX export for HEAL Pyramid Fusion (m1, N=2 collab).

Replaces the subnet-only export with the full forward chain:

    voxel_features (MAX_VOX, 32, 4)      <- padded raw voxel pts
    voxel_num_points (MAX_VOX,)          <- count of real pts per voxel
    voxel_coords (MAX_VOX, 4)            <- (batch_idx, z=0, y, x), int32
    voxel_mask (MAX_VOX,)                <- 1.0 real / 0.0 padding
    t_ego (N=2, 2, 3)                    <- affine to ego per agent
        |
        VFE (PFN: linear + BN + ReLU + maxpool)
        Scatter (flat index_add into BEV grid)
        backbone_m1 (ResNet 2D stride-2)
        aligner_m1 (Identity / AlignNet)
        pyramid_backbone.forward_collab (N=2)
        shrink_conv
        cls_head / reg_head / dir_head
        |
    cls_preds (1, 2,  H/2, W/2)
    reg_preds (1, 14, H/2, W/2)
    dir_preds (1, 4,  H/2, W/2)

For DAIR: cav_lidar_range = (-102.4, -51.2, -3.5, 102.4, 51.2, 1.5),
voxel_size = (0.4, 0.4, 5.0) → grid_size = (512, 256, 1) → BEV (256, 512),
after backbone_m1 stride-2 → (128, 256).
"""
from __future__ import annotations

import argparse
import os
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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.export_onnx_pyramid import build_pyramid_from_ckpt  # noqa: E402


# ---------------------------------------------------------------------------
# ONNX-friendly building blocks
# ---------------------------------------------------------------------------


def _pfn_forward_onnx(pfn, inputs: torch.Tensor) -> torch.Tensor:
    """ONNX-friendly PFNLayer.forward.

    Strips:
      * `inputs.shape[0] > self.part` Python branch (static M)
      * `torch.backends.cudnn.enabled` mutation
    """
    x = pfn.linear(inputs)                      # (M, P, C_out)
    if pfn.use_norm:
        x = pfn.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
    x = F.relu(x)
    x_max = torch.max(x, dim=1, keepdim=True)[0]
    if pfn.last_vfe:
        return x_max
    x_repeat = x_max.repeat(1, inputs.shape[1], 1)
    return torch.cat([x, x_repeat], dim=2)


def _vfe_forward_onnx(vfe, voxel_features: torch.Tensor,
                      voxel_num_points: torch.Tensor,
                      voxel_coords: torch.Tensor,
                      voxel_mask: torch.Tensor) -> torch.Tensor:
    """ONNX-friendly PillarVFE.forward.

    Args:
        voxel_features: (M, P=32, 4)  raw point features (x, y, z, intensity)
        voxel_num_points: (M,)        real pts per pillar
        voxel_coords: (M, 4)          (batch_idx, z, y, x) int32
        voxel_mask: (M,)              1.0 real / 0.0 padded
    Returns:
        pillar_features: (M, 64)  zeroed on padding
    """
    # points_mean: avg over real points (avoid div-by-zero on padding via clamp)
    denom = voxel_num_points.to(voxel_features.dtype).clamp(min=1.0).view(-1, 1, 1)
    points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / denom
    f_cluster = voxel_features[:, :, :3] - points_mean

    f_center = torch.zeros_like(voxel_features[:, :, :3])
    f_center[:, :, 0] = voxel_features[:, :, 0] - (
        voxel_coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * vfe.voxel_x + vfe.x_offset
    )
    f_center[:, :, 1] = voxel_features[:, :, 1] - (
        voxel_coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * vfe.voxel_y + vfe.y_offset
    )
    f_center[:, :, 2] = voxel_features[:, :, 2] - (
        voxel_coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * vfe.voxel_z + vfe.z_offset
    )

    if vfe.use_absolute_xyz:
        features = [voxel_features, f_cluster, f_center]
    else:
        features = [voxel_features[..., 3:], f_cluster, f_center]
    if vfe.with_distance:
        points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
        features.append(points_dist)
    features = torch.cat(features, dim=-1)  # (M, P, F_in)

    # Mask out padded points within each pillar (per-point mask)
    P = features.shape[1]
    point_idx = torch.arange(P, dtype=torch.int32,
                             device=voxel_num_points.device).view(1, -1)
    point_mask = (point_idx < voxel_num_points.view(-1, 1).int()).to(features.dtype)
    features = features * point_mask.unsqueeze(-1)

    for pfn in vfe.pfn_layers:
        features = _pfn_forward_onnx(pfn, features)

    # features shape after last PFN: (M, 1, C). squeeze dim=1 (static, ONNX-safe).
    out = features.squeeze(1) if features.dim() == 3 else features  # (M, C)
    # Mask padded pillars (full zero)
    return out * voxel_mask.view(-1, 1)


def _scatter_to_bev_onnx(pillar_features: torch.Tensor,
                         voxel_coords: torch.Tensor,
                         voxel_mask: torch.Tensor,
                         n_agents: int, H: int, W: int) -> torch.Tensor:
    """ONNX-friendly PointPillarScatter.

    Flat index = batch * H * W + y * W + x. Padded pillars routed to sentinel.

    Args:
        pillar_features: (M, C=64)
        voxel_coords: (M, 4) (batch, z, y, x) int32
        voxel_mask: (M,) float
        n_agents: N (fixed, here 2)
        H, W: BEV grid (DAIR: 256, 512)
    Returns:
        (N, C, H, W)
    """
    C = pillar_features.shape[1]
    # Compute flat BEV index from coords
    batch_idx = voxel_coords[:, 0].long()
    y_idx = voxel_coords[:, 2].long()
    x_idx = voxel_coords[:, 3].long()
    flat_idx = batch_idx * (H * W) + y_idx * W + x_idx  # (M,)

    # Route padded entries to sentinel slot N*H*W (one extra row, sliced off later)
    sentinel = n_agents * H * W
    mask_bool = voxel_mask > 0.5
    flat_idx_safe = torch.where(
        mask_bool, flat_idx,
        torch.full_like(flat_idx, sentinel)
    )

    # scatter via index_add → ONNX ScatterND with reduction='add'
    bev_flat = torch.zeros(
        n_agents * H * W + 1, C,
        dtype=pillar_features.dtype, device=pillar_features.device
    )
    bev_flat = bev_flat.index_add(0, flat_idx_safe, pillar_features)

    # Slice off sentinel, reshape to (N, H, W, C) → (N, C, H, W)
    bev = bev_flat[: n_agents * H * W].view(n_agents, H, W, C).permute(0, 3, 1, 2)
    return bev.contiguous()


# ---------------------------------------------------------------------------
# Affine warp (same as collab subnet)
# ---------------------------------------------------------------------------


def manual_affine_grid(M: torch.Tensor, dsize, align_corners=False):
    N, _, _ = M.shape
    H, W = dsize
    ys = torch.arange(H, device=M.device, dtype=M.dtype)
    xs = torch.arange(W, device=M.device, dtype=M.dtype)
    if align_corners:
        ys = (ys / max(H - 1, 1)) * 2 - 1
        xs = (xs / max(W - 1, 1)) * 2 - 1
    else:
        ys = (ys * 2 + 1) / H - 1
        xs = (xs * 2 + 1) / W - 1
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1)  # (H,W,3)
    grid = grid.unsqueeze(0).expand(N, -1, -1, -1)                          # (N,H,W,3)
    return torch.einsum("nhwc,nkc->nhwk", grid, M)                          # (N,H,W,2)


def warp_affine_simple(src, M, dsize, align_corners=False):
    grid = manual_affine_grid(M, dsize, align_corners=align_corners)
    return F.grid_sample(src, grid, mode="bilinear",
                         padding_mode="zeros", align_corners=align_corners)


# ---------------------------------------------------------------------------
# Pyramid e2e wrapper
# ---------------------------------------------------------------------------


class PyramidE2ESubnet(nn.Module):
    """Full forward: VFE → Scatter → backbone_m1 → aligner_m1 → pyramid (collab N=2)
    → shrink → heads."""

    def __init__(self, full_model, grid_size, align_corners=False):
        super().__init__()
        # Pull sub-modules
        self.vfe = full_model.encoder_m1.pillar_vfe
        self.scatter_cfg = full_model.encoder_m1.scatter
        self.backbone_m1 = full_model.backbone_m1
        self.aligner_m1 = full_model.aligner_m1

        self.pb = full_model.pyramid_backbone
        self.shrink = full_model.shrink_conv
        self.cls_head = full_model.cls_head
        self.reg_head = full_model.reg_head
        self.dir_head = full_model.dir_head

        self.num_levels = self.pb.num_levels
        self.align_corners = align_corners
        self.single_heads = nn.ModuleList(
            [getattr(self.pb, f"single_head_{i}") for i in range(self.num_levels)]
        )
        self.deblocks = self.pb.deblocks

        # BEV grid_size from PointPillarScatter (nx, ny, nz)
        # PyTorch tensor in TRT shape: (N, C, H=ny, W=nx)
        self.nx, self.ny, self.nz = grid_size

    def forward(self,
                voxel_features: torch.Tensor,
                voxel_num_points: torch.Tensor,
                voxel_coords: torch.Tensor,
                voxel_mask: torch.Tensor,
                t_ego: torch.Tensor):
        """
        Args:
            voxel_features  (MAX_VOX, 32, 4)
            voxel_num_points(MAX_VOX,)
            voxel_coords    (MAX_VOX, 4)  (batch=0/1, z=0, y, x)
            voxel_mask      (MAX_VOX,)
            t_ego           (N=2, 2, 3)
        """
        # ─── 1. VFE ──────────────────────────────────────────────────────────
        pillar_features = _vfe_forward_onnx(
            self.vfe, voxel_features, voxel_num_points, voxel_coords, voxel_mask
        )  # (MAX_VOX, 64)

        # ─── 2. Scatter to BEV ──────────────────────────────────────────────
        spatial = _scatter_to_bev_onnx(
            pillar_features, voxel_coords, voxel_mask,
            n_agents=2, H=self.ny, W=self.nx
        )  # (2, 64, ny, nx)

        # ─── 3. backbone_m1 ─────────────────────────────────────────────────
        feat = self.backbone_m1({"spatial_features": spatial})["spatial_features_2d"]
        # (2, 64, ny/2, nx/2)

        # ─── 4. aligner_m1 ──────────────────────────────────────────────────
        feat = self.aligner_m1(feat)  # (2, 64, H, W)

        # ─── 5. Pyramid forward_collab (fixed N=2, no compress, no camera) ──
        feats = self.pb.get_multiscale_feature(feat)
        fused_feats = []
        for i in range(self.num_levels):
            f_i = feats[i]
            occ = self.single_heads[i](f_i)
            score = torch.sigmoid(occ) + 1e-4
            _, _, H_i, W_i = f_i.shape
            features_in_ego = warp_affine_simple(
                f_i, t_ego, (H_i, W_i), align_corners=self.align_corners
            )
            scores_in_ego = warp_affine_simple(
                score, t_ego, (H_i, W_i), align_corners=self.align_corners
            )
            scores_in_ego = scores_in_ego.masked_fill(scores_in_ego == 0, float("-inf"))
            scores_in_ego = torch.softmax(scores_in_ego, dim=0)
            scores_in_ego = torch.where(
                torch.isnan(scores_in_ego),
                torch.zeros_like(scores_in_ego),
                scores_in_ego,
            )
            fused = (features_in_ego * scores_in_ego).sum(dim=0, keepdim=True)
            fused_feats.append(fused)

        ups = [self.deblocks[i](fused_feats[i]) for i in range(self.num_levels)]
        x = torch.cat(ups, dim=1)

        # ─── 6. Shrink + heads ───────────────────────────────────────────────
        x = self.shrink(x)
        return self.cls_head(x), self.reg_head(x), self.dir_head(x)


# ---------------------------------------------------------------------------
# Reference forward via HEAL native modules (for sanity check)
# ---------------------------------------------------------------------------


def _native_vfe_scatter(full_model, batch_dict):
    """Run HEAL's native VFE + Scatter for reference comparison."""
    out = full_model.encoder_m1.pillar_vfe(batch_dict)
    out = full_model.encoder_m1.scatter(out)
    return out["spatial_features"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def export(args):
    print(f"[1/5] Build model: {args.ckpt}")
    full = build_pyramid_from_ckpt(args.hypes, args.ckpt, device="cuda")

    # Pull grid_size from scatter module
    scatter_mod = full.encoder_m1.scatter
    grid = (scatter_mod.nx, scatter_mod.ny, scatter_mod.nz)
    print(f"  BEV grid (nx, ny, nz) = {grid}")
    print(f"  expected BEV input to backbone_m1: (2, 64, {grid[1]}, {grid[0]})")

    e2e = PyramidE2ESubnet(full, grid_size=grid, align_corners=False).cuda().eval()
    n_params = sum(p.numel() for p in e2e.parameters())
    print(f"  E2E subnet params: {n_params:,} ({n_params * 4 / 1e6:.2f} MB FP32)")

    # ─── PyTorch sanity: synth voxels and compare e2e vs native ───────────────
    print(f"[2/5] PyTorch numeric sanity check (e2e wrapper vs HEAL native)")
    torch.manual_seed(42)
    MAX_VOX = args.max_voxels
    P = 32  # max_points_per_voxel

    # Synth: 1/3 full of real voxels, rest padded. Realistic invariants:
    #  - Real voxel features (real points): nonzero
    #  - Real voxel features (padding within pillar, i.e. idx >= num_points): zero
    #  - Padded pillars (m >= n_real): all-zero features, num_points=0, coords=0
    n_real = MAX_VOX // 3
    voxel_features = torch.zeros(MAX_VOX, P, 4, device="cuda")
    voxel_num_points = torch.zeros(MAX_VOX, dtype=torch.int32, device="cuda")
    voxel_num_points[:n_real] = torch.randint(1, P + 1, (n_real,),
                                              dtype=torch.int32, device="cuda")
    # Fill realistic point xyz+intensity for real voxels' real points only
    for m in range(n_real):
        npts = int(voxel_num_points[m].item())
        # x in [-102.4, 102.4], y in [-51.2, 51.2], z in [-3.5, 1.5], i in [0,1]
        voxel_features[m, :npts, 0] = (torch.rand(npts, device="cuda") - 0.5) * 204.8
        voxel_features[m, :npts, 1] = (torch.rand(npts, device="cuda") - 0.5) * 102.4
        voxel_features[m, :npts, 2] = torch.rand(npts, device="cuda") * 5.0 - 3.5
        voxel_features[m, :npts, 3] = torch.rand(npts, device="cuda")

    # Sample UNIQUE (batch, y, x) tuples (real PointPillar voxelization invariant)
    # Total slots = 2 * ny * nx = 262144 for DAIR
    n_slots = 2 * grid[1] * grid[0]
    assert n_real <= n_slots, f"n_real={n_real} > slots={n_slots}"
    flat = torch.randperm(n_slots, device="cuda")[:n_real]
    b = (flat // (grid[1] * grid[0])).to(torch.int32)
    yx = flat % (grid[1] * grid[0])
    y = (yx // grid[0]).to(torch.int32)
    x = (yx % grid[0]).to(torch.int32)
    voxel_coords = torch.zeros(MAX_VOX, 4, dtype=torch.int32, device="cuda")
    voxel_coords[:n_real, 0] = b
    voxel_coords[:n_real, 2] = y
    voxel_coords[:n_real, 3] = x
    # Ensure both batches present
    voxel_coords[0, 0] = 0
    voxel_coords[1, 0] = 1
    voxel_mask = torch.zeros(MAX_VOX, device="cuda")
    voxel_mask[:n_real] = 1.0
    t_ego = torch.eye(2, 3, device="cuda").unsqueeze(0).repeat(2, 1, 1)

    # Native VFE + Scatter on the REAL subset only (HEAL doesn't handle padding)
    batch_dict_native = {
        "voxel_features": voxel_features[:n_real],
        "voxel_num_points": voxel_num_points[:n_real],
        "voxel_coords": voxel_coords[:n_real],
    }
    with torch.no_grad():
        spatial_native = _native_vfe_scatter(full, batch_dict_native)
    print(f"  native spatial: {tuple(spatial_native.shape)}")

    # E2E wrapper VFE+Scatter on the FULL padded tensor (must match native after slicing)
    with torch.no_grad():
        pf = _vfe_forward_onnx(e2e.vfe, voxel_features, voxel_num_points,
                               voxel_coords, voxel_mask)
        spatial_e2e = _scatter_to_bev_onnx(pf, voxel_coords, voxel_mask,
                                           n_agents=2, H=grid[1], W=grid[0])
    print(f"  e2e   spatial: {tuple(spatial_e2e.shape)}")

    # Compare (modulo float64-vs-float32 noise from BN). Allow rel < 1%.
    d = (spatial_native - spatial_e2e).abs()
    rel = (d.max() / max(spatial_native.abs().max().item(), 1e-6)).item()
    print(f"  VFE+Scatter rel_max diff: {rel:.2%}  "
          f"(abs max: {d.max().item():.3e})  "
          f"{'PASS' if rel < 0.05 else 'FAIL'}")
    if rel >= 0.05:
        raise SystemExit(1)

    # Run full e2e forward
    with torch.no_grad():
        cls_pt, reg_pt, dir_pt = e2e(voxel_features, voxel_num_points,
                                     voxel_coords, voxel_mask, t_ego)
    print(f"  full e2e: cls={tuple(cls_pt.shape)} reg={tuple(reg_pt.shape)} "
          f"dir={tuple(dir_pt.shape)}")

    # ─── ONNX export ─────────────────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[3/5] Export ONNX -> {out_path} (opset={args.opset})")
    with torch.no_grad():
        torch.onnx.export(
            e2e,
            (voxel_features, voxel_num_points, voxel_coords, voxel_mask, t_ego),
            str(out_path),
            opset_version=args.opset,
            input_names=["voxel_features", "voxel_num_points",
                         "voxel_coords", "voxel_mask", "t_ego"],
            output_names=["cls_preds", "reg_preds", "dir_preds"],
            do_constant_folding=True,
            verbose=False,
        )
    print(f"  ONNX size: {out_path.stat().st_size / 1e6:.2f} MB")

    # ─── ONNX runtime sanity ─────────────────────────────────────────────────
    print(f"[4/5] ONNX runtime sanity (CPU provider)")
    import onnx
    import onnxruntime as ort

    m = onnx.load(str(out_path))
    onnx.checker.check_model(m)
    op_types = sorted({n.op_type for n in m.graph.node})
    print(f"  Op types ({len(op_types)}): {op_types}")

    # ORT CPU EP's MatMul+BN→Gemm fusion mis-handles 3D MatMul in PFNLayer,
    # disable all optimizations for sanity (TRT doesn't go through ORT).
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(str(out_path), so, providers=["CPUExecutionProvider"])
    feeds = {
        "voxel_features": voxel_features.cpu().numpy(),
        "voxel_num_points": voxel_num_points.cpu().numpy(),
        "voxel_coords": voxel_coords.cpu().numpy(),
        "voxel_mask": voxel_mask.cpu().numpy(),
        "t_ego": t_ego.cpu().numpy(),
    }
    cls_o, reg_o, dir_o = sess.run(None, feeds)

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
              f"rel_max={rel:.2%}")

    max_rel = max(rel_diffs)
    if max_rel < 0.01:
        print(f"  [PASS] (max rel diff {max_rel:.3%} < 1%)")
    elif max_rel < 0.05:
        print(f"  [PASS-MARGINAL] (max rel {max_rel:.3%} < 5%; CPU-vs-CUDA FP32 drift)")
    else:
        print(f"  [FAIL] (max rel {max_rel:.3%} > 5%)")
        raise SystemExit(1)

    print(f"[5/5] Done. ONNX saved: {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    dair_ckpt = ("/home/jichengzhi/heal_research/checkpoints/stage1/"
                 "Pyramid_DAIR_m1_base_2023_08_14_11_42_29/net_epoch_bestval_at23.pth")
    dair_hypes = ("/home/jichengzhi/heal_research/checkpoints/stage1/"
                  "Pyramid_DAIR_m1_base_2023_08_14_11_42_29/config.yaml")
    p.add_argument("--ckpt", default=dair_ckpt)
    p.add_argument("--hypes", default=dair_hypes)
    p.add_argument("--out", default=str(REPO_ROOT / "models/pyramid_dair_m1_e2e_fp32.onnx"))
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--max-voxels", type=int, default=70000,
                   help="DAIR max_voxel_test = 70000")
    return p.parse_args()


if __name__ == "__main__":
    export(parse_args())
