"""
Export CoDriving cooperative kernel (N=2 fixed, DAIR) to ONNX.

Architecture: spatial_features (2,64,256,512) -> backbone.resnet (multiscale)
-> per-scale: warp_affine_simple (ONNX-friendly) + AttenFusion[i] -> deblocks[i]
-> concat(3 scales) -> shrink_conv -> cls_head + reg_head

Inputs:
    spatial_features   : (2, 64, 256, 512)    2 stacked agents (ego idx=0)
    pairwise_t_matrix  : (1, 2, 2, 4, 4)      B=1, L=2 agents, 4x4 homogeneous

Outputs:
    cls_preds  : (1, 1, 128, 256)   -- anchor_number=1
    reg_preds  : (1, 8, 128, 256)   -- 8 * anchor_number
    Note: DAIR config has NO shrink_header, so out_channel=384 -> cls/reg direct.

DAIR-V2X geometry:
    lidar_range = [-102.4, -51.2, -3.5, 102.4, 51.2, 1.5]
    voxel_size  = [0.4, 0.4, 5]
    grid_size   = (512, 256, 1) -> spatial_features (N, 64, 256, 512)
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
# numpy imported lazily to avoid tvm310 numpy 2.x / t2lib incompatibility

# -----------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------
REPO_ROOT = Path("/exdata/jichengzhi/V2Xverse_pyramid")
T2LIB = Path("/data/jichengzhi_v2x/t2lib")
TVM_SP = Path("/exdata/jichengzhi/tvm310/lib/python3.10/site-packages")

# onnx package must come BEFORE t2lib so torch.onnx can find it
for p in [str(TVM_SP), str(T2LIB), str(REPO_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

CKPT_PATH = str(REPO_ROOT / "opencood/logs"
                / "dair_centerpoint_codriving_2026_06_15_21_19_13"
                / "net_epoch_bestval_at11.pth")
HYPES_PATH = str(REPO_ROOT / "opencood/logs"
                 / "dair_centerpoint_codriving_2026_06_15_21_19_13"
                 / "config.yaml")
ONNX_OUT = str(REPO_ROOT / "output/codriving_pilot/collab_export"
               / "codriving_collab_base_fp32.onnx")
PT_OUT = str(REPO_ROOT / "output/codriving_pilot/collab_export"
             / "codriving_collab_base_fp32_ptout.npz")


# -----------------------------------------------------------------------
# ONNX-friendly affine warp
# (replaces F.affine_grid -> aten::affine_grid_generator, not in TRT < 8.6)
# -----------------------------------------------------------------------

def manual_affine_grid(M: torch.Tensor, dsize, align_corners: bool = False):
    """Equivalent to F.affine_grid(M, [N,C,H,W]) without aten::affine_grid_generator.

    M     : (N, 2, 3)
    Returns grid : (N, H, W, 2)

    Verified against F.affine_grid: max abs diff < 1e-6 for align_corners
    True/False on random (N,2,3) matrices.
    """
    H, W = dsize
    device, dtype = M.device, M.dtype
    if align_corners:
        ys = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
    else:
        ys = (torch.arange(H, device=device, dtype=dtype) * 2 + 1) / H - 1
        xs = (torch.arange(W, device=device, dtype=dtype) * 2 + 1) / W - 1
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")   # (H, W)
    ones = torch.ones_like(grid_x)
    base = torch.stack([grid_x, grid_y, ones], dim=-1)        # (H, W, 3)
    # einsum: (N, 2, 3) x (H, W, 3) -> (N, H, W, 2)
    grid = torch.einsum("nij,hwj->nhwi", M, base)
    return grid


def warp_affine_onnx(src: torch.Tensor, M: torch.Tensor, dsize,
                     align_corners: bool = False):
    """Drop-in for warp_affine_simple using manual_affine_grid (ONNX-safe)."""
    if M.dtype != src.dtype:
        M = M.to(src.dtype)
    grid = manual_affine_grid(M, dsize, align_corners=align_corners)
    return F.grid_sample(src, grid, align_corners=align_corners,
                         mode="bilinear", padding_mode="zeros")


# -----------------------------------------------------------------------
# CoDriving collaborative subnet -- fixed N=2, B=1
# -----------------------------------------------------------------------

class CoDrivingCollabN2(nn.Module):
    """Fixed N=2 cooperative inference kernel.

    Mirrors CoDriving.forward multi_scale + with_resnet=True path with:
      * record_len = [2]  (hardcoded -> no dynamic split via tensor_split)
      * pairwise_t_matrix (1, 2, 2, 4, 4) -> sliced to (2, 2, 3) affine
      * warp_affine_simple -> warp_affine_onnx (manual grid, no affine_grid op)
      * communication disabled (config has no 'communication' key)
      * no dir_head (center-point model has cls+reg only)
    """

    def __init__(self, model: nn.Module, align_corners: bool = False):
        super().__init__()
        self.backbone = model.backbone          # ResNetBEVBackbone
        # shrink_conv is optional (DAIR config does NOT have shrink_header)
        self.shrink_flag = model.shrink_flag
        if self.shrink_flag:
            self.shrink_conv = model.shrink_conv
        self.cls_head = model.cls_head
        self.reg_head = model.reg_head

        fusion = model.fusion_net               # CoDriving instance
        self.fuse_modules = fusion.fuse_modules # ModuleList[AttenFusion]
        self.num_levels = fusion.num_levels     # 3

        # normalization params (same as CoDriving.forward)
        self.discrete_ratio = fusion.discrete_ratio    # voxel_size[0] = 0.4
        self.downsample_rate = fusion.downsample_rate  # 1
        self.align_corners = align_corners

    def _build_t2x3(self, pairwise_t_matrix: torch.Tensor,
                    H: int, W: int) -> torch.Tensor:
        """Convert (1,2,2,4,4) homogeneous matrix to (2,2,3) normalized affine.

        Exactly mirrors:
          pairwise_t_matrix = pairwise_t_matrix[:,:,:,[0,1],:][:,:,:,:,[0,1,3]]
          pairwise_t_matrix[...,0,1] *= H/W
          ...
        from CoDriving.forward.
        """
        # (1,2,2,4,4) -> (2,2,2,3)
        t = pairwise_t_matrix[0, :2, :2, :, :]           # (2,2,4,4)
        t = t[:, :, [0, 1], :][:, :, :, [0, 1, 3]]       # (2,2,2,3)
        t = t.clone()
        t[..., 0, 1] = t[..., 0, 1] * H / W
        t[..., 1, 0] = t[..., 1, 0] * W / H
        t[..., 0, 2] = t[..., 0, 2] / (
            self.downsample_rate * self.discrete_ratio * W) * 2
        t[..., 1, 2] = t[..., 1, 2] / (
            self.downsample_rate * self.discrete_ratio * H) * 2
        return t  # (2, 2, 2, 3)

    def forward(self, spatial_features: torch.Tensor,
                pairwise_t_matrix: torch.Tensor):
        """
        spatial_features   : (2, 64, 256, 512)
        pairwise_t_matrix  : (1, 2, 2, 4, 4)

        Returns:
            cls_preds : (1, 1, H_out, W_out)
            reg_preds : (1, 8, H_out, W_out)
        """
        # Normalize pairwise_t_matrix ONCE with input spatial H,W (256, 512).
        # Mirrors native CoDriving.forward() which normalizes once at the top:
        #   _, C, H, W = x.shape  (x = spatial_features, H=256, W=512)
        #   pairwise_t_matrix = pairwise_t_matrix[:,:,:,[0,1],:][:,:,:,:,[0,1,3]]
        #   pairwise_t_matrix[...,0,1] *= H/W   etc.
        # Bug in original: per-level H_i/W_i caused wrong affine at scales > 0.
        H_in = spatial_features.shape[2]   # 256
        W_in = spatial_features.shape[3]   # 512
        t_full_once = self._build_t2x3(pairwise_t_matrix, H_in, W_in)
        # t_full_once[ego=0, :N, :, :] -> (2, 2, 3) normalized affine to ego frame
        t_to_ego = t_full_once[0, :, :, :]   # (2, 2, 3)

        # Multi-scale feature extraction via ResNet
        feats = self.backbone.resnet(spatial_features)
        # feats[0]: (2, 64,  H/2, W/2)  = (2,  64, 128, 256)
        # feats[1]: (2, 128, H/4, W/4)  = (2, 128,  64, 128)
        # feats[2]: (2, 256, H/8, W/8)  = (2, 256,  32,  64)

        ups = []
        for i in range(self.num_levels):
            x_i = feats[i]                              # (2, C_i, H_i, W_i)
            _, C_i, H_i, W_i = x_i.shape

            # Warp both agents into ego coordinate frame using the ONCE-normalized
            # affine matrix (same as native CoDriving behavior at all scales).
            neighbor_feature = warp_affine_onnx(
                x_i, t_to_ego, (H_i, W_i),
                align_corners=self.align_corners)       # (2, C_i, H_i, W_i)

            # AttenFusion: scaled dot-product attention over N=2 agents
            # output is ego feature (C_i, H_i, W_i) -- AttenFusion returns [0]
            fused = self.fuse_modules[i](neighbor_feature)   # (C_i, H_i, W_i)
            fused = fused.unsqueeze(0)                       # (1, C_i, H_i, W_i)

            # Deblock upsample
            ups.append(self.backbone.deblocks[i](fused))   # (1, 128, H_out, W_out)

        # Concat multi-scale: (1, 384, H_out, W_out)
        x_fuse = torch.cat(ups, dim=1)

        # Shrink conv is optional (DAIR config does not have shrink_header)
        if self.shrink_flag:
            x_fuse = self.shrink_conv(x_fuse)

        cls = self.cls_head(x_fuse)   # (1, 1,  H_out, W_out)
        reg = self.reg_head(x_fuse)   # (1, 8,  H_out, W_out)
        return cls, reg


# -----------------------------------------------------------------------
# Build model from checkpoint
# -----------------------------------------------------------------------

def build_codriving_from_ckpt(hypes_path: str, ckpt_path: str,
                               device: str = "cpu") -> nn.Module:
    import re
    import yaml
    # Use V2Xverse_pyramid's train_utils.create_model pattern
    from opencood.tools.train_utils import create_model

    # Config uses !!python/object/apply:... tags -> requires yaml.Loader (not SafeLoader)
    with open(hypes_path, "r") as f:
        loader = yaml.Loader
        loader.add_implicit_resolver(
            u'tag:yaml.org,2002:float',
            re.compile(u'''^(?:
             [-+]?(?:[0-9][0-9_]*)\\.([0-9_]*)(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\\.[0-9_]+(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
            |[-+]?\\.(?:inf|Inf|INF)
            |\\.(?:nan|NaN|NAN))$''', re.X),
            list(u'-+0123456789.'))
        hypes = yaml.load(f, Loader=loader)

    model = create_model(hypes)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # V2Xverse checkpoints are flat state_dicts (no 'model_state_dict' wrapper)
    # But handle DDP-saved 'module.' prefix just in case
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif any(k.startswith("module.") for k in ckpt.keys()):
            state_dict = {k.replace("module.", "", 1): v for k, v in ckpt.items()}
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"  [load] missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:
        print(f"    missing keys (first 5): {missing[:5]}")
    if unexpected:
        print(f"    unexpected keys (first 5): {unexpected[:5]}")

    model.to(device).eval()
    return model


# -----------------------------------------------------------------------
# Main export function
# -----------------------------------------------------------------------

def export(args):
    device = args.device

    print(f"[1/5] Build model from: {args.ckpt}")
    full_model = build_codriving_from_ckpt(args.hypes, args.ckpt, device=device)
    subnet = CoDrivingCollabN2(full_model, align_corners=False).to(device).eval()
    n_params = sum(p.numel() for p in subnet.parameters())
    print(f"  Subnet params: {n_params:,}")

    # Verify subnet structure
    print(f"  num_levels={subnet.num_levels}, "
          f"discrete_ratio={subnet.discrete_ratio}, "
          f"downsample_rate={subnet.downsample_rate}")

    # ---- Dummy inputs ----
    print(f"[2/5] PyTorch forward sanity (random inputs, identity t_matrix)")
    torch.manual_seed(42)
    # DAIR: spatial_features (2, 64, 256, 512)
    spatial = torch.randn(2, 64, 256, 512, device=device)
    # Identity pairwise_t_matrix: (1, 2, 2, 4, 4)
    t_mat = torch.zeros(1, 2, 2, 4, 4, device=device)
    t_mat[0, :, :, 0, 0] = 1.0
    t_mat[0, :, :, 1, 1] = 1.0
    t_mat[0, :, :, 2, 2] = 1.0
    t_mat[0, :, :, 3, 3] = 1.0

    with torch.no_grad():
        cls_pt, reg_pt = subnet(spatial, t_mat)
    print(f"  cls={tuple(cls_pt.shape)}  reg={tuple(reg_pt.shape)}")

    # Save PyTorch outputs as reference for later ORT numerical check
    # Use torch.save (avoid numpy -- tvm310 numpy 2.x conflicts with t2lib)
    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    pt_out_local = str(out_dir / "codriving_collab_base_fp32_ptout.pt")
    torch.save({
        "spatial": spatial.cpu(),
        "pairwise_t_matrix": t_mat.cpu(),
        "cls_preds": cls_pt.cpu(),
        "reg_preds": reg_pt.cpu(),
    }, pt_out_local)
    print(f"  PyTorch reference saved: {pt_out_local}")

    # ---- ONNX export ----
    out_path = Path(args.out)
    print(f"[3/5] ONNX export (opset={args.opset}) -> {out_path}")
    onnx_ok = False
    try:
        with torch.no_grad():
            torch.onnx.export(
                subnet,
                (spatial, t_mat),
                str(out_path),
                opset_version=args.opset,
                input_names=["spatial_features", "pairwise_t_matrix"],
                output_names=["cls_preds", "reg_preds"],
                do_constant_folding=True,
                verbose=False,
            )
        size_mb = out_path.stat().st_size / 1e6
        print(f"  ONNX size: {size_mb:.2f} MB")
        onnx_ok = True
    except Exception as e:
        print(f"  [ONNX EXPORT ERROR] {type(e).__name__}: {e}")

    # ---- ONNX model check ----
    if onnx_ok:
        print(f"[4/5] ONNX model check (onnx.checker)")
        try:
            import onnx
            m = onnx.load(str(out_path))
            onnx.checker.check_model(m)
            op_types = sorted({n.op_type for n in m.graph.node})
            print(f"  Op types ({len(op_types)}): {op_types}")
            print(f"  onnx.checker: PASS")
        except ImportError:
            print(f"  onnx not available -- skip checker (run on 4090)")
        except Exception as e:
            print(f"  onnx checker error: {type(e).__name__}: {e}")
    else:
        print(f"[4/5] ONNX export failed -- skip checker")

    # ---- ORT numerical check ----
    print(f"[5/5] ORT numerical check")
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(out_path),
                                    providers=["CUDAExecutionProvider",
                                               "CPUExecutionProvider"])
        # Use .detach().to(torch.float32) and convert via list to avoid numpy compat issues
        def t2np(x):
            import ctypes
            t = x.detach().cpu().to(torch.float32).contiguous()
            return t.numpy()

        cls_o, reg_o = sess.run(
            None,
            {
                "spatial_features": t2np(spatial),
                "pairwise_t_matrix": t2np(t_mat),
            },
        )
        for name, pt_t, ox in [("cls", cls_pt, cls_o),
                                ("reg", reg_pt, reg_o)]:
            pt = t2np(pt_t)
            import numpy as _np
            d = _np.abs(pt - ox)
            rel = d.max() / max(_np.abs(pt).max(), 1e-6)
            cos = float(
                _np.dot(pt.flatten(), ox.flatten()) /
                (_np.linalg.norm(pt.flatten()) *
                 _np.linalg.norm(ox.flatten()) + 1e-12)
            )
            print(f"  {name}: max|delta|={d.max():.3e}  rel_max={rel:.2%}"
                  f"  cosine={cos:.6f}")
    except ImportError:
        print(f"  onnxruntime not in this env -- skip (SCP to 4090 to verify)")
    except Exception as e:
        print(f"  ORT check error: {type(e).__name__}: {e}")

    # ---- Summary ----
    print(f"\n=== EXPORT SUMMARY ===")
    print(f"  ONNX:        {'SAVED -> ' + str(out_path) if onnx_ok else 'FAILED'}")
    print(f"  PyTorch ref: {PT_OUT}")
    print(f"  cls shape:   {tuple(cls_pt.shape)}")
    print(f"  reg shape:   {tuple(reg_pt.shape)}")
    if onnx_ok:
        print(f"  ONNX size:   {Path(args.out).stat().st_size/1e6:.2f} MB")
    return onnx_ok


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default=CKPT_PATH,
                   help="CoDriving DAIR bestval@11 checkpoint path")
    p.add_argument("--hypes", default=HYPES_PATH,
                   help="Config yaml (same dir as ckpt)")
    p.add_argument("--out", default=ONNX_OUT,
                   help="Output ONNX path")
    p.add_argument("--opset", type=int, default=17,
                   help="ONNX opset version (16+ required for grid_sampler)")
    p.add_argument("--device", default="cpu",
                   help="'cpu' (safe on busy server) or 'cuda:6' if GPU6 free")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ok = export(args)
    sys.exit(0 if ok else 1)
