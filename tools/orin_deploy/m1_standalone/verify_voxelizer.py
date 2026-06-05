"""
Voxelizer equivalence verification: spconv (HEAL) vs torch (ours).

Runs on ≥10 real DAIR-V2X val samples, compares:
  - Set of unique (z,y,x) voxel coordinates  → coord_match %
  - Per-voxel num_points for matched voxels   → num_pts maxdiff
  - Per-voxel actual point content            → point maxdiff (for matched voxels)

Usage (on 4090 with HEAL conda env):
  conda activate UniV2X_2.0
  cd /home/jichengzhi/UniV2X
  python tools/orin_deploy/m1_standalone/verify_voxelizer.py

Requirements: spconv, pypcd (HEAL env), torch
"""

import sys
import os
import json
import numpy as np
import torch

# ── HEAL on sys.path ──────────────────────────────────────────────────────────
HEAL_ROOT = "/home/jichengzhi/heal_research/HEAL"
if HEAL_ROOT not in sys.path:
    sys.path.insert(0, HEAL_ROOT)

from voxelizer_torch import voxelize_torch
from opencood.data_utils.pre_processor.sp_voxel_preprocessor import SpVoxelPreprocessor
from opencood.utils import pcd_utils

# ── Config (m1 DAIR) ──────────────────────────────────────────────────────────
PC_RANGE   = [-102.4, -51.2, -3.5, 102.4, 51.2, 1.5]
VOXEL_SIZE = [0.4, 0.4, 5.0]
MAX_PTS    = 32
MAX_VOXELS = 70000

DAIR_ROOT  = "/home/jichengzhi/heal_research/dataset/my_dair_v2x/v2x_c/cooperative-vehicle-infrastructure"
N_SAMPLES  = 10      # default; set env var VERIFY_N to override

# ── SpVoxelPreprocessor wrapper ───────────────────────────────────────────────
def make_spconv_preprocessor():
    # SpVoxelPreprocessor expects: params['cav_lidar_range'] and params['args']
    cfg = {
        "cav_lidar_range": PC_RANGE,
        "args": {
            "voxel_size": VOXEL_SIZE,
            "max_points_per_voxel": MAX_PTS,
            "max_voxel_train": 32000,
            "max_voxel_test": MAX_VOXELS,
        }
    }
    return SpVoxelPreprocessor(cfg, train=False)


def spconv_voxelize(preprocessor, pcd_np):
    """Run spconv-based voxelizer; return (voxels, coords, num_pts) numpy."""
    data = preprocessor.preprocess(pcd_np)
    voxels   = data["voxel_features"]   # [M, 32, 4]  float32
    coords   = data["voxel_coords"]     # [M, 3]       int32  (z, y, x)
    num_pts  = data["voxel_num_points"] # [M]          int32
    return voxels, coords, num_pts


# ── Comparison helpers ────────────────────────────────────────────────────────
def coord_key(coords_np):
    """Return set of (z,y,x) tuples."""
    return set(map(tuple, coords_np.tolist()))


def compare(sp_vox, sp_coords, sp_npts, th_vox, th_coords, th_npts):
    """Return dict of comparison metrics."""
    sp_set = coord_key(sp_coords)
    th_set = coord_key(th_coords.cpu().numpy())

    only_sp = sp_set - th_set
    only_th = th_set - sp_set
    common  = sp_set & th_set
    union   = sp_set | th_set

    coord_iou   = len(common) / max(len(union), 1)
    sp_extra    = len(only_sp)
    th_extra    = len(only_th)

    # For common voxels, compare num_points and actual data
    sp_idx = {tuple(row): i for i, row in enumerate(sp_coords.tolist())}
    th_idx = {tuple(row): i for i, row in enumerate(th_coords.cpu().numpy().tolist())}

    num_pts_maxdiff = 0
    point_maxdiff   = 0.0

    th_vox_np = th_vox.cpu().numpy()
    for key in common:
        si = sp_idx[key]
        ti = th_idx[key]
        np_diff = abs(int(sp_npts[si]) - int(th_npts.cpu()[ti].item()))
        num_pts_maxdiff = max(num_pts_maxdiff, np_diff)

        # compare voxel content (up to min num_pts)
        n = min(int(sp_npts[si]), int(th_npts.cpu()[ti].item()), MAX_PTS)
        if n > 0:
            data_diff = float(np.abs(sp_vox[si, :n, :] - th_vox_np[ti, :n, :]).max())
            point_maxdiff = max(point_maxdiff, data_diff)

    return {
        "n_sp":            len(sp_set),
        "n_th":            len(th_set),
        "n_common":        len(common),
        "coord_iou":       coord_iou,
        "sp_extra":        sp_extra,
        "th_extra":        th_extra,
        "num_pts_maxdiff": num_pts_maxdiff,
        "point_maxdiff":   point_maxdiff,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    n_samples = int(os.environ.get("VERIFY_N", N_SAMPLES))
    print(f"[verify_voxelizer] N_SAMPLES={n_samples}")

    val_ids  = json.load(open(os.path.join(DAIR_ROOT, "val.json")))
    co_info  = json.load(open(os.path.join(DAIR_ROOT, "cooperative/data_info.json")))
    co_by_veh = {
        item["vehicle_image_path"].split("/")[-1].replace(".jpg", ""): item
        for item in co_info
    }

    val_frames = [co_by_veh[v] for v in val_ids if v in co_by_veh]
    val_frames = val_frames[:n_samples]
    print(f"  → using {len(val_frames)} val frames")

    preprocessor = make_spconv_preprocessor()

    agg = {k: [] for k in ["n_sp","n_th","n_common","coord_iou","sp_extra","th_extra","num_pts_maxdiff","point_maxdiff"]}

    for idx, frame in enumerate(val_frames):
        # Load infra-side PCD (main agent in m1 single-agent inference)
        pcd_path = os.path.join(DAIR_ROOT, frame["infrastructure_pointcloud_path"])
        pcd_np, _ = pcd_utils.read_pcd(pcd_path)   # [N, 4] numpy float32

        # ── spconv path ────────────────────────────────────────────────
        sp_vox, sp_coords, sp_npts = spconv_voxelize(preprocessor, pcd_np)

        # ── torch path ────────────────────────────────────────────────
        pts_t = torch.from_numpy(pcd_np).float().cpu()
        th_vox, th_coords, th_npts = voxelize_torch(
            pts_t, VOXEL_SIZE, PC_RANGE, max_pts=MAX_PTS, max_voxels=MAX_VOXELS
        )

        metrics = compare(sp_vox, sp_coords, sp_npts, th_vox, th_coords, th_npts)

        print(
            f"  [{idx:02d}] {os.path.basename(pcd_path):<16s} "
            f"sp={metrics['n_sp']:5d} th={metrics['n_th']:5d} "
            f"IoU={metrics['coord_iou']:.4f} "
            f"sp_extra={metrics['sp_extra']:3d} th_extra={metrics['th_extra']:3d} "
            f"npts_maxΔ={metrics['num_pts_maxdiff']:2d} "
            f"pt_maxΔ={metrics['point_maxdiff']:.6f}"
        )

        for k, v in metrics.items():
            agg[k].append(v)

    print()
    print("=" * 72)
    print("[SUMMARY across all samples]")
    for k, vals in agg.items():
        print(f"  {k:<22s}: mean={np.mean(vals):.4f}  max={np.max(vals):.4f}  min={np.min(vals):.4f}")

    iou_min = min(agg["coord_iou"])
    pt_maxd = max(agg["point_maxdiff"])
    np_maxd = max(agg["num_pts_maxdiff"])

    print()
    print("=" * 72)
    if iou_min >= 0.99 and pt_maxd < 1e-5 and np_maxd == 0:
        print(f"✅  EQUIVALENCE PASS  (coord_iou_min={iou_min:.4f}, pt_maxdiff={pt_maxd:.2e}, num_pts_maxdiff={np_maxd})")
    else:
        print(f"⚠️  EQUIVALENCE WARN/FAIL")
        print(f"   coord_iou_min  = {iou_min:.4f}  (want ≥0.99)")
        print(f"   pt_maxdiff     = {pt_maxd:.2e}  (want <1e-5)")
        print(f"   num_pts_maxdiff= {np_maxd}     (want 0)")
        print("   Note: small discrepancies can arise from tie-breaking in point")
        print("   ordering (both are correct; the final PFN max-pool is order-agnostic).")
    print("=" * 72)


if __name__ == "__main__":
    main()
