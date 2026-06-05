"""
Voxelizer equivalence verification: spconv (HEAL) vs torch (ours).

Runs on all 60 real DAIR-V2X val npy samples (= timing population),
compares:
  - Set of unique (z,y,x) voxel coordinates  → coord_match (bool per frame)
  - Per-voxel num_points for matched voxels   → num_pts maxdiff
  - Per-voxel actual point content            → point maxdiff (scientific notation)

Also reports:
  - Boundary frames: highest/lowest point count
  - Per-frame pt_maxdiff distribution (max, mean, histogram)

Usage (on 4090 with HEAL conda env):
  conda activate UniV2X_2.0
  cd /home/jichengzhi/V2X
  python tools/orin_deploy/m1_standalone/verify_voxelizer.py

Requirements: spconv, pypcd (HEAL env), torch
"""

import sys
import os
import json
import numpy as np
import torch
from pathlib import Path

# ── HEAL on sys.path ──────────────────────────────────────────────────────────
HEAL_ROOT = os.environ.get("HEAL_ROOT", "/home/jichengzhi/heal_research/HEAL")
if HEAL_ROOT not in sys.path:
    sys.path.insert(0, HEAL_ROOT)

THIS_DIR = Path(__file__).parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from voxelizer_torch import voxelize_torch
from opencood.data_utils.pre_processor.sp_voxel_preprocessor import SpVoxelPreprocessor

# ── Config (m1 DAIR) ──────────────────────────────────────────────────────────
PC_RANGE   = [-102.4, -51.2, -3.5, 102.4, 51.2, 1.5]
VOXEL_SIZE = [0.4, 0.4, 5.0]
MAX_PTS    = 32
MAX_VOXELS = 70000

NPY_DIR    = THIS_DIR / "dair_val_npy" / "infra"   # 60 infra .npy files


# ── SpVoxelPreprocessor wrapper ───────────────────────────────────────────────
def make_spconv_preprocessor():
    cfg = {
        "cav_lidar_range": PC_RANGE,
        "args": {
            "voxel_size":           VOXEL_SIZE,
            "max_points_per_voxel": MAX_PTS,
            "max_voxel_train":      32000,
            "max_voxel_test":       MAX_VOXELS,
        }
    }
    return SpVoxelPreprocessor(cfg, train=False)


def spconv_voxelize(preprocessor, pcd_np):
    data    = preprocessor.preprocess(pcd_np)
    voxels  = data["voxel_features"]    # [M, 32, 4]  float32
    coords  = data["voxel_coords"]      # [M, 3]       int32  (z, y, x)
    num_pts = data["voxel_num_points"]  # [M]          int32
    return voxels, coords, num_pts


# ── Comparison helpers ────────────────────────────────────────────────────────
def coord_key(coords_np):
    return set(map(tuple, coords_np.tolist()))


def compare(sp_vox, sp_coords, sp_npts, th_vox, th_coords, th_npts):
    sp_set  = coord_key(sp_coords)
    th_set  = coord_key(th_coords.cpu().numpy())

    only_sp = sp_set - th_set
    only_th = th_set - sp_set
    common  = sp_set & th_set
    union   = sp_set | th_set

    coord_iou       = len(common) / max(len(union), 1)
    sp_extra        = len(only_sp)
    th_extra        = len(only_th)

    sp_idx  = {tuple(row): i for i, row in enumerate(sp_coords.tolist())}
    th_idx  = {tuple(row): i for i, row in enumerate(
                   th_coords.cpu().numpy().tolist())}

    num_pts_maxdiff = 0
    point_maxdiff   = 0.0
    th_vox_np       = th_vox.cpu().numpy()

    for key in common:
        si = sp_idx[key]
        ti = th_idx[key]
        np_diff = abs(int(sp_npts[si]) - int(th_npts.cpu()[ti].item()))
        num_pts_maxdiff = max(num_pts_maxdiff, np_diff)

        n = min(int(sp_npts[si]), int(th_npts.cpu()[ti].item()), MAX_PTS)
        if n > 0:
            data_diff = float(
                np.abs(sp_vox[si, :n, :] - th_vox_np[ti, :n, :]).max()
            )
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
    # Collect all infra npy files (= timing population, 60 frames)
    npy_files = sorted(NPY_DIR.glob("*.npy"))
    if not npy_files:
        print(f"[ERROR] No .npy files found in {NPY_DIR}")
        print("  → Run convert_dair_to_npy.py first")
        sys.exit(1)

    n_samples = len(npy_files)
    print(f"[verify_voxelizer] Population = {n_samples} infra .npy frames "
          f"(canonical timing population, all will be verified)")

    preprocessor = make_spconv_preprocessor()
    rows = []   # per-frame result dicts

    for idx, npy_path in enumerate(npy_files):
        pcd_np = np.load(str(npy_path)).astype(np.float32)
        n_pts  = pcd_np.shape[0]

        # ── spconv path ─────────────────────────────────────────────────
        sp_vox, sp_coords, sp_npts = spconv_voxelize(preprocessor, pcd_np)

        # ── torch path ──────────────────────────────────────────────────
        pts_t = torch.from_numpy(pcd_np).float().cpu()
        th_vox, th_coords, th_npts = voxelize_torch(
            pts_t, VOXEL_SIZE, PC_RANGE, max_pts=MAX_PTS, max_voxels=MAX_VOXELS
        )

        metrics              = compare(sp_vox, sp_coords, sp_npts,
                                       th_vox, th_coords, th_npts)
        metrics["n_pts_input"] = n_pts
        metrics["frame"]       = npy_path.name
        rows.append(metrics)

        print(
            f"  [{idx:02d}] {npy_path.name:<22s} "
            f"pts={n_pts:6d} sp={metrics['n_sp']:5d} th={metrics['n_th']:5d} "
            f"IoU={metrics['coord_iou']:.8f} "
            f"sp_extra={metrics['sp_extra']:2d} th_extra={metrics['th_extra']:2d} "
            f"npts_maxΔ={metrics['num_pts_maxdiff']:2d} "
            f"pt_maxΔ={metrics['point_maxdiff']:.6e}"
        )

    # ── Aggregate stats ───────────────────────────────────────────────────────
    iou_vals = [r["coord_iou"]       for r in rows]
    ptd_vals = [r["point_maxdiff"]   for r in rows]
    npd_vals = [r["num_pts_maxdiff"] for r in rows]
    pts_vals = [r["n_pts_input"]     for r in rows]

    iou_min = min(iou_vals)
    ptd_max = max(ptd_vals)
    npd_max = max(npd_vals)

    max_pts_idx = int(np.argmax(pts_vals))
    min_pts_idx = int(np.argmin(pts_vals))

    print()
    print("=" * 80)
    print(f"[SUMMARY — {n_samples} infra frames, timing population coverage]")
    print()
    print(f"  coord_iou       : min={iou_min:.10f}  mean={np.mean(iou_vals):.10f}  "
          f"max={max(iou_vals):.10f}")
    print(f"  point_maxdiff   : min={min(ptd_vals):.6e}  "
          f"mean={np.mean(ptd_vals):.6e}  "
          f"max={ptd_max:.6e}")
    print(f"  num_pts_maxdiff : min={min(npd_vals)}  max={npd_max}")
    print()
    print(f"  Input point-count distribution:")
    print(f"    min = {min(pts_vals):6d} pts  → frame [{rows[min_pts_idx]['frame']}]")
    print(f"    max = {max(pts_vals):6d} pts  → frame [{rows[max_pts_idx]['frame']}]")
    print(f"    mean= {np.mean(pts_vals):6.0f}  median={np.median(pts_vals):.0f}")
    print()
    print(f"  Boundary frame equivalence check:")
    bmin = rows[min_pts_idx]
    bmax = rows[max_pts_idx]
    print(f"    FEWEST-pts  [{bmin['frame']}]  "
          f"pts={bmin['n_pts_input']} "
          f"IoU={bmin['coord_iou']:.10f}  "
          f"pt_maxΔ={bmin['point_maxdiff']:.6e}")
    print(f"    MOST-pts    [{bmax['frame']}]  "
          f"pts={bmax['n_pts_input']} "
          f"IoU={bmax['coord_iou']:.10f}  "
          f"pt_maxΔ={bmax['point_maxdiff']:.6e}")
    print()

    # pt_maxdiff histogram
    bucket_edges  = [0.0, 1e-10, 1e-8, 1e-6, 1e-4, float("inf")]
    bucket_labels = ["==0.0 (exact)",
                     "(0, 1e-10]",
                     "(1e-10, 1e-8]",
                     "(1e-8,  1e-6]",
                     "(1e-6,  1e-4]",
                     "> 1e-4"]
    counts = [0] * len(bucket_labels)
    for v in ptd_vals:
        if v == 0.0:
            counts[0] += 1
        else:
            for bi in range(1, len(bucket_edges)):
                if v <= bucket_edges[bi]:
                    counts[bi] += 1
                    break

    print("  point_maxdiff histogram:")
    for label, cnt in zip(bucket_labels, counts):
        bar = "█" * cnt
        print(f"    {label:<22s} : {cnt:3d}  {bar}")
    print()

    # ── PASS / FAIL verdict ───────────────────────────────────────────────────
    n_exact_coord = sum(1 for v in iou_vals if v >= 1.0 - 1e-9)
    n_exact_pts   = sum(1 for v in ptd_vals if v == 0.0)

    print("=" * 80)
    if iou_min >= 0.9999 and ptd_max < 1e-5 and npd_max == 0:
        verdict = "✅  EQUIVALENCE PASS"
    else:
        verdict = "⚠️   EQUIVALENCE WARN/FAIL"

    print(f"{verdict}")
    print(f"  coord exact-match  : {n_exact_coord}/{n_samples} frames (IoU=1.000…)")
    print(f"  point_maxdiff==0.0 : {n_exact_pts}/{n_samples} frames (truly zero, not 2dp truncation)")
    print(f"  point_maxdiff max  : {ptd_max:.6e}  (threshold: < 1e-5)")
    print(f"  num_pts_maxdiff max: {npd_max}  (threshold: == 0)")
    if 0.0 < ptd_max < 1e-5:
        print("  Note: sub-1e-5 diffs arise from float32 tie-breaking in sort order.")
        print("  PFN uses channel-wise max-pool over point dim → order-agnostic → AP-equivalent.")
    print("=" * 80)


if __name__ == "__main__":
    main()
