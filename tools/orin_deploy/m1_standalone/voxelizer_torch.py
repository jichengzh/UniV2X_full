"""
Pure-PyTorch PointPillar voxelizer — replaces SpVoxelPreprocessor (spconv).

Equivalent to Point2VoxelCPU3d from spconv-2.x:
  - max_num_points_per_voxel = 32
  - point_cloud_range = [-102.4, -51.2, -3.5, 102.4, 51.2, 1.5]
  - voxel_size = [0.4, 0.4, 5]
  - max_num_voxels = 70000 (test)

No spconv / open3d / pypcd dependencies.
Compatible with Python 3.8 + PyTorch 1.12+.

Author: sw-optimizer (Task #12, 2026-06-06)
"""

import numpy as np
import torch


def _stable_argsort(values: torch.Tensor) -> torch.Tensor:
    """Stable argsort with a compatibility path for Jetson's PyTorch 1.12."""
    try:
        return torch.argsort(values, dim=0, stable=True)
    except TypeError:
        order = np.argsort(values.detach().cpu().numpy(), kind="stable")
        return torch.from_numpy(order).to(device=values.device)


def voxelize_torch(
    points: torch.Tensor,
    voxel_size: list,
    pc_range: list,
    max_pts: int = 32,
    max_voxels: int = 70000,
):
    """
    Voxelize point cloud into pillars.

    Args:
        points:      [N, 4+] float32 tensor (x, y, z, intensity, ...)
        voxel_size:  [vx, vy, vz]
        pc_range:    [x_min, y_min, z_min, x_max, y_max, z_max]
        max_pts:     max points per voxel (32)
        max_voxels:  max total voxels     (70000)

    Returns:
        voxels:      [M, max_pts, 4] float32  — padded with zeros
        coords:      [M, 3]         int32     — (z, y, x) order (OpenPCDet convention)
        num_points:  [M]            int32     — actual #points in each voxel
    """
    device = points.device

    pc_min = torch.tensor(pc_range[:3], dtype=torch.float32, device=device)
    pc_max = torch.tensor(pc_range[3:], dtype=torch.float32, device=device)
    vs = torch.tensor(voxel_size,       dtype=torch.float32, device=device)

    # ── 1. Filter points within lidar range (convention: [min, max) ) ──────
    valid = (
        (points[:, 0] >= pc_min[0]) & (points[:, 0] < pc_max[0]) &
        (points[:, 1] >= pc_min[1]) & (points[:, 1] < pc_max[1]) &
        (points[:, 2] >= pc_min[2]) & (points[:, 2] < pc_max[2])
    )
    pts = points[valid, :4].float()     # keep only (x, y, z, intensity)

    if pts.shape[0] == 0:
        return (
            torch.zeros(0, max_pts, 4, dtype=torch.float32, device=device),
            torch.zeros(0, 3,       dtype=torch.int32,      device=device),
            torch.zeros(0,          dtype=torch.int32,      device=device),
        )

    # ── 2. Per-point voxel indices (ix, iy, iz) ─────────────────────────────
    vox_ixyz = torch.floor((pts[:, :3] - pc_min) / vs).long()  # [N, 3]

    # Grid size along each axis
    gx = round((pc_range[3] - pc_range[0]) / voxel_size[0])    # x cells = 512
    gy = round((pc_range[4] - pc_range[1]) / voxel_size[1])    # y cells = 256
    # gz = round((pc_range[5] - pc_range[2]) / voxel_size[2])  # z cells = 1

    # ── 3. Flatten to a single z-major index (matches OpenPCDet coords order) ─
    flat = vox_ixyz[:, 2] * (gy * gx) + vox_ixyz[:, 1] * gx + vox_ixyz[:, 0]

    # ── 4. Sort points by flat voxel index ───────────────────────────────────
    order = _stable_argsort(flat)
    flat = flat[order]
    pts  = pts[order]

    # ── 5. Unique voxels + per-voxel point counts ───────────────────────────
    unique_flat, counts = torch.unique_consecutive(flat, return_counts=True)

    n_vox       = min(len(unique_flat), max_voxels)
    unique_flat = unique_flat[:n_vox]
    counts      = counts[:n_vox]

    # ── 6. Build output arrays (fully vectorized, no Python loops) ───────────
    # Cumulative start offset for each voxel in the sorted pts array
    cum = torch.zeros(n_vox + 1, dtype=torch.long, device=device)
    cum[1:] = torch.cumsum(counts, 0)
    n_pts = int(cum[n_vox].item())      # total points from the first n_vox voxels
    pts = pts[:n_pts]                   # drop points that belong to excess voxels

    # Voxel-id for every (truncated) point  →  [n_pts]
    vox_id = torch.repeat_interleave(
        torch.arange(n_vox, dtype=torch.long, device=device), counts
    )

    # In-voxel position (local index within each voxel's group)
    in_pos = torch.arange(n_pts, dtype=torch.long, device=device) - cum[vox_id]

    # Only keep positions < max_pts  (spconv drops extra points per voxel too)
    keep   = in_pos < max_pts
    vi     = vox_id[keep]
    pos    = in_pos[keep]
    pt_idx = torch.arange(n_pts, dtype=torch.long, device=device)[keep]

    voxels = torch.zeros(n_vox, max_pts, 4, dtype=torch.float32, device=device)
    voxels[vi, pos] = pts[pt_idx]

    num_points = torch.clamp(counts, max=max_pts).int()

    # ── 7. Decode flat back to (z, y, x) coordinates ─────────────────────────
    iz_out  = (unique_flat // (gy * gx)).int()
    rem     = unique_flat  % (gy * gx)
    iy_out  = (rem // gx).int()
    ix_out  = (rem  % gx).int()
    coords  = torch.stack([iz_out, iy_out, ix_out], dim=1)  # [M, 3] (z, y, x)

    return voxels, coords, num_points
