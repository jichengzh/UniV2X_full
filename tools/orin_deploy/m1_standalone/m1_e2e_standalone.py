"""
Standalone end-to-end inference + CUDA Event timing for HEAL m1 (PyramidFusion/DAIR).

No spconv required for voxelization (uses voxelizer_torch.py).
Uses HEAL's create_model + load_saved_model for correct weight loading.

Pipeline per frame:
  voxelize_torch  →  encoder_m1.pillar_vfe  →  encoder_m1.scatter  →
  backbone_m1  →  pyramid_backbone  →  shrink_conv  →  cls/reg/dir heads

CUDA Event timing stages:
  t_voxelize   — voxelize_torch + H2D copy + batch collation (CPU + GPU)
  t_encoder    — pillar_vfe + scatter  →  [B_agents, 64, 256, 512]
  t_backbone   — backbone_m1           →  [B_agents, 64, 128, 256]
  t_fusion     — pyramid_backbone      →  [1, 384, 128, 256]
  t_head       — shrink_conv + cls/reg/dir heads
  t_e2e_ms     — sum of all above (no NMS, no file I/O)

Usage:
  CUDA_VISIBLE_DEVICES=0 python m1_e2e_standalone.py --mode npy --batch 1 --n_frames 50
  CUDA_VISIBLE_DEVICES=0 python m1_e2e_standalone.py --mode npy --batch 2 --n_frames 50

Author: sw-optimizer (Task #12, 2026-06-06)
Compat: Python 3.8+, PyTorch 1.12+
"""

import sys
import os
import argparse
import json
import csv
import statistics
from pathlib import Path

import numpy as np
import torch

# ── HEAL on sys.path ─────────────────────────────────────────────────────────
HEAL_ROOT = os.environ.get("HEAL_ROOT", "/home/jichengzhi/heal_research/HEAL")
if HEAL_ROOT not in sys.path:
    sys.path.insert(0, HEAL_ROOT)

THIS_DIR = Path(__file__).parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from voxelizer_torch import voxelize_torch

# ── Constants ─────────────────────────────────────────────────────────────────
PC_RANGE   = [-102.4, -51.2, -3.5, 102.4, 51.2, 1.5]
VOXEL_SIZE = [0.4, 0.4, 5.0]
MAX_PTS    = 32
MAX_VOXELS = 70000

YAML_PATH  = (f"{HEAL_ROOT}/opencood/hypes_yaml/dairv2x/MoreModality/"
              "HEAL/stage1/m1_pyramid.yaml")
CKPT_DIR   = ("/home/jichengzhi/heal_research/checkpoints/stage1/"
              "Pyramid_DAIR_m1_base_2023_08_14_11_42_29")

DAIR_ROOT  = ("/home/jichengzhi/heal_research/dataset/my_dair_v2x/"
              "v2x_c/cooperative-vehicle-infrastructure")
NPY_DIR    = THIS_DIR / "dair_val_npy"


# ─────────────────────────────────────────────────────────────────────────────
#  Model loading via HEAL's standard utilities
# ─────────────────────────────────────────────────────────────────────────────

def load_heal_model(device):
    """
    Use HEAL's create_model + load_saved_model so weight keys match exactly.
    Returns the full HeterPyramidCollab model on `device`.
    """
    import yaml
    from opencood.tools import train_utils

    with open(YAML_PATH) as f:
        hypes = yaml.safe_load(f)

    print("[load_heal_model] Creating model …")
    model = train_utils.create_model(hypes)

    print(f"[load_heal_model] Loading ckpt from {CKPT_DIR} …")
    _, model = train_utils.load_saved_model(CKPT_DIR, model)
    model = model.to(device).eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
#  Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_npy(path):
    return np.load(str(path)).astype(np.float32)


def load_pcd(path):
    from opencood.utils import pcd_utils
    arr, _ = pcd_utils.read_pcd(str(path))
    return arr.astype(np.float32)


def get_frame_paths(mode, n_needed):
    val_ids   = json.load(open(os.path.join(DAIR_ROOT, "val.json")))
    co_info   = json.load(open(os.path.join(DAIR_ROOT, "cooperative/data_info.json")))
    co_by_veh = {
        item["vehicle_image_path"].split("/")[-1].replace(".jpg", ""): item
        for item in co_info
    }
    frames = [co_by_veh[v] for v in val_ids if v in co_by_veh][:n_needed]

    paths = []
    for frame in frames:
        if mode == "pcd":
            ip = os.path.join(DAIR_ROOT, frame["infrastructure_pointcloud_path"])
            vp = os.path.join(DAIR_ROOT, frame["vehicle_pointcloud_path"])
        else:
            i_name = os.path.basename(frame["infrastructure_pointcloud_path"]).replace(".pcd", ".npy")
            v_name = os.path.basename(frame["vehicle_pointcloud_path"]).replace(".pcd", ".npy")
            ip = str(NPY_DIR / "infra" / i_name)
            vp = str(NPY_DIR / "veh"   / v_name)
        paths.append((ip, vp))
    return paths


# ─────────────────────────────────────────────────────────────────────────────
#  Single-frame inference with per-stage CUDA Event timing
# ─────────────────────────────────────────────────────────────────────────────

def make_events():
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    return s, e


def run_frame(model, pts_list, device):
    """
    Args:
        model:    full HeterPyramidCollab (loaded + on device)
        pts_list: list of numpy [N_i, 4] arrays (one per agent, in order [infra, veh])
    Returns:
        dict of per-stage latencies (ms) + metadata
    """
    B = len(pts_list)   # number of agents

    # ── Stage 0: Voxelize + H2D + batch collation ─────────────────────────────
    s0, e0 = make_events()
    torch.cuda.synchronize(device)
    s0.record()

    vox_list, coord_list, npts_list = [], [], []
    for pts_np in pts_list:
        pts_t = torch.from_numpy(pts_np).to(device)
        vox, coord, npts = voxelize_torch(pts_t, VOXEL_SIZE, PC_RANGE,
                                          max_pts=MAX_PTS, max_voxels=MAX_VOXELS)
        vox_list.append(vox)
        coord_list.append(coord)
        npts_list.append(npts)

    vox_cat  = torch.cat(vox_list,  dim=0)
    npts_cat = torch.cat(npts_list, dim=0)
    coord_parts = []
    for b, c in enumerate(coord_list):
        bcol = torch.full((c.shape[0], 1), b, dtype=torch.int32, device=device)
        coord_parts.append(torch.cat([bcol, c], dim=1))
    coords_cat = torch.cat(coord_parts, dim=0)

    e0.record()
    torch.cuda.synchronize(device)
    t_vox = s0.elapsed_time(e0)

    # ── Stage 1: pillar_vfe + scatter → spatial_features [B, 64, 256, 512] ───
    s1, e1 = make_events()
    s1.record()

    bd = {
        "voxel_features":   vox_cat,
        "voxel_num_points": npts_cat,
        "voxel_coords":     coords_cat,
    }
    bd = model.encoder_m1.pillar_vfe(bd)   # adds 'pillar_features'
    bd = model.encoder_m1.scatter(bd)      # adds 'spatial_features'

    e1.record()
    torch.cuda.synchronize(device)
    t_enc = s1.elapsed_time(e1)

    # ── Stage 2: backbone_m1 → spatial_features_2d [B, 64, 128, 256] ─────────
    s2, e2 = make_events()
    s2.record()

    bd2 = model.backbone_m1({"spatial_features": bd["spatial_features"]})
    spatial = bd2["spatial_features_2d"]   # [B_agents, 64, 128, 256]

    e2.record()
    torch.cuda.synchronize(device)
    t_bb = s2.elapsed_time(e2)

    # ── Stage 3: PyramidFusion → fused features ──────────────────────────────
    s3, e3 = make_events()
    s3.record()

    if B == 1:
        # Single-agent: no inter-agent fusion
        fused, _ = model.pyramid_backbone.forward_single(spatial)  # [1, 384, 128, 256]
    else:
        # Multi-agent collab: identity affine (all agents already in infra frame)
        record_len = torch.tensor([B], dtype=torch.long, device=device)
        aff = torch.zeros(1, B, B, 2, 3, device=device, dtype=torch.float32)
        aff[:, :, :, 0, 0] = 1.0   # identity 2×3 affine
        aff[:, :, :, 1, 1] = 1.0
        fused, _ = model.pyramid_backbone.forward_collab(
            spatial, record_len, aff, None, None
        )

    e3.record()
    torch.cuda.synchronize(device)
    t_fuse = s3.elapsed_time(e3)

    # ── Stage 4: shrink_conv + detection heads ────────────────────────────────
    s4, e4 = make_events()
    s4.record()

    # fused: [1, 384, H, W] (ego agent output)
    ego = fused[:1] if fused.shape[0] > 1 else fused
    if model.shrink_flag:
        ego = model.shrink_conv(ego)       # [1, 256, 128, 256]
    _ = model.cls_head(ego)
    _ = model.reg_head(ego)
    _ = model.dir_head(ego)

    e4.record()
    torch.cuda.synchronize(device)
    t_head = s4.elapsed_time(e4)

    t_e2e = t_vox + t_enc + t_bb + t_fuse + t_head

    return dict(
        t_voxelize_ms=t_vox,
        t_encoder_ms=t_enc,
        t_backbone_ms=t_bb,
        t_fusion_ms=t_fuse,
        t_head_ms=t_head,
        t_e2e_ms=t_e2e,
        n_voxels=int(vox_cat.shape[0]),
        batch_agents=B,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",     default="npy", choices=["npy", "pcd"])
    parser.add_argument("--n_frames", default=50,  type=int)
    parser.add_argument("--batch",    default=1,   type=int, choices=[1, 2])
    parser.add_argument("--warmup",   default=5,   type=int)
    parser.add_argument("--gpu",      default=0,   type=int)
    parser.add_argument("--out",      default="results_e2e.csv")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[m1_e2e_standalone]  device={device}  mode={args.mode}  "
          f"batch={args.batch}  n_frames={args.n_frames}  out={args.out}")

    if device.type == "cuda":
        import subprocess
        ns = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             f"--id={args.gpu}", "--format=csv,noheader,nounits"],
            text=True
        ).strip()
        util, mem = ns.split(", ")
        print(f"  GPU {args.gpu}: util={util}%  mem={mem}MB")
        if int(util) > 10:
            print("  ⚠️  GPU util >10% — latency may be unreliable!")

    # ── Load model ────────────────────────────────────────────────────────────
    model = load_heal_model(device)

    # ── Collect frame paths ───────────────────────────────────────────────────
    n_total = args.warmup + args.n_frames
    frame_paths = get_frame_paths(args.mode, n_total)
    load_fn = load_npy if args.mode == "npy" else load_pcd
    print(f"  {len(frame_paths)} frames available (need {n_total})")

    def get_pts(ip, vp):
        pts = [load_fn(ip)]
        if args.batch == 2:
            pts.append(load_fn(vp) if os.path.exists(vp) else pts[0].copy())
        return pts

    # ── Warmup ────────────────────────────────────────────────────────────────
    print(f"[m1_e2e_standalone] Warmup ({args.warmup} frames) …")
    with torch.no_grad():
        for ip, vp in frame_paths[:args.warmup]:
            if os.path.exists(ip):
                run_frame(model, get_pts(ip, vp), device)

    # ── Measured run ──────────────────────────────────────────────────────────
    print(f"[m1_e2e_standalone] Measuring {args.n_frames} frames …")
    results = []
    with torch.no_grad():
        for idx, (ip, vp) in enumerate(frame_paths[args.warmup:args.warmup + args.n_frames]):
            if not os.path.exists(ip):
                print(f"  [SKIP] missing {ip}")
                continue
            metrics = run_frame(model, get_pts(ip, vp), device)
            metrics["frame"] = os.path.basename(ip)
            results.append(metrics)
            if (idx + 1) % 10 == 0 or idx == 0:
                r = metrics
                print(f"  [{idx+1:3d}] e2e={r['t_e2e_ms']:6.2f}ms  "
                      f"vox={r['t_voxelize_ms']:.2f}  enc={r['t_encoder_ms']:.2f}  "
                      f"bb={r['t_backbone_ms']:.2f}  fuse={r['t_fusion_ms']:.2f}  "
                      f"head={r['t_head_ms']:.2f}  nvox={r['n_voxels']}")

    # ── Write CSV ─────────────────────────────────────────────────────────────
    if results:
        keys = list(results[0].keys())
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
        print(f"\n[m1_e2e_standalone] CSV → {args.out}")

    # ── Summary stats ─────────────────────────────────────────────────────────
    def ptile(lst, pct):
        s = sorted(lst)
        return s[min(int(len(s) * pct / 100), len(s) - 1)]

    stages = ["t_voxelize_ms", "t_encoder_ms", "t_backbone_ms",
              "t_fusion_ms", "t_head_ms", "t_e2e_ms"]
    print("\n" + "=" * 68)
    print(f"  Platform : {device}  (batch_agents={args.batch})")
    print(f"  Frames   : {len(results)}")
    print(f"  {'Stage':<24s}  {'mean':>7s}  {'p50':>7s}  {'p95':>7s}  {'min':>7s}")
    print("  " + "-" * 56)
    for st in stages:
        vals = [r[st] for r in results]
        print(f"  {st:<24s}  {statistics.mean(vals):7.2f}  "
              f"{ptile(vals,50):7.2f}  {ptile(vals,95):7.2f}  {min(vals):7.2f}")
    print("=" * 68)
    print("  [口径] CUDA Event, PyTorch full-chain, no NMS, no pcd file I/O")
    print("         Full e2e ≈ t_e2e + NMS (est +1–4ms on 4090, +4–8ms on Orin)")
    print("=" * 68)


if __name__ == "__main__":
    main()
