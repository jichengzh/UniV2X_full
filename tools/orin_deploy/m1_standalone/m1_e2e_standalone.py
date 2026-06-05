"""
Standalone end-to-end inference + CUDA Event timing for HEAL m1 (PyramidFusion/DAIR).
v2: adds NMS stage (t_nms_ms) + independent wallclock (t_e2e_wallclock_ms, t_glue_ms).

No spconv required for voxelization (uses voxelizer_torch.py).
Uses HEAL's create_model + load_saved_model for correct weight loading.

Pipeline per frame:
  voxelize_torch  →  encoder_m1.pillar_vfe  →  encoder_m1.scatter  →
  backbone_m1  →  pyramid_backbone  →  shrink_conv  →  cls/reg/dir heads  →  NMS

CUDA Event timing stages:
  t_voxelize_ms       — voxelize_torch + H2D copy + batch collation (CPU + GPU)
  t_encoder_ms        — pillar_vfe + scatter           →  [B_agents, 64, 256, 512]
  t_backbone_ms       — backbone_m1                    →  [B_agents, 64, 128, 256]
  t_fusion_ms         — pyramid_backbone                →  [1, 384, 128, 256]
  t_head_ms           — shrink_conv + cls/reg/dir heads
  t_nms_ms            — delta_to_boxes3d + dir correction + rotated NMS (GPU mmcv)
  t_e2e_ms            — Σ(vox+enc+bb+fuse+head+nms) — stage-sum derived
  t_e2e_wallclock_ms  — independent outer CUDA Event, vox-start to NMS-end
  t_glue_ms           — wallclock - Σstages = Python/kernel launch overhead

Usage:
  CUDA_VISIBLE_DEVICES=0 python m1_e2e_standalone.py --mode npy --batch 1 --n_frames 50
  CUDA_VISIBLE_DEVICES=0 python m1_e2e_standalone.py --mode npy --batch 2 --n_frames 50

Author: sw-optimizer (Task #12 v2, 2026-06-06)
Compat: Python 3.8+, PyTorch 1.12+
"""

import sys
import os
import math
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

# NMS / postprocess params (from m1_pyramid.yaml postprocess section)
SCORE_THRESH   = 0.2
NMS_THRESH     = 0.05
DIR_OFFSET     = 0.7853    # pi/4
NUM_BINS       = 2

# Anchor params (from yaml anchor_args)
ANCHOR_L, ANCHOR_W, ANCHOR_H = 3.9, 1.6, 1.56
ANCHOR_R       = [0.0, 90.0]    # degrees
FEATURE_STRIDE = 2
ANCHOR_NUM     = 2
ANCHOR_ORDER   = 'hwl'

# Grid dims
GRID_X = round((PC_RANGE[3] - PC_RANGE[0]) / VOXEL_SIZE[0])   # 512
GRID_Y = round((PC_RANGE[4] - PC_RANGE[1]) / VOXEL_SIZE[1])   # 256
VH, VW = VOXEL_SIZE[1], VOXEL_SIZE[0]                          # 0.4, 0.4


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
#  Anchor generation (pre-computed once, moved to GPU lazily inside NMS decode)
# ─────────────────────────────────────────────────────────────────────────────

def build_anchor_tensor():
    """
    Build anchor box tensor [H//fs, W//fs, anchor_num, 7] in hwl order.
    Matches VoxelPostprocessor.generate_anchor_box() output exactly.
    """
    r_rad = [math.radians(x) for x in ANCHOR_R]

    x = np.linspace(PC_RANGE[0] + VW, PC_RANGE[3] - VW,
                    GRID_X // FEATURE_STRIDE)
    y = np.linspace(PC_RANGE[1] + VH, PC_RANGE[4] - VH,
                    GRID_Y // FEATURE_STRIDE)

    cx, cy = np.meshgrid(x, y)                     # [H//fs, W//fs]
    cx = np.tile(cx[..., np.newaxis], ANCHOR_NUM)   # [H//fs, W//fs, A]
    cy = np.tile(cy[..., np.newaxis], ANCHOR_NUM)
    cz = np.ones_like(cx) * (-1.0)

    ww = np.ones_like(cx) * ANCHOR_W
    ll = np.ones_like(cx) * ANCHOR_L
    hh = np.ones_like(cx) * ANCHOR_H

    rr = np.ones_like(cx)
    for i in range(ANCHOR_NUM):
        rr[..., i] = r_rad[i]

    # hwl order: [cx, cy, cz, h, w, l, r]
    anchors = np.stack([cx, cy, cz, hh, ww, ll, rr], axis=-1)
    return torch.from_numpy(anchors.astype(np.float32))


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
            i_name = os.path.basename(
                frame["infrastructure_pointcloud_path"]).replace(".pcd", ".npy")
            v_name = os.path.basename(
                frame["vehicle_pointcloud_path"]).replace(".pcd", ".npy")
            ip = str(NPY_DIR / "infra" / i_name)
            vp = str(NPY_DIR / "veh"   / v_name)
        paths.append((ip, vp))
    return paths


# ─────────────────────────────────────────────────────────────────────────────
#  NMS decode helper (inline, CUDA-timed externally via CUDA Events)
# ─────────────────────────────────────────────────────────────────────────────

def _limit_period(val, offset, period):
    """Numerically equivalent to limit_period from opencood.utils.common_utils."""
    return val - torch.floor(val / period + offset) * period


def decode_and_nms(cls_pred, reg_pred, dir_pred, anchor_t, device):
    """
    Run anchor decode + direction correction + rotated-NMS for timing.

    Args:
        cls_pred: [1, A,    H, W]  — raw logits (pre-sigmoid)
        reg_pred: [1, A*7,  H, W]
        dir_pred: [1, A*nb, H, W]  — nb = NUM_BINS = 2
        anchor_t: [H//fs, W//fs, A, 7]  float32 CPU tensor (moved to GPU once)
        device:   torch.device

    Returns:
        n_dets: int  (post-NMS detection count)
    """
    from opencood.utils import box_utils
    from opencood.data_utils.post_processor.voxel_postprocessor import \
        VoxelPostprocessor

    # ── Score threshold ────────────────────────────────────────────────────
    prob = torch.sigmoid(cls_pred.permute(0, 2, 3, 1)).reshape(1, -1)
    mask = prob > SCORE_THRESH          # [1, H*W*A]

    # ── Box decode ─────────────────────────────────────────────────────────
    anchor_dev = anchor_t.to(device)
    batch_box3d = VoxelPostprocessor.delta_to_boxes3d(reg_pred, anchor_dev)
    boxes3d = batch_box3d[0][mask[0]]   # [N_cands, 7]
    scores  = prob[0][mask[0]]           # [N_cands]

    if boxes3d.shape[0] == 0:
        return 0

    # ── Direction correction ───────────────────────────────────────────────
    period   = math.pi
    dm       = dir_pred.permute(0, 2, 3, 1).contiguous().reshape(1, -1, NUM_BINS)
    dir_cls  = dm[0][mask[0]]           # [N_cands, NUM_BINS]
    dir_lbl  = dir_cls.argmax(-1).float()

    dir_rot  = _limit_period(boxes3d[:, 6] - DIR_OFFSET, 0.0, period)
    boxes3d  = boxes3d.clone()          # avoid in-place on non-leaf
    boxes3d[:, 6] = dir_rot + DIR_OFFSET + period * dir_lbl
    boxes3d[:, 6] = _limit_period(boxes3d[:, 6], 0.5, 2.0 * math.pi)

    # ── Rotated NMS (GPU via mmcv) ─────────────────────────────────────────
    corners3d = box_utils.boxes_to_corners_3d(boxes3d, order=ANCHOR_ORDER)
    keep      = box_utils.nms_rotated(corners3d, scores, NMS_THRESH)
    return len(keep)


# ─────────────────────────────────────────────────────────────────────────────
#  Single-frame inference with per-stage CUDA Event timing
# ─────────────────────────────────────────────────────────────────────────────

def make_events():
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    return s, e


def run_frame(model, pts_list, device, anchor_t):
    """
    Args:
        model:    full HeterPyramidCollab (loaded + on device)
        pts_list: list of numpy [N_i, 4] arrays
        anchor_t: [H//fs, W//fs, A, 7] float32 CPU tensor (pre-built once)
    Returns:
        dict of per-stage latencies (ms) + metadata
    """
    B = len(pts_list)

    # ── Outer wallclock: start BEFORE Stage 0 ─────────────────────────────
    s_total, e_total = make_events()
    torch.cuda.synchronize(device)
    s_total.record()

    # ── Stage 0: Voxelize + H2D + batch collation ─────────────────────────
    s0, e0 = make_events()
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

    # ── Stage 1: pillar_vfe + scatter → spatial_features [B, 64, 256, 512] ─
    s1, e1 = make_events()
    s1.record()

    bd = {
        "voxel_features":   vox_cat,
        "voxel_num_points": npts_cat,
        "voxel_coords":     coords_cat,
    }
    bd = model.encoder_m1.pillar_vfe(bd)
    bd = model.encoder_m1.scatter(bd)

    e1.record()
    torch.cuda.synchronize(device)
    t_enc = s1.elapsed_time(e1)

    # ── Stage 2: backbone_m1 → spatial_features_2d [B, 64, 128, 256] ──────
    s2, e2 = make_events()
    s2.record()

    bd2     = model.backbone_m1({"spatial_features": bd["spatial_features"]})
    spatial = bd2["spatial_features_2d"]

    e2.record()
    torch.cuda.synchronize(device)
    t_bb = s2.elapsed_time(e2)

    # ── Stage 3: PyramidFusion → fused features ───────────────────────────
    s3, e3 = make_events()
    s3.record()

    if B == 1:
        fused, _ = model.pyramid_backbone.forward_single(spatial)
    else:
        record_len = torch.tensor([B], dtype=torch.long, device=device)
        aff = torch.zeros(1, B, B, 2, 3, device=device, dtype=torch.float32)
        aff[:, :, :, 0, 0] = 1.0
        aff[:, :, :, 1, 1] = 1.0
        fused, _ = model.pyramid_backbone.forward_collab(
            spatial, record_len, aff, None, None
        )

    e3.record()
    torch.cuda.synchronize(device)
    t_fuse = s3.elapsed_time(e3)

    # ── Stage 4: shrink_conv + detection heads ────────────────────────────
    s4, e4 = make_events()
    s4.record()

    ego = fused[:1] if fused.shape[0] > 1 else fused
    if model.shrink_flag:
        ego = model.shrink_conv(ego)
    cls_pred = model.cls_head(ego)    # [1, A,    H, W]
    reg_pred = model.reg_head(ego)    # [1, A*7,  H, W]
    dir_pred = model.dir_head(ego)    # [1, A*nb, H, W]

    e4.record()
    torch.cuda.synchronize(device)
    t_head = s4.elapsed_time(e4)

    # ── Stage 5: NMS — anchor decode + dir correction + rotated NMS ────────
    s5, e5 = make_events()
    s5.record()

    n_dets = decode_and_nms(cls_pred, reg_pred, dir_pred, anchor_t, device)

    e5.record()

    # ── Outer wallclock: stop AFTER Stage 5 ───────────────────────────────
    e_total.record()
    torch.cuda.synchronize(device)

    t_nms  = s5.elapsed_time(e5)
    t_wc   = s_total.elapsed_time(e_total)
    t_sum  = t_vox + t_enc + t_bb + t_fuse + t_head + t_nms

    return dict(
        t_voxelize_ms       = t_vox,
        t_encoder_ms        = t_enc,
        t_backbone_ms       = t_bb,
        t_fusion_ms         = t_fuse,
        t_head_ms           = t_head,
        t_nms_ms            = t_nms,
        t_e2e_ms            = t_sum,          # Σstages (breakdown reference)
        t_e2e_wallclock_ms  = t_wc,           # independent outer measurement
        t_glue_ms           = t_wc - t_sum,   # Python/kernel launch overhead
        n_voxels            = int(vox_cat.shape[0]),
        n_dets              = n_dets,
        batch_agents        = B,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",     default="npy", choices=["npy", "pcd"])
    parser.add_argument("--n_frames", default=50,  type=int)
    parser.add_argument("--batch",    default=1,   type=int, choices=[1, 2])
    parser.add_argument("--warmup",   default=10,  type=int,
                        help="Warmup frames excluded from stats (default 10)")
    parser.add_argument("--gpu",      default=0,   type=int)
    parser.add_argument("--out",      default="results_e2e.csv")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[m1_e2e_standalone v2]  device={device}  mode={args.mode}  "
          f"batch={args.batch}  n_frames={args.n_frames}  warmup={args.warmup}")
    print(f"  out={args.out}")

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

    # ── Load model ────────────────────────────────────────────────────────
    model = load_heal_model(device)

    # ── Pre-build anchor tensor (CPU → GPU lazily in NMS decode) ──────────
    anchor_t = build_anchor_tensor()
    A_H, A_W = anchor_t.shape[:2]
    print(f"  Anchors: [{A_H}, {A_W}, {ANCHOR_NUM}, 7]  "
          f"(H//fs={A_H}, W//fs={A_W}, A={ANCHOR_NUM})")

    # ── Collect frame paths ───────────────────────────────────────────────
    n_total = args.warmup + args.n_frames
    frame_paths = get_frame_paths(args.mode, n_total)
    load_fn = load_npy if args.mode == "npy" else load_pcd
    print(f"  {len(frame_paths)} frames available (need {n_total})")

    def get_pts(ip, vp):
        pts = [load_fn(ip)]
        if args.batch == 2:
            pts.append(load_fn(vp) if os.path.exists(vp) else pts[0].copy())
        return pts

    # ── Warmup ────────────────────────────────────────────────────────────
    print(f"[m1_e2e_standalone v2] Warmup ({args.warmup} frames) …")
    with torch.no_grad():
        for ip, vp in frame_paths[:args.warmup]:
            if os.path.exists(ip):
                run_frame(model, get_pts(ip, vp), device, anchor_t)

    # ── Measured run ──────────────────────────────────────────────────────
    print(f"[m1_e2e_standalone v2] Measuring {args.n_frames} frames …")
    results = []
    with torch.no_grad():
        for idx, (ip, vp) in enumerate(
                frame_paths[args.warmup:args.warmup + args.n_frames]):
            if not os.path.exists(ip):
                print(f"  [SKIP] missing {ip}")
                continue
            metrics = run_frame(model, get_pts(ip, vp), device, anchor_t)
            metrics["frame"] = os.path.basename(ip)
            results.append(metrics)
            if (idx + 1) % 10 == 0 or idx == 0:
                r = metrics
                print(
                    f"  [{idx+1:3d}] wc={r['t_e2e_wallclock_ms']:6.2f}ms "
                    f"sum={r['t_e2e_ms']:6.2f}ms glue={r['t_glue_ms']:5.2f}ms | "
                    f"vox={r['t_voxelize_ms']:.2f} enc={r['t_encoder_ms']:.2f} "
                    f"bb={r['t_backbone_ms']:.2f} fuse={r['t_fusion_ms']:.2f} "
                    f"head={r['t_head_ms']:.2f} nms={r['t_nms_ms']:.2f} "
                    f"dets={r['n_dets']} nvox={r['n_voxels']}"
                )

    # ── Write CSV ─────────────────────────────────────────────────────────
    if results:
        keys = list(results[0].keys())
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
        print(f"\n[m1_e2e_standalone v2] CSV → {args.out}")

    # ── Summary stats ─────────────────────────────────────────────────────
    def ptile(lst, pct):
        s = sorted(lst)
        return s[min(int(len(s) * pct / 100), len(s) - 1)]

    stages = ["t_voxelize_ms",  "t_encoder_ms",    "t_backbone_ms",
              "t_fusion_ms",    "t_head_ms",        "t_nms_ms",
              "t_e2e_ms",       "t_e2e_wallclock_ms", "t_glue_ms"]

    print("\n" + "=" * 76)
    print(f"  Platform : {device}  (batch_agents={args.batch})")
    print(f"  Frames   : {len(results)}")
    print(f"  {'Stage':<30s}  {'mean':>7s}  {'p50':>7s}  {'p95':>7s}  {'min':>7s}")
    print("  " + "-" * 62)
    for st in stages:
        vals = [r[st] for r in results]
        print(f"  {st:<30s}  {statistics.mean(vals):7.2f}  "
              f"{ptile(vals, 50):7.2f}  {ptile(vals, 95):7.2f}  {min(vals):7.2f}")
    print("=" * 76)
    print("  [口径 v2] CUDA Event, PyTorch + mmcv rotated-NMS full chain, no pcd I/O")
    print("  t_e2e_wallclock_ms = independent outer pair (authoritative e2e latency)")
    print("  t_e2e_ms           = Σstages (stage breakdown; matches wallclock closely)")
    print("  t_glue_ms          = wallclock − Σstages (Python & kernel-launch overhead)")
    print("=" * 76)


if __name__ == "__main__":
    main()
