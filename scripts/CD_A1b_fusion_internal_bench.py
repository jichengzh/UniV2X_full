"""
CD-A1b: CoDriving fusion_net 内部分段 CUDA Event 计时
插桩位置:
  - backbone.resnet(x)          L264 codriving_attn.py  → segment: fusion_backbone_resnet
  - 每 scale 的 warp_affine     L298                    → segment: fusion_warp (累加)
  - 每 scale 的 fuse_modules[i] L301 AttenFusion(bmm)   → segment: fusion_atten (累加)
  - 每 scale 的 deblocks[i]     L309                    → segment: fusion_deblock (累加)
  - fusion_net 整体(含上述全部)                         → segment: fusion_net_total

口径: max_cav=2, dense BEV [2,64,192,576], FP32, GPU 2
warmup=50, measure=200
真测: CUDA Event
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import sys
sys.path.insert(0, "/home/jichengzhi/V2Xverse")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import csv
import statistics

from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple
from opencood.models.fuse_modules.codriving_attn import AttenFusion, CoDriving

# ---- verify GPU is idle ----
print("GPU state check:")
import subprocess
result = subprocess.run(
    ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
     "--format=csv,noheader", "--id=2"],
    capture_output=True, text=True
)
print(result.stdout.strip())

device = torch.device("cuda:0")  # CUDA_VISIBLE_DEVICES=2 → remapped to cuda:0

# ---- build minimal backbone that matches CoDriving config ----
# CoDriving uses backbone.resnet + backbone.deblocks
# Matching config from center_point_codriving yaml (CoDriving standard):
# layer_nums: [3, 5, 8]  num_filters: [64, 128, 256]  num_upsample_filter: [128, 128, 128]
# This matches ResNetBEVBackbone; input channels=64
backbone_args = {
    "resnet": True,
    "layer_nums": [3, 4, 5],
    "layer_strides": [2, 2, 2],
    "num_filters": [64, 128, 256],
    "upsample_strides": [1, 2, 4],
    "num_upsample_filter": [128, 128, 128],
}
backbone = ResNetBEVBackbone(backbone_args, 64).to(device).eval()

# ---- fusion_net config (from codriving_multiclass_config.yaml) ----
fusion_args = {
    "voxel_size": [0.4, 0.4, 4],
    "downsample_rate": 1,
    "multi_scale": True,
    "layer_nums": [3, 4, 5],
    "num_filters": [64, 128, 256],
    "agg_operator": {
        "mode": "ATTEN",
        "feature_dim": 256,
    },
    # no communication key => no Communication module
}
fusion_net = CoDriving(fusion_args).to(device).eval()

# ---- check backbone has .resnet attribute ----
print(f"backbone has resnet: {hasattr(backbone, 'resnet')}")
print(f"backbone.deblocks length: {len(backbone.deblocks)}")
print(f"fusion_net.num_levels: {fusion_net.num_levels}")
print(f"fusion_net.fuse_modules: {len(fusion_net.fuse_modules)} AttenFusion modules")

# ---- dummy inputs ----
# max_cav=2, BN=2, C=64, H=192, W=576
BN = 2  # sum of cavs (batch=1, N_cav=2)
spatial_features = torch.randn(BN, 64, 192, 576, device=device)
psm_single = torch.randn(BN, 2, 192, 576, device=device)  # cls head output
record_len = torch.tensor([2], dtype=torch.long, device=device)
# pairwise_t_matrix: [B=1, L=2, L=2, 4, 4]
pairwise_t_matrix = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).unsqueeze(0)
pairwise_t_matrix = pairwise_t_matrix.expand(1, 2, 2, 4, 4).clone()

# ---- CUDA Event timer utility ----
def cuda_time_ms(fn, warmup=50, measure=200):
    """Returns list of ms timings."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(measure):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return times

# ============================================================
# 1. fusion_net整体 (re-use existing CoDriving.forward as-is)
# ============================================================
def run_fusion_net():
    with torch.no_grad():
        fusion_net(spatial_features, psm_single, record_len, pairwise_t_matrix,
                   backbone=backbone, waypoints=None)

print("\nTiming fusion_net_total ...")
t_fusion = cuda_time_ms(run_fusion_net)
fusion_mean = statistics.mean(t_fusion)
fusion_p50  = statistics.median(t_fusion)
fusion_p99  = np.percentile(t_fusion, 99)
print(f"  fusion_net_total: mean={fusion_mean:.3f} p50={fusion_p50:.3f} p99={fusion_p99:.3f} ms")

# ============================================================
# 2. backbone.resnet only (重跑 resnet on same input)
# ============================================================
def run_backbone_resnet():
    with torch.no_grad():
        backbone.resnet(spatial_features)

print("Timing fusion_backbone_resnet (backbone.resnet on [2,64,192,576]) ...")
t_bb_resnet = cuda_time_ms(run_backbone_resnet)
bb_resnet_mean = statistics.mean(t_bb_resnet)
bb_resnet_p50  = statistics.median(t_bb_resnet)
bb_resnet_p99  = np.percentile(t_bb_resnet, 99)
print(f"  fusion_backbone_resnet: mean={bb_resnet_mean:.3f} p50={bb_resnet_p50:.3f} p99={bb_resnet_p99:.3f} ms")

# ============================================================
# 3. warp_affine_simple per scale (use scale 0: [2,64,96,288])
# ============================================================
# We need to get the actual feature shapes at each scale to time warp
# Run a forward pass to capture shapes
warp_fn = warp_affine_simple

# Build t_matrix for warp: [N=2, 2, 3]
# pairwise_t_matrix processed inside CoDriving.forward:
t_matrix_raw = pairwise_t_matrix.clone()  # [1,2,2,4,4]
t_matrix_proc = t_matrix_raw[:,:,:,[0,1],:][:,:,:,:,[0,1,3]]  # [1,2,2,2,3]
# Scale corrections will depend on H,W per level; do approximate using scale-0 dims
# feats from backbone.resnet: ([2,64,96,288],[2,128,48,144],[2,256,24,72])
with torch.no_grad():
    feats = backbone.resnet(spatial_features)

scale_infos = []
for lvl, feat in enumerate(feats):
    C, H, W = feat.shape[1], feat.shape[2], feat.shape[3]
    scale_infos.append((lvl, C, H, W, feat))
    print(f"  scale {lvl}: {feat.shape}")

# downsample_rate=2, discrete_ratio=0.4 (voxel_size[0])
dr = 2
dc = 0.4
t_warp_total = []
t_atten_total = []
t_deblock_total = []

for lvl, C, H, W, feat in scale_infos:
    # Build corrected t_matrix for this level
    tm = t_matrix_proc.clone()  # [1,2,2,2,3]
    tm[...,0,1] = tm[...,0,1] * H / W
    tm[...,1,0] = tm[...,1,0] * W / H
    tm[...,0,2] = tm[...,0,2] / (dr * dc * W) * 2
    tm[...,1,2] = tm[...,1,2] / (dr * dc * H) * 2
    t_mat_b0 = tm[0, :2, :2, :, :]  # [2,2,2,3]
    t_mat_b0_ego = t_mat_b0[0, :, :, :]  # [2, 2, 3] ego's row

    node_features = feat  # [2, C, H, W]

    # warp_affine_simple expects node_features [N,C,H,W], t_matrix [N,2,3]
    def run_warp(nf=node_features, tm_ego=t_mat_b0_ego, h=H, w=W):
        with torch.no_grad():
            warp_fn(nf, tm_ego, (h, w))

    print(f"  Timing warp scale {lvl} ({H}x{W}) ...")
    tw = cuda_time_ms(run_warp)
    t_warp_total.append(statistics.mean(tw))
    print(f"    warp scale {lvl}: mean={statistics.mean(tw):.3f} p50={statistics.median(tw):.3f} ms")

    # AttenFusion: input neighbor_feature after warp, shape [2, C, H, W]
    with torch.no_grad():
        neighbor_feature = warp_fn(node_features, t_mat_b0_ego, (H, W))

    fuse_mod = fusion_net.fuse_modules[lvl]

    def run_atten(nf=neighbor_feature, fm=fuse_mod):
        with torch.no_grad():
            fm(nf)

    print(f"  Timing AttenFusion scale {lvl} ({H}x{W}) ...")
    ta = cuda_time_ms(run_atten)
    t_atten_total.append(statistics.mean(ta))
    print(f"    atten scale {lvl}: mean={statistics.mean(ta):.3f} p50={statistics.median(ta):.3f} ms")

    # deblocks[i] upsample: input x_fuse shape [B=1, C, H, W]
    x_fuse_sample = fuse_mod(neighbor_feature).unsqueeze(0)  # [1, C, H, W]
    deblock = backbone.deblocks[lvl]

    def run_deblock(xf=x_fuse_sample, db=deblock):
        with torch.no_grad():
            db(xf)

    print(f"  Timing deblock scale {lvl} ({H}x{W} -> upsample) ...")
    td = cuda_time_ms(run_deblock)
    t_deblock_total.append(statistics.mean(td))
    print(f"    deblock scale {lvl}: mean={statistics.mean(td):.3f} p50={statistics.median(td):.3f} ms")

# Final deblock (backbone.deblocks[-1] if num_levels < len(deblocks))
extra_deblock_mean = 0.0
if len(backbone.deblocks) > fusion_net.num_levels:
    # Simulate: after cat of 3 ups -> [1, 384, 96, 288] (3x128=384)
    H0, W0 = scale_infos[0][2], scale_infos[0][3]  # largest scale spatial dims
    x_fuse_cat = torch.randn(1, 384, H0, W0, device=device)
    extra_db = backbone.deblocks[-1]

    def run_extra_deblock(xf=x_fuse_cat, db=extra_db):
        with torch.no_grad():
            db(xf)

    print("  Timing extra final deblock ...")
    ted = cuda_time_ms(run_extra_deblock)
    extra_deblock_mean = statistics.mean(ted)
    print(f"    extra_deblock: mean={extra_deblock_mean:.3f} ms")

# ============================================================
# 4. Summary
# ============================================================
warp_sum   = sum(t_warp_total)
atten_sum  = sum(t_atten_total)
deblock_sum = sum(t_deblock_total) + extra_deblock_mean

print("\n======= CD-A1b SUMMARY =======")
print(f"fusion_net_total           : {fusion_mean:.3f} ms  (CUDA Event, warmup=50, n=200)")
print(f"  fusion_backbone_resnet   : {bb_resnet_mean:.3f} ms  ({bb_resnet_mean/fusion_mean*100:.1f}%)")
print(f"  fusion_warp (all scales) : {warp_sum:.3f} ms  ({warp_sum/fusion_mean*100:.1f}%)")
print(f"  fusion_atten (all scales): {atten_sum:.3f} ms  ({atten_sum/fusion_mean*100:.1f}%)")
print(f"  fusion_deblock(all+extra): {deblock_sum:.3f} ms  ({deblock_sum/fusion_mean*100:.1f}%)")
overhead = fusion_mean - bb_resnet_mean - warp_sum - atten_sum - deblock_sum
print(f"  overhead/other           : {overhead:.3f} ms  ({overhead/fusion_mean*100:.1f}%)")

# CD-A1 reference values
backbone_main_path_mean = 2.807  # backbone.resnet from CD-A1 (main path, same resnet)
e2e_mean = 10.166  # perception e2e from CD-A1

print(f"\n--- Lever analysis ---")
print(f"backbone.resnet usage:")
print(f"  main path (CD-A1):  {backbone_main_path_mean:.3f} ms")
print(f"  fusion internal:    {bb_resnet_mean:.3f} ms")
print(f"  total resnet:       {backbone_main_path_mean + bb_resnet_mean:.3f} ms")
print(f"  share of e2e:       {(backbone_main_path_mean + bb_resnet_mean)/e2e_mean*100:.1f}%")
print(f"  pruning lever (1x backbone prune -> reduce both): ~{(backbone_main_path_mean + bb_resnet_mean)/e2e_mean:.2f}x e2e impact factor")

# ============================================================
# 5. Write CSV
# ============================================================
csv_path = "/home/jichengzhi/V2X/results/CD_A1b_fusion_internal_4090.csv"

rows = []
rows.append({
    "module": "fusion_net_total",
    "scope": "fusion_internal",
    "mean_ms": f"{fusion_mean:.3f}",
    "p50_ms": f"{fusion_p50:.3f}",
    "p99_ms": f"{fusion_p99:.3f}",
    "caveat": "CUDA Event FP32, max_cav=2 BN=2 [2,64,192,576], warmup=50 n=200, GPU2 RTX4090"
})
rows.append({
    "module": "fusion_backbone_resnet",
    "scope": "fusion_internal",
    "mean_ms": f"{bb_resnet_mean:.3f}",
    "p50_ms": f"{bb_resnet_p50:.3f}",
    "p99_ms": f"{bb_resnet_p99:.3f}",
    "caveat": "backbone.resnet(x) inside CoDriving.forward L264, same input [2,64,192,576]"
})

for lvl, w_ms, a_ms, d_ms in zip(range(fusion_net.num_levels), t_warp_total, t_atten_total, t_deblock_total):
    H, W = scale_infos[lvl][2], scale_infos[lvl][3]
    rows.append({
        "module": f"fusion_warp_scale{lvl}",
        "scope": "fusion_internal",
        "mean_ms": f"{w_ms:.3f}",
        "p50_ms": "N/A",
        "p99_ms": "N/A",
        "caveat": f"warp_affine_simple scale{lvl} [{BN},{scale_infos[lvl][1]},{H},{W}]"
    })
    rows.append({
        "module": f"fusion_atten_scale{lvl}",
        "scope": "fusion_internal",
        "mean_ms": f"{a_ms:.3f}",
        "p50_ms": "N/A",
        "p99_ms": "N/A",
        "caveat": f"AttenFusion(bmm+softmax) scale{lvl} [{BN},{scale_infos[lvl][1]},{H},{W}]"
    })
    rows.append({
        "module": f"fusion_deblock_scale{lvl}",
        "scope": "fusion_internal",
        "mean_ms": f"{d_ms:.3f}",
        "p50_ms": "N/A",
        "p99_ms": "N/A",
        "caveat": f"backbone.deblocks[{lvl}] upsample after fuse, input [1,{scale_infos[lvl][1]},{H},{W}]"
    })

if extra_deblock_mean > 0:
    rows.append({
        "module": "fusion_deblock_extra",
        "scope": "fusion_internal",
        "mean_ms": f"{extra_deblock_mean:.3f}",
        "p50_ms": "N/A",
        "p99_ms": "N/A",
        "caveat": f"backbone.deblocks[-1] final concat deconv, input [1,384,{scale_infos[0][2]},{scale_infos[0][3]}]"
    })

rows.append({
    "module": "fusion_warp_all_scales",
    "scope": "fusion_internal",
    "mean_ms": f"{warp_sum:.3f}",
    "p50_ms": "N/A",
    "p99_ms": "N/A",
    "caveat": f"sum of warp_affine_simple across {fusion_net.num_levels} scales"
})
rows.append({
    "module": "fusion_atten_all_scales",
    "scope": "fusion_internal",
    "mean_ms": f"{atten_sum:.3f}",
    "p50_ms": "N/A",
    "p99_ms": "N/A",
    "caveat": f"sum of AttenFusion across {fusion_net.num_levels} scales"
})
rows.append({
    "module": "fusion_deblock_all",
    "scope": "fusion_internal",
    "mean_ms": f"{deblock_sum:.3f}",
    "p50_ms": "N/A",
    "p99_ms": "N/A",
    "caveat": f"sum of all deblocks in fusion path (incl. extra final)"
})
rows.append({
    "module": "fusion_pure_attn_overhead",
    "scope": "fusion_internal",
    "mean_ms": f"{warp_sum + atten_sum:.3f}",
    "p50_ms": "N/A",
    "p99_ms": "N/A",
    "caveat": "warp+AttenFusion, pure attention overhead (no backbone, no deblock)"
})

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["module","scope","mean_ms","p50_ms","p99_ms","caveat"])
    writer.writeheader()
    writer.writerows(rows)

print(f"\nResults written to {csv_path}")
print("Done.")
