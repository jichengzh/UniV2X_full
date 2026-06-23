"""
Pyramid m2 (Camera/LSS) DAIR structure profiling.
口径: fp32_pytorch CUDA-Event, DAIR camera_pyramid.yaml, random weights, dummy 4-cam input
授权: HANDOFF §二 2.3 预授权「profiling 微跑(随机权重+dummy 4-cam 合法, 口径标
      "fp32_pytorch hook-level random-weight")【此微跑同样已预授权, 引用本句】」
GPU: cuda:0 (util=0%, mem=1MiB confirmed)
"""
import sys, os, json, torch
sys.path.insert(0, '/home/jichengzhi/heal_research/HEAL')
os.chdir('/home/jichengzhi/heal_research/HEAL')

DEVICE = 'cuda:0'
WARMUP = 10
MEASURE = 50  # fewer iterations since LSS is heavier

from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.tools.train_utils import create_model

cfg = load_yaml('opencood/hypes_yaml/dairv2x/CameraOnly/camera_pyramid.yaml')
model = create_model(cfg).to(DEVICE)
model.eval()

# ---- Print param counts ----
total = sum(p.numel() for p in model.parameters())
print(f"Total params: {total:,} ({total/1e6:.3f}M)")
for name, module in model.named_children():
    p = sum(x.numel() for x in module.parameters())
    print(f"  {name}: {p:,} ({p/total*100:.2f}%)")

# camencode breakdown
cam_enc = model.encoder_m2.camencode
print("\n  camencode breakdown:")
for name, sub in cam_enc.named_children():
    p = sum(x.numel() for x in sub.parameters())
    print(f"    .{name}: {type(sub).__name__}  params={p:,}  ({p/total*100:.2f}% total)")

# ---- Grid conf (hardcoded from yaml: camera_pyramid.yaml) ----
# xbound: [-102.4, 102.4, 0.4]  → BEV W = 512
# ybound: [-51.2, 51.2, 0.4]    → BEV H = 256
# ddiscr: [2, 100, 98]           → D = 98 depth bins
# final_dim: [288, 512]          → cam H=288, W=512
# Ncams: 4
N_CAM = 4
IMG_H, IMG_W = 288, 512
BEV_H, BEV_W = 256, 512  # (ybound range / step, xbound range / step)
D = 98                     # depth bins
B = 2                      # 2 agents (DAIR V2I scenario)

print(f"\nGrid: BEV ({BEV_H}x{BEV_W}), depth bins={D}, cams={N_CAM}, img=({IMG_H}x{IMG_W})")

# ---- Dummy camera inputs ----
imgs      = torch.randn(B, N_CAM, 3, IMG_H, IMG_W, device=DEVICE)
rots      = torch.eye(3, device=DEVICE).unsqueeze(0).unsqueeze(0).expand(B, N_CAM, -1, -1)
trans     = torch.zeros(B, N_CAM, 3, device=DEVICE)
intrins   = torch.eye(3, device=DEVICE).unsqueeze(0).unsqueeze(0).expand(B, N_CAM, -1, -1)
# Reasonable focal length to avoid degenerate projection
intrins   = intrins.clone()
intrins[:, :, 0, 0] = 400.0  # fx
intrins[:, :, 1, 1] = 400.0  # fy
intrins[:, :, 0, 2] = IMG_W / 2  # cx
intrins[:, :, 1, 2] = IMG_H / 2  # cy
post_rots  = torch.eye(3, device=DEVICE).unsqueeze(0).unsqueeze(0).expand(B, N_CAM, -1, -1)
post_trans = torch.zeros(B, N_CAM, 3, device=DEVICE)

data_dict = {
    'inputs_m2': {
        'imgs':      imgs,
        'rots':      rots,
        'trans':     trans,
        'intrins':   intrins,
        'post_rots': post_rots,
        'post_trans': post_trans,
    }
}

# ---- Timer utility ----
def timed(fn, warmup=WARMUP, measure=MEASURE):
    """CUDA Event timer. Returns mean_ms."""
    with torch.no_grad():
        for _ in range(warmup):
            try:
                fn()
            except Exception:
                pass
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(measure):
            fn()
        e.record()
        torch.cuda.synchronize()
    return s.elapsed_time(e) / measure

# ---- Timing: full encoder_m2 (get_voxels = camencode + voxel_pooling) ----
print("\nProfiling encoder_m2.forward (get_voxels)...")
try:
    def fn_encoder():
        model.encoder_m2(data_dict, 'm2')
    t_encoder = timed(fn_encoder)
    print(f"encoder_m2 (LSS full):   {t_encoder:.4f} ms  input=({B},{N_CAM},3,{IMG_H},{IMG_W})")
except Exception as ex:
    print(f"encoder_m2 FAILED: {ex}")
    t_encoder = None

# ---- Timing: camencode only (EfficientNet + FPN + depth/image head) ----
# camencode.forward takes just images and camera params
def fn_camencode():
    with torch.no_grad():
        model.encoder_m2.camencode(imgs, rots, trans, intrins, post_rots, post_trans)

try:
    t_cam = timed(fn_camencode)
    print(f"encoder_m2.camencode:    {t_cam:.4f} ms  (EfficientNet+FPN+heads)")
except Exception as ex:
    print(f"camencode FAILED: {ex}")
    # Try an alternative signature
    try:
        def fn_camencode2():
            model.encoder_m2.camencode(imgs)
        t_cam = timed(fn_camencode2)
        print(f"encoder_m2.camencode(imgs only): {t_cam:.4f} ms")
    except Exception as ex2:
        print(f"camencode(imgs only) FAILED: {ex2}")
        t_cam = None

# ---- Timing: backbone_m2 ----
# backbone_m2 takes BEV pseudo-image
# First, get the BEV output from encoder
try:
    with torch.no_grad():
        bev = model.encoder_m2(data_dict, 'm2')
    if isinstance(bev, dict):
        bev_tensor = bev.get('spatial_features', bev.get('spatial_features_2d', list(bev.values())[0]))
    else:
        bev_tensor = bev
    print(f"\nencoder_m2 output shape: {bev_tensor.shape}")

    def fn_bb():
        model.backbone_m2({'spatial_features': bev_tensor})
    t_bb = timed(fn_bb)
    with torch.no_grad():
        bb_out = model.backbone_m2({'spatial_features': bev_tensor})
    bb_feat = bb_out.get('spatial_features_2d', bb_out.get('spatial_features'))
    print(f"backbone_m2:             {t_bb:.4f} ms  input={list(bev_tensor.shape)}")
    print(f"backbone_m2 output:      {list(bb_feat.shape)}")
except Exception as ex:
    print(f"backbone_m2 FAILED: {ex}")
    import traceback; traceback.print_exc()
    t_bb = None
    bb_feat = None

# ---- Timing: pyramid_backbone (shared with m1) ----
if bb_feat is not None:
    record_len = torch.tensor([2], device=DEVICE)
    affine_matrix = torch.zeros(1, 2, 2, 2, 3, device=DEVICE)
    affine_matrix[0, :, :, 0, 0] = 1.0
    affine_matrix[0, :, :, 1, 1] = 1.0
    agent_modality_list = ['m2', 'm2']
    try:
        def fn_pb():
            model.pyramid_backbone.forward_collab(
                bb_feat, record_len, affine_matrix, agent_modality_list, None
            )
        t_pb = timed(fn_pb)
        with torch.no_grad():
            fused_feat, _ = model.pyramid_backbone.forward_collab(
                bb_feat, record_len, affine_matrix, agent_modality_list, None
            )
        print(f"pyramid_backbone:        {t_pb:.4f} ms  input={list(bb_feat.shape)}")
        print(f"pyramid_backbone output: {list(fused_feat.shape)}")

        # ---- Timing: shrink_conv ----
        def fn_sc():
            model.shrink_conv(fused_feat)
        t_sc = timed(fn_sc)
        with torch.no_grad():
            shrunk = model.shrink_conv(fused_feat)
        print(f"shrink_conv:             {t_sc:.4f} ms  input={list(fused_feat.shape)}")
        print(f"shrink_conv output:      {list(shrunk.shape)}")

        # ---- Timing: heads ----
        def fn_heads():
            model.cls_head(shrunk)
            model.reg_head(shrunk)
            model.dir_head(shrunk)
        t_heads = timed(fn_heads)
        print(f"heads (cls+reg+dir):     {t_heads:.4f} ms  input={list(shrunk.shape)}")

    except Exception as ex:
        print(f"pyramid_backbone FAILED: {ex}")
        import traceback; traceback.print_exc()
        t_pb = t_sc = t_heads = None
        fused_feat = shrunk = None
else:
    t_pb = t_sc = t_heads = None
    fused_feat = shrunk = None

# ---- Full body chain (encoder+bb+pb+sc+heads) ----
print("\nProfiling full body chain...")
try:
    def fn_full():
        bev2 = model.encoder_m2(data_dict, 'm2')
        if isinstance(bev2, dict):
            bev2t = bev2.get('spatial_features', bev2.get('spatial_features_2d', list(bev2.values())[0]))
        else:
            bev2t = bev2
        bb2 = model.backbone_m2({'spatial_features': bev2t})
        hf2 = bb2.get('spatial_features_2d', bb2.get('spatial_features'))
        fused2, _ = model.pyramid_backbone.forward_collab(
            hf2, record_len, affine_matrix, agent_modality_list, None
        )
        sc2 = model.shrink_conv(fused2)
        model.cls_head(sc2); model.reg_head(sc2); model.dir_head(sc2)
    t_body = timed(fn_full)
    print(f"full body chain:         {t_body:.4f} ms  (enc+bb+pb+sc+heads)")
except Exception as ex:
    print(f"full body FAILED: {ex}")
    t_body = None

# ---- Summary ----
print("\n=== PYRAMID M2 (CAMERA/LSS) SUBMODULE TIMING ===")
print(f"[口径] fp32_pytorch CUDA-Event, DAIR dummy input (record_len=2, 4-cam), random weights, GPU0")
print(f"  encoder_m2 (LSS full):    {t_encoder if t_encoder else 'N/A':.4f} ms" if t_encoder else "  encoder_m2: N/A")
print(f"    .camencode:             {t_cam if t_cam else 'N/A':.4f} ms" if t_cam else "    .camencode: N/A")
if t_encoder and t_cam:
    print(f"    .voxel_pooling (derived): {t_encoder-t_cam:.4f} ms")
print(f"  backbone_m2:              {t_bb if t_bb else 'N/A':.4f} ms" if t_bb else "  backbone_m2: N/A")
print(f"  pyramid_backbone:         {t_pb if t_pb else 'N/A':.4f} ms" if t_pb else "  pyramid_backbone: N/A")
print(f"  shrink_conv:              {t_sc if t_sc else 'N/A':.4f} ms" if t_sc else "  shrink_conv: N/A")
print(f"  heads:                    {t_heads if t_heads else 'N/A':.4f} ms" if t_heads else "  heads: N/A")
print(f"  full body chain:          {t_body if t_body else 'N/A':.4f} ms" if t_body else "  full body: N/A")

# ---- Save ----
out = {
    'metadata': {
        'model': 'PyramidFusion_m2_Camera_DAIR',
        'device': DEVICE,
        'precision': 'fp32',
        'framework': 'pytorch_cuda_event',
        'ckpt': 'random_weights',
        'warmup': WARMUP,
        'measure': MEASURE,
        'input': f'DAIR_dummy B={B} N_cam={N_CAM} H={IMG_H} W={IMG_W}',
        'bev_grid': f'H={BEV_H} W={BEV_W} D={D}',
        'latency_kind': 'submodule_fp32_pytorch_cuda_event_random_weight',
        'authorization': 'HANDOFF §2.3 pre-authorized profiling micro-run',
        'note': 'NMS excluded (CPU); AP not measurable (no DAIR camera ckpt)'
    },
    'encoder_m2_lss_full_mean_ms': round(t_encoder, 4) if t_encoder else None,
    'camencode_mean_ms': round(t_cam, 4) if t_cam else None,
    'voxel_pooling_derived_ms': round(t_encoder - t_cam, 4) if (t_encoder and t_cam) else None,
    'backbone_m2_mean_ms': round(t_bb, 4) if t_bb else None,
    'pyramid_backbone_mean_ms': round(t_pb, 4) if t_pb else None,
    'shrink_conv_mean_ms': round(t_sc, 4) if t_sc else None,
    'heads_mean_ms': round(t_heads, 4) if t_heads else None,
    'full_body_chain_mean_ms': round(t_body, 4) if t_body else None,
    'param_counts': {
        'total_M': round(total / 1e6, 3),
        'encoder_m2_M': round(sum(p.numel() for p in model.encoder_m2.parameters()) / 1e6, 3),
        'backbone_m2_M': round(sum(p.numel() for p in model.backbone_m2.parameters()) / 1e6, 3),
        'camencode_trunk_efficientnet_M': round(sum(p.numel() for p in model.encoder_m2.camencode.trunk.parameters()) / 1e6, 3),
        'camencode_up1_M': round(sum(p.numel() for p in model.encoder_m2.camencode.up1.parameters()) / 1e6, 3),
        'camencode_up2_M': round(sum(p.numel() for p in model.encoder_m2.camencode.up2.parameters()) / 1e6, 3),
        'camencode_depth_head_M': round(sum(p.numel() for p in model.encoder_m2.camencode.depth_head.parameters()) / 1e6, 3),
        'camencode_image_head_M': round(sum(p.numel() for p in model.encoder_m2.camencode.image_head.parameters()) / 1e6, 3),
    }
}

path = '/home/jichengzhi/UniV2X/results/pyramid_m2_submodule_profile.json'
os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {path}")
