"""
Pyramid m1 DAIR submodule profiling.
口径: fp32_pytorch CUDA-Event chain-subtract, DAIR dummy input (record_len=2 V2I, random weights)
授权: HANDOFF §二 2.2 预授权「内部子模块 hook 计时若缺→profiling 微跑【此微跑已由 team-lead
      在本条消息预授权, 引用本句即可; 须空闲卡 nvidia-smi 实查 + 启动宣告】」
GPU: cuda:0 (util=0%, mem=1MiB confirmed by nvidia-smi pre-check)
"""
import sys, os, json, torch
sys.path.insert(0, '/home/jichengzhi/heal_research/HEAL')
os.chdir('/home/jichengzhi/heal_research/HEAL')

DEVICE = 'cuda:0'
WARMUP = 20
MEASURE = 200

from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.tools.train_utils import create_model

cfg = load_yaml('opencood/hypes_yaml/dairv2x/LiDAROnly/lidar_pyramid.yaml')
model = create_model(cfg).to(DEVICE)
model.eval()

def timed(fn, warmup=WARMUP, measure=MEASURE):
    """CUDA Event timer. Returns mean_ms."""
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(measure):
            fn()
        e.record()
        torch.cuda.synchronize()
    return s.elapsed_time(e) / measure

# ---- DAIR dummy inputs ----
B = 2  # 2 agents (ego + infra) for DAIR
sf = torch.randn(B, 64, 128, 256, device=DEVICE)  # PointPillar BEV pseudo-image

# Get backbone_m1 output shape
with torch.no_grad():
    bb_out = model.backbone_m1({'spatial_features': sf})
bb_feats = bb_out.get('spatial_features_2d', bb_out.get('spatial_features'))
C_bb, H_bb, W_bb = bb_feats.shape[1], bb_feats.shape[2], bb_feats.shape[3]
print(f"backbone_m1 output: (B={B}, C={C_bb}, H={H_bb}, W={W_bb})")

# affine_matrix: identity transform [B_scene=1, L=2, L=2, 2, 3]
affine_matrix = torch.zeros(1, 2, 2, 2, 3, device=DEVICE)
affine_matrix[0, :, :, 0, 0] = 1.0  # scale x=1
affine_matrix[0, :, :, 1, 1] = 1.0  # scale y=1 → identity 2D affine

record_len = torch.tensor([2], device=DEVICE)  # 1 scene with 2 agents
agent_modality_list = ['m1', 'm1']  # both LiDAR

# Pre-compute fixed backbone output for downstream timing
with torch.no_grad():
    heter_feat = bb_feats.clone()

print(f"forward_collab input: (sum_len={B}, C={C_bb}, H={H_bb}, W={W_bb})")

# ---- Timing: backbone_m1 ----
def fn_bb():
    model.backbone_m1({'spatial_features': sf})

t_bb = timed(fn_bb)
print(f"backbone_m1:      {t_bb:.4f} ms  input=(2,64,128,256)")

# ---- Timing: pyramid_backbone.forward_collab ----
def fn_pb():
    model.pyramid_backbone.forward_collab(
        heter_feat, record_len, affine_matrix, agent_modality_list, None
    )

t_pb = timed(fn_pb)
print(f"pyramid_backbone: {t_pb:.4f} ms  input=(2,{C_bb},{H_bb},{W_bb})")

# Get pyramid_backbone output
with torch.no_grad():
    fused_feat, _ = model.pyramid_backbone.forward_collab(
        heter_feat, record_len, affine_matrix, agent_modality_list, None
    )
C_pb, H_pb, W_pb = fused_feat.shape[1], fused_feat.shape[2], fused_feat.shape[3]
print(f"pyramid_backbone output: (1, {C_pb}, {H_pb}, {W_pb})")

# ---- Timing: shrink_conv ----
def fn_sc():
    model.shrink_conv(fused_feat)

t_sc = timed(fn_sc)
print(f"shrink_conv:      {t_sc:.4f} ms  input=(1,{C_pb},{H_pb},{W_pb})")

# Get shrink output
with torch.no_grad():
    shrunk = model.shrink_conv(fused_feat)
C_sc, H_sc, W_sc = shrunk.shape[1], shrunk.shape[2], shrunk.shape[3]
print(f"shrink_conv output: (1, {C_sc}, {H_sc}, {W_sc})")

# ---- Timing: heads ----
def fn_heads():
    model.cls_head(shrunk)
    model.reg_head(shrunk)
    model.dir_head(shrunk)

t_heads = timed(fn_heads)
print(f"heads (cls+reg+dir): {t_heads:.4f} ms  input=(1,{C_sc},{H_sc},{W_sc})")

# ---- Full body chain ----
def fn_full_body():
    sf2 = sf  # reuse constant input
    bb = model.backbone_m1({'spatial_features': sf2})
    hf = bb.get('spatial_features_2d', bb.get('spatial_features'))
    fused, _ = model.pyramid_backbone.forward_collab(
        hf, record_len, affine_matrix, agent_modality_list, None
    )
    shrunk2 = model.shrink_conv(fused)
    model.cls_head(shrunk2)
    model.reg_head(shrunk2)
    model.dir_head(shrunk2)

t_body = timed(fn_full_body)
print(f"full body chain:  {t_body:.4f} ms  (bb+pb+sc+heads sum={t_bb+t_pb+t_sc+t_heads:.4f})")

# ---- Summary ----
pct_total = lambda ms: f"{100*ms/t_body:.1f}%"
print("\n=== PYRAMID M1 SUB-MODULE TIMING ===")
print(f"[口径] fp32_pytorch CUDA-Event, DAIR dummy input (record_len=2), random weights, GPU0")
print(f"  encoder_m1 (PointPillar VFE):  NOT MEASURED (sparse, requires real voxel input)")
print(f"  backbone_m1 (BaseBEVBackbone):  {t_bb:.4f} ms  ({pct_total(t_bb)} body)")
print(f"  pyramid_backbone (forward_collab): {t_pb:.4f} ms  ({pct_total(t_pb)} body)")
print(f"  shrink_conv:                    {t_sc:.4f} ms  ({pct_total(t_sc)} body)")
print(f"  heads (cls+reg+dir):            {t_heads:.4f} ms  ({pct_total(t_heads)} body)")
print(f"  ---")
print(f"  full body (chain):              {t_body:.4f} ms")
print(f"  NMS: NOT MEASURED (CPU, separate)")

# ---- Save ----
out = {
    'metadata': {
        'model': 'PyramidFusion_m1_DAIR',
        'device': DEVICE,
        'precision': 'fp32',
        'framework': 'pytorch_cuda_event',
        'ckpt': 'random_weights',
        'warmup': WARMUP,
        'measure': MEASURE,
        'input': f'DAIR_dummy B=2 C=64 H=128 W=256',
        'latency_kind': 'submodule_fp32_pytorch_cuda_event_random_weight',
        'authorization': 'HANDOFF §2.2 pre-authorized profiling micro-run',
        'note': 'encoder_m1 excluded (sparse VFE, real voxel required); NMS excluded (CPU)'
    },
    'backbone_m1_mean_ms': round(t_bb, 4),
    'pyramid_backbone_mean_ms': round(t_pb, 4),
    'shrink_conv_mean_ms': round(t_sc, 4),
    'heads_mean_ms': round(t_heads, 4),
    'full_body_chain_mean_ms': round(t_body, 4),
    'encoder_m1_mean_ms': None,
    'shapes': {
        'input_sf': [2, 64, 128, 256],
        'backbone_m1_out': [B, C_bb, H_bb, W_bb],
        'pyramid_backbone_out': [1, C_pb, H_pb, W_pb],
        'shrink_conv_out': [1, C_sc, H_sc, W_sc],
    }
}
path = '/home/jichengzhi/UniV2X/results/pyramid_m1_submodule_profile.json'
with open(path, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {path}")
