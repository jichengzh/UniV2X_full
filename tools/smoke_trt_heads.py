"""Smoke-test: instantiate all Phase-2 TRT head variants from p2 config."""
import sys
import importlib.util
import os

sys.path.insert(0, '.')
sys.path.insert(0, 'projects')

# Patch mmcv registry to allow re-registration (avoids HungarianAssigner3D conflict)
import mmcv
from mmcv.utils.registry import Registry
_orig_register = Registry._register_module

def _force_register(self, module, module_name=None, force=False):
    try:
        _orig_register(self, module, module_name=module_name, force=False)
    except KeyError:
        pass  # already registered — skip silently

Registry._register_module = _force_register

# Safe to import now
import mmdet3d_plugin  # noqa: F401 — triggers all registrations

Registry._register_module = _orig_register  # restore strict mode

from mmdet3d_plugin.univ2x.dense_heads.occ_head import OccHeadTRT, OccHeadTRTP
from mmdet3d_plugin.univ2x.dense_heads.motion_head import MotionHeadTRT, MotionHeadTRTP
from mmdet3d_plugin.univ2x.dense_heads.planning_head import PlanningHeadSingleModeTRT

cfg = mmcv.Config.fromfile('projects/configs_e2e_univ2x/univ2x_coop_e2e_track_trt_p2.py')

CLS_MAP = {
    'OccHeadTRTP': OccHeadTRTP,
    'OccHeadTRT': OccHeadTRT,
    'MotionHeadTRTP': MotionHeadTRTP,
    'MotionHeadTRT': MotionHeadTRT,
    'PlanningHeadSingleModeTRT': PlanningHeadSingleModeTRT,
}


def build(cfg_dict):
    cls = CLS_MAP[cfg_dict.type]
    kwargs = {k: v for k, v in cfg_dict.items() if k != 'type'}
    return cls(**kwargs)


tests = [
    ('ego occ_head',      cfg.model_ego_agent.occ_head),
    ('ego motion_head',   cfg.model_ego_agent.motion_head),
    ('ego planning_head', cfg.model_ego_agent.planning_head),
    ('inf occ_head',      cfg.model_other_agent_inf.occ_head),
    ('inf motion_head',   cfg.model_other_agent_inf.motion_head),
]

all_ok = True
for name, hcfg in tests:
    try:
        build(hcfg)
        print(f'  {name} ({hcfg.type}): OK')
    except Exception as e:
        import traceback
        print(f'  {name} ({hcfg.type}): FAIL')
        traceback.print_exc()
        all_ok = False

print()
print('All heads OK' if all_ok else 'SOME FAILURES')

# ── Forward-pass smoke tests ──────────────────────────────────────────────────
import torch

print('\n--- Forward-pass smoke tests ---')

# ── OccHeadTRTP.forward_trt ───────────────────────────────────────────────────
occ = build(cfg.model_ego_agent.occ_head)
occ.eval()
BEV_H, BEV_W, C = 200, 200, 256
x = torch.randn(BEV_H * BEV_W, 1, C)          # (H*W, 1, C)
ins_query = torch.randn(1, 300, C)             # (1, num_query, C) — already mode-fused
try:
    with torch.no_grad():
        out = occ.forward_trt(x, ins_query)
    print(f'  OccHeadTRTP.forward_trt: OK  out keys={list(out.keys()) if isinstance(out, dict) else type(out)}')
except Exception as e:
    import traceback; traceback.print_exc()
    print(f'  OccHeadTRTP.forward_trt: FAIL -- {e}')

# ── MotionHeadTRTP.forward_trt ────────────────────────────────────────────────
mot = build(cfg.model_ego_agent.motion_head)
mot = mot.cuda().eval()
NUM_AGENT = 20
NUM_DEC = 6
bev_embed     = torch.randn(BEV_H * BEV_W, 1, C).cuda()  # (H*W, bs, C) seq-len first
track_query   = torch.randn(1, NUM_DEC, NUM_AGENT, C).cuda()  # (1, num_dec, A, C)
lane_query    = torch.randn(1, 300, C).cuda()
lane_query_pos= torch.randn(1, 300, C).cuda()
track_boxes_1 = torch.randn(NUM_AGENT, 10).cuda()    # last-step boxes
track_boxes_2 = torch.zeros(NUM_AGENT).long().cuda() # class labels (int)
gravity_center= torch.randn(NUM_AGENT, 3).cuda()
yaw           = torch.randn(NUM_AGENT).cuda()
try:
    with torch.no_grad():
        out = mot.forward_trt(bev_embed, track_query, lane_query, lane_query_pos,
                               track_boxes_1, track_boxes_2, gravity_center, yaw)
    print(f'  MotionHeadTRTP.forward_trt: OK  out type={type(out)}')
except Exception as e:
    import traceback; traceback.print_exc()
    print(f'  MotionHeadTRTP.forward_trt: FAIL -- {e}')

# ── PlanningHeadSingleModeTRT.forward_trt ────────────────────────────────────
plan = build(cfg.model_ego_agent.planning_head)
plan.eval()
occ_mask       = torch.randint(0, 2, (1, 10, BEV_H, BEV_W)).float()
bev_pos        = torch.randn(1, C, BEV_H, BEV_W)           # (b, c, h, w)
sdc_traj_query = torch.randn(NUM_DEC, 1, 6, C)              # (num_dec, 1, P, C)
sdc_track_query= torch.randn(1, C)                          # (1, C)
command        = torch.randint(0, 3, (1,))
bev_embed_flat = torch.randn(BEV_H * BEV_W, 1, C)          # (H*W, 1, C)
try:
    with torch.no_grad():
        out = plan.forward_trt(bev_embed_flat,
                                occ_mask, bev_pos, sdc_traj_query,
                                sdc_track_query, command)
    print(f'  PlanningHeadSingleModeTRT.forward_trt: OK  out type={type(out)}')
except Exception as e:
    import traceback; traceback.print_exc()
    print(f'  PlanningHeadSingleModeTRT.forward_trt: FAIL -- {e}')
