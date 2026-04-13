"""Smoke-test: D+E+F pipeline chain (motion → occ → planning)."""
import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
sys.path.insert(0, 'projects')

import mmcv
from mmcv.utils.registry import Registry
_orig = Registry._register_module
def _force(self, module, module_name=None, force=False):
    try:
        _orig(self, module, module_name=module_name, force=False)
    except KeyError:
        pass
Registry._register_module = _force
import mmdet3d_plugin  # noqa
Registry._register_module = _orig

import torch
from mmdet3d_plugin.univ2x.dense_heads.motion_head import MotionHeadTRTP
from mmdet3d_plugin.univ2x.dense_heads.occ_head import OccHeadTRTP
from mmdet3d_plugin.univ2x.dense_heads.planning_head import PlanningHeadSingleModeTRT

cfg = mmcv.Config.fromfile('projects/configs_e2e_univ2x/univ2x_coop_e2e_track_trt_p2.py')

CLS_MAP = {
    'OccHeadTRTP': OccHeadTRTP,
    'MotionHeadTRTP': MotionHeadTRTP,
    'PlanningHeadSingleModeTRT': PlanningHeadSingleModeTRT,
}

def build2(c):
    cls = CLS_MAP[c.type]
    return cls(**{k: v for k, v in c.items() if k != 'type'})

motion = build2(cfg.model_ego_agent.motion_head).cuda().eval()
occ    = build2(cfg.model_ego_agent.occ_head).eval()
plan   = build2(cfg.model_ego_agent.planning_head).eval()
print('Models built OK')

BH, BW, C, A = 200, 200, 256, 20
bev_embed       = torch.randn(BH * BW, 1, C).cuda()
track_query_in  = torch.randn(1, 6, A, C).cuda()
lane_query      = torch.zeros(1, 300, C).cuda()
lane_query_pos  = torch.zeros(1, 300, C).cuda()
track_scores    = torch.rand(A).cuda()
track_labels    = torch.zeros(A, dtype=torch.long).cuda()
gravity_center  = torch.randn(A, 3).cuda()
yaw             = torch.randn(A, 1).cuda()

with torch.no_grad():
    out = motion.forward_trt(bev_embed, track_query_in, lane_query, lane_query_pos,
                              track_scores, track_labels, gravity_center, yaw)
traj_scores, traj_preds, _v, inter_states, track_query, track_query_pos = out
print(f'Motion OK: traj_scores={traj_scores.shape}, inter_states={inter_states.shape}')

ins_query  = occ.merge_queries_trt(track_query.cpu(), track_query_pos.cpu(), inter_states.cpu())
occ_logits = occ.forward_trt(bev_embed.cpu(), ins_query)
print(f'Occ OK: {occ_logits.shape}')

sdc_traj_query  = inter_states[:, :, -1]   # (num_layers, 1, P, C)
sdc_track_query = track_query[:, -1]        # (1, C)
bev_pos = torch.randn(1, C, BH, BW)
occ_mask_dummy = torch.zeros(1, 1, BH, BW)
with torch.no_grad():
    sdc_traj = plan.forward_trt(
        bev_embed.cpu(), occ_mask_dummy, bev_pos,
        sdc_traj_query.cpu(), sdc_track_query.cpu(),
        torch.tensor(0),
    )
print(f'Planning OK: {sdc_traj.shape}')
print('ALL OK!')
