"""Isolate which aten ops come from which head (all on CUDA)."""
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.'); sys.path.insert(0, 'projects')

import mmcv
from mmcv.utils.registry import Registry
_orig = Registry._register_module
def _force(self, module, module_name=None, force=False):
    try: _orig(self, module, module_name=module_name, force=False)
    except KeyError: pass
Registry._register_module = _force
import mmdet3d_plugin
Registry._register_module = _orig

import torch
import onnx

cfg = mmcv.Config.fromfile('projects/configs_e2e_univ2x/univ2x_coop_e2e_track_trt_p2.py')

from mmdet3d_plugin.univ2x.dense_heads.planning_head import PlanningHeadSingleModeTRT
from mmdet3d_plugin.univ2x.dense_heads.occ_head import OccHeadTRTP
from mmdet3d_plugin.univ2x.dense_heads.motion_head import MotionHeadTRTP

def build2(c, cls): return cls(**{k:v for k,v in c.items() if k!='type'})
def aten_ops_in(path):
    m = onnx.load(path)
    return {n.op_type: sum(1 for x in m.graph.node if x.op_type == n.op_type and x.domain == 'org.pytorch.aten')
            for n in m.graph.node if n.domain == 'org.pytorch.aten'}

BH, BW, C, A = 200, 200, 256, 10

# OCC head on CUDA
occ = build2(cfg.model_ego_agent.occ_head, OccHeadTRTP).cuda().eval()
x = torch.randn(BH*BW, 1, C).cuda()
ins_q = torch.randn(1, A, C).cuda()
with torch.no_grad():
    torch.onnx.export(occ, (x, ins_q), '/tmp/occ_cuda.onnx', opset_version=16,
        input_names=['x', 'ins_q'], output_names=['out'],
        do_constant_folding=False,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH, verbose=False)
print('OCC head (CUDA) aten ops:', aten_ops_in('/tmp/occ_cuda.onnx'))

# Planning head on CUDA
plan = build2(cfg.model_ego_agent.planning_head, PlanningHeadSingleModeTRT).cuda().eval()
bev_embed = torch.randn(BH*BW, 1, C).cuda()
occ_mask = torch.zeros(1, 1, BH, BW).cuda()
bev_pos = torch.randn(1, C, BH, BW).cuda()
sdc_traj_query = torch.randn(3, 1, 6, C).cuda()
sdc_track_query = torch.randn(1, C).cuda()
command = torch.tensor([0]).cuda()
with torch.no_grad():
    torch.onnx.export(plan, (bev_embed, occ_mask, bev_pos, sdc_traj_query, sdc_track_query, command),
        '/tmp/plan_cuda.onnx', opset_version=16,
        input_names=['bev_embed', 'occ_mask', 'bev_pos', 'sdc_traj_query', 'sdc_track_query', 'command'],
        output_names=['sdc_traj'],
        do_constant_folding=False,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH, verbose=False)
print('Plan head (CUDA) aten ops:', aten_ops_in('/tmp/plan_cuda.onnx'))

# Motion head (just forward_trt, on CUDA)
# Note: can't easily export just motion head without wrapper, skip for now
print('Done')
