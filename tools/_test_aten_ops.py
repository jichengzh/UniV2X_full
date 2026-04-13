"""Check which aten ops come from OCC head vs planning head."""
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

def build2(c, cls): return cls(**{k:v for k,v in c.items() if k!='type'})
def aten_ops_in(path):
    m = onnx.load(path)
    ops = {}
    for n in m.graph.node:
        if n.domain == 'org.pytorch.aten':
            ops[n.op_type] = ops.get(n.op_type, 0) + 1
    return ops

# OCC head
occ = build2(cfg.model_ego_agent.occ_head, OccHeadTRTP).eval()
x = torch.randn(200*200, 1, 256)
ins_q = torch.randn(1, 10, 256)
with torch.no_grad():
    torch.onnx.export(occ, (x, ins_q), '/tmp/occ_only.onnx', opset_version=16,
        input_names=['x', 'ins_q'], output_names=['out'],
        do_constant_folding=False,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH, verbose=False)
print('OCC head aten ops:', aten_ops_in('/tmp/occ_only.onnx'))

# Planning head
plan = build2(cfg.model_ego_agent.planning_head, PlanningHeadSingleModeTRT).eval()
BH, BW, C = 200, 200, 256
P = 6
bev_embed = torch.randn(BH*BW, 1, C)
occ_mask = torch.zeros(1, 1, BH, BW)
bev_pos = torch.randn(1, C, BH, BW)
sdc_traj_query = torch.randn(3, 1, P, C)
sdc_track_query = torch.randn(1, C)
command = torch.tensor([0])
with torch.no_grad():
    torch.onnx.export(plan, (bev_embed, occ_mask, bev_pos, sdc_traj_query, sdc_track_query, command),
        '/tmp/plan_only.onnx', opset_version=16,
        input_names=['bev_embed', 'occ_mask', 'bev_pos', 'sdc_traj_query', 'sdc_track_query', 'command'],
        output_names=['sdc_traj'],
        do_constant_folding=False,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH, verbose=False)
print('Plan head aten ops:', aten_ops_in('/tmp/plan_only.onnx'))
