"""Smoke-test: ONNX tracing of DownstreamHeadsWrapper (random weights, small BEV)."""
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
import torch.nn as nn

# Re-use the wrapper class from the export script
sys.path.insert(0, 'tools')
from export_onnx_univ2x import DownstreamHeadsWrapper, onnx_compatible_attention, _register_onnx_symbolics

from mmdet3d_plugin.univ2x.dense_heads.motion_head import MotionHeadTRTP
from mmdet3d_plugin.univ2x.dense_heads.occ_head import OccHeadTRTP
from mmdet3d_plugin.univ2x.dense_heads.planning_head import PlanningHeadSingleModeTRT
from mmdet3d_plugin.univ2x.dense_heads.track_head import BEVFormerTrackHeadTRT

cfg = mmcv.Config.fromfile('projects/configs_e2e_univ2x/univ2x_coop_e2e_track_trt_p2.py')

CLS_MAP = {
    'OccHeadTRTP': OccHeadTRTP,
    'MotionHeadTRTP': MotionHeadTRTP,
    'PlanningHeadSingleModeTRT': PlanningHeadSingleModeTRT,
}

def build2(c):
    cls = CLS_MAP[c.type]
    return cls(**{k: v for k, v in c.items() if k != 'type'})

# Build a mock model object that has the required attributes
class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.motion_head   = build2(cfg.model_ego_agent.motion_head)
        self.occ_head      = build2(cfg.model_ego_agent.occ_head)
        self.planning_head = build2(cfg.model_ego_agent.planning_head)
        # pts_bbox_head — only needs bev_h, bev_w, positional_encoding
        from mmcv.cnn.bricks.transformer import build_positional_encoding
        from mmcv.utils import Registry as Reg
        class FakeDetHead(nn.Module):
            def __init__(self, bev_h, bev_w, embed_dims):
                super().__init__()
                self.bev_h = bev_h
                self.bev_w = bev_w
                from mmdet.models.utils.positional_encoding import LearnedPositionalEncoding
                self.positional_encoding = LearnedPositionalEncoding(
                    num_feats=embed_dims // 2, row_num_embed=bev_h, col_num_embed=bev_w)
        self.pts_bbox_head = FakeDetHead(200, 200, 256)

model = MockModel().cuda().eval()

pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
wrapper = DownstreamHeadsWrapper(model, pc_range).cuda().eval()

BH, BW, C = 200, 200, 256
num_query = 301
num_dec = 6
num_cls = 10
M = 300

bev_embed      = torch.randn(BH * BW, 1, C, device='cuda')
query_feats    = torch.randn(num_dec, 1, num_query, C, device='cuda')
all_bbox_preds = torch.randn(num_dec, 1, num_query, 10, device='cuda')
all_cls_scores = torch.randn(num_dec, 1, num_query, num_cls, device='cuda')
lane_query     = torch.zeros(1, M, C, device='cuda')
lane_query_pos = torch.zeros(1, M, C, device='cuda')
command        = torch.tensor(0, dtype=torch.long, device='cuda')

dummy = (bev_embed, query_feats, all_bbox_preds, all_cls_scores,
         lane_query, lane_query_pos, command)

with torch.no_grad():
    outs = wrapper(*dummy)
print(f'Forward OK — {len(outs)} outputs')
for name, o in zip(['traj_scores', 'traj_preds', 'occ_logits', 'sdc_traj'], outs):
    print(f'  {name}: {o.shape}')

# ONNX export (small BEV to speed up)
out_path = 'onnx/test_downstream_rand.onnx'
os.makedirs('onnx', exist_ok=True)
_register_onnx_symbolics()
print(f'Exporting to {out_path} ...')
with onnx_compatible_attention(wrapper), torch.no_grad():
    torch.onnx.export(
        wrapper, dummy, out_path,
        opset_version=16,
        training=torch.onnx.TrainingMode.TRAINING,
        input_names=['bev_embed', 'query_feats', 'all_bbox_preds', 'all_cls_scores',
                     'lane_query', 'lane_query_pos', 'command'],
        output_names=['traj_scores', 'traj_preds', 'occ_logits', 'sdc_traj'],
        do_constant_folding=False,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH,
        verbose=False,
    )
print(f'ONNX saved: {out_path}')

try:
    import onnx
    m = onnx.load(out_path)
    onnx.checker.check_model(m)
    op_types = {n.op_type for n in m.graph.node}
    plugin_nodes = {t for t in op_types if 'Plugin' in t}
    aten_ops = {n.op_type for n in m.graph.node if n.domain == 'org.pytorch.aten'}
    print(f'ONNX check passed. Plugin nodes: {plugin_nodes}')
    if aten_ops:
        print(f'  WARNING remaining ATen ops: {aten_ops}')
    else:
        print(f'  No ATen ops — fully TRT-compatible.')
except ImportError:
    print('onnx not installed, skip check')
