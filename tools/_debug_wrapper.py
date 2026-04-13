"""Debug: check that onnx_compatible_attention finds and modifies all transformer layers."""
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.'); sys.path.insert(0, 'projects'); sys.path.insert(0, 'tools')

import mmcv, torch, torch.nn as nn
from mmcv.utils.registry import Registry
_orig_reg = Registry._register_module
def _force_reg(self, module=None, module_name=None, force=False, **kwargs):
    try: _orig_reg(self, module=module, module_name=module_name, force=False)
    except (KeyError, TypeError): pass
Registry._register_module = _force_reg
import mmdet3d_plugin
Registry._register_module = _orig_reg

from export_onnx_univ2x import DownstreamHeadsWrapper, onnx_compatible_attention
from mmdet3d_plugin.univ2x.dense_heads.motion_head import MotionHeadTRTP
from mmdet3d_plugin.univ2x.dense_heads.occ_head import OccHeadTRTP
from mmdet3d_plugin.univ2x.dense_heads.planning_head import PlanningHeadSingleModeTRT
from mmdet.models.utils.positional_encoding import LearnedPositionalEncoding

cfg = mmcv.Config.fromfile('projects/configs_e2e_univ2x/univ2x_coop_e2e_track_trt_p2.py')

def build2(c, cls):
    return cls(**{k: v for k, v in c.items() if k != 'type'})

class FakeDetHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.bev_h = 200
        self.bev_w = 200
        self.positional_encoding = LearnedPositionalEncoding(128, 200, 200)

class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.motion_head   = build2(cfg.model_ego_agent.motion_head, MotionHeadTRTP)
        self.occ_head      = build2(cfg.model_ego_agent.occ_head, OccHeadTRTP)
        self.planning_head = build2(cfg.model_ego_agent.planning_head, PlanningHeadSingleModeTRT)
        self.pts_bbox_head = FakeDetHead()
    def num_query(self): return 300

model = MockModel().cuda().eval()
pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
wrapper = DownstreamHeadsWrapper(model, pc_range).cuda().eval()

enc = [m for m in wrapper.modules() if isinstance(m, nn.TransformerEncoderLayer)]
dec = [m for m in wrapper.modules() if isinstance(m, nn.TransformerDecoderLayer)]
print(f'Before context: enc={len(enc)} training={[m.training for m in enc][:3]}')
print(f'Before context: dec={len(dec)} training={[m.training for m in dec][:3]}')

with onnx_compatible_attention(wrapper):
    enc2 = [m for m in wrapper.modules() if isinstance(m, nn.TransformerEncoderLayer)]
    dec2 = [m for m in wrapper.modules() if isinstance(m, nn.TransformerDecoderLayer)]
    print(f'Inside context: enc training={[m.training for m in enc2][:3]}')
    print(f'Inside context: dec training={[m.training for m in dec2][:3]}')
    print(f'Inside context: enc self_attn training={[m.self_attn.training for m in enc2][:3]}')

enc3 = [m for m in wrapper.modules() if isinstance(m, nn.TransformerEncoderLayer)]
print(f'After context: enc training={[m.training for m in enc3][:3]}')
print('Done')
