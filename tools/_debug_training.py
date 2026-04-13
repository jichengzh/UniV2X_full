"""Debug: check that training=True is visible to transformer layers during ONNX tracing."""
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.'); sys.path.insert(0, 'projects'); sys.path.insert(0, 'tools')

import mmcv, torch
import torch.nn as nn
from mmcv.utils.registry import Registry
_orig = Registry._register_module
def _force(self, module=None, module_name=None, force=False, **kwargs):
    try: _orig(self, module=module, module_name=module_name, force=False)
    except (KeyError, TypeError): pass
Registry._register_module = _force
import mmdet3d_plugin
Registry._register_module = _orig

from mmdet3d_plugin.univ2x.dense_heads.motion_head import MotionHeadTRTP

cfg = mmcv.Config.fromfile('projects/configs_e2e_univ2x/univ2x_coop_e2e_track_trt_p2.py')
motion = MotionHeadTRTP(**{k: v for k, v in cfg.model_ego_agent.motion_head.items() if k != 'type'}).cuda().eval()

enc_layers = [m for m in motion.modules() if isinstance(m, nn.TransformerEncoderLayer)]
dec_layers = [m for m in motion.modules() if isinstance(m, nn.TransformerDecoderLayer)]
print(f'TransformerEncoderLayer: {len(enc_layers)}')
print(f'TransformerDecoderLayer: {len(dec_layers)}')
print(f'Before: enc training={[m.training for m in enc_layers[:2]]}, dec training={[m.training for m in dec_layers[:2]]}')

# Simulate context manager
for m in enc_layers + dec_layers:
    for sub in m.modules():
        if isinstance(sub, nn.Dropout):
            sub.p = 0.0
    m.train()
print(f'After train(): enc training={[m.training for m in enc_layers[:2]]}, dec training={[m.training for m in dec_layers[:2]]}')
print(f'self_attn training: {[m.self_attn.training for m in enc_layers[:2]]}')

# Now try forward with training=True and see if it avoids fast path
# (just check that forward works in training mode)
with torch.no_grad():
    IntentionInteraction = type(motion.motionformer.intention_interaction_layers)
    intention = motion.motionformer.intention_interaction_layers
    print(f'IntentionInteraction type: {type(intention)}')
    print(f'interaction_transformer type: {type(intention.interaction_transformer)}')
    print(f'interaction_transformer.training: {intention.interaction_transformer.training}')
    print(f'self_attn.training: {intention.interaction_transformer.self_attn.training}')
    print(f'self_attn.batch_first: {intention.interaction_transformer.self_attn.batch_first}')
