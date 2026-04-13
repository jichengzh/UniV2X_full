"""Check whether training=True prevents fast-path ops during ONNX tracing."""
import sys, warnings, onnx
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

# Use just IntentionInteraction (TransformerEncoderLayer, batch_first=True)
from projects.mmdet3d_plugin.univ2x.dense_heads.motion_head_plugin.modules import IntentionInteraction

layer = IntentionInteraction().cuda().eval()
x = torch.randn(2, 5, 256).cuda()  # (B*A, P, D) with batch_first=True

def aten_ops(path):
    m = onnx.load(path)
    return {n.op_type for n in m.graph.node if n.domain == 'org.pytorch.aten'}

# Export in eval mode (training=False)
with torch.no_grad():
    torch.onnx.export(layer, (x,), '/tmp/intention_eval.onnx', opset_version=16,
        do_constant_folding=False,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH, verbose=False)
print('Eval mode aten ops:', aten_ops('/tmp/intention_eval.onnx'))

# Set to training mode (should disable fast path)
layer.train()
for sub in layer.modules():
    if isinstance(sub, nn.Dropout): sub.p = 0.0

with torch.no_grad():
    torch.onnx.export(layer, (x,), '/tmp/intention_train.onnx', opset_version=16,
        do_constant_folding=False,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH, verbose=False)
print('Train mode aten ops:', aten_ops('/tmp/intention_train.onnx'))
