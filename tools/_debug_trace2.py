"""Minimal test: TransformerEncoderLayer eval vs train aten ops."""
import sys, warnings, onnx
warnings.filterwarnings('ignore')
sys.path.insert(0, '.'); sys.path.insert(0, 'projects')

import torch, torch.nn as nn
import torch.nn.functional as F

enc = nn.TransformerEncoderLayer(256, 8, dropout=0.0, batch_first=True).cuda().eval()
x = torch.randn(4, 10, 256).cuda()  # (B, L, C) with batch_first=True

def aten_ops(path):
    m = onnx.load(path)
    return {n.op_type: sum(1 for n2 in m.graph.node if n2.op_type == n.op_type and n2.domain == 'org.pytorch.aten')
            for n in m.graph.node if n.domain == 'org.pytorch.aten'}

# Eval mode
with torch.no_grad():
    torch.onnx.export(enc, (x,), '/tmp/enc_eval.onnx', opset_version=16,
        do_constant_folding=False,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH, verbose=False)
print('Eval mode aten ops:', aten_ops('/tmp/enc_eval.onnx'))

# Training mode
enc.train()
for sub in enc.modules():
    if isinstance(sub, nn.Dropout): sub.p = 0.0
print(f'Training mode: {enc.training}, self_attn.training: {enc.self_attn.training}')

# Patch SDPA
orig_sdpa = F.scaled_dot_product_attention
def _onnx_sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    scale_f = q.size(-1) ** -0.5 if scale is None else scale
    return torch.softmax(q @ k.transpose(-2, -1) * scale_f, dim=-1) @ v
F.scaled_dot_product_attention = _onnx_sdpa

with torch.no_grad():
    torch.onnx.export(enc, (x,), '/tmp/enc_train.onnx', opset_version=16,
        do_constant_folding=False,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH, verbose=False)
F.scaled_dot_product_attention = orig_sdpa
print('Train+SDPA patch aten ops:', aten_ops('/tmp/enc_train.onnx'))
