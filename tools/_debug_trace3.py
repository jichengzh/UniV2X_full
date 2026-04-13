"""Debug: check what actually triggers _transformer_encoder_layer_fwd."""
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.'); sys.path.insert(0, 'projects')

import torch, torch.nn as nn
import inspect

enc = nn.TransformerEncoderLayer(256, 8, dropout=0.0, batch_first=True)

# Print actual source of forward
src = inspect.getsource(enc.forward)
lines = src.split('\n')
# Show lines with 'training', 'fast_path', '_transformer'
for i, l in enumerate(lines):
    if any(kw in l for kw in ['training', 'fast_path', '_transformer']):
        print(f'{i}: {l}')

print()
print(f'enc.training before train(): {enc.training}')
enc.train()
print(f'enc.training after train(): {enc.training}')
print(f'self_attn.batch_first: {enc.self_attn.batch_first}')

# Try running and check trace output
x = torch.randn(4, 10, 256)

# Monkey-patch the forward to print training state
orig_fwd = type(enc).forward
def _patched_fwd(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
    print(f'[IN FORWARD] self.training={self.training}, self.self_attn.batch_first={self.self_attn.batch_first}')
    return orig_fwd(self, src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask, is_causal=is_causal)
type(enc).forward = _patched_fwd

with torch.no_grad():
    _ = enc(x)  # Eager execution

print('Eager forward done')
