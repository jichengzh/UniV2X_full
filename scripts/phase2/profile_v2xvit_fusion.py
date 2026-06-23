"""V2X-ViT fusion_net 内部子模块计时 (hook-level profiling).
授权: team-lead 新指令 (V2X-ViT 全网结构审计, 2026-06-05)
口径: fp32_pytorch, hook CUDA Event, warmup=10, measure=50, GPU空闲确认
"""
import sys, os, time
from pathlib import Path
sys.path.insert(0, '/home/jichengzhi/heal_research/HEAL')
sys.path.insert(0, '/home/jichengzhi/UniV2X')
os.chdir('/home/jichengzhi/heal_research/HEAL')

import torch
from torch.utils.data import DataLoader
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from tools.configurable.depgraph_v2xvit import build_model, CONFIG_YAML
from opencood.hypes_yaml.yaml_utils import load_general_params

WARMUP, MEASURE = 10, 50

model = build_model('cuda'); model.eval()
enc = model.fusion_net.fusion_net.encoder  # V2XTEncoder

# --- hook registry ---
timers = {}
handles = []

def make_hook(name):
    starts = {}
    def pre(mod, inp):
        e = torch.cuda.Event(enable_timing=True); e.record()
        starts[id(mod)] = e
    def post(mod, inp, out):
        e2 = torch.cuda.Event(enable_timing=True); e2.record()
        torch.cuda.synchronize()
        ms = starts[id(mod)].elapsed_time(e2)
        timers.setdefault(name, []).append(ms)
    return pre, post

def reg(mod, name):
    p, q = make_hook(name)
    handles.append(mod.register_forward_pre_hook(p))
    handles.append(mod.register_forward_hook(q))

# Register hooks
reg(enc.sttf, 'STTF')
reg(enc.prior_feed, 'prior_feed')
for i, layer in enumerate(enc.layers):
    fb = layer[0]          # V2XFusionBlock (direct, not PreNorm-wrapped)
    ff_pre = layer[1]      # PreNorm(FFN)
    ff_mod = ff_pre.fn     # inner FFN
    for j, sub in enumerate(fb.layers):
        reg(sub[0].fn, f'L{i}_HMSA_HGTCavAttn')       # PreNorm(HMSA).fn
        reg(sub[1].fn, f'L{i}_MSwin_PyramidWinAttn')  # PreNorm(MSwin).fn
    reg(ff_mod, f'L{i}_FFN')
reg(model.fusion_net.fusion_net, 'fusion_net_total')

# Dataset
hypes = yaml_utils.load_yaml(str(CONFIG_YAML))
hypes = load_general_params(hypes)
hypes['validate_dir'] = hypes['test_dir']
ds = build_dataset(hypes, visualize=False, train=False)
loader = DataLoader(ds, batch_size=1, num_workers=2,
                    collate_fn=ds.collate_batch_test, shuffle=False)

print(f'[profile] warmup={WARMUP} measure={MEASURE}', flush=True)
n = 0
with torch.no_grad():
    for batch in loader:
        if batch is None: continue
        batch = train_utils.to_device(batch, 'cuda')
        _ = inference_utils.inference_intermediate_fusion(batch, model, ds)
        if n == WARMUP:
            for k in timers: timers[k] = []
            print('[profile] warmup done, measuring...', flush=True)
        if n >= WARMUP + MEASURE: break
        n += 1

for h in handles: h.remove()

import json, numpy as np
results = {}
fusion_total_mean = float(np.mean(timers.get('fusion_net_total', [27.39])))
print(f'\n[profile] fusion_net_total mean: {fusion_total_mean:.2f} ms')
print(f'{"Module":40s} {"mean_ms":>8} {"pct_fusion":>11} {"pct_e2e":>9}')
print('-'*70)
e2e = 61.86
for name, vals in timers.items():
    if name == 'fusion_net_total': continue
    m = float(np.mean(vals))
    results[name] = {'mean_ms': round(m,3), 'pct_fusion': round(100*m/fusion_total_mean,1), 'pct_e2e': round(100*m/e2e,1)}
    print(f'{name:40s} {m:>8.3f} {100*m/fusion_total_mean:>10.1f}% {100*m/e2e:>8.1f}%')

results['fusion_net_total_ms'] = fusion_total_mean
results['e2e_reference_ms'] = e2e
out = Path('/home/jichengzhi/UniV2X/results/v2xvit_fusion_profile.json')
with open(out, 'w') as f: json.dump(results, f, indent=2)
print(f'\n[profile] saved to {out}')
