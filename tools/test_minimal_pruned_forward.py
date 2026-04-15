"""最小验证: 剪枝后的模型能 forward 吗"""
import sys, os, time, json
sys.path.insert(0, '/home/jichengzhi/UniV2X')
import torch
import mmdet3d
from mmcv.parallel import MMDataParallel

from tools.pruning_sensitivity_analysis import load_model_fresh, get_prune_target

print("Loading model...", flush=True)
t0 = time.time()
model, cfg = load_model_fresh(
    'projects/configs_e2e_univ2x/univ2x_coop_e2e_track.py',
    'ckpts/univ2x_coop_e2e_stg2.pth'
)
print(f"Loaded in {time.time()-t0:.1f}s", flush=True)

print("\nApplying P1 FFN 30% pruning...", flush=True)
with open('prune_configs/p1_ffn_30pct.json') as f:
    prune_cfg = json.load(f)

ego = get_prune_target(model)
params_before = sum(p.numel() for p in ego.parameters())

from projects.mmdet3d_plugin.univ2x.pruning.prune_univ2x import apply_prune_config
t0 = time.time()
apply_prune_config(ego, prune_cfg, dataloader=None)
print(f"Pruned in {time.time()-t0:.1f}s", flush=True)

params_after = sum(p.numel() for p in ego.parameters())
print(f"params: {params_before:,d} -> {params_after:,d} ({(1-params_after/params_before)*100:.2f}%)", flush=True)

print("\nBuilding test dataset...", flush=True)
from mmdet3d.datasets import build_dataset
from mmdet.datasets import replace_ImageToTensor
cfg.data.test.test_mode = True
cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
dataset = build_dataset(cfg.data.test)
print(f"Dataset size: {len(dataset)}", flush=True)

from projects.mmdet3d_plugin.datasets.builder import build_dataloader
dl = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=0, dist=False, shuffle=False)
print("Dataloader created.", flush=True)

model = MMDataParallel(model.cuda(), device_ids=[0])
model.eval()

t0 = time.time()
with torch.no_grad():
    for i, data in enumerate(dl):
        t_iter = time.time()
        try:
            result = model(return_loss=False, rescale=True, **data)
            print(f"  sample {i}: {time.time()-t_iter:.2f}s OK", flush=True)
        except Exception as e:
            print(f"  sample {i}: FAILED - {type(e).__name__}: {str(e)[:200]}", flush=True)
            import traceback
            traceback.print_exc()
            break
        if i >= 2:
            break
print(f"\nTotal: {time.time()-t0:.1f}s", flush=True)
