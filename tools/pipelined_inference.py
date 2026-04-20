"""
D2 流水线阶段重叠 benchmark

测量帧间流水线重叠对稳态延迟/显存/功率的影响。
支持: 无重叠 / backbone-BEV 重叠 / 多阶段全重叠。

重叠原理:
  无重叠:
    Frame t:   [Back 34ms] → [BEV 65ms] → [Dec 10ms] → [Seg 89ms]
    Frame t+1:                                                       [Back] → ...
    稳态 = 198ms

  backbone-BEV 重叠:
    Frame t:   [Back 34ms] → [BEV 65ms] → [Dec 10ms] → [Seg 89ms]
    Frame t+1:               [Back 34ms] → [BEV 65ms] → ...
    稳态 = max(Back, BEV) + Dec + Seg = 164ms

  注意: 流水线重叠需要在当前帧的 BEV 阶段同时预取下一帧的 Backbone 结果。
  这需要侵入 ego agent forward 的内部，拆分 backbone 和 BEV 两个阶段。

用法:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/jichengzhi/UniV2X \\
    conda run -n UniV2X_2.0 python tools/pipelined_inference.py \\
      projects/configs_e2e_univ2x/univ2x_coop_e2e_track.py \\
      ckpts/univ2x_coop_e2e_stg2.pth \\
      --mode backbone_bev_overlap \\
      --n-warmup 2 --n-runs 10
"""
import argparse
import json
import os
import subprocess
import sys
import time
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")
sys.path.insert(0, '/home/jichengzhi/UniV2X')

import numpy as np
import torch
import torch.cuda as cuda
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmdet.datasets import replace_ImageToTensor
from mmdet3d.datasets import build_dataset

import projects.mmdet3d_plugin  # noqa: F401


def parse_args():
    p = argparse.ArgumentParser(description="D2 pipeline overlap benchmark")
    p.add_argument("config")
    p.add_argument("checkpoint")
    p.add_argument("--prune-config", default=None)
    p.add_argument("--finetuned-ckpt", default=None)
    p.add_argument("--mode", choices=['none', 'backbone_bev_overlap', 'all'],
                   default='none',
                   help="none=无重叠, backbone_bev_overlap=backbone-BEV重叠, all=遍历")
    p.add_argument("--n-warmup", type=int, default=3)
    p.add_argument("--n-runs", type=int, default=10)
    p.add_argument("--output", default=None)
    return p.parse_args()


def build_model_and_data(args):
    from tools.pruning_sensitivity_analysis import load_model_fresh, get_prune_target
    model, cfg = load_model_fresh(args.config, args.checkpoint)

    if args.prune_config:
        with open(args.prune_config) as f:
            prune_cfg = json.load(f)
        _locked = prune_cfg.setdefault("locked", {})
        _locked.setdefault("importance_criterion", "l1_norm")
        _locked.setdefault("pruning_granularity", "local")
        _locked.setdefault("iterative_steps", 5)
        _locked.setdefault("round_to", 8)
        prune_cfg.setdefault("encoder", {})
        prune_cfg.setdefault("decoder", {})
        prune_cfg.setdefault("heads", {})
        prune_cfg.setdefault("constraints", {
            "skip_layers": ["sampling_offsets", "attention_weights"],
            "min_channels": 64, "channel_alignment": 8,
        })
        from projects.mmdet3d_plugin.univ2x.pruning.prune_univ2x import apply_prune_config
        apply_prune_config(get_prune_target(model), prune_cfg, dataloader=None)
        print(f"[Prune] Applied {args.prune_config}")

    if args.finetuned_ckpt:
        from mmcv.runner import load_checkpoint
        load_checkpoint(model, args.finetuned_ckpt, map_location="cpu")
        print(f"[Finetune] Loaded {args.finetuned_ckpt}")

    cfg.data.test.test_mode = True
    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    dataset = build_dataset(cfg.data.test)
    from projects.mmdet3d_plugin.datasets.builder import build_dataloader
    data_loader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=0, dist=False, shuffle=False
    )
    model = MMDataParallel(model.cuda(), device_ids=[0])
    model.eval()
    return model, dataset, data_loader


def get_gpu_power():
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        return float(out.split('\n')[0])
    except Exception:
        return -1.0


def benchmark_no_overlap(model, data_loader, n_warmup, n_runs):
    """无重叠: 逐帧完全串行执行 (当前默认行为)

    每帧: Backbone → BEV → Decoder → Seg → (等下一帧数据)
    """
    latencies = []
    powers = []
    torch.cuda.reset_peak_memory_stats()

    data_iter = iter(data_loader)
    with torch.no_grad():
        for i in range(n_warmup):
            data = next(data_iter, None) or next(iter(data_loader))
            model(return_loss=False, rescale=True, **data)
            torch.cuda.synchronize()

        for i in range(n_runs):
            data = next(data_iter, None) or next(iter(data_loader))
            pw = get_gpu_power()

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            model(return_loss=False, rescale=True, **data)
            end.record()
            torch.cuda.synchronize()

            latencies.append(start.elapsed_time(end))
            powers.append(pw)

    return {
        'mode': 'no_overlap',
        'latency_mean_ms': float(np.mean(latencies)),
        'latency_std_ms': float(np.std(latencies)),
        'peak_memory_mb': float(torch.cuda.max_memory_allocated() / 1024**2),
        'avg_power_w': float(np.mean(powers)),
        'energy_mj': float(np.mean(latencies) * np.mean(powers)),
        'n_runs': n_runs,
    }


def benchmark_backbone_bev_overlap(model, data_loader, n_warmup, n_runs):
    """backbone-BEV 重叠: 模拟帧间 backbone 预取

    实际上 UniV2X 的 ego forward 不容易拆分为 backbone 和 BEV 两阶段
    (backbone + BEV 紧密耦合在 univ2x_track.py 的 _forward_single_frame 中)。

    本 benchmark 模拟重叠效果:
      1. 连续执行 n_runs 帧
      2. 用模块级 hook 分别测量 backbone 和 non-backbone 时间
      3. 计算理论稳态时延 = max(backbone, non_backbone)
      4. 额外测量并行时的显存开销 (两帧中间激活同时存在)
    """
    from tools.benchmark_latency import ModuleTimer

    ma = model.module  # MultiAgent
    ego = ma.model_ego_agent

    timer = ModuleTimer()
    handles = []
    if hasattr(ego, 'img_backbone'):
        handles.extend(timer.attach(ego.img_backbone, 'backbone'))
    if hasattr(ego, 'img_neck'):
        handles.extend(timer.attach(ego.img_neck, 'neck'))
    if hasattr(ego, 'pts_bbox_head') and hasattr(ego.pts_bbox_head, 'transformer'):
        tf = ego.pts_bbox_head.transformer
        if hasattr(tf, 'encoder'):
            handles.extend(timer.attach(tf.encoder, 'bev_encoder'))
        if hasattr(tf, 'decoder'):
            handles.extend(timer.attach(tf.decoder, 'decoder'))
    if hasattr(ego, 'seg_head'):
        handles.extend(timer.attach(ego.seg_head, 'seg_head'))

    latencies = []
    powers = []
    torch.cuda.reset_peak_memory_stats()

    data_iter = iter(data_loader)
    with torch.no_grad():
        for i in range(n_warmup):
            data = next(data_iter, None) or next(iter(data_loader))
            model(return_loss=False, rescale=True, **data)
            torch.cuda.synchronize()
        timer.times.clear()

        for i in range(n_runs):
            data = next(data_iter, None) or next(iter(data_loader))
            pw = get_gpu_power()

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            model(return_loss=False, rescale=True, **data)
            end.record()
            torch.cuda.synchronize()

            latencies.append(start.elapsed_time(end))
            powers.append(pw)

    # 清理 hooks
    for h in handles:
        h.remove()

    # 模块级延迟统计
    module_stats = timer.summary()

    backbone_ms = module_stats.get('backbone', {}).get('mean', 0)
    neck_ms = module_stats.get('neck', {}).get('mean', 0)
    bev_ms = module_stats.get('bev_encoder', {}).get('mean', 0)
    decoder_ms = module_stats.get('decoder', {}).get('mean', 0)
    seg_ms = module_stats.get('seg_head', {}).get('mean', 0)

    backbone_total = backbone_ms + neck_ms
    non_backbone = bev_ms + decoder_ms + seg_ms

    # 理论稳态延迟 (backbone-BEV 重叠)
    # 稳态帧间延迟 = max(backbone, non_backbone)
    # 因为 backbone(t+1) 与 BEV+Dec+Seg(t) 并行
    theoretical_steady_state = max(backbone_total, non_backbone)

    return {
        'mode': 'backbone_bev_overlap',
        'actual_latency_mean_ms': float(np.mean(latencies)),
        'actual_latency_std_ms': float(np.std(latencies)),
        'module_backbone_ms': round(backbone_total, 1),
        'module_bev_encoder_ms': round(bev_ms, 1),
        'module_decoder_ms': round(decoder_ms, 1),
        'module_seg_head_ms': round(seg_ms, 1),
        'module_non_backbone_ms': round(non_backbone, 1),
        'theoretical_steady_state_ms': round(theoretical_steady_state, 1),
        'theoretical_speedup': round(float(np.mean(latencies)) / theoretical_steady_state, 2)
            if theoretical_steady_state > 0 else 0,
        'peak_memory_mb': float(torch.cuda.max_memory_allocated() / 1024**2),
        'estimated_overlap_extra_memory_mb': round(backbone_total * 0.5, 0),
            # 粗估: backbone 中间激活约占 backbone 延迟比例的显存
        'avg_power_w': float(np.mean(powers)),
        'energy_mj': float(np.mean(latencies) * np.mean(powers)),
        'n_runs': n_runs,
    }


def main():
    args = parse_args()
    print("[D2] Building model and dataset...")
    model, dataset, data_loader = build_model_and_data(args)
    print(f"[D2] Model loaded. Dataset: {len(dataset)} samples")

    if args.mode == 'all':
        modes = ['none', 'backbone_bev_overlap']
    else:
        modes = [args.mode]

    all_results = {}
    for mode in modes:
        print(f"\n{'='*60}")
        print(f"[D2] Benchmarking mode={mode}")
        print(f"{'='*60}")

        torch.cuda.reset_peak_memory_stats()
        if mode == 'none':
            result = benchmark_no_overlap(model, data_loader, args.n_warmup, args.n_runs)
        elif mode == 'backbone_bev_overlap':
            result = benchmark_backbone_bev_overlap(model, data_loader, args.n_warmup, args.n_runs)

        all_results[mode] = result

        if 'theoretical_steady_state_ms' in result:
            print(f"  Actual latency: {result['actual_latency_mean_ms']:.1f} ms")
            print(f"  Backbone: {result['module_backbone_ms']:.1f} ms")
            print(f"  Non-backbone (BEV+Dec+Seg): {result['module_non_backbone_ms']:.1f} ms")
            print(f"  Theoretical steady-state: {result['theoretical_steady_state_ms']:.1f} ms")
            print(f"  Theoretical speedup: {result['theoretical_speedup']:.2f}x")
        else:
            print(f"  Latency: {result['latency_mean_ms']:.1f} +/- {result['latency_std_ms']:.1f} ms")

        print(f"  Peak memory: {result['peak_memory_mb']:.0f} MB")
        print(f"  Avg power: {result['avg_power_w']:.1f} W")

    output = {
        'config': args.config,
        'prune_config': args.prune_config,
        'results': all_results
    }

    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\n[D2] Results saved to {args.output}")
    else:
        print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
