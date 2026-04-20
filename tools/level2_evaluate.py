"""
Level 2 真实评估管线

给定一个 (B1, B2, D) 联合配置，执行完整的:
  剪枝 → [量化] → 推理 → AMOTA/延迟/显存/能耗 四指标收集

这是联合搜索框架中 Level 2 的核心——真正运行模型并测量指标。
Level 1 是基于 LUT 的廉价预估(<1s)，Level 2 是实际测量(~15-20min)。

用法:
  CUDA_VISIBLE_DEVICES=4 PYTHONPATH=/home/jichengzhi/UniV2X \\
    conda run -n UniV2X_2.0 python tools/level2_evaluate.py \\
      --b1-prune-config prune_configs/decouple_enc10_07.json \\
      --b1-finetuned-ckpt work_dirs/ft_decouple_enc10_07/epoch_3.pth \\
      --d3-precision int8 --d3-frames 1 \\
      --d4-strategy defrag \\
      --n-eval 168 \\
      --output output/level2_d14_int8cache.json
"""
import argparse
import json
import os
import subprocess
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, '/home/jichengzhi/UniV2X')

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmdet.datasets import replace_ImageToTensor
from mmdet3d.datasets import build_dataset

import projects.mmdet3d_plugin  # noqa

from projects.mmdet3d_plugin.univ2x.pruning.temporal_cache import TemporalCacheManager


def parse_args():
    p = argparse.ArgumentParser(description="Level 2 real evaluation")
    # B1 剪枝
    p.add_argument("--b1-prune-config", default=None, help="剪枝配置 JSON")
    p.add_argument("--b1-finetuned-ckpt", default=None, help="剪枝微调 ckpt")
    # B2 量化 (暂为 placeholder — 后续集成 quick_eval_quant)
    p.add_argument("--b2-quant-config", default=None, help="量化配置 JSON (TODO)")
    # D 硬件配置
    p.add_argument("--d3-precision", choices=['fp16', 'int8'], default='fp16')
    p.add_argument("--d3-frames", type=int, default=1)
    p.add_argument("--d4-strategy", choices=['dynamic', 'defrag'], default='defrag')
    # 评估
    p.add_argument("--n-eval", type=int, default=168)
    p.add_argument("--output", default=None)
    return p.parse_args()


def setup_d4(strategy):
    """D4: 设置显存分配策略"""
    if strategy == 'defrag':
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def build_model(args):
    """构建模型（含剪枝）"""
    from tools.pruning_sensitivity_analysis import load_model_fresh, get_prune_target

    config_path = 'projects/configs_e2e_univ2x/univ2x_coop_e2e_track.py'
    ckpt_path = 'ckpts/univ2x_coop_e2e_stg2.pth'

    model, cfg = load_model_fresh(config_path, ckpt_path)

    # B1: 剪枝
    if args.b1_prune_config:
        with open(args.b1_prune_config) as f:
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

    if args.b1_finetuned_ckpt:
        from mmcv.runner import load_checkpoint
        load_checkpoint(model, args.b1_finetuned_ckpt, map_location="cpu")

    # B2: 量化 (TODO — 需要集成 quick_eval_quant 的 apply_quant_config)

    return model, cfg


def patch_d3_cache(model, precision, frames):
    """D3: Patch temporal cache"""
    from projects.mmdet3d_plugin.univ2x.detectors.univ2x_track import UniV2XTrack

    cache_mgr = TemporalCacheManager(precision=precision, max_frames=frames)
    ego = model.module.model_ego_agent
    ego._temporal_cache_mgr = cache_mgr

    original_fn = UniV2XTrack.simple_test_track

    def patched_simple_test_track(self_ego, img, l2g_t, l2g_r_mat, img_metas,
                                   timestamp, other_agent_results=None):
        mgr = getattr(self_ego, '_temporal_cache_mgr', None)
        if mgr is not None and hasattr(self_ego, 'prev_bev') and self_ego.prev_bev is not None:
            cached = mgr.retrieve()
            if cached is not None:
                self_ego.prev_bev = cached

        result = original_fn(self_ego, img, l2g_t, l2g_r_mat, img_metas,
                              timestamp, other_agent_results=other_agent_results)

        if mgr is not None:
            if hasattr(self_ego, 'prev_bev') and self_ego.prev_bev is not None:
                mgr.store(self_ego.prev_bev)
                if mgr.max_frames == 0:
                    self_ego.prev_bev = None
            else:
                mgr.reset()
        return result

    UniV2XTrack.simple_test_track = patched_simple_test_track
    return cache_mgr, original_fn


def get_gpu_power():
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        return float(out.split('\n')[0])
    except Exception:
        return -1.0


def evaluate(model, data_loader, n_eval, cache_mgr):
    """运行推理并收集四指标"""
    outputs = []
    latencies = []
    powers = []
    torch.cuda.reset_peak_memory_stats()

    data_iter = iter(data_loader)

    # Warmup 2 frames
    with torch.no_grad():
        for _ in range(2):
            data = next(data_iter, None) or next(iter(data_loader))
            model(return_loss=False, rescale=True, **data)
            torch.cuda.synchronize()

    # Evaluate
    with torch.no_grad():
        for i in range(min(n_eval, len(data_loader))):
            data = next(data_iter, None)
            if data is None:
                break

            pw = get_gpu_power()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = model(return_loss=False, rescale=True, **data)
            end.record()
            torch.cuda.synchronize()

            latencies.append(start.elapsed_time(end))
            powers.append(pw)
            outputs.extend(result)

            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{n_eval}] lat={latencies[-1]:.0f}ms")

    # AMOTA
    amota, mAP = None, None
    try:
        dataset = data_loader.dataset
        metrics = dataset.evaluate(
            {"bbox_results": outputs},
            jsonfile_prefix="output/level2_eval_tmp"
        )
        amota = metrics.get('pts_bbox_NuScenes/amota', None)
        mAP = metrics.get('pts_bbox_NuScenes/mAP', None)
    except Exception as e:
        print(f"  [WARN] Eval failed: {e}")

    peak_mem = torch.cuda.max_memory_allocated() / 1024**2
    cache_stats = cache_mgr.memory_stats if cache_mgr else {}

    return {
        'amota': amota,
        'mAP': mAP,
        'latency_mean_ms': float(np.mean(latencies)),
        'latency_std_ms': float(np.std(latencies)),
        'peak_memory_mb': float(peak_mem),
        'avg_power_w': float(np.mean(powers)),
        'energy_per_frame_mj': float(np.mean(latencies) * np.mean(powers)),
        'cache_stats': cache_stats,
        'n_eval': len(outputs),
    }


def main():
    args = parse_args()

    # D4: Memory strategy
    setup_d4(args.d4_strategy)

    print(f"[Level2] Building model...")
    print(f"  B1: prune={args.b1_prune_config}, ckpt={args.b1_finetuned_ckpt}")
    print(f"  D3: precision={args.d3_precision}, frames={args.d3_frames}")
    print(f"  D4: strategy={args.d4_strategy}")

    model, cfg = build_model(args)

    # Dataset
    cfg.data.test.test_mode = True
    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    dataset = build_dataset(cfg.data.test)
    from projects.mmdet3d_plugin.datasets.builder import build_dataloader
    data_loader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=0, dist=False, shuffle=False
    )

    model = MMDataParallel(model.cuda(), device_ids=[0])
    model.eval()

    # D3: Patch temporal cache
    cache_mgr, _ = patch_d3_cache(model, args.d3_precision, args.d3_frames)

    print(f"[Level2] Evaluating {args.n_eval} samples...")
    result = evaluate(model, data_loader, args.n_eval, cache_mgr)

    result['config'] = {
        'b1_prune_config': args.b1_prune_config,
        'b1_finetuned_ckpt': args.b1_finetuned_ckpt,
        'd3_precision': args.d3_precision,
        'd3_frames': args.d3_frames,
        'd4_strategy': args.d4_strategy,
    }

    print(f"\n[Level2] Results:")
    print(f"  AMOTA: {result['amota']}")
    print(f"  mAP: {result['mAP']}")
    print(f"  Latency: {result['latency_mean_ms']:.1f} +/- {result['latency_std_ms']:.1f} ms")
    print(f"  Peak memory: {result['peak_memory_mb']:.0f} MB")
    print(f"  Avg power: {result['avg_power_w']:.1f} W")
    print(f"  Energy/frame: {result['energy_per_frame_mj']:.0f} mJ")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"  Saved to {args.output}")


if __name__ == '__main__':
    main()
