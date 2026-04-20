"""
D3 时序缓存管理 benchmark

通过 monkey-patch univ2x_track.py 的 prev_bev 管理逻辑，
测量不同缓存配置（精度 x 帧数）对精度/显存的影响。

支持 5 种组合: (FP16,0), (FP16,1), (FP16,2), (INT8,1), (INT8,2)

用法:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/jichengzhi/UniV2X \\
    conda run -n UniV2X_2.0 python tools/benchmark_temporal_cache.py \\
      projects/configs_e2e_univ2x/univ2x_coop_e2e_track.py \\
      ckpts/univ2x_coop_e2e_stg2.pth \\
      --cache-precision int8 --cache-frames 1 \\
      --n-eval 168

  # sweep 全部 5 种配置
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/jichengzhi/UniV2X \\
    conda run -n UniV2X_2.0 python tools/benchmark_temporal_cache.py \\
      projects/configs_e2e_univ2x/univ2x_coop_e2e_track.py \\
      ckpts/univ2x_coop_e2e_stg2.pth \\
      --sweep --n-eval 168
"""
import argparse
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, '/home/jichengzhi/UniV2X')

import torch
import numpy as np
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmdet.datasets import replace_ImageToTensor
from mmdet3d.datasets import build_dataset

import projects.mmdet3d_plugin  # noqa: F401

from projects.mmdet3d_plugin.univ2x.pruning.temporal_cache import TemporalCacheManager


def parse_args():
    p = argparse.ArgumentParser(description="D3 temporal cache benchmark")
    p.add_argument("config")
    p.add_argument("checkpoint")
    p.add_argument("--prune-config", default=None)
    p.add_argument("--finetuned-ckpt", default=None)
    p.add_argument("--cache-precision", choices=['fp16', 'int8'], default='fp16')
    p.add_argument("--cache-frames", type=int, default=1, choices=[0, 1, 2])
    p.add_argument("--sweep", action="store_true",
                   help="遍历全部 5 种配置: (fp16,0), (fp16,1), (fp16,2), (int8,1), (int8,2)")
    p.add_argument("--n-eval", type=int, default=168,
                   help="评估样本数 (168=全量, 17=1/10)")
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


def patch_temporal_cache(model, cache_precision, cache_frames):
    """Monkey-patch ego agent 的 prev_bev 管理逻辑

    UniV2X 的 prev_bev 管理涉及两层:
      1. UniV2XTrack.simple_test_track (univ2x_track.py:~860-907):
         - self.prev_bev 读取 (line 893) 和写入 (line 907)
         - 场景切换时 self.prev_bev = None (line 874)
      2. UniV2X.forward_test (univ2x_e2e.py:~306):
         - self.prev_frame_info['prev_bev'] 用于 BEVFormer 的 prev_bev

    本 patch 作用在 simple_test_track 上:
      - 在写入 self.prev_bev 后，通过 cache_mgr 量化存储
      - 在读取 self.prev_bev 前，通过 cache_mgr 反量化
      - 场景切换时同步 reset cache_mgr
    """
    ego = model.module.model_ego_agent
    cache_mgr = TemporalCacheManager(
        precision=cache_precision,
        max_frames=cache_frames
    )
    ego._temporal_cache_mgr = cache_mgr

    # Patch simple_test_track (定义在 UniV2XTrack)
    from projects.mmdet3d_plugin.univ2x.detectors.univ2x_track import UniV2XTrack
    original_simple_test_track = UniV2XTrack.simple_test_track

    def patched_simple_test_track(self_ego, img, l2g_t, l2g_r_mat, img_metas,
                                   timestamp, other_agent_results=None):
        """在原始 simple_test_track 前后插入 cache 管理"""
        mgr = getattr(self_ego, '_temporal_cache_mgr', None)

        # PRE: 如果有 cache manager, 在 prev_bev 读取前做反量化
        if mgr is not None and hasattr(self_ego, 'prev_bev') and self_ego.prev_bev is not None:
            # 场景不变时, 用 cache 的值替换 prev_bev
            cached = mgr.retrieve()
            if cached is not None:
                self_ego.prev_bev = cached

        # 执行原始逻辑
        result = original_simple_test_track(
            self_ego, img, l2g_t, l2g_r_mat, img_metas,
            timestamp, other_agent_results=other_agent_results
        )

        # POST: 如果有 cache manager, 在 prev_bev 写入后做量化存储
        if mgr is not None:
            if hasattr(self_ego, 'prev_bev') and self_ego.prev_bev is not None:
                mgr.store(self_ego.prev_bev)
                # max_frames=0: 禁用时序
                if mgr.max_frames == 0:
                    self_ego.prev_bev = None
            else:
                # 场景切换 → reset cache
                mgr.reset()

        return result

    UniV2XTrack.simple_test_track = patched_simple_test_track
    return cache_mgr, original_simple_test_track


def unpatch_temporal_cache(model, original_simple_test_track):
    """恢复原始 simple_test_track"""
    from projects.mmdet3d_plugin.univ2x.detectors.univ2x_track import UniV2XTrack
    UniV2XTrack.simple_test_track = original_simple_test_track
    ego = model.module.model_ego_agent
    if hasattr(ego, '_temporal_cache_mgr'):
        del ego._temporal_cache_mgr


def evaluate_with_cache(model, data_loader, n_eval, cache_precision, cache_frames):
    """用指定缓存配置评估 AMOTA"""
    cache_mgr, original_ft = patch_temporal_cache(model, cache_precision, cache_frames)

    outputs = []
    latencies = []
    torch.cuda.reset_peak_memory_stats()

    data_iter = iter(data_loader)
    with torch.no_grad():
        for i in range(min(n_eval, len(data_loader))):
            data = next(data_iter, None)
            if data is None:
                break

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = model(return_loss=False, rescale=True, **data)
            end.record()
            torch.cuda.synchronize()

            latencies.append(start.elapsed_time(end))
            outputs.extend(result)

            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{n_eval}] latency: {latencies[-1]:.0f}ms")

    peak_mem = torch.cuda.max_memory_allocated() / 1024**2
    cache_stats = cache_mgr.memory_stats

    # 恢复
    unpatch_temporal_cache(model, original_ft)

    # 评估 AMOTA
    amota = None
    mAP = None
    try:
        dataset = data_loader.dataset
        result_dict = {"bbox_results": outputs}
        metrics = dataset.evaluate(result_dict, jsonfile_prefix=f"output/d3_cache_{cache_precision}_{cache_frames}")
        # 在返回的 metrics dict 中查找 AMOTA (可能在不同 key 下)
        for k, v in metrics.items():
            if 'amota' in k.lower() and not k.startswith('pts_bbox'):
                amota = v
                break
        if amota is None:
            amota = metrics.get('pts_bbox_NuScenes/amota', None)
        mAP = metrics.get('pts_bbox_NuScenes/mAP', None)
    except Exception as e:
        print(f"  [WARN] AMOTA evaluation failed: {e}")

    return {
        'cache_precision': cache_precision,
        'cache_frames': cache_frames,
        'amota': amota,
        'mAP': mAP,
        'latency_mean_ms': float(np.mean(latencies)),
        'latency_std_ms': float(np.std(latencies)),
        'peak_memory_mb': float(peak_mem),
        'cache_size_mb': cache_stats['cache_size_mb'],
        'total_memory_saved_mb': cache_stats['total_memory_saved_mb'],
        'n_eval': len(outputs),
    }


def main():
    args = parse_args()
    print("[D3] Building model and dataset...")
    model, dataset, data_loader = build_model_and_data(args)
    print(f"[D3] Model loaded. Dataset: {len(dataset)} samples, eval: {args.n_eval}")

    if args.sweep:
        configs = [
            ('fp16', 0), ('fp16', 1), ('fp16', 2),
            ('int8', 1), ('int8', 2),
        ]
    else:
        configs = [(args.cache_precision, args.cache_frames)]

    all_results = {}
    for prec, frames in configs:
        key = f"{prec}_{frames}frames"
        print(f"\n{'='*60}")
        print(f"[D3] Evaluating cache: precision={prec}, frames={frames}")
        print(f"{'='*60}")

        result = evaluate_with_cache(model, data_loader, args.n_eval, prec, frames)
        all_results[key] = result

        print(f"  AMOTA: {result['amota']}")
        print(f"  Latency: {result['latency_mean_ms']:.1f} +/- {result['latency_std_ms']:.1f} ms")
        print(f"  Peak memory: {result['peak_memory_mb']:.0f} MB")
        print(f"  Cache size: {result['cache_size_mb']} MB")

    output = {
        'config': args.config,
        'prune_config': args.prune_config,
        'n_eval': args.n_eval,
        'results': all_results,
    }

    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\n[D3] Results saved to {args.output}")
    else:
        print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
