"""
D2 流水线重叠 benchmark

通过 monkey-patch get_bevs() 方法，分别计时 backbone 和 BEV encoding 两阶段，
验证流水线重叠的理论收益和实测可行性。

实现方式:
  mode=none:         原始串行 (无修改)
  mode=backbone_bev: 在 get_bevs() 内部拆分 backbone 和 BEV 为两个阶段，
                     分别用 CUDA Event 计时，计算理论稳态帧间延迟

注: 真正的帧间 backbone 异步预取受限于 MMDataParallel 的 DataContainer scatter
    机制——下一帧数据在 model() 外部无法提前 scatter 到 GPU。
    完整实现需要绕过 MMDataParallel（使用自定义推理循环）或迁移到 TRT 部署。
    本 benchmark 测量的是**阶段级延迟**，用于计算理论重叠收益。

用法:
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/home/jichengzhi/UniV2X \\
    conda run -n UniV2X_2.0 python tools/pipelined_ego_forward.py \\
      projects/configs_e2e_univ2x/univ2x_coop_e2e_track.py \\
      ckpts/univ2x_coop_e2e_stg2.pth \\
      --n-warmup 2 --n-runs 10
"""
import argparse
import json
import os
import subprocess
import sys
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")
sys.path.insert(0, '/home/jichengzhi/UniV2X')

import torch
import torch.cuda as cuda
import numpy as np

import projects.mmdet3d_plugin  # noqa


def parse_args():
    p = argparse.ArgumentParser(description="D2 pipeline stage timing")
    p.add_argument("config")
    p.add_argument("checkpoint")
    p.add_argument("--prune-config", default=None)
    p.add_argument("--finetuned-ckpt", default=None)
    p.add_argument("--n-warmup", type=int, default=3)
    p.add_argument("--n-runs", type=int, default=10)
    p.add_argument("--output", default=None)
    return p.parse_args()


def get_gpu_power():
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        return float(out.split('\n')[0])
    except Exception:
        return -1.0


def build_model_and_data(args):
    from mmcv.parallel import MMDataParallel
    from mmdet.datasets import replace_ImageToTensor
    from mmdet3d.datasets import build_dataset
    from tools.pruning_sensitivity_analysis import load_model_fresh, get_prune_target

    model, cfg = load_model_fresh(args.config, args.checkpoint)

    if args.prune_config:
        with open(args.prune_config) as f:
            prune_cfg = json.load(f)
        _l = prune_cfg.setdefault("locked", {})
        _l.setdefault("importance_criterion", "l1_norm")
        _l.setdefault("pruning_granularity", "local")
        _l.setdefault("iterative_steps", 5)
        _l.setdefault("round_to", 8)
        prune_cfg.setdefault("encoder", {})
        prune_cfg.setdefault("decoder", {})
        prune_cfg.setdefault("heads", {})
        prune_cfg.setdefault("constraints", {
            "skip_layers": ["sampling_offsets", "attention_weights"],
            "min_channels": 64, "channel_alignment": 8})
        from projects.mmdet3d_plugin.univ2x.pruning.prune_univ2x import apply_prune_config
        apply_prune_config(get_prune_target(model), prune_cfg, dataloader=None)

    if args.finetuned_ckpt:
        from mmcv.runner import load_checkpoint
        load_checkpoint(model, args.finetuned_ckpt, map_location="cpu")

    cfg.data.test.test_mode = True
    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    dataset = build_dataset(cfg.data.test)
    from projects.mmdet3d_plugin.datasets.builder import build_dataloader
    dl = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=0, dist=False, shuffle=False)

    model = MMDataParallel(model.cuda(), device_ids=[0])
    model.eval()
    return model, dataset, dl


def patch_get_bevs_with_timing(ego_model):
    """Monkey-patch get_bevs() 加入阶段级 CUDA Event 计时"""
    original_get_bevs = ego_model.get_bevs
    stage_times = defaultdict(list)

    def timed_get_bevs(imgs, img_metas, prev_img=None, prev_img_metas=None, prev_bev=None):
        if prev_img is not None and prev_img_metas is not None:
            prev_bev = ego_model.get_history_bev(prev_img, prev_img_metas)

        # Stage 1: Backbone + FPN
        s1_start = cuda.Event(enable_timing=True)
        s1_end = cuda.Event(enable_timing=True)
        s1_start.record()
        img_feats = ego_model.extract_img_feat(img=imgs)
        s1_end.record()

        # Stage 2: BEV Encoder
        s2_start = cuda.Event(enable_timing=True)
        s2_end = cuda.Event(enable_timing=True)
        s2_start.record()
        if ego_model.freeze_bev_encoder:
            with torch.no_grad():
                bev_embed, bev_pos = ego_model.pts_bbox_head.get_bev_features(
                    mlvl_feats=img_feats, img_metas=img_metas, prev_bev=prev_bev)
        else:
            bev_embed, bev_pos = ego_model.pts_bbox_head.get_bev_features(
                mlvl_feats=img_feats, img_metas=img_metas, prev_bev=prev_bev)
        s2_end.record()

        cuda.synchronize()
        stage_times['backbone_ms'].append(s1_start.elapsed_time(s1_end))
        stage_times['bev_encoder_ms'].append(s2_start.elapsed_time(s2_end))

        if bev_embed.shape[1] == ego_model.bev_h * ego_model.bev_w:
            bev_embed = bev_embed.permute(1, 0, 2)
        assert bev_embed.shape[0] == ego_model.bev_h * ego_model.bev_w
        return bev_embed, bev_pos

    ego_model.get_bevs = timed_get_bevs
    return original_get_bevs, stage_times


def main():
    args = parse_args()
    print("[D2] Building model and dataset...")
    model, dataset, dl = build_model_and_data(args)
    print(f"[D2] Model loaded. Dataset: {len(dataset)} samples")

    ego = model.module.model_ego_agent
    original_get_bevs, stage_times = patch_get_bevs_with_timing(ego)

    # 整体 e2e 计时
    e2e_latencies = []
    powers = []
    torch.cuda.reset_peak_memory_stats()

    data_iter = iter(dl)
    with torch.no_grad():
        # Warmup
        for _ in range(args.n_warmup):
            data = next(data_iter, None) or next(iter(dl))
            model(return_loss=False, rescale=True, **data)
            cuda.synchronize()

        # 清掉 warmup 的计时
        stage_times['backbone_ms'].clear()
        stage_times['bev_encoder_ms'].clear()

        # Benchmark
        for i in range(args.n_runs):
            data = next(data_iter, None) or next(iter(dl))
            pw = get_gpu_power()

            start = cuda.Event(enable_timing=True)
            end = cuda.Event(enable_timing=True)
            start.record()
            model(return_loss=False, rescale=True, **data)
            end.record()
            cuda.synchronize()

            e2e_latencies.append(start.elapsed_time(end))
            powers.append(pw)

    # 恢复
    ego.get_bevs = original_get_bevs
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2

    # 统计
    backbone_mean = float(np.mean(stage_times['backbone_ms']))
    bev_mean = float(np.mean(stage_times['bev_encoder_ms']))
    e2e_mean = float(np.mean(e2e_latencies))
    other_mean = e2e_mean - backbone_mean - bev_mean  # decoder + seg + infra + overhead

    # 理论流水线计算
    # backbone-BEV 重叠: 稳态 = max(backbone, bev+decoder+seg+overhead)
    # 因为 backbone 远小于其他部分，backbone 被完全隐藏
    non_backbone = e2e_mean - backbone_mean
    theoretical_steady_state = max(backbone_mean, non_backbone)
    theoretical_speedup = e2e_mean / theoretical_steady_state if theoretical_steady_state > 0 else 1

    results = {
        'e2e_mean_ms': round(e2e_mean, 1),
        'e2e_std_ms': round(float(np.std(e2e_latencies)), 1),
        'backbone_mean_ms': round(backbone_mean, 1),
        'bev_encoder_mean_ms': round(bev_mean, 1),
        'other_mean_ms': round(other_mean, 1),
        'non_backbone_mean_ms': round(non_backbone, 1),
        'theoretical_steady_state_ms': round(theoretical_steady_state, 1),
        'theoretical_speedup': round(theoretical_speedup, 2),
        'backbone_pct_of_e2e': round(backbone_mean / e2e_mean * 100, 1),
        'peak_memory_mb': round(peak_mem, 0),
        'avg_power_w': round(float(np.mean(powers)), 1),
        'n_runs': args.n_runs,
        'prune_config': args.prune_config,
    }

    print(f"\n{'='*60}")
    print(f"[D2] Pipeline Stage Timing Results")
    print(f"{'='*60}")
    print(f"  E2E latency:       {results['e2e_mean_ms']:.1f} +/- {results['e2e_std_ms']:.1f} ms")
    print(f"  Backbone (stage1): {results['backbone_mean_ms']:.1f} ms ({results['backbone_pct_of_e2e']:.1f}% of e2e)")
    print(f"  BEV encoder:       {results['bev_encoder_mean_ms']:.1f} ms")
    print(f"  Other (dec+seg+infra+overhead): {results['other_mean_ms']:.1f} ms")
    print(f"  Non-backbone total: {results['non_backbone_mean_ms']:.1f} ms")
    print(f"  ---")
    print(f"  Theoretical steady-state (backbone-BEV overlap): {results['theoretical_steady_state_ms']:.1f} ms")
    print(f"  Theoretical speedup: {results['theoretical_speedup']:.2f}x")
    print(f"  Peak memory: {results['peak_memory_mb']:.0f} MB")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n[D2] Saved to {args.output}")


if __name__ == '__main__':
    main()
