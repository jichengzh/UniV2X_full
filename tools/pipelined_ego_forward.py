"""
D2 流水线重叠的真正实现：PipelinedEgoForward

通过 monkey-patch get_bevs() 方法，实现帧间 backbone-BEV 重叠：
  - 当前帧 BEV encoding 时，同时在独立 CUDA stream 上执行下一帧的 backbone
  - 下一帧到来时直接使用已缓存的 backbone 特征

三种模式:
  none:              原始串行 (无重叠)
  backbone_bev:      backbone(t+1) 与 BEV+Dec+Seg(t) 重叠
  full:              全阶段重叠（需要更复杂的拆分，当前降级为 backbone_bev）

用法:
  CUDA_VISIBLE_DEVICES=4 PYTHONPATH=/home/jichengzhi/UniV2X \\
    conda run -n UniV2X_2.0 python tools/pipelined_ego_forward.py \\
      projects/configs_e2e_univ2x/univ2x_coop_e2e_track.py \\
      ckpts/univ2x_coop_e2e_stg2.pth \\
      --mode backbone_bev --n-warmup 3 --n-runs 10 --compare
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

import torch
import torch.cuda as cuda
import numpy as np

import projects.mmdet3d_plugin  # noqa


def parse_args():
    p = argparse.ArgumentParser(description="D2 pipelined ego forward benchmark")
    p.add_argument("config")
    p.add_argument("checkpoint")
    p.add_argument("--prune-config", default=None)
    p.add_argument("--finetuned-ckpt", default=None)
    p.add_argument("--mode", choices=['none', 'backbone_bev', 'full', 'compare'],
                   default='compare')
    p.add_argument("--n-warmup", type=int, default=3)
    p.add_argument("--n-runs", type=int, default=10)
    p.add_argument("--output", default=None)
    return p.parse_args()


class PipelinedGetBevs:
    """在 backbone(t+1) 和 BEV(t) 之间实现帧间重叠

    原理:
      get_bevs() 内部分两步:
        1. extract_img_feat(img)      → img_feats   [backbone, ~32ms]
        2. get_bev_features(img_feats) → bev_embed  [BEV encoder, ~65ms]

      backbone_bev 重叠模式:
        - 保存当前帧 BEV encoding 完成后的状态
        - 在返回前，异步启动下一帧的 backbone (如果下一帧数据可用)
        - 下次调用 get_bevs 时，直接使用缓存的 backbone 结果
    """

    def __init__(self, ego_model, mode='none'):
        self.ego = ego_model
        self.mode = mode
        self.backbone_stream = cuda.Stream() if mode != 'none' else None
        self.cached_img_feats = None  # 缓存的下一帧 backbone 结果
        self.next_img = None  # 下一帧图像（由外部设置）
        self._original_get_bevs = ego_model.get_bevs

    def set_next_frame_img(self, img):
        """外部调用: 设置下一帧图像，以便异步预取 backbone"""
        self.next_img = img

    def patched_get_bevs(self, imgs, img_metas, prev_img=None, prev_img_metas=None, prev_bev=None):
        """替代原始 get_bevs()，加入流水线重叠逻辑"""

        if self.mode == 'none':
            return self._original_get_bevs(imgs, img_metas, prev_img, prev_img_metas, prev_bev)

        # Step 1: 获取当前帧的 backbone 特征
        if self.cached_img_feats is not None:
            # 使用上一帧预取的 backbone 结果
            img_feats = self.cached_img_feats
            self.cached_img_feats = None
        else:
            # 首帧或缓存不可用，同步执行 backbone
            if prev_img is not None and prev_img_metas is not None:
                prev_bev = self.ego.get_history_bev(prev_img, prev_img_metas)
            img_feats = self.ego.extract_img_feat(img=imgs)

        # Step 2: BEV encoding（在默认 stream 上）
        if self.ego.freeze_bev_encoder:
            with torch.no_grad():
                bev_embed, bev_pos = self.ego.pts_bbox_head.get_bev_features(
                    mlvl_feats=img_feats, img_metas=img_metas, prev_bev=prev_bev)
        else:
            bev_embed, bev_pos = self.ego.pts_bbox_head.get_bev_features(
                mlvl_feats=img_feats, img_metas=img_metas, prev_bev=prev_bev)

        # Step 3: 异步预取下一帧 backbone（在独立 stream 上）
        if self.mode in ('backbone_bev', 'full') and self.next_img is not None:
            with cuda.stream(self.backbone_stream):
                self.cached_img_feats = self.ego.extract_img_feat(img=self.next_img)
            self.next_img = None  # 消费掉

        # Step 4: BEV 形状处理（与原始 get_bevs 一致）
        if bev_embed.shape[1] == self.ego.bev_h * self.ego.bev_w:
            bev_embed = bev_embed.permute(1, 0, 2)
        assert bev_embed.shape[0] == self.ego.bev_h * self.ego.bev_w

        # 等待下一帧 backbone 完成（如果有）
        if self.backbone_stream is not None and self.cached_img_feats is not None:
            cuda.current_stream().wait_stream(self.backbone_stream)

        return bev_embed, bev_pos

    def enable(self):
        """激活流水线重叠"""
        self.ego.get_bevs = self.patched_get_bevs

    def disable(self):
        """恢复原始 get_bevs"""
        self.ego.get_bevs = self._original_get_bevs
        self.cached_img_feats = None
        self.next_img = None


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


def benchmark_pipeline_mode(model, data_loader, mode, n_warmup, n_runs):
    """Benchmark 一种流水线模式"""
    ego = model.module.model_ego_agent
    pipeline = PipelinedGetBevs(ego, mode=mode)
    pipeline.enable()

    latencies = []
    powers = []
    torch.cuda.reset_peak_memory_stats()

    data_iter = iter(data_loader)
    next_data = None

    with torch.no_grad():
        # Warmup
        for _ in range(n_warmup):
            data = next(data_iter, None) or next(iter(data_loader))
            model(return_loss=False, rescale=True, **data)
            torch.cuda.synchronize()

        # Pre-fetch first "next frame" for pipeline mode
        if mode != 'none':
            peek_data = next(data_iter, None) or next(iter(data_loader))
            # 提取下一帧的 img tensor
            if 'ego_agent_data' in peek_data and 'img' in peek_data['ego_agent_data']:
                pipeline.set_next_frame_img(peek_data['ego_agent_data']['img'])
            next_data = peek_data

        # Benchmark
        for i in range(n_runs):
            if next_data is not None:
                data = next_data
            else:
                data = next(data_iter, None) or next(iter(data_loader))

            # 准备下一帧数据
            next_data = next(data_iter, None) or next(iter(data_loader))
            if mode != 'none' and 'ego_agent_data' in next_data and 'img' in next_data['ego_agent_data']:
                pipeline.set_next_frame_img(next_data['ego_agent_data']['img'])

            pw = get_gpu_power()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            model(return_loss=False, rescale=True, **data)
            end.record()
            torch.cuda.synchronize()

            latencies.append(start.elapsed_time(end))
            powers.append(pw)

    pipeline.disable()
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2

    return {
        'mode': mode,
        'latency_mean_ms': float(np.mean(latencies)),
        'latency_std_ms': float(np.std(latencies)),
        'peak_memory_mb': float(peak_mem),
        'avg_power_w': float(np.mean(powers)),
        'energy_mj': float(np.mean(latencies) * np.mean(powers)),
        'n_runs': n_runs,
    }


def main():
    args = parse_args()
    print("[D2] Building model and dataset...")
    model, dataset, dl = build_model_and_data(args)
    print(f"[D2] Model loaded. Dataset: {len(dataset)} samples")

    if args.mode == 'compare':
        modes = ['none', 'backbone_bev']
    else:
        modes = [args.mode]

    results = {}
    for m in modes:
        print(f"\n{'='*60}")
        print(f"[D2] Benchmarking mode={m}")
        print(f"{'='*60}")

        torch.cuda.reset_peak_memory_stats()
        r = benchmark_pipeline_mode(model, dl, m, args.n_warmup, args.n_runs)
        results[m] = r

        print(f"  Latency: {r['latency_mean_ms']:.1f} +/- {r['latency_std_ms']:.1f} ms")
        print(f"  Peak memory: {r['peak_memory_mb']:.0f} MB")
        print(f"  Power: {r['avg_power_w']:.1f} W")

    if len(results) == 2:
        r_none = results['none']
        r_pipe = results['backbone_bev']
        delta = r_none['latency_mean_ms'] - r_pipe['latency_mean_ms']
        print(f"\n[D2] Pipeline saves: {delta:.1f} ms ({delta/r_none['latency_mean_ms']*100:.1f}%)")
        print(f"     Memory overhead: {r_pipe['peak_memory_mb'] - r_none['peak_memory_mb']:.0f} MB")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n[D2] Saved to {args.output}")


if __name__ == '__main__':
    main()
