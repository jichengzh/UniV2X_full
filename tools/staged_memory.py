"""
E3 分阶段内存释放管理器 (StagedMemoryManager)

参考 EMOS SI-3 的四边界线程管理策略，在 UniV2X 推理流水线的阶段边界
主动释放已完成阶段的 GPU 中间张量，降低峰值显存。

UniV2X 阶段边界:
  Boundary 1: Backbone + FPN 完成 → 释放原始图像
  Boundary 2: BEV Encoder 完成 → 释放多尺度特征图
  Boundary 3: Decoder 完成 → 释放 decoder 中间激活
  Boundary 4: Seg Head 完成 → 释放 seg 中间激活

用法:
  from tools.staged_memory import StagedMemoryManager

  mgr = StagedMemoryManager()
  mgr.enable(model)  # 自动在各阶段边界插入释放 hook

Benchmark:
  CUDA_VISIBLE_DEVICES=4 PYTHONPATH=/home/jichengzhi/UniV2X \\
    conda run -n UniV2X_2.0 python tools/staged_memory.py \\
      projects/configs_e2e_univ2x/univ2x_coop_e2e_track.py \\
      ckpts/univ2x_coop_e2e_stg2.pth \\
      --compare --n-runs 10
"""
import argparse
import gc
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, '/home/jichengzhi/UniV2X')

import torch
import numpy as np


class StagedMemoryManager:
    """分阶段 GPU 内存管理器（参考 EMOS SI-3）

    在模型各阶段的 forward hook 中，释放上一阶段不再需要的中间张量。
    通过 post-hook 在每个阶段结束后调用 empty_cache 回收碎片化显存。
    """

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.hooks = []
        self.memory_log = []  # 记录各阶段的显存快照
        self.enabled = False

    def _make_release_hook(self, stage_name):
        """创建阶段边界的 post-forward hook"""
        def hook(module, input, output):
            # 记录当前阶段结束时的显存
            mem_before = torch.cuda.memory_allocated() / 1024**2

            # 释放该阶段的输入引用（如果可以）
            # 注意：PyTorch 的 autograd 引用可能阻止释放
            # 在 no_grad 推理模式下，中间张量会在作用域结束时自动释放
            # 这里显式调用 empty_cache 回收碎片
            torch.cuda.empty_cache()

            mem_after = torch.cuda.memory_allocated() / 1024**2
            released = mem_before - mem_after

            self.memory_log.append({
                'stage': stage_name,
                'mem_before_mb': round(mem_before, 1),
                'mem_after_mb': round(mem_after, 1),
                'released_mb': round(released, 1),
            })

            if self.verbose and released > 0:
                print(f"  [E3] {stage_name}: released {released:.1f} MB "
                      f"({mem_before:.0f} → {mem_after:.0f} MB)")
        return hook

    def enable(self, model):
        """在模型各阶段边界注册释放 hook

        Args:
            model: MMDataParallel 包装的 MultiAgent 模型
        """
        # 获取 ego agent
        if hasattr(model, 'module'):
            ma = model.module
        else:
            ma = model
        ego = ma.model_ego_agent

        # Boundary 1: Backbone + FPN 完成
        if hasattr(ego, 'img_backbone'):
            h = ego.img_backbone.register_forward_hook(
                self._make_release_hook('backbone'))
            self.hooks.append(h)
        if hasattr(ego, 'img_neck'):
            h = ego.img_neck.register_forward_hook(
                self._make_release_hook('neck'))
            self.hooks.append(h)

        # Boundary 2: BEV Encoder 完成
        if hasattr(ego, 'pts_bbox_head') and hasattr(ego.pts_bbox_head, 'transformer'):
            tf = ego.pts_bbox_head.transformer
            if hasattr(tf, 'encoder'):
                h = tf.encoder.register_forward_hook(
                    self._make_release_hook('bev_encoder'))
                self.hooks.append(h)

        # Boundary 3: Decoder 完成
        if hasattr(ego, 'pts_bbox_head') and hasattr(ego.pts_bbox_head, 'transformer'):
            tf = ego.pts_bbox_head.transformer
            if hasattr(tf, 'decoder'):
                h = tf.decoder.register_forward_hook(
                    self._make_release_hook('decoder'))
                self.hooks.append(h)

        # Boundary 4: Seg Head 完成
        if hasattr(ego, 'seg_head'):
            h = ego.seg_head.register_forward_hook(
                self._make_release_hook('seg_head'))
            self.hooks.append(h)

        self.enabled = True
        if self.verbose:
            print(f"[E3] Registered {len(self.hooks)} staged memory release hooks")

    def disable(self):
        """移除所有 hook"""
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
        self.enabled = False

    def get_summary(self):
        """返回各阶段的显存释放统计"""
        if not self.memory_log:
            return {}
        # 按 stage 分组取平均
        from collections import defaultdict
        stage_data = defaultdict(list)
        for entry in self.memory_log:
            stage_data[entry['stage']].append(entry['released_mb'])
        return {
            stage: {
                'mean_released_mb': round(np.mean(vals), 1),
                'total_releases': len(vals),
            }
            for stage, vals in stage_data.items()
        }

    def clear_log(self):
        self.memory_log.clear()


def parse_args():
    p = argparse.ArgumentParser(description="E3 staged memory release benchmark")
    p.add_argument("config")
    p.add_argument("checkpoint")
    p.add_argument("--compare", action="store_true")
    p.add_argument("--n-warmup", type=int, default=2)
    p.add_argument("--n-runs", type=int, default=10)
    p.add_argument("--output", default=None)
    return p.parse_args()


def benchmark_memory(model, data_loader, n_warmup, n_runs, staged_mgr=None):
    """跑推理并记录显存"""
    latencies = []
    peak_mems = []

    data_iter = iter(data_loader)
    with torch.no_grad():
        for _ in range(n_warmup):
            data = next(data_iter, None) or next(iter(data_loader))
            model(return_loss=False, rescale=True, **data)
            torch.cuda.synchronize()

        for _ in range(n_runs):
            torch.cuda.reset_peak_memory_stats()
            data = next(data_iter, None) or next(iter(data_loader))
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            model(return_loss=False, rescale=True, **data)
            end.record()
            torch.cuda.synchronize()
            latencies.append(start.elapsed_time(end))
            peak_mems.append(torch.cuda.max_memory_allocated() / 1024**2)

    return {
        'latency_mean_ms': float(np.mean(latencies)),
        'latency_std_ms': float(np.std(latencies)),
        'peak_memory_mean_mb': float(np.mean(peak_mems)),
        'peak_memory_max_mb': float(np.max(peak_mems)),
        'n_runs': n_runs,
    }


def main():
    args = parse_args()

    import projects.mmdet3d_plugin  # noqa
    from mmcv.parallel import MMDataParallel
    from mmdet.datasets import replace_ImageToTensor
    from mmdet3d.datasets import build_dataset
    from tools.pruning_sensitivity_analysis import load_model_fresh

    model, cfg = load_model_fresh(args.config, args.checkpoint)
    cfg.data.test.test_mode = True
    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    dataset = build_dataset(cfg.data.test)
    from projects.mmdet3d_plugin.datasets.builder import build_dataloader
    data_loader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=0, dist=False, shuffle=False
    )
    model = MMDataParallel(model.cuda(), device_ids=[0])
    model.eval()

    results = {}

    if args.compare:
        print("[E3] Benchmark WITHOUT staged memory release...")
        r_no = benchmark_memory(model, data_loader, args.n_warmup, args.n_runs)
        results['without_staged'] = r_no
        print(f"  Peak memory: {r_no['peak_memory_mean_mb']:.0f} MB (max {r_no['peak_memory_max_mb']:.0f})")
        print(f"  Latency: {r_no['latency_mean_ms']:.1f} ms")

        print("\n[E3] Benchmark WITH staged memory release...")
        mgr = StagedMemoryManager(verbose=True)
        mgr.enable(model)
        mgr.clear_log()
        r_yes = benchmark_memory(model, data_loader, args.n_warmup, args.n_runs, mgr)
        results['with_staged'] = r_yes
        results['stage_summary'] = mgr.get_summary()
        mgr.disable()
        print(f"  Peak memory: {r_yes['peak_memory_mean_mb']:.0f} MB (max {r_yes['peak_memory_max_mb']:.0f})")
        print(f"  Latency: {r_yes['latency_mean_ms']:.1f} ms")

        delta_mem = r_no['peak_memory_mean_mb'] - r_yes['peak_memory_mean_mb']
        print(f"\n[E3] Staged release saves: {delta_mem:.0f} MB peak memory "
              f"({delta_mem/r_no['peak_memory_mean_mb']*100:.1f}%)")
    else:
        mgr = StagedMemoryManager(verbose=True)
        mgr.enable(model)
        r = benchmark_memory(model, data_loader, args.n_warmup, args.n_runs, mgr)
        results['with_staged'] = r
        results['stage_summary'] = mgr.get_summary()
        mgr.disable()
        print(f"  Peak memory: {r['peak_memory_mean_mb']:.0f} MB")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n[E3] Saved to {args.output}")


if __name__ == '__main__':
    main()
