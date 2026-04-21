"""
E2 数据异步预取器 (AsyncDataPrefetcher)

在当前帧 GPU 推理期间，CPU 异步加载并预处理下一帧数据。
实现 CPU-GPU 重叠，隐藏数据加载延迟。

这是 D2 流水线重叠的前置基础设施：
  E2: CPU 数据加载 与 GPU 推理并行  → CPU-GPU 重叠
  D2: GPU backbone(t+1) 与 GPU BEV(t) 并行 → GPU-GPU 重叠
  D2 需要 frame t+1 的数据提前在 GPU 上 → 这正是 E2 的功能

用法:
  from tools.async_prefetcher import AsyncDataPrefetcher

  prefetcher = AsyncDataPrefetcher(data_loader)
  data = prefetcher.get_next()
  while data is not None:
      result = model(**data)
      data = prefetcher.get_next()  # 在 model forward 结束时下一帧已在 GPU 上

Benchmark 用法:
  CUDA_VISIBLE_DEVICES=4 PYTHONPATH=/home/jichengzhi/UniV2X \\
    conda run -n UniV2X_2.0 python tools/async_prefetcher.py \\
      projects/configs_e2e_univ2x/univ2x_coop_e2e_track.py \\
      ckpts/univ2x_coop_e2e_stg2.pth \\
      --n-runs 20 --compare
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


class AsyncDataPrefetcher:
    """异步数据预取器

    在当前帧 GPU 推理期间，使用独立 CUDA stream 将下一帧数据
    从 CPU 异步搬运到 GPU（non_blocking=True）。

    工作原理:
      1. 初始化时预取第一帧数据
      2. get_next() 返回当前预取好的数据，同时启动下一帧预取
      3. GPU 推理与 CPU→GPU 数据传输重叠
    """

    def __init__(self, data_loader):
        self.loader = data_loader
        self.loader_iter = iter(data_loader)
        self.stream = cuda.Stream()
        self.next_data = None
        self._prefetch()

    def _prefetch(self):
        """异步预取下一帧数据到 GPU"""
        try:
            self.next_data = next(self.loader_iter)
        except StopIteration:
            self.next_data = None
            return

        with cuda.stream(self.stream):
            self._to_cuda_recursive(self.next_data)

    def _to_cuda_recursive(self, data):
        """递归地将嵌套 dict/list 中的 tensor 搬到 GPU"""
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    data[k] = v.cuda(non_blocking=True)
                elif isinstance(v, (dict, list)):
                    self._to_cuda_recursive(v)
        elif isinstance(data, list):
            for i, v in enumerate(data):
                if isinstance(v, torch.Tensor):
                    data[i] = v.cuda(non_blocking=True)
                elif isinstance(v, (dict, list)):
                    self._to_cuda_recursive(v)

    def get_next(self):
        """获取预取好的当前帧数据，同时启动下一帧预取

        Returns:
            当前帧数据 (已在 GPU 上), 或 None (数据耗尽)
        """
        cuda.current_stream().wait_stream(self.stream)
        data = self.next_data
        if data is not None:
            self._prefetch()
        return data

    def reset(self):
        """重置迭代器（用于多轮评估）"""
        self.loader_iter = iter(self.loader)
        self.next_data = None
        self._prefetch()


def get_gpu_power():
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        return float(out.split('\n')[0])
    except Exception:
        return -1.0


def benchmark_with_prefetch(model, data_loader, n_warmup, n_runs, use_prefetch):
    """对比有/无预取的推理性能"""
    latencies = []
    powers = []
    torch.cuda.reset_peak_memory_stats()

    if use_prefetch:
        prefetcher = AsyncDataPrefetcher(data_loader)
        with torch.no_grad():
            # Warmup
            for _ in range(n_warmup):
                data = prefetcher.get_next()
                if data is None:
                    prefetcher.reset()
                    data = prefetcher.get_next()
                model(return_loss=False, rescale=True, **data)
                torch.cuda.synchronize()

            # Benchmark
            for _ in range(n_runs):
                data = prefetcher.get_next()
                if data is None:
                    prefetcher.reset()
                    data = prefetcher.get_next()
                pw = get_gpu_power()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                model(return_loss=False, rescale=True, **data)
                end.record()
                torch.cuda.synchronize()
                latencies.append(start.elapsed_time(end))
                powers.append(pw)
    else:
        data_iter = iter(data_loader)
        with torch.no_grad():
            for _ in range(n_warmup):
                data = next(data_iter, None) or next(iter(data_loader))
                model(return_loss=False, rescale=True, **data)
                torch.cuda.synchronize()

            for _ in range(n_runs):
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

    peak_mem = torch.cuda.max_memory_allocated() / 1024**2
    return {
        'latency_mean_ms': float(np.mean(latencies)),
        'latency_std_ms': float(np.std(latencies)),
        'peak_memory_mb': float(peak_mem),
        'avg_power_w': float(np.mean(powers)),
        'energy_mj': float(np.mean(latencies) * np.mean(powers)),
        'n_runs': n_runs,
    }


def parse_args():
    p = argparse.ArgumentParser(description="E2 async data prefetcher benchmark")
    p.add_argument("config")
    p.add_argument("checkpoint")
    p.add_argument("--n-warmup", type=int, default=3)
    p.add_argument("--n-runs", type=int, default=20)
    p.add_argument("--compare", action="store_true", help="对比有/无预取")
    p.add_argument("--output", default=None)
    return p.parse_args()


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
        print("[E2] Benchmark WITHOUT prefetch...")
        r_no = benchmark_with_prefetch(model, data_loader, args.n_warmup, args.n_runs, False)
        results['no_prefetch'] = r_no
        print(f"  Latency: {r_no['latency_mean_ms']:.1f} +/- {r_no['latency_std_ms']:.1f} ms")
        print(f"  Peak memory: {r_no['peak_memory_mb']:.0f} MB")

        torch.cuda.reset_peak_memory_stats()
        print("\n[E2] Benchmark WITH prefetch...")
        r_yes = benchmark_with_prefetch(model, data_loader, args.n_warmup, args.n_runs, True)
        results['with_prefetch'] = r_yes
        print(f"  Latency: {r_yes['latency_mean_ms']:.1f} +/- {r_yes['latency_std_ms']:.1f} ms")
        print(f"  Peak memory: {r_yes['peak_memory_mb']:.0f} MB")

        delta = r_no['latency_mean_ms'] - r_yes['latency_mean_ms']
        print(f"\n[E2] Prefetch saves: {delta:.1f} ms/frame ({delta/r_no['latency_mean_ms']*100:.1f}%)")
    else:
        print("[E2] Benchmark WITH prefetch...")
        r = benchmark_with_prefetch(model, data_loader, args.n_warmup, args.n_runs, True)
        results['with_prefetch'] = r
        print(f"  Latency: {r['latency_mean_ms']:.1f} +/- {r['latency_std_ms']:.1f} ms")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n[E2] Saved to {args.output}")


if __name__ == '__main__':
    main()
