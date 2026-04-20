"""
D4 显存分配策略 benchmark

测量不同 PyTorch CUDA 显存分配策略对时延稳定性和峰值显存的影响。
支持: 动态分配 / 碎片整理 (expandable_segments)。

注意: 静态预分配在 PyTorch 中通过 set_per_process_memory_fraction 实现,
     但需要在所有 CUDA 操作之前设置。本脚本在模型加载前配置。

用法:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/jichengzhi/UniV2X \\
    conda run -n UniV2X_2.0 python tools/memory_strategy.py \\
      projects/configs_e2e_univ2x/univ2x_coop_e2e_track.py \\
      ckpts/univ2x_coop_e2e_stg2.pth \\
      --strategy dynamic \\
      --n-warmup 3 --n-runs 20
"""
import argparse
import json
import os
import subprocess
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, '/home/jichengzhi/UniV2X')


def parse_args():
    p = argparse.ArgumentParser(description="D4 memory strategy benchmark")
    p.add_argument("config")
    p.add_argument("checkpoint")
    p.add_argument("--prune-config", default=None)
    p.add_argument("--finetuned-ckpt", default=None)
    p.add_argument("--strategy", choices=['dynamic', 'defrag', 'static', 'all'],
                   default='dynamic')
    p.add_argument("--static-budget-mb", type=int, default=None,
                   help="静态预分配的显存预算 (MB)")
    p.add_argument("--n-warmup", type=int, default=3)
    p.add_argument("--n-runs", type=int, default=20)
    p.add_argument("--output", default=None)
    return p.parse_args()


def setup_memory_strategy(strategy, budget_mb=None):
    """在任何 CUDA 操作之前配置显存策略"""
    import torch

    if strategy == 'dynamic':
        pass  # PyTorch 默认

    elif strategy == 'defrag':
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    elif strategy == 'static':
        if budget_mb is not None:
            total_mem = torch.cuda.get_device_properties(0).total_memory
            fraction = budget_mb * 1024 * 1024 / total_mem
            fraction = min(fraction, 0.95)  # 不超过 95%
            torch.cuda.set_per_process_memory_fraction(fraction)
            print(f"[D4] Static memory: {budget_mb}MB = {fraction:.2%} of total")
        else:
            # 默认使用 80% GPU 显存
            torch.cuda.set_per_process_memory_fraction(0.8)
            print("[D4] Static memory: 80% of total")

    print(f"[D4] Memory strategy: {strategy}")


def benchmark_strategy(model, data_loader, n_warmup, n_runs):
    """运行推理并收集时延稳定性 + 显存统计"""
    import torch
    import numpy as np

    latencies = []
    memory_allocated = []
    memory_reserved = []

    torch.cuda.reset_peak_memory_stats()

    data_iter = iter(data_loader)
    with torch.no_grad():
        # Warmup
        for _ in range(n_warmup):
            data = next(data_iter, None) or next(iter(data_loader))
            model(return_loss=False, rescale=True, **data)
            torch.cuda.synchronize()

        # Benchmark
        for i in range(n_runs):
            data = next(data_iter, None) or next(iter(data_loader))

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            torch.cuda.synchronize()
            start.record()
            model(return_loss=False, rescale=True, **data)
            end.record()
            torch.cuda.synchronize()

            latencies.append(start.elapsed_time(end))
            memory_allocated.append(torch.cuda.memory_allocated() / 1024**2)
            memory_reserved.append(torch.cuda.memory_reserved() / 1024**2)

    peak_allocated = torch.cuda.max_memory_allocated() / 1024**2
    peak_reserved = torch.cuda.max_memory_reserved() / 1024**2

    return {
        'latency_mean_ms': float(np.mean(latencies)),
        'latency_std_ms': float(np.std(latencies)),
        'latency_min_ms': float(np.min(latencies)),
        'latency_max_ms': float(np.max(latencies)),
        'latency_p50_ms': float(np.median(latencies)),
        'latency_p90_ms': float(np.percentile(latencies, 90)),
        'latency_p99_ms': float(np.percentile(latencies, 99)) if n_runs >= 100 else None,
        'peak_allocated_mb': float(peak_allocated),
        'peak_reserved_mb': float(peak_reserved),
        'memory_waste_mb': float(peak_reserved - peak_allocated),
        'memory_waste_pct': float((peak_reserved - peak_allocated) / peak_reserved * 100)
            if peak_reserved > 0 else 0,
        'allocated_std_mb': float(np.std(memory_allocated)),
        'n_runs': n_runs,
    }


def main():
    args = parse_args()

    import numpy as np
    import torch
    from mmcv.parallel import MMDataParallel
    from mmdet.datasets import replace_ImageToTensor
    from mmdet3d.datasets import build_dataset
    import projects.mmdet3d_plugin  # noqa

    if args.strategy == 'all':
        strategies = ['dynamic', 'defrag']
    else:
        strategies = [args.strategy]

    all_results = {}
    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"[D4] Testing strategy: {strategy}")
        print(f"{'='*60}")

        # 策略需要在模型加载前设置
        # 但 defrag 只需设置环境变量, 可以在运行时切换
        if strategy == 'defrag':
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        elif strategy == 'dynamic':
            os.environ.pop('PYTORCH_CUDA_ALLOC_CONF', None)

        # 重建模型 (确保 allocator 生效)
        from tools.pruning_sensitivity_analysis import load_model_fresh, get_prune_target
        model, cfg = load_model_fresh(args.config, args.checkpoint)

        if args.prune_config:
            import json as json_mod
            with open(args.prune_config) as f:
                prune_cfg = json_mod.load(f)
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

        if args.finetuned_ckpt:
            from mmcv.runner import load_checkpoint
            load_checkpoint(model, args.finetuned_ckpt, map_location="cpu")

        cfg.data.test.test_mode = True
        cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
        dataset = build_dataset(cfg.data.test)
        from projects.mmdet3d_plugin.datasets.builder import build_dataloader
        data_loader = build_dataloader(
            dataset, samples_per_gpu=1, workers_per_gpu=0, dist=False, shuffle=False
        )

        model = MMDataParallel(model.cuda(), device_ids=[0])
        model.eval()

        result = benchmark_strategy(model, data_loader, args.n_warmup, args.n_runs)
        result['strategy'] = strategy
        all_results[strategy] = result

        print(f"  Latency: {result['latency_mean_ms']:.1f} +/- {result['latency_std_ms']:.1f} ms")
        print(f"  Latency p50/p90: {result['latency_p50_ms']:.1f} / {result['latency_p90_ms']:.1f} ms")
        print(f"  Peak allocated: {result['peak_allocated_mb']:.0f} MB")
        print(f"  Peak reserved: {result['peak_reserved_mb']:.0f} MB")
        print(f"  Memory waste: {result['memory_waste_mb']:.0f} MB ({result['memory_waste_pct']:.1f}%)")

        # 清理
        del model
        torch.cuda.empty_cache()

    output = {
        'config': args.config,
        'prune_config': args.prune_config,
        'results': all_results
    }

    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\n[D4] Results saved to {args.output}")
    else:
        print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
