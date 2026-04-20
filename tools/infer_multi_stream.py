"""
D1 多 Agent 并行度 benchmark

测量不同 CUDA stream 并行度下的推理时延/显存/功率。
支持 1/2 stream 配置（2=ego/infra BEV 并行, 3/4 需侵入模型内部，标记为 TODO）。

用法:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/jichengzhi/UniV2X \\
    conda run -n UniV2X_2.0 python tools/infer_multi_stream.py \\
      projects/configs_e2e_univ2x/univ2x_coop_e2e_track.py \\
      ckpts/univ2x_coop_e2e_stg2.pth \\
      --sweep --n-warmup 2 --n-runs 5 \\
      --output output/d1_baseline_sweep.json
"""
import argparse
import json
import os
import subprocess
import sys
import time
import warnings
from collections import defaultdict
from contextlib import contextmanager

warnings.filterwarnings("ignore")
sys.path.insert(0, '/home/jichengzhi/UniV2X')

import numpy as np
import torch
import torch.cuda as cuda
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmdet.datasets import replace_ImageToTensor
from mmdet3d.datasets import build_dataset

# 注册自定义数据集和模型
import projects.mmdet3d_plugin  # noqa: F401

from projects.mmdet3d_plugin.univ2x.detectors.multi_agent import MultiAgent


def parse_args():
    p = argparse.ArgumentParser(description="D1 multi-stream benchmark")
    p.add_argument("config")
    p.add_argument("checkpoint")
    p.add_argument("--prune-config", default=None)
    p.add_argument("--finetuned-ckpt", default=None)
    p.add_argument("--num-streams", type=int, default=1,
                   help="并行 stream 数: 1=串行, 2=ego/infra并行")
    p.add_argument("--sweep", action="store_true",
                   help="遍历 1/2 全部配置")
    p.add_argument("--n-warmup", type=int, default=3)
    p.add_argument("--n-runs", type=int, default=20)
    p.add_argument("--output", default=None)
    return p.parse_args()


# ============================================================
# Monkey-patch MultiAgent to support multi-stream forward_test
# ============================================================

def forward_test_serial(self, ego_agent_data=None, other_agent_data_dict={},
                        w_label=True, return_loss=False, **kwargs):
    """stream=1: 原始串行逻辑 (与 multi_agent.py 一致)"""
    other_agent_results = {}
    for name_other_agent in self.other_agent_names:
        other_agent_result = getattr(self, name_other_agent)(
            return_loss=return_loss,
            w_label=w_label,
            **(other_agent_data_dict[name_other_agent]),
            **kwargs
        )
        other_agent_result[0]['ego2other_rt'] = other_agent_data_dict[name_other_agent]['veh2inf_rt']
        other_agent_result[0]['pc_range'] = self.pc_range_dict[name_other_agent]
        other_agent_results[name_other_agent] = other_agent_result

    result = self.model_ego_agent(
        return_loss=return_loss, w_label=w_label,
        other_agent_results=other_agent_results,
        **ego_agent_data, **kwargs
    )
    return result


def forward_test_2stream(self, ego_agent_data=None, other_agent_data_dict={},
                         w_label=True, return_loss=False, **kwargs):
    """stream=2: infra 在独立 stream 上并行执行

    原理:
      - infra agent 的完整 forward 不依赖 ego 的任何输出
      - ego 只在 fusion 阶段(decoder 之前)需要 infra 的 track queries
      - 但当前 ego forward 是一个不可拆分的调用, 所以实际上:
        Stream 1: infra forward (异步启动)
        Stream 0: 等 infra 完成 → ego forward (含 fusion)
      - 真正的收益需要拆分 ego forward 为 backbone+BEV 和 decoder 两阶段
        (这是 D2 流水线重叠的工作)

    当前实现: infra 异步启动 + 同步后 ego forward
    这可以测量多 stream 的显存开销和 GPU 资源竞争影响。
    """
    stream_infra = cuda.Stream()
    other_agent_results = {}

    # 在独立 stream 上异步启动 infra
    for name_other_agent in self.other_agent_names:
        with cuda.stream(stream_infra):
            other_agent_result = getattr(self, name_other_agent)(
                return_loss=return_loss,
                w_label=w_label,
                **(other_agent_data_dict[name_other_agent]),
                **kwargs
            )
            other_agent_result[0]['ego2other_rt'] = \
                other_agent_data_dict[name_other_agent]['veh2inf_rt']
            other_agent_result[0]['pc_range'] = self.pc_range_dict[name_other_agent]
            other_agent_results[name_other_agent] = other_agent_result

    # 同步: 等 infra 完成
    cuda.current_stream().wait_stream(stream_infra)

    # ego forward (含 fusion)
    result = self.model_ego_agent(
        return_loss=return_loss, w_label=w_label,
        other_agent_results=other_agent_results,
        **ego_agent_data, **kwargs
    )
    return result


# ============================================================
# Model + Data construction
# ============================================================

def build_model_and_data(args):
    """构建模型和数据集"""
    from tools.pruning_sensitivity_analysis import load_model_fresh, get_prune_target

    model, cfg = load_model_fresh(args.config, args.checkpoint)

    # 可选: 剪枝
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

    # 可选: 加载微调 ckpt
    if args.finetuned_ckpt:
        from mmcv.runner import load_checkpoint
        load_checkpoint(model, args.finetuned_ckpt, map_location="cpu")
        print(f"[Finetune] Loaded {args.finetuned_ckpt}")

    # 构建数据集
    cfg.data.test.test_mode = True
    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    dataset = build_dataset(cfg.data.test)

    from projects.mmdet3d_plugin.datasets.builder import build_dataloader
    data_loader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=0,
        dist=False, shuffle=False
    )

    # 包装为 MMDataParallel (处理 DataContainer scatter)
    model = MMDataParallel(model.cuda(), device_ids=[0])
    model.eval()

    return model, dataset, data_loader


# ============================================================
# Patch and benchmark
# ============================================================

def get_gpu_power():
    """读取当前 GPU 功率 (W)"""
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        return float(out.split('\n')[0])
    except Exception:
        return -1.0


@contextmanager
def patch_forward_test(model, num_streams):
    """临时替换 MultiAgent.forward_test 为指定的 stream 策略"""
    # model 是 MMDataParallel 包装的, 实际 MultiAgent 在 model.module
    ma = model.module
    original = ma.forward_test

    if num_streams == 1:
        ma.forward_test = lambda **kw: forward_test_serial(ma, **kw)
    elif num_streams >= 2:
        ma.forward_test = lambda **kw: forward_test_2stream(ma, **kw)

    try:
        yield
    finally:
        ma.forward_test = original


def benchmark_config(model, data_loader, num_streams, n_warmup, n_runs):
    """对单个 stream 配置运行 benchmark"""
    latencies = []
    powers = []
    torch.cuda.reset_peak_memory_stats()

    with patch_forward_test(model, num_streams):
        data_iter = iter(data_loader)

        with torch.no_grad():
            # Warmup
            for i in range(n_warmup):
                try:
                    data = next(data_iter)
                except StopIteration:
                    data_iter = iter(data_loader)
                    data = next(data_iter)
                model(return_loss=False, rescale=True, **data)
                torch.cuda.synchronize()

            # Benchmark
            for i in range(n_runs):
                try:
                    data = next(data_iter)
                except StopIteration:
                    data_iter = iter(data_loader)
                    data = next(data_iter)

                power_before = get_gpu_power()

                torch.cuda.synchronize()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                start_event.record()
                model(return_loss=False, rescale=True, **data)
                end_event.record()
                torch.cuda.synchronize()

                latency_ms = start_event.elapsed_time(end_event)
                latencies.append(latency_ms)

                power_after = get_gpu_power()
                powers.append((power_before + power_after) / 2.0)

    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)

    return {
        'num_streams': num_streams,
        'latency_mean_ms': float(np.mean(latencies)),
        'latency_std_ms': float(np.std(latencies)),
        'latency_min_ms': float(np.min(latencies)),
        'latency_max_ms': float(np.max(latencies)),
        'peak_memory_mb': float(peak_mem),
        'avg_power_w': float(np.mean(powers)),
        'energy_per_frame_mj': float(np.mean(latencies) * np.mean(powers)),
        'n_runs': n_runs,
    }


def main():
    args = parse_args()
    print(f"[D1] Building model and dataset...")
    model, dataset, data_loader = build_model_and_data(args)
    print(f"[D1] Model loaded. Dataset: {len(dataset)} samples")

    if args.sweep:
        stream_configs = [1, 2]  # 3/4 需要侵入模型内部, 暂为 TODO
    else:
        stream_configs = [args.num_streams]

    all_results = {}
    for ns in stream_configs:
        print(f"\n{'='*60}")
        print(f"[D1] Benchmarking num_streams={ns}")
        print(f"{'='*60}")

        torch.cuda.reset_peak_memory_stats()
        result = benchmark_config(model, data_loader, ns, args.n_warmup, args.n_runs)
        all_results[f'stream_{ns}'] = result

        print(f"  Latency: {result['latency_mean_ms']:.1f} +/- {result['latency_std_ms']:.1f} ms")
        print(f"  Peak memory: {result['peak_memory_mb']:.0f} MB")
        print(f"  Avg power: {result['avg_power_w']:.1f} W")
        print(f"  Energy/frame: {result['energy_per_frame_mj']:.0f} mJ")

    output = {
        'config': args.config,
        'checkpoint': args.checkpoint,
        'prune_config': args.prune_config,
        'finetuned_ckpt': args.finetuned_ckpt,
        'results': all_results
    }

    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\n[D1] Results saved to {args.output}")
    else:
        print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
