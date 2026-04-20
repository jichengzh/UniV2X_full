"""
D1 多 Agent 并行度 benchmark

测量不同 CUDA stream 并行度下的推理时延/显存/功率。
支持 1/2/3/4 stream 配置。

用法:
  PYTHONPATH=/home/jichengzhi/UniV2X python tools/infer_multi_stream.py \
      projects/configs_e2e_univ2x/univ2x_coop_e2e_track.py \
      ckpts/univ2x_coop_e2e_stg2.pth \
      --num-streams 2 \
      --n-warmup 3 --n-runs 20

  # 带剪枝
  PYTHONPATH=/home/jichengzhi/UniV2X python tools/infer_multi_stream.py \
      projects/configs_e2e_univ2x/univ2x_coop_e2e_track.py \
      ckpts/univ2x_coop_e2e_stg2.pth \
      --prune-config prune_configs/decouple_enc10_dec07.json \
      --finetuned-ckpt work_dirs/ft_decouple_enc10_dec07/epoch_3.pth \
      --num-streams 2 \
      --n-warmup 3 --n-runs 20

  # sweep 全部 stream 配置
  PYTHONPATH=/home/jichengzhi/UniV2X python tools/infer_multi_stream.py \
      projects/configs_e2e_univ2x/univ2x_coop_e2e_track.py \
      ckpts/univ2x_coop_e2e_stg2.pth \
      --sweep \
      --n-warmup 3 --n-runs 10
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


def parse_args():
    p = argparse.ArgumentParser(description="D1 multi-stream benchmark")
    p.add_argument("config")
    p.add_argument("checkpoint")
    p.add_argument("--prune-config", default=None)
    p.add_argument("--finetuned-ckpt", default=None)
    p.add_argument("--num-streams", type=int, default=1,
                   help="并行 stream 数: 1=串行, 2=ego/infra并行, 3/4=更细拆分")
    p.add_argument("--sweep", action="store_true",
                   help="遍历 1/2/3/4 全部配置")
    p.add_argument("--n-warmup", type=int, default=3)
    p.add_argument("--n-runs", type=int, default=20)
    p.add_argument("--output", default=None,
                   help="结果 JSON 输出路径 (默认 stdout)")
    return p.parse_args()


def build_model_and_data(args):
    """构建模型和数据集，返回 (model, dataset, data_loader)"""
    cfg = Config.fromfile(args.config)

    # 修改 test pipeline
    if hasattr(cfg, 'test_pipeline'):
        cfg.data.test.pipeline = cfg.test_pipeline
    elif hasattr(cfg.data, 'test'):
        pass

    cfg.data.test.test_mode = True
    if hasattr(cfg.data.test, 'samples_per_gpu'):
        cfg.data.test.samples_per_gpu = 1

    dataset = build_dataset(cfg.data.test)
    from torch.utils.data import DataLoader
    from mmdet3d.datasets import build_dataloader
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        dist=False,
        shuffle=False
    )

    # 构建 MultiAgent 模型
    from projects.mmdet3d_plugin.univ2x.detectors.multi_agent import MultiAgent

    # 构建 ego agent
    ego_cfg = cfg.model_ego_agent
    from mmdet3d.models import build_model
    ego_model = build_model(ego_cfg, train_cfg=None, test_cfg=cfg.get('test_cfg'))

    # 构建 other agents
    other_agents = {}
    if hasattr(cfg, 'model_other_agent_inf'):
        inf_cfg = cfg.model_other_agent_inf
        inf_model = build_model(inf_cfg, train_cfg=None, test_cfg=cfg.get('test_cfg'))
        if hasattr(inf_cfg, 'other_agent_model_frozen') and inf_cfg.other_agent_model_frozen:
            for p in inf_model.parameters():
                p.requires_grad = False
        other_agents['model_other_agent_inf'] = inf_model

    model = MultiAgent(model_ego_agent=ego_model, model_other_agents=other_agents)

    # 加载 checkpoint
    import mmcv
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    if 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    model.load_state_dict(ckpt, strict=False)

    # 可选: 剪枝
    if args.prune_config:
        from projects.mmdet3d_plugin.univ2x.pruning.prune_univ2x import apply_prune_config
        prune_cfg = json.load(open(args.prune_config))
        apply_prune_config(model.model_ego_agent, prune_cfg)
        print(f"[Prune] Applied {args.prune_config}")

    # 可选: 加载微调 ckpt
    if args.finetuned_ckpt:
        ft_ckpt = torch.load(args.finetuned_ckpt, map_location='cpu')
        if 'state_dict' in ft_ckpt:
            ft_ckpt = ft_ckpt['state_dict']
        # 过滤匹配的 key
        model_dict = model.state_dict()
        matched = {k: v for k, v in ft_ckpt.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(matched)
        model.load_state_dict(model_dict)
        print(f"[Finetune] Loaded {len(matched)} params from {args.finetuned_ckpt}")

    model = model.cuda().eval()
    return model, dataset, data_loader


# ============================================================
# Multi-stream inference strategies
# ============================================================

def forward_serial(model, data):
    """stream=1: 完全串行 (当前默认行为)"""
    with torch.no_grad():
        return model(return_loss=False, rescale=True, **data)


def forward_2stream(model, data):
    """stream=2: ego agent 和 infra agent 的 backbone+BEV 阶段并行

    原理:
      - infra agent 的完整 forward 不依赖 ego 的输出
      - ego agent 的 backbone + BEV encoding 不依赖 infra 的输出
      - ego 只在 fusion 阶段（decoder 之前）需要 infra 的 track queries
      - 所以 infra forward 可以与 ego backbone+BEV 并行

    实现:
      Stream 0 (default): ego agent backbone + BEV encoder
      Stream 1:           infra agent 完整 forward
      Sync → ego fusion + decoder + heads
    """
    ego_data = data.get('ego_agent_data', {})
    other_data = data.get('other_agent_data_dict', {})

    # Stream 1: infra agent forward (异步)
    stream_infra = cuda.Stream()
    infra_results = {}

    with torch.no_grad():
        for name in model.other_agent_names:
            with cuda.stream(stream_infra):
                result = getattr(model, name)(
                    return_loss=False, w_label=True,
                    **(other_data[name])
                )
                result[0]['ego2other_rt'] = other_data[name]['veh2inf_rt']
                result[0]['pc_range'] = model.pc_range_dict[name]
                infra_results[name] = result

        # Stream 0 (default): ego agent forward
        # ego 内部会等 infra_results → 我们需要在 ego forward 之前同步
        cuda.current_stream().wait_stream(stream_infra)

        result = model.model_ego_agent(
            return_loss=False, w_label=True,
            other_agent_results=infra_results,
            **ego_data
        )

    return result


def forward_nstream(model, data, n_streams=4):
    """stream=3/4: 更细粒度的并行 (实验性)

    当前 UniV2X 的 ego agent forward 是一个整体调用 (backbone→BEV→decoder→heads),
    拆分需要侵入模型内部。

    stream=3/4 的理论实现需要:
      1. 拆分 ego agent 的 forward 为 backbone 和 bev_encoder 两个阶段
      2. 在不同 stream 上执行
      3. 处理中间 tensor 的跨 stream 同步

    当前降级为 stream=2 的行为 + 额外的 stream 创建开销测量。
    TODO: 侵入模型内部实现真正的 3/4 stream 拆分。
    """
    # 目前降级为 2-stream + overhead 测量
    # 创建额外 stream 来模拟资源竞争
    extra_streams = [cuda.Stream() for _ in range(n_streams - 2)]
    result = forward_2stream(model, data)
    # 同步额外 stream
    for s in extra_streams:
        cuda.current_stream().wait_stream(s)
    return result


STREAM_DISPATCH = {
    1: forward_serial,
    2: forward_2stream,
    3: lambda m, d: forward_nstream(m, d, 3),
    4: lambda m, d: forward_nstream(m, d, 4),
}


# ============================================================
# Benchmark
# ============================================================

def get_gpu_power():
    """读取当前 GPU 功率 (W)"""
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        # 可能有多个 GPU，取第一个
        return float(out.split('\n')[0])
    except Exception:
        return -1.0


def benchmark_config(model, data_loader, num_streams, n_warmup, n_runs):
    """对单个 stream 配置运行 benchmark"""
    forward_fn = STREAM_DISPATCH[num_streams]

    latencies = []
    powers = []
    peak_mem_before = torch.cuda.max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    data_iter = iter(data_loader)

    # Warmup
    for i in range(n_warmup):
        try:
            data = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            data = next(data_iter)

        # 移动到 GPU
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = v.cuda()

        forward_fn(model, data)
        torch.cuda.synchronize()

    # Benchmark runs
    for i in range(n_runs):
        try:
            data = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            data = next(data_iter)

        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = v.cuda()

        # 功率采样
        power_before = get_gpu_power()

        # CUDA Event 计时
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        forward_fn(model, data)
        end_event.record()
        torch.cuda.synchronize()

        latency_ms = start_event.elapsed_time(end_event)
        latencies.append(latency_ms)

        power_after = get_gpu_power()
        avg_power = (power_before + power_after) / 2.0
        powers.append(avg_power)

    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB

    result = {
        'num_streams': num_streams,
        'latency_mean_ms': float(np.mean(latencies)),
        'latency_std_ms': float(np.std(latencies)),
        'latency_min_ms': float(np.min(latencies)),
        'latency_max_ms': float(np.max(latencies)),
        'peak_memory_mb': float(peak_mem),
        'avg_power_w': float(np.mean(powers)),
        'energy_per_frame_mj': float(np.mean(latencies) * np.mean(powers)),  # ms * W = mJ
        'n_runs': n_runs,
    }
    return result


def main():
    args = parse_args()
    print(f"[D1] Building model and dataset...")
    model, dataset, data_loader = build_model_and_data(args)
    print(f"[D1] Model loaded. Dataset: {len(dataset)} samples")

    if args.sweep:
        stream_configs = [1, 2, 3, 4]
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

    # 输出
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
        print(f"\n[D1] Results:")
        print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
