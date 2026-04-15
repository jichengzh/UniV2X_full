"""
模型各模块 latency benchmark (对标李星峰 2026-04-11 周报的耗时表格)

用法:
  # baseline (未剪枝)
  PYTHONPATH=/home/jichengzhi/UniV2X python tools/benchmark_latency.py \
      projects/configs_e2e_univ2x/univ2x_coop_e2e_track.py \
      ckpts/univ2x_coop_e2e_stg2.pth \
      --n-warmup 3 --n-runs 20 \
      --output output/latency_baseline.json

  # 剪枝后
  PYTHONPATH=/home/jichengzhi/UniV2X python tools/benchmark_latency.py \
      projects/configs_e2e_univ2x/univ2x_coop_e2e_track.py \
      ckpts/univ2x_coop_e2e_stg2.pth \
      --prune-config prune_configs/p1_ffn_30pct.json \
      --n-warmup 3 --n-runs 20 \
      --output output/latency_pruned_p1_30.json
"""
import argparse
import json
import os
import sys
import time
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")
sys.path.insert(0, '/home/jichengzhi/UniV2X')

import torch
from mmcv.parallel import MMDataParallel
from mmcv import Config
from mmdet.datasets import replace_ImageToTensor
from mmdet3d.datasets import build_dataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("checkpoint")
    p.add_argument("--prune-config", default=None, help="可选: 剪枝配置")
    p.add_argument("--n-warmup", type=int, default=3, help="预热迭代数 (不计入统计)")
    p.add_argument("--n-runs", type=int, default=20, help="实际统计的迭代数")
    p.add_argument("--output", default="output/latency_results.json", help="结果保存路径")
    return p.parse_args()


# ==========================================
# 模块级 hook 记录
# ==========================================

class ModuleTimer:
    """为指定模块注册 forward_pre_hook + forward_hook, 用 CUDA Event 精确记录 latency"""

    def __init__(self):
        self.times = defaultdict(list)  # name -> list of ms
        self.events = {}  # name -> (start_event, end_event) for current iteration

    def make_pre_hook(self, name):
        def pre_hook(mod, inp):
            start = torch.cuda.Event(enable_timing=True)
            start.record()
            self.events[name] = start
        return pre_hook

    def make_post_hook(self, name):
        def post_hook(mod, inp, out):
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            torch.cuda.synchronize()
            start = self.events.pop(name, None)
            if start is not None:
                t_ms = start.elapsed_time(end)
                self.times[name].append(t_ms)
        return post_hook

    def attach(self, module, name):
        h1 = module.register_forward_pre_hook(self.make_pre_hook(name))
        h2 = module.register_forward_hook(self.make_post_hook(name))
        return [h1, h2]

    def summary(self):
        """返回统计: mean/std/p50/p90 per module (ms)"""
        import statistics
        result = {}
        for name, samples in self.times.items():
            if not samples:
                continue
            s = sorted(samples)
            result[name] = {
                "mean": round(statistics.mean(samples), 3),
                "std": round(statistics.stdev(samples) if len(samples) > 1 else 0, 3),
                "p50": round(s[len(s) // 2], 3),
                "p90": round(s[int(len(s) * 0.9)], 3) if len(s) > 10 else round(s[-1], 3),
                "n": len(samples),
            }
        return result


def get_target_modules(model_ego):
    """定义要测的模块 (按李星峰周报格式)"""
    targets = {}

    # backbone + neck
    if hasattr(model_ego, "img_backbone"):
        targets["backbone"] = model_ego.img_backbone
    if hasattr(model_ego, "img_neck"):
        targets["neck"] = model_ego.img_neck

    # track_head 整体 (pts_bbox_head)
    if hasattr(model_ego, "pts_bbox_head"):
        head = model_ego.pts_bbox_head
        targets["track_head"] = head

        # 再细分: BEV encoder + decoder
        if hasattr(head, "transformer"):
            tf = head.transformer
            if hasattr(tf, "encoder"):
                targets["bev_encoder"] = tf.encoder
            if hasattr(tf, "decoder"):
                targets["track_head_decoder"] = tf.decoder

    # seg_head 整体
    if hasattr(model_ego, "seg_head"):
        targets["seg_head"] = model_ego.seg_head
        seg = model_ego.seg_head
        if hasattr(seg, "transformer"):
            targets["seg_transformer"] = seg.transformer
        if hasattr(seg, "stuff_mask_head"):
            targets["seg_stuff_mask_head"] = seg.stuff_mask_head
        if hasattr(seg, "things_mask_head"):
            targets["seg_things_mask_head"] = seg.things_mask_head

    return targets


def main():
    args = parse_args()

    # 1. 加载模型
    print(f"Loading config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    from tools.pruning_sensitivity_analysis import load_model_fresh, get_prune_target
    model, cfg = load_model_fresh(args.config, args.checkpoint)

    # 2. 可选: 剪枝
    pruned = False
    params_before = sum(p.numel() for p in get_prune_target(model).parameters())
    if args.prune_config:
        print(f"\nApplying pruning: {args.prune_config}")
        with open(args.prune_config) as f:
            prune_cfg = json.load(f)
        from projects.mmdet3d_plugin.univ2x.pruning.prune_univ2x import apply_prune_config
        apply_prune_config(get_prune_target(model), prune_cfg, dataloader=None)
        pruned = True
    params_after = sum(p.numel() for p in get_prune_target(model).parameters())
    print(f"\nModel params: {params_before:,d} -> {params_after:,d} (reduction {(1-params_after/params_before)*100:.2f}%)")

    # 3. 构建测试数据 (只需要 1 个 sample)
    cfg.data.test.test_mode = True
    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    dataset = build_dataset(cfg.data.test)
    print(f"Dataset size: {len(dataset)}")

    from projects.mmdet3d_plugin.datasets.builder import build_dataloader
    dl = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=0, dist=False, shuffle=False)

    # 4. 设置 hooks
    ego = get_prune_target(model)
    targets = get_target_modules(ego)
    print(f"\nRegistering hooks on {len(targets)} modules:")
    for n in targets:
        p = sum(p.numel() for p in targets[n].parameters())
        print(f"  {n:<28} params={p/1e6:>6.2f}M")

    timer = ModuleTimer()
    handles = []
    for name, mod in targets.items():
        handles.extend(timer.attach(mod, name))

    # 5. 包装并 evaluate (含 warmup)
    model = MMDataParallel(model.cuda(), device_ids=[0])
    model.eval()

    e2e_times = []

    print(f"\n预热 {args.n_warmup} 次...")
    with torch.no_grad():
        it = iter(dl)
        # warmup
        for _ in range(args.n_warmup):
            try:
                data = next(it)
            except StopIteration:
                it = iter(dl)
                data = next(it)
            _ = model(return_loss=False, rescale=True, **data)
            torch.cuda.synchronize()

    # 清空 warmup 时的 timing
    timer.times.clear()

    print(f"\n正式统计 {args.n_runs} 次...")
    with torch.no_grad():
        for i in range(args.n_runs):
            try:
                data = next(it)
            except StopIteration:
                it = iter(dl)
                data = next(it)

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(return_loss=False, rescale=True, **data)
            torch.cuda.synchronize()
            t_ms = (time.perf_counter() - t0) * 1000
            e2e_times.append(t_ms)

            if (i + 1) % 5 == 0:
                print(f"  [{i+1}/{args.n_runs}] last e2e: {t_ms:.1f}ms")

    # 6. 统计
    import statistics
    e2e_mean = statistics.mean(e2e_times)
    e2e_std = statistics.stdev(e2e_times) if len(e2e_times) > 1 else 0

    print(f"\n{'='*65}")
    print(f"{'Module':<28} {'mean(ms)':>10} {'std':>8} {'p50':>8} {'p90':>8} {'n':>5}")
    print(f"{'-'*65}")
    module_stats = timer.summary()
    for name in ["backbone", "neck", "bev_encoder", "track_head_decoder", "track_head",
                 "seg_transformer", "seg_stuff_mask_head", "seg_things_mask_head", "seg_head"]:
        if name in module_stats:
            s = module_stats[name]
            print(f"{name:<28} {s['mean']:>10.3f} {s['std']:>8.3f} {s['p50']:>8.3f} {s['p90']:>8.3f} {s['n']:>5}")

    print(f"{'-'*65}")
    print(f"{'e2e (all)':<28} {e2e_mean:>10.3f} {e2e_std:>8.3f} {'-':>8} {'-':>8} {len(e2e_times):>5}")

    # 7. 保存
    out = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "prune_config": args.prune_config,
        "pruned": pruned,
        "params_before": params_before,
        "params_after": params_after,
        "reduction": round(1.0 - params_after / params_before, 4),
        "n_warmup": args.n_warmup,
        "n_runs": args.n_runs,
        "e2e_ms_mean": round(e2e_mean, 3),
        "e2e_ms_std": round(e2e_std, 3),
        "modules": module_stats,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n结果已保存: {args.output}")


if __name__ == "__main__":
    main()
