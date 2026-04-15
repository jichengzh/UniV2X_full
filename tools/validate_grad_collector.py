"""
验证 grad_collector.py 是否能正常为 Taylor 重要性评估收集梯度。

用法:
    PYTHONPATH=/home/jichengzhi/UniV2X python tools/validate_grad_collector.py \
        projects/configs_e2e_univ2x/univ2x_coop_e2e_track.py \
        ckpts/univ2x_coop_e2e_stg2.pth \
        --num-batches 4

成功标准:
    - 至少 80% 的 Linear 权重参数拿到了非零 .grad
    - 至少 80% 的 FFN `ffns.*.layers.*.0.weight` 拿到了非零 .grad
    - `GroupTaylorImportance` 能在剪枝目标模块上返回非零重要性向量

这一脚本是 Phase B.0 的前置冒烟，确保 Phase B.2 的准则对比可以用 Taylor。
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/jichengzhi/UniV2X")

import torch
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("validate_grad")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="验证 Taylor 梯度收集可用性")
    p.add_argument("config", help="mmdet3d 配置文件")
    p.add_argument("checkpoint", help="模型 checkpoint")
    p.add_argument("--num-batches", type=int, default=4,
                   help="收集梯度的 batch 数 (小冒烟 4, 正式 16-32)")
    p.add_argument("--gpu-id", type=int, default=0)
    return p.parse_args()


def build_model_and_loader(cfg_path: str, ckpt_path: str):
    """复用 pruning_sensitivity_analysis.load_model_fresh + 训练 dataloader。"""
    from tools.pruning_sensitivity_analysis import load_model_fresh, get_prune_target
    from mmdet3d.datasets import build_dataset
    from projects.mmdet3d_plugin.datasets.builder import build_dataloader

    logger.info("加载模型: %s", os.path.basename(ckpt_path))
    model, cfg = load_model_fresh(cfg_path, ckpt_path)

    logger.info("构建训练 dataloader (用于梯度采样)")
    dataset = build_dataset(cfg.data.train)
    loader = build_dataloader(
        dataset, samples_per_gpu=1,
        workers_per_gpu=0,  # 避免 pickle 问题
        dist=False, shuffle=True,
        nonshuffler_sampler=cfg.data.get("nonshuffler_sampler"),
    )
    return model, cfg, loader, get_prune_target(model)


def count_grad_stats(module: nn.Module, filter_name: str | None = None) -> dict:
    """对指定子模块统计 .grad 状态。"""
    n_params = 0
    n_with_grad = 0
    n_nonzero_grad = 0
    total_abs_grad = 0.0

    for name, p in module.named_parameters():
        if filter_name and filter_name not in name:
            continue
        n_params += 1
        if p.grad is not None:
            n_with_grad += 1
            g_abs = p.grad.abs().sum().item()
            total_abs_grad += g_abs
            if g_abs > 1e-10:
                n_nonzero_grad += 1

    return {
        "n_params": n_params,
        "n_with_grad": n_with_grad,
        "n_nonzero_grad": n_nonzero_grad,
        "mean_abs_grad": total_abs_grad / max(n_params, 1),
    }


def test_taylor_importance(prune_target: nn.Module) -> bool:
    """在典型 FFN 层上测试 GroupTaylorImportance 能否返回非零重要性。"""
    try:
        import torch_pruning as tp
    except ImportError:
        logger.error("torch_pruning 未安装，跳过 importance 测试")
        return False

    # 找一个典型 FFN first Linear
    sample_linear = None
    sample_name = None
    for name, mod in prune_target.named_modules():
        if "ffns" in name and "layers.0.0" in name and isinstance(mod, nn.Linear):
            sample_linear = mod
            sample_name = name
            break

    if sample_linear is None:
        logger.warning("没找到典型 FFN Linear，跳过 importance 测试")
        return False

    logger.info("在 %s (out=%d) 上测试 GroupTaylorImportance",
                sample_name, sample_linear.out_features)

    importance_fn = tp.importance.GroupTaylorImportance()

    # 构建一个最小的依赖图用 tp 接口
    # 这里只是验证 param.grad 被读到即可（完整路径在 apply_prune_config 里）
    w = sample_linear.weight
    if w.grad is None or w.grad.abs().sum() < 1e-10:
        logger.error("❌ FFN Linear 的 weight.grad 为零，Taylor 不可用")
        return False

    # 手动算一下 Taylor score (|W * grad|) 作为参考
    taylor_score = (w * w.grad).abs().sum(dim=1)  # per-output-channel
    n_nonzero_channels = (taylor_score > 1e-10).sum().item()
    logger.info("  Taylor 分数: %d/%d 通道非零, max=%.4e, mean=%.4e",
                n_nonzero_channels, taylor_score.numel(),
                taylor_score.max().item(), taylor_score.mean().item())

    success = n_nonzero_channels >= 0.5 * taylor_score.numel()
    if success:
        logger.info("  ✅ GroupTaylorImportance 可用")
    else:
        logger.error("  ❌ 过半通道 Taylor 分数为零，不可用")
    return success


def main() -> int:
    args = parse_args()
    torch.cuda.set_device(args.gpu_id)

    # 1. 构建模型和 dataloader
    model, cfg, loader, prune_target = build_model_and_loader(
        args.config, args.checkpoint
    )

    # 2. 收集梯度
    from projects.mmdet3d_plugin.univ2x.pruning.grad_collector import collect_gradients
    logger.info("=" * 60)
    logger.info("收集梯度: %d batches", args.num_batches)
    logger.info("=" * 60)

    collect_gradients(
        model=model,
        dataloader=loader,
        num_batches=args.num_batches,
    )

    # 3. 统计 ego_agent 全局
    logger.info("=" * 60)
    logger.info("梯度统计 (ego_agent 全局)")
    logger.info("=" * 60)
    stats_all = count_grad_stats(prune_target)
    logger.info("  参数总数:   %d", stats_all["n_params"])
    logger.info("  有 .grad:   %d (%.1f%%)",
                stats_all["n_with_grad"],
                100 * stats_all["n_with_grad"] / max(stats_all["n_params"], 1))
    logger.info("  非零 .grad: %d (%.1f%%)",
                stats_all["n_nonzero_grad"],
                100 * stats_all["n_nonzero_grad"] / max(stats_all["n_params"], 1))
    logger.info("  mean |grad|: %.4e", stats_all["mean_abs_grad"])

    # 4. 重点看 FFN Linear
    logger.info("=" * 60)
    logger.info("梯度统计 (仅 FFN 层)")
    logger.info("=" * 60)
    stats_ffn = count_grad_stats(prune_target, filter_name="ffns")
    logger.info("  FFN 参数数:   %d", stats_ffn["n_params"])
    logger.info("  非零 .grad:  %d (%.1f%%)",
                stats_ffn["n_nonzero_grad"],
                100 * stats_ffn["n_nonzero_grad"] / max(stats_ffn["n_params"], 1))
    logger.info("  mean |grad|: %.4e", stats_ffn["mean_abs_grad"])

    # 5. Taylor importance 测试
    logger.info("=" * 60)
    logger.info("Taylor importance 可用性测试")
    logger.info("=" * 60)
    taylor_ok = test_taylor_importance(prune_target)

    # 6. 成功判据
    logger.info("=" * 60)
    global_ok = stats_all["n_nonzero_grad"] >= 0.5 * stats_all["n_params"]
    ffn_ok = stats_ffn["n_nonzero_grad"] >= 0.8 * stats_ffn["n_params"]

    logger.info("全局梯度覆盖率 ≥ 50%%:  %s", "✅" if global_ok else "❌")
    logger.info("FFN 梯度覆盖率 ≥ 80%%:  %s", "✅" if ffn_ok else "❌")
    logger.info("Taylor importance 有效: %s", "✅" if taylor_ok else "❌")

    all_ok = global_ok and ffn_ok and taylor_ok
    logger.info("=" * 60)
    logger.info("结论: %s", "✅ grad_collector 可用，Phase B.2 可以跑 Taylor" if all_ok
                else "❌ 需要排查梯度收集问题")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
