"""最小搜索器 v0 (Phase 1A.6)

对应 v1.5 §6.1 Phase 1A.5 + Phase1_2 §2.3 关键设计.

只做随机采样 + 双向传播 + 硬约束过滤 (无评估器、无 Pareto).
存在意义: 验证 capability/约束/传播链路工作正常.
真正的多目标搜索器在 Stage 4 (NSGA-II / BoTorch) 实现.
"""

from __future__ import annotations

import random
from typing import Optional

from framework.capability_schema import HardwareCapability
from framework.config_schema import (
    Config,
    UNIV2X_MODULES,
    PRUNE_OBJECT_VALUES,
    PRUNE_CRITERION_VALUES,
    Q_BITS_VALUES,
    Q_GRAN_VALUES,
    Q_OBJ_VALUES,
)
from framework.constraints import is_legal
from framework.propagation import propagate_all

# Phase 2 §0.5 — 准则按子网分配的候选池
# CNN-like 偏 L1/FPGM, Transformer-like 偏 Taylor/Wanda
DEFAULT_CRITERION_POOL = {
    "backbone": ("L1", "FPGM"),
    "encoder": ("L1", "Taylor"),
    "decoder": ("Taylor", "Wanda"),
    "heads": ("L1",),
    "v2x_comm": ("L1",),
}

# 通道对齐网格 (256 通道下 = k/8) — 与 constraints._check_channel_alignment 对齐
ALIGNED_PRUNE_RATES = (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75)


def sample_random_config(
    hw: HardwareCapability,
    modules: tuple[str, ...] = UNIV2X_MODULES,
    rng: Optional[random.Random] = None,
    lock_d_to_gpu_only: bool = True,
) -> Config:
    """从合法搜索空间均匀采样一个候选 (不保证通过约束,留给 propagate+is_legal).

    lock_d_to_gpu_only=True 时,Phase 2 仅 B1+B2 搜索 (D 锁 GPU only).
    Phase 3 跨硬件迁移时设为 False,允许 DLA 路由.
    """
    rng = rng or random.Random()

    # 全局剪枝粒度 (channel/2:4/none 三选一,heads 类型不混搜)
    obj_choices = ["channel", "none"]
    if hw.features.sparse_tc:
        obj_choices.append("2:4")
    prune_object = rng.choice(obj_choices)

    prune_rate = {}
    prune_criterion = {}
    for m in modules:
        if prune_object == "none":
            prune_rate[m] = 0.0
            prune_criterion[m] = "none"
        elif prune_object == "2:4":
            prune_rate[m] = 0.5  # 2:4 强制 50%
            prune_criterion[m] = "L1"
        else:  # channel
            prune_rate[m] = rng.choice(ALIGNED_PRUNE_RATES)
            prune_criterion[m] = rng.choice(DEFAULT_CRITERION_POOL.get(m, ("L1",)))

    # B2: GPU 支持 INT8/FP16/(FP32),粒度二选一
    bits_pool = [b for b in ("INT8", "FP16", "FP32")
                 if b in hw.ips["gpu"].precisions]
    q_bits = {m: rng.choice(bits_pool) for m in modules}
    q_granularity = {}
    q_object = {}
    for m in modules:
        if q_bits[m] == "FP32":
            q_granularity[m] = "none"
            q_object[m] = "none"
        else:
            q_granularity[m] = rng.choice(("per-tensor", "per-channel"))
            q_object[m] = rng.choice(("W-only", "W+A"))

    # D: GPU only 时全部 GPU; 否则按硬件 IP 数 + DLA core 数随机
    if lock_d_to_gpu_only or not hw.has_dla:
        d_routing = {m: "GPU" for m in modules}
    else:
        d_choices = ["GPU"] + [f"DLA{i}" for i in range(hw.features.dla_count)]
        d_routing = {m: rng.choice(d_choices) for m in modules}

    return Config(
        prune_rate=prune_rate,
        prune_object=prune_object,
        prune_criterion=prune_criterion,
        q_bits=q_bits,
        q_granularity=q_granularity,
        q_object=q_object,
        d_routing=d_routing,
        source="phase1a_random",
    )


def random_search(
    hw: HardwareCapability,
    n_candidates: int = 50,
    max_attempts: int = 5000,
    seed: Optional[int] = None,
    lock_d_to_gpu_only: bool = True,
    verbose: bool = False,
) -> tuple[list[Config], dict]:
    """随机搜索: 采样 + 传播 + 过滤,直到收集到 n_candidates 个合法配置.

    返回:
        (candidates, stats) — stats 包含尝试次数、过滤通过率等诊断信息
    """
    rng = random.Random(seed)
    candidates: list[Config] = []
    seen_keys: set[tuple] = set()
    attempts = 0
    legal = 0
    duplicate = 0
    illegal_reasons: dict[str, int] = {}

    while len(candidates) < n_candidates and attempts < max_attempts:
        attempts += 1
        cfg = sample_random_config(hw, rng=rng, lock_d_to_gpu_only=lock_d_to_gpu_only)
        cfg = propagate_all(cfg, hw)

        ok, reason = is_legal(cfg, hw)
        if not ok:
            # 提取约束名做统计
            tag = reason.split("]")[0].lstrip("[") if "]" in reason else reason[:30]
            illegal_reasons[tag] = illegal_reasons.get(tag, 0) + 1
            continue

        # 去重 (按关键字段哈希)
        key = (
            cfg.prune_object,
            tuple(sorted(cfg.prune_rate.items())),
            tuple(sorted(cfg.q_bits.items())),
            tuple(sorted(cfg.q_granularity.items())),
            tuple(sorted(cfg.d_routing.items())),
        )
        if key in seen_keys:
            duplicate += 1
            continue
        seen_keys.add(key)
        # 给一个稳定的 config_id
        cfg = cfg.with_field(config_id=f"p1a_{len(candidates):04d}")
        candidates.append(cfg)
        legal += 1
        if verbose and len(candidates) % 10 == 0:
            print(f"  collected {len(candidates)}/{n_candidates} after {attempts} attempts")

    stats = {
        "n_candidates": len(candidates),
        "n_attempts": attempts,
        "n_legal": legal,
        "n_duplicate": duplicate,
        "n_illegal": attempts - legal - duplicate,
        "pass_rate": legal / max(attempts, 1),
        "illegal_breakdown": illegal_reasons,
    }
    return candidates, stats


__all__ = ["sample_random_config", "random_search", "DEFAULT_CRITERION_POOL", "ALIGNED_PRUNE_RATES"]
