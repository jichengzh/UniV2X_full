"""D ↔ B2 双向硬约束传播 (Phase 1A.5)

对应 v1.5 §0.2 重要区分 + Phase1_2 §2.3 关键设计.

D → B2 (DLA 路由触发):
  IF 模块路由到 DLA → B2 强制 (INT8, per-tensor, W+A)
  → B2 4 轴可行集合从 ~36 种缩为 1 种

B2 → D (精度需求触发):
  IF B2.粒度 = per-channel → 该模块禁止 DLA
  IF B2.位宽 = FP32 → 该模块禁止 DLA (DLA 不支持 FP32)

设计要点:
- 实现为不可变变换 (返回新 Config),不修改原 Config
- 迭代到不动点 (因为 D→B2 后可能反向触发更多约束)
- 限制最大迭代次数防止死循环
"""

from __future__ import annotations

from copy import copy
from dataclasses import replace

from framework.capability_schema import HardwareCapability
from framework.config_schema import Config


def propagate_d_to_b2(cfg: Config, hw: HardwareCapability) -> Config:
    """DLA 路由 → B2 强制约束传播.

    任何路由到 DLA 的模块,B2 配置 (位宽/粒度/对象) 强制改为 DLA 兼容值.
    """
    if not hw.has_dla:
        return cfg
    new_q_bits = dict(cfg.q_bits)
    new_q_gran = dict(cfg.q_granularity)
    new_q_obj = dict(cfg.q_object)
    changed = False
    for m, route in cfg.d_routing.items():
        if not route.startswith("DLA"):
            continue
        if new_q_bits.get(m) not in ("INT8",):  # 优先 INT8 (DLA 主目标)
            new_q_bits[m] = "INT8"
            changed = True
        if new_q_gran.get(m) != "per-tensor":
            new_q_gran[m] = "per-tensor"
            changed = True
        if new_q_obj.get(m) != "W+A":
            new_q_obj[m] = "W+A"
            changed = True
    if not changed:
        return cfg
    return replace(cfg, q_bits=new_q_bits, q_granularity=new_q_gran, q_object=new_q_obj)


def propagate_b2_to_d(cfg: Config, hw: HardwareCapability) -> Config:
    """B2 选择 → D 路由可行集合收窄.

    若某模块用了 DLA 不兼容的 B2 配置 (per-channel / FP32),把它从 DLA 路由踢回 GPU.
    """
    if not hw.has_dla:
        return cfg
    new_routing = dict(cfg.d_routing)
    changed = False
    for m, route in list(new_routing.items()):
        if not route.startswith("DLA"):
            continue
        # per-channel 量化 → DLA 不支持
        if cfg.q_granularity.get(m) == "per-channel":
            new_routing[m] = "GPU"
            changed = True
            continue
        # FP32 精度 → DLA 不支持 (DLA 仅 FP16/INT8)
        if cfg.q_bits.get(m) == "FP32":
            new_routing[m] = "GPU"
            changed = True
    if not changed:
        return cfg
    return replace(cfg, d_routing=new_routing)


def propagate_all(cfg: Config, hw: HardwareCapability, max_iter: int = 10) -> Config:
    """完整双向传播,迭代到不动点.

    防死循环: 设上限 max_iter (默认 10),超限抛 RuntimeError.
    实测 DLA + per-channel 这种典型冲突 1-2 次就稳定.
    """
    prev = None
    curr = cfg
    for i in range(max_iter):
        if curr == prev:
            return curr
        prev = curr
        curr = propagate_d_to_b2(curr, hw)
        curr = propagate_b2_to_d(curr, hw)
    raise RuntimeError(
        f"双向传播 {max_iter} 次未收敛,可能存在循环冲突. 最后状态: {curr}"
    )


__all__ = ["propagate_d_to_b2", "propagate_b2_to_d", "propagate_all"]
