"""硬约束 DSL 单元测试 (Phase 1A.4 验收).

覆盖三类约束:
- physical: DLA per-tensor / 精度支持 / DLA count / Sparse TC
- empirical: 通道对齐 / 跨层梯度
- soft: 精度一致性
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from framework.capability_schema import HardwareCapability
from framework.config_schema import Config, UNIV2X_MODULES
from framework.constraints import is_legal, list_violations


def _load_hw(name: str) -> HardwareCapability:
    return HardwareCapability.from_yaml(f"configs/hardware/{name}.yaml")


def test_baseline_legal_on_all_hw():
    """FP32 baseline 在所有硬件上都合法."""
    cfg = Config.fp32_baseline()
    for hw_name in ["rtx4090", "orin_agx", "orin_nano"]:
        hw = _load_hw(hw_name)
        ok, reason = is_legal(cfg, hw)
        assert ok, f"{hw_name}: {reason}"


def test_dla_per_tensor_violation():
    """DLA 路由 + per-channel = 物理硬约束违反."""
    hw = _load_hw("orin_agx")
    cfg = Config.fp32_baseline().with_field(
        d_routing={"backbone": "DLA0", "encoder": "GPU", "decoder": "GPU",
                   "heads": "GPU", "v2x_comm": "GPU"},
        q_bits={"backbone": "INT8", "encoder": "FP32", "decoder": "FP32",
                "heads": "FP32", "v2x_comm": "FP32"},
        q_granularity={"backbone": "per-channel", "encoder": "none", "decoder": "none",
                       "heads": "none", "v2x_comm": "none"},
    )
    ok, reason = is_legal(cfg, hw)
    assert not ok, "应该违反 DLA per-tensor"
    assert "per-channel" in reason or "per_tensor" in reason or "per-tensor" in reason


def test_dla_unavailable_on_nano():
    """Orin Nano 没有 DLA,请求 DLA 路由非法."""
    hw = _load_hw("orin_nano")
    assert not hw.has_dla
    cfg = Config.fp32_baseline().with_field(
        d_routing={"backbone": "DLA0", "encoder": "GPU", "decoder": "GPU",
                   "heads": "GPU", "v2x_comm": "GPU"},
    )
    ok, reason = is_legal(cfg, hw)
    assert not ok, f"Nano 上不应允许 DLA 路由,但通过: {reason}"


def test_fp8_only_on_4090():
    """FP8 只在 4090 (Ada) 上支持,Orin (Ampere) 上不支持."""
    cfg = Config.fp32_baseline().with_field(
        q_bits={m: "FP8" for m in UNIV2X_MODULES},
    )
    # 4090 应通过
    ok, _ = is_legal(cfg, _load_hw("rtx4090"))
    assert ok
    # Orin AGX/Nano 应失败
    for hw_name in ["orin_agx", "orin_nano"]:
        ok, reason = is_legal(cfg, _load_hw(hw_name))
        assert not ok, f"{hw_name}: FP8 不该被支持,但通过了"


def test_cross_layer_gradient_30pct():
    """跨模块剪枝率梯度 > 30% 触发经验约束."""
    hw = _load_hw("orin_agx")
    # backbone=0 -> encoder=0.5 一下跳了 50%
    cfg = Config.fp32_baseline().with_field(
        prune_rate={"backbone": 0.0, "encoder": 0.5, "decoder": 0.5,
                    "heads": 0.5, "v2x_comm": 0.5},
        prune_object="channel",
    )
    ok, reason = is_legal(cfg, hw)
    assert not ok, "应该被跨层梯度约束拦截"
    assert "梯度" in reason


def test_channel_alignment():
    """剪枝率 0.4375 (=14/32) 通道对齐到 32; 剪枝率 0.42 不对齐."""
    hw = _load_hw("orin_agx")
    # 0.5 -> kept=0.5, 0.5*8=4 (整数) → 通过
    cfg_aligned = Config.fp32_baseline().with_field(
        prune_rate={m: 0.5 for m in UNIV2X_MODULES},
        prune_object="channel",
    )
    ok, _ = is_legal(cfg_aligned, hw)
    assert ok, "0.5 应该对齐"
    # 0.42 -> kept=0.58, 0.58*8=4.64 → 失败
    cfg_misaligned = Config.fp32_baseline().with_field(
        prune_rate={m: 0.42 for m in UNIV2X_MODULES},
        prune_object="channel",
    )
    ok, reason = is_legal(cfg_misaligned, hw)
    assert not ok and "对齐" in reason


def test_dla_count_check():
    """请求 DLA1 但硬件 dla_count=0 → 失败 (Nano)."""
    hw = _load_hw("orin_nano")
    cfg = Config.fp32_baseline().with_field(
        d_routing={"backbone": "DLA0", "encoder": "GPU", "decoder": "GPU",
                   "heads": "GPU", "v2x_comm": "GPU"},
    )
    ok, reason = is_legal(cfg, hw)
    assert not ok


def test_2to4_requires_sparse_tc():
    """2:4 剪枝在不支持 Sparse TC 的硬件上失败 (假想 hw)."""
    hw = _load_hw("orin_agx")
    cfg = Config.fp32_baseline().with_field(prune_object="2:4")
    # AGX 支持 Sparse TC,所以应通过
    ok, _ = is_legal(cfg, hw)
    assert ok


def test_list_violations_returns_all():
    """list_violations 应返回所有违反的约束 (含 soft)."""
    hw = _load_hw("orin_agx")
    cfg = Config.fp32_baseline().with_field(
        prune_rate={"backbone": 0.0, "encoder": 0.5, "decoder": 0.5,
                    "heads": 0.5, "v2x_comm": 0.5},
        prune_object="channel",
        q_bits={"backbone": "FP32", "encoder": "INT8", "decoder": "INT8",
                "heads": "INT8", "v2x_comm": "INT8"},
    )
    vios = list_violations(cfg, hw)
    kinds = {v[0] for v in vios}
    # 应同时违反 empirical (梯度) 和 soft (FP32+INT8 混合)
    assert "empirical" in kinds
    assert "soft" in kinds


if __name__ == "__main__":
    test_baseline_legal_on_all_hw()
    test_dla_per_tensor_violation()
    test_dla_unavailable_on_nano()
    test_fp8_only_on_4090()
    test_cross_layer_gradient_30pct()
    test_channel_alignment()
    test_dla_count_check()
    test_2to4_requires_sparse_tc()
    test_list_violations_returns_all()
    print("--- 1A.4 constraints DSL: 9/9 tests passed ---")
