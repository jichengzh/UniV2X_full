"""双向传播单元测试 (Phase 1A.5 验收)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from framework.capability_schema import HardwareCapability
from framework.config_schema import Config, UNIV2X_MODULES
from framework.propagation import propagate_d_to_b2, propagate_b2_to_d, propagate_all
from framework.constraints import is_legal


def _hw(name: str) -> HardwareCapability:
    return HardwareCapability.from_yaml(f"configs/hardware/{name}.yaml")


def test_d_to_b2_forces_int8_per_tensor():
    """DLA 路由 → 自动改为 INT8 per-tensor W+A."""
    hw = _hw("orin_agx")
    cfg = Config.fp32_baseline().with_field(
        d_routing={"backbone": "DLA0", "encoder": "GPU", "decoder": "GPU",
                   "heads": "GPU", "v2x_comm": "GPU"},
    )
    new = propagate_d_to_b2(cfg, hw)
    assert new.q_bits["backbone"] == "INT8"
    assert new.q_granularity["backbone"] == "per-tensor"
    assert new.q_object["backbone"] == "W+A"
    # 其他 GPU 模块不变
    assert new.q_bits["encoder"] == "FP32"
    assert new.q_bits["heads"] == "FP32"


def test_b2_to_d_per_channel_kicks_dla_to_gpu():
    """B2 per-channel → DLA 路由自动改 GPU."""
    hw = _hw("orin_agx")
    cfg = Config.fp32_baseline().with_field(
        d_routing={"backbone": "DLA0", "encoder": "GPU", "decoder": "GPU",
                   "heads": "GPU", "v2x_comm": "GPU"},
        q_granularity={"backbone": "per-channel", "encoder": "none", "decoder": "none",
                       "heads": "none", "v2x_comm": "none"},
    )
    new = propagate_b2_to_d(cfg, hw)
    assert new.d_routing["backbone"] == "GPU"


def test_propagate_all_reaches_fixpoint():
    """完整传播应当收敛."""
    hw = _hw("orin_agx")
    # 先放一个 DLA + per-channel 的冲突 → propagate_d_to_b2 应改 per-tensor
    cfg = Config.fp32_baseline().with_field(
        d_routing={"backbone": "DLA0", "encoder": "GPU", "decoder": "GPU",
                   "heads": "GPU", "v2x_comm": "GPU"},
        q_granularity={"backbone": "per-channel", "encoder": "none", "decoder": "none",
                       "heads": "none", "v2x_comm": "none"},
    )
    fixed = propagate_all(cfg, hw)
    # 双向传播后,backbone 应在 DLA 上跑 INT8 per-tensor (D→B2 优先生效)
    assert fixed.d_routing["backbone"] == "DLA0"
    assert fixed.q_bits["backbone"] == "INT8"
    assert fixed.q_granularity["backbone"] == "per-tensor"

    # 验证收敛后合法
    ok, reason = is_legal(fixed, hw)
    assert ok, reason


def test_propagate_no_dla_hw_is_identity():
    """没有 DLA 的硬件 (4090),双向传播应是恒等."""
    hw = _hw("rtx4090")
    cfg = Config.fp32_baseline().with_field(
        d_routing={m: "GPU" for m in UNIV2X_MODULES},
        q_bits={m: "INT8" for m in UNIV2X_MODULES},
        q_granularity={m: "per-channel" for m in UNIV2X_MODULES},
    )
    fixed = propagate_all(cfg, hw)
    assert fixed == cfg


def test_immutability():
    """propagate_* 不修改原 Config."""
    hw = _hw("orin_agx")
    cfg = Config.fp32_baseline().with_field(
        d_routing={"backbone": "DLA0", "encoder": "GPU", "decoder": "GPU",
                   "heads": "GPU", "v2x_comm": "GPU"},
    )
    original_q_bits = dict(cfg.q_bits)
    _ = propagate_all(cfg, hw)
    assert cfg.q_bits == original_q_bits, "原 cfg 被修改"


if __name__ == "__main__":
    test_d_to_b2_forces_int8_per_tensor()
    test_b2_to_d_per_channel_kicks_dla_to_gpu()
    test_propagate_all_reaches_fixpoint()
    test_propagate_no_dla_hw_is_identity()
    test_immutability()
    print("--- 1A.5 propagation: 5/5 tests passed ---")
