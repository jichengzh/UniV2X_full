"""粗略 latency 估计器 (Phase 1A.8)

只用于 Phase 1A 端到端 sanity. 模块级 LUT 留给 Stage 1.1 实测.

模型:
    latency = baseline * (sum_modules: weight_m * speedup_factor(bits, prune_rate))

speedup_factor 取自 1.1/1.3 实验经验值:
    FP32 → 1.0
    FP16 → 0.16  (1.1 实测: 5640ms PyTorch FP32 → 90ms FP16 TRT, 取保守 1/6)
    INT8 → 0.10  (FP16 进一步 ~1.6× 加速)
    剪枝率 r → (1 - 0.7 * r) (compute-bound 部分剪枝才有收益)
"""

from __future__ import annotations

from framework.config_schema import Config

# 模块在总 latency 中的占比 (来自 1.3 实验报告 D2_pipeline_overlap)
DEFAULT_MODULE_WEIGHT = {
    "backbone": 0.20,    # 32ms / 157ms
    "encoder":  0.40,    # ~63ms (BEV encoder 主导)
    "decoder":  0.20,
    "heads":    0.15,
    "v2x_comm": 0.05,
}

# 位宽加速系数 (相对 FP32)
BITS_SPEEDUP = {
    "FP32": 1.00,
    "FP16": 0.16,
    "INT8": 0.10,
    "FP8":  0.07,
    "INT4": 0.06,
}

# 平台基线 latency (ms, 全 FP32 全 GPU)
BASELINE_LATENCY = {
    "rtx4090": 90.0,        # 1.1 实测 BEV FP16 ≈ 90ms,FP32 接口估计 ~600ms;但 Phase 1A 只关心相对值
    "orin_agx": 200.0,      # Orin AGX 频率约 4090 一半,带宽 1/5
    "orin_nano": 400.0,
}


def estimate_latency(cfg: Config, hardware: str = "rtx4090") -> float:
    """估计配置在指定硬件上的 latency (ms).

    限制: 不考虑 fusion / kernel 切换 / DLA 加速,只是 sum-of-modules 粗估.
    Stage 1 实测 LUT 后会替换.
    """
    base = BASELINE_LATENCY.get(hardware, 200.0)
    total_factor = 0.0
    for m, w in DEFAULT_MODULE_WEIGHT.items():
        bits = cfg.q_bits.get(m, "FP32")
        rate = cfg.prune_rate.get(m, 0.0)
        bits_factor = BITS_SPEEDUP.get(bits, 1.0)
        prune_factor = 1.0 - 0.7 * rate
        total_factor += w * bits_factor * prune_factor
    # DLA 路由 (若适用) — Orin AGX 上 DLA 比 GPU INT8 慢一点点 (~1.1×) 但能耗低
    # Phase 1A 不区分 DLA 收益,保持简单
    return base * total_factor


__all__ = ["estimate_latency", "BASELINE_LATENCY", "BITS_SPEEDUP", "DEFAULT_MODULE_WEIGHT"]
