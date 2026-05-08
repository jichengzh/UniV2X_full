"""HEAL Pyramid Fusion adapter (M4.4).

把 v1.5 Config schema 5 模块映射到 Pyramid Fusion (HEAL) 模块结构.

模块映射 (基于 opencood/models/heter_pyramid_collab.py + lidar_pyramid.yaml):
  v1.5 Config       → Pyramid Fusion (HEAL)
  ───────────────────────────────────────────────
  backbone           → encoder_{m} (PointPillar / Lift-Splat) + backbone_{m} (ResNet)
  encoder            → ResNetBEVBackbone 内部 conv blocks (BEV 编码)
  decoder            → pyramid_backbone (PyramidFusion, 3 stages 多尺度 CNN fusion)
                       ⚠️ 注意: Pyramid 的 decoder 是 CNN! (vs UniV2X / univ2x-tiny 是 Transformer)
  heads              → cls_head + reg_head + dir_head (3 个并行 head)
  v2x_comm           → aligner_{m} + multi-modality weighted fusion (跨 agent)

关键架构差异 vs UniV2X / univ2x-tiny:
  ✅ 全 CNN, 零自定义 plugin (无 MSDA/Rotate/DCN)
  ✅ 全模块可剪 (标准 conv, 含 backbone)
  ✅ 直接支持 NVIDIA SparsityINT8 (2:4)
  ✅ ONNX 跨平台无 plugin 依赖 (Orin 直接 trtexec build)
  ✅ M2 f 函数对 Pyramid 是 interpolation (M4.3 实测 4.49ms 在 M2 训练范围内)

PHYSICAL 约束:
  - 多模态情况 v2x_comm 必须保持 (HEAL 核心设计 — 不能剪掉)
  - decoder (PyramidFusion) layer_nums [3,5,8] 决定 3 stages, 不能改 stage 数
  - shrink_header 可选, 不存在时不影响搜索
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from framework.config_schema import Config, UNIV2X_MODULES

ROOT = Path(__file__).resolve().parents[2]
M2_F_JSON = ROOT / "results/m2_latency_mapping_f.json"
PYRAMID_4090_LATENCY = ROOT / "results/m4_3_pyramid_fusion_4090_latency.csv"

# Pyramid Fusion 物理约束
PYRAMID_CONSTRAINTS = {
    # 全模块可剪 (vs UniV2X DCN 反例)
    "all_modules_pruneable": True,
    # 支持 2:4 sparsity (DL4AGX SparsityINT8 现成)
    "sparse_24_supported": True,
    # 无自定义 plugin → ONNX 跨平台可用
    "onnx_portable": True,
    # decoder 是 CNN (PyramidFusion 继承 ResNetBEVBackbone)
    "decoder_arch": "CNN",
    # 多 agent 通信不能剪掉 (HEAL 核心)
    "v2x_comm_required": True,
    # Pyramid 3 stages 固定
    "fusion_stages": 3,
    # M4.3 实测 baseline (lidar_pyramid 配置)
    "params_M_default": 3.76,
    "lat_4090_pytorch_fp32_ms_default": 4.49,
    "lat_4090_pytorch_fp16_ms_default": 3.27,
}


@dataclass(frozen=True)
class PyramidFusionInfo:
    """Pyramid Fusion (HEAL) 元信息."""
    backbone: str = "ResNet (resnext)"
    fusion_stages: int = 3
    layer_nums: tuple = (3, 5, 8)
    num_filters: tuple = (64, 128, 256)
    has_v2x_comm: bool = True
    has_dcn: bool = False
    has_attention: bool = False
    is_pure_cnn: bool = True


def is_valid_for_pyramid(cfg: Config) -> tuple[bool, str]:
    """Pyramid Fusion 物理约束检查.

    Pyramid 是"宽容架构" — 全模块可剪/可量化, 仅 v2x_comm 不能完全剪掉.
    """
    # v2x_comm 是 HEAL 核心 fusion 模块, 不能完全剪掉 (rate < 1.0)
    # 但允许部分剪枝 (例如 50% rate 可以)
    if cfg.prune_rate.get("v2x_comm", 0.0) >= 1.0:
        return False, "[PHYSICAL/pyramid] v2x_comm 是 HEAL 核心 fusion 模块, 不能完全剪掉"

    # decoder (PyramidFusion) 不能用 Transformer 专用准则 (Taylor/Wanda)
    # 因为 Pyramid decoder 是 CNN, Taylor/Wanda 在 conv 上效果不如 L1/FPGM
    crit = cfg.prune_criterion.get("decoder", "L1")
    if crit in ("Taylor", "Wanda") and cfg.prune_rate.get("decoder", 0.0) > 0:
        return False, (
            f"[PHYSICAL/pyramid] Pyramid decoder 是 CNN, "
            f"准则 {crit} 是 Transformer 专用 (应改用 L1/FPGM)"
        )

    # 其它无限制 — 全 CNN 完全友好
    return True, ""


def config_to_baseline_row(cfg: Config) -> dict:
    """把 Config 转成 baseline 兼容的行 (与 baseline_4090.parquet schema 对齐).

    用于把搜索器候选 + 评估结果回写到 baseline 表 (M4.5 + Phase 2.5).
    """
    row: dict = {
        "config_id": cfg.config_id or "anonymous",
        "source": "pyramid_fusion",
        "model_class": "pyramid_fusion",
        "is_real_measured": False,
        "prune_object": cfg.prune_object,
    }
    for m in UNIV2X_MODULES:
        row[f"prune_rate__{m}"] = float(cfg.prune_rate.get(m, 0.0))
        row[f"prune_criterion__{m}"] = cfg.prune_criterion.get(m, "none")
        row[f"q_bits__{m}"] = cfg.q_bits.get(m, "FP32")
        row[f"q_granularity__{m}"] = cfg.q_granularity.get(m, "none")
        row[f"q_object__{m}"] = cfg.q_object.get(m, "none")
        row[f"d_routing__{m}"] = cfg.d_routing.get(m, "GPU")
    return row


def load_baseline() -> pd.DataFrame:
    """加载 Pyramid Fusion baseline (M4.3 实测的 4090 latency 1 行).

    ⚠️ 只有 1 行 — 是 lidar_pyramid 默认配置的 baseline.
    M4.5 跑 5-10 个 prune/quant configs 后再扩充到完整 baseline.
    """
    if not PYRAMID_4090_LATENCY.exists():
        raise FileNotFoundError(
            f"{PYRAMID_4090_LATENCY} 不存在. 请先运行 scripts/phase1/m4_3_pyramid_fusion_baseline.py"
        )
    return pd.read_csv(PYRAMID_4090_LATENCY)


def estimate_orin_latency(
    lat_4090_pytorch_fp32_ms: float, precision: str = "fp16",
) -> tuple[float, float]:
    """用 M2 f 函数把 4090 PyTorch FP32 latency 映射到 Orin AGX TRT.

    返回 (estimated_ms, uncertainty_ms).

    ✅ Pyramid Fusion 在 M2 f 训练范围内 (4.49ms ∈ [1, 7]ms),
    interpolation 而非 extrapolation, 估算可信度高.
    """
    if not M2_F_JSON.exists():
        raise FileNotFoundError(f"{M2_F_JSON} 不存在 (M2 未跑)")
    with open(M2_F_JSON, "r", encoding="utf-8") as f:
        fits = json.load(f)["fits"]

    key_map = {
        "fp16": "f_fp16_from_4090_fp32",
        "int8": "f_int8_from_4090_fp32",
    }
    key = key_map.get(precision)
    if key is None:
        raise ValueError(f"unsupported precision: {precision}")
    fit = fits[key]
    a, b, stderr = fit["intercept_a"], fit["slope_b"], fit["stderr"]
    est = a + b * lat_4090_pytorch_fp32_ms
    uncertainty = stderr * abs(lat_4090_pytorch_fp32_ms)
    return est, uncertainty


def get_module_subnet_type(module: str) -> str:
    """Pyramid Fusion 子网类型 (与 v1.5 §0.5 对齐).

    ⚠️ 关键差异 vs uniad_tiny adapter:
       Pyramid decoder 是 CNN (PyramidFusion 继承 ResNetBEVBackbone)
       而 uniad_tiny decoder 是 Transformer.
       这影响精度预测器 (lgb_v5_1) 的特征工程.
    """
    return {
        "backbone": "CNN",       # encoder_{m} + backbone_{m} (ResNet)
        "encoder": "CNN",        # ResNetBEVBackbone 内部
        "decoder": "CNN",        # ★ PyramidFusion 是 CNN, 与 uniad_tiny Transformer 不同
        "heads": "MLP",          # cls/reg/dir heads
        "v2x_comm": "CNN",       # ★ aligner + weighted fusion 也是 CNN, 与 uniad_tiny MLP 不同
    }[module]


def get_recommended_criterion_pool(module: str) -> tuple[str, ...]:
    """Pyramid Fusion 各模块推荐准则池 (按 v1.5 §0.5).

    Pyramid 全 CNN, 所有模块都用 L1/FPGM 系列 (而非 uniad_tiny 的混合方案).
    """
    if module == "heads":
        return ("L1",)  # MLP, 简单
    return ("L1", "FPGM")  # 所有 CNN 模块都用 L1/FPGM


__all__ = [
    "PyramidFusionInfo",
    "PYRAMID_CONSTRAINTS",
    "is_valid_for_pyramid",
    "config_to_baseline_row",
    "load_baseline",
    "estimate_orin_latency",
    "get_module_subnet_type",
    "get_recommended_criterion_pool",
]


# ============================================================
# Demo / smoke test
# ============================================================

if __name__ == "__main__":
    print("=== M4.4 pyramid_fusion adapter smoke test ===")

    # 1. load baseline
    df = load_baseline()
    print(f"✅ baseline loaded: {len(df)} row(s) (M4.3 实测)")
    print(df.to_string(index=False))

    # 2. estimate_orin_latency (interpolation, 可信度高)
    est_fp16, unc_fp16 = estimate_orin_latency(4.49, "fp16")
    est_int8, unc_int8 = estimate_orin_latency(4.49, "int8")
    print(f"\n✅ estimate_orin_latency(4.49ms 4090):")
    print(f"   fp16: {est_fp16:.3f} ± {unc_fp16:.3f} ms (M2 interpolation, 可信)")
    print(f"   int8: {est_int8:.3f} ± {unc_int8:.3f} ms")

    # 3. is_valid_for_pyramid (各种 case)
    cfg_ok = Config.fp32_baseline()
    ok, reason = is_valid_for_pyramid(cfg_ok)
    print(f"\n✅ FP32 baseline valid: {ok}")

    # bad case 1: v2x_comm 完全剪掉
    cfg_bad1 = cfg_ok.with_field(
        prune_object="channel",
        prune_rate={**{m: 0.0 for m in UNIV2X_MODULES}, "v2x_comm": 1.0},
    )
    ok, reason = is_valid_for_pyramid(cfg_bad1)
    print(f"   v2x_comm=100% prune valid: {ok}, reason: {reason}")

    # bad case 2: decoder 用 Taylor (Transformer 准则)
    cfg_bad2 = cfg_ok.with_field(
        prune_object="channel",
        prune_rate={**{m: 0.0 for m in UNIV2X_MODULES}, "decoder": 0.3},
        prune_criterion={**{m: "none" for m in UNIV2X_MODULES}, "decoder": "Taylor"},
    )
    ok, reason = is_valid_for_pyramid(cfg_bad2)
    print(f"   decoder Taylor (Pyramid CNN) valid: {ok}, reason: {reason}")

    # 4. subnet type 对比
    print("\n✅ subnet type (Pyramid vs UniAD-tiny 关键差异):")
    print(f"   {'module':<12} {'pyramid':<12} {'uniad_tiny':<12}")
    from framework.adapters import uniad_tiny as u_tiny
    for m in UNIV2X_MODULES:
        p_st = get_module_subnet_type(m)
        u_st = u_tiny.get_module_subnet_type(m)
        diff_marker = "  ← 差异" if p_st != u_st else ""
        print(f"   {m:<12} {p_st:<12} {u_st:<12}{diff_marker}")

    # 5. config_to_baseline_row
    row = config_to_baseline_row(cfg_ok.with_field(config_id="pyramid_demo"))
    print(f"\n✅ config_to_baseline_row: {len(row)} keys, source={row['source']}")
