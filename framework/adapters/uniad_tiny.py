"""univ2x-tiny variant adapter (M5.3).

把 v1.5 Config schema 5 模块映射到 univ2x-tiny variant 模块名.
univ2x-tiny variant: R50 + 50x50 BEV (vs UniV2X R101 + 200x200 + DCNv2)

模块映射 (基于 stage5_baseline_v3.csv 23 configs schema):
  v1.5 Config       → univ2x-tiny variant
  ────────────────────────────────────────────
  backbone           → ResNet-50 主干 (无 DCN)
  encoder            → BEV encoder (50x50, 减少 deformable attention)
  decoder            → tracking decoder (3 layers, 比 UniV2X 6 层少)
  heads              → planning + occupancy + map heads
  v2x_comm           → (univ2x-tiny variant 不含 V2X 通信, 此模块全 0/none)

Adapter 接口:
  - config_to_baseline_row(cfg) → dict (与 baseline_4090.parquet 兼容行)
  - is_valid_for_uniad_tiny(cfg) → (bool, reason) 检查 univ2x-tiny 物理约束
  - load_baseline() → pd.DataFrame (data/uniad_tiny_baseline.csv)
  - estimate_orin_latency(lat_4090, prec) → (mean_ms, uncertainty_ms)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from framework.config_schema import Config, UNIV2X_MODULES

# 路径
ROOT = Path(__file__).resolve().parents[2]
BASELINE_CSV = ROOT / "data/uniad_tiny_baseline.csv"
M2_F_JSON = ROOT / "results/m2_latency_mapping_f.json"

# univ2x-tiny variant 物理约束 (v1.5 §0.3 PHYSICAL 类别 — 架构限制)
TINY_VARIANT_CONSTRAINTS = {
    # 不含 V2X 通信模块 (与 UniV2X 主线最大差异)
    "no_v2x_comm": True,
    # decoder 只有 3 层 (vs UniV2X 6 层)
    "decoder_max_layers": 3,
    # 50x50 BEV size (vs UniV2X 200x200)
    "bev_size": (50, 50),
    # backbone R50 无 DCN, 标准 conv 全可剪
    "backbone_pruneable": True,
}


@dataclass(frozen=True)
class TinyVariantInfo:
    """univ2x-tiny variant 的元信息 (用于 adapter)."""
    backbone: str = "ResNet-50"
    bev_size: tuple = (50, 50)
    decoder_layers: int = 3
    has_v2x_comm: bool = False
    has_dcn: bool = False


def is_valid_for_uniad_tiny(cfg: Config) -> tuple[bool, str]:
    """检查 Config 是否与 univ2x-tiny variant 物理约束兼容.

    扩展 framework/constraints.py 的 PHYSICAL 类别 — 网络架构特定约束.
    """
    # v2x_comm 模块在 tiny variant 不存在, 必须 0 剪 / FP32 / GPU
    if cfg.prune_rate.get("v2x_comm", 0.0) > 0:
        return False, "[PHYSICAL/uniad_tiny] v2x_comm 模块在 tiny variant 不存在 (剪枝率必须 0)"
    if cfg.q_bits.get("v2x_comm", "FP32") != "FP32":
        return False, "[PHYSICAL/uniad_tiny] v2x_comm 模块在 tiny variant 不存在 (必须 FP32)"
    if cfg.d_routing.get("v2x_comm", "GPU") != "GPU":
        return False, "[PHYSICAL/uniad_tiny] v2x_comm 模块在 tiny variant 不存在 (路由必须 GPU)"

    # backbone 剪枝 — tiny variant 是 R50 标准 conv, 全可剪 (无 UniV2X DCN 限制)
    # 这是 tiny variant 相对 UniV2X 的优势 — 不加额外约束
    return True, ""


def config_to_baseline_row(cfg: Config) -> dict:
    """把 Config 转成 baseline_4090.parquet 兼容的行 (M5.5 + Phase 2.5 用).

    主要用于把搜索器候选 + 评估结果回写到 baseline 表.
    """
    row: dict = {
        "config_id": cfg.config_id or "anonymous",
        "source": "uniad_tiny_variant",
        "model_class": "uniad_tiny_variant",
        "is_real_measured": False,  # 默认估算; 真实测后改为 True
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
    """加载 univ2x-tiny baseline (data/uniad_tiny_baseline.csv, 23 configs).

    返回 DataFrame, 含 4090 实测 latency + M2 f 估算 Orin latency.
    """
    if not BASELINE_CSV.exists():
        raise FileNotFoundError(
            f"{BASELINE_CSV} 不存在. 请先运行 scripts/phase1/m5_2_uniad_tiny_baseline.py"
        )
    return pd.read_csv(BASELINE_CSV)


def estimate_orin_latency(
    lat_4090_pytorch_fp32_ms: float, precision: str = "fp16",
) -> tuple[float, float]:
    """用 M2 f 函数把 4090 PyTorch FP32 latency 映射到 Orin AGX TRT.

    返回 (estimated_ms, uncertainty_ms).
    precision: "fp16" 或 "int8".

    ⚠️ 外推 warning: M2 f 训练范围 ResNet 1-7ms, univ2x-tiny ~500ms 是 ~75× 外推.
    Phase 3 需用真实 Orin trtexec 验证.
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
        raise ValueError(f"unsupported precision: {precision} (use 'fp16' or 'int8')")

    fit = fits[key]
    a, b, stderr = fit["intercept_a"], fit["slope_b"], fit["stderr"]
    est = a + b * lat_4090_pytorch_fp32_ms
    uncertainty = stderr * abs(lat_4090_pytorch_fp32_ms)
    return est, uncertainty


def get_module_subnet_type(module: str) -> str:
    """univ2x-tiny variant 的子网类型映射 (与 v1.5 §0.5 对齐).

    用于 framework/lgb_v5_1 (精度预测器) 的特征工程.
    """
    return {
        "backbone": "CNN",       # ResNet-50
        "encoder": "CNN",        # BEV encoder (CNN-based, 50x50)
        "decoder": "Transformer",  # tracking decoder (attention)
        "heads": "MLP",          # planning + occupancy + map heads
        "v2x_comm": "MLP",       # tiny variant 无此模块, fallback
    }[module]


__all__ = [
    "TinyVariantInfo",
    "TINY_VARIANT_CONSTRAINTS",
    "is_valid_for_uniad_tiny",
    "config_to_baseline_row",
    "load_baseline",
    "estimate_orin_latency",
    "get_module_subnet_type",
]


# ============================================================
# Demo / smoke test
# ============================================================

if __name__ == "__main__":
    print("=== M5.3 univ2x-tiny adapter smoke test ===")

    # 1. load baseline
    df = load_baseline()
    print(f"✅ baseline loaded: {len(df)} configs")
    print(f"   amota: {df['amota'].min():.4f} - {df['amota'].max():.4f}")
    print(f"   lat_4090: {df['lat_4090_pytorch_fp32_ms'].min():.1f} - "
          f"{df['lat_4090_pytorch_fp32_ms'].max():.1f} ms")

    # 2. estimate_orin_latency
    est_fp16, unc_fp16 = estimate_orin_latency(540.0, "fp16")
    est_int8, unc_int8 = estimate_orin_latency(540.0, "int8")
    print(f"\n✅ estimate_orin_latency(540ms 4090):")
    print(f"   fp16: {est_fp16:.1f} ± {unc_fp16:.1f} ms")
    print(f"   int8: {est_int8:.1f} ± {unc_int8:.1f} ms")

    # 3. is_valid_for_uniad_tiny
    cfg_ok = Config.fp32_baseline()
    ok, reason = is_valid_for_uniad_tiny(cfg_ok)
    print(f"\n✅ FP32 baseline valid: {ok}, reason: '{reason}'")

    cfg_bad = Config(
        prune_rate={m: 0.0 for m in UNIV2X_MODULES},
        prune_object="none",
        prune_criterion={m: "none" for m in UNIV2X_MODULES},
        q_bits={m: "FP32" for m in UNIV2X_MODULES},
        q_granularity={m: "none" for m in UNIV2X_MODULES},
        q_object={m: "none" for m in UNIV2X_MODULES},
        d_routing={m: "GPU" for m in UNIV2X_MODULES},
    ).with_field(
        prune_rate={**{m: 0.0 for m in UNIV2X_MODULES}, "v2x_comm": 0.5}
    )
    ok, reason = is_valid_for_uniad_tiny(cfg_bad)
    print(f"   v2x_comm pruned 0.5 valid: {ok} (expected False)")
    print(f"   reason: {reason}")

    # 4. config_to_baseline_row
    row = config_to_baseline_row(cfg_ok.with_field(config_id="demo"))
    print(f"\n✅ config_to_baseline_row(FP32 baseline):")
    print(f"   {len(row)} keys, source={row['source']}")
