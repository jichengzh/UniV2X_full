"""特征工程: Config → 数值向量 (Stage S3.1)

对应 v1.5 §6.4 / 实施计划 §3.4 / §6.4.

设计原则:
- 类别特征不做 one-hot,LightGBM 原生支持(传字符串即可)
- 数值特征按 module 展开 + 全局派生
- 派生特征手工注入 (跨层 std / 加权平均 / 复合压力 / DLA flag)
- 输入: Config 对象 或 baseline_unified.parquet 行
- 输出: pd.DataFrame (单行 = 一个配置, 列 = 特征)
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from framework.config_schema import Config, UNIV2X_MODULES

# 模块在总 FLOPs 中的权重 (与 latency_estimator.DEFAULT_MODULE_WEIGHT 一致,但语义是 FLOPs)
MODULE_FLOPS_WEIGHT = {
    "backbone": 0.20,
    "encoder":  0.40,
    "decoder":  0.20,
    "heads":    0.15,
    "v2x_comm": 0.05,
}

# 位宽 → 数值映射 (用于加权平均)
BITS_TO_NUM = {
    "FP32": 32, "FP16": 16, "INT8": 8, "FP8": 8, "INT4": 4,
}

# 类别特征列 (LightGBM 把这些列当 categorical 处理)
CATEGORICAL_COLS = (
    "prune_object",
    *(f"prune_criterion__{m}" for m in UNIV2X_MODULES),
    *(f"q_bits__{m}" for m in UNIV2X_MODULES),
    *(f"q_granularity__{m}" for m in UNIV2X_MODULES),
    *(f"q_object__{m}" for m in UNIV2X_MODULES),
    *(f"d_routing__{m}" for m in UNIV2X_MODULES),
)


def encode_config(cfg: Config) -> dict:
    """单个 Config → 特征字典 (一行 DataFrame)."""
    row: dict = {}

    # ---------- 类别特征 (LightGBM 原生支持) ----------
    row["prune_object"] = cfg.prune_object
    for m in UNIV2X_MODULES:
        row[f"prune_criterion__{m}"] = cfg.prune_criterion.get(m, "none")
        row[f"q_bits__{m}"] = cfg.q_bits.get(m, "FP32")
        row[f"q_granularity__{m}"] = cfg.q_granularity.get(m, "none")
        row[f"q_object__{m}"] = cfg.q_object.get(m, "none")
        row[f"d_routing__{m}"] = cfg.d_routing.get(m, "GPU")

    # ---------- 数值特征 — 按 module ----------
    rates = []
    bit_nums = []
    for m in UNIV2X_MODULES:
        r = float(cfg.prune_rate.get(m, 0.0))
        b = BITS_TO_NUM.get(cfg.q_bits.get(m, "FP32"), 32)
        row[f"prune_rate__{m}"] = r
        row[f"q_bits_num__{m}"] = b
        rates.append(r)
        bit_nums.append(b)

    rates_arr = np.array(rates, dtype=float)
    bits_arr = np.array(bit_nums, dtype=float)
    weights_arr = np.array(
        [MODULE_FLOPS_WEIGHT[m] for m in UNIV2X_MODULES], dtype=float
    )

    # ---------- 派生特征 (跨模块统计 / 复合压力 / DLA flag) ----------
    row["prune_rate__avg"] = float(rates_arr.mean())
    row["prune_rate__std"] = float(rates_arr.std())                 # 跨层梯度替代量
    row["prune_rate__max"] = float(rates_arr.max())
    row["prune_rate__min"] = float(rates_arr.min())
    row["prune_rate__weighted_avg"] = float((rates_arr * weights_arr).sum())
    row["q_bits_num__min"] = float(bits_arr.min())                  # 最低位宽 = 瓶颈
    row["q_bits_num__weighted_avg"] = float((bits_arr * weights_arr).sum())
    row["q_bits_num__std"] = float(bits_arr.std())
    # criterion_diversity = 不同模块用了多少种剪枝准则 (除 'none')
    crits = {cfg.prune_criterion.get(m, "none") for m in UNIV2X_MODULES}
    crits.discard("none")
    row["criterion_diversity"] = len(crits)
    # 是否含 DLA 路由
    row["is_dla_routed"] = int(cfg.has_dla())
    # 复合压力 = 平均剪枝率 * (1 - 平均位宽/32)
    row["compound_pressure"] = float(
        row["prune_rate__avg"] * (1 - row["q_bits_num__weighted_avg"] / 32.0)
    )
    # 是否纯 baseline (剪枝率全 0 + 全 FP32)
    row["is_pure_baseline"] = int(rates_arr.sum() == 0 and bits_arr.min() == 32)

    return row


def encode_configs(configs: Iterable[Config]) -> pd.DataFrame:
    rows = [encode_config(c) for c in configs]
    df = pd.DataFrame(rows)
    # 把类别列转 category dtype (LightGBM 期望)
    for c in CATEGORICAL_COLS:
        if c in df.columns:
            df[c] = df[c].astype("category")
    return df


def encode_baseline_df(baseline_df: pd.DataFrame) -> pd.DataFrame:
    """把 baseline_unified.parquet 直接转特征矩阵 (不需要先变成 Config 对象).

    要求 baseline_df 至少包含字段:
      prune_object, prune_rate__{module}, q_bits__{module}, q_granularity__encoder,
      q_object__encoder, prune_criterion__encoder, d_routing__backbone
    缺失字段会按 baseline 默认填.
    """
    rows = []
    for _, r in baseline_df.iterrows():
        prune_rate = {m: float(r.get(f"prune_rate__{m}", 0.0)) for m in UNIV2X_MODULES}
        q_bits = {m: r.get(f"q_bits__{m}", "FP32") for m in UNIV2X_MODULES}
        # q_granularity / q_object 在 baseline_unified 里只 encoder 列存了,其它推断
        q_gran = {m: ("none" if q_bits[m] == "FP32" else r.get("q_granularity__encoder", "per-tensor"))
                  for m in UNIV2X_MODULES}
        q_obj = {m: ("none" if q_bits[m] == "FP32" else r.get("q_object__encoder", "W+A"))
                 for m in UNIV2X_MODULES}
        # prune_criterion 同上
        crit = r.get("prune_criterion__encoder", "L1")
        prune_crit = {m: ("none" if prune_rate[m] == 0 else crit) for m in UNIV2X_MODULES}
        d_route = {m: r.get("d_routing__backbone", "GPU") for m in UNIV2X_MODULES}

        cfg = Config(
            prune_rate=prune_rate,
            prune_object=r.get("prune_object", "none"),
            prune_criterion=prune_crit,
            q_bits=q_bits,
            q_granularity=q_gran,
            q_object=q_obj,
            d_routing=d_route,
            config_id=r.get("config_id"),
            source=r.get("source"),
        )
        rows.append(encode_config(cfg))

    df = pd.DataFrame(rows)
    for c in CATEGORICAL_COLS:
        if c in df.columns:
            df[c] = df[c].astype("category")
    return df


__all__ = [
    "encode_config",
    "encode_configs",
    "encode_baseline_df",
    "CATEGORICAL_COLS",
    "BITS_TO_NUM",
    "MODULE_FLOPS_WEIGHT",
]
