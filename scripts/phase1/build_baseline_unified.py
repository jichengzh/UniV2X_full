"""Phase 1A.7 — 1.1/1.2/1.3 数据整合为统一 baseline.

输入:
- quant_configs/*.json  (1.1 量化配置)
- prune_configs/*.json  (1.2 剪枝配置)
- 实验报告中手工提取的 (config_id, AMOTA) 数据点

输出:
- data/phase1/baseline_unified.parquet

后续:
- 由 Phase 2 Stage 2.4 数据完整性检查消费
- 由 Stage 2.5 主动采样补充联合配置
- 由 Stage 3 精度预测器训练消费
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from framework.config_schema import Config, UNIV2X_MODULES


# ============================================================
# 手工提取的 accuracy 数据点 (来自 1.1/1.2 实验报告)
# 字段: config_id, source, accuracy_amota, latency_ms, mAP (可选), notes
# ============================================================

ACCURACY_POINTS = [
    # ---------- 1.1 量化 ----------
    # 来源: paper_learning/1.1可配置量化实现记录/实验结果最终汇总/1_核心实验结果.md §二
    dict(config_id="all_fp32_pt", source="1.1_quant",
         accuracy_amota=0.338, latency_ms=5640.0, notes="all PyTorch FP32 baseline"),
    dict(config_id="bev_fp16_trt", source="1.1_quant",
         accuracy_amota=0.381, latency_ms=90.0, notes="BEV FP16 TRT, hookA"),
    dict(config_id="all_fp16_trt", source="1.1_quant",
         accuracy_amota=0.370, latency_ms=90.0, notes="all FP16 TRT, hookA+B+C+D"),
    dict(config_id="bev_int8_others_fp16", source="1.1_quant",
         accuracy_amota=0.364, latency_ms=None, notes="BEV INT8 + 其余 FP16, vanilla PTQ"),
    dict(config_id="heads_int8", source="1.1_quant",
         accuracy_amota=0.341, latency_ms=None, notes="下游头 INT8 修复后"),

    # ---------- 1.2 剪枝 Phase B (零微调 FFN 扫描) ----------
    # 来源: paper_learning/1.2可配置剪枝实现记录/实验结果最终汇总/1_核心实验结果.md §3.1
    dict(config_id="prune_baseline", source="1.2_prune",
         accuracy_amota=0.3298, mAP=0.0724, notes="prune baseline FP32"),
    dict(config_id="p1_ffn_20pct", source="1.2_prune",
         accuracy_amota=0.3055, mAP=0.0669, notes="FFN 20% 零微调"),
    dict(config_id="p1_ffn_30pct", source="1.2_prune",
         accuracy_amota=0.3189, mAP=0.0648, notes="FFN 30% 零微调"),
    dict(config_id="p1_ffn_40pct", source="1.2_prune",
         accuracy_amota=0.2973, mAP=0.0628, notes="FFN 40% 零微调"),
    dict(config_id="p1_ffn_50pct", source="1.2_prune",
         accuracy_amota=0.2668, mAP=0.0595, notes="FFN 50% 零微调"),
    dict(config_id="p1_ffn_60pct", source="1.2_prune",
         accuracy_amota=0.2366, mAP=0.0561, notes="FFN 60% 零微调"),

    # ---------- 1.2 剪枝 Phase B (微调后) ----------
    # 来源: §3.2 (q=1 / q=2 微调矩阵)
    dict(config_id="p1_ffn_30pct_ft_q1", source="1.2_prune_ft",
         accuracy_amota=0.3356, notes="FFN 30% q=1 微调 3 epoch"),
    dict(config_id="p1_ffn_30pct_ft_q2", source="1.2_prune_ft",
         accuracy_amota=0.3306, notes="FFN 30% q=2 微调"),
    dict(config_id="p1_ffn_50pct_ft_q1", source="1.2_prune_ft",
         accuracy_amota=0.2904, notes="FFN 50% q=1 微调"),
    dict(config_id="p1_ffn_50pct_ft_q2", source="1.2_prune_ft",
         accuracy_amota=0.3129, notes="FFN 50% q=2 微调"),
    dict(config_id="p1_ffn_60pct_ft_q2", source="1.2_prune_ft",
         accuracy_amota=0.3354, notes="FFN 60% q=2 微调 (Pareto 最优 trade)"),
    dict(config_id="p1_ffn_70pct_ft_q2", source="1.2_prune_ft",
         accuracy_amota=0.3102, notes="FFN 70% q=2 微调"),
    dict(config_id="p1_ffn_80pct_ft_q2", source="1.2_prune_ft",
         accuracy_amota=0.3102, notes="FFN 80% q=2 微调"),

    # ---------- 1.2 Phase D.2 剪枝 × 量化联合 ----------
    # 来源: §五
    dict(config_id="d22_bound_ffn60_int8", source="1.2_joint",
         accuracy_amota=0.287, mAP=0.071, notes="FFN 60% 绑定 + INT8 W+A"),
    dict(config_id="d27_decouple_enc10_07_int8", source="1.2_joint",
         accuracy_amota=0.360, mAP=0.074, notes="解耦 enc=1.0 dec=0.7 + INT8 W+A,联合最优"),

    # ---------- 1.2 全局 Pareto ----------
    # 来源: §8.1
    dict(config_id="d14_decouple_enc10_07_fp32", source="1.2_pareto",
         accuracy_amota=0.367, mAP=0.073, notes="enc=1.0 dec=0.7 FP32, Track 最优"),
    dict(config_id="d12_decouple_enc08_03_fp32", source="1.2_pareto",
         accuracy_amota=0.337, mAP=0.070, notes="enc=0.8 dec=0.3 FP32, 多任务平衡"),
]


# ============================================================
# 配置生成 — 把 accuracy 数据点和 quant_configs / prune_configs 文件对齐
# ============================================================

def _make_config_for_point(p: dict, root: Path) -> Config:
    """根据 accuracy 数据点的 config_id / source 构造 Config 对象."""
    cid = p["config_id"]
    src = p["source"]

    # 默认 baseline
    cfg = Config.fp32_baseline()

    if src == "1.1_quant":
        # 1.1 量化点 - 手工映射模块精度
        if cid == "all_fp32_pt":
            cfg = Config.fp32_baseline()
        elif cid == "bev_fp16_trt":
            cfg = cfg.with_field(
                q_bits={**cfg.q_bits, "encoder": "FP16"},
                q_granularity={**cfg.q_granularity, "encoder": "per-tensor"},
                q_object={**cfg.q_object, "encoder": "W+A"},
            )
        elif cid == "all_fp16_trt":
            cfg = cfg.with_field(
                q_bits={m: "FP16" for m in UNIV2X_MODULES},
                q_granularity={m: "per-tensor" for m in UNIV2X_MODULES},
                q_object={m: "W+A" for m in UNIV2X_MODULES},
            )
        elif cid == "bev_int8_others_fp16":
            cfg = cfg.with_field(
                q_bits={
                    "backbone": "FP16", "encoder": "INT8", "decoder": "FP16",
                    "heads": "FP16", "v2x_comm": "FP16",
                },
                q_granularity={m: "per-tensor" for m in UNIV2X_MODULES},
                q_object={m: "W+A" for m in UNIV2X_MODULES},
            )
        elif cid == "heads_int8":
            cfg = cfg.with_field(
                q_bits={**cfg.q_bits, "heads": "INT8"},
                q_granularity={**cfg.q_granularity, "heads": "per-tensor"},
                q_object={**cfg.q_object, "heads": "W+A"},
            )

    elif src.startswith("1.2_prune"):
        # 1.2 剪枝点 - 从 prune_configs/*.json 加载 (如果文件存在)
        # config_id 对应 prune_configs/{cid 去掉_ft_q*}.json
        base_id = cid.replace("_ft_q1", "").replace("_ft_q2", "")
        json_path = root / "prune_configs" / f"{base_id}.json"
        if json_path.exists():
            cfg = Config.from_prune_json(json_path)
        else:
            # baseline / 没有对应配置文件 → 用默认 + 手工填
            cfg = Config.fp32_baseline()
            cfg = cfg.with_field(
                prune_object="channel" if "ffn" in cid else "none",
                prune_criterion={m: "L1" for m in UNIV2X_MODULES},
            )

    elif src in ("1.2_joint", "1.2_pareto"):
        # 联合点 - 从 prune_configs/decouple_*.json 加载并叠加量化
        if "enc10_07" in cid:
            json_path = root / "prune_configs" / "decouple_enc10_07.json"
        elif "enc08_03" in cid:
            json_path = root / "prune_configs" / "decouple_enc08_03.json"
        elif "ffn60" in cid:
            json_path = root / "prune_configs" / "p1_ffn_60pct.json"
        else:
            json_path = None

        if json_path and json_path.exists():
            cfg = Config.from_prune_json(json_path)
        # 叠加量化
        if "int8" in cid.lower():
            cfg = cfg.with_field(
                q_bits={m: "INT8" for m in UNIV2X_MODULES},
                q_granularity={m: "per-tensor" for m in UNIV2X_MODULES},
                q_object={m: "W+A" for m in UNIV2X_MODULES},
            )

    cfg = cfg.with_field(config_id=cid, source=src)
    return cfg


def main() -> None:
    rows = []
    for p in ACCURACY_POINTS:
        cfg = _make_config_for_point(p, ROOT)
        row = {
            "config_id": p["config_id"],
            "source": p["source"],
            # target
            "accuracy_amota": p["accuracy_amota"],
            "accuracy_diff_vs_fp32": 0.330 - p["accuracy_amota"],  # 用 1.2 baseline 0.330
            "latency_ms": p.get("latency_ms"),
            "mAP": p.get("mAP"),
            # B1 剪枝
            "prune_object": cfg.prune_object,
            "prune_rate__backbone": cfg.prune_rate.get("backbone", 0.0),
            "prune_rate__encoder": cfg.prune_rate.get("encoder", 0.0),
            "prune_rate__decoder": cfg.prune_rate.get("decoder", 0.0),
            "prune_rate__heads": cfg.prune_rate.get("heads", 0.0),
            "prune_rate__v2x_comm": cfg.prune_rate.get("v2x_comm", 0.0),
            "prune_criterion__encoder": cfg.prune_criterion.get("encoder", "none"),
            # B2 量化
            "q_bits__backbone": cfg.q_bits.get("backbone", "FP32"),
            "q_bits__encoder": cfg.q_bits.get("encoder", "FP32"),
            "q_bits__decoder": cfg.q_bits.get("decoder", "FP32"),
            "q_bits__heads": cfg.q_bits.get("heads", "FP32"),
            "q_bits__v2x_comm": cfg.q_bits.get("v2x_comm", "FP32"),
            "q_granularity__encoder": cfg.q_granularity.get("encoder", "none"),
            "q_object__encoder": cfg.q_object.get("encoder", "none"),
            # D 路由 (Phase 1A 全 GPU)
            "d_routing__backbone": cfg.d_routing.get("backbone", "GPU"),
            # 派生
            "prune_rate__avg": cfg.avg_prune_rate(),
            "has_dla": cfg.has_dla(),
            "notes": p.get("notes", ""),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    out_path = ROOT / "data" / "phase1" / "baseline_unified.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    # 同时输出 csv 方便 review
    df.to_csv(out_path.with_suffix(".csv"), index=False)

    print(f"=== Phase 1A.7: baseline_unified ===")
    print(f"rows: {len(df)}")
    print(f"sources: {df['source'].value_counts().to_dict()}")
    print(f"AMOTA range: [{df['accuracy_amota'].min():.4f}, {df['accuracy_amota'].max():.4f}]")
    print(f"AMOTA span:  {df['accuracy_amota'].max() - df['accuracy_amota'].min():.4f}")
    print(f"输出: {out_path}")
    print(f"     {out_path.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
