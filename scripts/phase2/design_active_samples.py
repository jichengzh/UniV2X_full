"""Stage 2.5 主动采样设计 — 生成 30 个差异化 Config (Phase 2 §6.1).

设计依据: 2.0阶段_phase2_sanity报告.md §4 + Stage 3 sanity feature_importance 暴露的盲点.

5 类目标(共 30 个),每类直接对应一个被 sanity 模型忽略的特征族:

  A. 解锁 prune_rate__backbone (6)         — 当前 22 行全 0
  B. 解锁 prune_rate__heads (3)            — 当前 22 行全 0
  C. 解锁多模块剪枝差异化 prune_rate__std (4)— 让 std > 0.1
  D. 解锁多模块量化差异化 q_bits__* 间相关性 (5) — 当前 q_bits__* 几乎全同步
  E. 剪枝 × 量化联合 (12)                   — 核心 motivation,当前只有 4 行

设计原则:
- 每个配置必须通过 framework.constraints.is_legal (physical + empirical)
- 每个配置都要"反映现有 quant/prune_configs 的字段命名",方便 4090 上跑 PTQ 直接用
- 所有 d_routing 锁 GPU (Phase 1A 决策 D1 — DLA 留 Phase 3)
- 所有 prune_object = channel (与 1.2 一致;2:4 留 Stage 1 LUT 验证后再用)

输出:
- data/phase2/active_samples_plan.csv      30 行汇总
- prune_configs/active_NNN_*.json          30 个剪枝 JSON (沿用 1.2 schema)
- quant_configs/active_NNN_*.json          30 个量化 JSON (沿用 1.1 schema)
- paper_learning/2.5阶段_主动采样设计.md   给用户 review 的报告
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from framework.capability_schema import HardwareCapability
from framework.config_schema import Config, UNIV2X_MODULES
from framework.constraints import is_legal, list_violations
from framework.propagation import propagate_all


# ============================================================
# 30 个采样配置的"目标 + 高层描述" — 之后被代码具体化
# ============================================================

SAMPLE_PLAN = [
    # ---------- A. 解锁 prune_rate__backbone (6 个,纯剪枝, FP32) ----------
    dict(id="A1", goal="backbone 单独剪 0.25",
         prune={"backbone": 0.25}, q_bits=None),
    dict(id="A2", goal="backbone 单独剪 0.375",
         prune={"backbone": 0.375}, q_bits=None),
    dict(id="A3", goal="backbone 单独剪 0.5",
         prune={"backbone": 0.5}, q_bits=None),
    dict(id="A4", goal="backbone 0.25 + encoder 0.25 (同步剪)",
         prune={"backbone": 0.25, "encoder": 0.25}, q_bits=None),
    dict(id="A5", goal="backbone 0.375 + encoder 0.375",
         prune={"backbone": 0.375, "encoder": 0.375}, q_bits=None),
    dict(id="A6", goal="backbone 0.5 + encoder 0.5 + decoder 0.5",
         prune={"backbone": 0.5, "encoder": 0.5, "decoder": 0.5}, q_bits=None),

    # ---------- B. 解锁 prune_rate__heads (3 个) ----------
    dict(id="B1", goal="heads 单独剪 0.25",
         prune={"heads": 0.25}, q_bits=None),
    dict(id="B2", goal="heads 单独剪 0.375",
         prune={"heads": 0.375}, q_bits=None),
    dict(id="B3", goal="heads 0.25 + encoder 0.25",
         prune={"heads": 0.25, "encoder": 0.25}, q_bits=None),

    # ---------- C. 跨模块剪枝差异化 (让 prune_rate__std > 0.1) ----------
    dict(id="C1", goal="encoder 0.5 / decoder 0.25 / backbone 0 (差异 0.25)",
         prune={"encoder": 0.5, "decoder": 0.25}, q_bits=None),
    dict(id="C2", goal="encoder 0.625 / decoder 0.25 (差异 0.375)",
         prune={"encoder": 0.625, "decoder": 0.25}, q_bits=None),
    dict(id="C3", goal="encoder 0.5 / decoder 0.5 / heads 0.25",
         prune={"encoder": 0.5, "decoder": 0.5, "heads": 0.25}, q_bits=None),
    dict(id="C4", goal="encoder 0.5 / decoder 0.5 / backbone 0.25 / heads 0.25",
         prune={"encoder": 0.5, "decoder": 0.5, "backbone": 0.25, "heads": 0.25}, q_bits=None),

    # ---------- D. 模块差异化量化 (无剪枝, 突破 q_bits__* 同步) ----------
    dict(id="D1", goal="backbone INT8 + 其他 FP16 (per-tensor)",
         prune=None,
         q_bits={"backbone": "INT8", "encoder": "FP16", "decoder": "FP16",
                 "heads": "FP16", "v2x_comm": "FP16"},
         q_gran="per-tensor", q_obj="W+A"),
    dict(id="D2", goal="backbone FP16 + encoder INT8 + decoder INT8 + heads FP16",
         prune=None,
         q_bits={"backbone": "FP16", "encoder": "INT8", "decoder": "INT8",
                 "heads": "FP16", "v2x_comm": "FP16"},
         q_gran="per-tensor", q_obj="W+A"),
    dict(id="D3", goal="backbone INT8 + encoder INT8 + decoder FP16 + heads INT8",
         prune=None,
         q_bits={"backbone": "INT8", "encoder": "INT8", "decoder": "FP16",
                 "heads": "INT8", "v2x_comm": "FP16"},
         q_gran="per-tensor", q_obj="W+A"),
    dict(id="D4", goal="全 INT8 + per-channel (GPU 上对比 per-tensor)",
         prune=None,
         q_bits={m: "INT8" for m in UNIV2X_MODULES},
         q_gran="per-channel", q_obj="W+A"),
    dict(id="D5", goal="全 INT8 + W-only (对比 W+A 的精度损失)",
         prune=None,
         q_bits={m: "INT8" for m in UNIV2X_MODULES},
         q_gran="per-tensor", q_obj="W-only"),

    # ---------- E. 剪枝 × 量化联合 (12 个,核心 motivation) ----------
    # E1-E3: encoder 单轴扫 × INT8 全图
    dict(id="E1", goal="encoder 0.25 + 全 INT8 (per-tensor, W+A)",
         prune={"encoder": 0.25}, q_bits={m: "INT8" for m in UNIV2X_MODULES},
         q_gran="per-tensor", q_obj="W+A"),
    dict(id="E2", goal="encoder 0.5 + 全 INT8",
         prune={"encoder": 0.5}, q_bits={m: "INT8" for m in UNIV2X_MODULES},
         q_gran="per-tensor", q_obj="W+A"),
    dict(id="E3", goal="encoder 0.625 + 全 INT8 (探索极限)",
         prune={"encoder": 0.625}, q_bits={m: "INT8" for m in UNIV2X_MODULES},
         q_gran="per-tensor", q_obj="W+A"),
    # E4-E6: encoder × FP16 (验证量化压力小时剪枝行为)
    dict(id="E4", goal="encoder 0.25 + 全 FP16",
         prune={"encoder": 0.25}, q_bits={m: "FP16" for m in UNIV2X_MODULES},
         q_gran="per-tensor", q_obj="W+A"),
    dict(id="E5", goal="encoder 0.5 + 全 FP16",
         prune={"encoder": 0.5}, q_bits={m: "FP16" for m in UNIV2X_MODULES},
         q_gran="per-tensor", q_obj="W+A"),
    # E6-E8: 多模块剪枝 + 全 INT8
    dict(id="E6", goal="encoder 0.25 + decoder 0.5 + 全 INT8",
         prune={"encoder": 0.25, "decoder": 0.5},
         q_bits={m: "INT8" for m in UNIV2X_MODULES},
         q_gran="per-tensor", q_obj="W+A"),
    dict(id="E7", goal="encoder 0.5 + decoder 0.5 + 全 INT8",
         prune={"encoder": 0.5, "decoder": 0.5},
         q_bits={m: "INT8" for m in UNIV2X_MODULES},
         q_gran="per-tensor", q_obj="W+A"),
    dict(id="E8", goal="encoder 0.5 + decoder 0.5 + 全 INT8 + per-channel",
         prune={"encoder": 0.5, "decoder": 0.5},
         q_bits={m: "INT8" for m in UNIV2X_MODULES},
         q_gran="per-channel", q_obj="W+A"),
    # E9-E10: backbone × encoder × INT8
    dict(id="E9", goal="backbone 0.25 + encoder 0.25 + 全 INT8",
         prune={"backbone": 0.25, "encoder": 0.25},
         q_bits={m: "INT8" for m in UNIV2X_MODULES},
         q_gran="per-tensor", q_obj="W+A"),
    dict(id="E10", goal="backbone 0.25 + encoder 0.5 + 全 INT8 (混合压力)",
         prune={"backbone": 0.25, "encoder": 0.5},
         q_bits={m: "INT8" for m in UNIV2X_MODULES},
         q_gran="per-tensor", q_obj="W+A"),
    # E11-E12: 模块差异化量化 + 联合剪枝
    dict(id="E11", goal="encoder 0.5 + decoder 0.5 + backbone FP16 其他 INT8",
         prune={"encoder": 0.5, "decoder": 0.5},
         q_bits={"backbone": "FP16", "encoder": "INT8", "decoder": "INT8",
                 "heads": "INT8", "v2x_comm": "INT8"},
         q_gran="per-tensor", q_obj="W+A"),
    dict(id="E12", goal="encoder 0.625 + decoder 0.5 + 全 INT8 + W-only",
         prune={"encoder": 0.625, "decoder": 0.5},
         q_bits={m: "INT8" for m in UNIV2X_MODULES},
         q_gran="per-tensor", q_obj="W-only"),
]


# ============================================================
# 把 SAMPLE_PLAN 项实例化为 Config
# ============================================================

def build_config(spec: dict, criterion: str = "L1") -> Config:
    """把 spec dict 转为 Config (默认值: prune=channel/L1, quant=FP32 baseline)."""
    cfg = Config.fp32_baseline()

    # 剪枝
    prune_spec = spec.get("prune") or {}
    if prune_spec:
        new_rates = dict(cfg.prune_rate)
        new_crit = dict(cfg.prune_criterion)
        for m, r in prune_spec.items():
            new_rates[m] = float(r)
            new_crit[m] = criterion
        cfg = cfg.with_field(
            prune_rate=new_rates,
            prune_object="channel",
            prune_criterion=new_crit,
        )

    # 量化
    if spec.get("q_bits") is not None:
        new_bits = dict(cfg.q_bits)
        new_gran = dict(cfg.q_granularity)
        new_obj = dict(cfg.q_object)
        for m, b in spec["q_bits"].items():
            new_bits[m] = b
            if b == "FP32":
                new_gran[m] = "none"
                new_obj[m] = "none"
            else:
                new_gran[m] = spec.get("q_gran", "per-tensor")
                new_obj[m] = spec.get("q_obj", "W+A")
        cfg = cfg.with_field(
            q_bits=new_bits,
            q_granularity=new_gran,
            q_object=new_obj,
        )

    cfg = cfg.with_field(config_id=f"active_{spec['id']}", source="active_sample")
    return cfg


# ============================================================
# 转回 1.1/1.2 现有 JSON 格式 (4090 上跑 PTQ 不需要新代码)
# ============================================================

def to_prune_json(cfg: Config, goal: str) -> dict:
    """1.2 prune_configs/*.json schema."""
    # ffn_mid_ratio 是"保留比例", 与 prune_rate 互补
    encoder_ratio = round(1 - cfg.prune_rate.get("encoder", 0.0), 4)
    decoder_ratio = round(1 - cfg.prune_rate.get("decoder", 0.0), 4)
    head_ratio    = round(1 - cfg.prune_rate.get("heads", 0.0), 4)
    backbone_rate = cfg.prune_rate.get("backbone", 0.0)

    return {
        "version": "1.2-active",
        "_description": f"Stage 2.5 主动采样: {goal}",
        "_purpose": "解锁 sanity 模型未见过的特征维度",
        "locked": {
            "importance_criterion": "l1_norm",
            "pruning_granularity": "local",
            "iterative_steps": 5,
            "round_to": 8,
        },
        "encoder": {
            "ffn_mid_ratio": encoder_ratio,
            "attn_proj_ratio": 0.0,
            "head_pruning_ratio": 0.0,
        },
        "decoder": {
            "ffn_mid_ratio": decoder_ratio,
            "attn_proj_ratio": 0.0,
            "head_pruning_ratio": 0.0,
            "num_layers": 6,
        },
        "heads": {
            "head_mid_ratio": head_ratio,
        },
        "backbone": {
            "channel_pruning_ratio": float(backbone_rate),
            "_note": "1.2 没实现 backbone 剪枝, Stage 2.5 需要新实现或借助 Torch-Pruning",
        },
        "finetune": {
            "epochs": 0,
            "_note": "零微调, 与 1.2 PhaseB 一致, 直接评估剪枝后精度损失",
        },
        "constraints": {
            "skip_layers": ["sampling_offsets", "attention_weights"],
            "min_channels": 64,
            "channel_alignment": 32,
        },
    }


def to_quant_json(cfg: Config, goal: str) -> dict:
    """1.1 quant_configs/*.json schema."""
    # 主导位宽 = 出现最多的位宽
    bits_count: dict = {}
    for b in cfg.q_bits.values():
        bits_count[b] = bits_count.get(b, 0) + 1
    dominant = max(bits_count, key=bits_count.get) if bits_count else "FP32"

    bits_to_num = {"FP32": 32, "FP16": 16, "INT8": 8}
    gran_map = {"per-tensor": "per_tensor", "per-channel": "per_channel", "none": "per_tensor"}
    obj_map = {"W+A": "W+A", "W-only": "W", "W": "W", "none": "none"}

    enc = cfg.q_bits.get("encoder", "FP32")
    return {
        "version": "1.1-active",
        "_description": f"Stage 2.5 主动采样: {goal}",
        "global": {
            "symmetric": True,
            "scale_method": "minmax",
            "default_w_bits": bits_to_num.get(dominant, 32),
            "default_a_bits": bits_to_num.get(dominant, 32),
            "default_w_granularity": gran_map.get(cfg.q_granularity.get("encoder", "none"), "per_tensor"),
            "default_a_granularity": gran_map.get(cfg.q_granularity.get("encoder", "none"), "per_tensor"),
            "default_quant_target": obj_map.get(cfg.q_object.get("encoder", "none"), "none"),
        },
        "modules": {
            m: {
                "bits": bits_to_num.get(cfg.q_bits.get(m, "FP32"), 32),
                "granularity": gran_map.get(cfg.q_granularity.get(m, "none"), "per_tensor"),
                "target": obj_map.get(cfg.q_object.get(m, "none"), "none"),
            }
            for m in UNIV2X_MODULES
        },
        "layers": {},
        "v2x_comm": {
            "agent_query_precision": "int8" if cfg.q_bits.get("v2x_comm") == "INT8" else "fp16",
            "lane_query_precision": "int8" if cfg.q_bits.get("v2x_comm") == "INT8" else "fp16",
            "bev_scatter_precision": "fp16",
        },
    }


# ============================================================
# 主流程
# ============================================================

def main() -> None:
    print("=" * 60)
    print("Stage 2.5 — 主动采样规划生成")
    print("=" * 60)
    print(f"\n目标: 30 个差异化配置, 解锁 sanity 模型未见过的特征维度\n")

    hw = HardwareCapability.from_yaml(ROOT / "configs" / "hardware" / "rtx4090.yaml")

    rows = []
    physical_illegal: list = []
    empirical_violations: list = []

    for i, spec in enumerate(SAMPLE_PLAN, 1):
        cfg = build_config(spec)
        # 双向传播 (DLA 不参与, 但保留以防未来扩展)
        cfg = propagate_all(cfg, hw)
        # 只硬过滤 physical (build 失败会真停);empirical 是"探索性违反"
        physical_ok, physical_reason = is_legal(cfg, hw, kinds=("physical",))
        emp_vios = list_violations(cfg, hw, kinds=("empirical",))
        soft_vios = list_violations(cfg, hw, kinds=("soft",))

        if not physical_ok:
            physical_illegal.append((spec["id"], physical_reason))
        for kind, name, reason in emp_vios:
            empirical_violations.append((spec["id"], name, reason))

        rows.append({
            "id": spec["id"],
            "goal": spec["goal"],
            "physical_ok": physical_ok,
            "physical_violation": physical_reason or "",
            "empirical_violations": "; ".join(f"{v[1]}: {v[2]}" for v in emp_vios),
            "is_exploratory": len(emp_vios) > 0,        # ← 标记是否触发经验约束
            "soft_violations": "; ".join(v[1] for v in soft_vios),
            "prune_object": cfg.prune_object,
            "prune_rate__backbone": cfg.prune_rate.get("backbone", 0),
            "prune_rate__encoder": cfg.prune_rate.get("encoder", 0),
            "prune_rate__decoder": cfg.prune_rate.get("decoder", 0),
            "prune_rate__heads": cfg.prune_rate.get("heads", 0),
            "prune_rate__v2x_comm": cfg.prune_rate.get("v2x_comm", 0),
            "q_bits__backbone": cfg.q_bits.get("backbone"),
            "q_bits__encoder": cfg.q_bits.get("encoder"),
            "q_bits__decoder": cfg.q_bits.get("decoder"),
            "q_bits__heads": cfg.q_bits.get("heads"),
            "q_bits__v2x_comm": cfg.q_bits.get("v2x_comm"),
            "q_granularity__encoder": cfg.q_granularity.get("encoder"),
            "q_object__encoder": cfg.q_object.get("encoder"),
            "d_routing__backbone": cfg.d_routing.get("backbone"),
        })

        # 写 prune / quant JSON (兼容 1.1/1.2 schema)
        out_prune = ROOT / "prune_configs" / f"active_{spec['id']}.json"
        out_quant = ROOT / "quant_configs" / f"active_{spec['id']}.json"
        with open(out_prune, "w", encoding="utf-8") as f:
            json.dump(to_prune_json(cfg, spec["goal"]), f, indent=2, ensure_ascii=False)
        with open(out_quant, "w", encoding="utf-8") as f:
            json.dump(to_quant_json(cfg, spec["goal"]), f, indent=2, ensure_ascii=False)

    # 汇总 CSV
    df = pd.DataFrame(rows)
    out_csv = ROOT / "data" / "phase2" / "active_samples_plan.csv"
    df.to_csv(out_csv, index=False)

    # ---------- 报告输出 ----------
    print(f"\n[plan] 30 个配置已具体化")
    print(f"  物理硬约束通过 (build 一定成功): {df['physical_ok'].sum()}/{len(df)}")
    if physical_illegal:
        print(f"  ⚠️  物理约束违反 (必须修): {len(physical_illegal)}")
        for cid, r in physical_illegal:
            print(f"    {cid}: {r}")
    n_exploratory = df['is_exploratory'].sum()
    n_normal = len(df) - n_exploratory
    print(f"  常规配置 (经验约束也通过): {n_normal}")
    print(f"  探索性配置 (触发经验约束, 用于 Phase 1B.6 拐点验证): {n_exploratory}")
    soft_count = (df["soft_violations"] != "").sum()
    print(f"  软约束触发 (作为 negative 样本): {soft_count}/{len(df)}")

    # 按类别分布
    print("\n[breakdown]")
    df["category"] = df["id"].str[0]
    for cat, sub in df.groupby("category"):
        avg_prune = sub[["prune_rate__backbone","prune_rate__encoder","prune_rate__decoder","prune_rate__heads","prune_rate__v2x_comm"]].mean(axis=1).mean()
        n_int8 = (sub.filter(like="q_bits__") == "INT8").any(axis=1).sum()
        print(f"  {cat} ({len(sub)} 个): avg_prune={avg_prune:.3f}, 含 INT8={n_int8}")

    print(f"\n[outputs]")
    print(f"  {out_csv}")
    print(f"  prune_configs/active_*.json   ({len(SAMPLE_PLAN)} 个)")
    print(f"  quant_configs/active_*.json   ({len(SAMPLE_PLAN)} 个)")


if __name__ == "__main__":
    main()
