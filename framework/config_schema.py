"""协同加速配置 schema (Phase 1A.3)

对应 v1.5 §0.2 真实搜索空间 / Phase1_2 §1.2 模块接口契约.

设计原则:
- Config 是搜索器、预测器、评估器之间统一传递的数据对象
- 字段按 module 粒度组织 (UniV2X 模块: backbone/encoder/decoder/heads/v2x_comm)
- 兼容已有 quant_configs/*.json (1.1) 和 prune_configs/*.json (1.2) 的字段命名
- 不可变 (frozen=True) — 防止搜索过程中的隐式修改 (v1.5 §0 编码规范)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict, replace
from pathlib import Path
from typing import Literal, Optional

import yaml

# UniV2X 标准模块名 (来自 1.1 / 1.2 实验报告)
UNIV2X_MODULES = ("backbone", "encoder", "decoder", "heads", "v2x_comm")

# 取值范围 (与 v1.5 §0.2 对齐)
PRUNE_OBJECT_VALUES = ("channel", "head", "2:4", "element", "none")
PRUNE_CRITERION_VALUES = ("L1", "Taylor", "FPGM", "Wanda", "none")
Q_BITS_VALUES = ("INT8", "FP16", "FP32")
Q_GRAN_VALUES = ("per-tensor", "per-channel", "none")
Q_OBJ_VALUES = ("W-only", "W+A", "none")
ROUTING_VALUES = ("GPU", "DLA0", "DLA1", "CPU")


@dataclass(frozen=True)
class Config:
    """单个候选配置 (跨 B1/B2/D 联合搜索空间).

    设计为字典结构而非位置参数,因为模块数会变 (UniV2X 当前 5 个,以后可能更多).
    """

    # B1 剪枝
    prune_rate: dict[str, float] = field(default_factory=dict)
    # 全局粒度 (channel/head/2:4 三选一,因为不会混用)
    prune_object: str = "channel"
    prune_criterion: dict[str, str] = field(default_factory=dict)

    # B2 量化 (按模块)
    q_bits: dict[str, str] = field(default_factory=dict)
    q_granularity: dict[str, str] = field(default_factory=dict)
    q_object: dict[str, str] = field(default_factory=dict)

    # D 部署路由 (按模块)
    d_routing: dict[str, str] = field(default_factory=dict)

    # 元数据 (可选,用于追踪)
    config_id: Optional[str] = None
    source: Optional[str] = None  # '1.1_quant' / '1.2_prune' / 'phase1a' / 'active_sample'

    # ---------- 构造便利 ----------

    @classmethod
    def fp32_baseline(cls, modules: tuple[str, ...] = UNIV2X_MODULES) -> "Config":
        """所有 module 都是 FP32 + 不剪枝 + GPU only,作为对照基线."""
        return cls(
            prune_rate={m: 0.0 for m in modules},
            prune_object="none",
            prune_criterion={m: "none" for m in modules},
            q_bits={m: "FP32" for m in modules},
            q_granularity={m: "none" for m in modules},
            q_object={m: "none" for m in modules},
            d_routing={m: "GPU" for m in modules},
            source="baseline",
        )

    @classmethod
    def from_quant_json(cls, path: str | Path,
                        modules: tuple[str, ...] = UNIV2X_MODULES) -> "Config":
        """从 1.1 量化 JSON (quant_configs/*.json) 加载.

        1.1 格式:
            global: { default_w_bits, default_w_granularity, default_quant_target, ... }
            layers: { <layer>: { ... 逐层覆盖 ... } }
            v2x_comm: { agent_query_precision, ... }
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        g = data.get("global", {})
        bits = "INT8" if g.get("default_w_bits", 8) == 8 else "FP16"
        gran_map = {"per_tensor": "per-tensor", "per_channel": "per-channel"}
        gran = gran_map.get(g.get("default_w_granularity", "per_tensor"), "per-tensor")
        obj_map = {"W+A": "W+A", "W": "W-only", "W-only": "W-only", "none": "none"}
        qobj = obj_map.get(g.get("default_quant_target", "W+A"), "W+A")

        return cls(
            prune_rate={m: 0.0 for m in modules},
            prune_object="none",
            prune_criterion={m: "none" for m in modules},
            q_bits={m: bits for m in modules},
            q_granularity={m: gran for m in modules},
            q_object={m: qobj for m in modules},
            d_routing={m: "GPU" for m in modules},
            config_id=Path(path).stem,
            source="1.1_quant",
        )

    @classmethod
    def from_prune_json(cls, path: str | Path,
                        modules: tuple[str, ...] = UNIV2X_MODULES) -> "Config":
        """从 1.2 剪枝 JSON (prune_configs/*.json) 加载.

        1.2 格式:
            locked: { importance_criterion, pruning_granularity, ... }
            encoder: { ffn_mid_ratio, attn_proj_ratio, head_pruning_ratio }
            decoder: { ffn_mid_ratio, ... }
            heads: { head_mid_ratio }
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        crit_map = {
            "l1_norm": "L1", "l1": "L1",
            "taylor": "Taylor",
            "fpgm": "FPGM",
            "wanda": "Wanda",
        }
        crit = crit_map.get(
            data.get("locked", {}).get("importance_criterion", "l1_norm"), "L1"
        )

        # ffn_mid_ratio = 0.7 表示保留 70%, 即剪枝 30%
        prune_rate = {}
        for m in modules:
            mod_cfg = data.get(m, {})
            if not mod_cfg:
                prune_rate[m] = 0.0
                continue
            kept = mod_cfg.get("ffn_mid_ratio")
            if kept is None:
                kept = mod_cfg.get("head_mid_ratio", 1.0)
            try:
                kept = float(kept)
                prune_rate[m] = max(0.0, 1.0 - kept)
            except (TypeError, ValueError):
                prune_rate[m] = 0.0

        return cls(
            prune_rate=prune_rate,
            prune_object="channel",
            prune_criterion={m: crit for m in modules},
            q_bits={m: "FP32" for m in modules},
            q_granularity={m: "none" for m in modules},
            q_object={m: "none" for m in modules},
            d_routing={m: "GPU" for m in modules},
            config_id=Path(path).stem,
            source="1.2_prune",
        )

    # ---------- 序列化 ----------

    def to_dict(self) -> dict:
        return asdict(self)

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, allow_unicode=True, sort_keys=False)

    def to_json(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        return cls(**d)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f))

    # ---------- 不可变更新 ----------

    def with_field(self, **kwargs) -> "Config":
        """返回更新指定字段后的新 Config (immutable update)."""
        return replace(self, **kwargs)

    # ---------- 派生量 (给特征工程用) ----------

    def avg_prune_rate(self) -> float:
        if not self.prune_rate:
            return 0.0
        return sum(self.prune_rate.values()) / len(self.prune_rate)

    def has_dla(self) -> bool:
        return any(r.startswith("DLA") for r in self.d_routing.values())

    def modules(self) -> list[str]:
        """返回 Config 涉及的所有 module 名."""
        keys = set(self.prune_rate) | set(self.q_bits) | set(self.d_routing)
        return sorted(keys)


__all__ = [
    "Config",
    "UNIV2X_MODULES",
    "PRUNE_OBJECT_VALUES",
    "PRUNE_CRITERION_VALUES",
    "Q_BITS_VALUES",
    "Q_GRAN_VALUES",
    "Q_OBJ_VALUES",
    "ROUTING_VALUES",
]
