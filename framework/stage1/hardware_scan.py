"""Stage1 子流程 A — 硬件 capability 扫描 (归一化访问层).

设计文档: multi_agent/methods/design/stage1_graph_scan_partition_v1.md §1

职责:
  - 加载 + (尽量) 校验 hw_capability YAML (复用 framework/capability_schema.py)。
  - 提供归一化访问器, 吸收 schema v1.0 与旧 capability_schema 之间的字段名差异
    (如 alignment.int8_dense_channel vs int8_channel; quant_constraints 在根级 extra)。
  - 把硬件约束翻译成 graph_scan 需要的形式: round_to / legal_bits / op_whitelist / mem 预算。

不做: 探测真硬件 (probe) v1 留半人工 —— 直接读已写好的 YAML (如 configs/hardware/rtx4090.yaml)。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import yaml

_FW = Path(__file__).resolve().parents[1]  # framework/
if str(_FW.parent) not in sys.path:
    sys.path.insert(0, str(_FW.parent))

# 我们只搜 INT8 / FP16 两档 (设计裁剪: 不碰 FP8/INT4 搜索)
_SEARCH_BITS = ("INT8", "FP16")


class HwCapability:
    """归一化的硬件 capability 视图 (读 raw dict, 不依赖 pydantic 字段名)。"""

    def __init__(self, raw: dict, path: str | Path, validated: bool, validate_err: str = ""):
        self.raw = raw
        self.path = str(path)
        self.validated = validated
        self.validate_err = validate_err

    # ---- 加载 ----
    @classmethod
    def from_yaml(cls, path: str | Path) -> "HwCapability":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        validated, err = True, ""
        try:
            from framework.capability_schema import HardwareCapability
            HardwareCapability.from_yaml(path)
        except Exception as e:  # noqa: BLE001 — 校验失败不致命, 记录即可
            validated, err = False, f"{type(e).__name__}: {e}"
        return cls(raw, path, validated, err)

    # ---- 基本信息 ----
    @property
    def name(self) -> str:
        return (self.raw.get("basic", {}) or {}).get("name") or self.raw.get("name") or "unknown"

    @property
    def arch(self) -> str:
        a = self.raw.get("arch", {})
        if isinstance(a, dict):
            return (f"{a.get('family','')} {a.get('sm','')}").strip() or "unknown"
        return str(a) if a else "unknown"

    # ---- IP / DLA ----
    @property
    def has_dla(self) -> bool:
        dla = (self.raw.get("ips", {}) or {}).get("dla", {}) or {}
        if dla.get("enabled") is False:
            return False
        return bool(dla.get("enabled")) or dla.get("count", 0) > 0

    @property
    def dla_op_whitelist(self) -> Optional[list[str]]:
        if not self.has_dla:
            return None
        dla = (self.raw.get("ips", {}) or {}).get("dla", {}) or {}
        return dla.get("op_whitelist")

    @property
    def gpu_precisions(self) -> list[str]:
        gpu = (self.raw.get("ips", {}) or {}).get("gpu", {}) or {}
        return gpu.get("precisions", []) or []

    # ---- 对齐 (round_to) ----
    def _align(self) -> dict:
        return self.raw.get("alignment", {}) or {}

    @property
    def int8_align(self) -> int:
        a = self._align()
        return int(a.get("int8_dense_channel") or a.get("int8_channel") or 32)

    @property
    def fp16_align(self) -> int:
        a = self._align()
        return int(a.get("fp16_dense_channel") or a.get("fp16_channel") or 8)

    @property
    def int8_pack_factor(self) -> int:
        """dp4a/IMMA int8 沿 per-group 输入轴打包的 int8 个数 (reduction packing).

        分组卷积 int8 可建 ⟺ in_per_g % pack_factor == 0 ⟺
        width % (pack_factor×groups) == 0。有效 int8 对齐 =
        lcm(int8_align, pack_factor×groups)。实测定标 q_int8_dp4a_pairs.csv:
        dp4a=4 (4-int8 dot)。默认 4 (缺失即退化为 dp4a, 保守且最常见)。
        """
        return int(self._align().get("int8_pack_factor") or 4)

    @property
    def alignment_enforcement(self) -> str:
        return self._align().get("alignment_enforcement", "hard")

    def round_to_for(self, bits: str) -> int:
        """按量化档返回通道对齐 round_to. INT8→int8_align, 其余→fp16_align."""
        return self.int8_align if bits.upper() == "INT8" else self.fp16_align

    # ---- 量化约束 → B2 合法集合 ----
    def _qc(self) -> dict:
        return self.raw.get("quant_constraints", {}) or {}

    @property
    def legal_bits(self) -> list[str]:
        """我们只搜 {INT8, FP16}; 与硬件实际支持位宽取交集。"""
        qc = self._qc()
        bw = set(qc.get("bit_widths_w", [8, 16, 32]) or [8, 16, 32])
        out = []
        if 8 in bw:
            out.append("INT8")
        if 16 in bw:
            out.append("FP16")
        return out or list(_SEARCH_BITS)

    @property
    def legal_granularity_w(self) -> list[str]:
        return self._qc().get("granularity_w", ["per_tensor", "per_channel"]) or ["per_tensor"]

    @property
    def per_channel_act(self) -> bool:
        return bool(self._qc().get("per_channel_activation_supported", False))

    @property
    def symmetric_only(self) -> bool:
        return bool(self._qc().get("symmetric_only", True))

    # ---- 显存 (SLA 可行性闸门, B→A 耦合用) ----
    @property
    def mem_capacity_gb(self) -> Optional[float]:
        mem = self.raw.get("memory", {}) or {}
        v = mem.get("available_for_inference_gb") or mem.get("capacity_gb")
        return float(v) if v is not None else None

    # ---- 摘要 (写进 manifest) ----
    def summary(self) -> dict:
        return {
            "path": self.path,
            "name": self.name,
            "arch": self.arch,
            "schema_validated": self.validated,
            "schema_validate_err": self.validate_err or None,
            "has_dla": self.has_dla,
            "dla_op_whitelist": self.dla_op_whitelist,
            "int8_align": self.int8_align,
            "fp16_align": self.fp16_align,
            "alignment_enforcement": self.alignment_enforcement,
            "legal_bits": self.legal_bits,
            "legal_granularity_w": self.legal_granularity_w,
            "per_channel_act": self.per_channel_act,
            "symmetric_only": self.symmetric_only,
            "mem_capacity_gb": self.mem_capacity_gb,
        }


__all__ = ["HwCapability"]
