"""硬件 capability 描述 schema (Phase 1A.1)

对应 v1.5 §1 步骤 1 / Phase1_2 实施计划 §2.3。

设计原则:
- Pydantic v2 严格模式 (extra='forbid'),拼写错误立刻报错
- IP 列表是核心字段,不同硬件 IP 完全不同 (4090: 仅 GPU; Orin AGX: GPU+DLA; Orin Nano: 仅 GPU)
- 字段命名与 NVIDIA 文档对齐,便于人工查表
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

Precision = Literal[
    "FP32", "TF32", "BF16", "FP16", "INT8", "INT4",
    "FP8", "FP8_E4M3", "FP8_E5M2",
]


class IPCapability(BaseModel):
    """单个硬件 IP (GPU / DLA / NPU) 的能力描述."""

    model_config = ConfigDict(extra="allow")  # 允许 v1.0 schema 新字段（cores/streams_max/...）

    precisions: list[Precision] = Field(
        default_factory=list, description="该 IP 支持的精度集合 (CPU IP 可为空)"
    )
    tensor_core_gen: Optional[int] = Field(
        None, description="Tensor Core 代际 (3=Ampere, 4=Ada, 5=Hopper); 非 GPU 填 None"
    )
    sparse_tc: bool = Field(False, description="是否支持 2:4 Sparse Tensor Core")
    op_whitelist: Optional[list[str]] = Field(
        None, description="算子白名单 (DLA 必填; GPU 通常 None 表示通用)"
    )
    granularity: list[Literal["per-tensor", "per-channel"]] = Field(
        default_factory=lambda: ["per-tensor", "per-channel"],
        description="支持的量化粒度 (DLA 一般只有 per-tensor)",
    )


class Alignment(BaseModel):
    """对齐要求 (Tensor Core / DLA)."""

    model_config = ConfigDict(extra="allow")  # 允许 v1.0 schema 新字段

    int8_channel: int = Field(32, description="INT8 Tensor Core 通道对齐")
    fp16_channel: int = Field(16, description="FP16 Tensor Core 通道对齐")
    dla_layout: Optional[str] = Field(None, description="DLA 输入张量布局 (如 HWC4)")
    # ★ N2v2 实证驱动机制 (2026-05-06)
    # 让经验对齐约束的严重程度由 capability YAML 字段动态决定:
    #   "hard" — 默认; 违反就过滤掉 (旧 HW-NAS 做法)
    #   "soft" — N2v2 在 Orin sm87 上实测 implicit padding 抵消对齐效应,
    #            仅作为 penalty 项保留偏好但不过滤
    #   "auto" — 不加规则,让数据驱动学习 (Phase 2 精度/延迟预测器接管)
    alignment_enforcement: Literal["hard", "soft", "auto"] = Field(
        "hard",
        description="对齐约束的强度,由 capability YAML 写入的实证字段决定",
    )


class Features(BaseModel):
    """硬件/驱动级特性开关."""

    model_config = ConfigDict(extra="allow")

    mig: bool = False
    mps: bool = False
    sparse_tc: bool = False
    tensor_core: bool = False
    cuda_graph: bool = True
    dla_count: int = 0


class Toolchain(BaseModel):
    """工具链版本 (锁定后才能保证可复现)."""

    model_config = ConfigDict(extra="allow")

    trt_version: Optional[str] = None       # 旧字段; v1.0 schema 用 framework_version
    cuda_version: Optional[str] = None
    jetpack_version: Optional[str] = None
    driver_version: Optional[str] = None
    # v1.0 schema 字段 (新增)
    framework: Optional[str] = None
    framework_version: Optional[str] = None
    cuda: Optional[str] = None              # 旧字段 cuda_version 的 v1.0 别名
    cudnn: Optional[str] = None
    jetpack: Optional[str] = None           # 旧字段 jetpack_version 的 v1.0 别名
    l4t: Optional[str] = None
    python: Optional[str] = None
    driver: Optional[str] = None


class HardwareCapability(BaseModel):
    """完整硬件 capability 描述 (一份 YAML 对应一颗芯片)."""

    model_config = ConfigDict(extra="allow")

    name: str = Field(..., description="人类可读的硬件名,如 'Jetson AGX Orin 64GB'")
    arch: str = Field(..., description="架构标识,如 'Ampere sm87' / 'Ada sm89'")
    ips: dict[str, IPCapability] = Field(
        ..., description="可用 IP 字典,key 是 'gpu'/'dla'/'npu' 等"
    )
    alignment: Alignment = Field(default_factory=Alignment)
    features: Features = Field(default_factory=Features)
    toolchain: Toolchain = Field(default_factory=lambda: Toolchain())

    @field_validator("ips")
    @classmethod
    def _at_least_one_ip(cls, v: dict[str, IPCapability]) -> dict[str, IPCapability]:
        if not v:
            raise ValueError("至少要描述一个 IP")
        return v

    @property
    def has_dla(self) -> bool:
        """是否有可用 DLA. 兼容旧 (features.dla_count) 和新 (ips.dla.count) schema."""
        if "dla" not in self.ips:
            return False
        # 旧 schema: features.dla_count > 0
        if self.features.dla_count > 0:
            return True
        # 新 schema v1.0: ips.dla 节点下的 count + enabled (extra=allow 吸收)
        dla_extra = self.ips["dla"].model_extra or {}
        if dla_extra.get("enabled") is False:
            return False
        return dla_extra.get("count", 0) > 0 or len(self.ips["dla"].precisions) > 0

    @property
    def supported_precisions(self) -> set[str]:
        """所有 IP 支持精度的并集."""
        result: set[str] = set()
        for ip in self.ips.values():
            result.update(ip.precisions)
        return result

    @classmethod
    def from_yaml(cls, path: str | Path) -> "HardwareCapability":
        """从 YAML 文件加载,带 schema v1.0 兼容层.

        v1.0 schema 把 name/arch/cost 等放在 basic.* 与 arch.* 节点下,
        本层把它们 flatten 到根级,以保持向后兼容.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.model_validate(_flatten_schema_v1(data))


def _flatten_schema_v1(d: dict) -> dict:
    """schema.yaml v1.0 → 旧 capability_schema 的字段映射.

    v1.0 (新):                          → 旧 capability_schema:
      basic.name                          name
      basic.* (其余)                      (根级 extra=allow 吸收)
      arch.family + arch.sm                arch (拼成 'family smXX' 字符串)
      arch.* (其余)                       (extra=allow 吸收)
      ips.gpu.cores / streams_max ...      (IPCapability extra=allow)
      ips.cpu (没 precisions 字段)         (precisions 默认 [])
      ips.dla.enabled=False                如不启用直接删
      alignment.alignment_enforcement      Alignment 字段直接接受
    """
    out = dict(d)  # shallow copy
    basic = out.pop("basic", {}) or {}
    if "name" in basic and "name" not in out:
        out["name"] = basic["name"]
    # 把 basic 其它字段保留 (extra=allow 吸收)
    out.update({k: v for k, v in basic.items() if k != "name"})

    arch_dict = out.pop("arch", None)
    if isinstance(arch_dict, dict):
        family = arch_dict.get("family", "")
        sm = arch_dict.get("sm", "")
        out["arch"] = (family + " " + sm).strip() or family or sm or "unknown"
        # arch 其它字段保留
        for k, v in arch_dict.items():
            if k not in ("family", "sm"):
                out["arch_" + k] = v

    # ips: 移除 enabled=False 的 IP；移除 dla 的 op_blacklist 等额外字段(extra=allow 已吸收)
    ips = out.get("ips", {})
    if isinstance(ips, dict):
        ips_clean = {}
        for k, v in ips.items():
            if isinstance(v, dict) and v.get("enabled") is False:
                continue  # 不启用的 IP 跳过
            ips_clean[k] = v
        out["ips"] = ips_clean

    return out

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                self.model_dump(), f, allow_unicode=True, sort_keys=False
            )


__all__ = [
    "Precision",
    "IPCapability",
    "Alignment",
    "Features",
    "Toolchain",
    "HardwareCapability",
]
