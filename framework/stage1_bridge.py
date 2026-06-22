"""stage1_bridge.py — stage1 partition manifest → 搜索空间 SpaceSpec。

把 stage1 扫描产物 `framework/partitions/<model>_partition.yaml` 解析成搜索器
直接消费的搜索空间对象 (SpaceSpec)。这是 stage1 (网络-硬件协同刻画) 与 stage2
(协同搜索) 之间唯一的、自动的数据通道 —— 在它之前搜索器靠人工策展的 JSON 跑;
接通后, 导入任意网络即可自动建空间。

普适性三铁律 (设计文档 bridge_stage1_to_search_design_v1.md §3.1b 硬约束):
  1. **可建性 key 在对齐字段**: int8 dp4a 可建 ⟺ `w % int8_buildable_align == 0`,
     其中 int8_buildable_align = lcm(int8_align, pack_factor×groups) 已由 stage1
     算好。bridge 只读这个数, **不依赖整数组数 groups** → 天然跨架构。
  2. **每字段防御性默认**: 任一字段缺失即退化为"最宽松且合法", 用 `.get(k, DEFAULT)`,
     **绝不写 `if model == '...'`**。CoDriving 的 groups=1 / 无 DLA 行为从默认涌现。
  3. **硬件能力一次解析、按需消费**: HwCapability 解析全字段; 对齐/位宽/粒度立即可用,
     soft/auto enforcement、DLA 路由、IP 枚举先留接口骨架 (本阶段不消费)。

范围边界 (设计 §6): bridge 只接"空间 + 合法性约束"; 测量值 (latency LUT / AP) 是
另一条数据流, bridge 只消费其产物不生成。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

# 防御性默认 (字段缺失时退化为"最宽松且合法")。集中一处, 便于审计。
DEFAULTS = {
    "int8_align": 32,
    "fp16_align": 8,
    "int8_pack_factor": 4,          # dp4a 4-int8 dot (q_int8_dp4a_pairs.csv 定标)
    "alignment_enforcement": "hard",
    "legal_bits": ("INT8", "FP16"),
    "legal_granularity_w": ("per_tensor", "per_channel"),
    "per_channel_act": False,
    "symmetric_only": True,
    "round_to": 32,
    "max_rate": 0.0,
    "grouped_conv": False,
    "criterion_pool": ("L1",),
    "quantizable": True,
}


# ---------------------------------------------------------------------------
# 硬件能力 (manifest hw_capability 块 → 结构化访问; 含未消费字段骨架)
# ---------------------------------------------------------------------------
@dataclass
class HwCapability:
    name: str = "unknown"
    int8_align: int = DEFAULTS["int8_align"]
    fp16_align: int = DEFAULTS["fp16_align"]
    int8_pack_factor: int = DEFAULTS["int8_pack_factor"]
    alignment_enforcement: str = DEFAULTS["alignment_enforcement"]
    legal_bits: tuple[str, ...] = DEFAULTS["legal_bits"]
    legal_granularity_w: tuple[str, ...] = DEFAULTS["legal_granularity_w"]
    per_channel_act: bool = DEFAULTS["per_channel_act"]
    symmetric_only: bool = DEFAULTS["symmetric_only"]
    has_dla: bool = False
    dla_op_whitelist: Optional[list] = None
    mem_capacity_gb: Optional[float] = None
    # ---- 骨架: 本阶段不消费, 留字段供后续 (soft/auto enforcement、IP 枚举、工具链) ----
    ip_enum: Optional[dict] = None
    toolchain: Optional[dict] = None

    @classmethod
    def from_manifest(cls, hw: dict) -> "HwCapability":
        hw = hw or {}
        g = lambda k: hw.get(k, DEFAULTS.get(k))
        return cls(
            name=hw.get("name", "unknown"),
            int8_align=int(g("int8_align")),
            fp16_align=int(g("fp16_align")),
            int8_pack_factor=int(hw.get("int8_pack_factor", DEFAULTS["int8_pack_factor"])),
            alignment_enforcement=str(g("alignment_enforcement")),
            legal_bits=tuple(hw.get("legal_bits", DEFAULTS["legal_bits"])),
            legal_granularity_w=tuple(hw.get("legal_granularity_w", DEFAULTS["legal_granularity_w"])),
            per_channel_act=bool(hw.get("per_channel_act", DEFAULTS["per_channel_act"])),
            symmetric_only=bool(hw.get("symmetric_only", DEFAULTS["symmetric_only"])),
            has_dla=bool(hw.get("has_dla", False)),
            dla_op_whitelist=hw.get("dla_op_whitelist"),
            mem_capacity_gb=hw.get("mem_capacity_gb"),
            ip_enum=hw.get("ips"),
            toolchain=hw.get("toolchain"),
        )

    def effective_int8_align(self, groups: int) -> int:
        """分组 int8 的有效可建对齐 = pack_factor × groups (TVM dp4a/WMMA per-group 打包)。

        不叠 int8_align (那是 TRT 张量 tiling, 进 round_to 不进可建性)。仅当 manifest
        未提供 int8_buildable_align (旧版/外部 manifest) 时由 bridge 现算的兜底; 正常
        路径走 KnobSpec.int8_buildable_align (stage1 已算好)。
        """
        return self.int8_pack_factor * max(1, int(groups))


# ---------------------------------------------------------------------------
# 剪枝旋钮 (manifest view_b1_search_groups 一条 → 一个搜索旋钮)
# ---------------------------------------------------------------------------
@dataclass
class KnobSpec:
    search_group_id: str
    bucket: str
    cur_widths: tuple[int, ...]           # 成员当前宽度 (去重排序)
    round_to: int                          # 剪枝粒度 (合法宽度步长)
    int8_buildable_align: int              # dp4a int8 可建对齐 (合法宽度的可建子集)
    max_rate: float                        # 最紧剪枝率 (成员 min)
    grouped_conv: bool
    criterion_pool: tuple[str, ...]
    member_b1_groups: tuple[str, ...] = ()

    @classmethod
    def from_entry(cls, e: dict, hw: HwCapability) -> "KnobSpec":
        round_to = int(e.get("round_to", DEFAULTS["round_to"]))
        grouped = bool(e.get("grouped_conv", DEFAULTS["grouped_conv"]))
        # 防御性: int8_buildable_align 缺失 → 现算 (旧 manifest); 非分组退化为 round_to。
        ba = e.get("int8_buildable_align")
        if ba is None:
            # 无整数组数时按 grouped 标志保守: 分组用 hw.effective_int8_align 需 groups,
            # 但 view_b1_search_groups 不带整数组数 → 退回 round_to (不新增约束)。
            ba = round_to
        return cls(
            search_group_id=str(e.get("search_group_id", "knob")),
            bucket=str(e.get("bucket", "unknown")),
            cur_widths=tuple(sorted(int(w) for w in e.get("widths", []))),
            round_to=round_to,
            int8_buildable_align=int(ba),
            max_rate=float(e.get("max_rate", DEFAULTS["max_rate"])),
            grouped_conv=grouped,
            criterion_pool=tuple(e.get("criterion_pool", DEFAULTS["criterion_pool"])),
            member_b1_groups=tuple(e.get("member_b1_groups", ())),
        )

    @property
    def base_width(self) -> int:
        """未剪 (最大) 宽度。"""
        return max(self.cur_widths) if self.cur_widths else self.round_to

    def legal_widths(self) -> list[int]:
        """该旋钮的合法剪枝宽度 = [floor, base] 内 round_to 的倍数。

        floor 由 max_rate 决定 (剪到最紧), 向上取整到 round_to 且 ≥ round_to。
        这是"结构上可剪"的宽度集 (含 int8 不可建者); 是否可建 int8 另由
        buildable_int8 判定 —— 二者分离正是耦合陷阱的载体 (可剪≠可建 int8)。
        """
        base = self.base_width
        rt = max(1, self.round_to)
        lo = base * (1.0 - self.max_rate)
        floor = max(rt, int(math.ceil(lo / rt)) * rt)
        return [w for w in range(floor, base + 1, rt) if w % rt == 0]

    def buildable_int8(self, w: int) -> bool:
        """w 能否编译出 dp4a int8 快核 ⟺ w % int8_buildable_align == 0。

        铁律①: 仅依赖 stage1 算好的对齐数, 不依赖整数组数。CoDriving(g=1)
        align=round_to → 合法宽度恒可建; Pyramid 分组 align=128 → trap 宽度可剪
        但不可建 int8。
        """
        return int(w) % max(1, self.int8_buildable_align) == 0

    def buildable_int8_widths(self) -> list[int]:
        return [w for w in self.legal_widths() if self.buildable_int8(w)]


# ---------------------------------------------------------------------------
# 量化单元 / 路由段 (B2 / D 视图)
# ---------------------------------------------------------------------------
@dataclass
class QuantUnit:
    unit: str
    quantizable: bool
    legal_bits: tuple[str, ...]
    legal_granularity: tuple[str, ...]
    member_groups: tuple[str, ...]

    @classmethod
    def from_entry(cls, e: dict) -> "QuantUnit":
        return cls(
            unit=str(e.get("unit", "unit")),
            quantizable=bool(e.get("quantizable", DEFAULTS["quantizable"])),
            legal_bits=tuple(e.get("legal_bits", DEFAULTS["legal_bits"])),
            legal_granularity=tuple(e.get("legal_granularity_w",
                                          e.get("legal_granularity", DEFAULTS["legal_granularity_w"]))),
            member_groups=tuple(e.get("member_groups", ())),
        )


@dataclass
class RoutingSegment:
    device: str
    n_nodes: int = 0

    @classmethod
    def from_entry(cls, e: dict) -> "RoutingSegment":
        return cls(device=str(e.get("device", "gpu")), n_nodes=int(e.get("n_nodes", 0)))


# ---------------------------------------------------------------------------
# 搜索空间 (顶层; manifest → SpaceSpec)
# ---------------------------------------------------------------------------
@dataclass
class SpaceSpec:
    model: str
    hw: HwCapability
    knobs: list[KnobSpec]
    quant_units: list[QuantUnit] = field(default_factory=list)
    routing: list[RoutingSegment] = field(default_factory=list)
    raw: dict = field(default_factory=dict)

    # ---- 构造 ----
    @classmethod
    def from_manifest(cls, path: str | Path) -> "SpaceSpec":
        m = yaml.safe_load(Path(path).read_text())
        if m.get("scan_status") not in (None, "ok"):
            raise ValueError(f"manifest scan_status != ok: {path} ({m.get('scan_status')})")
        hw = HwCapability.from_manifest(m.get("hw_capability", {}))
        knobs = [KnobSpec.from_entry(e, hw) for e in m.get("view_b1_search_groups", [])]
        units = [QuantUnit.from_entry(e) for e in m.get("view_b2_quant_units", [])]
        seg_blk = m.get("view_d_routing_segments", {}) or {}
        routing = [RoutingSegment.from_entry(s) for s in seg_blk.get("segments", [])]
        return cls(model=str(m.get("model", "unknown")), hw=hw, knobs=knobs,
                   quant_units=units, routing=routing, raw=m)

    # ---- 旋钮访问 ----
    def knob(self, name: str) -> Optional[KnobSpec]:
        return next((k for k in self.knobs if k.search_group_id == name), None)

    def knobs_by_bucket(self, bucket: str) -> list[KnobSpec]:
        return [k for k in self.knobs if k.bucket == bucket]

    # ---- 耦合结构 (供 bridge 之后的耦合分数预测器; 本阶段只暴露不打分) ----
    def coupling_buckets(self) -> dict[str, list[str]]:
        """bucket → 该桶旋钮 id。同桶旋钮共享语义, 是耦合候选的粗划分。"""
        out: dict[str, list[str]] = {}
        for k in self.knobs:
            out.setdefault(k.bucket, []).append(k.search_group_id)
        return out

    def has_int8_buildability_cliff(self) -> bool:
        """是否存在"可剪但不可建 int8"的宽度 (耦合陷阱的结构性前提)。

        测量而非规则: 任一旋钮的合法宽度里有 int8 不可建者 → 该架构 P×Q 有耦合载体。
        CoDriving 全 g=1 → 恒 False (可分离); Pyramid 分组旋钮 → True (P-hub 耦合)。
        """
        return any(set(k.legal_widths()) - set(k.buildable_int8_widths()) for k in self.knobs)

    # ---- 自检报告 ----
    def summary(self) -> dict:
        return {
            "model": self.model,
            "hw": self.hw.name,
            "n_knobs": len(self.knobs),
            "n_quant_units": len(self.quant_units),
            "n_routing_segments": len(self.routing),
            "int8_buildability_cliff": self.has_int8_buildability_cliff(),
            "knobs": [
                {
                    "id": k.search_group_id, "grouped": k.grouped_conv,
                    "round_to": k.round_to, "int8_align": k.int8_buildable_align,
                    "legal": k.legal_widths(),
                    "int8_buildable": k.buildable_int8_widths(),
                }
                for k in self.knobs
            ],
        }


def load_spacespec(model: str,
                   partitions_dir: str | Path = None) -> SpaceSpec:
    """便捷加载: model 名 → SpaceSpec (默认 framework/partitions/)。"""
    base = Path(partitions_dir) if partitions_dir else Path(__file__).resolve().parent / "partitions"
    return SpaceSpec.from_manifest(base / f"{model}_partition.yaml")


# ---------------------------------------------------------------------------
# 自检: 解析全部 manifest, 打印可建性表 (python -m framework.stage1_bridge)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    pdir = Path(__file__).resolve().parent / "partitions"
    for mf in sorted(pdir.glob("*_partition.yaml")):
        model = mf.stem.replace("_partition", "")
        try:
            spec = SpaceSpec.from_manifest(mf)
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] {model}: {type(e).__name__}: {e}")
            continue
        s = spec.summary()
        print(f"\n=== {model}  hw={s['hw']}  knobs={s['n_knobs']}  "
              f"int8_cliff={s['int8_buildability_cliff']} ===")
        for k in s["knobs"]:
            unbuild = sorted(set(k["legal"]) - set(k["int8_buildable"]))
            print(f"  {k['id']:18s} grouped={str(k['grouped']):5s} "
                  f"round_to={k['round_to']:3d} int8_align={k['int8_align']:3d} "
                  f"legal={k['legal']} int8_unbuildable={unbuild}")
