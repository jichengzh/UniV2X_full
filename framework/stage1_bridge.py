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

LEGACY_PRUNE_RATES = (0.25, 0.50, 0.75)
STAGE_NAME_BY_SUFFIX = {
    "s0": "stage1",
    "s1": "stage2",
    "s2": "stage3",
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

    # ---- 逐-knob 耦合特征 (stage1→stage2 的分流信号; 解析+结构先验, 非2点回归) ----
    def cliff_strength(self) -> float:
        """P×Q 耦合强度 = log2(int8_buildable_align / round_to) (≥0)。

        剪枝宽度(P)是否决定 int8(Q)可达: 当 int8_buildable_align > round_to(分组对齐
        悬崖), 部分合法剪宽不可建 int8 → P 选择把 Q 轴锁死 → 内外耦合。纯解析,
        从 manifest 对齐数直接算: Pyramid 分组 log2(128/32)=2; 标准卷积 log2(1)=0。
        """
        rt = max(1, self.round_to)
        ratio = max(1.0, self.int8_buildable_align / rt)
        return math.log2(ratio)

    def schedule_headroom_prior(self, measured_gain: Optional[float] = None) -> float:
        """P×S 耦合强度 ∈ [0,1] = 调度自动调优的余量 (越大内环越该联合搜)。

        机理: 宽度一变, 合法 tile/快核/可融合性随之重建 → 调度最优点漂移; 余量大
        则内环(调度搜索)对最终延迟影响大、且依赖宽度 → P×S 强耦合。
        结构先验(无 per-block 实测时): 分组/非标卷积 TVM 调优增益大(实测跨模型定标:
        Pyramid 分组 ~8.7× vs CoDriving 标准 ~2×, HANDOFF_codesign_unified §2) → grouped
        高余量、标准低余量。measured_gain(该块 default/tuned 实测比)给了则优先用、归一化。
        """
        if measured_gain is not None and measured_gain > 1.0:
            # 实测优先: 归一化到 [0,1], 以跨模型上界 ~9× 为满刻度。
            return min(1.0, math.log2(measured_gain) / math.log2(9.0))
        # 结构先验 (定标自跨模型 TVM 增益: 分组≈8.7×→~1.0 / 标准≈2×→~0.32)。
        return (math.log2(8.7) if self.grouped_conv else math.log2(2.0)) / math.log2(9.0)

    def coupling_score(self, measured_gain: Optional[float] = None,
                       w_cliff: float = 0.5, w_sched: float = 0.5) -> float:
        """逐-knob 内外环(S↔P×Q)耦合分数 ∈ [0,1]。

        = w_cliff·(cliff_strength 归一化) + w_sched·schedule_headroom_prior。
        cliff 以 log2(32)=5(对齐悬崖经验上界)归一化。高分 → stage2 该旋钮联合搜
        (付内环); 低分 → 串行(默认调度下锁外环, 省内环预算)。
        """
        cliff_norm = min(1.0, self.cliff_strength() / 5.0)
        sched = self.schedule_headroom_prior(measured_gain)
        return round(w_cliff * cliff_norm + w_sched * sched, 4)


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
        CoDriving 全 g=1 → 恒 False (trace dense space 无 int8 buildability cliff);
        Pyramid 分组旋钮 → True (P-hub 耦合)。该信号不是模型级可分离证明。
        """
        return any(set(k.legal_widths()) - set(k.buildable_int8_widths()) for k in self.knobs)

    # ---- 旧逐-knob 诊断计划 (历史脚本兼容; 当前 Stage2 policy 见 stage2_search_space) ----
    def dispatch_plan(self, tau: float = 0.5,
                      measured_gains: Optional[dict] = None) -> list[dict]:
        """每个搜索旋钮的耦合分数 + legacy 诊断分流。

        保留给历史脚本和结构风险解释。当前 Stage2SearchSpace 的正式结论是
        模型级 `model_search_policy`，不把这里的逐 knob joint/serial 当作当前
        搜索空间停止目标。

        measured_gains: {search_group_id: default/tuned 实测比} 可选, 优先于结构先验。
        """
        mg = measured_gains or {}
        plan = []
        for k in self.knobs:
            score = k.coupling_score(measured_gain=mg.get(k.search_group_id))
            plan.append({
                "knob": k.search_group_id, "bucket": k.bucket,
                "grouped_conv": k.grouped_conv,
                "round_to": k.round_to, "int8_buildable_align": k.int8_buildable_align,
                "cliff_strength": round(k.cliff_strength(), 3),
                "schedule_headroom": round(k.schedule_headroom_prior(mg.get(k.search_group_id)), 3),
                "coupling_score": score,
                "dispatch": "joint" if score >= tau else "serial",
            })
        return sorted(plan, key=lambda p: -p["coupling_score"])

    def coupling_summary(self, tau: float = 0.5,
                         measured_gains: Optional[dict] = None) -> dict:
        plan = self.dispatch_plan(tau, measured_gains)
        n_joint = sum(1 for p in plan if p["dispatch"] == "joint")
        return {
            "model": self.model,
            "int8_buildability_cliff": self.has_int8_buildability_cliff(),
            "max_coupling_score": max((p["coupling_score"] for p in plan), default=0.0),
            "n_knobs": len(plan), "n_joint": n_joint, "n_serial": len(plan) - n_joint,
            "architecture_verdict": (
                "STRUCTURAL_COUPLING_RISK_DIAGNOSTIC"
                if n_joint
                else "STRUCTURAL_LOW_CLIFF_RISK_NOT_MODEL_SEPARABLE_PROOF"
            ),
            "dispatch_plan": plan,
        }

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

    # ---- Stage2 正式搜索空间契约 (模型级 policy, 非逐-knob dispatch) ----
    def stage2_search_space(self) -> dict:
        """Build the Stage2SearchSpace contract consumed by next-phase Stage2.

        This keeps the old SpaceSpec/dispatch_plan API intact for historical
        scripts, but exposes the current contract requested by the Stage2 gap
        plan: hierarchical blocks, P/Q software candidates, model-level policy,
        and dense-core/full-model claim scope.
        """
        software_candidates = [
            _software_candidate_from_knob(k, self.quant_units)
            for k in self.knobs
        ]
        hierarchical_blocks = _hierarchical_blocks(self.raw, software_candidates, self.routing)
        claim_scope = _claim_scope(self.raw, hierarchical_blocks)
        return {
            "schema": "stage2_search_space_v1",
            "model": self.model,
            "hardware_target": {
                "name": self.hw.name,
                "int8_align": self.hw.int8_align,
                "fp16_align": self.hw.fp16_align,
                "int8_pack_factor": self.hw.int8_pack_factor,
                "alignment_enforcement": self.hw.alignment_enforcement,
            },
            "optimized_scope": claim_scope["optimized_scope"],
            "software_candidates": software_candidates,
            "hardware_candidates": _hardware_candidates(self.hw, self.routing),
            "model_search_policy": _model_search_policy(self),
            "hierarchical_blocks": hierarchical_blocks,
            "quant_units": [_quant_unit_summary(q) for q in self.quant_units],
            "routing_segments": [_routing_segment_summary(r) for r in self.routing],
            "skipped_blocks": _skipped_blocks(self.raw),
            "legal_candidate_policy": {
                "primary_axis": "width_anchor",
                "prune_rate_is_report_field": True,
                "legacy_prune_rates": list(LEGACY_PRUNE_RATES),
                "default_anchors_per_dense_stage_group": "5-9_when_space_permits",
                "anchor_priority": [
                    "baseline",
                    "hardware_cliff_neighbors",
                    "ap_anchor_widths",
                    "schedule_lut_measured_widths",
                    "low_mid_high_prune_coverage",
                ],
            },
            "claim_scope": claim_scope,
            "unsupported_conclusions": sorted(set(
                _model_search_policy(self)["unsupported_conclusions"]
                + claim_scope["unsupported_conclusions"]
            )),
        }


def load_spacespec(model: str,
                   partitions_dir: str | Path = None) -> SpaceSpec:
    """便捷加载: model 名 → SpaceSpec (默认 framework/partitions/)。"""
    base = Path(partitions_dir) if partitions_dir else Path(__file__).resolve().parent / "partitions"
    return SpaceSpec.from_manifest(base / f"{model}_partition.yaml")


def load_stage2_search_space(path: str | Path) -> dict:
    """Load a partition manifest as the Stage2SearchSpace v1 contract."""
    return SpaceSpec.from_manifest(path).stage2_search_space()


def _stage_from_search_group(search_group_id: str, bucket: str) -> str:
    suffix = search_group_id.rsplit(".", 1)[-1] if "." in search_group_id else ""
    if suffix in STAGE_NAME_BY_SUFFIX:
        return STAGE_NAME_BY_SUFFIX[suffix]
    if bucket in {"neck", "heads"}:
        return bucket
    return "stage_unspecified"


def _nearest_legal_width(target: float, legal_widths: list[int]) -> int:
    return min(legal_widths, key=lambda w: (abs(w - target), -w))


def _quant_unit_for_knob(knob: KnobSpec, quant_units: list[QuantUnit]) -> Optional[QuantUnit]:
    for unit in quant_units:
        if unit.unit == knob.bucket:
            return unit
        if set(unit.member_groups) & set(knob.member_b1_groups):
            return unit
    return None


def _quant_policies(knob: KnobSpec, quant_units: list[QuantUnit]) -> tuple[str, list[dict]]:
    unit = _quant_unit_for_knob(knob, quant_units)
    if unit is None:
        return "evidence_only_attribute", [
            {
                "policy": "fp16",
                "backend_scope": "measured_h800_tvm",
                "provenance": "hw_capability_without_quant_unit",
                "status": "fixed",
            }
        ]
    if not unit.quantizable:
        return "fixed_quant_policy", [
            {
                "policy": "fp16",
                "backend_scope": "measured_h800_tvm",
                "provenance": f"quant_unit:{unit.unit}",
                "status": "fixed",
            }
        ]

    policies = [
        {
            "policy": "fp16",
            "backend_scope": "measured_h800_tvm",
            "provenance": f"quant_unit:{unit.unit}",
            "status": "active",
        }
    ]
    if "INT8" in unit.legal_bits:
        policies.append({
            "policy": "int8",
            "backend_scope": "measured_h800_tvm",
            "provenance": f"quant_unit:{unit.unit}:legal_bits",
            "status": "active",
        })
    return "active_software_candidate", policies


def _select_representative_widths(widths: list[int]) -> list[int]:
    if len(widths) <= 3:
        return widths
    idxs = {0, len(widths) // 2, len(widths) - 1}
    return [widths[i] for i in sorted(idxs)]


def _width_anchors(knob: KnobSpec) -> list[dict]:
    legal = knob.legal_widths()
    if not legal:
        return []

    base = knob.base_width
    buildable = set(knob.buildable_int8_widths())
    roles_by_width: dict[int, set[str]] = {}

    def add(width: int, role: str) -> None:
        if width in legal:
            roles_by_width.setdefault(width, set()).add(role)

    add(base, "baseline")
    for width in _select_representative_widths([w for w in legal if w in buildable]):
        add(width, "hardware_buildable_anchor")
    for width in _select_representative_widths(legal):
        add(width, "low_mid_high_prune_coverage")

    for rate in LEGACY_PRUNE_RATES:
        if rate <= knob.max_rate + 1e-9:
            mapped = _nearest_legal_width(base * (1.0 - rate), legal)
            add(mapped, f"legacy_{int(rate * 100)}pct_mapped")

    if set(legal) - buildable:
        for width in sorted(buildable):
            lower = [w for w in legal if w < width and w not in buildable]
            upper = [w for w in legal if w > width and w not in buildable]
            if lower:
                add(lower[-1], "hardware_cliff_neighbor")
            if upper:
                add(upper[0], "hardware_cliff_neighbor")

    def priority(item: tuple[int, set[str]]) -> tuple[int, int]:
        width, roles = item
        if "baseline" in roles:
            return (0, -width)
        if any(r.startswith("legacy_") for r in roles):
            return (1, -width)
        if "hardware_cliff_neighbor" in roles:
            return (2, -width)
        if "hardware_buildable_anchor" in roles:
            return (3, -width)
        return (4, -width)

    selected = sorted(roles_by_width.items(), key=priority)
    if len(selected) > 9:
        selected = selected[:9]
    selected = sorted(selected, key=lambda item: item[0])

    anchors = []
    for width, roles in selected:
        int8_buildable = knob.buildable_int8(width)
        anchor_type = "production" if int8_buildable else "diagnostic_trap"
        anchors.append({
            "width": width,
            "prune_rate": round(1.0 - (width / base), 6) if base else 0.0,
            "anchor_type": anchor_type,
            "roles": sorted(roles),
            "int8_buildable": int8_buildable,
            "round_to": knob.round_to,
            "int8_buildable_align": knob.int8_buildable_align,
        })
    return anchors


def _software_candidate_from_knob(knob: KnobSpec, quant_units: list[QuantUnit]) -> dict:
    quant_axis_status, quant_policies = _quant_policies(knob, quant_units)
    anchors = _width_anchors(knob)
    software_points = []
    for anchor in anchors:
        for policy in quant_policies:
            buildable = policy["policy"] != "int8" or anchor["int8_buildable"]
            software_points.append({
                "id": f"{knob.search_group_id}:w{anchor['width']}:{policy['policy']}",
                "width": anchor["width"],
                "prune_rate": anchor["prune_rate"],
                "quant_policy": policy["policy"],
                "buildable": buildable,
                "status": "active" if buildable else "diagnostic_only",
            })
    return {
        "id": knob.search_group_id,
        "search_group_id": knob.search_group_id,
        "bucket": knob.bucket,
        "dense_stage": _stage_from_search_group(knob.search_group_id, knob.bucket),
        "base_width": knob.base_width,
        "grouped_conv": knob.grouped_conv,
        "round_to": knob.round_to,
        "int8_buildable_align": knob.int8_buildable_align,
        "max_prune_rate": {
            "value": knob.max_rate,
            "bounds": ["structure", "hardware", "ap", "lut"],
            "bound_details": {
                "structure": "derived_from_stage1_prune_group_connectivity",
                "hardware": "derived_from_round_to_and_int8_buildable_align",
                "ap": "requires_ap_anchor_or_short_finetune_for_production_escalation",
                "lut": "requires_latency_lut_or_schedule_buildability_for_production_escalation",
            },
        },
        "width_anchors": anchors,
        "quant_axis_status": quant_axis_status,
        "quant_policies": quant_policies,
        "software_points": software_points,
    }


def _hardware_candidates(hw: HwCapability, routing: list[RoutingSegment]) -> list[dict]:
    return [
        {
            "id": "default_schedule",
            "backend_scope": "manifest_hw_capability",
            "hardware": hw.name,
            "schedule_policy": "default",
            "routing_scope": "fixed_single_gpu" if len(routing) <= 1 else "fixed_multi_segment",
        },
        {
            "id": "tvm_metaschedule_candidate",
            "backend_scope": "measured_h800_tvm",
            "hardware": hw.name,
            "schedule_policy": "tuned",
            "requires_latency_lut": True,
        },
    ]


def _quant_unit_summary(unit: QuantUnit) -> dict:
    return {
        "unit": unit.unit,
        "quantizable": unit.quantizable,
        "legal_bits": list(unit.legal_bits),
        "legal_granularity": list(unit.legal_granularity),
        "member_groups": list(unit.member_groups),
    }


def _routing_segment_summary(segment: RoutingSegment) -> dict:
    return {
        "device": segment.device,
        "n_nodes": segment.n_nodes,
        "status": "fixed_segment",
    }


def _skipped_blocks(raw: dict) -> list[dict]:
    trace = raw.get("trace", {}) or {}
    skipped_subgraphs = trace.get("skipped_subgraphs")
    if skipped_subgraphs:
        return [
            {
                "name": item.get("name", "skipped"),
                "type": item.get("type", "unknown"),
                "description": item.get("description", item.get("name", "")),
                "blocker_gate": item.get("blocker_gate", "coverage_caveat"),
                "full_model_verdict_blocker": bool(item.get("full_model_verdict_blocker", True)),
            }
            for item in skipped_subgraphs
        ]
    out = []
    for idx, desc in enumerate(trace.get("skipped_modules", []) or []):
        lowered = str(desc).lower()
        if any(token in lowered for token in ("fusion", "attention", "warp", "collab")):
            typ = "fusion_or_alignment"
            gate = "routing_fusion_coverage_anchor"
            blocker = True
        elif any(token in lowered for token in ("sparse", "vfe", "scatter")):
            typ = "sparse_or_geometry_preprocess"
            gate = "trace_sparse_frontend_boundary"
            blocker = False
        else:
            typ = "custom_or_unclassified"
            gate = "coverage_caveat"
            blocker = True
        name = str(desc).split(" ", 1)[0].strip() or f"skipped_{idx}"
        out.append({
            "name": name,
            "type": typ,
            "description": str(desc),
            "blocker_gate": gate,
            "full_model_verdict_blocker": blocker,
        })
    return out


def _hierarchical_blocks(raw: dict, software_candidates: list[dict],
                         routing: list[RoutingSegment]) -> list[dict]:
    blocks = []
    for candidate in software_candidates:
        block_type = "fused_dense_block" if candidate["grouped_conv"] else "serial_dense_block"
        blocks.append({
            "id": f"RSU:dense_perception:{candidate['dense_stage']}:{candidate['id']}",
            "device_scope": "RSU",
            "execution_block": "dense_perception",
            "dense_stage": candidate["dense_stage"],
            "search_group": candidate["id"],
            "knob": "width_anchor+quant_policy",
            "block_type": block_type,
        })
    for idx, segment in enumerate(routing):
        blocks.append({
            "id": f"RSU:routing_fixed:{idx}",
            "device_scope": "RSU",
            "execution_block": "routing_fixed",
            "dense_stage": "not_applicable",
            "search_group": None,
            "knob": "fixed_routing_segment",
            "block_type": "routing_fixed_block",
            "device": segment.device,
            "n_nodes": segment.n_nodes,
        })
    for skipped in _skipped_blocks(raw):
        if skipped["type"] == "fusion_or_alignment":
            execution = "fusion_attention_future"
            device_scope = "ego"
        else:
            execution = "skipped_blocker" if skipped["full_model_verdict_blocker"] else "skipped_caveat"
            device_scope = "RSU"
        blocks.append({
            "id": f"{device_scope}:{execution}:{skipped['name']}",
            "device_scope": device_scope,
            "execution_block": execution,
            "dense_stage": "not_applicable",
            "search_group": None,
            "knob": None,
            "block_type": skipped["type"],
            "blocker_gate": skipped["blocker_gate"],
        })
    return blocks


def _claim_scope(raw: dict, hierarchical_blocks: list[dict]) -> dict:
    future_blocks = []
    full_model_blockers = []
    for block in hierarchical_blocks:
        if block["execution_block"] == "fusion_attention_future":
            if "ego_fusion_attention_future" not in future_blocks:
                future_blocks.append("ego_fusion_attention_future")
            full_model_blockers.append(block["blocker_gate"])
        elif block["execution_block"] == "skipped_blocker":
            full_model_blockers.append(block["blocker_gate"])
    if not future_blocks:
        future_blocks.append("ego_fusion_attention_future")
    return {
        "optimized_scope": "rsu_dense_core",
        "trace_dense_core_latency_scope": "trace_net_only",
        "full_model_claim_allowed": False,
        "full_model_blockers": sorted(set(full_model_blockers)),
        "future_blocks": future_blocks,
        "unsupported_conclusions": [
            "dense_core_speedup_as_full_model_speedup",
            "ego_fusion_attention_acceleration_in_current_scope",
        ],
        "note": "RSU dense-core is the current closed loop; ego fusion/attention remains future scope.",
    }


def _model_search_policy(spec: SpaceSpec) -> dict:
    has_grouped = any(k.grouped_conv for k in spec.knobs)
    has_cliff = spec.has_int8_buildability_cliff()
    if has_grouped and has_cliff:
        selected = "joint"
        reasons = ["P_IC_BN_hub", "int8_buildability_cliff", "schedule_headroom_risk"]
    else:
        selected = "serial"
        reasons = ["standard_conv_dense_envelope", "no_int8_buildability_cliff_in_dense_core"]

    unsupported = [
        "per_knob_joint_serial",
        "dense_core_speedup_as_full_model_speedup",
    ]
    if not has_grouped:
        unsupported.extend([
            "groups1_not_model_separable_proof",
            "cross_model_extrapolation_from_standard_conv",
        ])
    return {
        "selected": selected,
        "decision_level": "model",
        "candidate_policies": ["joint", "serial", "noS/default"],
        "reasons": reasons,
        "features": {
            "has_grouped_conv": has_grouped,
            "has_int8_buildability_cliff": has_cliff,
            "n_search_groups": len(spec.knobs),
            "max_coupling_score": max(
                (k.coupling_score() for k in spec.knobs),
                default=0.0,
            ),
        },
        "unsupported_conclusions": unsupported,
    }


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
