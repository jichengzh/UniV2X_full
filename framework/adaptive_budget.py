"""自适应内环真测预算 (Stage2 SMBO 真测内环消费).

背景 (team-lead 计划三任务 §3b): bridge 主线下一阶段 = 耦合分数预测器 + 自适应内环预算。
耦合分数预测器**已存在**, 不需要重新造:
  1. framework/stage1_bridge.py::_model_search_policy() —— 解析式逐 knob 特征
     (KnobSpec.cliff_strength/schedule_headroom_prior/coupling_score), 输出
     model_search_policy.selected ∈ {joint, serial} + features.max_coupling_score。
  2. framework/stage1/coupling_predictor.py::predict_manifest() —— 更重的
     "Safe Predictor v0", 逐模型证据分级 verdict (MEASURED_SEPARABLE /
     PREDICTED_SEPARABLE_LOW_RISK / ANCHOR_PROBED_LOW_RISK / LOW_CONFIDENCE_* /
     FUSION_UNCOVERED_UNKNOWN / JOINT_OR_PAIR_SEARCH_REQUIRED_UNTIL_C4_C5_BOUND /
     P_HUB_CONTEXT_BLOCKS_MODEL_LEVEL_STANDARD_CONV_PROMOTION), 附 blockers/risk。

本模块只做"耦合信号 -> SMBO 内环真测预算(K/rounds/autotune 候选倍数)"这一件事,
是纯函数 + 一个便捷 from_manifest() 装配入口, 不重复实现耦合检测本身。

设计原则(遵循用户 Q2 拍板: 解析检测 + 逐 knob 特征, 非 2 点回归; 详见
memory/project-v2xvit-attention-bottleneck):
  - 任一耦合信号指向"高耦合/未证明可分离" -> 取宽预算 (保守 OR 逻辑)。
    不确定时不能省预算——省错了会永久漏掉全局最优且当场察觉不到。
  - 宽预算的真实依据: results/gap1_grid_corrected.json 实测 W_g(trap25)/P_g(pad64)
    陷阱对——窄的"仅默认调度前沿"内环预算会锁死在 trap25(1.96x), 永久错过
    pad64(7.706x), 3.93x regret, 只有把 autotune 候选widen 到"对齐邻居"才找得到。
  - 窄预算的真实依据: results/codriving_dair_grid_8point_clean.csv 实测显示,
    CoDriving(标准 conv, 无 int8_buildability_cliff)的 ~50% 预算子集
    {base_fp16, p50_fp16, p75_fp16, p50_int8} 已经零 regret 复现完整真 Pareto 前沿
    {p50_int8, p50_fp16, p75_fp16}——多余的内环真测预算对可分离模型是浪费。

不提 TRT——全程 TVM 口径。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

DEFAULT_TAU = 0.5

# coupling_predictor 的 verdict 分类: 未证明可分离 -> 必须保守按高耦合处理。
NEEDS_WIDE_BUDGET_VERDICTS = {
    "JOINT_OR_PAIR_SEARCH_REQUIRED_UNTIL_C4_C5_BOUND",  # v2xvit: C4/C5 gate 未清
    "P_HUB_CONTEXT_BLOCKS_MODEL_LEVEL_STANDARD_CONV_PROMOTION",  # pyramid: P-hub 上下文
    "LOW_CONFIDENCE_NEEDS_TARGETED_PROBE",
    "FUSION_UNCOVERED_UNKNOWN",
}
# 已有测量证据支持"可分离/低风险" -> 允许窄预算。
NARROW_BUDGET_OK_VERDICTS = {
    "MEASURED_SEPARABLE",
    "PREDICTED_SEPARABLE_LOW_RISK",
    "ANCHOR_PROBED_LOW_RISK",
}

GAP1_EVIDENCE = "results/gap1_grid_corrected.json (W_g trap25 1.96x vs P_g pad64 7.706x, 3.93x regret)"
CODRIVING_EVIDENCE = (
    "results/codriving_dair_grid_8point_clean.csv "
    "(~50% 预算子集零 regret 复现完整真 Pareto 前沿)"
)


@dataclass(frozen=True)
class BudgetPlan:
    """一次 SMBO 内环真测轮的预算方案."""

    k_per_round: int
    n_rounds: int
    autotune_candidate_multiplier: int
    tier: str  # "wide" | "narrow"
    rationale: list

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def total_real_measurements(self) -> int:
        return self.k_per_round * self.n_rounds


def adaptive_inner_loop_budget(
    *,
    coupling_score: Optional[float] = None,
    selected: Optional[str] = None,  # stage1_bridge model_search_policy.selected: "joint"|"serial"
    verdict: Optional[str] = None,  # coupling_predictor verdict string
    tau: float = DEFAULT_TAU,
    k_narrow: int = 4,
    k_wide: int = 12,
    rounds_narrow: int = 1,
    rounds_wide: int = 3,
    multiplier_narrow: int = 1,
    multiplier_wide: int = 4,
) -> BudgetPlan:
    """纯函数: 耦合信号 -> SMBO 内环真测预算.

    至少提供 coupling_score / selected / verdict 三者之一; 任一信号判定"高耦合"
    即整体取宽预算(保守 OR, 不确定不省)。
    """
    if coupling_score is None and selected is None and verdict is None:
        raise ValueError("must supply at least one of coupling_score/selected/verdict")

    rationale: list[str] = []
    wide = False

    if coupling_score is not None:
        if coupling_score >= tau:
            wide = True
            rationale.append(
                f"coupling_score={coupling_score:.3f} >= tau={tau} "
                "(stage1_bridge.KnobSpec.coupling_score 解析特征信号强)"
            )
        else:
            rationale.append(
                f"coupling_score={coupling_score:.3f} < tau={tau} (解析特征信号弱)"
            )

    if selected is not None:
        if selected == "joint":
            wide = True
            rationale.append(
                "model_search_policy.selected=joint "
                "(has_grouped_conv AND int8_buildability_cliff)"
            )
        elif selected == "serial":
            rationale.append(
                "model_search_policy.selected=serial (无 grouped_conv 或无 cliff)"
            )
        else:
            wide = True
            rationale.append(f"model_search_policy.selected={selected!r} 未识别 -> 保守取宽预算")

    if verdict is not None:
        if verdict in NEEDS_WIDE_BUDGET_VERDICTS:
            wide = True
            rationale.append(f"coupling_predictor verdict={verdict} 未证明可分离 -> 保守取宽预算")
        elif verdict in NARROW_BUDGET_OK_VERDICTS:
            rationale.append(f"coupling_predictor verdict={verdict} 已有测量证据支持可分离")
        else:
            wide = True
            rationale.append(f"coupling_predictor verdict={verdict} 未分类 -> 保守取宽预算")

    if wide:
        rationale.append(f"宽预算依据(真实测量): {GAP1_EVIDENCE}")
        return BudgetPlan(k_wide, rounds_wide, multiplier_wide, "wide", rationale)

    rationale.append(f"窄预算依据(真实测量): {CODRIVING_EVIDENCE}")
    return BudgetPlan(k_narrow, rounds_narrow, multiplier_narrow, "narrow", rationale)


def from_manifest(
    manifest_path: str | Path,
    *,
    evidence_dir: Optional[str | Path] = None,
    tau: float = DEFAULT_TAU,
    **budget_kwargs: Any,
) -> BudgetPlan:
    """便捷入口: 从 stage1 manifest 装配两路耦合信号(bridge 解析式 + Safe Predictor
    证据分级)后给出预算方案。两路信号任一判高耦合即取宽预算。
    """
    from framework.stage1.coupling_predictor import predict_manifest
    from framework.stage1_bridge import load_stage2_search_space

    space = load_stage2_search_space(manifest_path)
    policy = space.get("model_search_policy", {}) or {}
    coupling_score = (policy.get("features", {}) or {}).get("max_coupling_score")
    selected = policy.get("selected")

    report = predict_manifest(manifest_path, evidence_dir=evidence_dir)
    verdict = report.get("verdict")

    return adaptive_inner_loop_budget(
        coupling_score=coupling_score,
        selected=selected,
        verdict=verdict,
        tau=tau,
        **budget_kwargs,
    )


if __name__ == "__main__":
    import json
    from pathlib import Path as _Path

    root = _Path(__file__).resolve().parents[1]
    partitions = sorted((root / "framework/partitions").glob("*_partition.yaml"))
    for p in partitions:
        plan = from_manifest(p)
        print(p.name, "->", json.dumps(plan.to_dict(), ensure_ascii=False))
