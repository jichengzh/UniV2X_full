#!/usr/bin/env python3
"""耦合分数预测器 vs 已知真耦合 一致性核验 (sw-optimizer 计划三任务 4).

已知真耦合(来自本团队已核验的真实测数据, 非本脚本假设):
  - Pyramid: 强耦合(grouped_conv + int8_buildability_cliff, gap1_grid_corrected.json
    实测 W_g/P_g 陷阱对, 3.93x regret)。
  - CoDriving: 弱耦合/可分离(标准 conv, 无 alignment cliff,
    codriving_dair_grid_8point_clean.csv 实测窄预算零 regret)。
  - V2X-ViT: 真瓶颈在 attention/fusion(92.4% e2e latency, 见
    memory/project-v2xvit-attention-bottleneck), 本框架当前优化范围(conv backbone)
    对其结构性失效 —— 一致性核验的正确答案不是"backbone 耦合分数低就判 serial",
    而是"predict_manifest 的 verdict 必须诚实标出 attention gate 未解除,
    claim_scope 不能允许 full_model_claim"。

本脚本联立两路耦合信号:
  1. framework/stage1_bridge.py 的解析式逐 knob coupling_score(浅层, 只看
     backbone dense core 的 grouped_conv/int8_buildability_cliff)。
  2. framework/stage1/coupling_predictor.py 的 Safe Predictor v0(重, 证据分级
     verdict, 显式建模 attention/fusion 覆盖缺口)。
  3. framework/adaptive_budget.py 的合并结论(两路任一判高耦合 -> wide)。

输出: results/coupling_score_consistency_v1.json + 控制台摘要。
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARTITIONS = ROOT / "framework/partitions"

GROUND_TRUTH = {
    "pyramid_lidar_partition.yaml": {
        "model": "pyramid_lidar",
        "known_coupling": "strong",
        "expected_budget_tier": "wide",
        "evidence": "results/gap1_grid_corrected.json (W_g trap25 1.96x vs P_g pad64 7.706x)",
    },
    "pyramid_camera_partition.yaml": {
        "model": "pyramid_camera",
        "known_coupling": "strong",
        "expected_budget_tier": "wide",
        "evidence": "same backbone family as pyramid_lidar (P-hub context)",
    },
    "codriving_partition.yaml": {
        "model": "codriving",
        "known_coupling": "weak_separable",
        "expected_budget_tier": "narrow",
        "evidence": "results/codriving_dair_grid_8point_clean.csv (narrow budget zero regret)",
    },
    "v2xvit_partition.yaml": {
        "model": "v2xvit",
        "known_coupling": "attention_bottleneck_out_of_current_scope",
        "expected_budget_tier": "wide",
        "evidence": (
            "results/v2xvit_e2e_breakdown.json (attention/fusion=92.4% e2e latency, "
            "conv backbone=2.4%) -> conservative wide until C4/C5 gate resolved, "
            "framework structurally cannot claim full-model gain here"
        ),
    },
    "attfuse_partition.yaml": {
        "model": "attfuse",
        "known_coupling": "attention_fusion_uncovered",
        "expected_budget_tier": "wide",
        "evidence": "attention/fusion coverage not integrated into Stage1 v0 (conservative)",
    },
    "fcooper_partition.yaml": {
        "model": "fcooper",
        "known_coupling": "low_confidence_pending_anchor",
        "expected_budget_tier": "wide",
        "evidence": "MaxFusion/BaseBEVBackbone anchors not fully measured in v0 (conservative)",
    },
}


def main() -> None:
    from framework.adaptive_budget import from_manifest
    from framework.stage1.coupling_predictor import predict_manifest
    from framework.stage1_bridge import load_stage2_search_space

    rows = []
    n_match = 0
    for fname, gt in GROUND_TRUTH.items():
        path = PARTITIONS / fname
        space = load_stage2_search_space(path)
        policy = space.get("model_search_policy", {}) or {}
        analytical_score = (policy.get("features", {}) or {}).get("max_coupling_score")
        analytical_selected = policy.get("selected")

        report = predict_manifest(path)
        verdict = report.get("verdict")
        risk = report.get("risk", {})

        plan = from_manifest(path)
        match = plan.tier == gt["expected_budget_tier"]
        n_match += int(match)

        rows.append(
            {
                "manifest": fname,
                "model": gt["model"],
                "known_coupling": gt["known_coupling"],
                "known_coupling_evidence": gt["evidence"],
                "analytical_bridge_coupling_score": analytical_score,
                "analytical_bridge_selected": analytical_selected,
                "safe_predictor_verdict": verdict,
                "safe_predictor_risk": risk,
                "safe_predictor_blockers": report.get("blockers", []),
                "merged_budget_tier": plan.tier,
                "merged_budget_plan": plan.to_dict(),
                "expected_budget_tier": gt["expected_budget_tier"],
                "consistent_with_known_ground_truth": match,
            }
        )

    result = {
        "schema": "coupling_score_consistency_v1",
        "note": (
            "两路耦合信号(stage1_bridge 解析式逐knob特征 + "
            "stage1/coupling_predictor Safe Predictor v0 证据分级)经 "
            "framework/adaptive_budget.py 的保守OR合并后, 与团队已核验的真耦合结论"
            "(Pyramid强/CoDriving弱/V2X-ViT attention瓶颈框架结构性无效)逐模型核对。"
            "V2X-ViT/AttFuse/F-Cooper 的 backbone 浅层分析式分数其实偏低(标准conv), "
            "但 Safe Predictor 的 attention/fusion 覆盖缺口标记会把合并结论保守拉回 wide"
            "——这正是'不确定时不能靠2点回归/单一浅层分数拍板, 必须逐knob解析特征+"
            "证据分级联合判断'的设计意图所在证据(用户 Q2 拍板)。"
        ),
        "n_models": len(rows),
        "n_consistent_with_known_ground_truth": n_match,
        "all_consistent": n_match == len(rows),
        "rows": rows,
        "tvm_terminology_note": "全程 TVM 口径; 若历史数据文件字面含 TRT 字样, 原样引用不改写。",
    }

    out = ROOT / "results/coupling_score_consistency_v1.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    print(f"n_models={result['n_models']} n_consistent={n_match} all_consistent={result['all_consistent']}")
    for row in rows:
        flag = "OK" if row["consistent_with_known_ground_truth"] else "MISMATCH"
        print(
            f"  [{flag}] {row['model']:16s} known={row['known_coupling']:45s} "
            f"analytical_score={row['analytical_bridge_coupling_score']:.3f} "
            f"verdict={row['safe_predictor_verdict']:55s} -> tier={row['merged_budget_tier']}"
        )


if __name__ == "__main__":
    main()
