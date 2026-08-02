#!/usr/bin/env python3
"""量化"冷启动预测做搜索器首轮粗筛"的真测预算节省 (sw-optimizer 计划三任务 2)。

数据来源(不重跑训练, 直接复用已验证真实结果):
  results/coldstart_mape_report.json —— GradientBoostingRegressor(Pyramid original60
  富表训练) 在 CoDriving 12 个真锚点(base/p25/p50/p75 x fp32/fp16/int8)上的
  zero-anchor / leave-one-out(+anchor) 预测值, 已在 plan2 中核验 MAPE/Spearman。

本脚本不产生新的模型训练或新的真实测量, 只是把已验证的预测值接到
framework/coldstart_screen.py 做粗筛 precision/recall/预算节省量化(纯分析, CPU-only,
GPU 需求为零, 符合 team-lead "GPU 需求轻" 的任务定位)。

产出: results/coldstart_screen_eval_v1.json + 控制台摘要。
"""
from __future__ import annotations

import json
from pathlib import Path

from framework.coldstart_screen import min_k_for_full_recall, recall_at_budget, screen_sweep

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "results/coldstart_mape_report.json"


def _eval_target(name: str, labels, true_values, pred_zero, pred_loo) -> dict:
    n = len(labels)
    sweep_zero = screen_sweep(labels, true_values, pred_zero)
    sweep_loo = screen_sweep(labels, true_values, pred_loo)

    top_n_candidates = [n_top for n_top in (2, 3, 4) if n_top <= n]
    min_k = {
        f"top{n_top}": {
            "zero_anchor": min_k_for_full_recall(labels, true_values, pred_zero, n_top),
            "plus_anchor_loo": min_k_for_full_recall(labels, true_values, pred_loo, n_top),
            "full_real_measurement_budget": n,
        }
        for n_top in top_n_candidates
    }

    return {
        "target": name,
        "n_candidates": n,
        "labels": list(labels),
        "true_values": list(true_values),
        "screen_sweep_zero_anchor": sweep_zero,
        "screen_sweep_plus_anchor_loo": sweep_loo,
        "min_budget_for_full_recall_of_true_topN": min_k,
        "note": (
            "min_k_for_full_recall: 用冷启动预测排序做粗筛, 要 100% 保住真值 top-N 集合"
            "所需的最小真测预算(vs 全量真测=n_candidates); "
            "random_expected_recall 字段(在 screen_sweep 每个 k 里)是同预算随机筛选的"
            "期望召回, 作对照基线。"
        ),
    }


def main() -> None:
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))

    ap70 = report["targets"]["ap70"]
    lat = report["targets"]["latency_ms"]

    ap70_eval = _eval_target(
        "ap70",
        ap70["cod_labels"],
        ap70["cod_y_true"],
        ap70["pred_zero_anchor"],
        ap70["pred_leave_one_out"],
    )
    lat_eval = _eval_target(
        "latency_ms",
        lat["cod_labels"],
        lat["cod_y_true"],
        lat["pred_zero_anchor"],
        lat["pred_leave_one_out"],
    )

    # Budget-saved headline: "spend only k_headline real-measurement slots (out of 12) --
    # do we still retain all target_top_n=4 AP-optimal candidates for the precision inner
    # loop?" k_headline=5 is the smallest budget at which plus_anchor_LOO achieves 100%
    # recall of the true top-4 (see min_budget_for_full_recall_of_true_topN.top4 below).
    k_headline = 5
    target_top_n = 4
    labels_ap70 = ap70["cod_labels"]
    true_ap70 = ap70["cod_y_true"]
    loo_recall = recall_at_budget(
        labels_ap70, true_ap70, ap70["pred_leave_one_out"], k_headline, target_top_n
    )
    zero_recall = recall_at_budget(
        labels_ap70, true_ap70, ap70["pred_zero_anchor"], k_headline, target_top_n
    )
    headline = {
        "k": k_headline,
        "target_top_n": target_top_n,
        "n_total_candidates": ap70_eval["n_candidates"],
        "budget_reduction_vs_full_real_measurement_pct": round(
            100.0 * (1 - k_headline / ap70_eval["n_candidates"]), 1
        ),
        "plus_anchor_loo_recall_of_true_top4_at_k5": loo_recall["recall"],
        "zero_anchor_recall_of_true_top4_at_k5": zero_recall["recall"],
        "random_expected_recall_of_true_top4_at_k5": loo_recall["random_expected_recall"],
        "plus_anchor_loo_detail": loo_recall,
        "zero_anchor_detail": zero_recall,
    }

    result = {
        "schema": "coldstart_screen_eval_v1",
        "source": str(REPORT_PATH.relative_to(ROOT)),
        "method": (
            "复用 results/coldstart_mape_report.json 中已验证的 zero-anchor / "
            "leave-one-out(+anchor) 预测值(未重跑训练), 用 "
            "framework/coldstart_screen.py 做首轮粗筛 precision@k/recall@k 分析, "
            "对照同预算下随机筛选的期望召回(超几何分布解析解)。"
        ),
        "ap70": ap70_eval,
        "latency_ms": lat_eval,
        "headline_ap70_k5": headline,
        "verdict": {
            "ap70": (
                "AP70 目标: zero-anchor 预测排序几乎无用(需要锚点才谈得上粗筛价值, "
                "对齐 plan2 的 rank rho 0.039->0.556 结论)。plus_anchor_LOO 粗筛在预算 "
                f"k={k_headline}/12(节省 {headline['budget_reduction_vs_full_real_measurement_pct']}% "
                "真测预算)下对真值 top-4 的召回 = "
                f"{headline['plus_anchor_loo_recall_of_true_top4_at_k5']}"
                f"(随机基线期望召回仅 {headline['random_expected_recall_of_true_top4_at_k5']}), "
                "即: 用带锚点的冷启动预测做首轮粗筛, 在近一半真测预算下已保住全部 "
                "AP-最优候选集合送入精排内环, 而随机筛选大概率漏掉。"
            ),
            "latency_ms": (
                "latency 目标: zero-anchor 排序已经 rho=1.0(单调剪枝先验免费转移排序, "
                "对齐 plan2 结论), 即 latency 轴的粗筛/排序不需要任何跨模型锚点—— "
                "只需要剪枝率的结构先验即可 100% 召回; 冷启动模型此处的唯一增量价值是"
                "把预测值做仿射校正到正确量级(564%->254% MAPE), 而非排序本身。"
            ),
            "caveat": (
                "latency_ms n=3(只有 base/p25/p50 的 fp32 真实测量点, p75 tuned crash, "
                "int8 全模型未 build), 排序结论在小样本下要谨慎(见 plan2 small_n_caveat); "
                "AP70 n=12 结论更稳健。两目标都不提 TRT, 全程 TVM 口径, 数据源同"
                "coldstart_mape_report.json。"
            ),
        },
    }

    out = ROOT / "results/coldstart_screen_eval_v1.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    print(json.dumps(headline, ensure_ascii=False, indent=2))
    for n_top, row in ap70_eval["min_budget_for_full_recall_of_true_topN"].items():
        print(f"  ap70 min_k_for_full_recall[{n_top}] = {row}")


if __name__ == "__main__":
    main()
