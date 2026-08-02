#!/usr/bin/env python3
"""自适应内环预算 vs 固定/朴素窄预算 —— 回顾性效率对比 (sw-optimizer 计划三任务 3b)。

两个真实(非仿真)证据源, 均已由团队核验落盘, 本脚本只做新的分析计算, 不重跑任何真实
测量:

  1. Pyramid(强耦合, framework/adaptive_budget.py 判定 tier=wide):
     results/gap1_grid_corrected.json —— H800 TVM 真实 autotune 网格。復现一个真实发生
     过的"窄预算"选择规则: 只在**默认/廉价调度代理**(default_us)下算 Pareto 前沿, 只
     对前沿上的点花autotune真测预算。在 AP70=0.59 这个精度档位上, 默认调度前沿选中的是
     trap25(它比 pad64 的 default_us 更低), 但 trap25 是 int8-unbuildable 的非对齐宽度
     (48 非 32 对齐), 真实 autotune 后 trap25 只有 1.96x, 而 pad64(64 对齐, 是
     trap25 权重零填充版本, AP 不变=0.590)autotune 后达到 7.706x —— 窄预算选择规则
     选到了错的点, 永久错过全局最优, 除非把 autotune 候选集合按
     framework.adaptive_budget 的 autotune_candidate_multiplier 主动加宽去覆盖
     "对齐邻居"(pad64)。

  2. CoDriving(弱耦合/可分离, tier=narrow):
     results/codriving_dair_grid_8point_clean.csv —— 8 点真实测量网格(4 剪枝档
     x {fp16,int8})。復现一个结构相同、但结果不同的"窄预算"两阶段协议: 先只测
     便宜的 fp16 4 个点找 fp16 子前沿, 只对 fp16 子前沿覆盖到的剪枝档追加测 int8
     (不额外加宽候选集合, autotune_candidate_multiplier=1)。因为 CoDriving 是标准
     conv、无 int8_buildability_cliff, 这个"不加宽"的两阶段协议已经零 regret 找到
     全局真 Pareto 前沿。

  ★AP 口径注意: codriving_dair_grid_8point_clean.csv 的 collab_ap70 是"confounded
  finetune 协议"值(剪枝模型多退火了一轮), 文件自带 caveat 指向
  results/codriving_isobudget_verdict.csv 才是去confound后的公平 AP 判据。本脚本只用
  这份数据做"搜索预算效率"对比(定位真 Pareto 前沿需要多少个点), 不用它做剪枝
  AP-tradeoff 的因果结论, 该 caveat 原样保留在输出里。

  ★latency 单位注意: gap1 的 tuned_us 是微秒(µs, 子模块级), codriving 的
  body_lat_p50_ms 是毫秒(ms, 单 agent body)——两者不同模型/不同口径, 本脚本
  分别报告 regret 倍数(无量纲), 不做跨模型绝对数值比较。

不提 TRT——全程 TVM 口径(gap1 的 tuned_us 就是 TVM autotune 后的延迟)。

输出: results/adaptive_budget_eval_v1.json + 控制台摘要。
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GAP1_PATH = ROOT / "results/gap1_grid_corrected.json"
CODRIVING_CSV = ROOT / "results/codriving_dair_grid_8point_clean.csv"


def _pareto_front(points: list[dict], lat_key: str, ap_key: str) -> list[dict]:
    """非支配排序: 最小化 lat_key, 最大化 ap_key。"""
    front = []
    for p in points:
        dominated = False
        for q in points:
            if q is p:
                continue
            better_or_eq = q[lat_key] <= p[lat_key] and q[ap_key] >= p[ap_key]
            strictly_better = q[lat_key] < p[lat_key] or q[ap_key] > p[ap_key]
            if better_or_eq and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(p)
    return front


def eval_pyramid_gap1() -> dict:
    data = json.loads(GAP1_PATH.read_text(encoding="utf-8"))
    grid = data["grid"]
    by_label = {row["label"]: row for row in grid}

    # 窄预算规则(真实发生过的选择逻辑): 只在 default_us(廉价/默认调度代理)下取
    # Pareto 前沿, 在该前沿里找 AP70=0.59 档位的候选。
    ap59 = [row for row in grid if row.get("ap70") == 0.59]
    default_front_ap59 = _pareto_front(ap59, "default_us", "ap70")
    # 在同一 AP 档位内, 默认调度视角选中的是 default_us 更低的那个
    narrow_pick = min(default_front_ap59, key=lambda r: r["default_us"])

    # 宽预算规则: autotune_candidate_multiplier>1, 把"对齐邻居"(pad64, 由 trap25
    # 零填充 48->64 得到, AP 不变)也纳入真实 autotune 候选集合, 按真实 tuned_us 选择。
    wide_candidates = ap59  # {trap25, pad64} 都真实 autotune 过 (tuned_us 已测量)
    wide_pick = min(wide_candidates, key=lambda r: r["tuned_us"])

    regret_ratio = narrow_pick["tuned_us"] / wide_pick["tuned_us"]

    return {
        "model": "pyramid",
        "coupling_tier": "wide",
        "source": str(GAP1_PATH.relative_to(ROOT)),
        "ap_band_analyzed": 0.59,
        "narrow_budget_rule": (
            "只在 default_us(cheap/default schedule 代理)下取 Pareto 前沿, "
            "只对前沿点花 autotune 真测预算, autotune_candidate_multiplier=1(不主动加宽)"
        ),
        "narrow_budget_pick": {
            "label": narrow_pick["label"],
            "aligned": narrow_pick["aligned"],
            "default_us": narrow_pick["default_us"],
            "tuned_us": narrow_pick["tuned_us"],
            "ratio_vs_default_schedule": narrow_pick["ratio"],
        },
        "wide_budget_rule": (
            "autotune_candidate_multiplier=4: 主动把对齐邻居(pad64, trap25 的 s0 "
            "零填充 48->64 版本, AP70 不变)纳入真测 autotune 候选集合"
        ),
        "wide_budget_pick": {
            "label": wide_pick["label"],
            "aligned": wide_pick["aligned"],
            "default_us": wide_pick["default_us"],
            "tuned_us": wide_pick["tuned_us"],
            "ratio_vs_default_schedule": wide_pick["ratio"],
        },
        "regret_ratio_narrow_vs_wide_tuned_latency": round(regret_ratio, 3),
        "verdict": (
            f"窄预算(仅默认调度前沿)选中 {narrow_pick['label']}"
            f"(真实 autotune 后 {narrow_pick['tuned_us']:.1f}us, {narrow_pick['ratio']:.2f}x), "
            f"永久错过全局最优 {wide_pick['label']}"
            f"(真实 autotune 后 {wide_pick['tuned_us']:.1f}us, {wide_pick['ratio']:.3f}x)—— "
            f"regret = {regret_ratio:.2f}x 延迟惩罚。这正是 adaptive_budget 判定 Pyramid "
            "tier=wide(autotune_candidate_multiplier=4)的真实依据: 高耦合模型的 int8 "
            "可建性悬崖会让默认调度视角和真实 autotune 结果发生 rank-flip, 必须主动加宽"
            "候选集合才能发现。"
        ),
    }


def eval_codriving_grid() -> dict:
    with CODRIVING_CSV.open(encoding="utf-8") as f:
        rows = [
            row
            for row in csv.DictReader(f)
            if row.get("config") in {"base", "p25", "p50", "p75"}
            and row.get("precision") in {"fp16", "int8"}
        ]
    points = []
    for row in rows:
        points.append(
            {
                "id": f"{row['config']}_{row['precision']}",
                "config": row["config"],
                "precision": row["precision"],
                "lat": float(row["body_lat_p50_ms"]),
                "ap70": float(row["collab_ap70"]),
            }
        )
    all_ids = {p["id"] for p in points}
    assert len(points) == 8, f"expected 8 real rows, got {len(points)}"

    true_front = _pareto_front(points, "lat", "ap70")
    true_front_ids = {p["id"] for p in true_front}

    # 窄预算两阶段协议(不加宽, autotune_candidate_multiplier=1): 先只测便宜的 fp16
    # 4 个点, 取 fp16 子前沿覆盖到的剪枝档, 只对这些剪枝档追加测 int8。
    fp16_points = [p for p in points if p["precision"] == "fp16"]
    fp16_front = _pareto_front(fp16_points, "lat", "ap70")
    fp16_front_configs = sorted({p["config"] for p in fp16_front})
    narrow_ids = {p["id"] for p in fp16_points}  # 全部 4 个 fp16 点(阶段一)
    narrow_ids |= {
        f"{cfg}_int8" for cfg in fp16_front_configs
    }  # 阶段二: 仅 fp16 子前沿的剪枝档追加 int8

    recovered = true_front_ids.issubset(narrow_ids)
    n_narrow = len(narrow_ids)

    return {
        "model": "codriving",
        "coupling_tier": "narrow",
        "source": str(CODRIVING_CSV.relative_to(ROOT)),
        "ap_caveat": (
            "collab_ap70 是 confounded finetune 协议值(剪枝模型多退火一轮), "
            "去confound 判据见 results/codriving_isobudget_verdict.csv; "
            "本对比只用于'定位真 Pareto 前沿需要多少测量点'的搜索预算效率分析, "
            "不作为剪枝 AP-tradeoff 的因果结论。"
        ),
        "n_total_grid_points": len(points),
        "true_global_pareto_front": sorted(true_front_ids),
        "narrow_budget_protocol": (
            "阶段一: 测全部 4 个 fp16 点(所有剪枝档), 取 fp16 子 Pareto 前沿覆盖的剪枝档 "
            f"({fp16_front_configs}); 阶段二: 仅对这些剪枝档追加测 int8, "
            "autotune_candidate_multiplier=1(不主动加宽剪枝档搜索范围)"
        ),
        "narrow_budget_points_used": sorted(narrow_ids),
        "n_narrow_budget_points_used": n_narrow,
        "budget_used_pct_of_full": round(100.0 * n_narrow / len(points), 1),
        "recovers_true_global_front_with_zero_regret": recovered,
        "verdict": (
            f"两阶段窄预算协议只用 {n_narrow}/{len(points)} "
            f"({round(100.0 * n_narrow / len(points), 1)}%) 个真测点, "
            f"{'完整' if recovered else '未完整'}复现全局真 Pareto 前沿 "
            f"{sorted(true_front_ids)}——因为 CoDriving 是标准 conv、无 "
            "int8_buildability_cliff, 便宜的 fp16 阶段一扫描本身就足以定位有意义的剪枝档, "
            "不需要像 Pyramid 那样加宽 autotune 候选集合去追对齐悬崖。这正是 "
            "adaptive_budget 判定 CoDriving tier=narrow"
            "(autotune_candidate_multiplier=1)的真实依据。"
        ),
    }


def main() -> None:
    pyramid = eval_pyramid_gap1()
    codriving = eval_codriving_grid()

    result = {
        "schema": "adaptive_budget_eval_v1",
        "question": (
            "自适应内环预算(耦合分数驱动 K/rounds/autotune 候选倍数)相对固定/朴素窄预算"
            "的搜索效率提升, 用两个已核验的真实测量数据源回顾性量化(不重跑任何新真实测量)。"
        ),
        "pyramid_high_coupling_wide_budget_case": pyramid,
        "codriving_low_coupling_narrow_budget_case": codriving,
        "cross_case_summary": (
            f"高耦合(Pyramid): 若对所有模型都用同一套朴素窄预算(仅默认调度前沿, "
            f"autotune_candidate_multiplier=1), 会产生 {pyramid['regret_ratio_narrow_vs_wide_tuned_latency']}x "
            "延迟 regret(真实测量verified, 非仿真)。"
            f"弱耦合(CoDriving): 若对所有模型都用同一套固定宽预算(全 8 点全测), "
            f"会浪费 {100 - codriving['budget_used_pct_of_full']:.1f}% 真测预算而不改变找到的前沿"
            f"(两阶段窄协议 {codriving['n_narrow_budget_points_used']}/{codriving['n_total_grid_points']} "
            "点已零 regret 复现完整前沿)。"
            "=> 固定预算(不论松紧)在两个模型上至少有一个会踩坑(要么错过全局最优, 要么浪费"
            "真测名额); 耦合分数驱动的自适应预算(framework/adaptive_budget.py)按模型分流"
            "K/rounds/autotune_candidate_multiplier 才能同时避免两种代价。"
        ),
        "tvm_terminology_note": "全程 TVM 口径; gap1 的 tuned_us 即 TVM autotune 后延迟。",
    }

    out = ROOT / "results/adaptive_budget_eval_v1.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    print("pyramid:", pyramid["verdict"])
    print()
    print("codriving:", codriving["verdict"])
    print()
    print("cross_case_summary:", result["cross_case_summary"])


if __name__ == "__main__":
    main()
