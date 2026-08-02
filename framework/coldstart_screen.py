"""跨模型冷启动预测器 -> 搜索器首轮粗筛 (sw-optimizer 计划三任务 2).

背景: results/plan2_coldstart_value_rank.json / results/coldstart_mape_report.json 已经
用真实 CoDriving 12-anchor 数据验证过跨模型(Pyramid 富表 -> CoDriving 冷模型) cost-model
的 value/rank 迁移能力:
  - AP70: value MAPE 55.2%(zero-anchor) -> 3.2%(leave-one-out anchors), rank Spearman
    0.039 -> 0.556。价值 + 排序都需要锚点, 但锚点加入后价值/排序都变得可用。
  - latency_ms: rank 在 zero-anchor 就已经 rho=1.0(单调剪枝先验免费转移排序), 价值需要
    锚点做仿射校正(564%->254% MAPE), 排序不需要。

本模块把"冷启动预测值排序"操作化成一个可复用的**首轮粗筛**原语: 在花费昂贵的真实测量
(TVM autotune + 真 AP 评测)之前, 先用冷启动预测(或对 latency 这种目标, 直接用免费的
单调先验排序)筛出 Top-K 候选送入真测内环, 量化这样做相对于"随机选 K 个"和"全量真测"
省了多少真测预算, 以及是否保住了真 Pareto/Top-N 候选集合。

不提 TRT——全程 TVM 口径。冷启动 GBR 训练细节见
scripts/phase1/codriving_12anchor_coldstart.py(已跑过, 本模块直接复用其已验证的
LOO 预测结果做粗筛评估, 不重跑训练)。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence


@dataclass(frozen=True)
class ScreenEvalAtK:
    k: int
    true_top_k: list
    predicted_top_k: list
    retained: list  # true_top_k ∩ predicted_top_k
    precision_at_k: float  # |retained| / k
    recall_of_true_top_k: float  # |retained| / |true_top_k|
    random_expected_recall: float  # hypergeometric E[|retained|] / |true_top_k| for random K-subset

    def to_dict(self) -> dict:
        return asdict(self)


def _rank_desc(labels: Sequence[str], values: Sequence[float]) -> list[str]:
    """Descending-value ranking of labels, ties broken by label for determinism."""
    order = sorted(range(len(values)), key=lambda i: (-values[i], labels[i]))
    return [labels[i] for i in order]


def expected_random_recall(n_total: int, n_true_top: int, k: int) -> float:
    """随机不放回抽 k 个候选, 期望命中真 top-n_true_top 的比例(超几何分布期望)。

    E[|random_k ∩ true_top_n|] = k * n_true_top / n_total (hypergeometric mean);
    recall = 该期望 / n_true_top = k / n_total。
    """
    if n_true_top <= 0 or n_total <= 0:
        return 0.0
    k = min(k, n_total)
    expected_hits = k * n_true_top / n_total
    return expected_hits / n_true_top


def recall_at_budget(
    labels: Sequence[str],
    true_values: Sequence[float],
    pred_values: Sequence[float],
    budget_k: int,
    target_top_n: int,
) -> dict:
    """一般化版本: 粗筛预算 budget_k 和"要保住的真值 top-N 目标集合"大小可以不同
    (topk_screen_eval 是 budget_k == target_top_n 的特例)。

    典型用法: "只花 5 个真测名额(budget_k=5), 能保住全部 4 个 AP 最优候选
    (target_top_n=4)吗?" —— 这正是搜索器首轮粗筛决策要回答的问题。
    """
    n = len(labels)
    budget_k = min(budget_k, n)
    target_top_n = min(target_top_n, n)
    true_rank = _rank_desc(labels, true_values)
    pred_rank = _rank_desc(labels, pred_values)
    true_top_n = true_rank[:target_top_n]
    pred_top_k = pred_rank[:budget_k]
    retained = [label for label in true_top_n if label in pred_top_k]
    recall = len(retained) / len(true_top_n) if true_top_n else 0.0
    random_recall = expected_random_recall(n, target_top_n, budget_k)
    return {
        "budget_k": budget_k,
        "target_top_n": target_top_n,
        "true_top_n": true_top_n,
        "predicted_top_k": pred_top_k,
        "retained": retained,
        "recall": round(recall, 4),
        "random_expected_recall": round(random_recall, 4),
    }


def topk_screen_eval(
    labels: Sequence[str],
    true_values: Sequence[float],
    pred_values: Sequence[float],
    k: int,
) -> ScreenEvalAtK:
    """在预算 k 下评估"用 pred_values 排序做首轮粗筛"相对真值 top-k 的精度/召回,
    并给出同预算下随机筛选的期望召回作对照。
    """
    n = len(labels)
    k = min(k, n)
    true_rank = _rank_desc(labels, true_values)
    pred_rank = _rank_desc(labels, pred_values)
    true_top_k = true_rank[:k]
    pred_top_k = pred_rank[:k]
    retained = [label for label in pred_top_k if label in true_top_k]
    precision = len(retained) / k if k else 0.0
    recall = len(retained) / len(true_top_k) if true_top_k else 0.0
    random_recall = expected_random_recall(n, len(true_top_k), k)
    return ScreenEvalAtK(
        k=k,
        true_top_k=true_top_k,
        predicted_top_k=pred_top_k,
        retained=retained,
        precision_at_k=round(precision, 4),
        recall_of_true_top_k=round(recall, 4),
        random_expected_recall=round(random_recall, 4),
    )


def screen_sweep(
    labels: Sequence[str],
    true_values: Sequence[float],
    pred_values: Sequence[float],
    ks: Sequence[int] | None = None,
) -> list[dict]:
    """对一组预算 k 扫描 topk_screen_eval, 返回 dict 列表(便于 json.dump)。"""
    n = len(labels)
    if ks is None:
        ks = list(range(1, n + 1))
    return [topk_screen_eval(labels, true_values, pred_values, k).to_dict() for k in ks]


def min_k_for_full_recall(
    labels: Sequence[str],
    true_values: Sequence[float],
    pred_values: Sequence[float],
    target_top_n: int,
) -> int | None:
    """找到能 100% 召回"真值 top-target_top_n"集合所需的最小粗筛预算 k(<=n)。

    返回 None 代表即使 k=n(全量真测)也没有意义的粗筛(target_top_n<=0 等退化情形)。
    """
    n = len(labels)
    if target_top_n <= 0 or target_top_n > n:
        return None
    true_rank = _rank_desc(labels, true_values)
    target_set = set(true_rank[:target_top_n])
    pred_rank = _rank_desc(labels, pred_values)
    for k in range(target_top_n, n + 1):
        if target_set.issubset(set(pred_rank[:k])):
            return k
    return n
