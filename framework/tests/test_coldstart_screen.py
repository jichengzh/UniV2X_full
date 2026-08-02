from __future__ import annotations

from framework.coldstart_screen import (
    expected_random_recall,
    min_k_for_full_recall,
    screen_sweep,
    topk_screen_eval,
)


def test_perfect_predictor_full_precision_recall():
    labels = ["a", "b", "c", "d"]
    true_values = [4.0, 3.0, 2.0, 1.0]
    pred_values = [4.0, 3.0, 2.0, 1.0]
    ev = topk_screen_eval(labels, true_values, pred_values, k=2)
    assert ev.true_top_k == ["a", "b"]
    assert ev.predicted_top_k == ["a", "b"]
    assert ev.precision_at_k == 1.0
    assert ev.recall_of_true_top_k == 1.0


def test_anti_correlated_predictor_zero_overlap_at_half_budget():
    labels = ["a", "b", "c", "d"]
    true_values = [4.0, 3.0, 2.0, 1.0]
    pred_values = [1.0, 2.0, 3.0, 4.0]  # exactly inverted
    ev = topk_screen_eval(labels, true_values, pred_values, k=2)
    assert ev.retained == []
    assert ev.precision_at_k == 0.0


def test_expected_random_recall_matches_k_over_n():
    # recall = k / n_total when true_top == n_true_top (algebraic identity)
    assert expected_random_recall(n_total=12, n_true_top=4, k=6) == 6 / 12
    assert expected_random_recall(n_total=12, n_true_top=4, k=12) == 1.0
    assert expected_random_recall(n_total=0, n_true_top=4, k=6) == 0.0


def test_screen_sweep_covers_all_k_by_default():
    labels = ["a", "b", "c"]
    true_values = [3.0, 2.0, 1.0]
    pred_values = [3.0, 1.0, 2.0]
    rows = screen_sweep(labels, true_values, pred_values)
    assert [r["k"] for r in rows] == [1, 2, 3]
    assert rows[-1]["recall_of_true_top_k"] == 1.0  # k=n always fully recalls


def test_min_k_for_full_recall_perfect_predictor():
    labels = ["a", "b", "c", "d"]
    true_values = [4.0, 3.0, 2.0, 1.0]
    pred_values = [4.0, 3.0, 2.0, 1.0]
    assert min_k_for_full_recall(labels, true_values, pred_values, target_top_n=2) == 2


def test_min_k_for_full_recall_degenerate_target():
    labels = ["a", "b"]
    true_values = [1.0, 2.0]
    pred_values = [1.0, 2.0]
    assert min_k_for_full_recall(labels, true_values, pred_values, target_top_n=0) is None
    assert min_k_for_full_recall(labels, true_values, pred_values, target_top_n=5) is None
