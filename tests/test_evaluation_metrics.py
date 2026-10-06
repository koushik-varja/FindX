import math

import pytest

from evaluation.metrics import mrr, ndcg_at_k, overlap_recall_at_k, precision_at_k, recall_at_k


def test_precision_recall_and_mrr_known_case():
    ranked = ["a", "b", "c", "d"]
    relevant = {"b": 3, "c": 1, "z": 2}
    assert precision_at_k(ranked, relevant, 2) == pytest.approx(0.5)
    assert recall_at_k(ranked, relevant, 2) == pytest.approx(1 / 3)
    assert mrr(ranked, relevant) == pytest.approx(0.5)


def test_ndcg_uses_graded_relevance_and_is_one_for_ideal_order():
    relevant = {"a": 3, "b": 2, "c": 1}
    assert ndcg_at_k(["a", "b", "c"], relevant, 3) == pytest.approx(1.0)
    assert ndcg_at_k(["c", "b", "a"], relevant, 3) < 1.0
    assert math.isfinite(ndcg_at_k(["x", "a"], relevant, 2))


def test_ann_overlap_recall():
    assert overlap_recall_at_k(["a", "c", "x"], ["a", "b", "c"], 3) == pytest.approx(2 / 3)


def test_metric_k_must_be_positive():
    with pytest.raises(ValueError):
        precision_at_k([], {}, 0)
    with pytest.raises(ValueError):
        recall_at_k([], {}, 0)
    with pytest.raises(ValueError):
        ndcg_at_k([], {}, 0)
