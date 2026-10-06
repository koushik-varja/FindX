from ml.evaluation import mrr, ndcg_at_k, precision_at_k, recall_at_k


def test_metrics_known_case():
    ranked = ["a", "b", "c"]
    rel = {"b": 3, "c": 1}
    assert precision_at_k(ranked, rel, 2) == 0.5
    assert recall_at_k(ranked, rel, 2) == 0.5
    assert mrr(ranked, rel) == 0.5
    assert 0 < ndcg_at_k(ranked, rel, 3) <= 1
