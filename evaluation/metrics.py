from __future__ import annotations

import math


def _relevant_ids(relevant: dict[str, int | float]) -> set[str]:
    return {doc_id for doc_id, grade in relevant.items() if float(grade) > 0}


def precision_at_k(ranked: list[str], relevant: dict[str, int | float], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    relevant_ids = _relevant_ids(relevant)
    hits = sum(1 for doc_id in ranked[:k] if doc_id in relevant_ids)
    return hits / k


def recall_at_k(ranked: list[str], relevant: dict[str, int | float], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    relevant_ids = _relevant_ids(relevant)
    if not relevant_ids:
        return 0.0
    return len(set(ranked[:k]) & relevant_ids) / len(relevant_ids)


def mrr(ranked: list[str], relevant: dict[str, int | float]) -> float:
    relevant_ids = _relevant_ids(relevant)
    for rank, doc_id in enumerate(ranked, 1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def dcg_at_k(ranked: list[str], relevant: dict[str, int | float], k: int) -> float:
    score = 0.0
    for index, doc_id in enumerate(ranked[:k]):
        grade = float(relevant.get(doc_id, 0.0))
        score += (2.0**grade - 1.0) / math.log2(index + 2.0)
    return score


def ndcg_at_k(ranked: list[str], relevant: dict[str, int | float], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    ideal_ids = [doc_id for doc_id, _grade in sorted(relevant.items(), key=lambda item: (-float(item[1]), item[0]))]
    ideal = dcg_at_k(ideal_ids, relevant, k)
    return dcg_at_k(ranked, relevant, k) / ideal if ideal > 0 else 0.0


def overlap_recall_at_k(approx: list[str], exact: list[str], k: int) -> float:
    """ANN neighbour recall against an exact top-k reference set."""
    if k <= 0:
        raise ValueError("k must be positive")
    exact_set = set(exact[:k])
    if not exact_set:
        return 0.0
    return len(set(approx[:k]) & exact_set) / len(exact_set)
