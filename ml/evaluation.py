from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

from PIL import Image


def recall_at_k(ranked, relevant, k):
    relevant_ids = {product_id for product_id, grade in relevant.items() if grade > 0}
    return len(set(ranked[:k]) & relevant_ids) / max(1, len(relevant_ids))


def mrr(ranked, relevant):
    relevant_ids = {product_id for product_id, grade in relevant.items() if grade > 0}
    for index, product_id in enumerate(ranked, 1):
        if product_id in relevant_ids:
            return 1 / index
    return 0.0


def ndcg_at_k(ranked, relevant, k):
    def dcg(sequence):
        return sum((2 ** relevant.get(product_id, 0) - 1) / math.log2(index + 2) for index, product_id in enumerate(sequence[:k]))

    ideal = [product_id for product_id, _grade in sorted(relevant.items(), key=lambda item: -item[1])]
    denominator = dcg(ideal)
    return dcg(ranked) / denominator if denominator else 0.0


def _percentile(values, percentile):
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(percentile * len(ordered)) - 1))
    return ordered[index]


def evaluate(engine, eval_path: Path, modes=("bm25", "dense", "hybrid", "reranked"), k=10):
    rows = [json.loads(line) for line in eval_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    output = {
        "label_origin": "curated deterministic demo relevance labels authored for this repository; not a user study",
        "queries": len(rows),
        "k": k,
        "runtime_mode": engine.mode,
        "modes": {},
    }
    for mode in modes:
        recalls, reciprocal_ranks, ndcgs, latencies = [], [], [], []
        for query in rows:
            payload = engine.search_text(query["query"], k=k, mode=mode)
            ranked = [result["id"] for result in payload["results"]]
            recalls.append(recall_at_k(ranked, query["relevance"], k))
            reciprocal_ranks.append(mrr(ranked, query["relevance"]))
            ndcgs.append(ndcg_at_k(ranked, query["relevance"], k))
            latencies.append(payload["latency_ms"])
        output["modes"][mode] = {
            "recall_at_k": round(statistics.fmean(recalls), 4),
            "mrr": round(statistics.fmean(reciprocal_ranks), 4),
            "ndcg_at_k": round(statistics.fmean(ndcgs), 4),
            "latency_ms_p50": round(statistics.median(latencies), 3),
            "latency_ms_p95": round(_percentile(latencies, 0.95), 3),
        }
    return output


def evaluate_images(engine, eval_path: Path, k: int = 5):
    rows = [json.loads(line) for line in eval_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    recalls = []
    for row in rows:
        product = engine.by_id[row["query_product_id"]]
        with Image.open(engine.root / product["image_path"]) as image:
            payload = engine.search_image(image.convert("RGB"), k=min(len(engine.products), k + 1))
        ranked = [result["id"] for result in payload["results"] if result["id"] != row["query_product_id"]]
        relevant = {product_id: 1 for product_id in row["relevant_product_ids"]}
        recalls.append(recall_at_k(ranked, relevant, k))
    return {
        "metric": f"Recall@{k}",
        "value": round(statistics.fmean(recalls), 4) if recalls else 0.0,
        "queries": len(rows),
        "label_origin": "repository-authored same-category/style-proxy product pairs; query product itself is excluded",
    }
