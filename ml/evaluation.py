from __future__ import annotations

import json
import statistics
from pathlib import Path

from PIL import Image

from evaluation.metrics import mrr, ndcg_at_k, precision_at_k, recall_at_k


def _percentile(values, percentile):
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


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
        precisions, recalls, reciprocal_ranks, ndcgs, latencies = [], [], [], [], []
        for query in rows:
            payload = engine.search_text(query["query"], k=k, mode=mode)
            ranked = [result["id"] for result in payload["results"]]
            precisions.append(precision_at_k(ranked, query["relevance"], k))
            recalls.append(recall_at_k(ranked, query["relevance"], k))
            reciprocal_ranks.append(mrr(ranked, query["relevance"]))
            ndcgs.append(ndcg_at_k(ranked, query["relevance"], k))
            latencies.append(payload["latency_ms"])
        output["modes"][mode] = {
            "precision_at_k": round(statistics.fmean(precisions), 4),
            "recall_at_k": round(statistics.fmean(recalls), 4),
            "mrr": round(statistics.fmean(reciprocal_ranks), 4),
            "ndcg_at_k": round(statistics.fmean(ndcgs), 4),
            "latency_ms_p50": round(_percentile(latencies, 0.50), 3),
            "latency_ms_p95": round(_percentile(latencies, 0.95), 3),
            "latency_ms_p99": round(_percentile(latencies, 0.99), 3) if len(latencies) >= 50 else None,
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
