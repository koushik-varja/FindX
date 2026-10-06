from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import tempfile
import time
from typing import Callable

import numpy as np
from PIL import Image

from evaluation.dataset import GoldenQuery, load_golden_queries
from evaluation.metrics import mrr, ndcg_at_k, overlap_recall_at_k, precision_at_k, recall_at_k
from ml.ann import ExactIPIndex, FaissHNSWIndex, HNSWConfig, faiss_available
from ml.attributes import parse_attributes, passes_hard_filters
from ml.engine import SearchEngine


METHODS = ("bm25", "dense_ann", "dense_exact", "hybrid", "reranked")


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if not 0 <= p <= 1:
        raise ValueError("percentile p must be between 0 and 1")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * p
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def latency_summary(values: list[float]) -> dict[str, float | int | None]:
    return {
        "samples": len(values),
        "p50_ms": round(percentile(values, 0.50), 4),
        "p95_ms": round(percentile(values, 0.95), 4),
        "p99_ms": round(percentile(values, 0.99), 4) if len(values) >= 50 else None,
        "mean_ms": round(statistics.fmean(values), 4) if values else 0.0,
    }


def _rss_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except Exception:
        return None


def _directory_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _metrics(ranked: list[str], query: GoldenQuery, k: int) -> dict[str, float]:
    rel = query.graded_relevance
    return {
        f"precision_at_{k}": precision_at_k(ranked, rel, k),
        f"recall_at_{k}": recall_at_k(ranked, rel, k),
        "mrr": mrr(ranked, rel),
        f"ndcg_at_{k}": ndcg_at_k(ranked, rel, k),
    }


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {key: round(statistics.fmean(row[key] for row in rows), 4) for key in keys}


def _exact_dense(engine: SearchEngine, query: GoldenQuery, k: int) -> tuple[list[str], dict[str, float]]:
    total_started = time.perf_counter()
    started = time.perf_counter()
    normalized = engine.normalizer.normalize(query.query)
    attrs = parse_attributes(normalized.corrected)
    preprocessing = (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    vector = engine.dense.encode_query(normalized.corrected)
    embedding = (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    rows = engine.dense.exact_search_vector(vector, len(engine.products))
    retrieval = (time.perf_counter() - started) * 1000
    ranked = [
        engine.products[idx]["id"]
        for idx, _score in rows
        if passes_hard_filters(engine.products[idx], attrs)
    ][:k]
    total = (time.perf_counter() - total_started) * 1000
    return ranked, {
        "preprocessing": preprocessing,
        "dense_embedding": embedding,
        "dense_retrieval": retrieval,
        "total": total,
    }


def _engine_method(engine: SearchEngine, method: str, query: GoldenQuery, k: int):
    mode = "dense" if method == "dense_ann" else method
    payload = engine.search_text(query.query, k=k, mode=mode, debug=True)
    return [row["id"] for row in payload["results"]], payload.get("timings_ms", {}), payload


def _evaluate_text(engine: SearchEngine, queries: list[GoldenQuery], k: int, latency_repeats: int) -> dict:
    method_metrics: dict[str, list[dict[str, float]]] = {method: [] for method in METHODS}
    method_latencies: dict[str, list[float]] = {method: [] for method in METHODS}
    stage_latencies: dict[str, dict[str, list[float]]] = {method: defaultdict(list) for method in METHODS}
    per_query: list[dict] = []

    for query in queries:
        query_row = {
            "query_id": query.query_id,
            "query": query.query,
            "language": query.language,
            "query_type": query.query_type,
            "relevant_document_ids": list(query.relevant_document_ids),
            "methods": {},
        }
        for method in METHODS:
            first_ranked: list[str] | None = None
            first_metrics: dict[str, float] | None = None
            for _repeat in range(max(1, latency_repeats)):
                if method == "dense_exact":
                    ranked, timings = _exact_dense(engine, query, k)
                    payload = None
                else:
                    ranked, timings, payload = _engine_method(engine, method, query, k)
                method_latencies[method].append(float(timings.get("total", 0.0)))
                for stage, value in timings.items():
                    stage_latencies[method][stage].append(float(value))
                if first_ranked is None:
                    first_ranked = ranked
                    first_metrics = _metrics(ranked, query, k)
                    method_metrics[method].append(first_metrics)
                    debug_results = []
                    if payload is not None:
                        for result in payload.get("results", []):
                            debug = result.get("debug", {})
                            debug_results.append(
                                {
                                    "id": result["id"],
                                    "final_rank": result["rank"],
                                    "bm25_rank": debug.get("bm25_rank"),
                                    "dense_rank": debug.get("dense_rank"),
                                    "hybrid_rank": debug.get("hybrid_rank"),
                                    "lexical_score": debug.get("lexical_score"),
                                    "semantic_score": debug.get("semantic_score"),
                                    "fusion_score": debug.get("fusion_score"),
                                    "rerank_score": debug.get("rerank_score"),
                                }
                            )
                    query_row["methods"][method] = {
                        "ranked_ids": ranked,
                        "metrics": first_metrics,
                        "debug_results": debug_results,
                    }
        per_query.append(query_row)

    aggregate = {}
    for method in METHODS:
        aggregate[method] = {
            **_mean_metrics(method_metrics[method]),
            "latency": latency_summary(method_latencies[method]),
            "stage_latency": {
                stage: latency_summary(values)
                for stage, values in sorted(stage_latencies[method].items())
            },
        }

    by_language: dict[str, dict[str, dict[str, float]]] = {}
    for language in sorted({query.language for query in queries}):
        language_rows = [row for row in per_query if row["language"] == language]
        by_language[language] = {}
        for method in METHODS:
            by_language[language][method] = _mean_metrics(
                [row["methods"][method]["metrics"] for row in language_rows]
            )

    by_query_type: dict[str, dict[str, dict[str, float]]] = {}
    for query_type in sorted({query.query_type for query in queries}):
        type_rows = [row for row in per_query if row["query_type"] == query_type]
        by_query_type[query_type] = {}
        for method in METHODS:
            by_query_type[query_type][method] = _mean_metrics(
                [row["methods"][method]["metrics"] for row in type_rows]
            )

    return {
        "aggregate": aggregate,
        "by_language": by_language,
        "by_query_type": by_query_type,
        "queries": per_query,
    }


def _evaluate_multimodal(engine: SearchEngine, queries: list[GoldenQuery], k: int) -> dict:
    rows: list[dict] = []
    latencies: list[float] = []
    for query in queries:
        product = engine.by_id[query.query_image_id]
        with Image.open(engine.root / product["image_path"]) as source:
            image = source.convert("RGB")
            if query.modality == "image":
                payload = engine.search_image(image, k=min(len(engine.products), k + 1), debug=True)
                ranked = [
                    result["id"]
                    for result in payload["results"]
                    if result["id"] != query.query_image_id
                ][:k]
            else:
                payload = engine.search_multimodal(image, query.query, k=k, debug=True)
                ranked = [result["id"] for result in payload["results"]]
        latencies.append(float(payload["latency_ms"]))
        rows.append(
            {
                "query_id": query.query_id,
                "modality": query.modality,
                "query": query.query,
                "query_image_id": query.query_image_id,
                "ranked_ids": ranked,
                "metrics": _metrics(ranked, query, k),
                "timings_ms": payload.get("timings_ms", {}),
                "fusion_weights": payload.get("fusion_weights"),
            }
        )
    return {
        "aggregate": _mean_metrics([row["metrics"] for row in rows]),
        "latency": latency_summary(latencies),
        "queries": rows,
    }


def _hnsw_experiment(engine: SearchEngine, queries: list[GoldenQuery], k: int) -> dict:
    if not faiss_available():
        return {"available": False, "reason": "faiss-cpu is not installed"}
    vectors = np.asarray(engine.dense.vectors, dtype=np.float32)
    query_vectors: list[np.ndarray] = []
    for query in queries:
        normalized = engine.normalizer.normalize(query.query)
        query_vectors.append(engine.dense.encode_query(normalized.corrected))

    exact = ExactIPIndex().fit(vectors)
    exact_ids: list[list[str]] = []
    exact_latencies: list[float] = []
    for vector in query_vectors:
        started = time.perf_counter()
        exact_rows, _ = exact.search(vector, k)
        exact_latencies.append((time.perf_counter() - started) * 1000)
        exact_ids.append([engine.products[idx]["id"] for idx, _score in exact_rows])

    configs = [
        HNSWConfig(16, 16, 40),
        HNSWConfig(16, 64, 80),
        HNSWConfig(32, 16, 80),
        HNSWConfig(32, 64, 80),
        HNSWConfig(32, 128, 80),
        HNSWConfig(48, 128, 160),
    ]
    results = []
    for config in configs:
        started = time.perf_counter()
        index = FaissHNSWIndex(
            m=config.m,
            ef_search=config.ef_search,
            ef_construction=config.ef_construction,
        ).fit(vectors)
        build_ms = (time.perf_counter() - started) * 1000
        recalls: list[float] = []
        latencies: list[float] = []
        with tempfile.TemporaryDirectory(prefix="findx-hnsw-") as temp_dir:
            index_path = Path(temp_dir) / "index.faiss"
            index.save(index_path)
            index_bytes = index_path.stat().st_size
        for vector, exact_ranked in zip(query_vectors, exact_ids):
            started = time.perf_counter()
            approx_rows, _ = index.search(vector, k)
            latencies.append((time.perf_counter() - started) * 1000)
            approx_ranked = [engine.products[idx]["id"] for idx, _score in approx_rows]
            recalls.append(overlap_recall_at_k(approx_ranked, exact_ranked, k))
        results.append(
            {
                "m": config.m,
                "ef_construction": config.ef_construction,
                "ef_search": config.ef_search,
                "ann_recall_at_k_vs_exact": round(statistics.fmean(recalls), 4),
                "retrieval_latency": latency_summary(latencies),
                "build_ms": round(build_ms, 4),
                "index_bytes": index_bytes,
            }
        )
    return {
        "available": True,
        "k": k,
        "query_count": len(query_vectors),
        "exact_reference": {
            "backend": "exact inner product on normalized vectors",
            "retrieval_latency": latency_summary(exact_latencies),
            "vector_bytes": int(vectors.nbytes),
        },
        "configs": results,
    }


def _qps(engine: SearchEngine, queries: list[GoldenQuery], concurrency: int, limit: int = 24) -> dict:
    selected = queries[: min(limit, len(queries))]
    if not selected:
        return {"concurrency": concurrency, "queries": 0, "qps": 0.0}

    def run(query: GoldenQuery) -> None:
        engine.search_text(query.query, k=10, mode="reranked", debug=False)

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        list(pool.map(run, selected))
    wall = time.perf_counter() - started
    return {
        "concurrency": concurrency,
        "queries": len(selected),
        "wall_seconds": round(wall, 4),
        "qps": round(len(selected) / wall, 4) if wall > 0 else 0.0,
        "note": "Controlled local-process throughput; includes preprocessing, embedding, retrieval and reranking.",
    }


def run_benchmark(
    root: Path,
    *,
    mode: str,
    dataset_path: Path,
    k: int = 10,
    latency_repeats: int = 1,
    concurrency: int = 4,
) -> dict:
    memory_before = _rss_bytes()
    engine_started = time.perf_counter()
    engine = SearchEngine(root, mode=mode)
    engine_wall = time.perf_counter() - engine_started
    memory_after_engine = _rss_bytes()

    all_queries = load_golden_queries(dataset_path)
    text_queries = [query for query in all_queries if query.modality == "text"]
    image_queries = [query for query in all_queries if query.modality == "image"]
    multimodal_queries = [query for query in all_queries if query.modality == "multimodal"]

    text = _evaluate_text(engine, text_queries, k, latency_repeats)
    image = _evaluate_multimodal(engine, image_queries, min(k, 5)) if image_queries else None
    multimodal = _evaluate_multimodal(engine, multimodal_queries, min(k, 5)) if multimodal_queries else None
    hnsw = _hnsw_experiment(engine, text_queries, k) if mode == "full" else {
        "available": False,
        "reason": "HNSW experiment is meaningful for the Full FAISS path; run with --mode full.",
    }
    throughput = _qps(engine, text_queries, max(1, concurrency))
    memory_after_benchmark = _rss_bytes()

    index_dir = root / "artifacts/index/full"
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runtime_mode": mode,
        "dataset": {
            "path": str(dataset_path.relative_to(root)),
            "total_queries": len(all_queries),
            "text_queries": len(text_queries),
            "image_queries": len(image_queries),
            "multimodal_queries": len(multimodal_queries),
            "ground_truth": "Manually/repository-curated relevance labels; never generated by the system under evaluation.",
        },
        "engine": engine.status(),
        "performance": {
            "engine_build_or_load_wall_seconds": round(engine_wall, 4),
            "full_index_directory_bytes": _directory_bytes(index_dir) if mode == "full" else 0,
            "rss_bytes_before_engine": memory_before,
            "rss_bytes_after_engine": memory_after_engine,
            "rss_bytes_after_benchmark": memory_after_benchmark,
            "throughput": throughput,
        },
        "text": text,
        "image": image,
        "multimodal": multimodal,
        "hnsw": hnsw,
        "limitations": [
            "The relevance corpus is a small, repository-authored demo benchmark, not an industry benchmark or user study.",
            "Latency and QPS depend on hardware, process state, model cache state and background load.",
            "At 82 products exact vector search is already cheap; HNSW is evaluated to study ANN behaviour, not because this demo requires ANN for speed.",
            "Telugu queries measure the multilingual model/corpus directly; FindX has no explicit Telugu normalization table.",
        ],
    }
