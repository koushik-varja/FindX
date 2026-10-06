from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.benchmark import latency_summary
from evaluation.metrics import overlap_recall_at_k
from ml.ann import ExactIPIndex, FaissHNSWIndex, faiss_available


def normalized_random(rng: np.random.Generator, rows: int, dim: int) -> np.ndarray:
    data = rng.standard_normal((rows, dim), dtype=np.float32)
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    return data / np.maximum(norms, 1e-12)


def main() -> None:
    parser = argparse.ArgumentParser(description="Optional synthetic ANN scale benchmark; NOT a relevance benchmark")
    parser.add_argument("--sizes", nargs="+", type=int, default=[10000, 50000])
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation/results/synthetic-ann-scale.json")
    args = parser.parse_args()
    if not faiss_available():
        raise SystemExit("faiss-cpu is required for the synthetic ANN scale benchmark")

    rng = np.random.default_rng(20261006)
    output = {
        "synthetic_performance_only": True,
        "warning": "Random normalized vectors measure ANN mechanics only. Do not mix these numbers with FindX relevance-quality results.",
        "dimension": args.dim,
        "k": args.k,
        "results": [],
    }
    for size in args.sizes:
        vectors = normalized_random(rng, size, args.dim)
        queries = normalized_random(rng, args.queries, args.dim)
        exact = ExactIPIndex().fit(vectors)
        build_started = time.perf_counter()
        hnsw = FaissHNSWIndex(m=32, ef_search=64, ef_construction=80).fit(vectors)
        build_seconds = time.perf_counter() - build_started
        recalls = []
        exact_latency = []
        ann_latency = []
        for query in queries:
            started = time.perf_counter()
            exact_rows, _ = exact.search(query, args.k)
            exact_latency.append((time.perf_counter() - started) * 1000)
            started = time.perf_counter()
            ann_rows, _ = hnsw.search(query, args.k)
            ann_latency.append((time.perf_counter() - started) * 1000)
            exact_ids = [str(idx) for idx, _ in exact_rows]
            ann_ids = [str(idx) for idx, _ in ann_rows]
            recalls.append(overlap_recall_at_k(ann_ids, exact_ids, args.k))
        with tempfile.TemporaryDirectory(prefix="findx-ann-scale-") as temp_dir:
            path = Path(temp_dir) / "index.faiss"
            hnsw.save(path)
            index_bytes = path.stat().st_size
        output["results"].append(
            {
                "vectors": size,
                "hnsw_build_seconds": round(build_seconds, 4),
                "hnsw_index_bytes": index_bytes,
                "ann_recall_at_k_vs_exact": round(float(np.mean(recalls)), 4),
                "exact_latency": latency_summary(exact_latency),
                "hnsw_latency": latency_summary(ann_latency),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
