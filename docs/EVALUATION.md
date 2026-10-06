# Scientific Retrieval Evaluation

FindX deliberately keeps the production retrieval architecture stable while adding reproducible experiments around it.

## Ground truth

Run `python scripts/build_golden_dataset.py` to build `evaluation/golden_queries.jsonl`. The file combines the repository's pre-existing manually curated text labels and image proxy labels with a small manually curated Telugu slice and multimodal slice. Labels are not produced by the retrieval system itself.

The benchmark is a student/demo benchmark. It is not an industry-standard dataset and is not a user study.

## Run

Lightweight/offline-capable evaluation:

```powershell
python -m pip install -r backend/requirements-eval.txt
python scripts/build_golden_dataset.py
python scripts/benchmark_retrieval.py --mode lightweight
```

Full ML evaluation:

```powershell
python -m pip install -r backend/requirements-eval.txt
python -m pip install -r backend/requirements-full.txt
python scripts/build_golden_dataset.py
python scripts/benchmark_retrieval.py --mode full --output-dir evaluation/results/full
```

The runner writes machine-readable JSON, charts, `evaluation/report.md`, and `evaluation/error_analysis.md`.

## Methods compared

- BM25 only.
- Dense ANN: the production dense embedding plus its configured ANN index.
- Dense exact: the same embedding compared by exact inner product against all normalized vectors.
- Hybrid RRF: production BM25 + dense ANN reciprocal-rank fusion.
- Hybrid + reranker: the production deterministic reranker.
- Supported image-to-image and image+text multimodal queries.

`dense_exact` is an evaluation baseline, not a production architecture replacement.

## HNSW experiment

Full mode evaluates a small, intentional set of M/efConstruction/efSearch configurations against exact top-k neighbours. Query embedding time is excluded from the HNSW retrieval-only latency so the ANN algorithm itself can be compared fairly. The main report also keeps end-to-end query latency separate.

## Synthetic scale test

`python scripts/benchmark_ann_scale.py` optionally benchmarks 10K/50K or user-selected counts of random normalized vectors. Those outputs are explicitly marked **synthetic performance only** and must not be mixed with relevance-quality results.
