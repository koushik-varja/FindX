# Evaluation

Text retrieval is evaluated against the repository-authored labelled query set in `data/demo/eval_queries.jsonl`. The labels are a small deterministic demo set, not a user study. Metrics are Recall@K, MRR, NDCG@K and measured p50/p95 query latency.

Run each mode separately:

```bash
python scripts/evaluate.py --mode lightweight
python scripts/evaluate.py --mode full
```

The script compares BM25, Dense, Hybrid and Reranked retrieval for the selected runtime and writes timestamped JSON plus `latest-<mode>.json` under `artifacts/evaluation/`.

Image retrieval uses `data/demo/image_eval.jsonl`, a small repository-authored set of same-category/style-proxy pairs (for example black mesh running shoes or black smartwatches). The query product itself is removed before Recall@5 is calculated. This is only a demo consistency metric; it is not a claim of visual-search accuracy on an external benchmark.

Full-mode results must be generated on a machine where the SentenceTransformer/OpenCLIP dependencies and model weights are actually available. If Full mode performs worse on this tiny synthetic catalog, the JSON should be reported as measured rather than rewritten.
