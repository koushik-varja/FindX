# FindX

**Multimodal, Multilingual Intent-to-Result Search & Ranking Engine**

FindX is a focused commerce-search project for studying real retrieval and ranking behaviour: query normalization, typo tolerance, bounded Hindi/Hinglish handling, structured constraints, BM25, multilingual dense retrieval, FAISS HNSW ANN, Reciprocal Rank Fusion, deterministic reranking, OpenCLIP image retrieval, multimodal refinement and reproducible IR evaluation.

The project deliberately does **not** add RAG, agents, an LLM chatbot, another vector database or unrelated infrastructure. The evaluation upgrade measures the architecture already present instead of replacing it for novelty.

## Architecture

```text
Query
  ↓
Preprocessing / typo recovery / structured constraints
  ├───────────────────────────┐
  ↓                           ↓
BM25                        Embedding
  ↓                           ↓
Lexical candidates          ANN candidates
  └─────────────┬─────────────┘
                ↓
        Reciprocal Rank Fusion
                ↓
          Feature reranker
                ↓
              Results
```

Full multimodal refinement extends this with an OpenCLIP image embedding, OpenCLIP refinement-text/image compatibility, hybrid text relevance and structured-attribute matching.

See `docs/ARCHITECTURE.md`, `docs/SEARCH_PIPELINE.md`, `docs/FULL_MODE.md`, `docs/EVALUATION.md`, and `evaluation/IMPLEMENTATION_AUDIT.md`.

## Verified implementation

### FULL ML — flagship path

`FINDX_MODE=full` uses:

- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` for L2-normalized 384-dimensional text embeddings.
- OpenCLIP `ViT-B-32` / `laion2b_s34b_b79k` for L2-normalized 512-dimensional image and CLIP-text embeddings.
- FAISS `IndexHNSWFlat` with inner-product search. Production defaults remain `M=32`, `efConstruction=80`, `efSearch=64`.
- Inner product on normalized vectors, which is cosine-equivalent.
- Persisted vectors, FAISS indexes, model metadata and product-ID mappings under `artifacts/index/full/`.
- Catalog+image content hashing to reject stale persisted indexes.

Full mode does not silently downgrade to lightweight models when required ML dependencies are missing.

### LIGHTWEIGHT — local baseline/fallback

`FINDX_MODE=lightweight` keeps the same API/search stages while substituting:

- TF-IDF + Truncated SVD (LSI) dense retrieval.
- Handcrafted visual descriptors.
- Dependency-free random-hyperplane ANN.

The Index Status page reports the actual active models/index backends so the fallback cannot be mistaken for Full ML.

## Why Hybrid Search?

FindX uses **Reciprocal Rank Fusion (RRF)** rather than adding raw BM25 and cosine scores, because those score scales are not directly comparable.

In the previously verified Full run over the repository's 60 curated text queries:

| Method | Recall@10 | MRR | nDCG@10 |
|---|---:|---:|---:|
| BM25 | 0.9778 | 0.9639 | 0.9293 |
| Dense ANN | 0.9778 | 0.9297 | 0.8922 |
| Hybrid RRF | **0.9889** | 0.9750 | 0.9381 |
| Hybrid + reranker | 0.9833 | **0.9917** | **0.9535** |

That run shows a tradeoff rather than a universal winner: hybrid improved Recall@10, while reranking improved top-order quality (MRR/nDCG) but slightly reduced Recall@10. The upgraded evaluation framework regenerates current measurements and additionally compares exact dense search, HNSW settings, language slices and stage-level latency. Do not quote the table as production or industry-benchmark performance.

## Evaluation

The reproducible framework lives under `evaluation/`.

```powershell
python -m pip install -r backend/requirements-eval.txt
python scripts/build_golden_dataset.py
python scripts/benchmark_retrieval.py --mode lightweight
```

For the real SentenceTransformer/OpenCLIP/FAISS path:

```powershell
python -m pip install -r backend/requirements-eval.txt
python -m pip install -r backend/requirements-full.txt
python scripts/build_golden_dataset.py
python scripts/benchmark_retrieval.py --mode full --output-dir evaluation/results/full
```

The benchmark compares:

- BM25 only.
- Dense ANN (production dense retrieval).
- Dense exact reference search using the same embeddings.
- BM25 + dense ANN RRF hybrid.
- Hybrid + reranker.
- Supported image-to-image retrieval.
- Supported image + text multimodal refinement.

It computes Precision@K, Recall@K, MRR and graded nDCG@K; p50/p95/p99 latency where sample size permits; per-stage timings; controlled-concurrency QPS; observed RSS; persisted index size; and HNSW recall/latency/build-size experiments against exact neighbours.

Outputs include:

- `evaluation/golden_queries.jsonl`
- `evaluation/results/<mode>/results.json`
- `evaluation/report.md`
- `evaluation/error_analysis.md`
- `evaluation/results/<mode>/charts/*.png`

Ground truth is manually/repository curated. The search system never generates its own relevance labels and evaluates against them.

## Failure Cases

The benchmark automatically records difficult cases where, for example:

- BM25 beats dense retrieval on exact terminology.
- Dense retrieval beats BM25 on paraphrastic queries.
- reranking lowers per-query nDCG.
- ANN differs from exact dense neighbours.
- Hindi/Hinglish/Telugu slices miss labelled products.
- image retrieval confuses same-category/style proxies.

See [`evaluation/error_analysis.md`](evaluation/error_analysis.md) after running the benchmark. The report intentionally preserves regressions instead of cherry-picking only successes.

## Multilingual evaluation

The existing normalizer has a bounded Hindi/Hinglish rewrite vocabulary and typo table. Full dense retrieval is multilingual. Telugu has **no explicit rewrite dictionary**; Telugu benchmark queries are included specifically to measure what the multilingual model and corpus can do without pretending that Telugu preprocessing exists.

Metrics are reported by language (`en`, `hi`, `hinglish`, `te`) rather than only as one aggregate.

## Exact vs HNSW

At the included 82-product scale, exact vector search is already cheap. HNSW is therefore not claimed to be necessary for this tiny demo. The Full benchmark uses exact inner product as a reference and evaluates a small, sensible set of HNSW configurations to show the real recall/latency/build-size tradeoff.

Optional synthetic ANN-only scale testing is available:

```powershell
python scripts/benchmark_ann_scale.py --sizes 10000 50000
```

Those vectors are explicitly labelled synthetic performance data and are never mixed with relevance-quality results.

## Developer/evaluation panel

Debug/rank/timing internals are opt-in. Normal API responses do not expose them.

Set:

```text
FINDX_EXPOSE_DEBUG=1
```

Then the Search Lab can display BM25 rank, dense rank, hybrid rank, final rank, and stage timings such as preprocessing, dense embedding, ANN retrieval, fusion/ranking and total latency.

Keep `FINDX_EXPOSE_DEBUG=0` for normal use.

## Local demo data

The repository includes 82 generated demo products/images spanning shoes, apparel, electronics, bags, watches and home products. The benchmark labels are demo-scale, curated repository labels—not an industry benchmark or user study.

## Windows + Docker Desktop

```powershell
Copy-Item .env.example .env
```

Replace placeholder PostgreSQL/admin values.

### Lightweight

```text
FINDX_MODE=lightweight
FINDX_INSTALL_FULL=0
```

```powershell
docker compose up --build
```

### Full ML

```text
FINDX_MODE=full
FINDX_INSTALL_FULL=1
FINDX_DEVICE=auto
```

```powershell
docker compose up --build
```

First Full startup requires model downloads. Docker named volumes preserve model caches and persisted Full indexes across normal restarts.

## URLs

- Frontend: `http://localhost:5173`
- Backend: `http://localhost:8000`
- Swagger: `http://localhost:8000/docs`

## Direct index build

```powershell
python scripts/build_index.py --mode lightweight
python scripts/build_index.py --mode full
```

Use `--force` only when intentionally rebuilding a valid Full index.

## Tests and validation

```powershell
python -m pytest -q
python scripts/build_golden_dataset.py
python scripts/benchmark_retrieval.py --mode lightweight
cd frontend
npm ci
npm run typecheck
npm test
npm run build
```

Full integration validation:

```powershell
$env:FINDX_RUN_FULL_INTEGRATION="1"
python -m pytest -q -m full_ml
Remove-Item Env:FINDX_RUN_FULL_INTEGRATION
```

## Database and cache

PostgreSQL is the catalog/search-event database and Alembic owns schema migration. Redis is a non-critical query-result cache; search remains available when Redis is unavailable. Debug requests are intentionally not cached so measurement/debug payloads cannot be confused with normal cached responses.

## Limitations

- The corpus is only 82 demo products; quality numbers do not generalize to production commerce traffic.
- Relevance labels are manually/repository curated, not user judgments from a live search product.
- Hindi/Hinglish normalization is intentionally bounded.
- Telugu has no explicit normalization dictionary and must be judged from its measured slice.
- Full mode requires sizeable model/framework downloads and more RAM/disk.
- Latency/QPS are hardware- and cache-state-dependent; regenerate the report on the machine whose numbers you want to quote.
- Synthetic ANN scale tests measure index mechanics only, never relevance quality.

## License

MIT. See `LICENSE`.
