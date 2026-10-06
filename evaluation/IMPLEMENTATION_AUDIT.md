# FindX Retrieval Implementation Audit

This audit records the implementation verified from source before the evaluation upgrade. It intentionally distinguishes production behaviour from benchmark-only baselines.

| Area | Verified implementation |
|---|---|
| Full text embedding | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| Full text dimension | 384 |
| Full visual/text embedding | OpenCLIP `ViT-B-32` / `laion2b_s34b_b79k` |
| Full image dimension | 512 |
| Vector normalization | SentenceTransformer and OpenCLIP outputs are L2-normalized |
| FAISS index | `IndexHNSWFlat` |
| HNSW defaults | M=32, efConstruction=80, efSearch=64 |
| HNSW metric | inner product; cosine-equivalent because vectors are normalized |
| Lexical retrieval | custom BM25, k1=1.5, b=0.75 |
| Lexical tokenization | lowercase Unicode regex `[^\W]`/word-hyphen tokens via `r"[\w-]+"` |
| Hybrid fusion | Reciprocal Rank Fusion with k0=60 |
| Reranker | deterministic feature reranker over normalized lexical/dense/fusion signals, structured attributes, category bonus and typo-confidence term |
| Multimodal fusion | weighted visual similarity + reranked text relevance + OpenCLIP text/image compatibility + attribute match; defaults 0.50/0.20/0.20/0.10 with active-weight renormalization |
| Language handling | multilingual SentenceTransformer plus bounded Hindi/Hinglish normalization and deterministic typo aliases; no Telugu rewrite table |
| Persistence | Full vectors, FAISS/fallback indexes, model metadata and ID mappings under `artifacts/index/full`; invalidated by catalog+image content hash |
| Build pipeline | load catalog → build BM25 → load dense/visual models → validate persisted Full metadata → load or embed/build/persist indexes |
| Query pipeline | normalize/typo-recover → parse attributes → BM25 → dense embedding + ANN → RRF → optional reranker → filters/results |
| Existing tests | API, cache, normalization, search, image/multimodal, metrics, Full-mode contract and opt-in Full integration/persistence |
| Existing evaluation | repository-curated text relevance set and image proxy set with Recall/MRR/nDCG and basic latency summaries |

## Why the evaluation upgrade does not replace the architecture

The audit found no raw BM25/cosine score addition bug: FindX already uses rank-based RRF, avoiding incompatible score-scale addition. The production HNSW defaults and reranker are therefore retained. New exact-search and alternate-HNSW configurations are benchmark references, not silent production replacements.
