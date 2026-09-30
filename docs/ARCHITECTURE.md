# Architecture

## Flagship FULL ML path

```text
Text query ──> normalization / typo recovery / attribute extraction
                 │
                 ├──> BM25 lexical retrieval ───────────────┐
                 │                                          │
                 └──> Multilingual SentenceTransformer ─> ANN/FAISS
                                                            │
                                      Reciprocal Rank Fusion ┤
                                                            v
                                               feature reranker
                                                            │
                                                            v
                                                    final results

Image ──> OpenCLIP image embedding ──> image ANN/FAISS ──────┐
Text refinement ──> OpenCLIP text embedding ────────────────┤
Structured constraints + hybrid text relevance ─────────────┤
                                                            v
                                              weighted multimodal fusion
```

The public demo uses PostgreSQL for persistent catalog/search-event concepts, Alembic for schema migrations, Redis as a non-critical query-result cache, FastAPI for the API and React/TypeScript/Vite for the UI.

## Lightweight fallback

`FINDX_MODE=lightweight` keeps the same query normalization, BM25, hybrid fusion, reranker, APIs, Search Lab and evaluation pipeline, but substitutes:

- TF-IDF + Truncated SVD (LSI) for transformer dense embeddings.
- Handcrafted RGB/HSV/luminance/gradient descriptors in place of OpenCLIP visual embeddings.
- Dependency-free random-hyperplane ANN for vector search.

The UI's Index Status page reports the real runtime mode and active model/index names, so the fallback cannot be mistaken for the flagship ML stack.
