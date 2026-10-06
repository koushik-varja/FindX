# FindX Retrieval Evaluation Upgrade Manifest

This upgrade keeps the existing retrieval architecture and adds scientific measurement around it.

## Intentionally unchanged

- SentenceTransformer model choice
- OpenCLIP model choice
- BM25 implementation
- RRF hybrid strategy
- deterministic feature reranker design
- FastAPI / PostgreSQL / Redis stack
- React/TypeScript application architecture
- persisted Full-ML index design

## Added evaluation capabilities

- golden query schema and generator
- exact inner-product reference index
- configurable FAISS HNSW experiments
- ANN-vs-exact recall analysis
- stage timing instrumentation
- p50/p95/p99 latency
- controlled concurrency QPS
- process RSS and index-size reporting
- per-language slices
- image and multimodal slices
- automatic error analysis
- automatic Markdown report and charts
- optional synthetic ANN-only scale benchmark, clearly separated from relevance quality

## Ground truth

Labels are manually/repository curated. The system being evaluated does not generate its own relevance labels.

## Important

Any report values must come from a benchmark run on the machine where the report is generated. Previous verified Full-ML numbers are documented as historical results only, not fabricated into new benchmark output.
