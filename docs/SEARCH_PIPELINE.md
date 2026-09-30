# Search Pipeline

1. Normalize case, spacing, punctuation, common commerce shorthand and the supported Hindi/Hinglish demo vocabulary.
2. Apply confidence-aware typo recovery.
3. Extract structured price, colour, brand, category, material and related constraints.
4. Retrieve BM25 candidates.
5. Retrieve dense candidates using the active mode: LSI in `lightweight`, multilingual SentenceTransformer in `full`.
6. Fuse lexical and dense ranks with Reciprocal Rank Fusion rather than concatenation.
7. Apply a deterministic feature-based reranker using lexical/dense/fusion signals plus attribute and constraint features.
8. For image search, encode the query image with the active visual encoder and retrieve nearest indexed product-image vectors.
9. For image + text refinement, combine visual similarity, hybrid text relevance, structured attributes and—only in Full mode—OpenCLIP text/image compatibility.

Full mode persists its vector artifacts under `artifacts/index/full` and validates them against a content hash of the catalog and local images before reuse.
