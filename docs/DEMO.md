# Demo Flow

1. Open Search and confirm the runtime label shown with results.
2. Search `black running shoes under 3000`; inspect parsed colour/category/price constraints.
3. Search `samsoong wirless earbuds`; show the corrected query.
4. Search `3k ke andar black running shoes` and `काले रनिंग जूते`.
5. Open Search Lab and compare BM25, Dense, Hybrid and Reranked rankings from the live backend.
6. Upload a demo product image and run visual search.
7. Add `similar but blue under ₹2500` and run Image + text. In Full mode the result debug path includes OpenCLIP text-image compatibility; in Lightweight mode that term is disabled and the fusion weights are renormalized.
8. Open Evaluation and show the saved metrics for the active mode.
9. Open Index Status and explicitly point out `FULL ML` or `LIGHTWEIGHT`, dense model, visual model, vector-index backend and embedding dimensions.
