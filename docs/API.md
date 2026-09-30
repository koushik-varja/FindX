# API

- `POST /api/search/text` — JSON `{query, k, mode, debug}`; mode is `bm25`, `dense`, `hybrid`, or `reranked`.
- `POST /api/search/lab` — JSON `{query, k}`; returns all four ranking stages.
- `POST /api/search/image` — multipart `file` and `k`.
- `POST /api/search/multimodal` — multipart `file`, optional `text`, and `k`.
- `GET /api/products/{id}` — catalog item.
- `GET /api/search/debug/{query_id}` — in-process details for a recent text query.
- `GET /api/evaluation/latest` — latest saved measured evaluation JSON.
- `GET /api/index/status` — loaded index/model metadata.
- `GET /api/health` — service and cache status.
- `POST /api/admin/reindex` — protected by `X-Admin-Key`.

FastAPI generates the authoritative OpenAPI schema at `/docs` and `/openapi.json`.
