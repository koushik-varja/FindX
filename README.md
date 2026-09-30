# FindX

**Multimodal, Multilingual Intent-to-Result Search & Ranking Engine**

FindX is a local, keyless commerce-search system for studying real retrieval and ranking stages: query normalization, typo tolerance, a practical Hindi/Hinglish demo subset, structured constraints, BM25, dense retrieval, hybrid fusion, reranking, image retrieval, multimodal refinement and IR evaluation.

## Two explicit runtime modes

### FULL ML — flagship architecture

`FINDX_MODE=full` uses:

- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` for normalized multilingual product/query embeddings.
- OpenCLIP `ViT-B-32` / `laion2b_s34b_b79k` for product-image, query-image and refinement-text embeddings.
- FAISS HNSW inner-product ANN when `faiss-cpu` is installed; otherwise the repository's random-hyperplane ANN fallback is used and reported honestly.
- Persisted text/image vectors, vector indexes, model metadata and product-ID mapping under `artifacts/index/full/`.
- Weighted multimodal fusion of image similarity, hybrid text relevance, CLIP text-image compatibility and structured attribute matching.

Full mode does **not** silently downgrade to lightweight models when SentenceTransformer/OpenCLIP dependencies are missing.

### LIGHTWEIGHT — robust local fallback/baseline

`FINDX_MODE=lightweight` keeps the complete search/ranking/API/UI path while substituting:

- TF-IDF + Truncated SVD (LSI) dense retrieval.
- Handcrafted RGB/HSV/luminance/gradient visual descriptors.
- Dependency-free random-hyperplane ANN.

It is useful for low-memory machines, debugging and offline demonstrations after setup. The Index Status page always exposes the real active mode and component names.

## Search architecture

Text retrieval combines a custom BM25 index with dense retrieval using Reciprocal Rank Fusion, then applies a feature-based reranker. Parsed price/colour/category/etc. constraints participate in filtering and ranking. Image search uses actual pixels. Full mode adds OpenCLIP cross-modal text-image scoring for image + text refinement.

See `docs/ARCHITECTURE.md`, `docs/SEARCH_PIPELINE.md` and `docs/FULL_MODE.md`.

## Local demo data

The repository includes 82 generated local demo products/images across shoes, apparel, electronics, bags, watches and home products. Core functionality requires no paid API key.

## Windows + Docker Desktop

Copy the example environment file first:

```powershell
Copy-Item .env.example .env
```

Replace the placeholder PostgreSQL/admin values in `.env`.

### Lightweight Docker

Keep:

```text
FINDX_MODE=lightweight
FINDX_INSTALL_FULL=0
```

Then:

```powershell
docker compose up --build
```

### Full ML Docker

Set:

```text
FINDX_MODE=full
FINDX_INSTALL_FULL=1
```

Then:

```powershell
docker compose up --build
```

The first Full build/run requires internet access for Python packages and model weights. The selected multilingual MiniLM and OpenCLIP weight files are roughly 471 MB and 605 MB respectively, before framework/package overhead. Hugging Face/OpenCLIP caches and the generated Full vector indexes are mounted as Docker named volumes, so normal restarts reuse them. Full mode is substantially heavier than Lightweight mode.

Docker runs `alembic upgrade head` before the API. Normal application startup does not call `Base.metadata.create_all()`; that shortcut is restricted to explicit SQLite test configuration.

## URLs

- Frontend: `http://localhost:5173`
- Backend: `http://localhost:8000`
- Swagger: `http://localhost:8000/docs`

## Direct Python index build / evaluation

For Lightweight mode, install `backend/requirements.txt`. For Full mode, install `backend/requirements-full.txt` so SentenceTransformer, OpenCLIP and FAISS can be used when supported:

```powershell
python -m pip install -r backend/requirements-full.txt
python scripts/build_index.py --mode full
python scripts/smoke_test.py --mode full
python scripts/evaluate.py --mode full
```

Index-build commands for either mode are:

```powershell
python scripts/build_index.py --mode lightweight
python scripts/build_index.py --mode full
```

Use `--force` to rebuild valid persisted Full artifacts. A valid Full index is reused on later API startups rather than regenerating all product embeddings.

## Evaluation

```powershell
python scripts/evaluate.py --mode lightweight
python scripts/evaluate.py --mode full
```

The text set contains repository-authored demo relevance labels. The image set uses explicitly documented same-category/style proxies. Metrics are computed, never hardcoded.

## Tests

```powershell
pytest -q
cd frontend
npm ci
npm run typecheck
npm test
npm run build
```

Optional heavyweight integration validation (requires Full dependencies and model weights):

```powershell
$env:FINDX_RUN_FULL_INTEGRATION="1"
pytest -q -m full_ml
```

## Database and cache

PostgreSQL is the database source of truth and Alembic owns schema migration. Redis is an optional query-result cache; search continues when Redis is unavailable. Docker credentials and the admin key come from `.env`, which is ignored by Git.

## Limitations

- Hindi/Hinglish normalization is an intentionally bounded demo vocabulary, not universal Indian-language understanding.
- The included catalog and relevance labels are synthetic/demo-scale.
- Lightweight visual descriptors are not semantic vision models.
- Full mode requires downloading open model weights on first use and needs considerably more memory/disk than Lightweight mode.
- Search quality measurements on this demo set should not be generalized to production commerce traffic.
