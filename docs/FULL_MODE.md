# Full ML Mode

FindX has two explicit runtime modes. `lightweight` is a reliable baseline/fallback; `full` is the flagship ML architecture.

## Models

Full text retrieval uses `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. Product documents and queries are encoded by the same model with L2-normalized embeddings.

Full visual retrieval uses OpenCLIP `ViT-B-32` with pretrained weights `laion2b_s34b_b79k`. Product images and query images use the image encoder. Multimodal text refinements also use the CLIP text encoder, allowing direct text-image compatibility scoring.

FindX never silently replaces these models with the lightweight implementation when `FINDX_MODE=full`. Missing full-mode dependencies or weights cause an explicit startup/build error.

## Vector indexes

Full mode prefers FAISS HNSW with inner-product search over normalized vectors. If `faiss-cpu` is unavailable but the transformer and OpenCLIP models are installed, FindX uses its dependency-free random-hyperplane ANN fallback and reports that exact backend on Index Status. It never reports FAISS unless FAISS is active.

The full index directory is `artifacts/index/full/`. It contains model metadata, product-ID ordering, persisted text/image vectors and the selected vector-index files. The catalog plus local demo images are hashed; a valid persisted index is reused rather than rebuilding embeddings on every API startup.

Build or refresh explicitly:

```bash
python scripts/build_index.py --mode full
python scripts/build_index.py --mode full --force
```

## Multimodal fusion

For an image plus text refinement, Full mode combines four normalized terms:

`score = w_visual*visual + w_text*text + w_crossmodal*clip_text_image + w_attribute*attribute`

Default configured weights are 0.50 / 0.20 / 0.20 / 0.10. Disabled terms are removed and the remaining weights are renormalized. Override them through `FINDX_MM_VISUAL_WEIGHT`, `FINDX_MM_TEXT_WEIGHT`, `FINDX_MM_CROSSMODAL_WEIGHT`, and `FINDX_MM_ATTRIBUTE_WEIGHT`.

The visual term is query-image ↔ product-image cosine similarity. The text term is normalized hybrid/reranked search relevance. The cross-modal term is CLIP refinement-text ↔ product-image similarity. The attribute term rewards parsed commerce constraints such as colour, category and price.

## Model cache on Windows

The first Full-mode build requires internet access to obtain model weights. The multilingual MiniLM PyTorch/Safetensors weight file is roughly 471 MB and the selected OpenCLIP ViT-B/32 safetensors weight file is roughly 605 MB; Python packages and framework dependencies add additional disk usage. Hugging Face and OpenCLIP reuse their caches afterwards.

PowerShell example:

```powershell
$env:SENTENCE_TRANSFORMERS_HOME="$env:USERPROFILE\.cache\findx\sentence-transformers"
$env:OPENCLIP_CACHE_DIR="$env:USERPROFILE\.cache\findx\openclip"
python scripts/build_index.py --mode full
```

Docker uses named cache volumes so normal container restarts do not redownload model weights. Rebuilding after intentionally deleting those volumes requires downloading them again.
