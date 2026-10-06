from __future__ import annotations

from contextlib import asynccontextmanager
from io import BytesIO
import hashlib
import json
import time

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from sqlalchemy.orm import Session

from ml.engine import SearchEngine
from .cache import RedisCache
from .config import settings
from .db import engine as db_engine, get_db
from .models import Base, SearchEvent, SearchQuery
from .schemas import SearchLabRequest, TextSearchRequest


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.search = SearchEngine(settings.repo_root, settings.search_mode)
    app.state.cache = RedisCache(settings.redis_url)
    await app.state.cache.connect()
    if settings.test_schema_create and settings.database_url.startswith("sqlite"):
        Base.metadata.create_all(bind=db_engine)
    yield
    await app.state.cache.close()


app = FastAPI(title="FindX API", version="1.2.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount(
    "/demo-images",
    StaticFiles(directory=str(settings.repo_root / "data/demo/images")),
    name="demo-images",
)


def _cache_key(req: TextSearchRequest):
    return "search:" + hashlib.sha256(
        f"{settings.search_mode}|{req.query}|{req.k}|{req.mode}".encode()
    ).hexdigest()


def _open_image(raw: bytes):
    if len(raw) > 8 * 1024 * 1024:
        raise HTTPException(413, "Image exceeds 8 MB limit")
    try:
        image = Image.open(BytesIO(raw))
        image.verify()
        image = Image.open(BytesIO(raw))
        return image.convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(415, "Unsupported image format") from exc


def _persist_search(db: Session | None, payload: dict):
    if db is None:
        return
    try:
        query = SearchQuery(
            id=payload["query_id"],
            original_query=payload.get("original_query", "[image]"),
            normalized_query=payload.get("normalized_query", payload.get("corrected_query", "[image]")),
            parsed_attributes=payload.get("parsed_attributes", {}),
            mode=payload["mode"],
            latency_ms=payload["latency_ms"],
        )
        db.add(query)
        for result in payload.get("results", []):
            db.add(
                SearchEvent(
                    query_id=query.id,
                    product_id=result["id"],
                    rank=result["rank"],
                    score=result["score"],
                )
            )
        db.commit()
    except Exception:
        db.rollback()


def _debug_allowed(requested: bool) -> bool:
    return bool(requested and settings.expose_debug)


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "redis_cache": bool(app.state.cache.available),
        "search": app.state.search.status(),
    }


@app.post("/api/search/text")
async def search_text(req: TextSearchRequest, db: Session = Depends(get_db)):
    debug = _debug_allowed(req.debug)
    # Debug payloads contain stage/rank internals and are intentionally not cached.
    if not debug:
        key = _cache_key(req)
        cached = await app.state.cache.get(key)
        if cached:
            cached["cache_hit"] = True
            return cached
    output = app.state.search.search_text(req.query, req.k, req.mode, debug)
    output["cache_hit"] = False
    if not debug:
        await app.state.cache.set(_cache_key(req), output)
    _persist_search(db, output)
    return output


@app.post("/api/search/lab")
async def search_lab(req: SearchLabRequest):
    return app.state.search.search_lab(req.query, req.k, _debug_allowed(req.debug))


@app.post("/api/search/image")
async def search_image(
    file: UploadFile = File(...),
    k: int = Form(12),
    debug: bool = Form(False),
):
    if k < 1 or k > 50:
        raise HTTPException(422, "k must be between 1 and 50")
    return app.state.search.search_image(
        _open_image(await file.read()),
        k,
        _debug_allowed(debug),
    )


@app.post("/api/search/multimodal")
async def search_multimodal(
    file: UploadFile = File(...),
    text: str = Form(""),
    k: int = Form(12),
    debug: bool = Form(False),
):
    if k < 1 or k > 50:
        raise HTTPException(422, "k must be between 1 and 50")
    return app.state.search.search_multimodal(
        _open_image(await file.read()),
        text,
        k,
        _debug_allowed(debug),
    )


@app.get("/api/products/{product_id}")
async def product(product_id: str):
    product_row = app.state.search.by_id.get(product_id)
    if not product_row:
        raise HTTPException(404, "Product not found")
    return product_row


@app.get("/api/search/debug/{query_id}")
async def debug_query(query_id: str):
    if not settings.expose_debug:
        raise HTTPException(404, "Debug endpoint disabled")
    payload = app.state.search.debug.get(query_id)
    if not payload:
        raise HTTPException(404, "Debug record not found in this process")
    return payload


@app.get("/api/index/status")
async def index_status():
    return app.state.search.status()


@app.get("/api/evaluation/latest")
async def evaluation_latest():
    mode_specific = settings.repo_root / "artifacts/evaluation" / f"latest-{settings.search_mode}.json"
    fallback = settings.repo_root / "artifacts/evaluation/latest.json"
    path = mode_specific if mode_specific.exists() else fallback
    if not path.exists():
        raise HTTPException(404, f"No saved evaluation run for {settings.search_mode} mode")
    output = json.loads(path.read_text(encoding="utf-8"))
    output["artifact_mode_matches_runtime"] = output.get("runtime_mode") == settings.search_mode
    return output


@app.post("/api/admin/reindex")
async def reindex(x_admin_key: str | None = Header(default=None)):
    if x_admin_key != settings.admin_key:
        raise HTTPException(401, "Invalid admin key")
    started = time.perf_counter()
    app.state.search = SearchEngine(settings.repo_root, settings.search_mode, force_rebuild=True)
    return {
        "status": "rebuilt",
        "seconds": round(time.perf_counter() - started, 3),
        "index": app.state.search.status(),
    }
