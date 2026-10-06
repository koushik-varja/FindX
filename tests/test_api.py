import io
import os

os.environ["DATABASE_URL"] = "sqlite:///./test_findx.db"
os.environ["FINDX_MODE"] = "lightweight"
os.environ["FINDX_TEST_CREATE_SCHEMA"] = "1"
os.environ["FINDX_EXPOSE_DEBUG"] = "1"

from fastapi.testclient import TestClient
from PIL import Image

from backend.app.main import app


def test_api_validation_and_search():
    with TestClient(app) as client:
        assert client.post("/api/search/text", json={"query": ""}).status_code == 422
        response = client.post("/api/search/text", json={"query": "samsoong wirless earbuds", "k": 3})
        assert response.status_code == 200
        assert response.json()["results"][0]["brand"] == "Samsung"
        assert "timings_ms" not in response.json()
        assert client.get("/api/products/NOPE").status_code == 404


def test_debug_metadata_requires_explicit_request():
    with TestClient(app) as client:
        response = client.post(
            "/api/search/text",
            json={"query": "wireless earbuds", "k": 3, "debug": True},
        )
        payload = response.json()
        assert response.status_code == 200
        assert "timings_ms" in payload
        assert "bm25_rank" in payload["results"][0]["debug"]


def test_multimodal_request_validation():
    with TestClient(app) as client:
        bad = client.post(
            "/api/search/multimodal",
            data={"text": "similar but blue", "k": "12"},
            files={"file": ("bad.txt", b"not-an-image", "text/plain")},
        )
        assert bad.status_code == 415

        image = Image.new("RGB", (16, 16), "black")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        invalid_k = client.post(
            "/api/search/multimodal",
            data={"text": "similar", "k": "0"},
            files={"file": ("query.png", buffer.getvalue(), "image/png")},
        )
        assert invalid_k.status_code == 422


def test_admin_protection():
    with TestClient(app) as client:
        assert client.post("/api/admin/reindex").status_code == 401


def test_index_status_exposes_real_mode():
    with TestClient(app) as client:
        status = client.get("/api/index/status").json()
        assert status["runtime_label"] == "LIGHTWEIGHT"
        assert status["dense_model"] == "lsi-tfidf-svd-v1"
