import os

os.environ["DATABASE_URL"] = "sqlite:///./test_findx.db"
os.environ["FINDX_MODE"] = "lightweight"
os.environ["FINDX_TEST_CREATE_SCHEMA"] = "1"

from fastapi.testclient import TestClient

from backend.app.main import app


def test_api_validation_and_search():
    with TestClient(app) as client:
        assert client.post("/api/search/text", json={"query": ""}).status_code == 422
        response = client.post("/api/search/text", json={"query": "samsoong wirless earbuds", "k": 3})
        assert response.status_code == 200
        assert response.json()["results"][0]["brand"] == "Samsung"
        assert client.get("/api/products/NOPE").status_code == 404


def test_admin_protection():
    with TestClient(app) as client:
        assert client.post("/api/admin/reindex").status_code == 401


def test_index_status_exposes_real_mode():
    with TestClient(app) as client:
        status = client.get("/api/index/status").json()
        assert status["runtime_label"] == "LIGHTWEIGHT"
        assert status["dense_model"] == "lsi-tfidf-svd-v1"
