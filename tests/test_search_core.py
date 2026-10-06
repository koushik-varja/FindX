from pathlib import Path

import pytest

from ml.engine import SearchEngine

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def engine():
    return SearchEngine(ROOT, mode="lightweight")


def test_bm25_real(engine):
    result = engine.search_text("samsung wireless earbuds", k=5, mode="bm25")
    assert any(row["id"] == "ELE-001" for row in result["results"])


def test_dense(engine):
    result = engine.search_text("comfortable shoes for long distance running", k=10, mode="dense")
    assert any(row["id"].startswith("SHO-") for row in result["results"])


def test_hybrid_and_reranker_price(engine):
    result = engine.search_text("black running shoes under 3000", k=10, mode="reranked", debug=True)
    assert result["results"]
    assert all(row["price"] <= 3000 for row in result["results"])
    assert result["results"][0]["debug"]["rerank_score"] is not None
    assert result["timings_ms"]["dense_embedding"] >= 0
    assert result["timings_ms"]["dense_retrieval"] >= 0


def test_debug_metadata_is_opt_in(engine):
    normal = engine.search_text("wireless earbuds", k=3, mode="reranked", debug=False)
    debug = engine.search_text("wireless earbuds", k=3, mode="reranked", debug=True)
    assert "timings_ms" not in normal
    assert "debug" not in normal["results"][0]
    assert "timings_ms" in debug
    assert "bm25_rank" in debug["results"][0]["debug"]


def test_typo_recovery(engine):
    result = engine.search_text("samsoong wirless earbuds", k=3)
    assert result["corrected_query"] == "samsung wireless earbuds"
    assert result["results"][0]["brand"] == "Samsung"


def test_hinglish_and_hindi(engine):
    hinglish = engine.search_text("3k ke andar black running shoes", k=8)
    hindi = engine.search_text("काले रनिंग जूते", k=8)
    assert any(row["id"].startswith("SHO-") for row in hinglish["results"])
    assert any(row["id"].startswith("SHO-") for row in hindi["results"])


def test_search_lab(engine):
    lab = engine.search_lab("wireless earbuds", 3, debug=True)
    assert set(lab) == {"bm25", "dense", "hybrid", "reranked"}
    assert all(payload["results"] for payload in lab.values())


def test_lightweight_status_is_explicit(engine):
    status = engine.status()
    assert status["runtime_label"] == "LIGHTWEIGHT"
    assert status["dense_model"] == "lsi-tfidf-svd-v1"
    assert status["visual_model"] == "lightweight-visual-v1"
