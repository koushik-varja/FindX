from pathlib import Path

from ml.engine import SearchEngine


ROOT = Path(__file__).resolve().parents[1]


def test_ranking_is_deterministic_for_same_index_and_query():
    engine = SearchEngine(ROOT, mode="lightweight")
    first = engine.search_text("black running shoes under 3000", k=10, mode="reranked")
    second = engine.search_text("black running shoes under 3000", k=10, mode="reranked")
    assert [row["id"] for row in first["results"]] == [row["id"] for row in second["results"]]
    assert [row["score"] for row in first["results"]] == [row["score"] for row in second["results"]]
