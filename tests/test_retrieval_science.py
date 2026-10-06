import numpy as np
import pytest

from ml.ann import ExactIPIndex, FaissHNSWIndex, faiss_available
from ml.fusion import reciprocal_rank_fusion, sort_fused
from ml.normalization import QueryNormalizer
from ml.rerank import _norm


def test_rrf_is_rank_based_not_raw_score_addition():
    lexical = [(0, 1000.0), (1, 999.0)]
    dense = [(1, 0.99), (0, 0.01)]
    fused = reciprocal_rank_fusion([lexical, dense], k0=60)
    rows = sort_fused(fused, 2)
    assert {idx for idx, _ in rows} == {0, 1}
    assert fused[0] == pytest.approx(fused[1])


def test_rerank_normalization_stays_in_unit_interval():
    assert _norm(5, 0, 10) == pytest.approx(0.5)
    assert _norm(10, 0, 10) == pytest.approx(1.0)
    assert _norm(3, 3, 3) == 0.0


def test_exact_ip_deterministic():
    vectors = np.eye(4, dtype=np.float32)
    index = ExactIPIndex().fit(vectors)
    first, _ = index.search(np.array([1, 0, 0, 0], dtype=np.float32), 3)
    second, _ = index.search(np.array([1, 0, 0, 0], dtype=np.float32), 3)
    assert first == second
    assert first[0][0] == 0


def test_hnsw_matches_exact_on_small_fixture_when_faiss_available():
    if not faiss_available():
        pytest.skip("faiss-cpu not installed")
    rng = np.random.default_rng(7)
    vectors = rng.standard_normal((64, 16), dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    query = vectors[11]
    exact, _ = ExactIPIndex().fit(vectors).search(query, 5)
    ann, _ = FaissHNSWIndex(m=16, ef_search=64, ef_construction=80).fit(vectors).search(query, 5)
    assert [idx for idx, _ in ann] == [idx for idx, _ in exact]


def test_multilingual_preprocessing_is_explicit_about_telugu():
    normalizer = QueryNormalizer({"black", "running", "shoes", "wireless", "earbuds"})
    assert normalizer.normalize("काले रनिंग जूते").corrected == "black running shoes"
    assert normalizer.normalize("3k ke andar black running shoes").corrected.startswith("3000 under")
    telugu = "నల్ల రన్నింగ్ షూస్"
    # There is deliberately no Telugu rewrite table. Text survives Unicode preprocessing
    # so the multilingual model can be measured honestly rather than simulated.
    assert normalizer.normalize(telugu).normalized == telugu
