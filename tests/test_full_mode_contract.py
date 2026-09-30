from pathlib import Path
import importlib.util

import pytest

from ml.dense import SentenceTransformerDenseRetriever
from ml.engine import SearchEngine
from ml.images import OpenCLIPEmbedder

ROOT = Path(__file__).resolve().parents[1]


def test_full_components_do_not_silently_downgrade():
    if importlib.util.find_spec("sentence_transformers") is None:
        with pytest.raises(RuntimeError, match="requires sentence-transformers"):
            SentenceTransformerDenseRetriever()._load_model()
    if importlib.util.find_spec("open_clip") is None:
        with pytest.raises(RuntimeError, match="requires open_clip_torch"):
            OpenCLIPEmbedder()


def test_full_engine_fails_clearly_when_full_dependencies_missing():
    missing = importlib.util.find_spec("sentence_transformers") is None or importlib.util.find_spec("open_clip") is None
    if missing:
        with pytest.raises(RuntimeError):
            SearchEngine(ROOT, mode="full")
    else:
        pytest.skip("Full dependencies exist; integration execution is covered by the explicit full-mode validation command")
