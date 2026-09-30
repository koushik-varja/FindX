from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import pytest

from ml.ann import faiss_available
from ml.engine import SearchEngine


ROOT = Path(__file__).resolve().parents[1]
FULL_TEST_ENV = "FINDX_RUN_FULL_INTEGRATION"


def _require_full_integration() -> None:
    if os.getenv(FULL_TEST_ENV, "").strip().lower() not in {"1", "true", "yes"}:
        pytest.skip(f"optional FULL ML integration test; set {FULL_TEST_ENV}=1 to run")
    missing = [
        package
        for package in ("sentence_transformers", "open_clip", "torch")
        if importlib.util.find_spec(package) is None
    ]
    if missing:
        pytest.skip("FULL ML dependencies are unavailable: " + ", ".join(missing))


def _full_engine(root: Path, *, force_rebuild: bool) -> SearchEngine:
    try:
        return SearchEngine(root, mode="full", force_rebuild=force_rebuild)
    except (OSError, ConnectionError) as exc:
        pytest.skip(f"FULL ML model weights/cache are unavailable: {exc}")
    except RuntimeError as exc:
        message = str(exc).lower()
        unavailable_markers = (
            "download",
            "connection",
            "network",
            "offline",
            "cache",
            "model weights",
            "pretrained",
            "huggingface",
        )
        if any(marker in message for marker in unavailable_markers):
            pytest.skip(f"FULL ML model weights/cache are unavailable: {exc}")
        raise


@pytest.mark.full_ml
@pytest.mark.slow
def test_full_ml_end_to_end_and_persistence(tmp_path: Path):
    """Opt-in validation of the real transformer/CLIP/ANN path.

    This test intentionally does not run in normal CI because the models are large.
    With FINDX_RUN_FULL_INTEGRATION=1 it uses the real configured models and fails
    on functional regressions; it only skips when dependencies or model weights
    are genuinely unavailable.
    """

    _require_full_integration()

    isolated_root = tmp_path / "findx-full-integration"
    shutil.copytree(ROOT / "data" / "demo", isolated_root / "data" / "demo")

    engine = _full_engine(isolated_root, force_rebuild=True)
    status = engine.status()

    assert status["mode"] == "full"
    assert status["runtime_label"] == "FULL ML"
    assert "sentence-transformers" in str(status["dense_model"])
    assert str(status["visual_model"]).startswith("open_clip:")
    assert int(status["text_embedding_dimension"]) > 0
    assert int(status["image_embedding_dimension"]) > 0

    text_embedding = engine.dense.encode_query("comfortable running shoes")
    assert isinstance(text_embedding, np.ndarray)
    assert text_embedding.ndim == 1
    assert text_embedding.shape[0] == status["text_embedding_dimension"]
    assert np.isfinite(text_embedding).all()

    first_product = engine.products[0]
    image_path = isolated_root / first_product["image_path"]
    with Image.open(image_path) as source_image:
        image = source_image.convert("RGB")
        image_embedding = engine.visual.encode_pil(image)
        clip_text_embedding = engine.visual.encode_text("similar product")
        image_payload = engine.search_image(image, k=5)
        multimodal_payload = engine.search_multimodal(image, "similar but blue", k=5)

    assert image_embedding.ndim == 1
    assert image_embedding.shape[0] == status["image_embedding_dimension"]
    assert np.isfinite(image_embedding).all()
    assert clip_text_embedding.shape == image_embedding.shape
    assert np.isfinite(clip_text_embedding).all()

    expected_ann = "faiss-hnsw-ip-v1" if faiss_available() else "random-hyperplane-ann-v1"
    assert status["text_vector_index"] == expected_ann
    assert status["image_vector_index"] == expected_ann

    text_payload = engine.search_text("samsoong wirless earbuds", k=5, mode="reranked")
    assert text_payload["runtime_mode"] == "full"
    assert text_payload["results"]
    assert image_payload["runtime_mode"] == "full"
    assert image_payload["results"]
    assert multimodal_payload["runtime_mode"] == "full"
    assert multimodal_payload["results"]

    metadata_path = isolated_root / "artifacts" / "index" / "full" / "metadata.json"
    assert metadata_path.exists()

    reloaded = _full_engine(isolated_root, force_rebuild=False)
    reloaded_status = reloaded.status()
    assert reloaded_status["loaded_from_persisted_index"] is True
    assert reloaded_status["index_version"] == status["index_version"]
    assert reloaded_status["text_vector_index"] == status["text_vector_index"]
    assert reloaded_status["image_vector_index"] == status["image_vector_index"]
    assert reloaded.search_text("black running shoes", k=3, mode="reranked")["results"]
