from pathlib import Path

from PIL import Image

from ml.engine import SearchEngine
from ml.evaluation import evaluate_images

ROOT = Path(__file__).resolve().parents[1]


def test_image_and_multimodal():
    engine = SearchEngine(ROOT, mode="lightweight")
    image = Image.open(ROOT / "data/demo/images/SHO-001.png")
    result = engine.search_image(image, k=5)
    assert result["results"][0]["id"] == "SHO-001"
    multimodal = engine.search_multimodal(image, "similar but blue under 2500", k=6)
    assert multimodal["results"]
    assert all(row["price"] <= 2500 for row in multimodal["results"])
    assert any(row["colour"] == "blue" for row in multimodal["results"][:3])
    assert set(multimodal["fusion_weights"]) == {"visual", "text", "crossmodal", "attribute"}


def test_image_retrieval_metric_is_computed():
    engine = SearchEngine(ROOT, mode="lightweight")
    result = evaluate_images(engine, ROOT / "data/demo/image_eval.jsonl", k=5)
    assert result["queries"] >= 10
    assert 0 <= result["value"] <= 1
