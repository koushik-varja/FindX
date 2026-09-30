from __future__ import annotations

import sys
from pathlib import Path as _BootstrapPath

ROOT_BOOTSTRAP = _BootstrapPath(__file__).resolve().parents[1]
if str(ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(ROOT_BOOTSTRAP))

import argparse
from pathlib import Path

from PIL import Image

from ml.engine import SearchEngine

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["lightweight", "full"], default="lightweight")
    args = parser.parse_args()

    engine = SearchEngine(ROOT, mode=args.mode)
    checks = [
        "black running shoes under 3000",
        "3k ke andar black running shoes",
        "samsoong wirless earbuds",
        "काले रनिंग जूते",
    ]
    for query in checks:
        payload = engine.search_text(query, k=3)
        print(query, "=>", [(row["id"], row["title"]) for row in payload["results"]])

    with Image.open(ROOT / engine.by_id["SHO-001"]["image_path"]) as image:
        image = image.convert("RGB")
        print("image =>", [(row["id"], row["title"]) for row in engine.search_image(image, k=3)["results"]])
        print(
            "multimodal =>",
            [
                (row["id"], row["title"])
                for row in engine.search_multimodal(image, "similar but blue under 2500", k=3)["results"]
            ],
        )


if __name__ == "__main__":
    main()
