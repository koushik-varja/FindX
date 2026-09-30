from __future__ import annotations

import sys
from pathlib import Path as _BootstrapPath
ROOT_BOOTSTRAP = _BootstrapPath(__file__).resolve().parents[1]
if str(ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(ROOT_BOOTSTRAP))

import argparse
from pathlib import Path
import json
import time

from ml.engine import SearchEngine

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="Build FindX search indexes")
    parser.add_argument("--mode", choices=["lightweight", "full"], default="lightweight")
    parser.add_argument("--force", action="store_true", help="Force regeneration even when a valid full index exists")
    args = parser.parse_args()

    started = time.perf_counter()
    engine = SearchEngine(ROOT, mode=args.mode, force_rebuild=args.force)
    output = engine.status()
    output["wall_seconds"] = round(time.perf_counter() - started, 3)
    output["catalog_hash"] = engine.catalog_hash

    directory = ROOT / "artifacts/index" / args.mode
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "status.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
