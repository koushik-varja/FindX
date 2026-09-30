from __future__ import annotations

import sys
from pathlib import Path as _BootstrapPath
ROOT_BOOTSTRAP = _BootstrapPath(__file__).resolve().parents[1]
if str(ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(ROOT_BOOTSTRAP))

import argparse
from datetime import datetime, timezone
from pathlib import Path
import json

from ml.engine import SearchEngine
from ml.evaluation import evaluate, evaluate_images

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="Evaluate FindX retrieval")
    parser.add_argument("--mode", choices=["lightweight", "full"], default="lightweight")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    engine = SearchEngine(ROOT, mode=args.mode)
    output = evaluate(engine, ROOT / "data/demo/eval_queries.jsonl", k=args.k)
    output["generated_at"] = datetime.now(timezone.utc).isoformat()
    output["index"] = engine.status()
    output["image_retrieval"] = evaluate_images(engine, ROOT / "data/demo/image_eval.jsonl", k=5)

    directory = ROOT / "artifacts/evaluation"
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stamped = directory / f"eval-{args.mode}-{stamp}.json"
    latest_mode = directory / f"latest-{args.mode}.json"
    stamped.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    latest_mode.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    if args.mode == "lightweight":
        (directory / "latest.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
