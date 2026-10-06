from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.benchmark import run_benchmark
from evaluation.reporting import generate_charts, generate_error_analysis, generate_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the reproducible FindX retrieval benchmark")
    parser.add_argument("--mode", choices=["lightweight", "full"], default="lightweight")
    parser.add_argument("--dataset", type=Path, default=ROOT / "evaluation/golden_queries.jsonl")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--latency-repeats", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()

    if not args.dataset.exists():
        raise SystemExit(f"golden dataset not found: {args.dataset}; run python scripts/build_golden_dataset.py")
    output_dir = args.output_dir or ROOT / "evaluation/results" / args.mode
    output_dir.mkdir(parents=True, exist_ok=True)
    results = run_benchmark(
        ROOT,
        mode=args.mode,
        dataset_path=args.dataset,
        k=args.k,
        latency_repeats=max(1, args.latency_repeats),
        concurrency=max(1, args.concurrency),
    )
    result_path = output_dir / "results.json"
    result_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    charts = generate_charts(results, output_dir)
    annotations = ROOT / "evaluation/manual_annotations.json"
    generate_error_analysis(results, output_dir / "error_analysis.md", annotations)
    generate_report(results, output_dir / "report.md", charts)
    # Stable top-level copies are useful to the frontend/docs and final package.
    (ROOT / "evaluation/report.md").write_text((output_dir / "report.md").read_text(encoding="utf-8"), encoding="utf-8")
    (ROOT / "evaluation/error_analysis.md").write_text((output_dir / "error_analysis.md").read_text(encoding="utf-8"), encoding="utf-8")
    print(json.dumps({"results": str(result_path), "report": str(output_dir / 'report.md'), "charts": charts}, indent=2))


if __name__ == "__main__":
    main()
