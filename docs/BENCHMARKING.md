# Benchmarking

`python scripts/evaluate.py --mode <lightweight|full>` measures retrieval metrics and request-level search latency in the current process. These values are appropriate for comparing modes on the included demo catalog, not for making production-scale performance claims.

`python scripts/build_index.py --mode full --force` reports wall-clock index-build duration and runtime metadata. Full index size can be inspected directly under `artifacts/index/full/` after a successful build.

The repository includes lightweight results generated in the build environment. Full-mode numbers are intentionally absent unless the required model dependencies and weights were actually executed.
