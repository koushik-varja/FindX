from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHOD_LABELS = {
    "bm25": "BM25",
    "dense_ann": "Dense ANN",
    "dense_exact": "Dense exact",
    "hybrid": "Hybrid RRF",
    "reranked": "Hybrid + reranker",
}


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def generate_charts(results: dict, output_dir: Path) -> list[str]:
    charts = output_dir / "charts"
    charts.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []
    aggregate = results["text"]["aggregate"]
    k = int(results.get("hnsw", {}).get("k") or 10)

    methods = list(METHOD_LABELS)
    recalls = [aggregate[m].get("recall_at_10", aggregate[m].get(f"recall_at_{k}", 0)) for m in methods]
    p95s = [aggregate[m]["latency"]["p95_ms"] for m in methods]
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.scatter(p95s, recalls)
    for method, x, y in zip(methods, p95s, recalls):
        ax.annotate(METHOD_LABELS[method], (x, y), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel("p95 end-to-end latency (ms)")
    ax.set_ylabel("Recall@10")
    ax.set_title("Retrieval quality vs latency")
    ax.grid(True, alpha=0.25)
    path = charts / "recall_vs_p95_latency.png"
    _save(fig, path)
    generated.append(str(path))

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    x = range(len(methods))
    ax.bar(x, [aggregate[m]["mrr"] for m in methods])
    ax.set_xticks(list(x), [METHOD_LABELS[m] for m in methods], rotation=18, ha="right")
    ax.set_ylabel("MRR")
    ax.set_title("Lexical, dense, hybrid and reranked MRR")
    ax.set_ylim(0, 1.05)
    path = charts / "method_mrr.png"
    _save(fig, path)
    generated.append(str(path))

    languages = results["text"]["by_language"]
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    names = sorted(languages)
    width = 0.8 / len(methods)
    base = list(range(len(names)))
    for index, method in enumerate(methods):
        values = [languages[name][method].get("mrr", 0) for name in names]
        positions = [value - 0.4 + width / 2 + index * width for value in base]
        ax.bar(positions, values, width=width, label=METHOD_LABELS[method])
    ax.set_xticks(base, names)
    ax.set_ylabel("MRR")
    ax.set_title("MRR by language slice")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, ncol=2)
    path = charts / "mrr_by_language.png"
    _save(fig, path)
    generated.append(str(path))

    hybrid = aggregate["hybrid"]
    reranked = aggregate["reranked"]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    quality_delta = reranked["ndcg_at_10"] - hybrid["ndcg_at_10"]
    latency_delta = reranked["latency"]["p95_ms"] - hybrid["latency"]["p95_ms"]
    ax.bar(["Δ nDCG@10", "Δ p95 latency (ms)"], [quality_delta, latency_delta])
    ax.axhline(0, linewidth=0.8)
    ax.set_title("Reranker quality/latency delta vs hybrid")
    path = charts / "reranker_delta.png"
    _save(fig, path)
    generated.append(str(path))

    hnsw = results.get("hnsw") or {}
    if hnsw.get("available"):
        configs = hnsw["configs"]
        fig, ax = plt.subplots(figsize=(7.2, 4.5))
        for m in sorted({row["m"] for row in configs}):
            subset = sorted((row for row in configs if row["m"] == m), key=lambda row: row["ef_search"])
            ax.plot(
                [row["ef_search"] for row in subset],
                [row["ann_recall_at_k_vs_exact"] for row in subset],
                marker="o",
                label=f"M={m}",
            )
        ax.set_xlabel("efSearch")
        ax.set_ylabel("ANN neighbour Recall@10 vs exact")
        ax.set_title("HNSW search effort vs exact-neighbour recall")
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.25)
        path = charts / "hnsw_efsearch_recall.png"
        _save(fig, path)
        generated.append(str(path))

        fig, ax = plt.subplots(figsize=(7.2, 4.5))
        for m in sorted({row["m"] for row in configs}):
            subset = sorted((row for row in configs if row["m"] == m), key=lambda row: row["ef_search"])
            ax.plot(
                [row["ef_search"] for row in subset],
                [row["retrieval_latency"]["p95_ms"] for row in subset],
                marker="o",
                label=f"M={m}",
            )
        ax.set_xlabel("efSearch")
        ax.set_ylabel("p95 vector-retrieval latency (ms)")
        ax.set_title("HNSW search effort vs latency")
        ax.legend()
        ax.grid(True, alpha=0.25)
        path = charts / "hnsw_efsearch_latency.png"
        _save(fig, path)
        generated.append(str(path))
    return generated


def _rank_first_relevant(row: dict, method: str) -> int | None:
    relevant = set(row["relevant_document_ids"])
    for rank, doc_id in enumerate(row["methods"][method]["ranked_ids"], 1):
        if doc_id in relevant:
            return rank
    return None


def generate_error_analysis(results: dict, output_path: Path, annotations_path: Path | None = None) -> None:
    annotations = {}
    if annotations_path and annotations_path.exists():
        annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
    rows = results["text"]["queries"]
    failures = []
    for row in rows:
        hybrid_ndcg = row["methods"]["hybrid"]["metrics"]["ndcg_at_10"]
        rerank_ndcg = row["methods"]["reranked"]["metrics"]["ndcg_at_10"]
        bm_rank = _rank_first_relevant(row, "bm25")
        dense_rank = _rank_first_relevant(row, "dense_ann")
        exact_rank = _rank_first_relevant(row, "dense_exact")
        ann_rank = dense_rank
        reasons = []
        if bm_rank is not None and (dense_rank is None or bm_rank < dense_rank):
            reasons.append("BM25 places a relevant result earlier than dense ANN")
        if dense_rank is not None and (bm_rank is None or dense_rank < bm_rank):
            reasons.append("dense ANN places a relevant result earlier than BM25")
        if rerank_ndcg + 1e-12 < hybrid_ndcg:
            reasons.append("reranker decreases per-query nDCG@10")
        if exact_rank is not None and (ann_rank is None or ann_rank > exact_rank):
            reasons.append("ANN ranks the first relevant result below exact dense search")
        if row["language"] in {"hi", "te", "hinglish"} and row["methods"]["reranked"]["metrics"]["recall_at_10"] < 1.0:
            reasons.append(f"multilingual slice ({row['language']}) misses labelled relevant items")
        if reasons:
            failures.append((row, reasons, rerank_ndcg - hybrid_ndcg))

    failures.sort(key=lambda item: (item[2], item[0]["query_id"]))
    lines = [
        "# FindX Error Analysis",
        "",
        "This file is generated from actual benchmark output. It deliberately includes regressions and misses rather than showing only successful queries.",
        "",
        "## Important failing/difficult examples",
        "",
    ]
    for row, reasons, delta in failures[:20]:
        lines.extend(
            [
                f"### {row['query_id']} — `{row['query']}`",
                "",
                f"- Language: `{row['language']}`; type: `{row['query_type']}`",
                f"- First relevant rank — BM25: `{_rank_first_relevant(row, 'bm25')}`; dense ANN: `{_rank_first_relevant(row, 'dense_ann')}`; dense exact: `{_rank_first_relevant(row, 'dense_exact')}`; hybrid: `{_rank_first_relevant(row, 'hybrid')}`; reranked: `{_rank_first_relevant(row, 'reranked')}`",
                f"- Hybrid→reranked nDCG@10 delta: `{delta:+.4f}`",
                f"- Detected issue(s): {'; '.join(reasons)}.",
            ]
        )
        note = annotations.get(row["query"])
        if note:
            lines.append(f"- Manual annotation: {note}")
        lines.append("")

    if not failures:
        lines.extend(["No configured failure rule fired for this run. This does not imply the system is error-free.", ""])

    image = results.get("image")
    if image:
        lines.extend(["## Image retrieval", ""])
        weak = sorted(image["queries"], key=lambda row: row["metrics"]["recall_at_5"])
        for row in weak[:5]:
            lines.append(
                f"- `{row['query_id']}` from `{row['query_image_id']}` — Recall@5 `{row['metrics']['recall_at_5']:.4f}`; ranked `{', '.join(row['ranked_ids'][:5])}`."
            )
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def generate_report(results: dict, output_path: Path, charts: list[str]) -> None:
    k = 10
    dataset = results["dataset"]
    engine = results["engine"]
    perf = results["performance"]
    aggregate = results["text"]["aggregate"]
    lines = [
        "# FindX Retrieval Benchmark Report",
        "",
        f"Generated: `{results['generated_at']}`",
        "",
        "## 1. Dataset and ground truth",
        "",
        f"- Corpus size: **{engine['indexed_products']} products**.",
        f"- Golden queries: **{dataset['total_queries']}** total — {dataset['text_queries']} text, {dataset['image_queries']} image, {dataset['multimodal_queries']} multimodal.",
        f"- Ground truth: {dataset['ground_truth']}",
        "- The benchmark is a small student/demo benchmark. It is not presented as an industry-standard search benchmark or user study.",
        "",
        "## 2. Runtime configuration",
        "",
        f"- Runtime mode: `{results['runtime_mode']}`.",
        f"- Dense model: `{engine['dense_model']}` ({engine['text_embedding_dimension']} dimensions).",
        f"- Visual model: `{engine['visual_model']}` ({engine['image_embedding_dimension']} dimensions).",
        f"- Lexical index: `{engine['lexical_index']}`.",
        f"- Text vector index: `{engine['text_vector_index']}`.",
        f"- Image vector index: `{engine['image_vector_index']}`.",
        "- Full-mode vector similarity is inner product on L2-normalized embeddings, which is cosine-equivalent.",
        "",
        "## 3. Retrieval quality and latency",
        "",
        f"| Method | Precision@{k} | Recall@{k} | MRR | nDCG@{k} | p50 ms | p95 ms | p99 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, label in METHOD_LABELS.items():
        row = aggregate[method]
        latency = row["latency"]
        lines.append(
            f"| {label} | {_fmt(row[f'precision_at_{k}'])} | {_fmt(row[f'recall_at_{k}'])} | {_fmt(row['mrr'])} | {_fmt(row[f'ndcg_at_{k}'])} | {_fmt(latency['p50_ms'], 3)} | {_fmt(latency['p95_ms'], 3)} | {_fmt(latency['p99_ms'], 3)} |"
        )

    lines.extend(["", "## 4. Stage-level timing", ""])
    for method in ("dense_ann", "hybrid", "reranked"):
        lines.append(f"### {METHOD_LABELS[method]}")
        lines.append("")
        lines.append("| Stage | p50 ms | p95 ms |")
        lines.append("|---|---:|---:|")
        for stage, summary in aggregate[method].get("stage_latency", {}).items():
            lines.append(f"| {stage} | {_fmt(summary['p50_ms'], 3)} | {_fmt(summary['p95_ms'], 3)} |")
        lines.append("")

    lines.extend(["## 5. Per-language metrics", ""])
    for language, methods in results["text"]["by_language"].items():
        lines.append(f"### `{language}`")
        lines.append("")
        lines.append(f"| Method | Recall@{k} | MRR | nDCG@{k} |")
        lines.append("|---|---:|---:|---:|")
        for method, label in METHOD_LABELS.items():
            row = methods[method]
            lines.append(f"| {label} | {_fmt(row[f'recall_at_{k}'])} | {_fmt(row['mrr'])} | {_fmt(row[f'ndcg_at_{k}'])} |")
        lines.append("")

    lines.extend(["## 6. Multimodal evaluation", ""])
    for name in ("image", "multimodal"):
        section = results.get(name)
        if section:
            lines.append(f"- **{name}**: Recall@5 `{section['aggregate'].get('recall_at_5', 0):.4f}`, MRR `{section['aggregate'].get('mrr', 0):.4f}`, nDCG@5 `{section['aggregate'].get('ndcg_at_5', 0):.4f}`, p95 `{section['latency']['p95_ms']:.3f} ms`.")
    lines.append("")

    lines.extend(["## 7. Exact vs approximate search / HNSW", ""])
    hnsw = results.get("hnsw", {})
    if hnsw.get("available"):
        lines.extend(
            [
                f"Exact dense reference p95 retrieval latency: `{hnsw['exact_reference']['retrieval_latency']['p95_ms']:.4f} ms`.",
                "",
                "| M | efConstruction | efSearch | ANN recall@10 vs exact | p95 retrieval ms | build ms | index bytes |",
                "|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in hnsw["configs"]:
            lines.append(
                f"| {row['m']} | {row['ef_construction']} | {row['ef_search']} | {row['ann_recall_at_k_vs_exact']:.4f} | {row['retrieval_latency']['p95_ms']:.4f} | {row['build_ms']:.3f} | {row['index_bytes']} |"
            )
    else:
        lines.append(f"HNSW experiment not run: {hnsw.get('reason', 'unavailable')}.")
    lines.append("")

    hybrid = aggregate["hybrid"]
    reranked = aggregate["reranked"]
    lines.extend(
        [
            "## 8. Hybrid and reranker ablation",
            "",
            f"Hybrid RRF vs dense ANN Recall@10 delta: `{hybrid['recall_at_10'] - aggregate['dense_ann']['recall_at_10']:+.4f}`.",
            f"Reranker vs hybrid MRR delta: `{reranked['mrr'] - hybrid['mrr']:+.4f}`.",
            f"Reranker vs hybrid nDCG@10 delta: `{reranked['ndcg_at_10'] - hybrid['ndcg_at_10']:+.4f}`.",
            f"Reranker added p95 latency: `{reranked['latency']['p95_ms'] - hybrid['latency']['p95_ms']:+.3f} ms`.",
            "",
            "These deltas are reported as measured. No claim of improvement should be made when a delta is negative.",
            "",
            "## 9. Performance summary",
            "",
            f"- Engine build/load wall time: `{perf['engine_build_or_load_wall_seconds']:.4f} s`.",
            f"- Persisted Full index directory size: `{perf['full_index_directory_bytes']}` bytes.",
            f"- Observed RSS after engine load: `{perf['rss_bytes_after_engine']}` bytes.",
            f"- Controlled concurrency QPS: `{perf['throughput']['qps']}` at concurrency `{perf['throughput']['concurrency']}` over `{perf['throughput']['queries']}` queries.",
            "",
            "## 10. Charts",
            "",
        ]
    )
    for chart in charts:
        relative = Path(chart).relative_to(output_path.parent)
        lines.append(f"- ![]({relative.as_posix()})")

    lines.extend(["", "## 11. Error analysis", "", "See [`error_analysis.md`](error_analysis.md) for measured difficult and failing examples.", "", "## 12. Limitations", ""])
    for limitation in results.get("limitations", []):
        lines.append(f"- {limitation}")
    lines.extend(
        [
            "- Benchmark results should be regenerated on the target machine before quoting latency/QPS numbers.",
            "- The 82-item relevance corpus is too small to demonstrate production-scale ANN necessity; HNSW remains useful here as an architecture/algorithm experiment.",
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
