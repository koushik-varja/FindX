from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import time
import uuid

import numpy as np
from PIL import Image

from .ann import build_vector_index, load_vector_index, save_vector_index
from .attributes import attribute_match, parse_attributes, passes_hard_filters
from .bm25 import BM25Index, tokenize
from .dense import make_dense
from .fusion import reciprocal_rank_fusion, sort_fused
from .images import make_visual
from .normalization import QueryNormalizer
from .rerank import rerank


class SearchEngine:
    def __init__(
        self,
        repo_root: Path | str | None = None,
        mode: str = "lightweight",
        force_rebuild: bool = False,
    ):
        self.root = Path(repo_root or Path(__file__).resolve().parents[1])
        self.mode = mode.strip().lower()
        if self.mode not in {"lightweight", "full"}:
            raise ValueError("mode must be 'lightweight' or 'full'")
        self.force_rebuild = force_rebuild
        self.debug: dict[str, dict] = {}
        self._build()

    def _catalog_hash(self) -> str:
        digest = hashlib.sha256()
        catalog = self.root / "data/demo/catalog.jsonl"
        digest.update(catalog.read_bytes())
        for product in self.products:
            path = self.root / product["image_path"]
            digest.update(product["image_path"].encode("utf-8"))
            digest.update(path.read_bytes())
        return digest.hexdigest()

    def _search_document(self, product: dict) -> str:
        return " ".join(
            [
                product["title"],
                product["description"],
                product["category"],
                product["brand"],
                product["colour"],
                " ".join(map(str, product["attributes"].values())),
                " ".join(product["tags"]),
                product.get("search_text_hi", ""),
            ]
        )

    def _full_index_dir(self) -> Path:
        return self.root / "artifacts/index/full"

    def _full_metadata_valid(self, metadata: dict) -> bool:
        return (
            metadata.get("mode") == "full"
            and metadata.get("catalog_hash") == self.catalog_hash
            and metadata.get("dense_model") == getattr(self.dense, "model_name", self.dense.name)
            and metadata.get("visual_model") == self.visual.name
            and metadata.get("product_ids") == [product["id"] for product in self.products]
        )

    def _load_full_indexes(self, metadata: dict) -> None:
        directory = self._full_index_dir()
        self.dense.load(directory, metadata["text"])
        self.image_vectors = np.asarray(np.load(directory / metadata["image"]["vector_file"]), dtype=np.float32)
        self.image_ann = load_vector_index(directory, metadata["image"]["vector_index"])
        self.image_index_backend = self.image_ann.backend_name

    def _persist_full_indexes(self) -> None:
        directory = self._full_index_dir()
        directory.mkdir(parents=True, exist_ok=True)
        text_metadata = self.dense.save(directory)
        np.save(directory / "image_vectors.npy", self.image_vectors)
        image_ann_metadata = save_vector_index(self.image_ann, directory, "image_ann")
        product_ids = [product["id"] for product in self.products]
        (directory / "text_ids.json").write_text(json.dumps(product_ids, indent=2) + "\n", encoding="utf-8")
        (directory / "image_ids.json").write_text(json.dumps(product_ids, indent=2) + "\n", encoding="utf-8")
        metadata = {
            "index_version": self.index_version,
            "mode": "full",
            "build_timestamp": self.last_rebuild_at,
            "catalog_hash": self.catalog_hash,
            "indexed_products": len(self.products),
            "product_ids": product_ids,
            "text_id_map": "text_ids.json",
            "image_id_map": "image_ids.json",
            "dense_model": getattr(self.dense, "model_name", self.dense.name),
            "visual_model": self.visual.name,
            "text": text_metadata,
            "image": {
                "model": self.visual.name,
                "embedding_dimension": int(self.image_vectors.shape[1]),
                "vector_file": "image_vectors.npy",
                "vector_index": image_ann_metadata,
            },
        }
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        self.persisted_metadata = metadata

    def _build(self) -> None:
        started = time.perf_counter()
        catalog = self.root / "data/demo/catalog.jsonl"
        self.products = [json.loads(line) for line in catalog.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.by_id = {product["id"]: product for product in self.products}
        self.catalog_hash = self._catalog_hash()

        documents: list[str] = []
        vocabulary: set[str] = set()
        for product in self.products:
            text = self._search_document(product)
            documents.append(text)
            vocabulary.update(tokenize(text))

        self.normalizer = QueryNormalizer(vocabulary)
        self.bm25 = BM25Index().fit(documents)
        self.dense = make_dense(self.mode)
        self.visual = make_visual(self.mode)
        self.loaded_from_persisted_index = False
        self.persisted_metadata: dict | None = None

        if self.mode == "full":
            metadata_path = self._full_index_dir() / "metadata.json"
            metadata = None
            if metadata_path.exists() and not self.force_rebuild:
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    metadata = None
            if metadata and self._full_metadata_valid(metadata):
                self._load_full_indexes(metadata)
                self.persisted_metadata = metadata
                self.index_version = metadata["index_version"]
                self.last_rebuild_at = metadata["build_timestamp"]
                self.loaded_from_persisted_index = True
            else:
                self.dense.fit(documents)
                image_paths = [self.root / product["image_path"] for product in self.products]
                batch_size = int(os.getenv("FINDX_CLIP_BATCH_SIZE", "16"))
                self.image_vectors = self.visual.encode_paths(image_paths, batch_size=batch_size)
                self.image_ann = build_vector_index(self.image_vectors, prefer_faiss=True)
                self.image_index_backend = self.image_ann.backend_name
                self.index_version = f"full-{len(self.products)}-{self.catalog_hash[:12]}"
                self.last_rebuild_at = datetime.now(timezone.utc).isoformat()
                self._persist_full_indexes()
        else:
            self.dense.fit(documents)
            image_paths = [self.root / product["image_path"] for product in self.products]
            self.image_vectors = self.visual.encode_paths(image_paths)
            self.image_ann = build_vector_index(self.image_vectors, prefer_faiss=False)
            self.image_index_backend = self.image_ann.backend_name
            self.index_version = f"lightweight-{len(self.products)}-{self.catalog_hash[:12]}"
            self.last_rebuild_at = datetime.now(timezone.utc).isoformat()

        self.build_seconds = time.perf_counter() - started

    @staticmethod
    def _rank_map(rows: list[tuple[int, float]]) -> dict[int, int]:
        return {idx: rank for rank, (idx, _score) in enumerate(rows, 1)}

    @staticmethod
    def _elapsed_ms(started: float) -> float:
        return round((time.perf_counter() - started) * 1000, 4)

    def _pack(
        self,
        idx,
        rank,
        score,
        result_mode,
        *,
        include_debug: bool = False,
        bm=0.0,
        dense=0.0,
        rerank_score=None,
        matched=None,
        reason=None,
        debug_extra: dict | None = None,
    ):
        product = dict(self.products[idx])
        product["rank"] = rank
        product["score"] = round(float(score), 6)
        product["mode"] = result_mode
        if include_debug:
            product["debug"] = {
                "lexical_score": round(float(bm), 6),
                "semantic_score": round(float(dense), 6),
                "rerank_score": None if rerank_score is None else round(float(rerank_score), 6),
                "matched_attributes": matched or [],
                "reason": reason or "",
                **(debug_extra or {}),
            }
        return product

    def search_text(self, query: str, k: int = 12, mode: str = "reranked", debug: bool = False):
        total_started = time.perf_counter()
        timings: dict[str, float] = {}

        started = time.perf_counter()
        normalized = self.normalizer.normalize(query)
        attributes = parse_attributes(normalized.corrected)
        timings["preprocessing"] = self._elapsed_ms(started)

        candidate_k = min(60, len(self.products))
        started = time.perf_counter()
        bm25_list = self.bm25.topk(normalized.corrected, candidate_k)
        timings["bm25"] = self._elapsed_ms(started)
        bm25_scores = dict(bm25_list)

        started = time.perf_counter()
        query_vector = self.dense.encode_query(normalized.corrected)
        timings["dense_embedding"] = self._elapsed_ms(started)

        started = time.perf_counter()
        dense_list, ann_fallback = self.dense.search_vector(query_vector, candidate_k)
        timings["dense_retrieval"] = self._elapsed_ms(started)
        dense_scores = dict(dense_list)

        started = time.perf_counter()
        fused = reciprocal_rank_fusion([bm25_list, dense_list])
        fused_list = sort_fused(fused, candidate_k)
        timings["fusion"] = self._elapsed_ms(started)

        bm25_ranks = self._rank_map(bm25_list)
        dense_ranks = self._rank_map(dense_list)
        hybrid_ranks = self._rank_map(fused_list)

        def extra(idx: int, final_rank: int, parts: dict | None = None) -> dict:
            payload = {
                "bm25_rank": bm25_ranks.get(idx),
                "dense_rank": dense_ranks.get(idx),
                "hybrid_rank": hybrid_ranks.get(idx),
                "final_rank": final_rank,
                "fusion_score": round(float(fused.get(idx, 0.0)), 6),
            }
            if parts:
                payload["rerank_components"] = {key: round(float(value), 6) for key, value in parts.items()}
            return payload

        started = time.perf_counter()
        if mode == "bm25":
            selected = [(idx, score) for idx, score in bm25_list if passes_hard_filters(self.products[idx], attributes)][:k]
            results = [
                self._pack(
                    idx,
                    rank,
                    score,
                    "bm25",
                    include_debug=debug,
                    bm=score,
                    matched=attribute_match(self.products[idx], attributes)[1],
                    reason="lexical term match",
                    debug_extra=extra(idx, rank),
                )
                for rank, (idx, score) in enumerate(selected, 1)
            ]
        elif mode == "dense":
            selected = [(idx, score) for idx, score in dense_list if passes_hard_filters(self.products[idx], attributes)][:k]
            results = [
                self._pack(
                    idx,
                    rank,
                    score,
                    "dense",
                    include_debug=debug,
                    dense=score,
                    matched=attribute_match(self.products[idx], attributes)[1],
                    reason="dense vector similarity",
                    debug_extra=extra(idx, rank),
                )
                for rank, (idx, score) in enumerate(selected, 1)
            ]
        elif mode == "hybrid":
            selected = [(idx, score) for idx, score in fused_list if passes_hard_filters(self.products[idx], attributes)][:k]
            results = [
                self._pack(
                    idx,
                    rank,
                    score,
                    "hybrid",
                    include_debug=debug,
                    bm=bm25_scores.get(idx, 0),
                    dense=dense_scores.get(idx, 0),
                    matched=attribute_match(self.products[idx], attributes)[1],
                    reason="reciprocal-rank fusion",
                    debug_extra=extra(idx, rank),
                )
                for rank, (idx, score) in enumerate(selected, 1)
            ]
        elif mode == "reranked":
            candidates = [idx for idx, _ in fused_list]
            reranked = rerank(
                self.products,
                candidates,
                bm25_scores,
                dense_scores,
                fused,
                attributes,
                normalized.correction_confidence,
                k,
            )
            results = []
            for rank, (idx, score, matched, parts) in enumerate(reranked, 1):
                reason = "hybrid relevance"
                if matched:
                    reason += " + " + ", ".join(matched[:3])
                results.append(
                    self._pack(
                        idx,
                        rank,
                        score,
                        "reranked",
                        include_debug=debug,
                        bm=bm25_scores.get(idx, 0),
                        dense=dense_scores.get(idx, 0),
                        rerank_score=score,
                        matched=matched,
                        reason=reason,
                        debug_extra=extra(idx, rank, parts),
                    )
                )
        else:
            raise ValueError("mode must be bm25, dense, hybrid or reranked")
        timings["ranking"] = self._elapsed_ms(started)
        timings["total"] = self._elapsed_ms(total_started)

        query_id = str(uuid.uuid4())
        payload = {
            "query_id": query_id,
            "original_query": query,
            "normalized_query": normalized.normalized,
            "corrected_query": normalized.corrected,
            "corrections": normalized.corrections,
            "correction_confidence": round(normalized.correction_confidence, 4),
            "parsed_attributes": attributes,
            "mode": mode,
            "runtime_mode": self.mode,
            "latency_ms": timings["total"],
            "ann_exact_fallback": ann_fallback,
            "results": results,
        }
        if debug:
            payload["timings_ms"] = timings
            self.debug[query_id] = payload
            if len(self.debug) > 200:
                self.debug.pop(next(iter(self.debug)))
        return payload

    def search_lab(self, query: str, k: int = 8, debug: bool = False):
        return {
            mode: self.search_text(query, k=k, mode=mode, debug=debug)
            for mode in ["bm25", "dense", "hybrid", "reranked"]
        }

    def search_image(self, image: Image.Image, k: int = 12, debug: bool = False):
        total_started = time.perf_counter()
        started = time.perf_counter()
        query_vector = self.visual.encode_pil(image)
        embedding_ms = self._elapsed_ms(started)
        started = time.perf_counter()
        rows, ann_fallback = self.image_ann.search(query_vector, k)
        retrieval_ms = self._elapsed_ms(started)
        results = [
            self._pack(
                idx,
                rank,
                score,
                "image",
                include_debug=debug,
                reason="visual embedding cosine similarity",
                debug_extra={"final_rank": rank},
            )
            for rank, (idx, score) in enumerate(rows, 1)
        ]
        payload = {
            "query_id": str(uuid.uuid4()),
            "mode": "image",
            "runtime_mode": self.mode,
            "visual_model": self.visual.name,
            "vector_index": self.image_index_backend,
            "ann_exact_fallback": ann_fallback,
            "latency_ms": self._elapsed_ms(total_started),
            "results": results,
        }
        if debug:
            payload["timings_ms"] = {
                "image_embedding": embedding_ms,
                "image_retrieval": retrieval_ms,
                "total": payload["latency_ms"],
            }
        return payload

    def _multimodal_weights(self, has_text: bool) -> dict[str, float]:
        visual = float(os.getenv("FINDX_MM_VISUAL_WEIGHT", "0.50"))
        text = float(os.getenv("FINDX_MM_TEXT_WEIGHT", "0.20")) if has_text else 0.0
        crossmodal = (
            float(os.getenv("FINDX_MM_CROSSMODAL_WEIGHT", "0.20"))
            if has_text and getattr(self.visual, "supports_text", False)
            else 0.0
        )
        attribute = float(os.getenv("FINDX_MM_ATTRIBUTE_WEIGHT", "0.10")) if has_text else 0.0
        total = visual + text + crossmodal + attribute
        if total <= 0:
            raise ValueError("Multimodal fusion weights must sum to a positive value")
        return {
            "visual": visual / total,
            "text": text / total,
            "crossmodal": crossmodal / total,
            "attribute": attribute / total,
        }

    def search_multimodal(self, image: Image.Image, text: str = "", k: int = 12, debug: bool = False):
        total_started = time.perf_counter()
        timings: dict[str, float] = {}

        started = time.perf_counter()
        query_visual = self.visual.encode_pil(image)
        timings["image_embedding"] = self._elapsed_ms(started)

        started = time.perf_counter()
        visual_scores = self.image_vectors @ query_visual
        timings["visual_scoring"] = self._elapsed_ms(started)

        normalized = self.normalizer.normalize(text or "similar")
        attributes = parse_attributes(normalized.corrected)
        anchor_idx = int(np.argmax(visual_scores))
        anchor_category = self.products[anchor_idx]["category"]

        text_scores: dict[str, float] = {}
        started = time.perf_counter()
        if text.strip():
            text_payload = self.search_text(text, k=len(self.products), mode="reranked", debug=False)
            text_scores = {row["id"]: row["score"] for row in text_payload["results"]}
        timings["text_retrieval"] = self._elapsed_ms(started)
        max_text = max(text_scores.values(), default=1.0)

        crossmodal_scores = np.zeros(len(self.products), dtype=np.float32)
        started = time.perf_counter()
        if text.strip() and getattr(self.visual, "supports_text", False):
            clip_text_vector = self.visual.encode_text(normalized.corrected)
            crossmodal_scores = self.image_vectors @ clip_text_vector
        timings["crossmodal_text_embedding_and_scoring"] = self._elapsed_ms(started)

        started = time.perf_counter()
        weights = self._multimodal_weights(bool(text.strip()))
        rows = []
        for idx, product in enumerate(self.products):
            if not passes_hard_filters(product, attributes):
                continue
            if "category" not in attributes and product["category"] != anchor_category:
                continue
            attribute_score, matched = attribute_match(product, attributes)
            normalized_text_score = text_scores.get(product["id"], 0.0) / max(max_text, 1e-9)
            visual_score = max(0.0, float(visual_scores[idx]))
            crossmodal_score = max(0.0, float(crossmodal_scores[idx]))
            score = (
                weights["visual"] * visual_score
                + weights["text"] * normalized_text_score
                + weights["crossmodal"] * crossmodal_score
                + weights["attribute"] * attribute_score
            )
            rows.append(
                (
                    idx,
                    score,
                    matched,
                    normalized_text_score,
                    crossmodal_score,
                    visual_score,
                    attribute_score,
                )
            )
        rows.sort(key=lambda row: (-row[1], self.products[row[0]]["price"]))
        timings["fusion_and_ranking"] = self._elapsed_ms(started)

        results = []
        for rank, row in enumerate(rows[:k], 1):
            idx, score, matched, text_score, crossmodal_score, visual_score, attribute_score = row
            debug_extra = {
                "final_rank": rank,
                "visual_similarity": round(visual_score, 6),
                "clip_text_image_similarity": round(crossmodal_score, 6),
                "attribute_score": round(attribute_score, 6),
            }
            results.append(
                self._pack(
                    idx,
                    rank,
                    score,
                    "multimodal",
                    include_debug=debug,
                    dense=text_score,
                    rerank_score=score,
                    matched=matched,
                    reason="weighted visual + textual + cross-modal + attribute fusion",
                    debug_extra=debug_extra,
                )
            )

        timings["total"] = self._elapsed_ms(total_started)
        payload = {
            "query_id": str(uuid.uuid4()),
            "mode": "multimodal",
            "runtime_mode": self.mode,
            "original_query": text,
            "corrected_query": normalized.corrected,
            "parsed_attributes": attributes,
            "visual_model": self.visual.name,
            "reference_category": anchor_category,
            "fusion_weights": weights,
            "latency_ms": timings["total"],
            "results": results,
        }
        if debug:
            payload["timings_ms"] = timings
        return payload

    def status(self):
        dense_vectors = getattr(self.dense, "vectors", np.empty(0, dtype=np.float32))
        dense_bytes = int(dense_vectors.nbytes)
        image_bytes = int(self.image_vectors.nbytes)
        output = {
            "runtime_label": "FULL ML" if self.mode == "full" else "LIGHTWEIGHT",
            "mode": self.mode,
            "index_version": self.index_version,
            "indexed_products": len(self.products),
            "lexical_index": "custom-bm25-v1",
            "dense_model": getattr(self.dense, "model_name", self.dense.name),
            "visual_model": self.visual.name,
            "text_vector_index": self.dense.index_backend,
            "image_vector_index": self.image_index_backend,
            "text_embedding_dimension": int(dense_vectors.shape[1]) if dense_vectors.ndim == 2 else 0,
            "image_embedding_dimension": int(self.image_vectors.shape[1]),
            "loaded_from_persisted_index": self.loaded_from_persisted_index,
            "last_rebuild_at": self.last_rebuild_at,
            "startup_seconds": round(self.build_seconds, 3),
            "numeric_vector_bytes": dense_bytes + image_bytes,
        }
        text_ann = getattr(self.dense, "ann", None)
        if getattr(text_ann, "backend_name", "") == "faiss-hnsw-ip-v1":
            output["text_hnsw"] = {
                "m": text_ann.m,
                "ef_construction": text_ann.ef_construction,
                "ef_search": text_ann.ef_search,
                "metric": "inner_product_on_l2_normalized_vectors",
            }
        if getattr(self.image_ann, "backend_name", "") == "faiss-hnsw-ip-v1":
            output["image_hnsw"] = {
                "m": self.image_ann.m,
                "ef_construction": self.image_ann.ef_construction,
                "ef_search": self.image_ann.ef_search,
                "metric": "inner_product_on_l2_normalized_vectors",
            }
        return output
