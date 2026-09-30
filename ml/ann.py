from __future__ import annotations

from pathlib import Path
import numpy as np


class HyperplaneANNIndex:
    """Small, dependency-free random-hyperplane ANN fallback.

    Unit vectors are hashed by signs of random projections. Search scans nearby
    Hamming signatures and falls back to an exact cosine scan when the candidate
    pool is too small. The fallback flag is returned with every search.
    """

    backend_name = "random-hyperplane-ann-v1"

    def __init__(self, n_planes: int = 18, seed: int = 23):
        self.n_planes = n_planes
        self.seed = seed

    def fit(self, vectors: np.ndarray) -> "HyperplaneANNIndex":
        x = np.asarray(vectors, dtype=np.float32)
        self.vectors = x
        rng = np.random.default_rng(self.seed)
        self.planes = rng.standard_normal((x.shape[1], self.n_planes), dtype=np.float32)
        self.signatures = (x @ self.planes >= 0).astype(np.uint8)
        return self

    def _hamming(self, query_signature: np.ndarray) -> np.ndarray:
        return np.sum(self.signatures != query_signature, axis=1)

    def search(self, vector: np.ndarray, k: int = 20):
        query = np.asarray(vector, dtype=np.float32).reshape(-1)
        query_signature = (query @ self.planes >= 0).astype(np.uint8)
        hamming = self._hamming(query_signature)
        pool = np.where(hamming <= 1)[0]
        used_exact_fallback = False
        if len(pool) < k:
            pool = np.where(hamming <= 2)[0]
        if len(pool) < k:
            pool = np.arange(len(self.vectors))
            used_exact_fallback = True
        sims = self.vectors[pool] @ query
        order = np.argsort(-sims)[:k]
        return [(int(pool[i]), float(sims[i])) for i in order], used_exact_fallback

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            vectors=self.vectors,
            planes=self.planes,
            signatures=self.signatures,
            n_planes=np.asarray([self.n_planes], dtype=np.int32),
            seed=np.asarray([self.seed], dtype=np.int32),
        )

    @classmethod
    def load(cls, path: Path) -> "HyperplaneANNIndex":
        data = np.load(path, allow_pickle=False)
        obj = cls(int(data["n_planes"][0]), int(data["seed"][0]))
        obj.vectors = np.asarray(data["vectors"], dtype=np.float32)
        obj.planes = np.asarray(data["planes"], dtype=np.float32)
        obj.signatures = np.asarray(data["signatures"], dtype=np.uint8)
        return obj


class FaissHNSWIndex:
    """FAISS HNSW cosine/inner-product ANN index for normalized vectors."""

    backend_name = "faiss-hnsw-ip-v1"

    def __init__(self, m: int = 32, ef_search: int = 64, ef_construction: int = 80):
        self.m = m
        self.ef_search = ef_search
        self.ef_construction = ef_construction

    @staticmethod
    def _faiss():
        try:
            import faiss  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "FAISS is not installed. Install backend/requirements-full.txt or use the documented hyperplane ANN fallback."
            ) from exc
        return faiss

    def fit(self, vectors: np.ndarray) -> "FaissHNSWIndex":
        faiss = self._faiss()
        x = np.ascontiguousarray(vectors, dtype=np.float32)
        index = faiss.IndexHNSWFlat(x.shape[1], self.m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = self.ef_construction
        index.hnsw.efSearch = self.ef_search
        index.add(x)
        self.index = index
        self.dimension = x.shape[1]
        self.count = x.shape[0]
        return self

    def search(self, vector: np.ndarray, k: int = 20):
        query = np.ascontiguousarray(np.asarray(vector, dtype=np.float32).reshape(1, -1))
        scores, indices = self.index.search(query, k)
        rows = [
            (int(idx), float(score))
            for idx, score in zip(indices[0].tolist(), scores[0].tolist())
            if idx >= 0
        ]
        return rows, False

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._faiss().write_index(self.index, str(path))

    @classmethod
    def load(cls, path: Path, ef_search: int = 64) -> "FaissHNSWIndex":
        faiss = cls._faiss()
        obj = cls(ef_search=ef_search)
        obj.index = faiss.read_index(str(path))
        if hasattr(obj.index, "hnsw"):
            obj.index.hnsw.efSearch = ef_search
        obj.dimension = obj.index.d
        obj.count = obj.index.ntotal
        return obj


def faiss_available() -> bool:
    try:
        import faiss  # noqa: F401
        return True
    except ImportError:
        return False


def build_vector_index(vectors: np.ndarray, prefer_faiss: bool = False, seed: int = 23):
    if prefer_faiss and faiss_available():
        return FaissHNSWIndex().fit(vectors)
    n_planes = min(24, max(10, vectors.shape[1] // 16))
    return HyperplaneANNIndex(n_planes=n_planes, seed=seed).fit(vectors)


def save_vector_index(index, directory: Path, prefix: str) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    if isinstance(index, FaissHNSWIndex):
        filename = f"{prefix}.faiss"
    else:
        filename = f"{prefix}.npz"
    index.save(directory / filename)
    return {"backend": index.backend_name, "file": filename}


def load_vector_index(directory: Path, metadata: dict):
    backend = metadata["backend"]
    path = directory / metadata["file"]
    if backend == FaissHNSWIndex.backend_name:
        return FaissHNSWIndex.load(path)
    if backend == HyperplaneANNIndex.backend_name:
        return HyperplaneANNIndex.load(path)
    raise ValueError(f"Unsupported vector index backend: {backend}")
