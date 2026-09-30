from __future__ import annotations

from pathlib import Path
import os
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer

from .ann import build_vector_index, load_vector_index, save_vector_index


FULL_TEXT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class LSIDenseRetriever:
    """Classical dense baseline used only in FINDX_MODE=lightweight."""

    name = "lsi-tfidf-svd-v1"

    def __init__(self, dim: int = 48, seed: int = 23):
        self.dim = dim
        self.seed = seed
        self.index_backend = "random-hyperplane-ann-v1"

    def fit(self, docs: list[str]):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
            strip_accents="unicode",
        )
        matrix = self.vectorizer.fit_transform(docs)
        n_components = max(2, min(self.dim, matrix.shape[0] - 1, matrix.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=n_components, random_state=self.seed)
        dense = self.svd.fit_transform(matrix)
        self.normalizer = Normalizer(copy=False)
        dense = self.normalizer.fit_transform(dense).astype(np.float32)
        self.vectors = dense
        self.ann = build_vector_index(dense, prefer_faiss=False, seed=self.seed)
        self.index_backend = self.ann.backend_name
        return self

    @property
    def embedding_dimension(self) -> int:
        return int(self.vectors.shape[1])

    def encode_query(self, query: str) -> np.ndarray:
        matrix = self.vectorizer.transform([query])
        vector = self.svd.transform(matrix)
        return self.normalizer.transform(vector).astype(np.float32)[0]

    def topk(self, query: str, k: int = 20):
        return self.ann.search(self.encode_query(query), k)


class SentenceTransformerDenseRetriever:
    """Flagship multilingual dense retriever for FINDX_MODE=full."""

    name = FULL_TEXT_MODEL

    def __init__(
        self,
        model_name: str | None = None,
        cache_dir: str | None = None,
        prefer_faiss: bool = True,
        batch_size: int = 32,
    ):
        self.model_name = model_name or os.getenv("FINDX_TEXT_MODEL", self.name)
        self.cache_dir = cache_dir or os.getenv("SENTENCE_TRANSFORMERS_HOME") or None
        self.prefer_faiss = prefer_faiss
        self.batch_size = batch_size
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "FINDX_MODE=full requires sentence-transformers. Install backend/requirements-full.txt. "
                "FindX will not silently downgrade FULL mode to the lightweight LSI baseline."
            ) from exc
        kwargs = {}
        if self.cache_dir:
            kwargs["cache_folder"] = self.cache_dir
        self._model = SentenceTransformer(self.model_name, **kwargs)
        return self._model

    def fit(self, docs: list[str]):
        model = self._load_model()
        self.vectors = np.asarray(
            model.encode(
                docs,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )
        self.ann = build_vector_index(self.vectors, prefer_faiss=self.prefer_faiss)
        self.index_backend = self.ann.backend_name
        return self

    @property
    def embedding_dimension(self) -> int:
        return int(self.vectors.shape[1])

    def encode_query(self, query: str) -> np.ndarray:
        model = self._load_model()
        return np.asarray(
            model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0],
            dtype=np.float32,
        )

    def topk(self, query: str, k: int = 20):
        return self.ann.search(self.encode_query(query), k)

    def save(self, directory: Path) -> dict:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "text_vectors.npy", self.vectors)
        ann_meta = save_vector_index(self.ann, directory, "text_ann")
        return {
            "model": self.model_name,
            "embedding_dimension": self.embedding_dimension,
            "vector_file": "text_vectors.npy",
            "vector_index": ann_meta,
        }

    def load(self, directory: Path, metadata: dict):
        if metadata.get("model") != self.model_name:
            raise ValueError("Persisted text index model does not match configured full-mode model")
        self._load_model()
        self.vectors = np.asarray(np.load(directory / metadata["vector_file"]), dtype=np.float32)
        self.ann = load_vector_index(directory, metadata["vector_index"])
        self.index_backend = self.ann.backend_name
        return self


def make_dense(mode: str = "lightweight"):
    if mode == "full":
        return SentenceTransformerDenseRetriever()
    if mode == "lightweight":
        return LSIDenseRetriever()
    raise ValueError("FINDX_MODE must be 'lightweight' or 'full'")
