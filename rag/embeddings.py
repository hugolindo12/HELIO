"""
HEILO Dense Embedding Backends

Priority (auto mode):
  1. sentence-transformers  (if installed + HEILO_USE_ST=1 or auto finds it)
  2. Ollama /api/embeddings (if server reachable)
  3. LSA dense (TF-IDF + truncated SVD) — always available, pure NumPy
  4. TF-IDF sparse normalized (legacy)

Dense vectors = fixed-size continuous representations for cosine RAG.
"""
from __future__ import annotations

import json
import math
import os
import re
import urllib.request
from abc import ABC, abstractmethod
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-ZÀ-ÿ_][a-zA-ZÀ-ÿ0-9_]{1,}", (text or "").lower())


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return mat / norms


class EmbeddingBackend(ABC):
    name: str = "base"

    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """Return (n, dim) float32 L2-normalized matrix."""

    def fit(self, texts: List[str]) -> None:
        """Optional corpus fit (LSA / TF-IDF)."""
        return None

    @property
    def dim(self) -> int:
        return 0

    def info(self) -> dict:
        return {"name": self.name, "dim": self.dim}


class TfidfBackend(EmbeddingBackend):
    """Sparse-like but stored dense; baseline."""

    name = "tfidf"

    def __init__(self):
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}

    @property
    def dim(self) -> int:
        return len(self.vocab)

    def fit(self, texts: List[str]) -> None:
        df: Counter = Counter()
        for t in texts:
            for tok in set(tokenize(t)):
                df[tok] += 1
        n = max(len(texts), 1)
        self.vocab = {tok: i for i, tok in enumerate(sorted(df.keys()))}
        self.idf = {tok: math.log((1 + n) / (1 + c)) + 1.0 for tok, c in df.items()}

    def embed(self, texts: List[str]) -> np.ndarray:
        if not self.vocab:
            self.fit(texts)
        d = max(len(self.vocab), 1)
        mat = np.zeros((len(texts), d), dtype=np.float32)
        for i, t in enumerate(texts):
            counts = Counter(tokenize(t))
            if not counts:
                continue
            max_f = max(counts.values()) or 1
            for tok, c in counts.items():
                j = self.vocab.get(tok)
                if j is None:
                    continue
                mat[i, j] = (0.5 + 0.5 * c / max_f) * self.idf.get(tok, 1.0)
        return l2_normalize(mat)


class LSADenseBackend(EmbeddingBackend):
    """
    Dense semantic vectors via Latent Semantic Analysis:
    TF-IDF matrix → truncated SVD → k-dimensional dense space.

    Learns latent topics from the HEILO knowledge corpus itself
    (no external model download).
    """

    name = "lsa"

    def __init__(self, n_components: int = 64):
        self.n_components = n_components
        self.tfidf = TfidfBackend()
        self.components_: Optional[np.ndarray] = None  # (k, vocab)
        self._dim = n_components

    @property
    def dim(self) -> int:
        return self._dim

    def fit(self, texts: List[str]) -> None:
        self.tfidf.fit(texts)
        X = self.tfidf.embed(texts)  # already L2-normalized TF-IDF
        # Undo normalize for SVD stability — use raw-ish
        X_raw = X * np.maximum(
            np.linalg.norm(
                self.tfidf.embed(texts), axis=1, keepdims=True
            ),  # still normalized; use X as is
            1e-12,
        )
        X_raw = self._tfidf_raw(texts)
        n_docs, n_terms = X_raw.shape
        k = int(min(self.n_components, n_docs, n_terms, max(n_docs - 1, 1)))
        k = max(k, 1)
        # economy SVD
        try:
            # X ≈ U S Vt ; dense docs = U * S
            U, S, Vt = np.linalg.svd(X_raw, full_matrices=False)
            self.components_ = Vt[:k].astype(np.float32)  # (k, terms)
            self._dim = k
            self._singular = S[:k].astype(np.float32)
        except Exception:
            self.components_ = None
            self._dim = n_terms

    def _tfidf_raw(self, texts: List[str]) -> np.ndarray:
        d = max(len(self.tfidf.vocab), 1)
        mat = np.zeros((len(texts), d), dtype=np.float64)
        for i, t in enumerate(texts):
            counts = Counter(tokenize(t))
            if not counts:
                continue
            max_f = max(counts.values()) or 1
            for tok, c in counts.items():
                j = self.tfidf.vocab.get(tok)
                if j is None:
                    continue
                mat[i, j] = (0.5 + 0.5 * c / max_f) * self.tfidf.idf.get(tok, 1.0)
        return mat

    def embed(self, texts: List[str]) -> np.ndarray:
        if self.components_ is None:
            self.fit(texts)
        if self.components_ is None:
            return self.tfidf.embed(texts)
        X = self._tfidf_raw(texts)  # (n, terms)
        # project: X @ Vt.T → (n, k)
        dense = (X @ self.components_.T).astype(np.float32)
        return l2_normalize(dense)


class OllamaDenseBackend(EmbeddingBackend):
    """Dense embeddings via local Ollama (e.g. nomic-embed-text)."""

    name = "ollama"

    def __init__(
        self,
        base_url: str = None,
        model: str = None,
    ):
        self.base_url = (base_url or os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")).rstrip(
            "/"
        )
        self.model = model or os.getenv("HEILO_OLLAMA_EMBED_MODEL", "nomic-embed-text")
        self._dim = 0

    @property
    def dim(self) -> int:
        return self._dim

    def available(self) -> bool:
        try:
            req = urllib.request.Request(
                f"{self.base_url}/api/tags", method="GET"
            )
            with urllib.request.urlopen(req, timeout=2) as resp:
                return resp.status == 200
        except Exception:
            return False

    def embed(self, texts: List[str]) -> np.ndarray:
        vectors = []
        for t in texts:
            body = json.dumps({"model": self.model, "prompt": t}).encode()
            req = urllib.request.Request(
                f"{self.base_url}/api/embeddings",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode())
            emb = data.get("embedding") or data.get("embeddings")
            if emb is None:
                raise RuntimeError(f"Ollama embeddings response missing embedding: {data.keys()}")
            vectors.append(emb)
        mat = np.asarray(vectors, dtype=np.float32)
        self._dim = mat.shape[1]
        return l2_normalize(mat)


class SentenceTransformersBackend(EmbeddingBackend):
    name = "sentence-transformers"

    def __init__(self, model_name: str = None):
        self.model_name = model_name or os.getenv(
            "HEILO_ST_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"
        )
        self._model = None
        self._dim = 0

    @property
    def dim(self) -> int:
        return self._dim

    def available(self) -> bool:
        try:
            import sentence_transformers  # noqa: F401

            return True
        except Exception:
            return False

    def embed(self, texts: List[str]) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        emb = self._model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        mat = np.asarray(emb, dtype=np.float32)
        self._dim = mat.shape[1]
        return mat


def resolve_backend(prefer: str = None) -> EmbeddingBackend:
    """
    prefer: auto | lsa | ollama | st | tfidf | sentence-transformers
    """
    prefer = (prefer or os.getenv("HEILO_EMBED_BACKEND", "auto")).lower()

    if prefer in ("st", "sentence-transformers"):
        b = SentenceTransformersBackend()
        if b.available():
            return b
        print("[Embeddings] sentence-transformers unavailable, falling back")

    if prefer == "ollama":
        b = OllamaDenseBackend()
        if b.available():
            return b
        print("[Embeddings] Ollama unavailable, falling back")

    if prefer == "tfidf":
        return TfidfBackend()

    if prefer == "lsa":
        return LSADenseBackend(
            n_components=int(os.getenv("HEILO_LSA_DIM", "64"))
        )

    # auto
    if os.getenv("HEILO_USE_ST", "0") == "1":
        b = SentenceTransformersBackend()
        if b.available():
            return b
    b = OllamaDenseBackend()
    if b.available():
        return b
    return LSADenseBackend(n_components=int(os.getenv("HEILO_LSA_DIM", "64")))
