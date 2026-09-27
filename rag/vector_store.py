"""
HEILO Vector Store — Dense RAG

Uses embedding backends from heilo.rag.embeddings (LSA dense by default,
Ollama / sentence-transformers when available).
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from heilo.config import BASE_DIR
from heilo.rag.embeddings import EmbeddingBackend, resolve_backend, tokenize


class LocalVectorStore:
    def __init__(
        self,
        collection: str = "heilo_knowledge",
        root: Path = None,
        backend: Optional[EmbeddingBackend] = None,
        backend_name: Optional[str] = None,
    ):
        self.root = (root or (BASE_DIR / "rag" / "data")) / collection
        self.root.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.root / "documents.json"
        self.emb_path = self.root / "embeddings.npy"
        self.backend_meta_path = self.root / "backend.json"
        self.documents: List[dict] = []
        self.embeddings: Optional[np.ndarray] = None
        self.backend: EmbeddingBackend = backend or resolve_backend(backend_name)
        self._load()

    def _load(self):
        if self.meta_path.exists():
            try:
                self.documents = json.loads(self.meta_path.read_text(encoding="utf-8"))
            except Exception:
                self.documents = []
        if self.emb_path.exists() and self.documents:
            try:
                self.embeddings = np.load(self.emb_path)
                if len(self.embeddings) != len(self.documents):
                    self.embeddings = None
            except Exception:
                self.embeddings = None

    def _save(self):
        self.meta_path.write_text(
            json.dumps(self.documents, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if self.embeddings is not None:
            np.save(self.emb_path, self.embeddings)
        self.backend_meta_path.write_text(
            json.dumps(self.backend.info(), indent=2), encoding="utf-8"
        )

    def _doc_id(self, text: str, source: str) -> str:
        return hashlib.sha256(f"{source}::{text[:200]}".encode()).hexdigest()[:16]

    def _rebuild_embeddings(self):
        texts = [d["text"] for d in self.documents]
        if not texts:
            self.embeddings = None
            return
        self.backend.fit(texts)
        self.embeddings = self.backend.embed(texts)
        self._save()

    def add(
        self,
        texts: List[str],
        sources: Optional[List[str]] = None,
        metadatas: Optional[List[dict]] = None,
    ):
        if not texts:
            return
        sources = sources or ["unknown"] * len(texts)
        metadatas = metadatas or [{}] * len(texts)
        existing = {d["id"] for d in self.documents}
        added = False
        for text, source, meta in zip(texts, sources, metadatas):
            did = self._doc_id(text, source)
            if did in existing:
                continue
            self.documents.append(
                {
                    "id": did,
                    "text": text,
                    "source": source,
                    "metadata": meta,
                    "added_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            existing.add(did)
            added = True
        if added:
            self._rebuild_embeddings()

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.05,
    ) -> List[Tuple[str, float, dict]]:
        if not self.documents:
            return []
        if self.embeddings is None or len(self.embeddings) != len(self.documents):
            self._rebuild_embeddings()
        if self.embeddings is None:
            return []

        q = self.backend.embed([query])[0]
        if q.shape[0] != self.embeddings.shape[1]:
            self._rebuild_embeddings()
            q = self.backend.embed([query])[0]

        scores = self.embeddings @ q
        top_idx = np.argsort(scores)[::-1][: max(top_k * 3, top_k)]
        results = []
        for i in top_idx:
            sc = float(scores[i])
            if sc < min_score:
                continue
            doc = self.documents[i]
            results.append(
                (
                    doc["text"],
                    sc,
                    {
                        "source": doc["source"],
                        "id": doc["id"],
                        "backend": self.backend.name,
                        "dim": int(q.shape[0]),
                        **doc.get("metadata", {}),
                    },
                )
            )
            if len(results) >= top_k:
                break
        return results

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.05,
        keyword_boost: float = 0.12,
    ) -> List[Tuple[str, float, dict]]:
        semantic = self.search(query, top_k=top_k * 2, min_score=0.0)
        q_tokens = set(tokenize(query))
        boosted = []
        for text, score, meta in semantic:
            t_tokens = set(tokenize(text))
            overlap = len(q_tokens & t_tokens) / max(len(q_tokens), 1)
            final = float(score) + keyword_boost * overlap
            meta = {
                **meta,
                "semantic_score": float(score),
                "keyword_overlap": round(overlap, 3),
            }
            boosted.append((text, final, meta))
        boosted.sort(key=lambda x: -x[1])
        return [(t, s, m) for t, s, m in boosted if s >= min_score][:top_k]

    def count(self) -> int:
        return len(self.documents)

    def info(self) -> dict:
        return {
            "documents": len(self.documents),
            "backend": self.backend.name,
            "dim": int(self.embeddings.shape[1])
            if self.embeddings is not None
            else self.backend.dim,
            "embeddings_shape": list(self.embeddings.shape)
            if self.embeddings is not None
            else None,
            "backend_info": self.backend.info(),
        }

    def clear(self):
        self.documents = []
        self.embeddings = None
        for p in (self.meta_path, self.emb_path, self.backend_meta_path):
            if p.exists():
                p.unlink()
