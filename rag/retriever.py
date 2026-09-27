"""
HEILO RAG Retriever
Semantic (+ hybrid) search over knowledge/ and learning notes.
"""
from __future__ import annotations

from typing import List, Optional, Dict, Any

from heilo.knowledge.store import KnowledgeStore
from heilo.rag.vector_store import LocalVectorStore


class Retriever:
    def __init__(
        self,
        knowledge: Optional[KnowledgeStore] = None,
        vector_store: Optional[LocalVectorStore] = None,
        auto_index: bool = True,
        mode: str = "hybrid",  # hybrid | semantic | keyword
        min_score: float = 0.08,
    ):
        self.knowledge = knowledge or KnowledgeStore()
        self.vector = vector_store or LocalVectorStore()
        self.mode = mode
        self.min_score = min_score
        self._indexed = False
        if auto_index:
            try:
                self.index_knowledge()
            except Exception as e:
                print(f"[Retriever] Initial index skipped: {e}")

    def index_knowledge(self, force: bool = False):
        """Scan knowledge/ (and verified solutions) into the vector store."""
        if self._indexed and not force:
            return
        if force:
            self.vector.clear()
        texts, sources = [], []
        root = self.knowledge.root
        for path in root.rglob("*"):
            if path.suffix.lower() not in (".md", ".txt"):
                # skip .py of HEILO itself if accidentally under knowledge
                if path.suffix.lower() == ".py" and path.name in (
                    "store.py",
                    "git_sync.py",
                    "__init__.py",
                ):
                    continue
                if path.suffix.lower() != ".py":
                    continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
                if len(text.strip()) < 20:
                    continue
                rel = str(path.relative_to(root))
                for i, chunk in enumerate(self._chunk(text, max_chars=900)):
                    texts.append(chunk)
                    sources.append(rel + (f"#chunk{i}" if i else ""))
            except Exception:
                continue
        if texts:
            self.vector.add(texts, sources=sources)
        self._indexed = True

    def _chunk(self, text: str, max_chars: int = 900) -> List[str]:
        if len(text) <= max_chars:
            return [text]
        chunks = []
        paragraphs = text.split("\n\n")
        current = ""
        for p in paragraphs:
            if len(current) + len(p) + 2 <= max_chars:
                current = (current + "\n\n" + p).strip()
            else:
                if current:
                    chunks.append(current)
                # hard-split long paragraphs
                while len(p) > max_chars:
                    chunks.append(p[:max_chars])
                    p = p[max_chars:]
                current = p
        if current:
            chunks.append(current)
        return chunks or [text[:max_chars]]

    def search(
        self,
        query: str,
        top_k: int = 5,
        mode: Optional[str] = None,
        min_score: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Structured semantic search.
        Returns list of {text, score, source, backend, ...}
        """
        mode = mode or self.mode
        min_score = self.min_score if min_score is None else min_score
        results: List[Dict[str, Any]] = []

        if mode in ("hybrid", "semantic"):
            try:
                if mode == "hybrid":
                    hits = self.vector.hybrid_search(
                        query, top_k=top_k, min_score=min_score
                    )
                else:
                    hits = self.vector.search(query, top_k=top_k, min_score=min_score)
                for text, score, meta in hits:
                    results.append(
                        {
                            "text": text,
                            "score": round(float(score), 4),
                            "source": meta.get("source", ""),
                            "backend": meta.get("backend", "tfidf"),
                            "semantic_score": meta.get("semantic_score"),
                            "keyword_overlap": meta.get("keyword_overlap"),
                        }
                    )
            except Exception as e:
                print(f"[Retriever] semantic search error: {e}")

        if len(results) < top_k and mode in ("hybrid", "keyword"):
            for kw in self.knowledge.search(query, limit=top_k - len(results)):
                results.append(
                    {
                        "text": kw,
                        "score": 0.2,
                        "source": "keyword",
                        "backend": "keyword",
                    }
                )

        return results[:top_k]

    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        """String list for agents (backward compatible)."""
        hits = self.search(query, top_k=top_k)
        out = []
        for h in hits:
            src = h.get("source") or ""
            sc = h.get("score", 0)
            out.append(f"[{src}] (score={sc:.2f}) {h['text'][:500]}")
        return out

    def info(self) -> dict:
        info = self.vector.info()
        info["mode"] = self.mode
        info["min_score"] = self.min_score
        info["indexed"] = self._indexed
        return info
