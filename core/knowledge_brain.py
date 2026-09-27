"""
HEILO Knowledge Brain
Answers from *owned* knowledge first, without calling an external LLM.

Goal: the repository (knowledge + memory + verified solutions) is the primary
source of understanding. The LLM is optional fallback for synthesis/generation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class BrainAnswer:
    found: bool
    confidence: float  # 0..1
    answer: str
    sources: List[str] = field(default_factory=list)
    mode: str = "knowledge"  # knowledge | verified_solution | hybrid | none


class KnowledgeBrain:
    """
    Lightweight local understanding over HEILO's own stores.
    No external model required for high-confidence hits.
    """

    def __init__(self, retriever=None, memory=None, knowledge=None, learning=None):
        self.retriever = retriever
        self.memory = memory
        self.knowledge = knowledge
        self.learning = learning

    def understand(self, query: str, top_k: int = 5) -> BrainAnswer:
        query = (query or "").strip()
        if not query:
            return BrainAnswer(found=False, confidence=0.0, answer="", mode="none")

        sources: List[str] = []
        chunks: List[str] = []
        scores: List[float] = []

        # 1) Verified solutions (highest trust)
        if self.memory:
            try:
                sols = self.memory.list_solutions(limit=20)
                q_tokens = set(self._tokens(query))
                for sol in sols:
                    text = f"{sol.get('problem', '')} {sol.get('result_preview', '')}"
                    overlap = self._overlap(q_tokens, self._tokens(text))
                    if overlap >= 0.25 and sol.get("status") == "verified":
                        chunks.append(
                            f"[verified] {sol.get('problem', '')[:200]}\n"
                            f"{sol.get('result_preview', '')[:500]}"
                        )
                        sources.append(f"memory/verified_solutions ({sol.get('status')})")
                        scores.append(min(0.95, 0.5 + overlap))
            except Exception:
                pass

        # 1b) Learning store (errors + corrections)
        if self.learning:
            try:
                for hit in self.learning.search(query, limit=5):
                    conf = 0.7 if hit.get("confidence") == "verified" else 0.45
                    chunks.append(
                        f"[learning:{hit.get('kind')}] {hit.get('problem', '')[:200]}\n"
                        f"error: {hit.get('error_message', '')[:200]}\n"
                        f"fix: {hit.get('correction', '')[:400]}"
                    )
                    sources.append(f"memory/learning/{hit.get('entry_id')}")
                    scores.append(conf)
            except Exception:
                pass

        # 2) RAG / knowledge files
        if self.retriever:
            try:
                hits = self.retriever.retrieve(query, top_k=top_k)
                for h in hits:
                    # parse optional score= from retriever formatting
                    m = re.search(r"score=([0-9.]+)", h)
                    sc = float(m.group(1)) if m else 0.35
                    chunks.append(h)
                    sources.append(h.split("]")[0].replace("[", "") if "]" in h else "knowledge")
                    scores.append(sc)
            except Exception:
                pass

        # 3) Keyword scan on knowledge store directly
        if self.knowledge and not chunks:
            try:
                hits = self.knowledge.search(query, limit=top_k)
                for h in hits:
                    chunks.append(h)
                    sources.append("knowledge")
                    scores.append(0.3)
            except Exception:
                pass

        if not chunks:
            return BrainAnswer(found=False, confidence=0.0, answer="", sources=[], mode="none")

        conf = max(scores) if scores else 0.0
        # Boost if multiple agreeing sources
        if len(chunks) >= 2 and conf >= 0.25:
            conf = min(0.98, conf + 0.1)

        answer = self._synthesize_local(query, chunks)
        mode = "verified_solution" if any("verified" in s for s in sources) else "knowledge"
        if any("verified" in s for s in sources) and any("knowledge" in s or "/" in s for s in sources):
            mode = "hybrid"

        return BrainAnswer(
            found=conf >= 0.2,
            confidence=round(conf, 3),
            answer=answer,
            sources=list(dict.fromkeys(sources))[:12],
            mode=mode,
        )

    def _synthesize_local(self, query: str, chunks: List[str]) -> str:
        """Extractive synthesis — no LLM."""
        lines = [f"**Conhecimento HEILO** (sem modelo externo)\n"]
        lines.append(f"Pergunta: {query}\n")
        lines.append("Trechos relevantes do repositório:\n")
        for i, c in enumerate(chunks[:5], 1):
            excerpt = re.sub(r"\s+", " ", c)[:400]
            lines.append(f"{i}. {excerpt}")
        lines.append(
            "\n_Resposta baseada apenas em knowledge/memory locais. "
            "Use um modelo só se precisar de síntese criativa ou código novo._"
        )
        return "\n".join(lines)

    @staticmethod
    def _tokens(text: str) -> List[str]:
        return re.findall(r"[a-zA-ZÀ-ÿ0-9_]{3,}", (text or "").lower())

    @staticmethod
    def _overlap(a: set, b) -> float:
        b = set(b)
        if not a or not b:
            return 0.0
        return len(a & b) / max(len(a), 1)
