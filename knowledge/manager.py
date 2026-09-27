"""
HEILO Knowledge — conhecimento FORA dos pesos do modelo.

    documentos / ensinamentos
        ↓ processamento (chunks)
        ↓ embeddings (rag/embeddings.py — LSA local por padrão)
        ↓ índice (rag/vector_store.py)
        ↓ retrieval (rag/retriever.py)
        ↓ HEILO Core → modelo

Guardar uma informação aqui NÃO exige retreinar nenhum modelo: ela passa a ser
recuperável na hora. Tudo local, funciona offline.
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from heilo.knowledge.store import KnowledgeStore

TAUGHT_CATEGORY = "taught"


class KnowledgeManager:
    def __init__(self, store: Optional[KnowledgeStore] = None, retriever=None):
        self.store = store or KnowledgeStore()
        self._retriever = retriever

    @property
    def root(self) -> Path:
        return self.store.root

    @property
    def retriever(self):
        if self._retriever is None:
            from heilo.rag.retriever import Retriever
            self._retriever = Retriever(self.store)
        return self._retriever

    # ---------------------------------------------------------------- escrever
    def add_document(self, title: str, content: str, category: str = "general") -> Path:
        slug = re.sub(r"[^\w-]+", "_", title.lower()).strip("_")[:60] or "documento"
        path = self.store.add_document(category, slug, content)
        self._index(content, f"{category}/{path.name}")
        return path

    def add_taught(self, question: str, answer: str) -> Path:
        """Registra um ensinamento (/ensinar) como conhecimento recuperável."""
        pasta = self.root / TAUGHT_CATEGORY
        pasta.mkdir(parents=True, exist_ok=True)
        arq = pasta / "ensinamentos.md"
        bloco = (
            f"\n## {question.strip()}\n\n"
            f"{answer.strip()}\n\n"
            f"_Ensinado em {datetime.now():%Y-%m-%d %H:%M}._\n"
        )
        novo = not arq.exists()
        with open(arq, "a", encoding="utf-8", newline="\n") as f:
            if novo:
                f.write("# Ensinamentos da HEILO\n\nRespostas ensinadas pelo usuário com /ensinar.\n")
            f.write(bloco)
        self._index(f"{question.strip()}\n{answer.strip()}", f"{TAUGHT_CATEGORY}/ensinamentos.md")
        return arq

    def _index(self, text: str, source: str) -> None:
        """Indexa só o trecho novo (sem reindexar tudo). Falha de índice não perde o arquivo."""
        try:
            self.retriever.vector.add([text], sources=[source])
        except Exception as e:
            print(f"[HEILO Knowledge] índice não atualizado agora ({e}); o arquivo foi salvo.")

    # ----------------------------------------------------------------- ler
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        return self.retriever.search(query, top_k=top_k)

    def documents(self) -> List[Path]:
        return sorted(p for p in self.root.rglob("*") if p.suffix.lower() in (".md", ".txt"))

    def stats(self) -> Dict:
        docs = self.documents()
        return {"documentos": len(docs), "pasta": str(self.root)}
