"""Utilitários de teste: plataforma HEILO isolada em pasta temporária + motores falsos."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from heilo.models.base import ModelAdapter, ModelCard


class FakeAdapter(ModelAdapter):
    """Motor falso e determinístico (sem torch/transformers)."""

    def __init__(self, key: str, resposta: str = "resposta falsa", disponivel: bool = True):
        self.key = key
        self.resposta = resposta
        self.disponivel = disponivel
        self.chamadas: List[List[Dict]] = []

    def card(self) -> ModelCard:
        return ModelCard(name=f"Fake {self.key}", role=self.key, provider="test",
                         base_model="fake-model", license="test", purpose="test", status="test")

    def availability(self) -> Tuple[bool, str]:
        return (True, "ok") if self.disponivel else (False, "indisponível (teste)")

    def generate(self, messages, **opts) -> str:
        self.chamadas.append(messages)
        return f"{self.resposta}"


def plataforma(tmp_path: Path, modo: str = "auto", teacher_enabled: bool = True,
               teacher_in_chat: bool = False, adapters: Optional[Dict] = None,
               seed_weights: Optional[Path] = None, com_orquestrador: bool = True):
    """Monta Core + Memory + Knowledge + Training em tmp_path. Nada toca os dados reais."""
    from heilo.core.model_manager import ModelManager
    from heilo.knowledge.manager import KnowledgeManager
    from heilo.knowledge.store import KnowledgeStore
    from heilo.memory.manager import MemoryManager
    from heilo.rag.retriever import Retriever
    from heilo.rag.vector_store import LocalVectorStore
    from heilo.training.pipeline import TrainingPipeline

    seed_weights = seed_weights or (tmp_path / "seed.pt")
    models = ModelManager(mode=modo, teacher_enabled=teacher_enabled,
                          teacher_in_chat=teacher_in_chat, seed_weights=seed_weights,
                          adapters=adapters, persona="persona de teste")
    memory = MemoryManager(root=tmp_path / "memory")
    kstore = KnowledgeStore(root=tmp_path / "knowledge")
    retr = Retriever(kstore, vector_store=LocalVectorStore(root=tmp_path / "vec",
                                                          backend_name="tfidf"))
    knowledge = KnowledgeManager(store=kstore, retriever=retr)
    training = TrainingPipeline(data_dir=tmp_path / "data", models=models, memory=memory,
                                knowledge=knowledge, seed_weights=seed_weights)
    orch = None
    if com_orquestrador:
        from heilo.core.orchestrator import Orchestrator
        orch = Orchestrator(models=models, memory_manager=memory,
                            knowledge_manager=knowledge, training=training)
    return orch, models, memory, knowledge, training


def escrever_aprovados(training, pares: List[Tuple[str, str]], origem: str = "curated") -> None:
    from heilo.training.records import append_jsonl, make_example
    append_jsonl(training.approved_file, [
        make_example([{"role": "user", "content": p}, {"role": "assistant", "content": r}], origem)
        for p, r in pares
    ])


def treinar_seed_minusculo(training, passos: int = 3) -> dict:
    """Treino real, mas com um Seed minúsculo (rápido) — requer torch."""
    from heilo.models.seed.gpt import GPTConfig
    escrever_aprovados(training, [(f"pergunta {i}", f"resposta número {i}") for i in range(12)])
    training.build_dataset()
    return training.train_seed(passos=passos, lote=2, log=lambda *_: None, salvar_cada=0,
                               config=GPTConfig(block_size=32, n_layer=1, n_head=2, n_embd=32))


def sha(path: Path) -> str:
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest() if Path(path).exists() else ""


def jsonl(path: Path) -> List[Dict]:
    p = Path(path)
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []
