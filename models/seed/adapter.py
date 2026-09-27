"""Adaptador do HEILO Seed para o Model Manager."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from heilo.models.base import ModelAdapter, ModelCard
from heilo.models.seed import SEED_WEIGHTS


class SeedAdapter(ModelAdapter):
    key = "seed"

    def __init__(self, weights: Optional[Path] = None):
        self.weights = Path(weights or SEED_WEIGHTS)
        self._chat = None
        self._erro: Optional[str] = None

    def card(self) -> ModelCard:
        info = self._chat.info if self._chat is not None else {}
        from heilo.models.linhas import nome_modelo
        n = self._chat.modelo.n_params() if self._chat is not None and self._chat.modelo is not None else None
        return ModelCard(
            name=nome_modelo(n),
            role="seed",
            provider="heilo",
            base_model="HEILO Seed (GPT decoder próprio, treinado do zero)",
            adapter="",
            license="Propriedade do projeto HEILO (código e pesos próprios)",
            purpose="chat",
            status="experimental",
            origin=str(self.weights),
            version=str(info.get("passos_totais", "")),
        )

    def availability(self) -> Tuple[bool, str]:
        from heilo.models.seed import gpt
        if not gpt.TORCH_OK:
            return False, "PyTorch não instalado (pip install torch)"
        from heilo.models.seed.gpt import tem_modelo
        if not tem_modelo(self.weights):
            return False, f"pesos do Seed não encontrados em {self.weights} (rode: python -m heilo.main treinar)"
        if self._erro:
            return False, f"falha ao carregar o Seed: {self._erro}"
        return True, "ok"

    def _load(self):
        if self._chat is None:
            from heilo.models.seed.gpt import MiniGPTChat
            try:
                self._chat = MiniGPTChat(arquivo=self.weights)
            except Exception as e:  # arquivo corrompido, versão incompatível...
                self._erro = str(e)
                self._chat = None
        return self._chat

    def generate(self, messages: List[Dict[str, str]], **opts) -> str:
        if not self.available:
            return ""
        chat = self._load()
        if chat is None or not chat.pronto:
            return ""
        # Na conversa: desencoraja repetição (a avaliação usa os padrões desligados).
        # Só com tokenizer BPE: no modelo byte a byte as letras se repetem naturalmente.
        bpe = type(getattr(chat.modelo, "tokenizer", None)).__name__ == "BPETokenizer"
        kw = {"penalidade": 1.2, "sem_repetir": 4} if bpe else {}
        kw.update({k: v for k, v in opts.items()
                   if k in ("temperatura", "max_novos", "penalidade", "sem_repetir")})
        return chat.responder(messages, **kw)

    def reload(self) -> None:
        self._chat = None
        self._erro = None

    def info(self) -> Dict:
        chat = self._load() if self.available else None
        return chat.info if chat is not None else {}
