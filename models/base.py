"""
HEILO Models — contrato comum dos motores de linguagem.

    HEILO Core
        ↓
    Model Manager          (heilo/core/model_manager.py)
        ↓
    ModelAdapter           (esta interface)
        ├── HEILO Seed     (heilo/models/seed)     — modelo próprio, treinado do zero
        ├── HEILO Teacher  (heilo/models/teacher)  — modelo externo temporário [OPCIONAL]
        └── futuro: modelo na HEILO Cloud

O Core só conhece esta interface. Ele nunca importa bibliotecas de modelo
(torch, transformers, peft) nem sabe qual modelo externo está por trás.

Nota: `heilo.core.model_adapter.ModelAdapter` é outra coisa — o provedor de LLM
com tool-calling usado pelos agentes (OpenAI-compatível/stub). Ele não faz parte
da conversa do Core com o Seed/Teacher.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

TEACHER_REQUIRED_MSG = "Esta função requer o HEILO Teacher."


@dataclass
class ModelCard:
    """Metadados de transparência/licença de um motor."""
    name: str                  # nome na arquitetura: "HEILO Seed", "HEILO Teacher"
    role: str                  # seed | teacher | cloud
    provider: str              # heilo (próprio) | external
    base_model: str = ""       # modelo real (ex.: "Qwen/Qwen2.5-0.5B-Instruct")
    adapter: str = ""          # ex.: "LoRA" ou ""
    license: str = ""          # licença do modelo base/pesos
    purpose: str = ""          # ex.: "chat", "training_teacher"
    status: str = ""           # experimental | temporary | ...
    origin: str = ""           # de onde vêm os pesos
    version: str = ""
    extra: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


class ModelAdapter(ABC):
    """Contrato mínimo que todo motor precisa cumprir."""

    #: identificador curto usado pelo Model Manager ("seed", "teacher", ...)
    key: str = ""

    @abstractmethod
    def card(self) -> ModelCard:
        """Metadados (nome, modelo real, licença, finalidade...)."""

    @abstractmethod
    def availability(self) -> Tuple[bool, str]:
        """(disponível?, motivo). Nunca lança exceção."""

    @abstractmethod
    def generate(self, messages: List[Dict[str, str]], **opts) -> str:
        """Gera a próxima resposta do assistente. Retorna "" se não conseguir."""

    def reload(self) -> None:
        """Recarrega pesos do disco (depois de um treino)."""

    # conveniências -------------------------------------------------------
    @property
    def available(self) -> bool:
        return self.availability()[0]

    def status(self) -> Dict:
        ok, motivo = self.availability()
        return {"key": self.key, "available": ok, "reason": motivo, **self.card().to_dict()}


class UnavailableAdapter(ModelAdapter):
    """Marca um componente ausente/desativado sem quebrar o Core."""

    def __init__(self, key: str, name: str, reason: str, role: Optional[str] = None):
        self.key = key
        self._name = name
        self._reason = reason
        self._role = role or key

    def card(self) -> ModelCard:
        return ModelCard(name=self._name, role=self._role, provider="-", status="unavailable")

    def availability(self) -> Tuple[bool, str]:
        return False, self._reason

    def generate(self, messages, **opts) -> str:
        return ""
