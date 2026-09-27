"""
HEILO Model Manager — ponte entre o Core e os motores de linguagem.

    HEILO Core → ModelManager → ModelAdapter (seed | teacher | futuro cloud)

O Core pede "responda esta conversa" e recebe texto. Ele não sabe qual biblioteca
ou modelo está por trás. Regras:

- auto:    usa o HEILO Seed. O Teacher só entra se teacher_in_chat=true.
           O Teacher nunca substitui o Seed como cérebro principal por conta própria.
- seed:    só o Seed.
- teacher: só o Teacher (se estiver habilitado e disponível).
- off:     nenhum modelo; o Core usa as respostas prontas / conhecimento local.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from heilo.config import normalize_brain_mode
from heilo.models.base import TEACHER_REQUIRED_MSG, ModelAdapter
from heilo.models import registry


@dataclass
class ModelReply:
    text: str
    model: str = ""        # "seed" | "teacher" | ""
    error: str = ""


class ModelManager:
    def __init__(
        self,
        mode: str = "auto",
        teacher_enabled: bool = True,
        teacher_in_chat: bool = False,
        seed_weights: Optional[Path] = None,
        adapters: Optional[Dict[str, ModelAdapter]] = None,
        persona: Optional[str] = None,
    ):
        self.mode = normalize_brain_mode(mode)
        self.teacher_enabled = teacher_enabled
        self.teacher_in_chat = teacher_in_chat
        self.persona = persona
        self._seed_weights = seed_weights
        # adaptadores injetados (testes, cloud futura) têm prioridade
        self._adapters: Dict[str, ModelAdapter] = dict(adapters or {})

    @classmethod
    def from_config(cls, cfg, **kw) -> "ModelManager":
        from heilo.core.persona import PERSONA_PROMPT
        return cls(
            mode=getattr(cfg, "brain_mode", "auto"),
            teacher_enabled=getattr(cfg, "teacher_enabled", True),
            teacher_in_chat=getattr(cfg, "teacher_in_chat", False),
            persona=PERSONA_PROMPT,
            **kw,
        )

    # ----------------------------------------------------------- adaptadores
    def get(self, key: str) -> ModelAdapter:
        if key not in self._adapters:
            if key == "seed":
                self._adapters[key] = registry.build_seed(self._seed_weights)
            elif key == "teacher":
                self._adapters[key] = registry.build_teacher(enabled=self.teacher_enabled)
            else:
                raise KeyError(f"modelo desconhecido: {key}")
        adapter = self._adapters[key]
        if key == "teacher" and not self.teacher_enabled:
            return registry.build_teacher(enabled=False)
        return adapter

    def is_available(self, key: str) -> bool:
        try:
            return self.get(key).available
        except KeyError:
            return False

    def teacher_available(self) -> bool:
        return self.teacher_enabled and self.is_available("teacher")

    def reload(self) -> None:
        for a in self._adapters.values():
            a.reload()

    # ------------------------------------------------------------------ modo
    def set_mode(self, mode: str) -> str:
        """Troca o modo. Retorna mensagem para o usuário."""
        novo = normalize_brain_mode(mode)
        if novo == "teacher" and not self.teacher_available():
            motivo = self.get("teacher").availability()[1]
            return f"{TEACHER_REQUIRED_MSG} ({motivo}). Modo continua: {self.mode}."
        self.mode = novo
        return f"Modo do cérebro: {self.mode}."

    def _ordem(self, mode: Optional[str]) -> List[str]:
        mode = normalize_brain_mode(mode or self.mode)
        if mode == "off":
            return []
        if mode == "seed":
            return ["seed"]
        if mode == "teacher":
            return ["teacher"]
        ordem = ["seed"]
        if self.teacher_enabled and self.teacher_in_chat:
            ordem.append("teacher")
        return ordem

    # --------------------------------------------------------------- geração
    def _gerar(self, key: str, messages: List[Dict[str, str]], **opts) -> str:
        adapter = self.get(key)
        if not adapter.available:
            return ""
        if key == "teacher" and self.persona:
            opts.setdefault("system", self.persona)
        try:
            return adapter.generate(messages, **opts) or ""
        except Exception as e:  # um motor nunca derruba o Core
            print(f"[ModelManager] {key}: {e}")
            return ""

    def generate(self, messages: List[Dict[str, str]], mode: Optional[str] = None,
                 **opts) -> ModelReply:
        """Resposta do melhor motor disponível para o modo. Texto vazio = use o fallback."""
        for key in self._ordem(mode):
            texto = self._gerar(key, messages, **opts)
            if texto:
                return ModelReply(text=texto, model=key)
        return ModelReply(text="", model="", error="nenhum modelo disponível")

    def generate_with(self, key: str, messages: List[Dict[str, str]], **opts) -> ModelReply:
        """Usa um motor específico (ex.: o Training pedindo exemplos ao Teacher)."""
        if key == "teacher" and not self.teacher_available():
            return ModelReply(text="", model="teacher", error=TEACHER_REQUIRED_MSG)
        texto = self._gerar(key, messages, **opts)
        return ModelReply(text=texto, model=key, error="" if texto else "sem resposta")

    def compare(self, messages: List[Dict[str, str]]) -> Dict[str, str]:
        """Ferramenta de AVALIAÇÃO: mesma conversa nos dois motores. Não decide quem está certo."""
        out: Dict[str, str] = {}
        for key in ("seed", "teacher"):
            if key == "teacher" and not self.teacher_available():
                out[key] = TEACHER_REQUIRED_MSG
                continue
            adapter = self.get(key)
            if not adapter.available:
                out[key] = f"(indisponível: {adapter.availability()[1]})"
                continue
            texto = self._gerar(key, messages)
            if not texto:
                ok, motivo = adapter.availability()
                texto = "(sem resposta)" if ok else f"(indisponível: {motivo})"
            out[key] = texto
        return out

    # ---------------------------------------------------------------- status
    def status(self) -> Dict:
        return {
            "mode": self.mode,
            "teacher_enabled": self.teacher_enabled,
            "teacher_in_chat": self.teacher_in_chat,
            "seed": self.get("seed").status(),
            "teacher": self.get("teacher").status(),
        }
