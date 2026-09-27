"""
HEILO Teacher via Ollama (modelo externo rodando no PC, ex.: qwen3-vl:8b).

Mesmo papel do Teacher do Hugging Face: gerar exemplos NÃO verificados para revisão.
Não é a HEILO e não entra nos pesos dela. Usa a API local do Ollama
(http://localhost:11434), sem internet. Escolha com HEILO_TEACHER_BACKEND=ollama.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from heilo.models.base import ModelAdapter, ModelCard
from heilo.models.teacher.adapter import TEACHER_DIR, load_metadata

META_OLLAMA = TEACHER_DIR / "teacher_ollama.json"


class OllamaTeacherAdapter(ModelAdapter):
    key = "teacher"

    def __init__(self, metadata_path: Optional[Path] = None, url: Optional[str] = None):
        self.meta = load_metadata(metadata_path or META_OLLAMA)
        self.url = (url or os.getenv("HEILO_OLLAMA_URL") or self.meta.get("url") or "http://localhost:11434").rstrip("/")
        self.modelo = os.getenv("HEILO_OLLAMA_MODEL") or self.meta.get("ollama_model", "")
        self._erro: Optional[str] = None

    def card(self) -> ModelCard:
        m = self.meta
        return ModelCard(name=m.get("name", "HEILO Teacher (Ollama)"), role="teacher",
                         provider="external (Ollama, local)", base_model=self.modelo, adapter="nenhum",
                         license=m.get("license", ""), purpose="training_teacher",
                         status=m.get("status", "temporary"), origin=m.get("origin", ""),
                         version=m.get("version", ""))

    def _pedir(self, caminho: str, dados: Optional[Dict] = None, tempo: float = 5.0) -> Dict:
        req = urllib.request.Request(self.url + caminho, method="POST" if dados is not None else "GET",
                                     data=json.dumps(dados).encode() if dados is not None else None,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=tempo) as r:
            return json.loads(r.read().decode("utf-8"))

    def availability(self) -> Tuple[bool, str]:
        if not self.modelo:
            return False, "modelo do Ollama não definido (teacher_ollama.json ou HEILO_OLLAMA_MODEL)"
        try:
            nomes = [m.get("name", "") for m in self._pedir("/api/tags").get("models", [])]
        except (urllib.error.URLError, OSError, ValueError) as e:
            return False, f"Ollama não está rodando em {self.url} ({e}). Abra o Ollama."
        if not any(n == self.modelo or n.split(":")[0] == self.modelo for n in nomes):
            return False, f"modelo {self.modelo} não está no Ollama (instalados: {', '.join(nomes) or 'nenhum'})"
        return True, "ok"

    def generate(self, messages: List[Dict[str, str]], system: Optional[str] = None,
                 temperatura: float = 0.5, max_novos: int = 256, **_) -> str:
        conversa = ([{"role": "system", "content": system}] if system else []) + [
            m for m in messages if m.get("role") in ("user", "assistant")][-10:]
        corpo = {"model": self.modelo, "messages": conversa, "stream": False, "think": False,
                 "options": {"temperature": temperatura, "num_predict": max(64, max_novos)}}
        try:
            r = self._pedir("/api/chat", corpo, tempo=float(os.getenv("HEILO_OLLAMA_TIMEOUT", "300")))
        except (urllib.error.URLError, OSError, ValueError) as e:
            self._erro = str(e)
            print(f"[HEILO Teacher/Ollama] {e}")
            return ""
        texto = (r.get("message") or {}).get("content", "")
        return re.sub(r"<think>.*?</think>", "", texto, flags=re.S).strip()
