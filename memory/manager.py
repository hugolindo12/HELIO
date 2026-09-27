"""
HEILO Memory — memória persistente e LOCAL da plataforma.

Guarda histórico, sessões e trocas de conversa em memory/sessions/*.jsonl.

Memória NÃO é treinamento: conversar nunca altera pesos de modelo.
O caminho até o treino é sempre explícito:

    conversa → Memory → seleção (/aprender) → revisão → dataset → treino opcional

Os arquivos de sessão ficam fora do Git (.gitignore): GitHub não é memória.
"""
from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from heilo.config import MEMORY_DIR


def _agora() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class MemoryManager:
    def __init__(self, root: Optional[Path] = None, session_id: Optional[str] = None,
                 enabled: bool = True):
        self.root = Path(root or (MEMORY_DIR / "sessions"))
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self.enabled = enabled

    # ---------------------------------------------------------------- gravar
    def _arquivo(self) -> Path:
        return self.root / f"{datetime.now():%Y-%m}.jsonl"

    def record(self, user: str, assistant: str, source: str = "", kind: str = "message",
               model: str = "") -> Optional[Dict]:
        """Registra uma troca. Formato compatível com o log antigo (brain/data/conversas)."""
        if not self.enabled or not user or not assistant:
            return None
        item = {
            "sessao": self.session_id,
            "ts": _agora(),
            "user": user,
            "assistant": assistant,
            "fonte": source,
            "tipo": kind,
            "modelo": model,
        }
        self.root.mkdir(parents=True, exist_ok=True)
        with open(self._arquivo(), "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
        return item

    # --------------------------------------------------------------- recuperar
    def iter_all(self) -> Iterable[Dict]:
        if not self.root.exists():
            return
        for arq in sorted(self.root.glob("*.jsonl")):
            with open(arq, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue

    def sessions(self) -> Dict[str, List[Dict]]:
        out: Dict[str, List[Dict]] = {}
        for r in self.iter_all():
            out.setdefault(r.get("sessao", "?"), []).append(r)
        return out

    def recent(self, n: int = 10, session_id: Optional[str] = None) -> List[Dict]:
        rows = [r for r in self.iter_all() if session_id is None or r.get("sessao") == session_id]
        return rows[-n:]

    def search(self, query: str, limit: int = 5) -> List[Dict]:
        """Busca simples por palavras (sem modelo, funciona offline)."""
        tokens = set(re.findall(r"\w+", (query or "").lower()))
        if not tokens:
            return []
        scored = []
        for r in self.iter_all():
            texto = f"{r.get('user', '')} {r.get('assistant', '')}".lower()
            score = sum(1 for t in tokens if t in texto)
            if score:
                scored.append((score, r.get("ts", ""), r))
        scored.sort(key=lambda x: x[1], reverse=True)   # mais recentes primeiro
        scored.sort(key=lambda x: -x[0])                # depois por relevância (estável)
        return [r for _, _, r in scored[:limit]]

    def stats(self) -> Dict:
        rows = list(self.iter_all())
        return {
            "trocas": len(rows),
            "sessoes": len({r.get("sessao") for r in rows}),
            "pasta": str(self.root),
            "gravando": self.enabled,
        }
