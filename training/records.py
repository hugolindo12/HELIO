"""Formato dos exemplos de treino da HEILO + utilitários JSONL."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

# Origem de cada exemplo — rastreável até o dataset e o treino.
ORIGINS = ("curated", "taught", "teacher", "memory")


def agora() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def example_id(messages: List[Dict[str, str]]) -> str:
    """ID determinístico pelo conteúdo: o mesmo exemplo sempre tem o mesmo id (dedupe)."""
    norm = [(m.get("role", ""), " ".join((m.get("content") or "").split()).lower())
            for m in messages]
    return hashlib.sha1(json.dumps(norm, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


def make_example(messages: List[Dict[str, str]], origin: str, meta: Optional[Dict] = None) -> Dict:
    return {
        "id": example_id(messages),
        "origin": origin,
        "messages": messages,
        "created": agora(),
        "meta": meta or {},
    }


def read_jsonl(path: Path) -> List[Dict]:
    path = Path(path)
    if not path.exists():
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def append_jsonl(path: Path, rows: Iterable[Dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def write_jsonl(path: Path, rows: Iterable[Dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    n = 0
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    tmp.replace(path)
    return n
