"""
Simple disk cache for web search results.
"""
from __future__ import annotations
import json
import hashlib
import time
from pathlib import Path
from typing import Any, Optional
from heilo.config import BASE_DIR


class WebSearchCache:
    def __init__(self, ttl_seconds: int = 3600, root: Path = None):
        self.ttl = ttl_seconds
        self.root = root or (BASE_DIR / "rag" / "web_cache")
        self.root.mkdir(parents=True, exist_ok=True)

    def _key(self, query: str) -> str:
        return hashlib.sha256(query.strip().lower().encode()).hexdigest()[:24]

    def get(self, query: str) -> Optional[Any]:
        path = self.root / f"{self._key(query)}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if time.time() - data.get("ts", 0) > self.ttl:
                path.unlink(missing_ok=True)
                return None
            return data.get("results")
        except Exception:
            return None

    def set(self, query: str, results: Any):
        path = self.root / f"{self._key(query)}.json"
        try:
            path.write_text(
                json.dumps({"ts": time.time(), "query": query, "results": results}, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:
            pass
