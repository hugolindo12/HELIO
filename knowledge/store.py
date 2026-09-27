"""
HEILO Knowledge Store + basic retriever (keyword for v1)
"""
from pathlib import Path
from typing import List, Optional
import re
from heilo.config import KNOWLEDGE_DIR


class KnowledgeStore:
    def __init__(self, root: Path = None):
        self.root = root or KNOWLEDGE_DIR
        self.root.mkdir(parents=True, exist_ok=True)
        for sub in ["general", "programming", "projects", "documentation",
                    "verified_solutions", "agents"]:
            (self.root / sub).mkdir(exist_ok=True)

    def add_document(self, category: str, name: str, content: str):
        cat = self.root / category
        cat.mkdir(exist_ok=True)
        path = cat / f"{name}.md"
        path.write_text(content, encoding="utf-8")
        return path

    def search(self, query: str, limit: int = 5) -> List[str]:
        """
        Very basic keyword search across markdown/txt files.
        Ready to be replaced by embedding + vector DB.
        """
        tokens = set(re.findall(r"\w+", query.lower()))
        if not tokens:
            return []

        scored = []
        for path in self.root.rglob("*"):
            if path.suffix.lower() not in (".md", ".txt", ".py"):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
                lower = text.lower()
                score = sum(1 for t in tokens if t in lower)
                if score > 0:
                    preview = text[:400].replace("\n", " ")
                    scored.append((score, f"[{path.relative_to(self.root)}] {preview}"))
            except Exception:
                continue

        scored.sort(key=lambda x: -x[0])
        return [s[1] for s in scored[:limit]]
