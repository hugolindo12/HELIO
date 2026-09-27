"""
HEILO Research Tools
Knowledge base search, web search (lightweight), and note-taking.
"""
from typing import Optional, List
import json
import urllib.request
import urllib.parse
import re
from heilo.tools.base import BaseTool, ToolResult
from heilo.security.permissions import PermissionLevel
from heilo.tools.web_cache import WebSearchCache


class SearchKnowledgeTool(BaseTool):
    name = "search_knowledge"
    description = (
        "Search the HEILO knowledge base and RAG vector store for relevant documents. "
        "Use for documentation, verified solutions, programming notes."
    )
    required_permission = PermissionLevel.READ
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "top_k": {"type": "integer", "default": 5},
        },
        "required": ["query"],
    }

    def __init__(self, retriever=None):
        self.retriever = retriever

    def execute(self, query: str, top_k: int = 5) -> ToolResult:
        if not self.retriever:
            return ToolResult(success=False, error="Retriever not configured")
        try:
            hits = self.retriever.retrieve(query, top_k=int(top_k))
            return ToolResult(
                success=True,
                output=hits,
                metadata={"count": len(hits), "query": query},
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class SearchWebTool(BaseTool):
    name = "search_web"
    description = (
        "Search the public web for technical information. "
        "Returns titles, URLs and short snippets. Prefer for APIs, errors, docs."
    )
    required_permission = PermissionLevel.READ
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "max_results": {"type": "integer", "default": 5},
        },
        "required": ["query"],
    }

    _cache = WebSearchCache(ttl_seconds=3600)

    # Domains often low-value for technical research
    _BLOCKED = ("pinterest.", "facebook.com", "twitter.com", "x.com", "instagram.", "tiktok.")

    def execute(self, query: str, max_results: int = 5) -> ToolResult:
        """
        Lightweight web search via DuckDuckGo HTML (no API key).
        Uses disk cache and basic quality filtering.
        """
        try:
            max_results = int(max_results)
            cached = self._cache.get(query)
            if cached is not None:
                return ToolResult(
                    success=True,
                    output=cached[:max_results],
                    metadata={"count": len(cached[:max_results]), "query": query, "source": "cache"},
                )

            q = urllib.parse.quote(query)
            url = f"https://html.duckduckgo.com/html/?q={q}"
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "HEILO-Research/0.2 (educational agent)"},
            )
            with urllib.request.urlopen(req, timeout=12) as resp:
                html = resp.read().decode("utf-8", errors="ignore")

            results = []
            for m in re.finditer(
                r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>.*?class="result__snippet"[^>]*>(.*?)</(?:a|td|div)',
                html,
                re.DOTALL | re.IGNORECASE,
            ):
                href, title, snippet = m.group(1), m.group(2), m.group(3)
                title = re.sub(r"<[^>]+>", "", title).strip()
                snippet = re.sub(r"<[^>]+>", "", snippet).strip()
                if "uddg=" in href:
                    parsed = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
                    href = urllib.parse.unquote(parsed.get("uddg", [href])[0])
                if any(b in href.lower() for b in self._BLOCKED):
                    continue
                if len(title) < 5:
                    continue
                results.append({"title": title[:200], "url": href[:300], "snippet": snippet[:300]})
                if len(results) >= max_results * 2:
                    break

            if not results:
                for m in re.finditer(r'href="(https?://[^"]+)"[^>]*>([^<]{10,120})</a>', html):
                    href, title = m.group(1), m.group(2).strip()
                    if "duckduckgo" in href or any(b in href.lower() for b in self._BLOCKED):
                        continue
                    results.append({"title": title, "url": href, "snippet": ""})
                    if len(results) >= max_results:
                        break

            # Prefer results with snippets
            results.sort(key=lambda r: (0 if r.get("snippet") else 1, -len(r.get("snippet") or "")))
            results = results[:max_results]
            self._cache.set(query, results)

            return ToolResult(
                success=True,
                output=results,
                metadata={"count": len(results), "query": query, "source": "duckduckgo"},
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Web search failed: {e}",
                metadata={"query": query},
            )


class FetchUrlTool(BaseTool):
    name = "fetch_url"
    description = "Fetch text content from a public URL (HTML stripped to plain text)."
    required_permission = PermissionLevel.READ
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string"},
            "max_chars": {"type": "integer", "default": 4000},
        },
        "required": ["url"],
    }

    def execute(self, url: str, max_chars: int = 4000) -> ToolResult:
        try:
            if not url.startswith(("http://", "https://")):
                return ToolResult(success=False, error="Only http/https URLs allowed")
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "HEILO-Research/0.2"},
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                raw = resp.read().decode("utf-8", errors="ignore")
            # Strip scripts/styles/tags
            raw = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", raw)
            text = re.sub(r"<[^>]+>", " ", raw)
            text = re.sub(r"\s+", " ", text).strip()
            return ToolResult(
                success=True,
                output=text[: int(max_chars)],
                metadata={"url": url, "length": len(text)},
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class SaveResearchNoteTool(BaseTool):
    name = "save_research_note"
    description = "Save a verified research finding into the HEILO knowledge base."
    required_permission = PermissionLevel.WRITE
    parameters = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "content": {"type": "string"},
            "category": {
                "type": "string",
                "default": "documentation",
                "description": "general|programming|documentation|verified_solutions",
            },
        },
        "required": ["title", "content"],
    }

    def __init__(self, knowledge_store=None, retriever=None):
        self.knowledge = knowledge_store
        self.retriever = retriever

    def execute(self, title: str, content: str, category: str = "documentation") -> ToolResult:
        try:
            if not self.knowledge:
                return ToolResult(success=False, error="Knowledge store not configured")
            safe = re.sub(r"[^\w\-]+", "_", title)[:80]
            path = self.knowledge.add_document(category, safe, f"# {title}\n\n{content}")
            if self.retriever:
                try:
                    self.retriever.add_document(content, source=f"{category}/{safe}.md")
                except Exception:
                    pass
            return ToolResult(
                success=True,
                output=f"Saved note: {path}",
                metadata={"path": str(path), "category": category},
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))
