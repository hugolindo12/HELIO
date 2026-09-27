"""
HEILO Learning Store
Records errors, corrections and outcomes so the system accumulates knowledge
from real usage (self-improvement over time).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from heilo.config import MEMORY_DIR, KNOWLEDGE_DIR, BASE_DIR


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug(text: str, max_len: int = 60) -> str:
    s = re.sub(r"[^\w\-]+", "_", (text or "").lower().strip())
    return s[:max_len].strip("_") or "entry"


@dataclass
class LearningEntry:
    """One learning unit: problem → investigation → fix → outcome."""
    entry_id: str
    created_at: str
    kind: str  # error | correction | verified_solution | insight
    problem: str
    error_message: str = ""
    cause: str = ""
    correction: str = ""
    files: List[str] = field(default_factory=list)
    tests_passed: Optional[bool] = None
    project: str = ""
    agent: str = ""
    source: str = "runtime"  # runtime | user | research | import
    confidence: str = "generated"  # generated | tested | verified
    tags: List[str] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_markdown(self) -> str:
        lines = [
            f"# {self.kind}: {self.problem[:80]}",
            "",
            f"- **id:** `{self.entry_id}`",
            f"- **created:** {self.created_at}",
            f"- **confidence:** {self.confidence}",
            f"- **project:** {self.project or '-'}",
            f"- **agent:** {self.agent or '-'}",
            f"- **tests_passed:** {self.tests_passed}",
            "",
            "## Problem",
            self.problem,
            "",
        ]
        if self.error_message:
            lines += ["## Error", "```", self.error_message[:2000], "```", ""]
        if self.cause:
            lines += ["## Cause", self.cause, ""]
        if self.correction:
            lines += ["## Correction", self.correction, ""]
        if self.files:
            lines += ["## Files", *[f"- `{f}`" for f in self.files], ""]
        if self.tags:
            lines += ["## Tags", ", ".join(self.tags), ""]
        return "\n".join(lines)


class LearningStore:
    """
    Persists learning entries in:
      memory/learning/*.json
      knowledge/verified_solutions/*.md   (when verified)
    Ready to sync to a GitHub knowledge folder.
    """

    def __init__(self, root: Path = None, knowledge_dir: Path = None):
        self.root = Path(root or (MEMORY_DIR / "learning"))
        self.root.mkdir(parents=True, exist_ok=True)
        self.knowledge_dir = Path(knowledge_dir or (KNOWLEDGE_DIR / "verified_solutions"))
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)

    def _new_id(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")

    def record(self, entry: LearningEntry) -> Path:
        path = self.root / f"{entry.entry_id}.json"
        path.write_text(
            json.dumps(entry.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        # Promote verified knowledge into the knowledge repo tree
        if entry.confidence == "verified" or entry.tests_passed is True:
            md_name = f"{entry.entry_id}_{_slug(entry.problem)}.md"
            md_path = self.knowledge_dir / md_name
            md_path.write_text(entry.to_markdown(), encoding="utf-8")
        return path

    def record_from_task(
        self,
        problem: str,
        result: Dict[str, Any],
        project: str = "",
        agent: str = "",
    ) -> Optional[LearningEntry]:
        """
        Extract learning from a CODE/TEST/PIPELINE task result.
        """
        status = result.get("status")
        steps = result.get("steps") or []
        raw = str(result.get("result") or result.get("content") or "")

        errors = []
        files = []
        tests_passed = None
        for s in steps:
            tool = s.get("tool") or ""
            if tool in ("run_tests", "run_command", "build_project"):
                if s.get("success"):
                    tests_passed = True if tests_passed is None else tests_passed
                else:
                    tests_passed = False
                    errors.append(str(s.get("output_preview") or s.get("error") or "test failed"))
            if tool in ("edit_file", "write_file"):
                args = s.get("args") or {}
                if args.get("path"):
                    files.append(str(args["path"]))
            if not s.get("success") and tool:
                errors.append(f"{tool}: {s.get('output_preview', '')[:200]}")

        if status == "success" and (files or tests_passed):
            conf = "verified" if tests_passed else "tested"
            kind = "verified_solution"
        elif status in ("aborted", "failed") or tests_passed is False:
            conf = "generated"
            kind = "error"
        else:
            # skip noise
            if not files and not errors:
                return None
            conf = "generated"
            kind = "insight"

        entry = LearningEntry(
            entry_id=self._new_id(),
            created_at=_utc_iso(),
            kind=kind,
            problem=problem[:800],
            error_message="\n".join(errors)[:2000],
            cause="",
            correction=raw[:2000] if kind != "error" else "",
            files=list(dict.fromkeys(files)),
            tests_passed=tests_passed,
            project=project,
            agent=agent or str(result.get("agent") or ""),
            source="runtime",
            confidence=conf,
            tags=self._auto_tags(problem, files),
            raw={"status": status, "step_count": len(steps)},
        )
        self.record(entry)
        return entry

    def record_error_and_fix(
        self,
        problem: str,
        error_message: str,
        correction: str,
        files: Optional[List[str]] = None,
        tests_passed: bool = False,
        project: str = "",
        agent: str = "HEILO_CODE",
    ) -> LearningEntry:
        conf = "verified" if tests_passed else "tested"
        entry = LearningEntry(
            entry_id=self._new_id(),
            created_at=_utc_iso(),
            kind="verified_solution" if tests_passed else "correction",
            problem=problem[:800],
            error_message=(error_message or "")[:2000],
            correction=(correction or "")[:2000],
            files=files or [],
            tests_passed=tests_passed,
            project=project,
            agent=agent,
            source="runtime",
            confidence=conf,
            tags=self._auto_tags(problem, files or []),
        )
        self.record(entry)
        return entry

    def list_entries(self, limit: int = 50, kind: Optional[str] = None) -> List[Dict]:
        items = []
        for p in sorted(self.root.glob("*.json"), reverse=True):
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                if kind and d.get("kind") != kind:
                    continue
                items.append(d)
            except Exception:
                continue
            if len(items) >= limit:
                break
        return items

    def search(self, query: str, limit: int = 10) -> List[Dict]:
        tokens = set(re.findall(r"\w{3,}", query.lower()))
        scored = []
        for d in self.list_entries(limit=200):
            blob = " ".join([
                d.get("problem", ""),
                d.get("error_message", ""),
                d.get("correction", ""),
                " ".join(d.get("files") or []),
            ]).lower()
            score = sum(1 for t in tokens if t in blob)
            if score:
                scored.append((score, d))
        scored.sort(key=lambda x: -x[0])
        return [d for _, d in scored[:limit]]

    def _auto_tags(self, problem: str, files: List[str]) -> List[str]:
        tags = []
        pl = (problem or "").lower()
        for word in ("discount", "desconto", "sketch", "pytest", "async", "rag", "permission"):
            if word in pl:
                tags.append(word)
        for f in files:
            if f.endswith(".py"):
                tags.append("python")
            if f.endswith(".cs"):
                tags.append("csharp")
        return list(dict.fromkeys(tags))[:12]
