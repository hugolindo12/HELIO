"""
HEILO Knowledge → GitHub sync

Keeps knowledge/ + memory/learning/ + verified_solutions as a git repo
so the user can push to GitHub and own all accumulated knowledge.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from heilo.config import BASE_DIR, KNOWLEDGE_DIR, MEMORY_DIR


class KnowledgeGitSync:
    """
    Manage a git repository dedicated to HEILO knowledge.

    Typical layout (can be the whole heilo/ or a dedicated knowledge-repo/):
      knowledge/
      memory/learning/
      memory/verified_solutions/   (if mirrored)
      README.md
    """

    def __init__(self, repo_root: Optional[Path] = None):
        # Default: use heilo/ as repo root so knowledge + memory are tracked
        self.repo_root = Path(repo_root or BASE_DIR).resolve()

    def _run(self, args: List[str], timeout: int = 60) -> Dict:
        try:
            r = subprocess.run(
                args,
                cwd=str(self.repo_root),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return {
                "ok": r.returncode == 0,
                "stdout": (r.stdout or "").strip(),
                "stderr": (r.stderr or "").strip(),
                "code": r.returncode,
            }
        except Exception as e:
            return {"ok": False, "stdout": "", "stderr": str(e), "code": -1}

    def is_git_repo(self) -> bool:
        return (self.repo_root / ".git").exists()

    def init_repo(self, remote_url: Optional[str] = None) -> Dict:
        """Initialize git repo if needed and optionally set remote origin."""
        results = []
        if not self.is_git_repo():
            results.append(self._run(["git", "init"]))
            # sensible defaults
            gitignore = self.repo_root / ".gitignore"
            if not gitignore.exists():
                gitignore.write_text(
                    "\n".join([
                        "__pycache__/",
                        "*.pyc",
                        ".pytest_cache/",
                        "htmlcov/",
                        ".coverage",
                        "logs/",
                        "rag/data/**/*.npy",
                        "workspace/**/.heilo/",
                        ".env",
                        "",
                    ]),
                    encoding="utf-8",
                )
            readme = self.repo_root / "KNOWLEDGE_REPO.md"
            if not readme.exists():
                readme.write_text(
                    "# HEILO Knowledge Repository\n\n"
                    "Este repositório guarda o conhecimento acumulado da HEILO:\n\n"
                    "- `knowledge/` — documentos, manuais, soluções verificadas\n"
                    "- `memory/learning/` — erros, correções e aprendizados em JSON\n"
                    "- `memory/verified_solutions/` — soluções com teste PASS\n\n"
                    "A HEILO consulta este material antes de depender de modelos externos.\n",
                    encoding="utf-8",
                )
        if remote_url:
            # set or update origin
            check = self._run(["git", "remote", "get-url", "origin"])
            if check["ok"]:
                results.append(self._run(["git", "remote", "set-url", "origin", remote_url]))
            else:
                results.append(self._run(["git", "remote", "add", "origin", remote_url]))
        return {
            "ok": all(r.get("ok", True) for r in results) if results else True,
            "repo": str(self.repo_root),
            "is_git": self.is_git_repo(),
            "steps": results,
        }

    def status(self) -> Dict:
        if not self.is_git_repo():
            return {"ok": False, "is_git": False, "message": "Not a git repository"}
        st = self._run(["git", "status", "--porcelain"])
        branch = self._run(["git", "branch", "--show-current"])
        remote = self._run(["git", "remote", "-v"])
        return {
            "ok": True,
            "is_git": True,
            "branch": branch.get("stdout") or "",
            "dirty": bool(st.get("stdout")),
            "changes": st.get("stdout") or "",
            "remotes": remote.get("stdout") or "",
            "repo": str(self.repo_root),
        }

    def commit_knowledge(self, message: Optional[str] = None) -> Dict:
        """Stage knowledge-related paths and commit."""
        if not self.is_git_repo():
            init = self.init_repo()
            if not init.get("ok") and not self.is_git_repo():
                return {"ok": False, "error": "git init failed", "detail": init}

        paths = [
            "knowledge",
            "memory/learning",
            "memory/verified_solutions",
            "memory/user_preferences",
            # dados de treino APROVADOS/registrados e métricas (não a memória de conversas)
            "data/taught",
            "data/approved",
            "data/rejected",
            "data/teacher",
            "data/training",
            # pesos + model card do HEILO Seed (release do modelo próprio)
            "models/seed/weights",
            "models/seed/MODEL_CARD.md",
            "KNOWLEDGE_REPO.md",
        ]
        # Only add paths that exist
        existing = [p for p in paths if (self.repo_root / p).exists()]
        if not existing:
            return {"ok": False, "error": "No knowledge paths to commit"}

        add = self._run(["git", "add"] + existing)
        msg = message or f"heilo: update knowledge {__import__('datetime').datetime.now().isoformat(timespec='seconds')}"
        commit = self._run(["git", "commit", "-m", msg])
        # commit fails with 1 if nothing to commit — treat as soft ok
        nothing = "nothing to commit" in (commit.get("stdout") + commit.get("stderr")).lower()
        return {
            "ok": commit.get("ok") or nothing,
            "added": existing,
            "add": add,
            "commit": commit,
            "nothing_to_commit": nothing,
        }

    def push(self, remote: str = "origin", branch: Optional[str] = None) -> Dict:
        if not self.is_git_repo():
            return {"ok": False, "error": "Not a git repository"}
        if not branch:
            b = self._run(["git", "branch", "--show-current"])
            branch = b.get("stdout") or "main"
        return self._run(["git", "push", "-u", remote, branch], timeout=120)

    def sync(self, message: Optional[str] = None, push: bool = False) -> Dict:
        """commit_knowledge + optional push."""
        commit = self.commit_knowledge(message=message)
        result = {"commit": commit}
        if push and commit.get("ok"):
            result["push"] = self.push()
        return result
