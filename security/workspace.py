"""
HEILO Workspace Security Layer
"""
from pathlib import Path
from typing import Optional, Union
import re
from heilo.config import config, WORKSPACE_ROOT
from heilo.security.permissions import PermissionLevel


class WorkspaceError(Exception):
    pass


class WorkspaceManager:
    """
    Ensures all file operations stay inside an authorized workspace.
    """

    def __init__(self, root: Optional[Path] = None):
        self.root = (root or config.security.default_workspace).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._blocked = [re.compile(p, re.IGNORECASE) for p in config.security.blocked_patterns]

    def set_workspace(self, path: Union[str, Path]):
        p = Path(path).resolve()
        if not p.exists():
            p.mkdir(parents=True, exist_ok=True)
        self.root = p

    def resolve(self, relative_or_abs: Union[str, Path], must_exist: bool = False) -> Path:
        """
        Resolve a path safely inside the workspace.
        Rejects path traversal and absolute paths outside root.
        """
        raw = Path(relative_or_abs)

        # Block obvious dangerous patterns
        s = str(raw)
        for pat in self._blocked:
            if pat.search(s):
                raise WorkspaceError(f"Path blocked by security policy: {s}")

        if raw.is_absolute():
            if not config.security.allow_absolute_paths:
                # Only allow if it is already under root
                try:
                    resolved = raw.resolve()
                    resolved.relative_to(self.root)
                    return resolved
                except ValueError:
                    raise WorkspaceError(
                        f"Absolute path outside workspace not allowed: {raw}"
                    )
            resolved = raw.resolve()
        else:
            resolved = (self.root / raw).resolve()

        # Final containment check
        try:
            resolved.relative_to(self.root)
        except ValueError:
            raise WorkspaceError(
                f"Path escapes workspace ({self.root}): {resolved}"
            )

        if must_exist and not resolved.exists():
            raise WorkspaceError(f"Path does not exist: {resolved}")

        return resolved

    def is_inside(self, path: Path) -> bool:
        try:
            path.resolve().relative_to(self.root)
            return True
        except ValueError:
            return False

    def list_files(self, subpath: str = ".", pattern: str = "*") -> list[str]:
        base = self.resolve(subpath)
        if not base.is_dir():
            return []
        results = []
        for p in base.rglob(pattern):
            if p.is_file():
                rel = p.relative_to(self.root)
                results.append(str(rel).replace("\\", "/"))
        return sorted(results)

    def read_text(self, path: str, encoding: str = "utf-8") -> str:
        p = self.resolve(path, must_exist=True)
        return p.read_text(encoding=encoding, errors="replace")

    def write_text(self, path: str, content: str, encoding: str = "utf-8") -> Path:
        p = self.resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding=encoding)
        return p

    def exists(self, path: str) -> bool:
        try:
            return self.resolve(path).exists()
        except WorkspaceError:
            return False

    def get_info(self) -> dict:
        return {
            "root": str(self.root),
            "exists": self.root.exists(),
            "file_count": len(self.list_files()),
        }
