"""
HEILO Filesystem Tools
"""
from typing import Optional
from pathlib import Path
import re
from heilo.tools.base import BaseTool, ToolResult
from heilo.security.permissions import PermissionLevel
from heilo.security.workspace import WorkspaceManager, WorkspaceError


class ListFilesTool(BaseTool):
    name = "list_files"
    description = "List files in the workspace. Optionally filter by pattern or subpath."
    required_permission = PermissionLevel.READ
    parameters = {
        "type": "object",
        "properties": {
            "subpath": {"type": "string", "description": "Relative subdirectory", "default": "."},
            "pattern": {"type": "string", "description": "Glob pattern", "default": "*"},
        },
    }

    def __init__(self, workspace: WorkspaceManager):
        self.ws = workspace

    def execute(self, subpath: str = ".", pattern: str = "*") -> ToolResult:
        try:
            files = self.ws.list_files(subpath, pattern)
            return ToolResult(success=True, output=files, metadata={"count": len(files)})
        except WorkspaceError as e:
            return ToolResult(success=False, error=str(e))


class ReadFileTool(BaseTool):
    name = "read_file"
    description = "Read the content of a file inside the workspace."
    required_permission = PermissionLevel.READ
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Relative path to the file"},
        },
        "required": ["path"],
    }

    def __init__(self, workspace: WorkspaceManager):
        self.ws = workspace

    def execute(self, path: str) -> ToolResult:
        try:
            content = self.ws.read_text(path)
            return ToolResult(
                success=True,
                output=content,
                metadata={"path": path, "length": len(content)},
            )
        except WorkspaceError as e:
            return ToolResult(success=False, error=str(e))
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class SearchCodeTool(BaseTool):
    name = "search_code"
    description = "Search for a text or regex pattern across files in the workspace."
    required_permission = PermissionLevel.READ
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Text or regex to search"},
            "file_pattern": {"type": "string", "default": "*"},
            "case_sensitive": {"type": "boolean", "default": False},
            "max_results": {"type": "integer", "default": 30},
        },
        "required": ["query"],
    }

    def __init__(self, workspace: WorkspaceManager):
        self.ws = workspace

    def execute(
        self,
        query: str,
        file_pattern: str = "*",
        case_sensitive: bool = False,
        max_results: int = 30,
    ) -> ToolResult:
        try:
            flags = 0 if case_sensitive else re.IGNORECASE
            try:
                pattern = re.compile(query, flags)
            except re.error:
                pattern = re.compile(re.escape(query), flags)

            matches = []
            files = self.ws.list_files(".", file_pattern)
            for rel in files:
                try:
                    text = self.ws.read_text(rel)
                    for i, line in enumerate(text.splitlines(), 1):
                        if pattern.search(line):
                            matches.append({
                                "file": rel,
                                "line": i,
                                "content": line.strip()[:200],
                            })
                            if len(matches) >= max_results:
                                break
                except Exception:
                    continue
                if len(matches) >= max_results:
                    break

            return ToolResult(
                success=True,
                output=matches,
                metadata={"count": len(matches), "query": query},
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class WriteFileTool(BaseTool):
    name = "write_file"
    description = "Create or overwrite a file with the given content."
    required_permission = PermissionLevel.WRITE
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["path", "content"],
    }

    def __init__(self, workspace: WorkspaceManager, checkpoint_manager=None):
        self.ws = workspace
        self.checkpoints = checkpoint_manager

    def execute(self, path: str, content: str) -> ToolResult:
        try:
            existed = self.ws.exists(path)
            original = None
            if existed:
                try:
                    original = self.ws.read_text(path)
                except Exception:
                    original = ""
            if self.checkpoints:
                self.checkpoints.snapshot_file(path, original, existed=existed)

            p = self.ws.write_text(path, content)
            return ToolResult(
                success=True,
                output=f"Written {len(content)} bytes to {path}",
                metadata={"path": str(p), "size": len(content), "checkpointed": bool(self.checkpoints)},
            )
        except WorkspaceError as e:
            return ToolResult(success=False, error=str(e))
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class EditFileTool(BaseTool):
    name = "edit_file"
    description = "Replace a specific string in a file (old_string → new_string)."
    required_permission = PermissionLevel.WRITE
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "old_string": {"type": "string"},
            "new_string": {"type": "string"},
            "replace_all": {"type": "boolean", "default": False},
        },
        "required": ["path", "old_string", "new_string"],
    }

    def __init__(self, workspace: WorkspaceManager, checkpoint_manager=None):
        self.ws = workspace
        self.checkpoints = checkpoint_manager

    def execute(
        self,
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> ToolResult:
        try:
            content = self.ws.read_text(path)
            if old_string not in content:
                # Handle CRLF vs LF line ending mismatch (common on Windows)
                norm_content = content.replace("\r\n", "\n")
                norm_old = old_string.replace("\r\n", "\n")
                if norm_old in norm_content:
                    norm_new = new_string.replace("\r\n", "\n")
                    if replace_all:
                        new_content = norm_content.replace(norm_old, norm_new)
                        count = norm_content.count(norm_old)
                    else:
                        new_content = norm_content.replace(norm_old, norm_new, 1)
                        count = 1
                    if "\r\n" in content:
                        new_content = new_content.replace("\n", "\r\n")
                    if self.checkpoints:
                        self.checkpoints.snapshot_file(path, content, existed=True)
                    self.ws.write_text(path, new_content)
                    return ToolResult(
                        success=True,
                        output=f"Replaced {count} occurrence(s) in {path} (normalized line endings)",
                        metadata={"path": path, "replacements": count, "checkpointed": bool(self.checkpoints)},
                    )

                return ToolResult(
                    success=False,
                    error=f"old_string not found in {path}",
                )
            if self.checkpoints:
                self.checkpoints.snapshot_file(path, content, existed=True)

            if replace_all:
                new_content = content.replace(old_string, new_string)
                count = content.count(old_string)
            else:
                new_content = content.replace(old_string, new_string, 1)
                count = 1
            self.ws.write_text(path, new_content)
            return ToolResult(
                success=True,
                output=f"Replaced {count} occurrence(s) in {path}",
                metadata={"path": path, "replacements": count, "checkpointed": bool(self.checkpoints)},
            )
        except WorkspaceError as e:
            return ToolResult(success=False, error=str(e))
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class FindSymbolTool(BaseTool):
    name = "find_symbol"
    description = "Find definitions of a class, function or variable by name."
    required_permission = PermissionLevel.READ
    parameters = {
        "type": "object",
        "properties": {
            "symbol": {"type": "string"},
            "file_pattern": {"type": "string", "default": "*.py"},
        },
        "required": ["symbol"],
    }

    def __init__(self, workspace: WorkspaceManager):
        self.ws = workspace

    def execute(self, symbol: str, file_pattern: str = "*.py") -> ToolResult:
        # Simple regex-based symbol finder (good enough for v1)
        patterns = [
            rf"^\s*(def|class|async def)\s+{re.escape(symbol)}\b",
            rf"^\s*{re.escape(symbol)}\s*=",
        ]
        matches = []
        files = self.ws.list_files(".", file_pattern)
        for rel in files:
            try:
                text = self.ws.read_text(rel)
                for i, line in enumerate(text.splitlines(), 1):
                    for pat in patterns:
                        if re.search(pat, line):
                            matches.append({
                                "file": rel,
                                "line": i,
                                "content": line.strip()[:200],
                            })
            except Exception:
                continue
        return ToolResult(success=True, output=matches, metadata={"count": len(matches)})
