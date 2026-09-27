"""
HEILO Execution Tools (run_command, tests, build)
"""
import subprocess
import shlex
import time
from typing import Optional, List
from heilo.tools.base import BaseTool, ToolResult
from heilo.security.permissions import PermissionLevel
from heilo.security.workspace import WorkspaceManager
from heilo.config import config


class RunCommandTool(BaseTool):
    name = "run_command"
    description = "Execute a shell command inside the workspace directory. Use with caution."
    required_permission = PermissionLevel.EXECUTE
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command to run"},
            "timeout": {"type": "integer", "default": 60},
        },
        "required": ["command"],
    }

    # Dangerous patterns that escalate to CRITICAL
    CRITICAL_PATTERNS = [
        r"\brm\s+-rf\b", r"\bdel\s+/[sf]\b", r"\bformat\b",
        r"\bmkfs\b", r"\bdd\s+if=", r">\s*/dev/",
        r"\bshutdown\b", r"\breboot\b", r"\bchmod\s+777\b",
    ]

    def __init__(self, workspace: WorkspaceManager):
        self.ws = workspace
        self._command_count = 0

    def execute(self, command: str, timeout: int = 60) -> ToolResult:
        self._command_count += 1
        if self._command_count > config.limits.max_commands:
            return ToolResult(
                success=False,
                error=f"Command limit exceeded ({config.limits.max_commands})",
            )

        # Detect critical commands
        import re
        for pat in self.CRITICAL_PATTERNS:
            if re.search(pat, command, re.IGNORECASE):
                return ToolResult(
                    success=False,
                    error="CRITICAL_COMMAND",
                    metadata={
                        "permission_request": {
                            "level": "CRITICAL",
                            "action": "run_command",
                            "target": command,
                            "reason": "Potentially destructive command detected",
                        }
                    },
                )

        try:
            start = time.time()
            result = subprocess.run(
                command,
                shell=True,
                cwd=str(self.ws.root),
                capture_output=True,
                text=True,
                timeout=min(timeout, config.limits.timeout_seconds),
            )
            elapsed = time.time() - start
            output = (result.stdout or "") + (result.stderr or "")
            return ToolResult(
                success=result.returncode == 0,
                output=output.strip()[:8000],
                error=None if result.returncode == 0 else f"Exit code {result.returncode}",
                metadata={
                    "returncode": result.returncode,
                    "elapsed": round(elapsed, 2),
                    "command": command,
                },
            )
        except subprocess.TimeoutExpired:
            return ToolResult(success=False, error="Command timed out")
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class RunTestsTool(BaseTool):
    name = "run_tests"
    description = "Run project tests (pytest, unittest, or custom command)."
    required_permission = PermissionLevel.EXECUTE
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Test command (default: pytest -q)",
                "default": "python -m pytest -q",
            },
            "timeout": {"type": "integer", "default": 120},
        },
    }

    def __init__(self, workspace: WorkspaceManager):
        self.runner = RunCommandTool(workspace)

    def execute(self, command: str = "python -m pytest -q", timeout: int = 120) -> ToolResult:
        return self.runner.execute(command=command, timeout=timeout)


class BuildProjectTool(BaseTool):
    name = "build_project"
    description = "Build the project (e.g. python setup.py, npm run build, etc.)."
    required_permission = PermissionLevel.EXECUTE
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "default": "python -m compileall .",
            },
            "timeout": {"type": "integer", "default": 180},
        },
    }

    def __init__(self, workspace: WorkspaceManager):
        self.runner = RunCommandTool(workspace)

    def execute(self, command: str = "python -m compileall .", timeout: int = 180) -> ToolResult:
        return self.runner.execute(command=command, timeout=timeout)
