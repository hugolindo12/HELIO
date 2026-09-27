"""
HEILO Git Tools
"""
from heilo.tools.base import BaseTool, ToolResult
from heilo.security.permissions import PermissionLevel
from heilo.security.workspace import WorkspaceManager
from heilo.tools.execution import RunCommandTool


class GetGitDiffTool(BaseTool):
    name = "get_git_diff"
    description = "Show git diff of the current workspace (if it is a git repo)."
    required_permission = PermissionLevel.READ
    parameters = {
        "type": "object",
        "properties": {
            "staged": {"type": "boolean", "default": False},
        },
    }

    def __init__(self, workspace: WorkspaceManager):
        self.runner = RunCommandTool(workspace)

    def execute(self, staged: bool = False) -> ToolResult:
        cmd = "git diff --cached" if staged else "git diff"
        result = self.runner.execute(command=cmd, timeout=30)
        if not result.success and "not a git repository" in (result.error or result.output or ""):
            return ToolResult(success=True, output="Not a git repository", metadata={"is_git": False})
        return result
