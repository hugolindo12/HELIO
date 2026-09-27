"""
HEILO Tool Registry
"""
from typing import Dict, List, Optional
from heilo.tools.base import BaseTool, ToolSchema, ToolResult
from heilo.security.permissions import PermissionManager, PermissionLevel


class ToolRegistry:
    def __init__(self, permission_manager: Optional[PermissionManager] = None):
        self._tools: Dict[str, BaseTool] = {}
        self.permissions = permission_manager or PermissionManager()

    def register(self, tool: BaseTool):
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)

    def list_tools(self) -> List[ToolSchema]:
        return [t.schema() for t in self._tools.values()]

    def list_names(self) -> List[str]:
        return list(self._tools.keys())

    def execute(self, name: str, **kwargs) -> ToolResult:
        tool = self.get(name)
        if not tool:
            return ToolResult(success=False, error=f"Tool not found: {name}")

        # Permission check
        if not self.permissions.check(tool.required_permission, name, kwargs.get("path", "")):
            req = self.permissions.request(
                tool.required_permission,
                action=name,
                target=str(kwargs.get("path", kwargs.get("command", ""))),
                reason=f"Tool '{name}' requires {tool.required_permission.name}",
            )
            return ToolResult(
                success=False,
                error="PERMISSION_REQUIRED",
                metadata={"permission_request": {
                    "level": req.level.name,
                    "action": req.action,
                    "target": req.target,
                    "reason": req.reason,
                }},
            )

        try:
            return tool.execute(**kwargs)
        except Exception as e:
            return ToolResult(success=False, error=str(e))
