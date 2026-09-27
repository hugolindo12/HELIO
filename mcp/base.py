"""
HEILO MCP Layer (Model Context Protocol style)
MCP is a TOOL LAYER, not the brain of the AI.
Future: HL-3D, Blender, SAP, Windows, etc. register here.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from heilo.tools.base import BaseTool, ToolResult, ToolSchema
from heilo.security.permissions import PermissionLevel


@dataclass
class MCPServerInfo:
    name: str
    description: str
    version: str = "0.1.0"
    capabilities: List[str] = field(default_factory=list)


class MCPServer(ABC):
    """
    Abstract MCP server. Each external system (Blender, SAP, HL-3D...)
    implements this and exposes tools to HEILO.
    """

    name: str = "mcp_base"
    description: str = ""
    version: str = "0.1.0"

    @abstractmethod
    def list_tools(self) -> List[ToolSchema]:
        ...

    @abstractmethod
    def call_tool(self, name: str, arguments: Dict[str, Any]) -> ToolResult:
        ...

    def info(self) -> MCPServerInfo:
        return MCPServerInfo(
            name=self.name,
            description=self.description,
            version=self.version,
            capabilities=[t.name for t in self.list_tools()],
        )

    def is_available(self) -> bool:
        return True


class MCPToolAdapter(BaseTool):
    """Wraps an MCP server tool as a native HEILO tool."""

    def __init__(
        self,
        server: MCPServer,
        tool_name: str,
        description: str,
        parameters: Dict[str, Any],
        required_permission: PermissionLevel = PermissionLevel.EXECUTE,
    ):
        self.server = server
        self.name = f"mcp_{server.name}_{tool_name}"
        self._mcp_tool_name = tool_name
        self.description = f"[MCP:{server.name}] {description}"
        self.parameters = parameters
        self.required_permission = required_permission

    def execute(self, **kwargs) -> ToolResult:
        try:
            return self.server.call_tool(self._mcp_tool_name, kwargs)
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class MCPRegistry:
    """
    Central registry of MCP servers.
    HEILO discovers tools from all registered servers.
    """

    def __init__(self):
        self._servers: Dict[str, MCPServer] = {}

    def register(self, server: MCPServer):
        self._servers[server.name] = server

    def unregister(self, name: str):
        self._servers.pop(name, None)

    def get(self, name: str) -> Optional[MCPServer]:
        return self._servers.get(name)

    def list_servers(self) -> List[MCPServerInfo]:
        return [s.info() for s in self._servers.values()]

    def all_tools(self) -> List[MCPToolAdapter]:
        tools = []
        for server in self._servers.values():
            if not server.is_available():
                continue
            for schema in server.list_tools():
                tools.append(MCPToolAdapter(
                    server=server,
                    tool_name=schema.name,
                    description=schema.description,
                    parameters=schema.parameters,
                    required_permission=schema.required_permission,
                ))
        return tools

    def call(self, server_name: str, tool_name: str, arguments: dict) -> ToolResult:
        server = self.get(server_name)
        if not server:
            return ToolResult(success=False, error=f"MCP server not found: {server_name}")
        return server.call_tool(tool_name, arguments)
