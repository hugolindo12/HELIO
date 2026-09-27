"""
Built-in MCP server implementations (examples + stubs for future).
"""
from typing import Any, Dict, List
from heilo.mcp.base import MCPServer
from heilo.tools.base import ToolSchema, ToolResult
from heilo.security.permissions import PermissionLevel
import platform
import os
from datetime import datetime, timezone


class SystemInfoMCPServer(MCPServer):
    """Exposes safe system information tools."""

    name = "system"
    description = "Safe system information (OS, time, env summary)"
    version = "0.1.0"

    def list_tools(self) -> List[ToolSchema]:
        return [
            ToolSchema(
                name="get_os_info",
                description="Get operating system name, version and architecture",
                parameters={"type": "object", "properties": {}},
                required_permission=PermissionLevel.READ,
            ),
            ToolSchema(
                name="get_time",
                description="Get current UTC time",
                parameters={"type": "object", "properties": {}},
                required_permission=PermissionLevel.READ,
            ),
            ToolSchema(
                name="list_env_keys",
                description="List environment variable NAMES only (not values)",
                parameters={"type": "object", "properties": {}},
                required_permission=PermissionLevel.READ,
            ),
        ]

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> ToolResult:
        if name == "get_os_info":
            return ToolResult(success=True, output={
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
                "python": platform.python_version(),
            })
        if name == "get_time":
            return ToolResult(success=True, output=datetime.now(timezone.utc).isoformat() + "Z")
        if name == "list_env_keys":
            keys = sorted(os.environ.keys())[:50]
            return ToolResult(success=True, output=keys, metadata={"count": len(keys)})
        return ToolResult(success=False, error=f"Unknown tool: {name}")


class HL3DMCPServer(MCPServer):
    """
    Stub for future HL-3D / CAD integration.
    Tools are declared but return 'not connected' until the real bridge exists.
    """

    name = "hl3d"
    description = "HL-3D CAD integration (stub – connect real bridge later)"
    version = "0.0.1"
    connected: bool = False

    def is_available(self) -> bool:
        return True  # visible, but tools report not connected

    def list_tools(self) -> List[ToolSchema]:
        return [
            ToolSchema(
                name="list_sketches",
                description="List sketches in the current HL-3D project",
                parameters={"type": "object", "properties": {}},
                required_permission=PermissionLevel.READ,
            ),
            ToolSchema(
                name="get_sketch",
                description="Get sketch data by name",
                parameters={
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
                required_permission=PermissionLevel.READ,
            ),
            ToolSchema(
                name="update_sketch",
                description="Update sketch geometry or properties",
                parameters={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "data": {"type": "object"},
                    },
                    "required": ["name", "data"],
                },
                required_permission=PermissionLevel.WRITE,
            ),
        ]

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> ToolResult:
        if not self.connected:
            return ToolResult(
                success=False,
                error="HL-3D MCP bridge not connected. Start the HL-3D plugin and register the bridge.",
                metadata={"server": "hl3d", "connected": False},
            )
        return ToolResult(success=False, error=f"Tool {name} not implemented in stub")


class BlenderMCPServer(MCPServer):
    """Stub for future Blender MCP integration."""

    name = "blender"
    description = "Blender 3D integration (stub)"
    version = "0.0.1"
    connected: bool = False

    def list_tools(self) -> List[ToolSchema]:
        return [
            ToolSchema(
                name="list_objects",
                description="List objects in the active Blender scene",
                parameters={"type": "object", "properties": {}},
                required_permission=PermissionLevel.READ,
            ),
            ToolSchema(
                name="run_script",
                description="Execute a Python script inside Blender",
                parameters={
                    "type": "object",
                    "properties": {"script": {"type": "string"}},
                    "required": ["script"],
                },
                required_permission=PermissionLevel.CRITICAL,
            ),
        ]

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> ToolResult:
        return ToolResult(
            success=False,
            error="Blender MCP bridge not connected.",
            metadata={"server": "blender", "connected": False},
        )
