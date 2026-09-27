"""
HEILO Tool Base Classes
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Callable
from heilo.security.permissions import PermissionLevel


@dataclass
class ToolResult:
    success: bool
    output: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "metadata": self.metadata,
        }


@dataclass
class ToolSchema:
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON-schema like
    required_permission: PermissionLevel = PermissionLevel.READ


class BaseTool(ABC):
    name: str = "base"
    description: str = ""
    required_permission: PermissionLevel = PermissionLevel.READ
    parameters: Dict[str, Any] = {}

    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
            required_permission=self.required_permission,
        )

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        ...

    def __call__(self, **kwargs) -> ToolResult:
        return self.execute(**kwargs)
