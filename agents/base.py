"""
HEILO Generic Agent Base
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
from heilo.core.model_adapter import ModelAdapter, Message
from heilo.tools.registry import ToolRegistry
from heilo.security.permissions import PermissionManager
from heilo.core.task import TaskLog


class BaseAgent(ABC):
    """
    Every HEILO agent has:
    - identity
    - instructions
    - tools
    - permissions
    - memory (optional)
    - knowledge (optional)
    - model (via adapter)
    """

    name: str = "base"
    description: str = ""
    instructions: str = ""

    def __init__(
        self,
        model: ModelAdapter,
        tools: ToolRegistry,
        permissions: PermissionManager,
        memory_store=None,
        knowledge_store=None,
    ):
        self.model = model
        self.tools = tools
        self.permissions = permissions
        self.memory = memory_store
        self.knowledge = knowledge_store

    @abstractmethod
    def run(self, task: str, task_log: TaskLog, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Execute the agent on a task.
        Returns a dict with at least: status, result, details
        """
        ...

    def _system_prompt(self) -> str:
        tool_list = "\n".join(
            f"- {t.name}: {t.description}" for t in self.tools.list_tools()
        )
        return (
            f"You are {self.name}, a specialized agent of the HEILO platform.\n"
            f"{self.description}\n\n"
            f"INSTRUCTIONS:\n{self.instructions}\n\n"
            f"AVAILABLE TOOLS:\n{tool_list}\n\n"
            "When you need to use a tool, respond with a line in the format:\n"
            "TOOL: tool_name key1=value1 key2=value2\n"
            "You may use multiple tools step by step. Think carefully before acting.\n"
            "Always work only inside the authorized workspace.\n"
        )
