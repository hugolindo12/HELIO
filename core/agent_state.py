"""
HEILO Resumable Agent State
Allows Code Agent (and others) to pause on permission requests and resume later.
"""
from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict
from heilo.config import LOGS_DIR
from heilo.core.model_adapter import Message


@dataclass
class AgentState:
    """Serializable state of a running agent cycle."""
    task_id: str
    agent_name: str
    task: str
    phase: str
    attempt: int
    status: str  # running | waiting_permission | success | failed | aborted
    history: List[Dict[str, Any]] = field(default_factory=list)  # serialized Messages
    steps: List[Dict[str, Any]] = field(default_factory=list)
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    pending_permission: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_messages(self) -> List[Message]:
        return [
            Message(
                role=m.get("role", "user"),
                content=m.get("content", ""),
                name=m.get("name"),
                tool_call_id=m.get("tool_call_id"),
            )
            for m in self.history
        ]

    def set_history(self, messages: List[Message]):
        self.history = [
            {
                "role": m.role,
                "content": m.content,
                "name": m.name,
                "tool_call_id": m.tool_call_id,
            }
            for m in messages
        ]
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AgentState":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class AgentStateStore:
    """Persists agent states to disk so they survive permission pauses."""

    def __init__(self, directory: Path = None):
        self.dir = directory or (LOGS_DIR / "agent_states")
        self.dir.mkdir(parents=True, exist_ok=True)

    def save(self, state: AgentState) -> Path:
        state.updated_at = datetime.now(timezone.utc).isoformat()
        path = self.dir / f"{state.task_id}.json"
        try:
            path.write_text(
                json.dumps(state.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as e:
            # Sandbox / filesystem issues should not abort the agent
            print(f"[AgentStateStore] save warning: {e}")
        return path

    def load(self, task_id: str) -> Optional[AgentState]:
        path = self.dir / f"{task_id}.json"
        if not path.exists():
            return None
        try:
            return AgentState.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            return None

    def delete(self, task_id: str):
        path = self.dir / f"{task_id}.json"
        if path.exists():
            path.unlink()

    def list_waiting(self) -> List[AgentState]:
        result = []
        for p in self.dir.glob("*.json"):
            try:
                st = AgentState.from_dict(json.loads(p.read_text(encoding="utf-8")))
                if st.status == "waiting_permission":
                    result.append(st)
            except Exception:
                continue
        return result
