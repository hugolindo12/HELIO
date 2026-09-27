"""
HEILO Task & Execution Log
"""
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import uuid
import json
from pathlib import Path
from heilo.config import LOGS_DIR


@dataclass
class ToolCallRecord:
    tool: str
    arguments: Dict[str, Any]
    result_success: bool
    result_summary: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class TaskLog:
    task_id: str
    user_message: str
    agent: str
    model: str
    status: str = "running"  # running | success | failed | aborted | waiting_permission
    files_analyzed: List[str] = field(default_factory=list)
    tools_used: List[ToolCallRecord] = field(default_factory=list)
    changes: List[Dict[str, Any]] = field(default_factory=list)
    tests: List[Dict[str, Any]] = field(default_factory=list)
    result: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    finished_at: Optional[str] = None
    elapsed_seconds: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_tool(self, tool: str, arguments: dict, success: bool, summary: str):
        self.tools_used.append(ToolCallRecord(
            tool=tool,
            arguments=arguments,
            result_success=success,
            result_summary=summary[:500],
        ))

    def finish(self, status: str, result: str = None):
        self.status = status
        self.result = result
        self.finished_at = datetime.now(timezone.utc).isoformat()
        try:
            start = datetime.fromisoformat(self.started_at)
            end = datetime.fromisoformat(self.finished_at)
            self.elapsed_seconds = (end - start).total_seconds()
        except Exception:
            pass

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def save(self, directory: Path = None):
        directory = directory or LOGS_DIR
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"task_{self.task_id}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return path


def new_task_id() -> str:
    return str(uuid.uuid4())[:8]
