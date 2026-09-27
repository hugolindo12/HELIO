"""
HEILO Checkpoint & Rollback System
Creates snapshots before WRITE operations and restores on failure.
"""
from __future__ import annotations
import json
import shutil
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict


@dataclass
class FileSnapshot:
    path: str          # relative path inside workspace
    content: str
    existed: bool      # False if file was created by the change


@dataclass
class Checkpoint:
    checkpoint_id: str
    task_id: str
    created_at: str
    files: List[FileSnapshot] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Checkpoint":
        files = [FileSnapshot(**f) for f in d.get("files", [])]
        return cls(
            checkpoint_id=d["checkpoint_id"],
            task_id=d["task_id"],
            created_at=d["created_at"],
            files=files,
            metadata=d.get("metadata", {}),
        )


class CheckpointManager:
    """
    Manages file-level checkpoints inside <workspace>/.heilo/checkpoints/
    """

    def __init__(self, workspace_root: Path):
        self.root = Path(workspace_root)
        self.dir = self.root / ".heilo" / "checkpoints"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._active: Optional[Checkpoint] = None

    def _new_id(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")

    def start(self, task_id: str, metadata: Optional[dict] = None) -> Checkpoint:
        cp = Checkpoint(
            checkpoint_id=self._new_id(),
            task_id=task_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            metadata=metadata or {},
        )
        self._active = cp
        return cp

    def snapshot_file(self, relative_path: str, current_content: Optional[str], existed: bool):
        """Record original content of a file before it is modified."""
        if self._active is None:
            self.start(task_id="anonymous")
        # Avoid duplicate snapshots of the same file in one checkpoint
        for f in self._active.files:
            if f.path == relative_path:
                return
        self._active.files.append(FileSnapshot(
            path=relative_path,
            content=current_content if current_content is not None else "",
            existed=existed,
        ))

    def commit(self) -> Optional[Path]:
        """Persist the active checkpoint to disk."""
        if self._active is None or not self._active.files:
            self._active = None
            return None
        path = self.dir / f"{self._active.checkpoint_id}.json"
        path.write_text(
            json.dumps(self._active.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        saved = self._active
        self._active = None
        return path

    def rollback(self, checkpoint: Optional[Checkpoint] = None) -> List[str]:
        """
        Restore files from the given (or last active/saved) checkpoint.
        Returns list of restored relative paths.
        """
        cp = checkpoint or self._active
        if cp is None:
            # try last saved
            files = sorted(self.dir.glob("*.json"), reverse=True)
            if not files:
                return []
            cp = Checkpoint.from_dict(json.loads(files[0].read_text(encoding="utf-8")))

        restored = []
        for snap in cp.files:
            target = self.root / snap.path
            if not snap.existed:
                # File was created by the agent → delete it
                if target.exists():
                    target.unlink()
                    restored.append(f"deleted:{snap.path}")
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with open(target, "w", encoding="utf-8", newline="") as f:
                    f.write(snap.content)
                restored.append(f"restored:{snap.path}")
        self._active = None
        return restored

    def list_checkpoints(self, limit: int = 20) -> List[dict]:
        result = []
        for p in sorted(self.dir.glob("*.json"), reverse=True)[:limit]:
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                result.append({
                    "id": d.get("checkpoint_id"),
                    "task_id": d.get("task_id"),
                    "created_at": d.get("created_at"),
                    "files": [f["path"] for f in d.get("files", [])],
                })
            except Exception:
                continue
        return result
