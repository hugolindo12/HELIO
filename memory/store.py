"""
HEILO Basic Memory Store
"""
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from heilo.config import MEMORY_DIR


class MemoryStore:
    def __init__(self, root: Path = None):
        self.root = root or MEMORY_DIR
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "conversations").mkdir(exist_ok=True)
        (self.root / "verified_solutions").mkdir(exist_ok=True)
        (self.root / "user_preferences").mkdir(exist_ok=True)

    def save_conversation(self, session_id: str, messages: List[Dict]):
        path = self.root / "conversations" / f"{session_id}.json"
        path.write_text(json.dumps({
            "session_id": session_id,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "messages": messages,
        }, indent=2, ensure_ascii=False), encoding="utf-8")

    def maybe_store_solution(self, problem: str, result: Dict[str, Any]):
        """
        Only store if the task succeeded and we have meaningful changes/tests.
        Classification: generated vs verified.
        """
        if result.get("status") != "success":
            return
        if not result.get("steps"):
            return

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "problem": problem[:500],
            "status": "verified" if any(
                s.get("tool") in ("run_tests", "run_command") and s.get("success")
                for s in result.get("steps", [])
            ) else "generated",
            "result_preview": str(result.get("result", ""))[:1000],
            "tools_used": [s.get("tool") for s in result.get("steps", [])],
        }
        fname = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + ".json"
        path = self.root / "verified_solutions" / fname
        path.write_text(json.dumps(entry, indent=2, ensure_ascii=False), encoding="utf-8")

    def list_solutions(self, limit: int = 10) -> List[Dict]:
        sols = []
        for p in sorted((self.root / "verified_solutions").glob("*.json"), reverse=True)[:limit]:
            try:
                sols.append(json.loads(p.read_text(encoding="utf-8")))
            except Exception:
                pass
        return sols

    def _prefs_path(self) -> Path:
        return self.root / "user_preferences" / "settings.json"

    def get_preferences(self) -> Dict[str, Any]:
        path = self._prefs_path()
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def set_preference(self, key: str, value: Any) -> Dict[str, Any]:
        prefs = self.get_preferences()
        prefs[key] = value
        prefs["updated_at"] = datetime.now(timezone.utc).isoformat()
        path = self._prefs_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(prefs, indent=2, ensure_ascii=False), encoding="utf-8")
        return prefs

    def get_timezone(self, default: str = "America/Sao_Paulo") -> str:
        return str(self.get_preferences().get("timezone") or default)
