"""
HEILO Permission System
"""
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, Set


class PermissionLevel(Enum):
    READ = auto()
    WRITE = auto()
    EXECUTE = auto()
    CRITICAL = auto()


@dataclass
class PermissionRequest:
    level: PermissionLevel
    action: str
    target: str
    reason: str
    approved: Optional[bool] = None


class PermissionManager:
    """Controls what the agent is allowed to do."""

    def __init__(self, require_confirmation: bool = True):
        self.require_confirmation = require_confirmation
        self._pending: list[PermissionRequest] = []
        self._session_grants: Set[str] = set()  # "WRITE:path" etc.

    def check(self, level: PermissionLevel, action: str, target: str = "") -> bool:
        """Return True if allowed without confirmation."""
        key = f"{level.name}:{target}"
        if key in self._session_grants:
            return True
        if level in (PermissionLevel.READ,):
            return True
        if level == PermissionLevel.WRITE and not self.require_confirmation:
            return True
        if level == PermissionLevel.EXECUTE and not self.require_confirmation:
            return True
        # CRITICAL always needs confirmation
        return False

    def request(self, level: PermissionLevel, action: str, target: str, reason: str) -> PermissionRequest:
        req = PermissionRequest(level=level, action=action, target=target, reason=reason)
        self._pending.append(req)
        return req

    def approve(self, req: PermissionRequest, grant_session: bool = False):
        req.approved = True
        if grant_session:
            key = f"{req.level.name}:{req.target}"
            self._session_grants.add(key)
        if req in self._pending:
            self._pending.remove(req)

    def deny(self, req: PermissionRequest):
        req.approved = False
        if req in self._pending:
            self._pending.remove(req)

    def get_pending(self) -> list[PermissionRequest]:
        return list(self._pending)

    def clear_session_grants(self):
        self._session_grants.clear()
