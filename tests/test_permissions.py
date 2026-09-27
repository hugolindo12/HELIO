"""Unit tests for heilo.security.permissions"""
import pytest
from heilo.security.permissions import (
    PermissionLevel,
    PermissionManager,
    PermissionRequest,
)


class TestPermissionManager:
    def test_read_always_allowed(self):
        pm = PermissionManager(require_confirmation=True)
        assert pm.check(PermissionLevel.READ, "read_file", "a.py") is True

    def test_write_requires_confirmation_by_default(self):
        pm = PermissionManager(require_confirmation=True)
        assert pm.check(PermissionLevel.WRITE, "write_file", "a.py") is False

    def test_write_allowed_without_confirmation(self):
        pm = PermissionManager(require_confirmation=False)
        assert pm.check(PermissionLevel.WRITE, "write_file", "a.py") is True
        assert pm.check(PermissionLevel.EXECUTE, "run_tests", "") is True

    def test_critical_always_blocked_until_approve(self):
        pm = PermissionManager(require_confirmation=False)
        assert pm.check(PermissionLevel.CRITICAL, "rm", "/") is False

    def test_request_and_approve(self):
        pm = PermissionManager(require_confirmation=True)
        req = pm.request(PermissionLevel.WRITE, "edit_file", "main.py", "fix bug")
        assert req in pm.get_pending()
        assert req.approved is None
        pm.approve(req)
        assert req.approved is True
        assert req not in pm.get_pending()

    def test_approve_session_grant(self):
        pm = PermissionManager(require_confirmation=True)
        req = pm.request(PermissionLevel.WRITE, "edit_file", "main.py", "fix")
        pm.approve(req, grant_session=True)
        assert pm.check(PermissionLevel.WRITE, "edit_file", "main.py") is True
        # different target still blocked
        assert pm.check(PermissionLevel.WRITE, "edit_file", "other.py") is False

    def test_deny(self):
        pm = PermissionManager(require_confirmation=True)
        req = pm.request(PermissionLevel.EXECUTE, "run_command", "ls", "list")
        pm.deny(req)
        assert req.approved is False
        assert req not in pm.get_pending()

    def test_clear_session_grants(self):
        pm = PermissionManager(require_confirmation=True)
        req = pm.request(PermissionLevel.WRITE, "w", "f.py", "r")
        pm.approve(req, grant_session=True)
        pm.clear_session_grants()
        assert pm.check(PermissionLevel.WRITE, "w", "f.py") is False

    def test_get_pending_copy(self):
        pm = PermissionManager()
        pm.request(PermissionLevel.WRITE, "a", "t", "r")
        pending = pm.get_pending()
        pending.clear()
        assert len(pm.get_pending()) == 1
