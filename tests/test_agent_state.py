"""
Unit tests for heilo.core.agent_state

Covers:
- AgentState serialization (Message ↔ dict)
- from_dict field filtering
- AgentStateStore save / load / delete
- list_waiting filter
- resume-oriented status transitions
- OSError resilience on save
"""
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone

import pytest

from heilo.core.agent_state import AgentState, AgentStateStore
from heilo.core.model_adapter import Message


@pytest.fixture
def store(tmp_path):
    return AgentStateStore(directory=tmp_path / "agent_states")


def _sample_state(**overrides) -> AgentState:
    base = dict(
        task_id="task01",
        agent_name="HEILO_CODE",
        task="fix discount bug",
        phase="INSPECT",
        attempt=1,
        status="running",
    )
    base.update(overrides)
    return AgentState(**base)


class TestAgentState:
    """In-memory AgentState behavior."""

    def test_defaults(self):
        st = _sample_state()
        assert st.history == []
        assert st.steps == []
        assert st.pending_permission is None
        assert st.metadata == {}
        assert st.started_at
        assert st.updated_at

    def test_set_history_and_to_messages_roundtrip(self):
        st = _sample_state()
        msgs = [
            Message(role="system", content="You are CODE"),
            Message(role="user", content="fix it"),
            Message(role="assistant", content="TOOL: list_files"),
            Message(role="user", content="TOOL_RESULT (list_files): ok", name="tool"),
        ]
        st.set_history(msgs)
        assert len(st.history) == 4
        assert st.history[0]["role"] == "system"
        assert st.history[1]["content"] == "fix it"

        restored = st.to_messages()
        assert len(restored) == 4
        assert all(isinstance(m, Message) for m in restored)
        assert restored[0].role == "system"
        assert restored[2].content == "TOOL: list_files"
        assert restored[3].name == "tool"

    def test_set_history_updates_timestamp(self):
        st = _sample_state()
        before = st.updated_at
        st.set_history([Message(role="user", content="x")])
        assert st.updated_at >= before

    def test_to_dict_contains_core_fields(self):
        st = _sample_state(status="waiting_permission", phase="APPLY")
        st.pending_permission = {"level": "WRITE", "action": "edit_file"}
        st.steps = [{"tool": "read_file", "success": True}]
        d = st.to_dict()
        assert d["task_id"] == "task01"
        assert d["status"] == "waiting_permission"
        assert d["phase"] == "APPLY"
        assert d["pending_permission"]["action"] == "edit_file"
        assert d["steps"][0]["tool"] == "read_file"

    def test_from_dict_roundtrip(self):
        st = _sample_state(attempt=3, status="waiting_permission")
        st.set_history([Message(role="user", content="hello")])
        st.steps = [{"tool": "list_files"}]
        st2 = AgentState.from_dict(st.to_dict())
        assert st2.task_id == st.task_id
        assert st2.attempt == 3
        assert st2.status == "waiting_permission"
        assert len(st2.history) == 1
        assert st2.history[0]["content"] == "hello"
        assert st2.steps[0]["tool"] == "list_files"

    def test_from_dict_ignores_unknown_fields(self):
        d = _sample_state().to_dict()
        d["future_field"] = "should_be_ignored"
        d["another"] = 123
        st = AgentState.from_dict(d)
        assert st.task_id == "task01"
        assert not hasattr(st, "future_field")

    def test_from_dict_missing_optional_ok(self):
        minimal = {
            "task_id": "t",
            "agent_name": "HEILO_CODE",
            "task": "x",
            "phase": "PLAN",
            "attempt": 0,
            "status": "running",
        }
        st = AgentState.from_dict(minimal)
        assert st.history == []
        assert st.steps == []


class TestAgentStateStore:
    """Persistence layer."""

    def test_save_and_load(self, store):
        st = _sample_state(task_id="abc123")
        st.set_history([Message(role="user", content="go")])
        path = store.save(st)
        assert path.exists()
        assert path.name == "abc123.json"

        loaded = store.load("abc123")
        assert loaded is not None
        assert loaded.task_id == "abc123"
        assert loaded.history[0]["content"] == "go"
        assert loaded.phase == "INSPECT"

    def test_load_missing_returns_none(self, store):
        assert store.load("does_not_exist") is None

    def test_load_corrupt_json_returns_none(self, store):
        bad = store.dir / "bad.json"
        bad.write_text("{not-json", encoding="utf-8")
        assert store.load("bad") is None

    def test_delete(self, store):
        st = _sample_state(task_id="delme")
        store.save(st)
        assert store.load("delme") is not None
        store.delete("delme")
        assert store.load("delme") is None

    def test_delete_missing_is_safe(self, store):
        store.delete("never_saved")  # must not raise

    def test_save_overwrites_same_task_id(self, store):
        st = _sample_state(task_id="same", phase="UNDERSTAND", attempt=0)
        store.save(st)
        st.phase = "APPLY"
        st.attempt = 2
        store.save(st)
        loaded = store.load("same")
        assert loaded.phase == "APPLY"
        assert loaded.attempt == 2

    def test_list_waiting_filters(self, store):
        store.save(_sample_state(task_id="r1", status="running"))
        store.save(_sample_state(task_id="w1", status="waiting_permission", phase="APPLY"))
        store.save(_sample_state(task_id="s1", status="success"))
        store.save(_sample_state(task_id="w2", status="waiting_permission", phase="TEST"))

        waiting = store.list_waiting()
        ids = {w.task_id for w in waiting}
        assert ids == {"w1", "w2"}
        assert all(w.status == "waiting_permission" for w in waiting)

    def test_list_waiting_empty(self, store):
        assert store.list_waiting() == []

    def test_list_waiting_skips_corrupt(self, store):
        store.save(_sample_state(task_id="w1", status="waiting_permission"))
        (store.dir / "corrupt.json").write_text("{{{", encoding="utf-8")
        waiting = store.list_waiting()
        assert len(waiting) == 1
        assert waiting[0].task_id == "w1"

    def test_save_updates_updated_at(self, store):
        st = _sample_state(task_id="ts")
        old = st.updated_at
        store.save(st)
        assert st.updated_at >= old

    def test_persisted_json_structure(self, store):
        st = _sample_state(task_id="struct")
        st.pending_permission = {"level": "WRITE", "target": "main.py"}
        store.save(st)
        raw = json.loads((store.dir / "struct.json").read_text(encoding="utf-8"))
        assert raw["task_id"] == "struct"
        assert raw["pending_permission"]["target"] == "main.py"
        assert "history" in raw
        assert "steps" in raw


class TestResumeScenarios:
    """Scenarios that mirror CodeAgent pause/resume."""

    def test_pause_on_permission_then_resume_payload(self, store):
        # Simulate pause
        st = _sample_state(
            task_id="perm1",
            status="waiting_permission",
            phase="APPLY",
            attempt=1,
        )
        st.set_history([
            Message(role="system", content="CODE"),
            Message(role="user", content="fix discount"),
            Message(role="assistant", content="TOOL: edit_file ..."),
        ])
        st.steps = [
            {"tool": "list_files", "success": True},
            {"tool": "read_file", "success": True},
        ]
        st.pending_permission = {
            "level": "WRITE",
            "action": "edit_file",
            "target": "main.py",
            "reason": "modify code",
        }
        store.save(st)

        # Simulate load on resume
        loaded = store.load("perm1")
        assert loaded is not None
        assert loaded.status == "waiting_permission"
        assert loaded.phase == "APPLY"
        assert loaded.attempt == 1
        assert len(loaded.to_messages()) == 3
        assert loaded.pending_permission["action"] == "edit_file"

        # After approve, agent would clear pending and set running
        loaded.status = "running"
        loaded.pending_permission = None
        store.save(loaded)
        again = store.load("perm1")
        assert again.status == "running"
        assert again.pending_permission is None
        # history preserved for continuity
        assert len(again.history) == 3

    def test_cleanup_after_success(self, store):
        st = _sample_state(task_id="done1", status="running")
        store.save(st)
        # finish
        st.status = "success"
        store.delete(st.task_id)
        assert store.load("done1") is None

    def test_orphan_running_detectable(self, store):
        """Orphan = running on disk, no active process — GC candidate."""
        store.save(_sample_state(task_id="orphan", status="running", phase="ANALYZE"))
        all_files = list(store.dir.glob("*.json"))
        assert len(all_files) == 1
        loaded = store.load("orphan")
        assert loaded.status == "running"
        # list_waiting should NOT include orphans in running state
        assert store.list_waiting() == []
