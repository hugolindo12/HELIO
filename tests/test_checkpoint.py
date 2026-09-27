"""Unit tests for heilo.security.checkpoint"""
import pytest
from pathlib import Path
from heilo.security.checkpoint import CheckpointManager, Checkpoint, FileSnapshot


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "main.py").write_text("ORIGINAL\n", encoding="utf-8")
    return root


@pytest.fixture
def cp(project):
    return CheckpointManager(project)


class TestCheckpointManager:
    def test_start_and_snapshot(self, cp, project):
        ck = cp.start("task1")
        assert ck.task_id == "task1"
        cp.snapshot_file("main.py", "ORIGINAL\n", existed=True)
        assert len(cp._active.files) == 1
        assert cp._active.files[0].path == "main.py"

    def test_duplicate_snapshot_ignored(self, cp):
        cp.start("t")
        cp.snapshot_file("main.py", "A", existed=True)
        cp.snapshot_file("main.py", "B", existed=True)
        assert len(cp._active.files) == 1
        assert cp._active.files[0].content == "A"

    def test_commit_persists(self, cp, project):
        cp.start("task-commit")
        cp.snapshot_file("main.py", "ORIGINAL\n", existed=True)
        path = cp.commit()
        assert path is not None
        assert path.exists()
        assert cp._active is None
        listed = cp.list_checkpoints()
        assert any(c["task_id"] == "task-commit" for c in listed)

    def test_commit_empty_returns_none(self, cp):
        cp.start("empty")
        assert cp.commit() is None

    def test_rollback_restores_content(self, cp, project):
        main = project / "main.py"
        cp.start("rb")
        cp.snapshot_file("main.py", "ORIGINAL\n", existed=True)
        main.write_text("BROKEN\n", encoding="utf-8")
        restored = cp.rollback()
        assert any("restored:main.py" in r for r in restored)
        assert main.read_text(encoding="utf-8") == "ORIGINAL\n"

    def test_rollback_deletes_new_file(self, cp, project):
        cp.start("del")
        cp.snapshot_file("created.py", "", existed=False)
        new = project / "created.py"
        new.write_text("new file\n", encoding="utf-8")
        restored = cp.rollback()
        assert any("deleted:created.py" in r for r in restored)
        assert not new.exists()

    def test_rollback_from_disk(self, cp, project):
        cp.start("disk")
        cp.snapshot_file("main.py", "ORIGINAL\n", existed=True)
        cp.commit()
        (project / "main.py").write_text("CHANGED\n", encoding="utf-8")
        # no active checkpoint — load last from disk
        restored = cp.rollback()
        assert (project / "main.py").read_text(encoding="utf-8") == "ORIGINAL\n"

    def test_checkpoint_roundtrip_dict(self):
        ck = Checkpoint(
            checkpoint_id="id1",
            task_id="t1",
            created_at="2026-01-01T00:00:00",
            files=[FileSnapshot(path="a.py", content="x", existed=True)],
        )
        d = ck.to_dict()
        ck2 = Checkpoint.from_dict(d)
        assert ck2.checkpoint_id == "id1"
        assert ck2.files[0].content == "x"

    def test_list_checkpoints_limit(self, cp):
        for i in range(3):
            cp.start(f"t{i}")
            cp.snapshot_file("main.py", f"v{i}", existed=True)
            cp.commit()
        listed = cp.list_checkpoints(limit=2)
        assert len(listed) <= 2
