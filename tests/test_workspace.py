"""Unit tests for heilo.security.workspace"""
import pytest
from pathlib import Path
from heilo.security.workspace import WorkspaceManager, WorkspaceError


@pytest.fixture
def ws(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "src").mkdir()
    (root / "src" / "app.py").write_text("print(1)\n", encoding="utf-8")
    (root / "readme.md").write_text("# hi\n", encoding="utf-8")
    return WorkspaceManager(root)


class TestWorkspaceManager:
    def test_resolve_relative(self, ws):
        p = ws.resolve("src/app.py")
        assert p.exists()
        assert p.name == "app.py"

    def test_resolve_must_exist(self, ws):
        with pytest.raises(WorkspaceError, match="does not exist"):
            ws.resolve("missing.py", must_exist=True)

    def test_path_traversal_blocked(self, ws):
        with pytest.raises(WorkspaceError):
            ws.resolve("../../../etc/passwd")

    def test_absolute_outside_blocked(self, ws):
        # /etc is blocked by policy pattern; also absolute outside is rejected
        with pytest.raises(WorkspaceError):
            ws.resolve("/etc/passwd")
        with pytest.raises(WorkspaceError):
            ws.resolve("/tmp/definitely_outside_heilo_ws_xyz")

    def test_blocked_patterns(self, ws):
        with pytest.raises(WorkspaceError, match="blocked"):
            ws.resolve("foo/System32/bar")

    def test_list_files(self, ws):
        files = ws.list_files()
        assert "src/app.py" in files or any(f.endswith("app.py") for f in files)
        assert any("readme" in f for f in files)

    def test_list_files_pattern(self, ws):
        py = ws.list_files(pattern="*.py")
        assert all(f.endswith(".py") for f in py)

    def test_read_write_text(self, ws):
        ws.write_text("src/new.py", "x = 1\n")
        assert ws.read_text("src/new.py") == "x = 1\n"
        assert ws.exists("src/new.py")

    def test_is_inside(self, ws):
        inside = ws.resolve("readme.md")
        assert ws.is_inside(inside) is True
        assert ws.is_inside(Path("/tmp")) is False

    def test_set_workspace(self, tmp_path):
        a = tmp_path / "a"
        b = tmp_path / "b"
        a.mkdir()
        mgr = WorkspaceManager(a)
        mgr.set_workspace(b)
        assert mgr.root == b.resolve()
        assert b.exists()

    def test_get_info(self, ws):
        info = ws.get_info()
        assert info["exists"] is True
        assert info["file_count"] >= 2
        assert "root" in info

    def test_edit_file_crlf_normalization(self, ws):
        from heilo.tools.filesystem import EditFileTool
        p = ws.resolve("crlf_sample.py")
        p.write_bytes(b"line1\r\nline2\r\nline3\r\n")
        
        tool = EditFileTool(ws)
        # Search using LF, matching CRLF content
        res = tool.execute("crlf_sample.py", old_string="line1\nline2", new_string="line1_fixed\nline2_fixed")
        assert res.success is True
        content = p.read_bytes()
        assert b"line1_fixed\r\nline2_fixed" in content


def test_edit_file_preserves_lf(tmp_path):
    from heilo.security.workspace import WorkspaceManager
    from heilo.tools.filesystem import EditFileTool
    ws = WorkspaceManager(tmp_path)
    p = ws.resolve("lf_sample.py")
    p.write_bytes(b"a\nb\n")
    assert EditFileTool(ws).execute("lf_sample.py", old_string="a", new_string="x").success
    assert p.read_bytes() == b"x\nb\n"
