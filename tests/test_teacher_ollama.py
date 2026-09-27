"""Teacher via Ollama: fala com a API local (aqui simulada) e remove o <think>."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from heilo.models.teacher.ollama_adapter import OllamaTeacherAdapter


class _Falso(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj):
        b = json.dumps(obj).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        self._json({"models": [{"name": "qwen3-vl:8b"}]})

    def do_POST(self):
        corpo = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert corpo["stream"] is False and corpo["messages"][0]["role"] == "system"
        self._json({"message": {"content": "<think>pensando...</think>A capital de Portugal é Lisboa."}})


def test_ollama_teacher():
    srv = HTTPServer(("127.0.0.1", 0), _Falso)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    t = OllamaTeacherAdapter(url=f"http://127.0.0.1:{srv.server_port}")
    assert t.availability() == (True, "ok")
    assert t.generate([{"role": "user", "content": "capital de portugal?"}], system="seja breve") == \
        "A capital de Portugal é Lisboa."
    srv.shutdown()
    assert OllamaTeacherAdapter(url="http://127.0.0.1:9").availability()[0] is False
