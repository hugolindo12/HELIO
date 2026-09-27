"""
HEILO API Tests
Validates FastAPI endpoints including status, chat, SSE stream, and checkpoints.
"""
import pytest
from fastapi.testclient import TestClient
from heilo.ui.api import app, orchestrator


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_api_status(client):
    res = client.get("/api/status")
    assert res.status_code == 200
    data = res.json()
    assert "workspace" in data
    assert "model" in data
    assert "agents" in data
    assert "HEILO_CODE" in data["agents"]
    assert "HEILO_RESEARCH" in data["agents"]
    assert "HEILO_TEST" in data["agents"]


def test_api_tools(client):
    res = client.get("/api/tools")
    assert res.status_code == 200
    data = res.json()
    assert "tools" in data
    assert len(data["tools"]) >= 10


def test_api_mcp_servers(client):
    res = client.get("/api/mcp")
    assert res.status_code == 200
    data = res.json()
    assert "servers" in data
    names = [s.get("name") for s in data["servers"]]
    assert "system" in names


def test_api_checkpoints_and_rollback(client):
    # List checkpoints
    res = client.get("/api/checkpoints")
    assert res.status_code == 200
    data = res.json()
    assert "checkpoints" in data
    assert isinstance(data["checkpoints"], list)

    # Rollback
    res = client.post("/api/checkpoints/rollback")
    assert res.status_code == 200
    data = res.json()
    assert "restored" in data
    assert "count" in data


def test_api_chat(client):
    res = client.post("/api/chat", json={"message": "Olá HEILO"})
    assert res.status_code == 200
    data = res.json()
    assert "content" in data or "type" in data


def test_api_chat_stream(client):
    res = client.post("/api/chat/stream", json={"message": "Status do sistema"})
    assert res.status_code == 200
    assert "text/event-stream" in res.headers.get("content-type", "")
    content = res.text
    assert "event: start" in content
    assert "event: intent" in content
    assert "event: result" in content
    assert "event: done" in content
