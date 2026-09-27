"""Interface de conversa: rotas novas respondem e não disparam agentes."""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture(scope="module")
def cliente():
    from heilo.ui.api import app
    return TestClient(app)


def test_pagina_e_info(cliente):
    assert "HEILO" in cliente.get("/").text
    d = cliente.get("/api/heilo").json()
    assert "modo" in d and "versoes" in d


def test_comando_na_conversa(cliente):
    r = cliente.post("/api/conversa", json={"message": "/cerebro", "history": []}).json()
    assert r["type"] == "comando" and "Modo" in r["content"]


def test_conversa_nao_chama_agente(cliente):
    r = cliente.post("/api/conversa", json={"message": "crie um arquivo main.py com um bug corrigido",
                                            "history": [{"role": "user", "content": "oi"},
                                                        {"role": "assistant", "content": "Oi!"}]}).json()
    assert r.get("type") != "permission_request"
    assert "autorização" not in r.get("content", "")


def test_ensinar_valida(cliente):
    assert cliente.post("/api/ensinar", json={"pergunta": " ", "resposta": "x"}).status_code == 400
