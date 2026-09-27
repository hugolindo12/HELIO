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


def test_versao_inexistente(cliente):
    r = cliente.post("/api/conversa", json={"message": "oi", "history": [], "versao": "v9.9"}).json()
    assert r["type"] == "erro" and "não encontrada" in r["content"]


def test_versao_sem_caminho_malicioso(cliente):
    r = cliente.post("/api/conversa", json={"message": "oi", "history": [], "versao": "../../etc"}).json()
    assert r["type"] == "erro"


def test_escritos_viram_pendentes_e_so_aprovam_na_revisao(tmp_path):
    from heilo.training.pipeline import TrainingPipeline
    from heilo.training.records import make_example, append_jsonl
    tp = TrainingPipeline(data_dir=tmp_path)
    ex = make_example([{"role": "user", "content": "qual é a capital do japão?"},
                       {"role": "assistant", "content": "A capital do Japão é Tóquio."}],
                      origin="escrito_claude", meta={"status": "nao_verificado"})
    append_jsonl(tp.escritos_dir / "lote_teste.jsonl", [ex])
    assert [p["id"] for p in tp.pending()] == [ex["id"]]
    assert not tp.approved()
    assert tp.review(ex["id"], True)["ok"]
    assert [a["id"] for a in tp.approved()] == [ex["id"]] and not tp.pending()
