"""HEILO Dia a Dia: respostas exatas e sem falsos positivos em conversa normal."""
from datetime import datetime

import pytest

from heilo.tools.diaadia import DiaADia

AGORA = datetime(2026, 9, 27, 13, 5)


@pytest.fixture
def d(tmp_path):
    return DiaADia(tmp_path, relogio=lambda: AGORA)


@pytest.mark.parametrize("q,esperado", [
    ("que horas são?", "13:05"),
    ("que dia é hoje?", "domingo, 27 de setembro de 2026"),
    ("quantos dias faltam pro natal?", "89 dias"),
    ("25/12/2026 cai em que dia da semana?", "sexta-feira"),
    ("quanto é 17% de 2.380?", "404,6"),
    ("quanto é 12 x 7?", "84"),
    ("quanto é 1.250,50 mais 300", "1.550,5"),
    ("raiz de 81", "9"),
    ("10/0", "dividir por zero"),
    ("quantos mm tem 2 polegadas", "50,8 mm"),
    ("30 graus celsius em fahrenheit", "86 °F"),
    ("2 horas em minutos", "120 minutos"),
])
def test_respostas_exatas(d, q, esperado):
    r = d.responder(q)
    assert r is not None and esperado in r["content"], (q, r)


@pytest.mark.parametrize("q", ["oi, tudo bem?", "quem é você?", "quanto é bom estudar?", "você lembra de mim?",
                               "nota 10 pra você", "o que é uma função em python?", "me fala sobre o brasil",
                               "/status", "estou com fome"])
def test_conversa_normal_vai_para_o_modelo(d, q):
    assert d.responder(q) is None


def test_lembrete_notas_listas(d):
    assert "15:00" in d.responder("me lembra de ligar pro fornecedor às 15h")["content"]
    assert "ligar pro fornecedor" in d.responder("meus lembretes")["content"]
    assert "Me diz quando" in d.responder("me lembra de pagar a conta")["content"]
    d.relogio = lambda: datetime(2026, 9, 27, 15, 1)
    assert [l["texto"] for l in d.vencidos()] == ["ligar pro fornecedor"]
    assert d.vencidos() == []                                   # avisa uma vez só
    d.responder("anota: peça 4520 precisa de revisão")
    assert "4520" in d.responder("o que eu anotei sobre a peça 4520?")["content"]
    d.responder("coloca leite, pão e ovos na lista de compras")
    d.responder("tira pão da lista de compras")
    lista = d.responder("mostra a lista de compras")["content"]
    assert "leite" in lista and "ovos" in lista and "pão" not in lista


def test_orchestrator_usa_ferramenta(tmp_path, monkeypatch):
    from heilo.core.orchestrator import Orchestrator
    o = Orchestrator()
    o.diaadia = DiaADia(tmp_path, relogio=lambda: AGORA)
    r = o.chat("quanto é 17% de 2.380?")
    assert r["source"] == "ferramenta:calculadora" and "404,6" in r["content"]
