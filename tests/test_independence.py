"""HEILO Independence Test automatizado (ver docs/HEILO_INDEPENDENCE_TEST.md).

Copia a HEILO para uma pasta temporária e roda, sem internet e sem git:
A) Teacher habilitado  B) Teacher desligado  C) pasta do Teacher APAGADA.
As funções essenciais precisam funcionar nas três rodadas.
"""
import pytest

from heilo.independence import ESSENCIAIS, run_independence_test
from heilo.models.base import TEACHER_REQUIRED_MSG


def test_heilo_independence(tmp_path):
    rel = run_independence_test(tmp_path)
    for rodada in ("A_teacher_habilitado", "B_teacher_desligado", "C_teacher_removido"):
        r = rel[rodada]
        falhas = {k: r.get(k) for k in ESSENCIAIS if not r.get(k, {}).get("ok")}
        assert not falhas, f"{rodada}: {falhas}"
    for rodada in ("B_teacher_desligado", "C_teacher_removido"):
        assert TEACHER_REQUIRED_MSG in rel[rodada]["gerar_com_teacher"]["detalhe"]
        assert rel[rodada]["comparar"]["detalhe"] == TEACHER_REQUIRED_MSG
    assert "não instalado" in rel["C_teacher_removido"]["teacher"]["detalhe"]
