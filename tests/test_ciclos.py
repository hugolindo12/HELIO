"""Ciclos de treino do HEILO Seed: validação, avaliação e promoção de versões."""
import json
import re
import shutil

import pytest

from heilo.models.seed import gpt
from heilo.tests.helpers import FakeAdapter, escrever_aprovados, plataforma
from heilo.training.eval_set import load_eval
from heilo.training.evaluate import acertou, avaliar
from heilo.training.qualidade import avaliar_exemplo, contaminado

EVAL_QS = [i["pergunta"] for i in load_eval()]


def test_eval_set_sem_sobreposicao_com_treino_aprovado():
    from pathlib import Path
    from heilo.training.records import read_jsonl
    aprovados = Path(__file__).resolve().parents[1] / "data" / "approved"
    perguntas = [r["messages"][0]["content"] for f in aprovados.glob("*.jsonl")
                 for r in read_jsonl(f)]
    iguais = [q for q in perguntas if q.strip().lower() in {e.lower() for e in EVAL_QS}]
    assert iguais == []


def test_criterios_de_acerto():
    assert acertou("7 vezes 9 é 63.", {"any": [["63"]]})
    assert not acertou("7 vezes 9 é 64.", {"any": [["63"]]})
    assert acertou("Sim.", {"regex": r"^\s*sim[.!]?\s*$"})
    assert not acertou("Sim, claro que sou", {"regex": r"^\s*sim[.!]?\s*$"})
    assert not acertou("G02 é anti-horário", {"any": [["horario"]], "none": ["anti-horario"]})


def test_avaliar_com_modelo_falso():
    itens = load_eval()[:5]
    r = avaliar(itens, lambda q, t, k: "Oi! Tudo tranquilo por aqui.", amostras=2)
    assert r["itens"] == 5 and 0 <= r["acerto"] <= 1 and r["perplexidade"] is None


def test_qualidade_filtra_saidas_ruins_do_teacher():
    ok = avaliar_exemplo("o que é g01?", "G01 é interpolação linear.", "cnc", EVAL_QS, set(), True)
    assert ok["status"] == "aprovado"
    errado = avaliar_exemplo("o que é g02?", "G02 é anti-horário.", "cnc", EVAL_QS, set(), True)
    assert errado["status"] == "rejeitado"
    conta = avaliar_exemplo("quanto é 3 vezes 4?", "Dá 13.", "matematica", EVAL_QS, set(), False)
    assert conta["status"] == "rejeitado"
    outro = avaliar_exemplo("quem é você?", "Sou o Qwen, um modelo de linguagem.", "conversa",
                            EVAL_QS, set(), False)
    assert outro["status"] == "rejeitado"
    chines = avaliar_exemplo("oi", "你好", "conversa", EVAL_QS, set(), False)
    assert chines["status"] == "rejeitado"
    nao_verificavel = avaliar_exemplo("o que é o blender?", "É um programa 3D.", "ferramentas",
                                      EVAL_QS, set(), True)
    assert nao_verificavel["status"] == "revisao"      # factual sem verificação → humano


def test_contaminacao_com_a_avaliacao():
    assert contaminado("7x9", EVAL_QS)
    assert contaminado("quanto é 9 vezes 7?", EVAL_QS)      # mesma conta, outra forma
    assert contaminado("Como você se chama?", EVAL_QS)
    assert not contaminado("quanto é 11 vezes 13?", EVAL_QS)


class _Professor(FakeAdapter):
    def generate(self, messages, system=None, **k):
        q = messages[-1]["content"]
        if "Escreva 8 perguntas" in q:
            return "- o que é uma string?\n- como leio um csv?"
        if "g01" in q.lower():
            return "G01 é interpolação linear."
        nums = re.findall(r"\d+", q)
        if "vezes" in q and len(nums) >= 2:
            return f"{nums[0]} vezes {nums[1]} dá {int(nums[0]) * int(nums[1])}."
        return "Claro, posso ajudar!"


@pytest.mark.skipif(not gpt.TORCH_OK, reason="PyTorch não instalado")
def test_ciclo_completo_cria_versao_e_so_promove_com_ganho(tmp_path):
    from heilo.training.ciclos import CiclosSeed
    prof = _Professor("teacher")
    _o, models, _m, _k, pipe = plataforma(tmp_path, adapters={"teacher": prof},
                                          com_orquestrador=False)
    escrever_aprovados(pipe, [(f"pergunta {i}", f"resposta {i}") for i in range(12)])
    pesos = tmp_path / "seed.pt"
    pipe.build_dataset()
    pipe.train_seed(passos=3, lote=2, log=lambda *_: None, salvar_cada=0, pesos=pesos,
                    config=gpt.GPTConfig(block_size=64, n_layer=1, n_head=2, n_embd=32))
    c = CiclosSeed(pipe, models=models, data_dir=pipe.root, versions_dir=tmp_path / "v",
                   pesos_ativos=pesos, log=lambda *_: None)
    c.itens_eval = c.itens_eval[:6]
    antes = pesos.read_bytes()
    rel = c.executar(max_ciclos=1, passos=3, base_n=4, lote=2, salvar_cada=0)
    reg = c.registro()
    assert [v["versao"] for v in reg["versoes"]] == ["v0.1", "v0.2"]
    v2 = reg["versoes"][1]
    assert v2["status"] in ("ativa", "rejeitada") and "comparacao" in v2
    if v2["status"] == "rejeitada":
        assert pesos.read_bytes() == antes           # versão anterior preservada
    d = rel["ciclos"][0]["dados"]
    assert d["por_origem"]["teacher"]["aprovados"] >= 1      # G01 e contas verificadas
    # nenhuma pergunta da avaliação entrou nos aprovados
    aprovados = {r["messages"][0]["content"].lower() for r in pipe.approved()}
    assert not aprovados & {q.lower() for q in EVAL_QS}
    uso = [json.loads(l) for l in (pipe.root / "training" / "uso_dados.jsonl").read_text().splitlines()]
    assert uso[0]["versao_seed"] == "v0.2" and uso[0]["ids"]


@pytest.mark.skipif(not gpt.TORCH_OK, reason="PyTorch não instalado")
def test_ciclo_sem_professor_funciona(tmp_path):
    from heilo.training.ciclos import CiclosSeed
    _o, models, _m, _k, pipe = plataforma(tmp_path, teacher_enabled=False, com_orquestrador=False)
    escrever_aprovados(pipe, [(f"pergunta {i}", f"resposta {i}") for i in range(12)])
    pesos = tmp_path / "seed.pt"
    pipe.build_dataset()
    pipe.train_seed(passos=2, lote=2, log=lambda *_: None, salvar_cada=0, pesos=pesos,
                    config=gpt.GPTConfig(block_size=64, n_layer=1, n_head=2, n_embd=32))
    c = CiclosSeed(pipe, models=models, data_dir=pipe.root, versions_dir=tmp_path / "v",
                   pesos_ativos=pesos, log=lambda *_: None)
    c.itens_eval = c.itens_eval[:4]
    rel = c.executar(max_ciclos=1, passos=2, base_n=3, lote=2, salvar_cada=0)
    assert rel["ciclos"][0]["dados"]["teacher_chamadas"] == 0


def test_quarentena_tira_do_treino_ate_revisao(tmp_path):
    _o, _m, _mem, _k, pipe = plataforma(tmp_path, com_orquestrador=False)
    escrever_aprovados(pipe, [("o que é x?", "x é uma coisa."), ("oi", "Oi!")], origem="teacher_qwen")
    ids = [r["id"] for r in pipe.approved()]
    pipe.quarantine([ids[0]], "aprovado sem verificação")
    assert [r["id"] for r in pipe.approved()] == [ids[1]]
    assert [r["id"] for r in pipe.pending()] == [ids[0]]
    assert pipe.build_dataset()["teacher_examples"] + pipe.latest_dataset()["val_examples"] <= 1
    assert pipe.review(ids[0], True)["ok"]                 # humano aprova → volta
    assert {r["id"] for r in pipe.approved()} == set(ids)
    assert pipe.pending() == []
