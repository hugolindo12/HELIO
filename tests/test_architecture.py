"""
Testes da arquitetura oficial da HEILO.

Core + Seed + Teacher [opcional] + Memory + Knowledge + Training.
Numeração = lista de testes pedida na especificação.
"""
import socket
import subprocess
import sys

import pytest

from heilo.models.base import TEACHER_REQUIRED_MSG
from heilo.models.seed import gpt
from heilo.tests.helpers import (
    FakeAdapter, escrever_aprovados, jsonl, plataforma, sha, treinar_seed_minusculo,
)

precisa_torch = pytest.mark.skipif(not gpt.TORCH_OK, reason="PyTorch não instalado")


@pytest.fixture
def sem_teacher(monkeypatch):
    """Simula a REMOÇÃO do Teacher: pacote e bibliotecas do modelo externo ausentes."""
    for mod in ("heilo.models.teacher", "heilo.models.teacher.adapter", "transformers", "peft"):
        monkeypatch.setitem(sys.modules, mod, None)


@pytest.fixture
def sem_internet(monkeypatch):
    def bloqueado(*a, **k):
        raise OSError("sem internet (teste)")
    monkeypatch.setattr(socket.socket, "connect", bloqueado)
    monkeypatch.setattr(socket, "create_connection", bloqueado)


# 1 + 2 ----------------------------------------------------------------------
def test_01_02_inicia_heilo_e_carrega_core(tmp_path):
    orch, models, memory, knowledge, training = plataforma(tmp_path)
    st = orch.get_status()
    assert "models" in st and "memory" in st
    assert orch.models is models and orch.training is training


def test_core_nao_importa_modelo_externo():
    """O Core não pode conter dependência direta do modelo externo."""
    from pathlib import Path
    core = Path(__file__).resolve().parents[1] / "core"
    for arq in core.glob("*.py"):
        texto = arq.read_text(encoding="utf-8").lower()
        for proibido in ("qwen", "import transformers", "from transformers", "import peft",
                         "from peft", "models.teacher"):
            assert proibido not in texto, f"{arq.name} menciona '{proibido}'"


# 3 --------------------------------------------------------------------------
@precisa_torch
def test_03_carrega_seed(tmp_path):
    _, models, *_rest, training = plataforma(tmp_path, com_orquestrador=False)
    assert not models.is_available("seed")          # ainda sem pesos
    treinar_seed_minusculo(training)
    models.reload()
    assert models.is_available("seed")
    assert models.get("seed").card().provider == "heilo"


# 4 --------------------------------------------------------------------------
def test_04_carrega_teacher_com_metadados_de_licenca():
    from heilo.models.teacher.adapter import TeacherAdapter
    card = TeacherAdapter().card().to_dict()
    assert card["name"] == "HEILO Teacher" and card["role"] == "teacher"
    assert card["provider"] == "external"
    assert "Qwen2.5-0.5B" in card["base_model"]           # transparência: modelo real
    assert card["license"] == "Apache-2.0"
    assert card["purpose"] == "training_teacher" and card["status"] == "temporary"


# 5 --------------------------------------------------------------------------
def test_05_teacher_desativado(tmp_path):
    t = FakeAdapter("teacher", "sou o teacher")
    orch, models, *_ = plataforma(tmp_path, teacher_enabled=False,
                                  adapters={"teacher": t, "seed": FakeAdapter("seed", "sou o seed")})
    assert not models.teacher_available()
    assert "desativado" in models.status()["teacher"]["reason"]
    assert TEACHER_REQUIRED_MSG in models.set_mode("teacher")
    assert models.compare([{"role": "user", "content": "oi"}])["teacher"] == TEACHER_REQUIRED_MSG
    assert orch.chat("oi")["content"] == "sou o seed"
    assert t.chamadas == []


# 6 --------------------------------------------------------------------------
def test_06_teacher_removido_core_continua(tmp_path, sem_teacher):
    orch, models, memory, *_ = plataforma(tmp_path, adapters={"seed": FakeAdapter("seed", "oi do seed")})
    assert not models.teacher_available()
    assert "não instalado" in models.status()["teacher"]["reason"]
    r = orch.chat("oi")
    assert r["content"] == "oi do seed"
    from heilo.core import commands
    assert TEACHER_REQUIRED_MSG in commands.handle(orch, "/gerar o que é python?")


# 7 --------------------------------------------------------------------------
@precisa_torch
def test_07_conversa_com_seed_real(tmp_path):
    orch, models, _m, _k, training = plataforma(tmp_path)
    treinar_seed_minusculo(training)
    models.set_mode("seed")
    r = orch.chat("oi")
    assert r["source"] in ("model_seed", "conversational_brain")
    assert r["content"]


# 8 --------------------------------------------------------------------------
def test_08_conversa_com_teacher_quando_escolhido(tmp_path):
    t = FakeAdapter("teacher", "resposta do teacher")
    orch, models, *_ = plataforma(tmp_path, adapters={"teacher": t,
                                                      "seed": FakeAdapter("seed", disponivel=False)})
    assert models.set_mode("teacher") == "Modo do cérebro: teacher."
    r = orch.chat("oi")
    assert r["content"] == "resposta do teacher" and r["model"] == "teacher"
    assert t.chamadas, "Teacher deveria ter sido chamado"


# 9 --------------------------------------------------------------------------
def test_09_auto_prefere_seed_e_nao_promove_teacher(tmp_path):
    seed, t = FakeAdapter("seed", "seed fala"), FakeAdapter("teacher", "teacher fala")
    orch, models, *_ = plataforma(tmp_path, adapters={"seed": seed, "teacher": t})
    assert orch.chat("oi")["content"] == "seed fala"
    seed.disponivel = False
    r = orch.chat("oi")                      # Seed fora e teacher_in_chat=false
    assert r["content"] != "teacher fala"    # → fallback, NÃO o Teacher
    assert t.chamadas == []
    models.teacher_in_chat = True            # só com permissão explícita
    assert orch.chat("oi")["content"] == "teacher fala"


# 10 -------------------------------------------------------------------------
def test_10_ensinar_registra_sem_alterar_pesos(tmp_path):
    pesos = tmp_path / "seed.pt"
    pesos.write_bytes(b"pesos-fixos")
    orch, _m, _mem, knowledge, training = plataforma(tmp_path, seed_weights=pesos)
    antes = sha(pesos)
    from heilo.core import commands
    out = commands.handle(orch, "/ensinar quem te criou? => Fui criada pelo Hugo.")
    assert "Nenhum peso foi alterado" in out
    assert sha(pesos) == antes
    assert jsonl(training.taught_file)[0]["origin"] == "taught"
    assert any(r["origin"] == "taught" for r in training.approved())
    assert (knowledge.root / "taught" / "ensinamentos.md").exists()
    # inválido → rejeitado, não aprovado
    assert "Não registrei" in commands.handle(orch, "/ensinar oi => ")


# 11 -------------------------------------------------------------------------
def test_11_aprender_prepara_dataset_sem_treinar_sem_teacher_sem_git(tmp_path, monkeypatch):
    t = FakeAdapter("teacher")
    pesos = tmp_path / "seed.pt"
    pesos.write_bytes(b"pesos-fixos")
    orch, _m, memory, _k, training = plataforma(
        tmp_path, adapters={"teacher": t, "seed": FakeAdapter("seed", "olá!")}, seed_weights=pesos)
    antes = sha(pesos)
    chamadas_git = []
    real_run = subprocess.run
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: chamadas_git.append(a) or real_run(*a, **k))
    orch.chat("oi")
    orch.chat("tudo bem?")
    r = orch.aprender()
    assert r["coletados_da_memoria"] == 2
    assert r["pendentes_revisao"] == 2                       # candidatos esperam revisão
    assert r["dataset"]["train_examples"] + r["dataset"]["val_examples"] == 0  # nada aprovado ainda
    assert sha(pesos) == antes
    assert t.chamadas == [] and chamadas_git == []


# 12 + 13 --------------------------------------------------------------------
def test_12_13_memoria_salva_e_recupera(tmp_path):
    orch, _m, memory, *_ = plataforma(tmp_path, adapters={"seed": FakeAdapter("seed", "claro!")})
    orch.chat("oi! hoje eu mexi com G-code")
    assert memory.stats()["trocas"] == 1
    achado = memory.search("g-code")
    assert achado and "G-code" in achado[0]["user"]
    assert memory.recent(1)[0]["modelo"] == "seed"


# 14 -------------------------------------------------------------------------
def test_14_knowledge_sem_retreinar(tmp_path):
    _o, _m, _mem, knowledge, _t = plataforma(tmp_path, com_orquestrador=False)
    knowledge.add_document("Fanuc G41", "G41 ativa a compensação de raio à esquerda.", "cnc")
    hits = knowledge.search("compensação de raio G41")
    assert any("G41" in h["text"] for h in hits)


# 15 -------------------------------------------------------------------------
def test_15_dataset_so_com_aprovados_e_teacher_nao_entra_sem_revisao(tmp_path):
    t = FakeAdapter("teacher", "Python é uma linguagem de programação.")
    _o, _m, _mem, _k, training = plataforma(tmp_path, adapters={"teacher": t},
                                            com_orquestrador=False)
    escrever_aprovados(training, [("oi", "Oi! Tudo bem?")])
    training.generate_with_teacher(["o que é python?"])
    m1 = training.build_dataset()
    assert m1["teacher_examples"] == 0                        # não verificado → fora
    pend = training.pending()
    assert len(pend) == 1 and pend[0]["origin"] == "teacher"
    training.review(pend[0]["id"], approve=True)
    m2 = training.build_dataset()
    total = m2["train_examples"] + m2["val_examples"]
    assert total == 2 and m2["version"] == 2
    met = training.metrics()["teacher"]
    assert met["gerados"] == 1 and met["aprovados"] == 1 and met["pendentes"] == 0


def test_validacao_rejeita_saida_ruim_do_teacher(tmp_path):
    t = FakeAdapter("teacher", "<|im_start|>assistant lixo")
    _o, _m, _mem, _k, training = plataforma(tmp_path, adapters={"teacher": t},
                                            com_orquestrador=False)
    training.generate_with_teacher(["pergunta"])
    r = training.validate_pending()
    assert r["rejeitados_auto"] == 1 and training.pending() == []
    assert "artefato" in training.rejected()[0]["reasons"][0]


# 16 -------------------------------------------------------------------------
@precisa_torch
def test_16_treino_registra_run_com_origem_e_validacao(tmp_path):
    _o, models, _mem, _k, training = plataforma(tmp_path, com_orquestrador=False)
    run = treinar_seed_minusculo(training)
    runs = jsonl(training.runs_file)
    assert len(runs) == 1 and runs[0]["dataset_version"] == 1
    assert runs[0]["por_origem"] == {"curated": run["exemplos_treino"]}
    assert runs[0]["teacher_examples_used"] == 0
    assert runs[0]["perda_treino"] is not None
    # continuar treino soma passos
    run2 = training.train_seed(passos=2, lote=2, log=lambda *_: None, salvar_cada=0)
    assert run2["passos_totais"] == run["passos_totais"] + 2


# 17 + 19 --------------------------------------------------------------------
def test_17_19_funciona_offline(tmp_path, sem_internet):
    orch, *_rest, training = plataforma(tmp_path, adapters={"seed": FakeAdapter("seed", "offline ok")})
    assert orch.chat("oi")["content"] == "offline ok"
    assert orch.ensinar("oi", "E aí!")["ok"]
    assert orch.aprender()["aprovados"] == 1


# 18 -------------------------------------------------------------------------
def test_18_sem_qwen(tmp_path, sem_teacher, sem_internet):
    orch, models, *_ = plataforma(tmp_path, adapters={"seed": FakeAdapter("seed", "sem qwen ok")})
    assert orch.chat("oi")["content"] == "sem qwen ok"
    assert models.compare([{"role": "user", "content": "oi"}])["teacher"] == TEACHER_REQUIRED_MSG


# 20 -------------------------------------------------------------------------
def test_20_sem_github(tmp_path, monkeypatch):
    real_run = subprocess.run

    def sem_git(args, *a, **k):
        if args and args[0] == "git":
            raise FileNotFoundError("git não instalado (teste)")
        return real_run(args, *a, **k)
    orch, *_ = plataforma(tmp_path, adapters={"seed": FakeAdapter("seed", "ok")})
    monkeypatch.setattr(subprocess, "run", sem_git)
    assert orch.chat("oi")["content"] == "ok"
    assert orch.aprender()["dataset"]["version"] == 1
    r = orch.knowledge_git.sync(push=True)                   # publicar falha com elegância
    assert r["commit"]["ok"] is False


# 21 -------------------------------------------------------------------------
def test_21_modelo_inexistente(tmp_path):
    orch, models, *_ = plataforma(tmp_path, seed_weights=tmp_path / "nao_existe.pt")
    ok, motivo = models.get("seed").availability()
    assert not ok and ("não encontrados" in motivo or "PyTorch" in motivo)
    r = orch.chat("oi")
    assert r["content"] and r["source"] == "conversational_brain"   # fallback, sem quebrar


# 22 -------------------------------------------------------------------------
def test_22_configuracao_invalida(monkeypatch, tmp_path):
    from heilo.config import env_bool, normalize_brain_mode
    assert normalize_brain_mode("xyz") == "auto"
    assert normalize_brain_mode("mini") == "seed" and normalize_brain_mode("lora") == "teacher"
    monkeypatch.setenv("HEILO_TEACHER_ENABLED", "talvez")
    assert env_bool("HEILO_TEACHER_ENABLED", False) is False
    _o, models, *_ = plataforma(tmp_path, modo="qualquer-coisa", com_orquestrador=False)
    assert models.mode == "auto"
    assert "Modo do cérebro: seed" in models.set_mode("mini")
