"""HEILO Aurora: corpus em partes, acumulação de gradiente e modelo grande em partes para o GitHub."""
import subprocess
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from heilo.models.seed import gpt  # noqa: E402
from heilo.models.seed.gpt import GPTConfig, carregar, dividir_em_partes, montar_partes, salvar, MiniGPT  # noqa: E402
from heilo.models.seed.tokenizer import BPETokenizer  # noqa: E402
from heilo.training import aurora, pretrain  # noqa: E402


def _tok():
    return BPETokenizer.treinar(iter(["O Brasil é um país da América do Sul. " * 40, "def f(x):\n    return x\n" * 40]),
                                vocab_size=300)


def test_pretreino_com_partes_e_acumulacao(tmp_path):
    tok = _tok()
    for k, txt in enumerate(["O Brasil é um país da América do Sul. " * 30, "def f(x):\n    return x\n" * 30], 1):
        pretrain.preparar_corpus(iter([txt] * 40), tok, tmp_path / "web" / f"parte_{k:02d}", max_tokens=10**9,
                                 log=lambda s: None)
    pastas = pretrain.pastas_corpus([tmp_path / "web"])
    assert len(pastas) == 2
    cfg = GPTConfig(block_size=32, n_layer=1, n_head=2, n_embd=32, dropout=0.0, arquitetura="moderna", n_kv_head=1)
    r = pretrain.pretreinar(cfg, tok, [tmp_path / "web"], tmp_path / "ck.pt", passos=6, lote=2, acumular=3,
                            avaliar_cada=3, salvar_cada=100, aquecimento=1, log=lambda s: None)
    assert r["passo"] == 6 and r["hist"]
    r2 = pretrain.pretreinar(cfg, tok, [tmp_path / "web"], tmp_path / "ck.pt", passos=8, lote=2, acumular=3,
                             avaliar_cada=100, salvar_cada=100, aquecimento=1, log=lambda s: None)
    assert r2["passo"] == 8                                     # retomou do checkpoint


def test_partes_do_modelo_ida_e_volta(tmp_path, monkeypatch):
    m = MiniGPT(GPTConfig(block_size=16, n_layer=1, n_head=2, n_embd=32, dropout=0.0))
    arq = salvar(m, {"versao": "x"}, tmp_path / "grande.pt")
    man = dividir_em_partes(arq, tamanho=20_000)
    assert len(man["partes"]) > 1
    original = arq.read_bytes()
    arq.unlink()
    assert gpt.tem_modelo(arq)
    m2, info = carregar(arq, dispositivo="cpu")                 # remonta sozinho
    assert info["versao"] == "x" and arq.read_bytes() == original
    (tmp_path / man["partes"][0]).write_bytes(b"lixo")
    arq.unlink(); Path(str(arq) + ".montado").unlink()
    with pytest.raises(ValueError):
        montar_partes(arq)


def test_preparar_para_git(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "models" / "seed" / "versions" / "aurora-1.0").mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    grande = repo / "models" / "seed" / "versions" / "aurora-1.0" / "heilo_seed.pt"
    pequeno = repo / "models" / "seed" / "weights" / "heilo_seed.pt"
    pequeno.parent.mkdir(parents=True)
    grande.write_bytes(b"a" * 50_000); pequeno.write_bytes(b"b" * 100)
    monkeypatch.setattr(aurora, "LIMITE_GITHUB", 10_000)
    monkeypatch.setattr(gpt, "TAMANHO_PARTE", 20_000)
    monkeypatch.setattr(aurora, "dividir_em_partes", lambda p: dividir_em_partes(p, tamanho=20_000))
    add = aurora.preparar_para_git(repo, log=lambda s: None)
    assert "models/seed/weights/heilo_seed.pt" in add
    assert "models/seed/versions/aurora-1.0/heilo_seed.pt.partes.json" in add
    assert sum(1 for a in add if ".parte0" in a) == 3
    assert "models/seed/versions/aurora-1.0/heilo_seed.pt" in (repo / ".gitignore").read_text()


def test_config_aurora_tem_cerca_de_150M():
    import copy
    c = copy.deepcopy(aurora.CONFIG_AURORA)
    c.vocab_size = 16000
    assert 140e6 < MiniGPT(c).n_params() < 160e6
    assert aurora.lote_para_gpu("Tesla T4", 15)["acumular"] * aurora.lote_para_gpu("Tesla T4", 15)["micro"] == 32


def test_regua_sonda_v2_sem_falsos_positivos():
    from heilo.training.eval_set import load_sonda
    from heilo.training.evaluate import acertou
    s = {it["pergunta"]: it["criterios"] for it in load_sonda()}
    assert not acertou("10 menos 12 é 12.", s["Qual é maior: 9 ou 12?"])
    assert acertou("12 é maior que 9.", s["Qual é maior: 9 ou 12?"])
    assert not acertou("Não tenho um modelo de frio.", s["Qual é o oposto de quente?"])
    assert acertou("O oposto de quente é frio.", s["Qual é o oposto de quente?"])
    assert acertou("3 mais 2 dá 5.", s["Tenho 3 maçãs e ganho mais 2. Com quantas fico?"])


def test_regua_palavra_inteira():
    from heilo.training.eval_set import load_sonda
    from heilo.training.evaluate import acertou
    c = {it["pergunta"]: it["criterios"] for it in load_sonda()}["Qual palavra-chave cria uma classe em Python?"]
    assert not acertou("Uma classe é um programa de computador.", c)
    assert acertou("Use a palavra-chave class.", c)


def test_agenda_por_tempo(tmp_path):
    tok = _tok()
    pretrain.preparar_corpus(iter(["O Brasil é um país da América do Sul. " * 30] * 40), tok, tmp_path / "c",
                             max_tokens=10**9, log=lambda s: None)
    cfg = GPTConfig(block_size=32, n_layer=1, n_head=2, n_embd=32, dropout=0.0, arquitetura="moderna", n_kv_head=1)
    r = pretrain.pretreinar(cfg, tok, tmp_path / "c", tmp_path / "ck.pt", passos=10**9, lote=2, aquecimento=2,
                            avaliar_cada=10**9, salvar_cada=10**9, minutos_alvo=0.03, log=lambda s: None)
    assert r["passo"] > 5                                  # parou pelo TEMPO, não pelos passos
    st = torch.load(tmp_path / "ck.pt", weights_only=False)
    assert st["minutos_agenda"] >= 0.03
    r2 = pretrain.pretreinar(cfg, tok, tmp_path / "c", tmp_path / "ck.pt", passos=10**9, lote=2, aquecimento=2,
                             avaliar_cada=10**9, salvar_cada=10**9, minutos_alvo=0.03, log=lambda s: None)
    assert r2["passo"] == r["passo"]                       # já estava completo: não treina de novo
