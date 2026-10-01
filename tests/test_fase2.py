"""Fase 2: contas geradas certas, mistura em degraus, pesos no corpus e agenda WSD."""
import json
import re

import pytest


def _br(s):
    return float(s.replace(".", "").replace(",", "."))


def test_contas_geradas_estao_certas():
    import random
    from heilo.training import fase2
    rnd = random.Random(7)
    vistos = 0
    for _ in range(3000):
        t = fase2.gerar_conta(rnd)
        for a, op, b, c in re.findall(r"([\d.,]+) ([+−×]) ([\d.,]+) = (?:R\$ )?([\d.,]+)", t):
            a, b, c = _br(a), _br(b), _br(c.rstrip(".,"))
            esperado = {"+": a + b, "−": a - b, "×": a * b}[op]
            assert abs(esperado - c) < 1e-6, t
            vistos += 1
        m = re.search(r"Quanto é (\d+)% de ([\d.,]+)\? .* = ([\d.,]+)\.$", t)
        if m:
            assert abs(int(m.group(1)) * _br(m.group(2)) / 100 - _br(m.group(3))) < 0.01, t
        m = re.search(r"Resolva (\d+)x \+ (\d+) = (\d+)\..* x = (\d+)\.", t)
        if m:
            k, c, r, x = map(int, m.groups())
            assert k * x + c == r, t
    assert vistos > 1000


def test_numero_no_jeito_brasileiro():
    from heilo.training import fase2
    assert fase2._num(1234) == "1.234" and fase2._num(12.5) == "12,5" and fase2._num(1234.56) == "1.234,56"


def test_conversa_vira_texto():
    from heilo.training import fase2
    ex = [{"messages": [{"role": "user", "content": " oi "}, {"role": "assistant", "content": "Olá!"}]},
          {"messages": [{"role": "user", "content": "só pergunta"}]}]
    assert list(fase2.textos_conversa(ex)) == ["Pergunta: oi\nHEILO: Olá!"]


def test_codigo_so_python_de_licenca_permissiva():
    from heilo.training import fase2
    ok = {"language": "Python", "license": "MIT", "code": "x = 1\n" * 50}
    assert fase2.codigo_aceito(ok)
    assert not fase2.codigo_aceito({**ok, "license": "gpl-3.0"})
    assert not fase2.codigo_aceito({**ok, "language": "Java"})
    assert not fase2.codigo_aceito({**ok, "code": "x"})


def test_degraus_e_pesos():
    from heilo.training import fase2
    todas = {f: ["p"] for f in fase2.MISTURA}
    for d, alvo in enumerate(fase2.DEGRAUS):
        assert sum(fase2.pesos(d, todas).values()) == pytest.approx(alvo)
    assert fase2.pesos(2, todas)["geral_en"] == pytest.approx(0.21)          # fim: 60% português
    sem_codigo = {**todas, "codigo": []}
    assert "codigo" not in fase2.pesos(2, sem_codigo)                       # fonte sem parte não entra
    ini, n = 91552, fase2.PASSOS_FASE2
    assert [fase2.degrau(ini + x, ini) for x in (0, n // 3 + 1, 2 * n // 3 + 1, n)] == [0, 1, 2, 2]


def test_preparar_fontes_e_aplicar_pesos(tmp_path, monkeypatch):
    from heilo.models.seed.tokenizer import BPETokenizer
    from heilo.training import fase2, pretrain
    tok = BPETokenizer.treinar(["texto de exemplo em português, 123 e code"] * 50, vocab_size=300,
                               digitos_separados=True)
    monkeypatch.setattr(fase2, "TOKENS_FONTE", {"geral_en": 3000, "codigo": 3000, "contas": 3000, "conversa": None})
    monkeypatch.setattr(fase2, "textos_geral_en", lambda n, log=None: iter(["Knowledge text. " * 40] * 200))

    def sem_internet(n, log=None):
        raise ConnectionError("sem internet")
        yield  # pragma: no cover
    monkeypatch.setattr(fase2, "textos_codigo", sem_internet)
    conv = [{"messages": [{"role": "user", "content": "oi"}, {"role": "assistant", "content": "Olá, Hugo!"}]}] * 30
    logs = []
    prontas = fase2.preparar_fontes(tmp_path / "f2", tok, conv, log=logs.append)
    assert len(prontas["geral_en"]) == 1 and len(prontas["contas"]) == 1 and len(prontas["conversa"]) == 1
    assert prontas["codigo"] == [] and any("não consegui preparar codigo" in l for l in logs)
    w = fase2.pesos(0, prontas)
    pt = tmp_path / "pt"
    pretrain.preparar_corpus(["Português de verdade. " * 30] * 100, tok, pt, max_tokens=10 ** 9, log=lambda s: None)
    fase2.aplicar_pesos({**prontas, "pt": [pt]}, w)
    meta = json.loads((prontas["geral_en"][0] / "meta.json").read_text())
    assert meta["peso_fixo"] == pytest.approx(w["geral_en"])
    assert "peso_fixo" not in json.loads((pt / "meta.json").read_text())


def test_agenda_wsd_reaquece_fica_constante_e_cai_no_fim(tmp_path):
    torch = pytest.importorskip("torch")
    from heilo.models.seed.gpt import GPTConfig, MiniGPT
    from heilo.models.seed.tokenizer import BPETokenizer
    from heilo.training import faisca_grande, pretrain
    tok = BPETokenizer.treinar(["um texto qualquer para treinar 1 2 3"] * 50, vocab_size=300, digitos_separados=True)
    pretrain.preparar_corpus(["um texto qualquer para treinar. " * 20] * 200, tok, tmp_path / "c",
                             max_tokens=10 ** 9, log=lambda s: None)
    cfg = GPTConfig(block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0, vocab_size=tok.vocab_size,
                    arquitetura="moderna", n_kv_head=1)
    m = MiniGPT(cfg)
    torch.save({"modelo": m.state_dict(), "opt": torch.optim.AdamW(m.parameters()).state_dict(), "passo": 100,
                "hist": [], "config": cfg.__dict__}, tmp_path / "f1.pt")
    faisca_grande.crescer_profundidade(tmp_path / "f1.pt", tmp_path / "f2.pt", 4, log=lambda s: None)
    import dataclasses
    res = pretrain.pretreinar(dataclasses.replace(cfg, n_layer=4), tok, tmp_path / "c", tmp_path / "f2.pt",
                              passos=200, lote=2, lr=1e-3, lr_min=1e-4, aquecimento=10, avaliar_cada=10,
                              salvar_cada=1000, log=lambda s: None, agenda="wsd", frac_queda=0.2)
    lrs = {h["passo"]: h["lr"] for h in res["hist"] if "lr" in h}
    assert lrs[120] == pytest.approx(1e-3) and lrs[180] == pytest.approx(1e-3)   # constante
    assert lrs[190] < 1e-3 and lrs[200] == pytest.approx(1e-4, rel=0.1)        # queda só no fim
    assert res["passo"] == 200
