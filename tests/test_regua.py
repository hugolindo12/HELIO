"""Régua fixa (R1 perplexidade congelada, R2 gramática em pares) e o portão anti-esquecimento."""
import json

import numpy as np
import pytest


def _corpus(pasta, tokens):
    pasta.mkdir(parents=True)
    np.array(tokens, dtype=np.uint16).tofile(pasta / "val.bin")
    (pasta / "meta.json").write_text(json.dumps({"dtype": "uint16", "tokens_val": len(tokens)}))
    return pasta


def test_r2_tem_50_pares_validos():
    from heilo.training import regua
    pares = regua.carregar_r2()
    assert len(pares) == 50
    assert all(p["certa"] != p["errada"] and p["tipo"] for p in pares)
    assert len({p["certa"] for p in pares}) == 50


def test_r1_congela_uma_vez_e_nunca_muda(tmp_path):
    from heilo.training import regua
    a = _corpus(tmp_path / "wiki", list(range(1, 400)))
    b = _corpus(tmp_path / "web" / "parte_01", list(range(500, 900)))
    m1 = regua.congelar_r1([a, b], tmp_path / "regua", n_tokens=200, log=lambda s: None)
    t1 = np.load(tmp_path / "regua" / "r1_tokens.npy")
    assert m1["tokens"] == 200 and len(m1["fontes"]) == 2 and t1[0] == 1 and t1[100] == 500
    _corpus(tmp_path / "web" / "parte_02", list(range(1000, 1400)))
    m2 = regua.congelar_r1([tmp_path / "web" / "parte_02"], tmp_path / "regua", n_tokens=50, log=lambda s: None)
    assert m2 == m1 and (np.load(tmp_path / "regua" / "r1_tokens.npy") == t1).all()


def test_medir_e_portao(tmp_path):
    torch = pytest.importorskip("torch")
    from heilo.models.seed.gpt import GPTConfig, MiniGPT
    from heilo.models.seed.tokenizer import BPETokenizer
    from heilo.training import regua
    textos = [p["certa"] + " " + p["errada"] for p in regua.carregar_r2()] * 4
    tok = BPETokenizer.treinar(textos, vocab_size=400, digitos_separados=True)
    cfg = GPTConfig(block_size=32, n_layer=2, n_head=4, n_embd=64, dropout=0.0, vocab_size=tok.vocab_size,
                    arquitetura="moderna", n_kv_head=2)
    torch.manual_seed(0)
    m = MiniGPT(cfg)
    a = _corpus(tmp_path / "wiki", [int(i) % tok.vocab_size for i in range(3000)])
    regua.congelar_r1([a], tmp_path / "regua", n_tokens=2000, log=lambda s: None)
    r = regua.medir(m, tok, tmp_path / "regua", rotulo="teste", log=lambda s: None)
    assert r["r1_ppl"] > 1 and 0 <= r["r2_acerto"] <= 1
    assert regua.medir_r1(m, tmp_path / "regua") == pytest.approx(r["r1_ppl"], rel=1e-3)   # sempre igual
    assert len(json.loads((tmp_path / "regua" / "historico.json").read_text())) == 1
    antes = {"r1_ppl": 20.0, "r2_acerto": 0.80}
    assert regua.portao(antes, {"r1_ppl": 20.5, "r2_acerto": 0.80})["aprovado"]
    assert not regua.portao(antes, {"r1_ppl": 20.7, "r2_acerto": 0.80})["aprovado"]
    assert regua.portao(antes, {"r1_ppl": 19.0, "r2_acerto": 0.70})["aviso_r2"]
