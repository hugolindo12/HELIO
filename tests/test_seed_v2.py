"""HEILO Seed v2.x: arquitetura moderna (RoPE, RMSNorm, GeGLU, GQA, QK-norm)."""
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from heilo.models.seed.gpt import GPTConfig, MiniGPT, MiniGPTChat, carregar, salvar  # noqa: E402
from heilo.models.seed.tokenizer import BPETokenizer  # noqa: E402


def _tok():
    txt = ["Olá! Eu sou a HEILO, criada pelo Hugo. " * 30, "def soma(a, b):\n    return a + b\n" * 30]
    return BPETokenizer.treinar(iter(txt), vocab_size=300)


def _modelo(tok, **kw):
    cfg = GPTConfig(block_size=64, n_layer=2, n_head=4, n_embd=64, dropout=0.0, vocab_size=tok.vocab_size,
                    arquitetura="moderna", n_kv_head=2, **kw)
    m = MiniGPT(cfg)
    m.tokenizer = tok
    return m


def test_forward_backward_e_aprende():
    tok = _tok()
    torch.manual_seed(0)
    m = _modelo(tok)
    dados = torch.tensor(tok.encode("Olá! Eu sou a HEILO, criada pelo Hugo. " * 10))[:65]
    x, y = dados[:-1][None], dados[1:][None]
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    inicial = m(x, y)[1].item()
    for _ in range(40):
        loss = m(x, y)[1]
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < inicial * 0.5


def test_salvar_carregar_mesma_saida(tmp_path):
    tok = _tok()
    m = _modelo(tok).eval()
    ids = torch.tensor([tok.encode("def soma")])
    antes = m(ids)[0]
    arq = salvar(m, {"versao": "teste"}, tmp_path / "v2.pt")
    m2, info = carregar(arq, dispositivo="cpu")
    assert m2.cfg.arquitetura == "moderna" and m2.cfg.n_kv_head == 2 and info["versao"] == "teste"
    assert torch.allclose(antes, m2(ids)[0], atol=1e-5)
    assert isinstance(MiniGPTChat(arquivo=arq).responder([{"role": "user", "content": "oi"}], max_novos=5), str)


def test_contexto_menor_que_o_bloco_e_geracao():
    tok = _tok()
    m = _modelo(tok).eval()
    novos = m.gerar(tok.encode("Olá"), max_novos=80, temperatura=0.8, parar_em=())
    assert len(novos) == 80                     # passa do block_size: janela desliza sem erro


def test_checkpoint_antigo_gpt2_continua_carregando(tmp_path):
    cfg = GPTConfig(block_size=32, n_layer=1, n_head=2, n_embd=32, dropout=0.0)
    m = MiniGPT(cfg)
    arq = salvar(m, {}, tmp_path / "antigo.pt")
    ck = torch.load(arq, weights_only=False)
    for k in ("arquitetura", "n_kv_head", "rope_base"):   # simula checkpoint salvo antes da v2
        ck["config"].pop(k)
    torch.save(ck, arq)
    m2, _ = carregar(arq, dispositivo="cpu")
    assert m2.cfg.arquitetura == "gpt2" and hasattr(m2, "pos")
