"""Destilação com vocabulários diferentes: alinhamento por caractere + projeção do vocabulário."""
import itertools
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from heilo.models.seed.gpt import GPTConfig, MiniGPT  # noqa: E402
from heilo.models.seed.tokenizer import BPETokenizer  # noqa: E402
from heilo.training import destilacao as d  # noqa: E402

TEXTO = "O Brasil é um país da América do Sul. A capital do Brasil é Brasília. " * 20


class _HFFalso:
    """Imita um tokenizer rápido do Hugging Face (o do professor) com outro vocabulário."""

    def __init__(self, bpe):
        self.bpe, self.pad_token_id = bpe, bpe.pad

    def __len__(self):
        return self.bpe.vocab_size

    def __call__(self, texto, add_special_tokens=False, return_offsets_mapping=True):
        e = self.bpe.tok.encode(texto, add_special_tokens=False)
        return {"input_ids": e.ids, "offset_mapping": e.offsets}

    def decode(self, ids):
        return self.bpe.tok.decode(ids)


class _ProfessorFalso(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask=None):
        return SimpleNamespace(logits=self.m(input_ids)[0])


def test_alinhar_e_projetar():
    assert d.alinhar([(0, 1), (1, 3), (3, 6)], [(0, 3), (3, 4), (4, 6)]) == [(1, 0)]
    mapa = torch.tensor([2, -1, 0])
    alvo = d.alvo_projetado(torch.tensor([[5.0, 5.0, 0.0]]), mapa, vocab_heilo=3, k=3, temperatura=1.0)
    assert torch.allclose(alvo.sum(), torch.tensor(1.0)) and alvo[0, 2] > alvo[0, 0] and alvo[0, 1] == 0


def test_destilar_com_vocabularios_diferentes():
    aluno_tok = BPETokenizer.treinar(iter([TEXTO] * 5), vocab_size=320)
    prof_tok = BPETokenizer.treinar(iter([TEXTO.upper() + TEXTO] * 5), vocab_size=400)
    hf = _HFFalso(prof_tok)
    mapa = d.mapa_vocab(d.TokHF(hf).textos_dos_tokens(), d.TokHeilo(aluno_tok))
    assert (mapa >= 0).float().mean() > 0.5
    prof = _ProfessorFalso(MiniGPT(GPTConfig(block_size=128, n_layer=1, n_head=2, n_embd=32, vocab_size=400)))
    cfg = GPTConfig(block_size=128, n_layer=1, n_head=2, n_embd=32, dropout=0.0, vocab_size=320,
                    arquitetura="moderna", n_kv_head=1)
    aluno = MiniGPT(cfg)
    textos = itertools.cycle([TEXTO[i:i + 300] for i in range(0, 1200, 37)])
    r = d.destilar(aluno, aluno_tok, textos, prof, hf, mapa, minutos=0.05, lote=2, max_tokens=128,
                   aquecimento=2, log=lambda *_: None)
    assert r["passos"] > 3
