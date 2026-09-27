"""
Destilação por logits: o HEILO Seed aprende a IMITAR A DISTRIBUIÇÃO de probabilidades
de um professor (ex.: Qwen2.5-1.5B), e não só o texto final.

O problema: os vocabulários são diferentes (HEILO ~16 mil tokens × Qwen ~151 mil).
A solução usada aqui:

1. ALINHAR POSIÇÕES pelo caractere: o mesmo texto é tokenizado pelos dois; só usamos as
   posições em que os dois tokenizers terminam um token no MESMO caractere (fronteiras
   em comum, quase sempre entre palavras). Nesses pontos, "qual é o próximo pedaço?" é a
   mesma pergunta para os dois.
2. PROJETAR O VOCABULÁRIO: cada token do professor vira o 1º token da HEILO que escreve o
   mesmo texto (tabela calculada uma vez). A distribuição top-k do professor é somada nos
   tokens correspondentes da HEILO → vira um alvo "suave" no vocabulário dela.
3. PERDA = alfa × entropia cruzada normal (o texto real) + beta × destilação (o alvo suave),
   com temperatura. Os pesos da HEILO continuam sendo só dela; o professor é removível.

Aproximação honesta: quando o próximo token da HEILO é mais longo que o do professor, a
projeção fica um pouco "curta". Por isso a entropia cruzada normal continua junto.
"""
from __future__ import annotations

import math
import time
import unicodedata
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

Log = Callable[[str], None]


# ------------------------------------------------------------- tokenizers
class TokHeilo:
    """Interface mínima sobre o BPETokenizer da HEILO."""

    def __init__(self, bpe):
        self.bpe = bpe
        self.vocab_size = bpe.vocab_size

    def encode_offsets(self, texto: str) -> Tuple[List[int], List[Tuple[int, int]]]:
        e = self.bpe.tok.encode(texto, add_special_tokens=False)
        return e.ids, e.offsets

    def encode_lote(self, textos: List[str]) -> List[List[int]]:
        return self.bpe.encode_lote(textos)


class TokHF:
    """Interface mínima sobre um tokenizer rápido do Hugging Face (o do professor)."""

    def __init__(self, hf):
        self.hf = hf
        self.vocab_size = len(hf)

    def encode_offsets(self, texto: str) -> Tuple[List[int], List[Tuple[int, int]]]:
        e = self.hf(texto, add_special_tokens=False, return_offsets_mapping=True)
        return e["input_ids"], [tuple(o) for o in e["offset_mapping"]]

    def textos_dos_tokens(self) -> List[str]:
        return [self.hf.decode([i]) for i in range(self.vocab_size)]


# -------------------------------------------------------------- mapeamento
def mapa_vocab(textos_prof: Sequence[str], tok_heilo: TokHeilo) -> torch.Tensor:
    """mapa[t] = 1º token da HEILO que escreve o texto do token t do professor (-1 = sem par)."""
    out = torch.full((len(textos_prof),), -1, dtype=torch.long)
    validos = [(i, t) for i, t in enumerate(textos_prof) if t and "�" not in t]
    for (i, _), ids in zip(validos, tok_heilo.encode_lote([t for _, t in validos])):
        if ids:
            out[i] = ids[0]
    return out


def alinhar(off_a: Sequence[Tuple[int, int]], off_b: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Pares (i, j) em que o token i de A e o token j de B terminam no mesmo caractere,
    e ambos ainda têm um próximo token (a posição prevê alguma coisa)."""
    fim_b = {}
    for j, (_, fim) in enumerate(off_b[:-1]):
        fim_b.setdefault(fim, j)
    pares = []
    for i, (_, fim) in enumerate(off_a[:-1]):
        j = fim_b.get(fim)
        if j is not None and fim > 0:
            pares.append((i, j))
    return pares


def alvo_projetado(logits_prof: torch.Tensor, mapa: torch.Tensor, vocab_heilo: int,
                   k: int = 50, temperatura: float = 2.0, bloco: int = 256) -> torch.Tensor:
    """[N, Vp] logits do professor → [N, Vh] distribuição-alvo no vocabulário da HEILO.
    Processa em blocos: com o vocabulário do Qwen (~152 mil), N×Vp em float32 não cabe na GPU."""
    m = mapa.to(logits_prof.device)
    if m.numel() < logits_prof.size(-1):   # o modelo costuma ter mais saídas que tokens reais (ex.: Qwen)
        m = torch.cat([m, torch.full((logits_prof.size(-1) - m.numel(),), -1, dtype=m.dtype, device=m.device)])
    saidas = []
    for a in range(0, logits_prof.size(0), bloco):
        lg = logits_prof[a:a + bloco].float() / temperatura
        lse = torch.logsumexp(lg, dim=-1, keepdim=True)
        top_l, top_i = lg.topk(min(k, lg.size(-1)), dim=-1)
        top_p = (top_l - lse).exp()
        destino = m[top_i]
        ok = destino >= 0
        alvo = torch.zeros(lg.size(0), vocab_heilo, device=lg.device)
        alvo.scatter_add_(1, destino.clamp(min=0), top_p * ok)
        saidas.append(alvo / alvo.sum(-1, keepdim=True).clamp(min=1e-8))
    return torch.cat(saidas)


def perda_destilacao(logits_aluno: torch.Tensor, alvo: torch.Tensor, temperatura: float = 2.0) -> torch.Tensor:
    """Entropia cruzada suave (≡ KL até uma constante), escalada por T² (Hinton et al.)."""
    logp = F.log_softmax(logits_aluno.float() / temperatura, dim=-1)
    return -(alvo * logp).sum(-1).mean() * temperatura ** 2


# ------------------------------------------------------------------- treino
def _normalizar(t: str) -> str:
    return unicodedata.normalize("NFC", t)


def preparar_exemplo(texto: str, th: TokHeilo, tp, doc_id: int, max_tokens: int):
    """Corta o texto para caber nos dois tokenizers e devolve ids + pares alinhados."""
    texto = _normalizar(texto)
    ids_h, off_h = th.encode_offsets(texto)
    if len(ids_h) > max_tokens - 1:
        corte = off_h[max_tokens - 2][1]
        texto = texto[:corte]
        ids_h, off_h = th.encode_offsets(texto)
    ids_p, off_p = tp.encode_offsets(texto)
    ids_p, off_p = ids_p[:max_tokens], off_p[:max_tokens]
    ids_h = [doc_id] + ids_h                                  # <doc> no começo, como no pré-treino
    off_h = [(0, 0)] + list(off_h)
    pares = alinhar(off_h, off_p)
    return ids_h, ids_p, pares


def destilar(modelo, tok_bpe, textos: Iterator[str], professor, tok_prof_hf, mapa: torch.Tensor,
             minutos: float = 90, lote: int = 16, max_tokens: int = 512, lr: float = 1e-4,
             lr_min: float = 1e-5, alfa: float = 0.5, beta: float = 0.5, temperatura: float = 2.0,
             k: int = 50, aquecimento: int = 200, log: Log = print,
             salvar: Optional[Callable[[int], None]] = None, salvar_cada: int = 500) -> Dict:
    """Treina o aluno (HEILO) com CE + destilação do professor, por `minutos` (pelo relógio)."""
    dev = next(modelo.parameters()).device
    amp = dev.type == "cuda"
    max_tokens = min(max_tokens, modelo.cfg.block_size)
    th, tp = TokHeilo(tok_bpe), TokHF(tok_prof_hf)
    pad_h = tok_bpe.pad
    pad_p = tok_prof_hf.pad_token_id if tok_prof_hf.pad_token_id is not None else 0
    opt = torch.optim.AdamW(modelo.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    professor.eval()
    modelo.train()
    inicio, passo, hist = time.time(), 0, []
    media_ce = media_kd = None
    while (time.time() - inicio) / 60 < minutos:
        exs = []
        while len(exs) < lote:
            ex = preparar_exemplo(next(textos), th, tp, tok_bpe.doc, max_tokens)
            if len(ex[0]) > 8 and ex[2]:
                exs.append(ex)
        Lh = max(len(e[0]) for e in exs)
        Lp = max(len(e[1]) for e in exs)
        xh = torch.full((lote, Lh), pad_h, dtype=torch.long)
        yh = torch.full((lote, Lh), -100, dtype=torch.long)
        xp = torch.full((lote, Lp), pad_p, dtype=torch.long)
        mp = torch.zeros((lote, Lp), dtype=torch.long)
        for b, (ih, ip, _) in enumerate(exs):
            xh[b, :len(ih)] = torch.tensor(ih)
            yh[b, :len(ih) - 1] = torch.tensor(ih[1:])
            xp[b, :len(ip)] = torch.tensor(ip)
            mp[b, :len(ip)] = 1
        xh, yh, xp, mp = xh.to(dev), yh.to(dev), xp.to(dev), mp.to(dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            lp = professor(input_ids=xp, attention_mask=mp).logits
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            lh, _ = modelo(xh)
        ce = F.cross_entropy(lh.float().reshape(-1, lh.size(-1)), yh.reshape(-1),
                             ignore_index=-100)
        bi = [b for b, (_, _, pares) in enumerate(exs) for _ in pares]
        ii = [i for _, _, pares in exs for i, _ in pares]
        jj = [j for _, _, pares in exs for _, j in pares]
        bi_t = torch.tensor(bi, device=dev)
        alvo = alvo_projetado(lp[bi_t, torch.tensor(jj, device=dev)], mapa, lh.size(-1), k, temperatura)
        kd = perda_destilacao(lh[bi_t, torch.tensor(ii, device=dev)], alvo, temperatura)
        perda = alfa * ce + beta * kd
        frac = min(1.0, (time.time() - inicio) / 60 / minutos)
        taxa = lr * (passo + 1) / aquecimento if passo < aquecimento else \
            lr_min + (lr - lr_min) * 0.5 * (1 + math.cos(math.pi * frac))
        for g in opt.param_groups:
            g["lr"] = taxa
        opt.zero_grad(set_to_none=True)
        scaler.scale(perda).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        passo += 1
        media_ce = ce.item() if media_ce is None else 0.98 * media_ce + 0.02 * ce.item()
        media_kd = kd.item() if media_kd is None else 0.98 * media_kd + 0.02 * kd.item()
        if passo % 100 == 0:
            alinh = len(ii) / max(1, sum(len(e[0]) for e in exs))
            hist.append({"passo": passo, "ce": round(media_ce, 4), "kd": round(media_kd, 4),
                         "posicoes_alinhadas": round(alinh, 3), "min": round((time.time() - inicio) / 60, 1)})
            log(f"  passo {passo} | CE {media_ce:.3f} | destilação {media_kd:.3f} | "
                f"posições alinhadas {alinh:.0%} | {(time.time() - inicio) / 60:.1f} min")
        if salvar and passo % salvar_cada == 0:
            salvar(passo)
    modelo.eval()
    return {"passos": passo, "hist": hist, "minutos": round((time.time() - inicio) / 60, 1)}
