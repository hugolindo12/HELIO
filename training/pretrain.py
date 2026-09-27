"""
Pré-treino + ajuste de conversa (SFT) do HEILO Seed v1.x.

Por que existe (auditoria de 27/09): o Seed v0.x (3,3 M, bytes, ~500 conversas)
só decorava. Para aprender português ele precisa antes ler MUITO texto geral.

Etapas (pensadas para GPU no Colab, mas rodam na CPU em miniatura para testes):
  1. tokenizer BPE próprio   (models/seed/tokenizer.py)
  2. corpus → arquivos .bin  (preparar_corpus)
  3. pré-treino              (pretreinar)  — prever o próximo token em texto geral
  4. SFT com o dataset HEILO (ajustar_conversa) — perda só nas falas da HEILO,
                                                   parada antecipada pela validação
  5. exportar checkpoint único (pesos + config + tokenizer)

Nenhum peso de modelo externo é usado: pesos aleatórios → aprendizado próprio.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from heilo.models.seed.gpt import GPTConfig, MiniGPT, encode_chat, salvar

Log = Callable[[str], None]


# ------------------------------------------------------------------ corpus
def preparar_corpus(textos: Iterable[str], tok, destino: Path, max_tokens: int,
                    frac_val: float = 0.005, lote: int = 2000, log: Log = print) -> Dict:
    """Tokeniza documentos em <doc> ... <fim> e grava train.bin / val.bin (uint16/uint32)."""
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    dtype = np.uint16 if tok.vocab_size < 65535 else np.uint32
    tr = open(destino / "train.bin", "wb")
    va = open(destino / "val.bin", "wb")
    n_tr = n_va = n_docs = 0
    buf: List[str] = []

    def despejar():
        nonlocal n_tr, n_va, n_docs
        for ids in tok.encode_lote(buf) if hasattr(tok, "encode_lote") else map(tok.encode, buf):
            arr = np.array([tok.doc] + ids + [tok.fim], dtype=dtype)
            n_docs += 1
            if (n_docs % int(1 / frac_val)) == 0:
                arr.tofile(va); n_va += len(arr)
            else:
                arr.tofile(tr); n_tr += len(arr)
        buf.clear()

    for t in textos:
        buf.append(t)
        if len(buf) >= lote:
            despejar()
            log(f"  corpus: {n_docs:,} documentos, {n_tr + n_va:,} tokens")
            if n_tr + n_va >= max_tokens:
                break
    if buf:
        despejar()
    tr.close(); va.close()
    if n_va == 0 and n_tr > 0:   # corpus pequeno demais para separar documentos: usa o final
        dados = np.fromfile(destino / "train.bin", dtype=dtype)
        corte = max(1, len(dados) // 100)
        dados[-corte:].tofile(destino / "val.bin")
        dados[:-corte].tofile(destino / "train.bin")
        n_tr, n_va = len(dados) - corte, corte
        log("  aviso: corpus pequeno; validação = último 1% dos tokens")
    meta = {"docs": n_docs, "tokens_treino": n_tr, "tokens_val": n_va, "dtype": np.dtype(dtype).name,
            "vocab_size": tok.vocab_size}
    (destino / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


class CorpusMulti:
    """Vários corpora (pastas com meta.json) vistos como um só: cada exemplo do lote
    vem de uma parte sorteada proporcionalmente ao tamanho dela."""

    def __init__(self, partes, fixos=None):
        """fixos[i] = fração fixa do lote para a parte i (ex.: 0.3 = 30% dos exemplos);
        None = proporcional ao tamanho, dividindo o que sobrar."""
        fixos = list(fixos or [None] * len(partes))
        pares = [(p, f) for p, f in zip(partes, fixos) if len(p) > 1]
        self.partes = [p for p, _ in pares]
        self.fixos = [f for _, f in pares]

    def _pesos(self, T: int):
        ok = [len(p) > T + 2 for p in self.partes]
        fixo_total = sum(f for f, v in zip(self.fixos, ok) if v and f)
        livres = sum(len(p) for p, f, v in zip(self.partes, self.fixos, ok) if v and not f)
        resto = max(0.0, 1.0 - fixo_total)
        w = np.array([0.0 if not v else (f if f else (resto * len(p) / livres if livres else 0.0))
                      for p, f, v in zip(self.partes, self.fixos, ok)], dtype=np.float64)
        return w / w.sum()

    def __len__(self):
        return int(sum(len(p) for p in self.partes))

    def lote(self, T: int, B: int, dev: str):
        escolha = np.random.choice(len(self.partes), size=B, p=self._pesos(T))
        xs, ys = [], []
        for k in escolha:
            d = self.partes[k]
            i = np.random.randint(0, len(d) - T - 1)
            xs.append(d[i:i + T].astype(np.int64)); ys.append(d[i + 1:i + 1 + T].astype(np.int64))
        x, y = torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))
        return x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)


def pastas_corpus(pasta) -> List[Path]:
    """Aceita uma pasta com meta.json, uma pasta com subpastas parte_* ou uma lista delas."""
    if isinstance(pasta, (list, tuple)):
        out: List[Path] = []
        for p in pasta:
            out += pastas_corpus(p)
        return out
    pasta = Path(pasta)
    if (pasta / "meta.json").exists():
        return [pasta]
    return sorted(p for p in pasta.glob("parte_*") if (p / "meta.json").exists())


def _abrir_bin(pasta, nome: str):
    pastas = pastas_corpus(pasta)
    if not pastas:
        raise FileNotFoundError(f"nenhum corpus pronto em {pasta}")
    arrays, fixos = [], []
    for p in pastas:
        meta = json.loads((p / "meta.json").read_text(encoding="utf-8"))
        arrays.append(np.memmap(p / f"{nome}.bin", dtype=meta["dtype"], mode="r"))
        fixos.append(meta.get("peso_fixo"))
    if len(arrays) == 1 and not fixos[0]:
        return arrays[0]
    return CorpusMulti(arrays, fixos)


def _lote(dados, T: int, B: int, dev: str):
    if isinstance(dados, CorpusMulti):
        return dados.lote(T, B, dev)
    ix = np.random.randint(0, len(dados) - T - 1, size=B)
    x = torch.from_numpy(np.stack([dados[i:i + T].astype(np.int64) for i in ix]))
    y = torch.from_numpy(np.stack([dados[i + 1:i + 1 + T].astype(np.int64) for i in ix]))
    return x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)


@torch.no_grad()
def _perda_val(modelo, dados, T, B, dev, lotes=20, amp=False):
    modelo.eval()
    ps = []
    for _ in range(lotes):
        x, y = _lote(dados, T, B, dev)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            _, loss = modelo(x, y)
        ps.append(loss.item())
    modelo.train()
    return sum(ps) / len(ps)


# ---------------------------------------------------------------- pré-treino
def pretreinar(cfg: GPTConfig, tok, pasta_corpus: Path, ckpt: Path, passos: int = 20000,
               lote: int = 32, lr: float = 6e-4, lr_min: float = 6e-5, aquecimento: int = 500,
               tempo_max_min: Optional[float] = None, avaliar_cada: int = 500,
               salvar_cada: int = 1000, log: Log = print, acumular: int = 1,
               minutos_alvo: Optional[float] = None) -> Dict:
    """Pré-treino com retomada: se `ckpt` existir, continua de onde parou.

    minutos_alvo=M: a agenda da taxa de aprendizado segue o TEMPO (não uma estimativa
    de passos): treina M minutos no total (somando retomadas) e termina com a taxa no
    mínimo. Evita o erro da v2.0, em que a estimativa de velocidade errou e o treino
    parou com 24 min em vez de 210.

    acumular=k: soma o gradiente de k micro-lotes antes de cada passo (lote efetivo
    = lote × k), para caber modelos maiores na memória da GPU."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    amp = dev == "cuda"
    tr, va = _abrir_bin(pasta_corpus, "train"), _abrir_bin(pasta_corpus, "val")
    cfg.vocab_size = tok.vocab_size
    modelo = MiniGPT(cfg).to(dev)
    modelo.tokenizer = tok
    opt = torch.optim.AdamW(modelo.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    passo, hist, min_antes, inicio_agenda = 0, [], 0.0, 0
    ckpt = Path(ckpt)
    if ckpt.exists():
        st = torch.load(ckpt, map_location=dev, weights_only=False)
        modelo.load_state_dict(st["modelo"])
        opt.load_state_dict(st["opt"])
        passo, hist = st["passo"], st["hist"]
        min_antes = st.get("minutos_agenda", 0.0) if minutos_alvo else 0.0
        inicio_agenda = st.get("inicio_agenda", passo) if minutos_alvo and "minutos_agenda" in st else passo
        log(f"[pré-treino] retomando do passo {passo}" + (f" ({min_antes:.0f} min já feitos)" if minutos_alvo else ""))
    log(f"[pré-treino] {modelo.n_params()/1e6:.1f} M parâmetros | {len(tr):,} tokens de treino | "
        f"{dev} | lote {lote}x{cfg.block_size}")

    def lr_em(p):
        if p - inicio_agenda < aquecimento:
            return lr * (p - inicio_agenda + 1) / aquecimento
        if minutos_alvo:
            prog = min(1.0, (min_antes + (time.time() - inicio) / 60) / minutos_alvo)
        else:
            prog = min(1.0, (p - aquecimento) / max(1, passos - aquecimento))
        return lr_min + (lr - lr_min) * 0.5 * (1 + math.cos(math.pi * prog))

    def guardar():
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        tmp = ckpt.with_suffix(".tmp")
        torch.save({"modelo": modelo.state_dict(), "opt": opt.state_dict(), "passo": passo,
                    "hist": hist, "config": cfg.__dict__, "inicio_agenda": inicio_agenda,
                    "minutos_agenda": min_antes + (time.time() - inicio) / 60}, tmp)
        tmp.replace(ckpt)

    inicio, media = time.time(), None
    modelo.train()
    while passo < passos:
        if tempo_max_min and (time.time() - inicio) / 60 > tempo_max_min:
            log(f"[pré-treino] limite de tempo atingido no passo {passo}")
            break
        if minutos_alvo and min_antes + (time.time() - inicio) / 60 >= minutos_alvo:
            log(f"[pré-treino] {minutos_alvo:.0f} min de treino completos no passo {passo}")
            break
        for g in opt.param_groups:
            g["lr"] = lr_em(passo)
        opt.zero_grad(set_to_none=True)
        for _ in range(max(1, acumular)):
            x, y = _lote(tr, cfg.block_size, lote, dev)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                _, loss = modelo(x, y)
            scaler.scale(loss / max(1, acumular)).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        passo += 1
        media = loss.item() if media is None else 0.98 * media + 0.02 * loss.item()
        if passo % avaliar_cada == 0 or passo == passos:
            pv = _perda_val(modelo, va, cfg.block_size, lote, dev, amp=amp)
            hist.append({"passo": passo, "perda_treino": round(media, 4), "perda_val": round(pv, 4),
                         "ppl_val": round(math.exp(pv), 2), "min": round((time.time() - inicio) / 60, 1)})
            log(f"  passo {passo}/{passos} | treino {media:.3f} | val {pv:.3f} (ppl {math.exp(pv):.1f}) "
                f"| {(time.time() - inicio) / 60:.1f} min")
        if passo % salvar_cada == 0:
            guardar()
    guardar()
    modelo.eval()
    return {"modelo": modelo, "passo": passo, "hist": hist,
            "minutos": round((time.time() - inicio) / 60, 1)}


# -------------------------------------------------------------------- SFT
def _exemplos_sft(exemplos: List[Dict], tok, T: int):
    """(ids, rótulos) com perda SÓ nas falas da HEILO (+ <fim>)."""
    out = []
    for ex in exemplos:
        ids, lab = [], []
        for m in ex["messages"]:
            pedaco = encode_chat([m], tok=tok)
            ids += pedaco
            lab += (pedaco if m["role"] == "assistant" else [-100] * len(pedaco))
        ids, lab = ids[:T + 1], lab[:T + 1]
        if len(ids) >= 2 and any(l != -100 for l in lab[1:]):
            out.append((ids, lab))
    return out


def _lote_sft(itens, B, pad, dev, rnd=None):
    esc = [itens[i] for i in (rnd or np.random).randint(0, len(itens), size=B)]
    L = max(len(i) for i, _ in esc) - 1
    x = torch.full((B, L), pad, dtype=torch.long)
    y = torch.full((B, L), -100, dtype=torch.long)
    for k, (ids, lab) in enumerate(esc):
        n = len(ids) - 1
        x[k, :n] = torch.tensor(ids[:-1])
        y[k, :n] = torch.tensor(lab[1:])
    return x.to(dev), y.to(dev)


def _perda_sft(modelo, itens, pad, dev, amp):
    modelo.eval()
    s = n = 0
    with torch.no_grad():
        for i in range(0, len(itens), 16):
            parte = itens[i:i + 16]
            L = max(len(a) for a, _ in parte) - 1
            x = torch.full((len(parte), L), pad, dtype=torch.long)
            y = torch.full((len(parte), L), -100, dtype=torch.long)
            for k, (ids, lab) in enumerate(parte):
                x[k, :len(ids) - 1] = torch.tensor(ids[:-1])
                y[k, :len(ids) - 1] = torch.tensor(lab[1:])
            x, y = x.to(dev), y.to(dev)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                logits, _ = modelo(x)
            l = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1),
                                ignore_index=-100, reduction="sum")
            s += l.item(); n += (y != -100).sum().item()
    modelo.train()
    return s / max(1, n)


def ajustar_conversa(modelo, treino: List[Dict], val: List[Dict], passos: int = 1500, lote: int = 32,
                     lr: float = 1e-4, avaliar_cada: int = 100, paciencia: int = 4,
                     log: Log = print, pasta_corpus: Optional[Path] = None,
                     peso_texto: float = 0.3, lote_texto: int = 8) -> Dict:
    """SFT com parada antecipada: guarda o estado com MENOR perda de validação.

    pasta_corpus (opcional): a cada passo soma `peso_texto` × perda de modelagem de
    linguagem num lote do corpus do pré-treino. Evita que um dataset de conversa
    pequeno faça o modelo esquecer o português e colar em poucas frases prontas
    (o que aconteceu na v1.0)."""
    dev = next(modelo.parameters()).device.type
    amp = dev == "cuda"
    tok = modelo.tokenizer
    T = modelo.cfg.block_size
    it_tr, it_va = _exemplos_sft(treino, tok, T), _exemplos_sft(val, tok, T)
    texto = _abrir_bin(pasta_corpus, "train") if pasta_corpus else None
    opt = torch.optim.AdamW(modelo.parameters(), lr=lr, weight_decay=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    melhor, melhor_estado, sem_melhora, hist = None, None, 0, []
    rnd = np.random.RandomState(0)
    base = _perda_sft(modelo, it_va, tok.pad, dev, amp) if it_va else None
    log(f"[SFT] {len(it_tr)} exemplos de treino, {len(it_va)} de validação | perda val inicial {base}")
    modelo.train()
    for p in range(1, passos + 1):
        x, y = _lote_sft(it_tr, min(lote, len(it_tr)), tok.pad, dev, rnd)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            logits, _ = modelo(x)
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)
        if texto is not None and peso_texto > 0:
            xt, yt = _lote(texto, T, lote_texto, dev)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                _, perda_txt = modelo(xt, yt)
            loss_total = loss + peso_texto * perda_txt.float()
        else:
            loss_total = loss
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss_total).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        if it_va and (p % avaliar_cada == 0 or p == passos):
            pv = _perda_sft(modelo, it_va, tok.pad, dev, amp)
            hist.append({"passo": p, "perda_treino": round(loss.item(), 4), "perda_val": round(pv, 4)})
            log(f"  SFT passo {p} | treino {loss.item():.3f} | val {pv:.3f}")
            if melhor is None or pv < melhor - 1e-4:
                melhor, sem_melhora = pv, 0
                melhor_estado = {k: v.detach().clone() for k, v in modelo.state_dict().items()}
            else:
                sem_melhora += 1
                if sem_melhora >= paciencia:
                    log(f"[SFT] parada antecipada no passo {p} (validação parou de melhorar)")
                    break
    if melhor_estado is not None:
        modelo.load_state_dict(melhor_estado)
    modelo.eval()
    return {"perda_val_inicial": base, "melhor_perda_val": melhor, "hist": hist}


def exportar(modelo, arquivo: Path, info: Dict) -> Path:
    """Checkpoint único (pesos fp16 + config + tokenizer) — carrega com gpt.carregar()."""
    modelo.cfg.dropout = 0.0
    return salvar(modelo, info, Path(arquivo), meia_precisao=True)
