"""
Régua fixa do cérebro HEILO — mede sempre do MESMO jeito, para comparar sessões, fases e tamanhos.

Por que existe: a "val" do pré-treino muda toda vez que entra uma parte nova de texto, então a
perplexidade do log não compara bem um dia com o outro. A régua é congelada uma vez e nunca muda.

- R1: perplexidade num conjunto FIXO de texto em português que o modelo nunca treinou
      (pedaços da validação da Wikipédia e da 1ª parte da web). Menor é melhor.
- R2: gramática em pares. Para cada par (frase certa × frase errada que difere por um detalhe),
      o modelo acerta se achar a certa mais provável. Maior é melhor.
- R2b: o mesmo, com 60 pares difíceis (crase, sujeito distante, verbos impessoais, subjuntivo,
      particípios, plurais irregulares). Criado quando a Faísca 200M chegou a 50/50 no R2.

Portão anti-esquecimento (plano de evolução): depois de crescer ou treinar outra fase,
R1 não pode piorar mais de 3% em relação à versão anterior.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

Log = Callable[[str], None]

ARQ_R2 = Path(__file__).resolve().parents[1] / "data" / "eval" / "regua_r2_gramatica.jsonl"
ARQ_R2B = ARQ_R2.parent / "regua_r2b_gramatica_dificil.jsonl"
TOLERANCIA_R1 = 0.03   # R1 pode piorar no máximo 3%


# ------------------------------------------------------------------ R1
def congelar_r1(pastas_val: List[Path], destino: Path, n_tokens: int = 1_000_000,
                log: Log = print) -> Dict:
    """Cria (uma única vez) o conjunto fixo do R1 a partir do val.bin das pastas dadas,
    em partes iguais. Se já existe, NÃO mexe: a régua não pode mudar."""
    destino = Path(destino)
    arq, meta_arq = destino / "r1_tokens.npy", destino / "r1_meta.json"
    if arq.exists() and meta_arq.exists():
        return json.loads(meta_arq.read_text(encoding="utf-8"))
    destino.mkdir(parents=True, exist_ok=True)
    por_fonte = n_tokens // max(1, len(pastas_val))
    pedacos, fontes = [], []
    for p in pastas_val:
        p = Path(p)
        meta = json.loads((p / "meta.json").read_text(encoding="utf-8"))
        va = np.memmap(p / "val.bin", dtype=meta["dtype"], mode="r")
        pedacos.append(np.array(va[:por_fonte]))
        fontes.append({"pasta": f"{p.parent.name}/{p.name}", "tokens": int(min(por_fonte, len(va)))})
    tokens = np.concatenate(pedacos)
    tmp = destino / "r1_tokens.tmp.npy"
    np.save(tmp, tokens)
    tmp.replace(arq)
    meta = {"tokens": int(len(tokens)), "fontes": fontes, "criado": time.strftime("%Y-%m-%d %H:%M"),
            "dtype": str(tokens.dtype)}
    meta_arq.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"régua R1 congelada: {len(tokens):,} tokens de {len(fontes)} fonte(s)")
    return meta


def medir_r1(modelo, pasta_regua: Path, T: int = 1024, B: int = 8, dev: Optional[str] = None,
             max_janelas: Optional[int] = None) -> float:
    """Perplexidade no conjunto fixo, em janelas seguidas (sem sorteio: sempre o mesmo número)."""
    import torch
    import torch.nn.functional as F
    dev = dev or next(modelo.parameters()).device
    tokens = np.load(Path(pasta_regua) / "r1_tokens.npy").astype(np.int64)
    T = min(T, modelo.cfg.block_size)
    n = (len(tokens) - 1) // T
    if max_janelas:
        n = min(n, max_janelas)
    if n < 1:
        raise ValueError("conjunto R1 pequeno demais para o contexto do modelo")
    era_treino = modelo.training
    modelo.eval()
    soma, cont = 0.0, 0
    usar_amp = str(dev).startswith("cuda")
    with torch.no_grad():
        for i in range(0, n, B):
            idx = range(i, min(n, i + B))
            x = torch.from_numpy(np.stack([tokens[j * T:(j + 1) * T] for j in idx])).to(dev)
            y = torch.from_numpy(np.stack([tokens[j * T + 1:(j + 1) * T + 1] for j in idx])).to(dev)
            with torch.autocast("cuda", dtype=torch.float16, enabled=usar_amp):
                logits, _ = modelo(x)
            perda = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1), reduction="sum")
            soma += perda.item()
            cont += y.numel()
    if era_treino:
        modelo.train()
    return math.exp(soma / cont)


# ------------------------------------------------------------------ R2
def carregar_r2(arq: Path = ARQ_R2) -> List[Dict]:
    return [json.loads(l) for l in Path(arq).read_text(encoding="utf-8").splitlines() if l.strip()]


def _logprob(modelo, tok, texto: str, dev) -> float:
    import torch
    import torch.nn.functional as F
    ids = [tok.doc] + tok.encode(texto)
    x = torch.tensor([ids[:-1]], device=dev)
    y = torch.tensor([ids[1:]], device=dev)
    logits, _ = modelo(x)
    lp = F.log_softmax(logits.float(), dim=-1).gather(-1, y.unsqueeze(-1))
    return float(lp.sum())


def medir_r2(modelo, tok, pares: Optional[List[Dict]] = None, dev: Optional[str] = None) -> Dict:
    """Acerto nos pares de gramática: a frase certa precisa ser mais provável que a errada."""
    import torch
    pares = pares if pares is not None else carregar_r2()
    dev = dev or next(modelo.parameters()).device
    era_treino = modelo.training
    modelo.eval()
    acertos, por_tipo = 0, {}
    with torch.no_grad():
        for p in pares:
            ok = _logprob(modelo, tok, p["certa"], dev) > _logprob(modelo, tok, p["errada"], dev)
            acertos += ok
            t = por_tipo.setdefault(p.get("tipo", "outro"), [0, 0])
            t[0] += ok
            t[1] += 1
    if era_treino:
        modelo.train()
    return {"acerto": round(acertos / max(1, len(pares)), 4), "acertos": acertos, "total": len(pares),
            "por_tipo": {k: f"{a}/{n}" for k, (a, n) in sorted(por_tipo.items())}}


# ------------------------------------------------------------------ juntos
def medir(modelo, tok, pasta_regua: Path, rotulo: str = "", log: Log = print,
          max_janelas: Optional[int] = None) -> Dict:
    """Mede R1 e R2, grava no histórico da régua (pasta_regua/historico.json) e devolve o resultado."""
    t0 = time.time()
    r1 = medir_r1(modelo, pasta_regua, max_janelas=max_janelas)
    r2 = medir_r2(modelo, tok)
    r2b = medir_r2(modelo, tok, carregar_r2(ARQ_R2B)) if ARQ_R2B.exists() else None
    res = {"rotulo": rotulo, "data": time.strftime("%Y-%m-%d %H:%M"),
           "params_M": round(sum(p.numel() for p in modelo.parameters()) / 1e6, 1),
           "r1_ppl": round(r1, 3), "r2_acerto": r2["acerto"], "r2_por_tipo": r2["por_tipo"]}
    if r2b:
        res.update({"r2b_acerto": r2b["acerto"], "r2b_por_tipo": r2b["por_tipo"]})
    hist_arq = Path(pasta_regua) / "historico.json"
    hist = json.loads(hist_arq.read_text(encoding="utf-8")) if hist_arq.exists() else []
    hist.append(res)
    hist_arq.write_text(json.dumps(hist, indent=1, ensure_ascii=False), encoding="utf-8")
    log(f"RÉGUA {rotulo}: R1 perplexidade fixa {r1:.2f} | R2 gramática {r2['acertos']}/{r2['total']}"
        + (f" | R2b difícil {r2b['acertos']}/{r2b['total']}" if r2b else "")
        + f" ({time.time() - t0:.0f} s)")
    return res


def portao(antes: Dict, depois: Dict, tol_r1: float = TOLERANCIA_R1) -> Dict:
    """Aprova o modelo novo só se o português não piorou: R1 no máximo +3%.
    R2 é informativo (um aviso se cair mais de 5 pontos)."""
    lim = antes["r1_ppl"] * (1 + tol_r1)
    ok = depois["r1_ppl"] <= lim
    var = depois["r1_ppl"] / antes["r1_ppl"] - 1
    aviso = any(k in antes and k in depois and depois[k] - antes[k] < -0.05 for k in ("r2_acerto", "r2b_acerto"))
    motivo = (f"R1 {antes['r1_ppl']:.2f} → {depois['r1_ppl']:.2f} ({var:+.1%}; limite +{tol_r1:.0%})"
              f" | R2 {antes['r2_acerto']:.0%} → {depois['r2_acerto']:.0%}"
              + (f" | R2b {antes['r2b_acerto']:.0%} → {depois['r2b_acerto']:.0%}"
                 if "r2b_acerto" in antes and "r2b_acerto" in depois else "")
              + (" — atenção: gramática caiu" if aviso else ""))
    return {"aprovado": ok, "motivo": motivo, "variacao_r1": round(var, 4), "aviso_r2": aviso}
