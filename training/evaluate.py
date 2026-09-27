"""
Avaliação REAL do HEILO Seed num conjunto fixo que nunca entra no treino.

Métricas (todas calculadas, nenhuma estimada):
- perda / perplexidade: da resposta de referência, dado a pergunta (sem amostragem)
- acerto (greedy): resposta mais provável cumpre os critérios objetivos do item
- acerto amostrado: média de acerto em N amostras (temperatura 0.7)
- consistência: quão estável o acerto é entre as N amostras (1.0 = sempre igual)
- seguir instruções: acerto nos itens do tipo "instrucao"
- generalização: acerto em "generalizacao" (assunto/combinação não vista) vs "paráfrase"
- por categoria: onde o Seed é fraco → alimenta o próximo ciclo do Teacher
"""
from __future__ import annotations

import math
import re
import time
import unicodedata
from typing import Callable, Dict, List, Optional, Tuple


def normalizar(texto: str) -> str:
    t = unicodedata.normalize("NFD", (texto or "").lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return " ".join(t.split())


def acertou(resposta: str, criterios: Dict) -> bool:
    r = normalizar(resposta)
    if not r:
        return False
    if "regex" in criterios and not re.match(criterios["regex"], r):
        return False
    for grupo in criterios.get("any", []):
        if not any(_contem(r, p) for p in grupo):
            return False
    for proibida in criterios.get("none", []):
        if _contem(r, proibida):
            return False
    if "so_numeros" in criterios:
        # todos os números da resposta precisam estar entre os permitidos
        # ("10 menos 12 é 12" NÃO acerta "qual é maior: 9 ou 12?")
        permitidos = {str(n) for n in criterios["so_numeros"]}
        if any(n not in permitidos for n in re.findall(r"\d+", r)):
            return False
    return True


def _contem(resposta_norm: str, termo: str) -> bool:
    """Termo precisa começar numa fronteira de palavra ("ia" não casa com "programação";
    prefixos como "descans" continuam valendo para "descansa")."""
    inteira = termo.startswith("=")          # "=class": só a palavra inteira (não "classe")
    t = normalizar(termo.lstrip("="))
    if not t:
        return False
    padrao = (r"(?<![\w])" if t[0].isalnum() else "") + re.escape(t)
    if inteira:
        padrao += r"(?![\w])"
    if t[-1].isdigit():
        padrao += r"(?!\d)"          # "100" não pode casar com "1000"
    return re.search(padrao, resposta_norm) is not None


def avaliar(
    itens: List[Dict],
    gerar: Callable[[str, float, int], str],
    perda: Optional[Callable[[str, str], Optional[Tuple[float, int]]]] = None,
    amostras: int = 3,
    log: Callable[[str], None] = lambda *_: None,
) -> Dict:
    """gerar(pergunta, temperatura, top_k) -> resposta; perda(pergunta, ref) -> (soma, n)."""
    t0 = time.time()
    detalhes = []
    soma_perda, n_tok, n_car = 0.0, 0, 0
    for it in itens:
        q, crit = it["pergunta"], it["criterios"]
        greedy = gerar(q, 1.0, 1)
        ok_greedy = acertou(greedy, crit)
        amostradas = [gerar(q, 0.7, 40) for _ in range(amostras)]
        oks = [acertou(a, crit) for a in amostradas]
        p = sum(oks) / len(oks) if oks else 0.0
        d = {"id": it["id"], "categoria": it["categoria"], "tipo": it["tipo"],
             "pergunta": q, "resposta": greedy, "acerto": ok_greedy,
             "acerto_amostrado": p, "consistencia": max(p, 1 - p)}
        if perda is not None:
            r = perda(q, it["referencia"])
            if r:
                soma_perda += r[0]
                n_tok += r[1]
                n_car += len(it["referencia"]) + 1
                d["perda"] = r[0] / max(1, r[1])
        detalhes.append(d)

    def media(chave, filtro=lambda d: True):
        xs = [float(d[chave]) for d in detalhes if filtro(d)]
        return round(sum(xs) / len(xs), 4) if xs else None

    por_cat: Dict[str, Dict] = {}
    for cat in sorted({d["categoria"] for d in detalhes}):
        por_cat[cat] = {"acerto": media("acerto", lambda d, c=cat: d["categoria"] == c),
                        "itens": sum(1 for d in detalhes if d["categoria"] == cat)}
    perda_media = soma_perda / n_tok if n_tok else None
    return {
        "itens": len(detalhes),
        "perda": None if perda_media is None else round(perda_media, 4),
        "perplexidade": None if perda_media is None else round(math.exp(perda_media), 3),
        # perda por CARACTERE: comparável entre tokenizers diferentes (bytes × BPE)
        "perda_por_caractere": round(soma_perda / n_car, 4) if n_car else None,
        "acerto": media("acerto"),
        "acerto_amostrado": media("acerto_amostrado"),
        "consistencia": media("consistencia"),
        "seguir_instrucoes": media("acerto", lambda d: d["tipo"] == "instrucao"),
        "generalizacao": media("acerto", lambda d: d["tipo"] == "generalizacao"),
        "parafrase": media("acerto", lambda d: d["tipo"] == "paráfrase"),
        "por_categoria": por_cat,
        "segundos": round(time.time() - t0, 1),
        "detalhes": detalhes,
    }


def avaliar_seed(chat, itens: List[Dict], amostras: int = 3) -> Dict:
    """Atalho para um MiniGPTChat carregado."""
    def gerar(q, temp, top_k):
        return chat.responder([{"role": "user", "content": q}], temperatura=temp,
                              top_k=top_k, max_novos=160)
    return avaliar(itens, gerar, perda=chat.perda_resposta, amostras=amostras)
