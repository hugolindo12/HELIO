"""
Memória verificada da HEILO: antes de "chutar", a HEILO procura a pergunta entre os
exemplos APROVADOS (revisados por uma pessoa) e, se achar uma pergunta equivalente,
responde com a resposta verificada.

- Não usa nenhuma IA externa: busca por palavras com peso (as raras pesam mais).
- É conservadora: se a pergunta tem uma palavra importante que o exemplo não tem
  ("capital da ITÁLIA" × "capital da ALEMANHA"), NÃO usa. Na dúvida, quem responde é
  o HEILO Seed.
- Tudo que você ensina (/ensinar, 👍, Corrigir, revisão) entra aqui na hora.
"""
from __future__ import annotations

import math
import re
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PARADAS = set("""
a o as os um uma uns umas de do da dos das no na nos nas em por para pra pro com sem e ou que
qual quais quanto quantos quantas como onde quando porque por que se me te voce vc eu tu ele ela
isso isto esse essa este esta aquele aquela e eh sao ser estar esta estou tem ter ha
fala diz explica explique me diga sobre coisa algo pode poderia consegue sabe favor
heilo oi ola faz fazer serve servir significa funciona usa usar uso pra
""".split())


_FORMATO = re.compile(r"\b(responda|uma palavra|so o codigo|so com|apenas|diga so|diga apenas)\b")


def _sem_acento(t: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")


def _radical(p: str) -> str:
    for suf in ("mente", "coes", "cao", "oes", "aes", "ais", "eis", "es", "s"):
        if len(p) > len(suf) + 3 and p.endswith(suf):
            return p[: -len(suf)]
    return p


def palavras(texto: str) -> List[str]:
    t = _sem_acento((texto or "").lower())
    toks = re.findall(r"[a-z0-9_]+(?:\(\))?|[#%*+/=<>-]+", t)
    return [_radical(p) for p in toks if p not in PARADAS]


class MemoriaVerificada:
    def __init__(self, pipeline, limiar: float = 0.6, recarregar_seg: float = 20.0):
        self.pipeline = pipeline
        self.limiar = limiar
        self.recarregar_seg = recarregar_seg
        self._carregado_em = 0.0
        self._itens: List[Tuple[Counter, str, str]] = []
        self._idf: Dict[str, float] = {}

    # ------------------------------------------------------------ índice
    def _carregar(self) -> None:
        if self._itens and time.time() - self._carregado_em < self.recarregar_seg:
            return
        itens = []
        for r in self.pipeline.approved():
            msgs = r.get("messages") or []
            if len(msgs) != 2 or msgs[0].get("role") != "user":      # só pergunta → resposta simples
                continue
            q, a = msgs[0]["content"], msgs[1]["content"]
            toks = palavras(q)
            if toks and a.strip():
                itens.append((Counter(toks), q, a))
        df = Counter(t for c, _, _ in itens for t in c)
        n = max(1, len(itens))
        self._idf = {t: math.log((n + 1) / (d + 0.5)) + 1.0 for t, d in df.items()}
        self._itens = itens
        self._carregado_em = time.time()

    def _peso(self, t: str) -> float:
        return self._idf.get(t, math.log(len(self._itens) + 2) + 1.0)   # palavra nunca vista = muito rara

    # ------------------------------------------------------------- busca
    def buscar(self, pergunta: str) -> Optional[Dict]:
        self._carregar()
        q = Counter(palavras(pergunta))
        if not q or not self._itens:
            return None
        melhor, melhor_s = None, 0.0
        pede_formato = bool(_FORMATO.search(_sem_acento(pergunta.lower())))
        for c, perg, resp in self._itens:
            if not pede_formato and _FORMATO.search(_sem_acento(perg.lower())):
                continue      # exemplo de "responda só com…" não serve para pergunta normal
            comum = sum(self._peso(t) for t in q if t in c)
            if not comum:
                continue
            uniao = sum(self._peso(t) for t in set(q) | set(c))
            s = comum / uniao
            if s > melhor_s:
                melhor, melhor_s = (c, perg, resp), s
        if melhor is None or melhor_s < self.limiar:
            return None
        c, perg, resp = melhor
        # palavras importantes de UM lado que faltam no OUTRO → é outra pergunta
        # (o exemplo pode ter uma palavra a mais, ex.: "o que é o g71 NO TORNO?")
        faltando = [t for t in set(q) - set(c) if self._peso(t) >= 3.0]
        if faltando:
            return None
        return {"pergunta": perg, "resposta": resp, "similaridade": round(melhor_s, 3)}

    def responder(self, mensagem: str) -> Optional[Dict]:
        if not mensagem or mensagem.strip().startswith("/") or len(mensagem) > 200:
            return None
        achado = self.buscar(mensagem)
        if not achado:
            return None
        return {"type": "message", "content": achado["resposta"], "agent": "HEILO",
                "source": "memória verificada", "model": "memoria",
                "memoria": {"pergunta": achado["pergunta"], "similaridade": achado["similaridade"]}}
