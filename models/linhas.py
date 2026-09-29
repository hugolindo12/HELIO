"""
Linhas de modelos da HEILO — do menor/mais rápido para o maior/mais capaz.

  HEILO Faísca  — pequeno e rápido (até ~500 M parâmetros; meta: 400 M). Roda em qualquer PC.
  HEILO Aurora  — médio (~500 M a 1 bi; meta: 600 M–1 bi). Mais capaz, ainda roda no PC.
  HEILO Zênite  — o maior (1 bi ou mais; meta: 2 bi): o sol no ponto mais alto.

"Seed" continua sendo o nome do motor/projeto de treino (HEILO Seed v2.0 = motor);
a linha diz o TAMANHO. Ex.: "HEILO Faísca 2.0". Modelos de código levam "Code":
"HEILO Faísca Code 0.2".
"""
from __future__ import annotations

from typing import Optional

LINHAS = [
    ("Faísca", 500_000_000, "pequeno e rápido: roda em qualquer PC"),
    ("Aurora", 1_000_000_000, "médio: mais capaz, ainda roda no PC"),
    ("Zênite", None, "o maior: mais capaz, mais pesado"),
]


def linha(n_params: Optional[int]) -> str:
    if not n_params:
        return "Faísca"
    for nome, limite, _ in LINHAS:
        if limite is None or n_params < limite:
            return nome
    return LINHAS[-1][0]


def nome_modelo(n_params: Optional[int], versao: str = "", codigo: bool = False) -> str:
    import re
    v = re.sub(r"^(v|aurora-|zenite-|faisca-)", "", (versao or ""), flags=re.I)
    return " ".join(p for p in ("HEILO", linha(n_params), "Code" if codigo else "", v) if p)
