"""
Linhas de modelos da HEILO — do menor/mais rápido para o maior/mais capaz.

  HEILO Faísca  — pequeno e rápido (até ~80 M parâmetros). Roda em qualquer PC.
  HEILO Aurora  — médio (~80–250 M). Mais capaz, ainda leve.
  HEILO Zênite  — o maior (250 M ou mais): o sol no ponto mais alto.

"Seed" continua sendo o nome do motor/projeto de treino (HEILO Seed v2.0 = motor);
a linha diz o TAMANHO. Ex.: "HEILO Faísca 2.0". Modelos de código levam "Code":
"HEILO Faísca Code 0.2".
"""
from __future__ import annotations

from typing import Optional

LINHAS = [
    ("Faísca", 80_000_000, "pequeno e rápido: roda em qualquer PC"),
    ("Aurora", 250_000_000, "médio: mais capaz, ainda leve"),
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
