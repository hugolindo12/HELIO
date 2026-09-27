"""
Validação de exemplos antes de entrarem no dataset HEILO.

Nada é tratado como verdade automaticamente — nem o que o Teacher gera.
A validação automática só pega problemas objetivos; conteúdo factual continua
precisando de revisão humana (/revisar).
"""
from __future__ import annotations

import re
from typing import Dict, List, Set, Tuple

MAX_CHARS = 1500
MIN_RESPOSTA = 2
# restos de template de chat / tokens especiais de modelos externos
_ARTEFATOS = re.compile(r"<\|[^|>]{1,40}\|>|\[/?INST\]|<s>|</s>|<think>|</think>")


def validate(example: Dict, known_ids: Set[str] = frozenset()) -> Tuple[bool, List[str]]:
    """Retorna (ok, lista_de_problemas)."""
    erros: List[str] = []
    msgs = example.get("messages")
    if not isinstance(msgs, list) or len(msgs) < 2:
        return False, ["estrutura: precisa de pelo menos 1 pergunta e 1 resposta"]

    for i, m in enumerate(msgs):
        if not isinstance(m, dict) or m.get("role") not in ("user", "assistant"):
            erros.append(f"estrutura: mensagem {i} com papel inválido")
            continue
        esperado = "user" if i % 2 == 0 else "assistant"
        if m["role"] != esperado:
            erros.append(f"estrutura: mensagem {i} deveria ser '{esperado}'")
        texto = m.get("content")
        if not isinstance(texto, str) or not texto.strip():
            erros.append(f"vazio: mensagem {i} sem conteúdo")
            continue
        if len(texto) > MAX_CHARS:
            erros.append(f"tamanho: mensagem {i} com mais de {MAX_CHARS} caracteres")
        if "�" in texto:
            erros.append(f"codificação: mensagem {i} com caractere inválido (�)")
        if _ARTEFATOS.search(texto):
            erros.append(f"artefato: mensagem {i} contém token de template de modelo")

    if msgs and msgs[-1].get("role") != "assistant":
        erros.append("estrutura: a última mensagem precisa ser da HEILO (assistant)")

    resp = (msgs[-1].get("content") or "").strip() if msgs else ""
    perg = (msgs[-2].get("content") or "").strip() if len(msgs) >= 2 else ""
    if resp and len(resp) < MIN_RESPOSTA:
        erros.append("resposta curta demais")
    if resp and perg and resp.lower() == perg.lower():
        erros.append("resposta apenas repete a pergunta")
    palavras = resp.lower().split()
    if len(palavras) >= 12 and len(set(palavras)) / len(palavras) < 0.3:
        erros.append("resposta muito repetitiva (provável degeneração do modelo)")

    if example.get("id") in known_ids:
        erros.append("duplicado: exemplo idêntico já existe")

    return (not erros), erros
