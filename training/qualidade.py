"""
Validação/limpeza dos exemplos gerados (Teacher ou programáticos) antes do treino.

O Teacher (Qwen 0.5B) erra, mistura idiomas e às vezes diz que é outro modelo.
Aqui só passa o que é objetivamente aceitável. Conteúdo factual que não dá para
verificar automaticamente NÃO é aprovado sozinho: fica pendente de revisão humana.

qualidade (0–1) = média de checagens objetivas; aprovado automaticamente se
todas as checagens obrigatórias passam e qualidade ≥ LIMIAR.
"""
from __future__ import annotations

import ast
import re
from typing import Dict, List, Optional, Set, Tuple

from heilo.training.evaluate import normalizar

LIMIAR = 0.75
MAX_RESPOSTA = 220          # o HEILO Seed tem janela de 256 bytes
MAX_PERGUNTA = 140

_CJK = re.compile(r"[぀-ヿ㐀-鿿가-힯]")
_OUTRA_IDENTIDADE = re.compile(
    r"\b(qwen|alibaba|openai|chatgpt|gpt-?\d|anthropic|claude|llama|meta ai|"
    r"assistente (de ia |virtual )?(criado|desenvolvido)|modelo de linguagem)\b")
_PT = {"o", "a", "de", "que", "e", "do", "da", "em", "um", "uma", "para", "com", "não",
       "nao", "é", "os", "as", "no", "na", "se", "por", "mais", "você", "voce", "isso",
       "seu", "sua", "como", "pra", "está", "esta", "ser", "tem", "são", "sao", "ao",
       "qualquer", "coisa", "aqui", "tô", "to", "tá", "ta", "eu", "me", "te", "muito", "bem",
       "sim", "já", "ja", "só", "so", "falou", "valeu", "obrigado", "obrigada", "oi", "olá", "ola"}
_EN = {"the", "and", "is", "are", "you", "your", "this", "that", "with", "for", "of",
       "to", "in", "it", "can", "what", "how", "i", "am", "be"}

# Fatos verificáveis (CNC/Fanuc): código → palavras que a resposta PRECISA conter
FATOS_CNC = {
    "g00": ["rapid"], "g0": ["rapid"], "g01": ["linear", "linha reta"], "g1": ["linear", "linha reta"],
    "g02": ["horario"], "g03": ["anti-horario", "anti horario"], "g17": ["xy"],
    "g20": ["polegada"], "g21": ["milimetro"], "g28": ["referencia"],
    "g40": ["cancel"], "g41": ["esquerda"], "g42": ["direita"], "g43": ["comprimento"],
    "g54": ["zero", "coordenada", "origem"], "g80": ["cancel"], "g81": ["fura"],
    "g83": ["fura", "cavaco", "pica"], "g90": ["absolut"], "g91": ["incremental"],
    "m03": ["horario", "liga"], "m05": ["deslig", "para"], "m06": ["troca"],
    "m08": ["refrigera", "fluido"], "m09": ["deslig"], "m30": ["fim", "encerra", "termina"],
}

_ARIT = re.compile(r"(\d+)\s*(x|\*|vezes|mais|\+|menos|-|dividido por|/)\s*(\d+)")


def _tokens(t: str) -> List[str]:
    return re.findall(r"\w+", normalizar(t))


def jaccard(a: str, b: str) -> float:
    sa, sb = set(_tokens(a)), set(_tokens(b))
    return len(sa & sb) / len(sa | sb) if sa and sb else 0.0


_OPS = {"x": "*", "*": "*", "vezes": "*", "mais": "+", "+": "+", "menos": "-", "-": "-",
        "dividido por": "/", "/": "/"}


def conta_da_pergunta(pergunta: str) -> Optional[Tuple[int, str, int]]:
    m = _ARIT.search(normalizar(pergunta))
    if not m:
        return None
    a, op, b = int(m.group(1)), _OPS[m.group(2)], int(m.group(3))
    if op in "*+":          # comutativa: 7x9 e 9x7 são a mesma conta
        a, b = min(a, b), max(a, b)
    return a, op, b


def contaminado(pergunta: str, perguntas_eval: List[str], limiar: float = 0.85) -> bool:
    """True se a pergunta é igual/quase igual a uma pergunta da AVALIAÇÃO
    (inclui a MESMA conta escrita de outro jeito: "7x9" = "quanto é 7 vezes 9")."""
    n = normalizar(pergunta)
    conta = conta_da_pergunta(pergunta)
    for e in perguntas_eval:
        if n == normalizar(e) or jaccard(pergunta, e) >= limiar:
            return True
        if conta is not None and conta == conta_da_pergunta(e):
            return True
    return False


def parece_portugues(texto: str) -> bool:
    toks = _tokens(texto)
    if len(toks) < 4:
        return True
    pt = sum(t in _PT for t in toks)
    en = sum(t in _EN for t in toks)
    return pt >= en


def conta_certa(pergunta: str, resposta: str) -> Optional[bool]:
    """Se a pergunta é uma conta, confere o resultado. None = não é conta."""
    m = _ARIT.search(normalizar(pergunta))
    if not m:
        return None
    a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
    if op in ("x", "*", "vezes"):
        v = a * b
    elif op in ("mais", "+"):
        v = a + b
    elif op in ("menos", "-"):
        v = a - b
    else:
        if b == 0:
            return None
        if a % b:
            return None
        v = a // b
    return str(v) in re.findall(r"-?\d+", resposta.replace(".", ""))


def fato_cnc(pergunta: str, resposta: str) -> Optional[bool]:
    """Se a pergunta é sobre um código G/M conhecido, confere a resposta. None = não se aplica."""
    q = normalizar(pergunta)
    if "sim ou nao" in q or "sim ou não" in q:
        return None      # pergunta de sim/não: a resposta não descreve o código
    codigos = re.findall(r"\b([gm])\s?0?(\d{1,2})\b", q)
    if len(codigos) != 1:
        return None
    letra, num = codigos[0]
    chave = f"{letra}{int(num):02d}" if f"{letra}{int(num):02d}" in FATOS_CNC else f"{letra}{int(num)}"
    esperado = FATOS_CNC.get(chave)
    if not esperado:
        return None
    r = normalizar(resposta)
    if chave == "g02" and "anti" in r:
        return False
    return any(e in r for e in esperado)


_INICIO_CODIGO = re.compile(r"^(def |for |if |while |class |import |from |with |try:|print\(|[a-zA-Z_]\w*\s*=[^=])")


def codigo_python_ok(resposta: str) -> Optional[bool]:
    """Se a resposta traz um bloco de código Python (2+ linhas), ele precisa compilar."""
    linhas = resposta.split("\n")
    ini = next((k for k, l in enumerate(linhas) if _INICIO_CODIGO.match(l)), None)
    if ini is None:
        return None
    bloco = [linhas[ini]]
    for l in linhas[ini + 1:]:   # só o trecho contínuo de código (para na 1ª frase em prosa)
        if l.startswith((" ", "\t")) or _INICIO_CODIGO.match(l) or l.startswith(("else", "elif", "except", "finally")):
            bloco.append(l)
        else:
            break
    if len(bloco) < 2:
        return None
    try:
        ast.parse("\n".join(bloco))
        return True
    except SyntaxError:
        return False


def avaliar_exemplo(pergunta: str, resposta: str, categoria: str,
                    perguntas_eval: List[str], conhecidos: Set[str],
                    factual: bool, verificar_fatos: bool = True) -> Dict:
    """Retorna {"aprovado", "qualidade", "problemas", "status"}.

    status: aprovado | rejeitado | revisao (válido, mas factual e não verificável)
    """
    problemas: List[str] = []
    obrig = []  # checagens obrigatórias (bool)

    def check(ok: bool, msg: str, obrigatoria: bool = True):
        if not ok:
            problemas.append(msg)
        if obrigatoria:
            obrig.append(ok)
        return ok

    p, r = (pergunta or "").strip(), (resposta or "").strip()
    check(bool(p) and bool(r), "vazio")
    check(len(p) <= MAX_PERGUNTA, "pergunta longa demais")
    check(len(r) <= MAX_RESPOSTA, f"resposta com mais de {MAX_RESPOSTA} caracteres")
    check(len(r) >= 2, "resposta curta demais")
    check(not _CJK.search(p + r), "caracteres em chinês/japonês/coreano")
    prosa = "\n".join(l for l in r.split("\n") if not _INICIO_CODIGO.match(l) and not l.startswith(" "))
    check(parece_portugues(prosa), "resposta não parece português")
    if verificar_fatos:   # só saídas de modelo; dados escritos à mão podem citar o Qwen como professor
        check(not _OUTRA_IDENTIDADE.search(normalizar(r)), "resposta assume outra identidade/modelo")
    check("<|" not in r and "�" not in r, "artefato de template/codificação")
    check(normalizar(p) != normalizar(r), "resposta repete a pergunta")
    palavras = normalizar(r).split()
    check(not (len(palavras) >= 12 and len(set(palavras)) / len(palavras) < 0.35),
          "resposta repetitiva")
    check(not contaminado(p, perguntas_eval), "pergunta igual/quase igual à AVALIAÇÃO")
    check(normalizar(p) not in conhecidos, "pergunta duplicada")

    verificado = None
    for f in ((conta_certa, fato_cnc) if verificar_fatos else ()):
        v = f(p, r)
        if v is not None:
            verificado = v
            check(v, "fato/conta errado")
    cod = codigo_python_ok(r)
    if cod is not None:
        check(cod, "código Python não compila")

    # checagens de estilo (não obrigatórias, só afetam a nota)
    total_estilo = 3
    estilo = 0
    estilo += len(r) <= 160
    estilo += r[:1].isupper() or r[:1].isdigit()
    estilo += r.endswith((".", "!", "?", ")", "\"")) or "\n" in r
    qualidade = round(0.7 * (sum(obrig) / len(obrig)) + 0.3 * (estilo / total_estilo), 3)

    if not all(obrig):
        status = "rejeitado"
    elif factual and verificado is None:
        status = "revisao"          # válido, mas não dá para confirmar o fato sozinho
    elif qualidade >= LIMIAR:
        status = "aprovado"
    else:
        status = "revisao"
    return {"aprovado": status == "aprovado", "status": status,
            "qualidade": qualidade, "problemas": problemas,
            "verificado": verificado}
