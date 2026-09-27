"""
HEILO Dia a Dia — ferramentas determinísticas para o uso diário.

Um modelo pequeno não sabe a hora, erra contas e não conhece as suas anotações.
Estas ferramentas respondem com EXATIDÃO, rodando só no seu PC:

- data e hora, dia da semana, quantos dias faltam
- calculadora (contas, porcentagem, raiz, potência)
- conversões de unidades (comprimento, massa, volume, tempo, temperatura)
- lembretes e timer ("me lembra de ... às 15h", "timer de 10 minutos")
- anotações ("anota: ...", "o que eu anotei sobre ...")
- listas ("coloca leite na lista de compras", "mostra a lista de compras")

Os dados pessoais ficam em memory/pessoal/ (fora do Git: nunca vão para o GitHub).
Se nenhuma ferramenta reconhecer a mensagem, devolve None e quem responde é o HEILO Seed.
"""
from __future__ import annotations

import ast
import json
import math
import operator
import re
import unicodedata
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional

DIAS = ["segunda-feira", "terça-feira", "quarta-feira", "quinta-feira", "sexta-feira", "sábado", "domingo"]
MESES = ["janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho", "agosto",
         "setembro", "outubro", "novembro", "dezembro"]


def _sem_acento(t: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", _sem_acento(t.lower())).strip(" ?!.")


def _num(s: str) -> float:
    """'2.380' / '2,5' / '1.234,5' → float (formato brasileiro)."""
    s = s.strip()
    if "," in s:
        s = s.replace(".", "").replace(",", ".")
    elif re.fullmatch(r"\d{1,3}(\.\d{3})+", s):
        s = s.replace(".", "")
    return float(s)


def _fmt(x: float, casas: int = 4) -> str:
    if abs(x - round(x)) < 1e-9 and abs(x) < 1e15:
        return f"{int(round(x)):,}".replace(",", ".")
    txt = f"{x:,.{casas}f}".rstrip("0").rstrip(".")
    return txt.replace(",", "X").replace(".", ",").replace("X", ".")


def _data_br(d: date) -> str:
    return f"{DIAS[d.weekday()]}, {d.day} de {MESES[d.month - 1]} de {d.year}"


# ------------------------------------------------------------ calculadora
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv,
        ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUNCS = {"raiz": math.sqrt, "sqrt": math.sqrt, "sen": lambda x: math.sin(math.radians(x)),
          "cos": lambda x: math.cos(math.radians(x)), "tan": lambda x: math.tan(math.radians(x)),
          "log": math.log10, "ln": math.log, "abs": abs}


def _avaliar(no):
    if isinstance(no, ast.Expression):
        return _avaliar(no.body)
    if isinstance(no, ast.Constant) and isinstance(no.value, (int, float)):
        return no.value
    if isinstance(no, ast.Name) and no.id == "pi":
        return math.pi
    if isinstance(no, ast.BinOp) and type(no.op) in _OPS:
        a, b = _avaliar(no.left), _avaliar(no.right)
        if isinstance(no.op, ast.Pow) and abs(b) > 100:
            raise ValueError("potência grande demais")
        return _OPS[type(no.op)](a, b)
    if isinstance(no, ast.UnaryOp) and type(no.op) in _OPS:
        return _OPS[type(no.op)](_avaliar(no.operand))
    if isinstance(no, ast.Call) and isinstance(no.func, ast.Name) and no.func.id in _FUNCS and len(no.args) == 1:
        return _FUNCS[no.func.id](_avaliar(no.args[0]))
    raise ValueError("expressão não suportada")


def calcular(texto: str) -> Optional[str]:
    t = _norm(texto)
    t = re.sub(r"^(quanto (e|da|fica|vale)|calcula|calcule|conta|resultado de|qual (e )?o resultado de)\s*:?\s*", "", t)
    # porcentagem: "17% de 2380"
    m = re.fullmatch(r"([\d.,]+)\s*(%|por ?cento) de ([\d.,]+)", t)
    if m:
        p, v = _num(m.group(1)), _num(m.group(3))
        return f"{_fmt(p)}% de {_fmt(v)} é **{_fmt(p * v / 100)}**."
    m = re.fullmatch(r"(?:a )?raiz (?:quadrada )?de ([\d.,]+)", t)
    if m:
        v = _num(m.group(1))
        return f"A raiz quadrada de {_fmt(v)} é **{_fmt(math.sqrt(v))}**."
    m = re.fullmatch(r"([\d.,]+) (?:elevado|elevada) (?:a|ao|na) ([\d.,]+)", t)
    if m:
        a, b = _num(m.group(1)), _num(m.group(2))
        if b <= 100:
            return f"{_fmt(a)} elevado a {_fmt(b)} é **{_fmt(a ** b)}**."
    if not re.search(r"\d", t) or not re.search(r"\d\s*([-+*/x×÷^]|mais|menos|vezes|dividido)\s*\(?-?\d|raiz|sqrt|\bsen\b|\bcos\b", t):
        return None
    expr = t
    for a, b in ((r"\bmais\b", "+"), (r"\bmenos\b", "-"), (r"\bvezes\b", "*"), (r"\bdividido por\b", "/"),
                 (r"(?<=\d)\s*[x×]\s*(?=[\d(])", "*"), ("÷", "/"), (r"\^", "**"), (r"\braiz de\b", "raiz")):
        expr = re.sub(a, b, expr)
    expr = re.sub(r"(\d)\.(\d{3})(?!\d)", r"\1\2", expr)      # 2.380 → 2380
    expr = re.sub(r"(\d),(\d)", r"\1.\2", expr)               # 2,5 → 2.5
    expr = re.sub(r"raiz\s*(\d+(?:\.\d+)?)", r"raiz(\1)", expr)
    expr = expr.replace("=", "").strip()
    if not re.fullmatch(r"[\d\s.+\-*/%()a-z]+", expr) or len(expr) > 120:
        return None
    try:
        v = _avaliar(ast.parse(expr, mode="eval"))
    except (SyntaxError, ValueError, ZeroDivisionError, TypeError, OverflowError):
        if re.search(r"/\s*0(?!\.\d)", expr):
            return "Não dá para dividir por zero."
        return None
    return f"**{_fmt(v)}**"


# ------------------------------------------------------------ conversões
_UNID = {  # nome → (grandeza, fator para a unidade base)
    "mm": ("comp", 0.001), "milimetro": ("comp", 0.001), "milimetros": ("comp", 0.001),
    "cm": ("comp", 0.01), "centimetro": ("comp", 0.01), "centimetros": ("comp", 0.01),
    "m": ("comp", 1.0), "metro": ("comp", 1.0), "metros": ("comp", 1.0),
    "km": ("comp", 1000.0), "quilometro": ("comp", 1000.0), "quilometros": ("comp", 1000.0),
    "pol": ("comp", 0.0254), "polegada": ("comp", 0.0254), "polegadas": ("comp", 0.0254), "in": ("comp", 0.0254),
    "pe": ("comp", 0.3048), "pes": ("comp", 0.3048), "ft": ("comp", 0.3048),
    "milha": ("comp", 1609.344), "milhas": ("comp", 1609.344),
    "g": ("massa", 1.0), "grama": ("massa", 1.0), "gramas": ("massa", 1.0),
    "kg": ("massa", 1000.0), "quilo": ("massa", 1000.0), "quilos": ("massa", 1000.0),
    "t": ("massa", 1e6), "tonelada": ("massa", 1e6), "toneladas": ("massa", 1e6),
    "lb": ("massa", 453.59237), "libra": ("massa", 453.59237), "libras": ("massa", 453.59237),
    "ml": ("vol", 0.001), "mililitro": ("vol", 0.001), "mililitros": ("vol", 0.001),
    "l": ("vol", 1.0), "litro": ("vol", 1.0), "litros": ("vol", 1.0),
    "galao": ("vol", 3.785411784), "galoes": ("vol", 3.785411784),
    "s": ("tempo", 1.0), "seg": ("tempo", 1.0), "segundo": ("tempo", 1.0), "segundos": ("tempo", 1.0),
    "min": ("tempo", 60.0), "minuto": ("tempo", 60.0), "minutos": ("tempo", 60.0),
    "h": ("tempo", 3600.0), "hora": ("tempo", 3600.0), "horas": ("tempo", 3600.0),
    "dia": ("tempo", 86400.0), "dias": ("tempo", 86400.0),
    "semana": ("tempo", 604800.0), "semanas": ("tempo", 604800.0),
    "km/h": ("vel", 1 / 3.6), "m/s": ("vel", 1.0), "mph": ("vel", 0.44704),
}
_TEMP = {"c": "C", "celsius": "C", "graus celsius": "C", "graus": "C", "f": "F", "fahrenheit": "F",
         "graus fahrenheit": "F", "k": "K", "kelvin": "K"}


def converter(texto: str) -> Optional[str]:
    t = _norm(texto).replace("°", " ")
    m = re.search(r"([\d.,]+)\s*([a-z/ ]+?)\s+(?:em|para|pra|p/|to|sao quantos?|equivale a quantos?|da quantos?)\s+"
                  r"(?:quantos?\s+|quantas?\s+)?([a-z/ ]+)$", t)
    if not m:
        m2 = re.search(r"quant[oa]s?\s+([a-z/ ]+?)\s+(?:tem|ha|da|dao|sao)\s+(?:em\s+)?([\d.,]+)\s*([a-z/ ]+)$", t)
        if not m2:
            return None
        para, valor, de = m2.group(1).strip(), m2.group(2), m2.group(3).strip()
    else:
        valor, de, para = m.group(1), m.group(2).strip(), m.group(3).strip()
    try:
        v = _num(valor)
    except ValueError:
        return None
    if de in _TEMP and para in _TEMP:
        a, b = _TEMP[de], _TEMP[para]
        c = {"C": v, "F": (v - 32) * 5 / 9, "K": v - 273.15}[a]
        r = {"C": c, "F": c * 9 / 5 + 32, "K": c + 273.15}[b]
        return f"{_fmt(v, 2)} °{a} = **{_fmt(r, 2)} °{b}**." if b != "K" else f"{_fmt(v, 2)} °{a} = **{_fmt(r, 2)} K**."
    if de not in _UNID or para not in _UNID or _UNID[de][0] != _UNID[para][0]:
        return None
    r = v * _UNID[de][1] / _UNID[para][1]
    return f"{_fmt(v)} {de} = **{_fmt(r)} {para}**."


# ------------------------------------------------------------ datas
def _parse_data(t: str, hoje: date) -> Optional[date]:
    m = re.search(r"(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?", t)
    if m:
        d, mes = int(m.group(1)), int(m.group(2))
        ano = int(m.group(3)) if m.group(3) else hoje.year
        ano = ano + 2000 if ano < 100 else ano
        try:
            alvo = date(ano, mes, d)
        except ValueError:
            return None
        if not m.group(3) and alvo < hoje:
            alvo = alvo.replace(year=ano + 1)
        return alvo
    m = re.search(r"(\d{1,2}) de (" + "|".join(_sem_acento(x) for x in MESES) + r")(?: de (\d{4}))?", t)
    if m:
        mes = [_sem_acento(x) for x in MESES].index(m.group(2)) + 1
        ano = int(m.group(3)) if m.group(3) else hoje.year
        try:
            alvo = date(ano, mes, int(m.group(1)))
        except ValueError:
            return None
        if not m.group(3) and alvo < hoje:
            alvo = alvo.replace(year=ano + 1)
        return alvo
    fixas = {"natal": (12, 25), "ano novo": (1, 1), "reveillon": (12, 31), "dia das criancas": (10, 12)}
    for nome, (mes, d) in fixas.items():
        if nome in t:
            alvo = date(hoje.year, mes, d)
            return alvo if alvo >= hoje else alvo.replace(year=hoje.year + 1)
    return None


def datas(texto: str, agora: datetime) -> Optional[str]:
    t = _norm(texto)
    hoje = agora.date()
    if re.search(r"\b(que horas? (sao|e)|me (fala|diz) a hora|horas agora|que hora e agora)\b", t):
        return f"Agora são **{agora:%H:%M}**."
    if re.search(r"\b(que dia e hoje|qual (e )?a data (de )?hoje|data de hoje|hoje e que dia|que dia (da semana )?e hoje)\b", t):
        return f"Hoje é **{_data_br(hoje)}**."
    if re.search(r"\b(amanha e que dia|que dia (e|sera) amanha)\b", t):
        return f"Amanhã é **{_data_br(hoje + timedelta(days=1))}**."
    if re.search(r"\b(ontem foi que dia|que dia foi ontem)\b", t):
        return f"Ontem foi **{_data_br(hoje - timedelta(days=1))}**."
    if re.search(r"\b(quantos dias faltam|falta quantos dias|faltam quantos dias)\b", t):
        alvo = _parse_data(t, hoje)
        if alvo:
            n = (alvo - hoje).days
            return ("É hoje!" if n == 0 else
                    f"Faltam **{n} dia{'s' if n != 1 else ''}** para {_data_br(alvo)}.")
    if re.search(r"\b(que dia da semana|cai (em )?que dia|dia da semana)\b", t):
        alvo = _parse_data(t, hoje)
        if alvo:
            return f"{alvo:%d/%m/%Y} cai numa **{DIAS[alvo.weekday()]}**."
    m = re.search(r"\b(?:que dia (?:vai ser|sera|e) )?daqui a (\d+) (dias?|semanas?)\b", t)
    if m and ("que dia" in t or "data" in t):
        n = int(m.group(1)) * (7 if m.group(2).startswith("semana") else 1)
        return f"Daqui a {m.group(1)} {m.group(2)} será **{_data_br(hoje + timedelta(days=n))}**."
    return None


# ------------------------------------------------------------ armazenamento pessoal
class Pessoal:
    """Lembretes, notas e listas em memory/pessoal/ (local, fora do Git)."""

    def __init__(self, pasta: Path):
        self.pasta = Path(pasta)

    def _arq(self, nome):
        return self.pasta / f"{nome}.json"

    def ler(self, nome) -> list:
        try:
            return json.loads(self._arq(nome).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []

    def gravar(self, nome, dados) -> None:
        self.pasta.mkdir(parents=True, exist_ok=True)
        tmp = self._arq(nome).with_suffix(".tmp")
        tmp.write_text(json.dumps(dados, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self._arq(nome))


def _hora_alvo(t: str, agora: datetime) -> Optional[datetime]:
    m = re.search(r"\bem (\d+) ?(minutos?|min|horas?|h)\b|\bdaqui a (\d+) ?(minutos?|min|horas?|h)\b", t)
    if m:
        n, u = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
        return agora + (timedelta(hours=int(n)) if u.startswith("h") else timedelta(minutes=int(n)))
    m = re.search(r"\b(?:as|a|para as|pras)\s*(\d{1,2})(?:[:h](\d{2}))?\s*(?:h|horas?)?\b", t)
    if m:
        h, mi = int(m.group(1)), int(m.group(2) or 0)
        if h > 23 or mi > 59:
            return None
        alvo = agora.replace(hour=h, minute=mi, second=0, microsecond=0)
        if "amanha" in t:
            alvo += timedelta(days=1)
        elif alvo <= agora:
            alvo += timedelta(days=1)
        return alvo
    return None


def lembretes(texto: str, agora: datetime, p: Pessoal) -> Optional[str]:
    t = _norm(texto)
    m = re.search(r"\b(?:timer|temporizador|cronometro)\s+(?:de\s+)?(\d+)\s*(minutos?|min|segundos?|seg|s|horas?|h)\b", t)
    if m:
        n, u = int(m.group(1)), m.group(2)
        delta = timedelta(hours=n) if u.startswith("h") else (timedelta(seconds=n) if u.startswith("s") else timedelta(minutes=n))
        return _novo_lembrete(p, agora + delta, f"Timer de {n} {u} terminou!", agora)
    if re.search(r"\b(meus lembretes|quais (sao )?(os )?(meus )?lembretes|lista de lembretes|mostra (os )?lembretes)\b", t):
        ls = sorted([l for l in p.ler("lembretes") if not l.get("feito")], key=lambda l: l["quando"])
        if not ls:
            return "Você não tem lembretes pendentes."
        return "Seus lembretes:\n" + "\n".join(
            f"- {datetime.fromisoformat(l['quando']):%d/%m %H:%M} — {l['texto']}" for l in ls)
    if re.search(r"\b(apaga|cancela|remove|limpa) (todos )?(os )?(meus )?lembretes\b", t):
        p.gravar("lembretes", [l for l in p.ler("lembretes") if l.get("feito")])
        return "Pronto, apaguei os lembretes pendentes."
    m = re.search(r"^(?:heilo,? )?(?:por favor,? )?(?:me (?:lembra|lembre|avisa|avise)|lembre-me|avise-me|"
                  r"cria(?:r)? um lembrete|lembrete)\b\s*(.*)", t)
    if m:
        quando = _hora_alvo(t, agora)
        if quando is None:
            return ("Posso lembrar, sim! Me diz quando, por exemplo: "
                    "\"me lembra de tomar água às 15h\" ou \"me lembra em 20 minutos de tirar o bolo\".")
        # texto do lembrete: frase original sem a parte do horário
        orig = re.sub(r"(?i)^.*?\b(lembra|lembre|avisa|avise|lembrete)(-me)?\s*(de|que|para|pra|:)?\s+", "", texto.strip(), count=1)
        orig = re.sub(r"(?i)\s*\b(amanhã|amanha)\b", "", orig)
        orig = re.sub(r"(?i)\s*\b(às|as|a|para as|pras|em|daqui a)\s*\d{1,2}([:h]\d{2})?\s*(h|horas?|minutos?|min)?\b", "", orig)
        orig = re.sub(r"(?i)^(de|que|para|pra)\s+", "", orig.strip(" ,.!?")) or "lembrete"
        return _novo_lembrete(p, quando, orig, agora)
    return None


def _novo_lembrete(p: Pessoal, quando: datetime, texto: str, agora: datetime) -> str:
    ls = p.ler("lembretes")
    ls.append({"id": uuid.uuid4().hex[:8], "quando": quando.isoformat(timespec="seconds"),
               "texto": texto, "criado": agora.isoformat(timespec="seconds"), "feito": False})
    p.gravar("lembretes", ls)
    dia = "hoje" if quando.date() == agora.date() else ("amanhã" if quando.date() == agora.date() + timedelta(days=1)
                                                        else f"{quando:%d/%m}")
    return f"Combinado! Vou te lembrar **{dia} às {quando:%H:%M}**: {texto}"


def lembretes_vencidos(p: Pessoal, agora: datetime) -> List[Dict]:
    ls = p.ler("lembretes")
    vencidos = [l for l in ls if not l.get("feito") and datetime.fromisoformat(l["quando"]) <= agora]
    if vencidos:
        for l in ls:
            if l in vencidos:
                l["feito"] = True
        p.gravar("lembretes", ls)
    return vencidos


def notas(texto: str, agora: datetime, p: Pessoal) -> Optional[str]:
    t = _norm(texto)
    m = re.match(r"(?i)^\s*(?:heilo,?\s+)?(anota|anote|anotar|salva a nota|guarda isso:|nova nota:)\s*(isso|aí|ai)?\s*[:\-,]?\s+(.+)$",
                 texto.strip())
    if m and len(m.group(3).strip()) > 1:
        ns = p.ler("notas")
        ns.append({"id": uuid.uuid4().hex[:8], "texto": m.group(3).strip(), "quando": agora.isoformat(timespec="seconds")})
        p.gravar("notas", ns)
        return "Anotado! ✎"
    if re.search(r"\b(minhas notas|minhas anotacoes|mostra (as )?(minhas )?(notas|anotacoes)|o que eu anotei)$", t):
        ns = p.ler("notas")[-15:]
        if not ns:
            return "Você ainda não tem anotações. É só dizer \"anota: ...\"."
        return "Suas anotações mais recentes:\n" + "\n".join(
            f"- {datetime.fromisoformat(n['quando']):%d/%m} — {n['texto']}" for n in reversed(ns))
    m = re.search(r"\b(?:o que (?:eu )?anotei|minhas notas|anotacoes?) (?:sobre|de|do|da)\s+(.+)$", t)
    if m:
        termos = [w for w in m.group(1).split() if len(w) > 2] or m.group(1).split()
        achadas = [n for n in p.ler("notas") if all(w in _norm(n["texto"]) for w in termos)]
        if not achadas:
            return f"Não encontrei nenhuma anotação sobre \"{m.group(1)}\"."
        return "Encontrei:\n" + "\n".join(
            f"- {datetime.fromisoformat(n['quando']):%d/%m} — {n['texto']}" for n in achadas[-10:])
    return None


def _itens_originais(texto: str, verbo: str, ate: str) -> List[str]:
    """Itens como o usuário escreveu (com acentos)."""
    m = re.search(verbo + r"\s+(.+?)\s+" + ate, texto.lower())
    return [i.strip(" .!?") for i in re.split(r",|\s+e\s+", m.group(1)) if i.strip(" .!?")] if m else []


def listas(texto: str, p: Pessoal) -> Optional[str]:
    t = _norm(texto)
    m = re.search(r"\b(?:coloca|coloque|adiciona|adicione|poe|bota|inclui|inclua|anota)\s+(.+?)\s+na lista(?: de| do| da)?\s*([a-z ]*)$", t)
    if m:
        nome = (m.group(2).strip() or "compras")
        itens = _itens_originais(texto, r"(?:coloca|coloque|adiciona|adicione|p[oõ]e|bota|inclui|inclua|anota)",
                                 "na lista") or [i.strip() for i in re.split(r",| e ", m.group(1)) if i.strip()]
        todas = {l["nome"]: l for l in p.ler("listas")}
        lst = todas.setdefault(nome, {"nome": nome, "itens": []})
        existentes = {_norm(i) for i in lst["itens"]}
        novos = [i for i in itens if _norm(i) not in existentes]
        lst["itens"] += novos
        p.gravar("listas", list(todas.values()))
        return f"Coloquei na lista de {nome}: {', '.join(itens)}. ({len(lst['itens'])} item(ns) no total)"
    m = re.search(r"\b(?:tira|tire|remove|remova|apaga|risca)\s+(.+?)\s+da lista(?: de| do| da)?\s*([a-z ]*)$", t)
    if m:
        nome = m.group(2).strip() or "compras"
        todas = {l["nome"]: l for l in p.ler("listas")}
        if nome not in todas:
            return f"Não achei a lista de {nome}."
        itens = _itens_originais(texto, r"(?:tira|tire|remove|remova|apaga|risca)", "da lista") or \
            [i.strip() for i in re.split(r",| e ", m.group(1)) if i.strip()]
        antes = len(todas[nome]["itens"])
        fora = {_norm(i) for i in itens}
        todas[nome]["itens"] = [i for i in todas[nome]["itens"] if _norm(i) not in fora]
        p.gravar("listas", list(todas.values()))
        return (f"Tirei da lista de {nome}: {', '.join(itens)}." if len(todas[nome]["itens"]) < antes
                else f"Não achei {', '.join(itens)} na lista de {nome}.")
    m = re.search(r"\b(?:limpa|apaga|zera)\s+a lista(?: de| do| da)?\s*([a-z ]*)$", t)
    if m:
        nome = m.group(1).strip() or "compras"
        todas = {l["nome"]: l for l in p.ler("listas")}
        todas.pop(nome, None)
        p.gravar("listas", list(todas.values()))
        return f"Lista de {nome} apagada."
    m = re.search(r"\b(?:mostra|mostre|ver|qual (?:e )?|o que tem na|como esta)\s*a? ?lista(?: de| do| da)?\s*([a-z ]*)$", t)
    if m:
        nome = m.group(1).strip() or "compras"
        todas = {l["nome"]: l for l in p.ler("listas")}
        if nome not in todas or not todas[nome]["itens"]:
            return f"A lista de {nome} está vazia."
        return f"Lista de {nome}:\n" + "\n".join(f"- {i}" for i in todas[nome]["itens"])
    return None


# ------------------------------------------------------------ roteador
class DiaADia:
    def __init__(self, pasta: Path, relogio: Callable[[], datetime] = datetime.now):
        self.p = Pessoal(pasta)
        self.relogio = relogio

    def responder(self, mensagem: str) -> Optional[Dict]:
        if not mensagem or mensagem.strip().startswith("/") or len(mensagem) > 300:
            return None
        agora = self.relogio()
        for nome, fn in (("datas", lambda m: datas(m, agora)),
                         ("lembretes", lambda m: lembretes(m, agora, self.p)),
                         ("notas", lambda m: notas(m, agora, self.p)),
                         ("listas", lambda m: listas(m, self.p)),
                         ("conversao", converter),
                         ("calculadora", calcular)):
            try:
                r = fn(mensagem)
            except Exception:  # uma ferramenta nunca derruba a conversa
                r = None
            if r:
                return {"type": "message", "content": r, "agent": "HEILO",
                        "source": f"ferramenta:{nome}", "model": "ferramenta"}
        return None

    def vencidos(self) -> List[Dict]:
        return lembretes_vencidos(self.p, self.relogio())
