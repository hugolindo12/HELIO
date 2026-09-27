"""
Geradores de dados para os ciclos de treino do HEILO Seed.

- programáticos (sem Teacher): contas verificadas, seguir instruções, honestidade,
  identidade — respostas corretas por construção.
- com o HEILO Teacher: perguntas novas (self-instruct) e respostas do Teacher,
  que depois passam pela validação em training/qualidade.py.

Nada daqui usa perguntas do conjunto de AVALIAÇÃO (filtro `contaminado`).
"""
from __future__ import annotations

import random
import re
from typing import Callable, Dict, List, Optional, Tuple

from heilo.training.qualidade import FATOS_CNC, contaminado

Par = Tuple[str, str]

# ---------------------------------------------------------------- programáticos
_FORMAS_MULT = ["{a}x{b}", "quanto é {a} vezes {b}?", "{a} vezes {b}", "quanto dá {a}x{b}?",
                "me diz quanto é {a} vezes {b}"]
_FORMAS_SOMA = ["quanto é {a} mais {b}?", "{a}+{b}", "soma {a} com {b}", "quanto dá {a} mais {b}?"]
_FORMAS_SUB = ["quanto é {a} menos {b}?", "{a}-{b}", "quanto dá {a} menos {b}?"]
_FORMAS_DIV = ["quanto é {a} dividido por {b}?", "{a}/{b}", "quanto dá {a} dividido por {b}?"]


def matematica(n: int, rnd: random.Random, eval_qs: List[str]) -> List[Par]:
    out: List[Par] = []
    tentativas = 0
    while len(out) < n and tentativas < n * 50:
        tentativas += 1
        tipo = rnd.choice(["mult", "mult", "soma", "sub", "div", "instr"])
        a, b = rnd.randint(2, 20), rnd.randint(2, 20)
        if tipo == "mult":
            q, r = rnd.choice(_FORMAS_MULT).format(a=a, b=b), f"{a} vezes {b} é {a*b}."
        elif tipo == "soma":
            a, b = rnd.randint(1, 99), rnd.randint(1, 99)
            q, r = rnd.choice(_FORMAS_SOMA).format(a=a, b=b), f"{a} mais {b} é {a+b}."
        elif tipo == "sub":
            a, b = rnd.randint(10, 99), rnd.randint(1, 60)
            if b > a:
                a, b = b, a
            q, r = rnd.choice(_FORMAS_SUB).format(a=a, b=b), f"{a} menos {b} é {a-b}."
        elif tipo == "div":
            q_, b = rnd.randint(2, 12), rnd.randint(2, 12)
            a = q_ * b
            q, r = rnd.choice(_FORMAS_DIV).format(a=a, b=b), f"{a} dividido por {b} é {q_}."
        else:  # seguir instrução: só o número
            q = f"Responda só com o número: quanto é {a} vezes {b}?"
            r = str(a * b)
        if contaminado(q, eval_qs):
            continue
        out.append((q, r))
    return out


_SIM_NAO = [
    ("o Python é uma linguagem de programação?", "Sim."),
    ("G02 é interpolação anti-horária?", "Não."),
    ("G03 é interpolação anti-horária?", "Sim."),
    ("você foi criada pelo Hugo?", "Sim."),
    ("você funciona sem internet?", "Sim."),
    ("o M05 liga o eixo-árvore?", "Não."),
    ("o M08 liga o refrigerante?", "Sim."),
    ("a Unity usa Python como linguagem principal?", "Não."),
    ("o Blender é um programa de modelagem 3D?", "Sim."),
    ("uma hora tem 60 minutos?", "Sim."),
    ("você sabe a previsão do tempo?", "Não."),
    ("o G91 é modo incremental?", "Sim."),
    ("o G90 é modo incremental?", "Não."),
    ("você é uma pessoa?", "Não."),
    ("o Teacher é o seu cérebro principal?", "Não."),
]
_UMA_PALAVRA = [
    ("qual função do Python mostra texto na tela?", "print"),
    ("qual palavra cria uma função em Python?", "def"),
    ("qual método adiciona um item numa lista?", "append"),
    ("qual função conta os itens de uma lista?", "len"),
    ("qual o seu nome?", "HEILO"),
    ("quem te criou?", "Hugo"),
    ("qual formato exporta do Blender para a Unity?", "FBX"),
    ("qual programa de CAD mecânico é da Dassault?", "SolidWorks"),
]
_SO_CODIGO = [
    ("qual código G liga o modo incremental?", "G91"),
    ("qual código G é movimento rápido?", "G00"),
    ("qual código G cancela a compensação de raio?", "G40"),
    ("qual código M desliga o refrigerante?", "M09"),
    ("qual código M faz a troca de ferramenta?", "M06"),
    ("qual código G define milímetros?", "G21"),
    ("qual código G é interpolação linear?", "G01"),
    ("qual código M liga o eixo-árvore no sentido horário?", "M03"),
]


def instrucoes(eval_qs: List[str], rnd: random.Random) -> List[Par]:
    out: List[Par] = []
    prefixos_sn = ["Responda apenas sim ou não: ", "Só sim ou não: ", "Responda com sim ou não: "]
    for q, r in _SIM_NAO:
        for p in rnd.sample(prefixos_sn, 2):
            out.append((p + q, r))
    for q, r in _UMA_PALAVRA:
        out.append(("Responda com uma palavra: " + q, r))
        out.append(("Uma palavra só: " + q, r))
    for q, r in _SO_CODIGO:
        out.append(("Só o código, sem explicação: " + q, r))
        out.append(("Responda só com o código: " + q, r))
    return [(q, r) for q, r in out if not contaminado(q, eval_qs)]


_SEM_ACESSO = [
    ("qual a cotação do euro hoje?", "Não tenho cotações em tempo real. Vale conferir num site de câmbio."),
    ("vai fazer sol amanhã?", "Não tenho acesso à previsão do tempo. Um app de clima te responde na hora."),
    ("qual a temperatura agora?", "Não consigo ver a temperatura daqui. Confira num app de clima."),
    ("quem venceu a eleição ontem?", "Não tenho acesso a notícias em tempo real."),
    ("que horas são agora?", "Não tenho relógio aqui. Dá para ver no canto da tela."),
    ("qual o resultado do jogo do flamengo?", "Não tenho acesso a resultados em tempo real."),
    ("quanto está o bitcoin?", "Não tenho cotações em tempo real."),
    ("quais as notícias de hoje?", "Não tenho acesso a notícias. Posso ajudar com outra coisa?"),
    ("tem trânsito agora na minha rua?", "Não tenho acesso ao trânsito em tempo real. Um app de mapas mostra isso."),
    ("qual o preço do arroz no mercado?", "Não tenho preços atualizados. Vale conferir no mercado."),
    ("quem é o técnico da seleção hoje?", "Não tenho informações atualizadas sobre isso."),
    ("qual a previsão pro fim de semana?", "Não tenho acesso à previsão do tempo."),
]


def honestidade(eval_qs: List[str]) -> List[Par]:
    return [(q, r) for q, r in _SEM_ACESSO if not contaminado(q, eval_qs)]


_IDENTIDADE = [
    ("qual seu nome, IA?", "Meu nome é HEILO."),
    ("como te chamam?", "Me chamam de HEILO."),
    ("quem te programou?", "Fui criada pelo Hugo."),
    ("quem é seu dono?", "Fui criada pelo Hugo."),
    ("você é humana?", "Não, sou uma inteligência artificial."),
    ("você é um robô?", "Sou uma inteligência artificial, um programa de computador."),
    ("você é o qwen disfarçado?", "Não. Sou a HEILO. O Qwen é só um professor temporário."),
    ("o qwen é você?", "Não. O Qwen é o HEILO Teacher, um professor temporário. Eu sou a HEILO."),
    ("qual é o seu modelo?", "Eu uso o HEILO Seed, meu modelo próprio treinado do zero."),
    ("seu cérebro é o qwen?", "Não. Meu modelo é o HEILO Seed. O Qwen só ajuda no treino."),
    ("você roda na nuvem?", "Não preciso de nuvem. Eu rodo localmente no computador."),
    ("você funciona offline?", "Funciono sim! O HEILO Seed roda localmente."),
]


def identidade(eval_qs: List[str]) -> List[Par]:
    return [(q, r) for q, r in _IDENTIDADE if not contaminado(q, eval_qs)]


# ------------------------------------------------------------- com o Teacher
DESCRICOES = {
    "conversa": "conversa do dia a dia, sentimentos, cumprimentos, pedidos de conselho simples",
    "python": "dúvidas básicas de programação em Python (listas, funções, laços, arquivos, erros)",
    "ferramentas": "Blender, Unity, SolidWorks, CAD/CAM e SAP, perguntas curtas de iniciante",
}

_FORMAS_CNC = ["o que faz o {c}?", "para que serve o {c}?", "o que significa {c} no Fanuc?",
               "me explica o {c}", "qual a função do {c} no CNC?"]

SISTEMA_PERGUNTAS = ("Você cria perguntas para treinar uma assistente. Escreva só as perguntas, "
                     "uma por linha, em português do Brasil, sem numeração e sem respostas.")
SISTEMA_RESPOSTA = ("Você é a HEILO, uma IA criada pelo Hugo. Responda em português do Brasil, "
                    "de forma correta, simpática e curta: no máximo 2 frases (até 160 caracteres). "
                    "Nunca diga que é outro modelo.")
SISTEMA_RACIOCINIO = ("Você é a HEILO. Explique em português do Brasil, em no máximo 2 frases curtas, "
                      "mostrando a conta e o resultado final.")


def perguntas_cnc(n: int, rnd: random.Random, eval_qs: List[str]) -> List[str]:
    codigos = [c.upper() for c in FATOS_CNC if len(c) == 3]
    out, tent = [], 0
    while len(out) < n and tent < n * 30:
        tent += 1
        q = rnd.choice(_FORMAS_CNC).format(c=rnd.choice(codigos))
        if q not in out and not contaminado(q, eval_qs):
            out.append(q)
    return out


def _limpar_linha(l: str) -> str:
    l = re.sub(r"^\s*([-*•]|\d+[.)-])\s*", "", l.strip())
    return l.strip().strip('"').strip()


def perguntas_teacher(teacher: Callable[[List[Dict], str, float, int], str], categoria: str,
                      exemplos: List[str], n: int, eval_qs: List[str],
                      conhecidos: set, tentativas: int = 4) -> List[str]:
    """Self-instruct: o Teacher inventa perguntas novas para uma categoria."""
    out: List[str] = []
    for _ in range(tentativas):
        if len(out) >= n:
            break
        ex = "\n".join(f"- {e}" for e in exemplos[:4])
        pedido = (f"Escreva 8 perguntas curtas e diferentes sobre: {DESCRICOES[categoria]}.\n"
                  f"Exemplos do estilo:\n{ex}")
        texto = teacher([{"role": "user", "content": pedido}], SISTEMA_PERGUNTAS, 0.9, 220)
        for l in texto.split("\n"):
            q = _limpar_linha(l)
            if not (6 <= len(q) <= 120) or q.lower() in conhecidos:
                continue
            if contaminado(q, eval_qs) or q in out:
                continue
            out.append(q)
    return out[:n]


def perguntas_raciocinio(n: int, rnd: random.Random, eval_qs: List[str]) -> List[str]:
    out, tent = [], 0
    while len(out) < n and tent < n * 30:
        tent += 1
        a, b = rnd.randint(3, 15), rnd.randint(3, 15)
        q = rnd.choice([f"explica como calcular {a} vezes {b}",
                        f"por que {a} vezes {b} dá {a*b}?",
                        f"me mostra a conta de {a} vezes {b}"])
        if not contaminado(q, eval_qs) and q not in out:
            out.append(q)
    return out
