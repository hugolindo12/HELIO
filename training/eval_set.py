"""
Conjunto de AVALIAÇÃO fixo do HEILO Seed (heilo_eval_v1).

Regras:
- Nenhuma pergunta daqui pode entrar no treino (o gerador de dados do Teacher
  filtra perguntas parecidas com estas → evita "decorar a prova").
- Cada item tem uma resposta de referência (para perda/perplexidade) e critérios
  objetivos de acerto (palavras que PRECISAM aparecer / não podem aparecer / regex).
- tipo:  "paráfrase"     → assunto visto no treino, pergunta escrita diferente
         "generalizacao" → combinação/assunto NÃO visto no treino
         "instrucao"     → pede um formato específico (seguir instruções)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

EVAL_VERSION = "heilo_eval_v1"

# (categoria, tipo, pergunta, referência, criterios)
# criterios: any=[[alternativas], ...] (cada grupo precisa ter 1 presente),
#            none=[proibidas], regex="padrão que a resposta inteira deve casar"
_ITENS = [
    # --- saudação / conversa
    ("conversa", "paráfrase", "opa, tudo tranquilo?", "Tudo tranquilo por aqui! E contigo?", {"any": [["tudo", "tranquilo", "bem", "otimo", "certo"]]}),
    ("conversa", "paráfrase", "bom dia, heilo!", "Bom dia! O que vamos fazer hoje?", {"any": [["bom dia"]]}),
    ("conversa", "paráfrase", "boa noite pra você", "Boa noite! Posso ajudar em algo?", {"any": [["boa noite"]]}),
    ("conversa", "paráfrase", "muito obrigado pela ajuda", "De nada! Sempre que precisar.", {"any": [["nada", "prazer", "tamo junto", "disponha", "sempre"]]}),
    ("conversa", "paráfrase", "tchauzinho", "Tchau! Até a próxima.", {"any": [["tchau", "ate", "falou"]]}),
    ("conversa", "paráfrase", "hoje eu tô muito cansado", "Descansa um pouco! Quer que eu adiante algo leve?", {"any": [["descans", "pausa", "calma", "imagino", "cansa"]]}),
    ("conversa", "paráfrase", "estou feliz demais hoje", "Que bom! O que aconteceu?", {"any": [["que bom", "que massa", "que otimo", "aconteceu", "parabens", "legal"]]}),
    ("conversa", "paráfrase", "me conta uma piadinha", "Por que o livro de matemática ficou triste? Porque tinha muitos problemas.", {"any": [["por que", "porque", "o que"]]}),
    ("conversa", "generalizacao", "to com fome", "Então vai comer alguma coisa! Depois a gente continua.", {"any": [["com", "fome", "lanche", "comida", "comer"]], "none": ["g01", "python"]}),
    ("conversa", "generalizacao", "meu cachorro fugiu", "Poxa, sinto muito! Espero que ele volte logo.", {"any": [["sinto", "poxa", "espero", "calma", "procur"]]}),
    # --- identidade
    ("identidade", "paráfrase", "como você se chama?", "Eu me chamo HEILO.", {"any": [["heilo"]]}),
    ("identidade", "paráfrase", "quem é o seu criador?", "Fui criada pelo Hugo.", {"any": [["hugo"]]}),
    ("identidade", "paráfrase", "você é uma pessoa de verdade?", "Não, sou uma inteligência artificial.", {"any": [["nao", "inteligencia artificial", "programa de computador"]]}),
    ("identidade", "paráfrase", "você precisa do qwen pra funcionar?", "Não. O Qwen é só um professor temporário.", {"any": [["nao"]], "none": ["sim, preciso"]}),
    ("identidade", "paráfrase", "o que é o seed?", "É o meu modelo próprio, treinado do zero.", {"any": [["modelo", "proprio", "zero"]]}),
    ("identidade", "generalizacao", "você roda no meu computador?", "Sim, eu rodo localmente no seu computador.", {"any": [["sim", "local", "computador"]]}),
    # --- honestidade
    ("honestidade", "paráfrase", "como vai estar o clima amanhã?", "Não tenho acesso à previsão do tempo.", {"any": [["nao tenho", "nao consigo", "nao sei", "app de clima", "previsao"]]}),
    ("honestidade", "paráfrase", "quanto está o dólar agora?", "Não tenho cotações em tempo real.", {"any": [["nao tenho", "nao consigo", "nao sei", "tempo real"]]}),
    ("honestidade", "generalizacao", "quem é o presidente da frança hoje?", "Não tenho informações atualizadas sobre isso.", {"any": [["nao tenho", "nao sei", "nao consigo", "atualizad"]]}),
    ("honestidade", "generalizacao", "qual o placar do jogo de hoje?", "Não tenho acesso a resultados em tempo real.", {"any": [["nao tenho", "nao sei", "nao consigo", "tempo real"]]}),
    ("honestidade", "paráfrase", "sua resposta está errada", "Obrigada por avisar! Qual é a resposta certa?", {"any": [["obrigad", "desculp", "certa", "ensinar", "corrig"]]}),
    # --- python
    ("python", "paráfrase", "como mostro um texto na tela em python?", "Use print, por exemplo: print(\"Olá\")", {"any": [["print"]]}),
    ("python", "paráfrase", "como declaro uma função em python?", "Use def: def nome(): ...", {"any": [["def"]]}),
    ("python", "paráfrase", "como faço um laço de repetição em python?", "Use for: for i in range(10): ...", {"any": [["for", "while"]]}),
    ("python", "paráfrase", "como pego o que o usuário digitou?", "Use input: nome = input()", {"any": [["input"]]}),
    ("python", "paráfrase", "como coloco um elemento no final da lista?", "Use append: lista.append(x)", {"any": [["append"]]}),
    ("python", "paráfrase", "como descubro quantos itens tem uma lista?", "Use len(lista).", {"any": [["len"]]}),
    ("python", "paráfrase", "como transformo a string \"7\" em inteiro?", "Use int: int(\"7\")", {"any": [["int("]]}),
    ("python", "paráfrase", "pra que serve o try?", "Para tratar erros sem parar o programa.", {"any": [["erro", "except", "excec"]]}),
    ("python", "paráfrase", "como instalo o numpy?", "pip install numpy", {"any": [["pip install"]]}),
    ("python", "generalizacao", "como ordeno uma lista em python?", "Use sorted(lista) ou lista.sort().", {"any": [["sort"]]}),
    ("python", "generalizacao", "como junto duas listas?", "Use +: lista1 + lista2", {"any": [["+", "extend"]]}),
    ("python", "generalizacao", "o que é uma lista vazia?", "É uma lista sem itens: []", {"any": [["[]", "sem itens", "nenhum item", "vazia"]]}),
    # --- cnc
    ("cnc", "paráfrase", "pra que serve o código g01?", "Movimento linear com avanço.", {"any": [["linear", "linha reta"]]}),
    ("cnc", "paráfrase", "o g00 faz o quê?", "Movimento rápido, sem usinar.", {"any": [["rapido"]]}),
    ("cnc", "paráfrase", "qual a função do g02?", "Interpolação circular no sentido horário.", {"any": [["horario"]], "none": ["anti-horario"]}),
    ("cnc", "paráfrase", "g03 é horário ou anti-horário?", "Anti-horário.", {"any": [["anti-horario", "anti horario"]]}),
    ("cnc", "paráfrase", "qual código cancela a compensação de raio?", "G40.", {"any": [["g40"]]}),
    ("cnc", "paráfrase", "compensação de raio à direita é qual código?", "G42.", {"any": [["g42"]]}),
    ("cnc", "paráfrase", "o que o m08 faz?", "Liga o refrigerante.", {"any": [["refrigera", "fluido", "liquido"]], "none": ["desliga"]}),
    ("cnc", "paráfrase", "qual código desliga o fuso?", "M05.", {"any": [["m05", "m5"]]}),
    ("cnc", "paráfrase", "qual o código de fim de programa?", "M30.", {"any": [["m30", "m02", "m2"]]}),
    ("cnc", "paráfrase", "g91 é absoluto ou incremental?", "Incremental.", {"any": [["incremental"]]}),
    ("cnc", "generalizacao", "qual código usar para trabalhar em milímetros?", "G21.", {"any": [["g21"]]}),
    ("cnc", "generalizacao", "qual ciclo usar para furar com quebra de cavaco?", "G83.", {"any": [["g83"]]}),
    # --- ferramentas
    ("ferramentas", "paráfrase", "pra que serve o blender?", "Modelagem e animação 3D.", {"any": [["3d", "modelagem", "animacao"]]}),
    ("ferramentas", "paráfrase", "a unity usa qual linguagem?", "C#.", {"any": [["c#"]]}),
    ("ferramentas", "paráfrase", "o que é um erp?", "Sistema de gestão empresarial, como o SAP.", {"any": [["gestao", "empresa", "sap"]]}),
    ("ferramentas", "generalizacao", "qual formato uso para levar um modelo do blender para a unity?", "FBX.", {"any": [["fbx"]]}),
    # --- matemática (pares NÃO vistos no treino)
    ("matematica", "generalizacao", "quanto é 6 vezes 8?", "6 vezes 8 é 48.", {"any": [["48"]]}),
    ("matematica", "generalizacao", "7x9", "7 vezes 9 é 63.", {"any": [["63"]]}),
    ("matematica", "generalizacao", "quanto é 4 vezes 7?", "4 vezes 7 é 28.", {"any": [["28"]]}),
    ("matematica", "generalizacao", "9x6", "9 vezes 6 é 54.", {"any": [["54"]]}),
    ("matematica", "generalizacao", "3x8", "3 vezes 8 é 24.", {"any": [["24"]]}),
    ("matematica", "generalizacao", "quanto é 12 mais 30?", "12 mais 30 é 42.", {"any": [["42"]]}),
    ("matematica", "generalizacao", "quanto é 8 mais 5?", "8 mais 5 é 13.", {"any": [["13"]]}),
    ("matematica", "generalizacao", "quanto é 40 menos 15?", "40 menos 15 é 25.", {"any": [["25"]]}),
    ("matematica", "generalizacao", "quanto é 90 dividido por 3?", "90 dividido por 3 é 30.", {"any": [["30"]]}),
    ("matematica", "paráfrase", "quanto dá 10 vezes 15?", "10 vezes 15 é 150.", {"any": [["150"]]}),
    ("matematica", "paráfrase", "quantos minutos há em uma hora?", "60 minutos.", {"any": [["60"]]}),
    ("matematica", "paráfrase", "um metro tem quantos milímetros?", "1000 milímetros.", {"any": [["1000", "1.000", "mil milimetros"]]}),
    # --- seguir instruções
    ("instrucao", "instrucao", "Responda só com o número: quanto é 5 vezes 5?", "25", {"regex": r"^\s*25\.?\s*$"}),
    ("instrucao", "instrucao", "Responda apenas sim ou não: você é uma IA?", "Sim.", {"regex": r"^\s*sim[.!]?\s*$"}),
    ("instrucao", "instrucao", "Responda apenas sim ou não: você é o Qwen?", "Não.", {"regex": r"^\s*nao[.!]?\s*$"}),
    ("instrucao", "instrucao", "Diga só o seu nome.", "HEILO", {"regex": r"^\s*(eu sou a |meu nome e )?heilo[.!]?\s*$"}),
    ("instrucao", "instrucao", "Responda com uma palavra: qual linguagem a Unity usa?", "C#", {"regex": r"^\s*c#[.!]?\s*$"}),
    ("instrucao", "instrucao", "Só o código, sem explicação: qual código G liga o modo absoluto?", "G90", {"regex": r"^\s*g90[.!]?\s*$"}),
]


def load_eval() -> List[Dict]:
    out = []
    for i, (cat, tipo, q, ref, crit) in enumerate(_ITENS):
        out.append({"id": f"{EVAL_VERSION}-{i:03d}", "categoria": cat, "tipo": tipo,
                    "pergunta": q, "referencia": ref, "criterios": crit})
    return out


def export(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    itens = load_eval()
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for it in itens:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    return len(itens)


SONDA_FILE = Path(__file__).resolve().parents[1] / "data" / "eval" / "sonda_independente_v1.jsonl"


def load_sonda(path: Path = None) -> List[Dict]:
    """Sonda independente (24 perguntas novas, criada na auditoria). Vazia se o arquivo não existir."""
    path = Path(path or SONDA_FILE)
    if not path.exists():
        return []
    itens = []
    for l in path.read_text(encoding="utf-8").splitlines():
        if l.strip():
            it = json.loads(l)
            it.setdefault("referencia", "")
            itens.append(it)
    return itens
