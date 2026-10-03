"""
HEILO Fase 2 — a Faísca 3.0 (200M, só português) esticada para 400M e treinada com
conhecimento geral, código e contas, SEM esquecer o português.

Mistura (fim da fase; ver MISTURA): 60% português (revisão do corpus da Fase 1),
21% conhecimento geral (FineWeb-Edu, inglês, ODC-By), 12% código Python de licença
permissiva (github-code-clean: só MIT/Apache/BSD/ISC/Unlicense/CC0), 6% contas e
raciocínio gerados aqui (respostas calculadas por código) e 1% conversa (exemplos
aprovados pelo Hugo).

Degraus: o português começa em 80% e desce para 70% e 60% (um terço da fase cada).
Taxa de aprendizado: reaquecimento até 1,5e-4, constante, e queda só nos últimos 20%
("WSD"; Ibrahim et al. 2024). Régua: R1 não pode piorar mais de 3% em relação à 3.0.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional

from heilo.training import faisca_grande, pretrain

Log = Callable[[str], None]

CONFIG = faisca_grande.CONFIG_FAISCA_400M
TOKENS_FASE2 = 3_000_000_000
PASSOS_FASE2 = TOKENS_FASE2 // (faisca_grande.LOTE_SEQUENCIAS * CONFIG.block_size)   # 91.552

# fração FINAL de cada fonte nova no lote; o resto (60%) é português da Fase 1
MISTURA = {"geral_en": 0.21, "codigo": 0.12, "contas": 0.06, "conversa": 0.01}
# quantos tokens preparar de cada fonte (o que for menor que a fatia é relido algumas vezes)
TOKENS_FONTE = {"geral_en": 250_000_000, "codigo": 200_000_000, "contas": 40_000_000, "conversa": None}
PARTES_MAX = {"geral_en": 3, "codigo": 2, "contas": 1, "conversa": 1}
DEGRAUS = (0.20, 0.30, 0.40)        # parte nova do lote em cada terço da fase (PT = 80/70/60%)
LICENCAS_OK = {"mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "isc", "unlicense", "cc0-1.0"}


# ------------------------------------------------------------------ fontes
def textos_geral_en(parte: int, n_divisoes: int = 64, log: Log = print) -> Iterator[str]:
    """FineWeb-Edu (amostra de 10 bi tokens), um pedaço diferente por parte."""
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    try:
        n = min(n_divisoes, ds.n_shards)
        ds = ds.shard(num_shards=n, index=(parte - 1) % n)
    except Exception as e:
        log(f"aviso: não consegui dividir o FineWeb-Edu ({e}); lendo do início")
    for item in ds:
        t = (item.get("text") or "").strip()
        if len(t) >= 300:
            yield t


def codigo_aceito(item: Dict) -> bool:
    return (item.get("language") == "Python" and str(item.get("license", "")).lower() in LICENCAS_OK
            and 200 <= len(item.get("code") or "") <= 60_000)


def textos_codigo(parte: int, arquivos_por_parte: int = 40, log: Log = print) -> Iterator[str]:
    """Código Python de licença permissiva (github-code-clean, arquivos parquet)."""
    from datasets import load_dataset
    ini = (parte - 1) * arquivos_por_parte
    # lê os parquet direto pelo leitor "parquet": o repositório ainda tem um script de carga
    # (github-code-clean.py) que as versões novas do `datasets` recusam
    base = "hf://datasets/codeparrot/github-code-clean/data"
    arqs = [f"{base}/train-{i:05d}-of-00880.parquet" for i in range(ini, min(880, ini + arquivos_por_parte))]
    ds = load_dataset("parquet", data_files=arqs, split="train", streaming=True)
    for item in ds:
        if codigo_aceito(item):
            yield f"# arquivo: {item.get('path', '')}\n{item['code']}"


# ------------------------------------------------------------------ contas (geradas aqui)
def _num(n: float) -> str:
    """Número no jeito brasileiro: 1.234,5 (sem zeros sobrando)."""
    if float(n).is_integer():
        return f"{int(n):,}".replace(",", ".")
    s = f"{n:,.2f}".rstrip("0").rstrip(".")
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def gerar_conta(rnd: random.Random) -> str:
    """Um exercício resolvido em português; a resposta é sempre calculada, nunca inventada."""
    t = rnd.randrange(12)
    a, b = rnd.randint(2, 999), rnd.randint(2, 999)
    if t == 0:
        return f"Quanto é {_num(a)} + {_num(b)}? {_num(a)} + {_num(b)} = {_num(a + b)}."
    if t == 1:
        a, b = max(a, b), min(a, b)
        return f"Quanto é {_num(a)} − {_num(b)}? {_num(a)} − {_num(b)} = {_num(a - b)}."
    if t == 2:
        a, b = rnd.randint(2, 99), rnd.randint(2, 99)
        return f"Quanto é {a} vezes {b}? {a} × {b} = {_num(a * b)}."
    if t == 3:
        a, b = rnd.randint(1, 10), rnd.randint(1, 10)
        return f"Tabuada: {a} × {b} = {a * b}."
    if t == 4:
        b = rnd.randint(2, 30)
        q, r = rnd.randint(1, 99), rnd.randint(0, b - 1)
        a = b * q + r
        return (f"Quanto é {_num(a)} dividido por {b}? {_num(a)} ÷ {b} = {q}" +
                (f", com resto {r}." if r else " exatamente."))
    if t == 5:
        p = rnd.choice([5, 10, 12, 15, 17, 20, 25, 30, 40, 50, 75])
        v = rnd.randint(1, 500) * rnd.choice([1, 10])
        return f"Quanto é {p}% de {_num(v)}? {p}% de {_num(v)} = {_num(v * p / 100)}."
    if t == 6:
        mm = rnd.randint(1, 500)
        return f"{mm} mm em polegadas: {mm} ÷ 25,4 = {_num(round(mm / 25.4, 2))} pol."
    if t == 7:
        h, m = rnd.randint(1, 23), rnd.randint(0, 59)
        return f"{h} horas e {m} minutos são {h * 60 + m} minutos."
    if t == 8:
        km = rnd.randint(1, 900)
        return f"{km} km são {_num(km * 1000)} metros."
    if t == 9:
        x, k = rnd.randint(1, 50), rnd.randint(2, 12)
        c = rnd.randint(0, 99)
        return f"Resolva {k}x + {c} = {k * x + c}. Tiramos {c} dos dois lados: {k}x = {k * x}. Dividimos por {k}: x = {x}."
    if t == 10:
        n = rnd.randint(2, 31)
        return f"A raiz quadrada de {n * n} é {n}, porque {n} × {n} = {n * n}."
    preco, qtd = rnd.randint(2, 90), rnd.randint(2, 12)
    pago = (preco * qtd // 10 + 1) * 10 + rnd.choice([0, 10, 50])
    return (f"Comprei {qtd} itens de R$ {preco} cada e paguei com R$ {pago}. "
            f"O total é {qtd} × {preco} = R$ {qtd * preco}, e o troco é {pago} − {qtd * preco} = R$ {pago - qtd * preco}.")


def textos_contas(n_docs: int = 400_000, por_doc: int = 12, seed: int = 2026) -> Iterator[str]:
    rnd = random.Random(seed)
    for _ in range(n_docs):
        yield "\n".join(gerar_conta(rnd) for _ in range(por_doc))


# ------------------------------------------------------------------ conversa (aprovados)
def textos_conversa(exemplos: Iterable[Dict]) -> Iterator[str]:
    """Exemplos APROVADOS (só o split de treino), como texto: Pergunta / Resposta."""
    for ex in exemplos:
        linhas = []
        for m in ex.get("messages", []):
            if m.get("role") == "user":
                linhas.append(f"Pergunta: {m['content'].strip()}")
            elif m.get("role") == "assistant":
                linhas.append(f"HEILO: {m['content'].strip()}")
        if len(linhas) >= 2:
            yield "\n".join(linhas)


# ------------------------------------------------------------------ preparo e pesos
def preparar_fontes(pasta: Path, tok, exemplos_conversa: Optional[List[Dict]] = None,
                    novas_por_sessao: int = 1, log: Log = print) -> Dict[str, List[Path]]:
    """Garante as partes de cada fonte em pasta/<fonte>/parte_NN. Por sessão prepara no máximo
    `novas_por_sessao` parte nova de cada fonte da web (o resto já está no Drive)."""
    pasta = Path(pasta)
    geradores = {
        "geral_en": lambda n: textos_geral_en(n, log=log),
        "codigo": lambda n: textos_codigo(n, log=log),
        "contas": lambda n: textos_contas(seed=2026 + n),
        "conversa": lambda n: textos_conversa(exemplos_conversa or []),
    }
    for fonte, gerar in geradores.items():
        if fonte == "conversa" and not exemplos_conversa:
            continue
        dirf = pasta / fonte
        dirf.mkdir(parents=True, exist_ok=True)
        feitas = 0
        while feitas < novas_por_sessao:
            prontas = pretrain.pastas_corpus(dirf)
            if len(prontas) >= PARTES_MAX[fonte]:
                break
            n = len(prontas) + 1
            destino = dirf / f"parte_{n:02d}"
            if destino.exists():
                for f in destino.glob("*"):
                    f.unlink()
            try:
                meta = pretrain.preparar_corpus(gerar(n), tok, destino,
                                                max_tokens=TOKENS_FONTE[fonte] or 10 ** 12, log=lambda s: None)
                log(f"{fonte}/parte_{n:02d} pronta: {meta['tokens_treino']:,} tokens")
            except Exception as e:
                log(f"não consegui preparar {fonte} hoje (continua com o que já tem): {e}")
                break
            feitas += 1
    return {f: pretrain.pastas_corpus(pasta / f) for f in MISTURA}


def degrau(passo: int, passo_inicial_fase: int, passos_fase: int = PASSOS_FASE2) -> int:
    prog = (passo - passo_inicial_fase) / max(1, passos_fase)
    return 0 if prog < 1 / 3 else (1 if prog < 2 / 3 else 2)


def pesos(d: int, fontes_prontas: Dict[str, List[Path]]) -> Dict[str, float]:
    """Fração fixa do lote para cada fonte nova no degrau d (só as que têm partes prontas);
    o que sobra fica para o português."""
    ativos = {f: MISTURA[f] for f, ps in fontes_prontas.items() if ps}
    if not ativos:
        return {}
    escala = DEGRAUS[d] / sum(MISTURA.values())
    return {f: round(w * escala, 6) for f, w in ativos.items()}


def aplicar_pesos(partes_locais: Dict[str, List[Path]], w: Dict[str, float]) -> None:
    """Escreve `peso_fixo` no meta.json das cópias locais: o peso da fonte é dividido entre
    as partes dela. Partes de português ficam sem peso fixo (dividem o resto)."""
    for fonte, ps in partes_locais.items():
        for p in ps:
            mp = Path(p) / "meta.json"
            meta = json.loads(mp.read_text(encoding="utf-8"))
            if fonte in w and w[fonte] > 0:
                meta["peso_fixo"] = w[fonte] / len(ps)
            else:
                meta.pop("peso_fixo", None)
            mp.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def preparar_cerebro(marco_fase1: Path, ck_fase2: Path, log: Log = print) -> Dict[str, int]:
    """Na 1ª sessão: estica a 3.0 (16 camadas) para 32 camadas. Depois só continua.
    Devolve o passo em que a Fase 2 começou e o passo atual."""
    import torch
    ck_fase2 = Path(ck_fase2)
    if not ck_fase2.exists():
        faisca_grande.crescer_profundidade(marco_fase1, ck_fase2, CONFIG.n_layer, log=log)
    st = torch.load(ck_fase2, map_location="cpu", weights_only=False)
    return {"inicio": int(st.get("inicio_agenda", st["passo"])), "passo": int(st["passo"])}
