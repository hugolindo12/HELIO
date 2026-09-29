"""
HEILO Faísca 200M — a Faísca "de verdade": ~210 M parâmetros, treinada do zero em sessões
diárias de ~4 h no Colab (continua de onde parou).

O que muda em relação à Faísca 2.x (30 M):
- 7× maior (16 camadas × 1024 dimensões, 16 cabeças / 4 de chave-valor, contexto 1024).
- Tokenizer NOVO, com 32 mil pedaços e ALGARISMOS SEPARADOS ("48" → "4", "8"): o antigo
  juntava números inteiros num token só, e é por isso que a 2.x errava contas.
- Corpus: Wikipédia PT + texto da web em português (FineWeb-2), tokenizados com o novo
  tokenizer (numa pasta própria no Drive: heilo_faisca200).
- Ajuste de conversa com a seleção corrigida (pelas respostas geradas).

Reaproveita a infraestrutura da Aurora (partes da web, lote por GPU, envio em partes ao GitHub).
"""
from __future__ import annotations

import itertools
from pathlib import Path
from typing import Callable, Dict, Iterator

import dataclasses

from heilo.models.seed.gpt import GPTConfig
from heilo.training import aurora

# 16 camadas, 1024 dimensões, 16 cabeças (4 de chave/valor), contexto 1024, vocab 32k → ~210 M
VOCAB = 32000
CONFIG_FAISCA_200M = GPTConfig(block_size=1024, n_layer=16, n_head=16, n_embd=1024, dropout=0.0,
                               arquitetura="moderna", n_kv_head=4, vocab_size=VOCAB)
TOKENS_ALVO = 3_000_000_000          # ~15 tokens por parâmetro (o ideal "Chinchilla" seria ~4 bi)

# Faísca 400M: a 200M "esticada" em profundidade (16 → 32 camadas) → ~390 M.
# Não começa do zero: cada camada da 200M vira duas (ver crescer_profundidade).
CONFIG_FAISCA_400M = dataclasses.replace(CONFIG_FAISCA_200M, n_layer=32)
TOKENS_ALVO_400M = 3_000_000_000     # dá para aumentar depois: o treino só continua
LOTE_SEQUENCIAS = 32                 # lote efetivo: 32 × 1024 tokens por passo
CARACTERES_TOKENIZER = 300_000_000   # amostra usada para treinar o tokenizer (wiki + web)


def lote_para_gpu(nome_gpu: str, memoria_gb: float, n_layer: int = 16) -> Dict:
    """Micro-lote que cabe na GPU; o lote efetivo fica sempre em 32 × 1024 tokens.
    Com 32 camadas (400M) as ativações dobram, então o micro-lote cai pela metade."""
    micro = 16 if memoria_gb >= 38 else (8 if memoria_gb >= 20 else 4)
    if n_layer > 16:
        micro = max(2, micro // 2)
    return {"micro": micro, "acumular": LOTE_SEQUENCIAS // micro}


def _amostra_tokenizer(max_chars: int, log: Callable[[str], None]) -> Iterator[str]:
    """Metade Wikipédia, metade web: o tokenizer aprende os dois estilos de texto."""
    from heilo.training import seed_v1
    metade = max_chars // 2
    wiki = seed_v1.textos_wikipedia(metade, log=log)

    def web():
        total = 0
        for t in aurora.textos_web_pt(1, log=log):
            total += len(t)
            yield t
            if total >= metade:
                return
    # intercala para não depender da ordem
    for a, b in itertools.zip_longest(wiki, web()):
        if a:
            yield a
        if b:
            yield b


def preparar_tokenizer(pasta: Path, log: Callable[[str], None] = print,
                       max_chars: int = CARACTERES_TOKENIZER, textos=None):
    """Treina (uma vez) e salva o tokenizer da Faísca 200M em pasta/tokenizer.json."""
    from heilo.models.seed.tokenizer import BPETokenizer
    arq = Path(pasta) / "tokenizer.json"
    if arq.exists():
        return BPETokenizer.from_json(arq.read_text(encoding="utf-8"))
    arq.parent.mkdir(parents=True, exist_ok=True)
    tok = BPETokenizer.treinar(textos if textos is not None else _amostra_tokenizer(max_chars, log),
                               vocab_size=VOCAB, digitos_separados=True)
    tmp = arq.with_suffix(".tmp")
    tmp.write_text(tok.tok.to_str(), encoding="utf-8")
    tmp.replace(arq)
    log(f"tokenizer novo: {tok.vocab_size} pedaços, algarismos separados")
    return tok


def crescer_profundidade(ck_origem: Path, ck_destino: Path, n_layer_novo: int = 32,
                         log: Callable[[str], None] = print) -> Dict:
    """Cria o checkpoint de um modelo MAIS FUNDO a partir de um já treinado, sem perder o
    que ele sabe (técnica parecida com a do SOLAR / "depth up-scaling").

    Cada camada i vira duas seguidas: a original e uma cópia. Na cópia, as projeções de
    saída (atenção `o` e MLP `baixo`) começam em ZERO, então ela não altera nada no início:
    o modelo novo responde exatamente como o antigo e vai usando as camadas novas conforme
    treina. O otimizador recomeça (o dele não serve para o tamanho novo) e a taxa de
    aprendizado faz um novo aquecimento a partir deste passo."""
    import torch
    from heilo.models.seed.gpt import MiniGPT
    st = torch.load(ck_origem, map_location="cpu", weights_only=False)
    cfg_antigo = GPTConfig(**st["config"])
    n_antigo = cfg_antigo.n_layer
    if n_layer_novo % n_antigo:
        raise ValueError(f"{n_layer_novo} camadas não é múltiplo de {n_antigo}")
    rep = n_layer_novo // n_antigo
    velho = st["modelo"]
    novo = {}
    for k, v in velho.items():
        if not k.startswith("blocos."):
            novo[k] = v.clone()
    for i in range(n_antigo):
        pref = f"blocos.{i}."
        for r in range(rep):
            j = i * rep + r
            for k, v in velho.items():
                if k.startswith(pref):
                    nome = k[len(pref):]
                    w = v.clone()
                    if r > 0 and nome in ("o.weight", "baixo.weight"):
                        w.zero_()
                    novo[f"blocos.{j}.{nome}"] = w
    cfg_novo = dataclasses.replace(cfg_antigo, n_layer=n_layer_novo)
    modelo = MiniGPT(cfg_novo)
    modelo.load_state_dict(novo)
    opt = torch.optim.AdamW(modelo.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.1)
    passo = st["passo"]
    info = {"de": f"{n_antigo} camadas", "para": f"{n_layer_novo} camadas", "passo": passo,
            "parametros_M": round(sum(p.numel() for p in modelo.parameters()) / 1e6, 1)}
    Path(ck_destino).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(ck_destino).with_suffix(".tmp")
    torch.save({"modelo": modelo.state_dict(), "opt": opt.state_dict(), "passo": passo,
                "hist": st.get("hist", []) + [{"passo": passo, "crescimento": info}],
                "config": cfg_novo.__dict__, "inicio_agenda": passo, "minutos_agenda": 0.0}, tmp)
    tmp.replace(ck_destino)
    log(f"cérebro esticado: {info['de']} → {info['para']} ({info['parametros_M']} M parâmetros), "
        f"continua do passo {passo}")
    return info
