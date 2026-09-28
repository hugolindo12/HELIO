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

from heilo.models.seed.gpt import GPTConfig
from heilo.training import aurora

# 16 camadas, 1024 dimensões, 16 cabeças (4 de chave/valor), contexto 1024, vocab 32k → ~210 M
VOCAB = 32000
CONFIG_FAISCA_200M = GPTConfig(block_size=1024, n_layer=16, n_head=16, n_embd=1024, dropout=0.0,
                               arquitetura="moderna", n_kv_head=4, vocab_size=VOCAB)
TOKENS_ALVO = 3_000_000_000          # ~15 tokens por parâmetro (o ideal "Chinchilla" seria ~4 bi)
LOTE_SEQUENCIAS = 32                 # lote efetivo: 32 × 1024 tokens por passo
CARACTERES_TOKENIZER = 300_000_000   # amostra usada para treinar o tokenizer (wiki + web)


def lote_para_gpu(nome_gpu: str, memoria_gb: float) -> Dict:
    """Micro-lote que cabe na GPU (o modelo é ~40% maior que a Aurora); lote efetivo fixo."""
    micro = 16 if memoria_gb >= 38 else (8 if memoria_gb >= 20 else 4)
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
