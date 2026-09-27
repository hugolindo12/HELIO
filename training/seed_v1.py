"""
Receita do HEILO Seed v1.x (usada pelo notebook training/colab/heilo_seed_v1.ipynb).

    corpus geral PT → tokenizer BPE próprio → pré-treino → SFT (dataset HEILO) →
    avaliação (heilo_eval_v1 + sonda independente) → registro/promoção só se melhorar
"""
from __future__ import annotations

import json
import re
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional

from heilo.config import DATA_DIR
from heilo.models.seed.gpt import GPTConfig
from heilo.models.seed.tokenizer import BPETokenizer
from heilo.training import pretrain
from heilo.training.records import read_jsonl

# Configuração padrão do v1.0 (~30 M parâmetros com vocabulário de 16k)
CONFIG_V1 = GPTConfig(block_size=512, n_layer=8, n_head=8, n_embd=512, dropout=0.05)
# v2.x: mesmo tamanho (~30 M), "motor" moderno no estilo Gemma — RoPE, RMSNorm, GeGLU,
# GQA (2 cabeças de chave/valor) e QK-norm. Treinada do zero; só as técnicas vêm do Gemma.
CONFIG_V2 = GPTConfig(block_size=512, n_layer=8, n_head=8, n_embd=512, dropout=0.0,
                      arquitetura="moderna", n_kv_head=2)
VOCAB_V1 = 16000


def limpar_artigo(texto: str) -> str:
    """Parágrafos de texto corrido (tira listas, tabelas e títulos curtos)."""
    bons = []
    for p in texto.split("\n"):
        p = p.strip()
        if len(p) < 120 or p.startswith(("|", "{", "*", "#", "=")):
            continue
        bons.append(re.sub(r"\s+", " ", p))
    return "\n".join(bons)


def textos_wikipedia(max_chars: int, config: str = "20231101.pt",
                     log: Callable[[str], None] = print) -> Iterator[str]:
    """Wikipédia em português (Hugging Face, streaming). Só roda onde há internet (Colab)."""
    from datasets import load_dataset
    ds = load_dataset("wikimedia/wikipedia", config, split="train", streaming=True)
    total = 0
    for art in ds:
        t = limpar_artigo(art["text"])
        if len(t) < 300:
            continue
        total += len(t)
        yield t
        if total >= max_chars:
            log(f"  corpus: {total:,} caracteres")
            return


def textos_heilo(data_dir: Optional[Path] = None) -> List[str]:
    """Falas dos exemplos aprovados (entram no treino do tokenizer para ele conhecer o estilo)."""
    out = []
    for arq in sorted(Path(data_dir or DATA_DIR, "approved").glob("*.jsonl")):
        for r in read_jsonl(arq):
            out += [m["content"] for m in r.get("messages", [])]
    return out


def treinar_tokenizer(textos: Iterable[str], extras: List[str], vocab_size: int = VOCAB_V1) -> BPETokenizer:
    def todos():
        yield from textos
        yield from extras
    return BPETokenizer.treinar(todos(), vocab_size=vocab_size)


def dividir_sft(pipeline) -> Dict:
    """Usa o dataset HEILO versionado (só aprovados, validação separada por id)."""
    m = pipeline.build_dataset()
    pasta = pipeline.root / "datasets"
    return {"manifest": m, "treino": read_jsonl(pasta / m["train_file"]),
            "val": read_jsonl(pasta / m["val_file"])}


def registrar(ciclos, arquivo: Path, nome: str, manifest: Dict, relatorio: Dict,
              promover: bool = True,
              origem: str = "pré-treino em texto geral PT + SFT com o dataset HEILO (tokenizer BPE próprio)") -> Dict:
    return ciclos.registrar_versao_externa(
        arquivo, nome, ciclo=None, pai=None, dataset_version=manifest["version"],
        origem=origem, promover=promover, extra={"treino": relatorio})
