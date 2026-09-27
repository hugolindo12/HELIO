"""
HEILO Aurora — linha média (~150 M parâmetros), treinada do zero em várias sessões.

Por que ~150 M e não 300 M: com o Colab Pro (T4, ~4 h/dia) o que cabe são ~2 bilhões
de tokens em 2–3 semanas. Com essa quantidade de GPU, ~150 M é o tamanho que aprende
mais; 300 M ficaria "treinado pela metade" e pior (leis de escala de Chinchilla).

- Motor moderno (RoPE, RMSNorm, GeGLU, GQA, QK-norm), tokenizer BPE da HEILO (o da v1).
- Corpus: Wikipédia PT (já no Drive) + texto da web em português (FineWeb-2, por_Latn),
  preparado em PARTES: uma parte nova por sessão, o corpus cresce a cada dia.
- Checkpoint no Drive a cada 500 passos: cada sessão continua de onde a outra parou.
- Arquivo final ~300 MB: vai ao GitHub em partes de 90 MB (limite de 100 MB).
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Callable, Dict, Iterator, List

from heilo.models.seed.gpt import GPTConfig, dividir_em_partes
from heilo.training import pretrain

# 12 camadas, 1024 dimensões, 16 cabeças (4 de chave/valor), contexto 1024 → ~149 M
CONFIG_AURORA = GPTConfig(block_size=1024, n_layer=12, n_head=16, n_embd=1024, dropout=0.0,
                          arquitetura="moderna", n_kv_head=4)
TOKENS_ALVO = 2_000_000_000
TOKENS_POR_PARTE = 250_000_000
LIMITE_GITHUB = 95 * 1024 * 1024


def lote_para_gpu(nome_gpu: str, memoria_gb: float) -> Dict:
    """Micro-lote que cabe na GPU; o lote efetivo fica sempre em 32 × 1024 tokens."""
    micro = 32 if memoria_gb >= 38 else (16 if memoria_gb >= 20 else 8)
    return {"micro": micro, "acumular": 32 // micro}


def textos_web_pt(parte: int, n_divisoes: int = 64,
                  log: Callable[[str], None] = print) -> Iterator[str]:
    """Texto da web em português (FineWeb-2 por_Latn, filtrado por qualidade), em streaming.
    Cada parte lê um PEDAÇO DIFERENTE do dataset (por arquivos), sem baixar os anteriores."""
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb-2", name="por_Latn", split="train", streaming=True)
    try:
        n = min(n_divisoes, ds.n_shards)
        ds = ds.shard(num_shards=n, index=(parte - 1) % n)
    except Exception as e:  # versões antigas do datasets: lê do começo (pode repetir textos)
        log(f"aviso: não consegui dividir o dataset ({e}); lendo do início")
    for item in ds:
        t = (item.get("text") or "").strip()
        if len(t) >= 300:
            yield t


def preparar_proxima_parte(pasta: Path, tok, log: Callable[[str], None] = print) -> Dict:
    """Prepara UMA parte nova do corpus web (~250 M tokens) numa pasta parte_NN."""
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    prontas = sorted(p for p in pasta.glob("parte_*") if (p / "meta.json").exists())
    n = len(prontas) + 1
    destino = pasta / f"parte_{n:02d}"
    if destino.exists():                       # parte incompleta de uma sessão que caiu
        for f in destino.glob("*"):
            f.unlink()
    meta = pretrain.preparar_corpus(textos_web_pt(n, log=log), tok, destino,
                                    max_tokens=TOKENS_POR_PARTE, log=lambda s: None)
    log(f"parte {n:02d} pronta: {meta['tokens_treino']:,} tokens")
    return meta


def preparar_para_git(repo: Path, log: Callable[[str], None] = print) -> List[str]:
    """Antes do push: todo .pt > 95 MB em models/ vira partes de 90 MB (e o .pt sai do Git);
    .pt pequenos voltam a ser versionados normalmente. Retorna os caminhos a adicionar."""
    repo = Path(repo)
    gi = repo / ".gitignore"
    linhas = gi.read_text(encoding="utf-8").splitlines() if gi.exists() else []
    grandes_ini = "# HEILO: modelos grandes vão em partes (o .pt é remontado no PC)"
    adicionar: List[str] = []
    for pt in sorted((repo / "models").rglob("*.pt")):
        rel = pt.relative_to(repo).as_posix()
        if pt.stat().st_size > LIMITE_GITHUB:
            man = dividir_em_partes(pt)
            subprocess.run(["git", "-C", str(repo), "rm", "--cached", "-q", "--ignore-unmatch", rel])
            if rel not in linhas:
                if grandes_ini not in linhas:
                    linhas.append(grandes_ini)
                linhas.append(rel)
            adicionar += [rel + ".partes.json"] + [pt.parent.relative_to(repo).as_posix() + "/" + p
                                                   for p in man["partes"]]
            log(f"{rel}: {pt.stat().st_size / 1e6:.0f} MB → {len(man['partes'])} partes")
        else:
            for velha in pt.parent.glob(pt.name + ".parte*"):
                subprocess.run(["git", "-C", str(repo), "rm", "-q", "--ignore-unmatch",
                                velha.relative_to(repo).as_posix()])
                velha.unlink(missing_ok=True)
            mp = Path(str(pt) + ".partes.json")
            if mp.exists():
                subprocess.run(["git", "-C", str(repo), "rm", "-q", "--ignore-unmatch", mp.relative_to(repo).as_posix()])
                mp.unlink(missing_ok=True)
            if rel in linhas:
                linhas.remove(rel)
            adicionar.append(rel)
    gi.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    return adicionar + [".gitignore"]
