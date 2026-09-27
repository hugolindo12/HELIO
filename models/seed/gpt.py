"""
HEILO Seed — modelo próprio da HEILO, treinado do zero (antes chamado "mini-GPT").

Nasce sem saber nada (pesos aleatórios) e aprende apenas com o dataset HEILO
(exemplos aprovados em data/approved → data/datasets). Não usa nem copia pesos
de nenhum modelo externo.

- Tokenizer em bytes (UTF-8): 256 bytes + 4 tokens especiais. Não precisa de
  vocabulário externo e nunca "cresce", então o treino pode continuar sempre
  do último checkpoint (aprendizado incremental).
- Transformer decoder (estilo GPT) pequeno, treinável na CPU.

Uso (pela plataforma):
    python -m heilo.main treinar --passos 3000
    python -m heilo.main cli        →  /cerebro seed
"""
from __future__ import annotations

import argparse
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_OK = True
except ImportError:  # o resto da HEILO funciona sem torch
    torch = None
    nn = None
    F = None
    TORCH_OK = False

from heilo.models.seed import SEED_WEIGHTS

# ----------------------------------------------------------------- tokenizer
USUARIO, HEILO, FIM, DOC = 256, 257, 258, 259
VOCAB_SIZE = 260
_TAG = {"user": USUARIO, "assistant": HEILO}


def encode(texto: str) -> List[int]:
    return list(texto.encode("utf-8"))


def decode(ids: List[int]) -> str:
    return bytes(i for i in ids if i < 256).decode("utf-8", errors="ignore")


def encode_chat(messages: List[Dict], fechar_ultimo: bool = True) -> List[int]:
    """<USUARIO>oi<FIM><HEILO>Oi! Tudo bem?<FIM>..."""
    ids: List[int] = []
    for i, m in enumerate(messages):
        tag = _TAG.get(m.get("role"))
        if tag is None:
            continue
        ids.append(tag)
        ids += encode(m.get("content", ""))
        if fechar_ultimo or i < len(messages) - 1:
            ids.append(FIM)
    return ids


# ------------------------------------------------------------------- modelo
@dataclass
class GPTConfig:
    block_size: int = 256
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.1
    vocab_size: int = VOCAB_SIZE


if TORCH_OK:

    class Bloco(nn.Module):
        def __init__(self, c: GPTConfig):
            super().__init__()
            self.ln1 = nn.LayerNorm(c.n_embd)
            self.qkv = nn.Linear(c.n_embd, 3 * c.n_embd)
            self.proj = nn.Linear(c.n_embd, c.n_embd)
            self.ln2 = nn.LayerNorm(c.n_embd)
            self.mlp = nn.Sequential(
                nn.Linear(c.n_embd, 4 * c.n_embd),
                nn.GELU(),
                nn.Linear(4 * c.n_embd, c.n_embd),
                nn.Dropout(c.dropout),
            )
            self.n_head = c.n_head
            self.dropout = c.dropout

        def forward(self, x):
            B, T, C = x.shape
            q, k, v = self.qkv(self.ln1(x)).split(C, dim=2)
            q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
            )
            y = y.transpose(1, 2).contiguous().view(B, T, C)
            x = x + self.proj(y)
            return x + self.mlp(self.ln2(x))

    class MiniGPT(nn.Module):
        def __init__(self, c: GPTConfig):
            super().__init__()
            self.cfg = c
            self.tok = nn.Embedding(c.vocab_size, c.n_embd)
            self.pos = nn.Embedding(c.block_size, c.n_embd)
            self.drop = nn.Dropout(c.dropout)
            self.blocos = nn.ModuleList([Bloco(c) for _ in range(c.n_layer)])
            self.ln = nn.LayerNorm(c.n_embd)
            self.head = nn.Linear(c.n_embd, c.vocab_size, bias=False)
            self.head.weight = self.tok.weight  # weight tying
            self.apply(self._init)

        @staticmethod
        def _init(m):
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

        def forward(self, idx, alvo=None):
            B, T = idx.shape
            pos = torch.arange(T, device=idx.device)
            x = self.drop(self.tok(idx) + self.pos(pos))
            for b in self.blocos:
                x = b(x)
            logits = self.head(self.ln(x))
            loss = None
            if alvo is not None:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), alvo.view(-1))
            return logits, loss

        def n_params(self) -> int:
            return sum(p.numel() for p in self.parameters())

        @torch.no_grad()
        def gerar(self, ids: List[int], max_novos: int = 200, temperatura: float = 0.8,
                  top_k: int = 40, parar_em=(FIM, USUARIO)) -> List[int]:
            self.eval()
            dev = next(self.parameters()).device
            x = torch.tensor([ids[-self.cfg.block_size:]], dtype=torch.long, device=dev)
            novos: List[int] = []
            for _ in range(max_novos):
                logits, _ = self(x[:, -self.cfg.block_size:])
                logits = logits[:, -1, :] / max(temperatura, 1e-4)
                if top_k:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("inf")
                prox = torch.multinomial(F.softmax(logits, dim=-1), 1)
                t = int(prox.item())
                if t in parar_em:
                    break
                novos.append(t)
                x = torch.cat([x, prox], dim=1)
            return novos


# ----------------------------------------------------------- checkpoint I/O
def _dispositivo() -> str:
    return "cuda" if TORCH_OK and torch.cuda.is_available() else "cpu"


def salvar(modelo, info: Dict, arquivo: Path = None) -> Path:
    arquivo = Path(arquivo or SEED_WEIGHTS)
    arquivo.parent.mkdir(parents=True, exist_ok=True)
    tmp = arquivo.with_suffix(".tmp")
    torch.save({"config": asdict(modelo.cfg), "modelo": modelo.state_dict(), "info": info}, tmp)
    tmp.replace(arquivo)
    return arquivo


def carregar(arquivo: Path = None, dispositivo: str = None):
    """Retorna (modelo, info) ou (None, {}) se ainda não existe cérebro treinado."""
    arquivo = Path(arquivo or SEED_WEIGHTS)
    if not TORCH_OK or not arquivo.exists():
        return None, {}
    dispositivo = dispositivo or _dispositivo()
    ck = torch.load(arquivo, map_location=dispositivo, weights_only=False)
    modelo = MiniGPT(GPTConfig(**ck["config"])).to(dispositivo)
    modelo.load_state_dict(ck["modelo"])
    modelo.eval()
    return modelo, ck.get("info", {})


# ------------------------------------------------------------------- treino
def montar_fluxo(dataset: List[Dict], corpus: str = "", repetir_chat: int = 1,
                 semente: int = 0) -> List[int]:
    """Transforma dataset + documentos num fluxo único de tokens."""
    rnd = random.Random(semente)
    pedacos: List[List[int]] = []
    for _ in range(max(1, repetir_chat)):
        for ex in dataset:
            pedacos.append(encode_chat(ex["messages"]))
    for doc in (corpus or "").split("\n\n"):
        doc = doc.strip()
        if len(doc) > 20:
            pedacos.append([DOC] + encode(doc) + [FIM])
    rnd.shuffle(pedacos)
    fluxo: List[int] = []
    for p in pedacos:
        fluxo += p
    return fluxo


def treinar(passos: int = 2000, lote: int = 16, lr: float = 3e-4, novo: bool = False,
            config: Optional[GPTConfig] = None, arquivo: Path = None,
            dataset_file: Path = None, corpus_file: Path = None,
            repetir_chat: int = 4, log_cada: int = 100, salvar_cada: int = 500,
            semente: int = 1337, log=print,
            avaliacao: Optional[List[Dict]] = None) -> Dict:
    """Treina (ou continua treinando) o HEILO Seed com um dataset HEILO."""
    if not TORCH_OK:
        raise RuntimeError("PyTorch não está instalado. Rode: pip install torch")
    from heilo.training.records import read_jsonl

    torch.manual_seed(semente)
    random.seed(semente)
    dispositivo = _dispositivo()
    arquivo = Path(arquivo or SEED_WEIGHTS)

    if dataset_file is None:
        raise RuntimeError("Informe o dataset HEILO (use heilo.training.TrainingPipeline).")
    dataset = read_jsonl(Path(dataset_file))
    corpus_path = Path(corpus_file) if corpus_file else None
    corpus = corpus_path.read_text(encoding="utf-8") if corpus_path and corpus_path.exists() else ""
    if not dataset and not corpus:
        raise RuntimeError("Dataset vazio. Rode antes: python -m heilo.main aprender")

    fluxo = montar_fluxo(dataset, corpus, repetir_chat=repetir_chat, semente=semente)
    dados = torch.tensor(fluxo, dtype=torch.long)

    modelo, info = (None, {}) if novo else carregar(arquivo, dispositivo)
    if modelo is None:
        modelo = MiniGPT(config or GPTConfig()).to(dispositivo)
        info = {"passos_totais": 0, "historico": []}
        log(f"[HEILO Seed] Modelo NOVO (do zero): {modelo.n_params()/1e6:.2f}M parâmetros")
    else:
        log(f"[HEILO Seed] Continuando treino: {info.get('passos_totais', 0)} passos já feitos")

    T = modelo.cfg.block_size
    if len(dados) <= T + 1:  # dataset minúsculo: repete até caber uma janela
        dados = dados.repeat(math.ceil((T + 2) / max(1, len(dados))))
    log(f"[HEILO Seed] {len(dataset)} exemplos de chat, {len(fluxo):,} tokens, dispositivo={dispositivo}")

    opt = torch.optim.AdamW(modelo.parameters(), lr=lr, weight_decay=0.1)
    aquecimento = min(100, passos // 10)

    def lr_em(p):
        if p < aquecimento:
            return lr * (p + 1) / aquecimento
        prog = (p - aquecimento) / max(1, passos - aquecimento)
        return lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))

    modelo.train()
    base = info.get("passos_totais", 0)
    inicio, media = time.time(), None
    for p in range(passos):
        for g in opt.param_groups:
            g["lr"] = lr_em(p)
        ix = torch.randint(len(dados) - T - 1, (lote,))
        x = torch.stack([dados[i:i + T] for i in ix]).to(dispositivo)
        y = torch.stack([dados[i + 1:i + 1 + T] for i in ix]).to(dispositivo)
        _, loss = modelo(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
        opt.step()
        media = loss.item() if media is None else 0.95 * media + 0.05 * loss.item()
        if (p + 1) % log_cada == 0 or p == 0:
            log(f"  passo {p + 1}/{passos}  perda={media:.3f}  ({time.time() - inicio:.0f}s)")
        if salvar_cada and (p + 1) % salvar_cada == 0:
            info["passos_totais"] = base + p + 1
            salvar(modelo, info, arquivo)

    perda_val = avaliar(modelo, avaliacao) if avaliacao else None
    info["passos_totais"] = base + passos
    info.setdefault("historico", []).append({
        "data": time.strftime("%Y-%m-%d %H:%M"),
        "passos": passos,
        "perda": round(media or 0.0, 4),
        "perda_validacao": None if perda_val is None else round(perda_val, 4),
        "exemplos": len(dataset),
    })
    salvar(modelo, info, arquivo)
    extra = f", validação {perda_val:.3f}" if perda_val is not None else ""
    log(f"[HEILO Seed] Salvo em {arquivo} (perda final {media:.3f}{extra})")
    return {"perda": media, "perda_validacao": perda_val,
            "passos_totais": info["passos_totais"], "arquivo": str(arquivo)}


@torch.no_grad() if TORCH_OK else (lambda f: f)
def avaliar(modelo, exemplos: List[Dict]) -> Optional[float]:
    """Perda média (cross-entropy por token) em exemplos NÃO usados no treino.

    Métrica real e reprodutível: quanto menor, melhor o Seed prevê conversas
    aprovadas que ele nunca viu. Retorna None se não houver exemplos.
    """
    if not TORCH_OK or not exemplos:
        return None
    modelo.eval()
    dev = next(modelo.parameters()).device
    T = modelo.cfg.block_size
    perdas = []
    for ex in exemplos:
        ids = encode_chat(ex["messages"])[-(T + 1):]
        if len(ids) < 2:
            continue
        x = torch.tensor([ids[:-1]], device=dev)
        y = torch.tensor([ids[1:]], device=dev)
        logits, _ = modelo(x)
        perdas.append(F.cross_entropy(logits[0], y[0]).item())
    modelo.train()
    return sum(perdas) / len(perdas) if perdas else None


# ---------------------------------------------------------------- conversa
class MiniGPTChat:
    """Interface de conversa em cima do HEILO Seed treinado."""

    def __init__(self, arquivo: Path = None):
        self.modelo, self.info = carregar(arquivo)

    @property
    def pronto(self) -> bool:
        return self.modelo is not None

    def responder(self, messages: List[Dict], temperatura: float = 0.7,
                  max_novos: int = 240) -> str:
        if not self.pronto:
            return ""
        conversa = [m for m in messages if m.get("role") in ("user", "assistant")][-8:]
        ids = encode_chat(conversa) + [HEILO]
        # garante espaço para a resposta dentro da janela
        bloco = self.modelo.cfg.block_size
        limite = max(bloco // 2, bloco - 64)  # sempre sobra espaço para a resposta
        ids = ids[-limite:]
        novos = self.modelo.gerar(ids, max_novos=max_novos, temperatura=temperatura)
        return decode(novos).strip()
