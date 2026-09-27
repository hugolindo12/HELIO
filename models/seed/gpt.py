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
from typing import Dict, List, Optional, Tuple

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
from heilo.models.seed.tokenizer import BYTE, from_dict as tokenizer_from_dict

# ----------------------------------------------------------------- tokenizer
# Constantes do tokenizer em bytes (v0.x). Modelos com BPE (v1.x) usam o
# tokenizer salvo no próprio checkpoint — todas as funções aceitam `tok`.
USUARIO, HEILO, FIM, DOC = BYTE.usuario, BYTE.heilo, BYTE.fim, BYTE.doc
VOCAB_SIZE = BYTE.vocab_size


def encode(texto: str, tok=BYTE) -> List[int]:
    return tok.encode(texto)


def decode(ids: List[int], tok=BYTE) -> str:
    return tok.decode(ids)


def encode_chat(messages: List[Dict], fechar_ultimo: bool = True, tok=BYTE) -> List[int]:
    """<USUARIO>oi<FIM><HEILO>Oi! Tudo bem?<FIM>..."""
    tags = {"user": tok.usuario, "assistant": tok.heilo}
    ids: List[int] = []
    for i, m in enumerate(messages):
        tag = tags.get(m.get("role"))
        if tag is None:
            continue
        ids.append(tag)
        ids += tok.encode(m.get("content", ""))
        if fechar_ultimo or i < len(messages) - 1:
            ids.append(tok.fim)
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
    # "gpt2" (v0.x/v1.x) ou "moderna" (v2.x, técnicas do estilo Gemma: RoPE, RMSNorm,
    # GeGLU, GQA e QK-norm). Checkpoints antigos não têm o campo → "gpt2".
    arquitetura: str = "gpt2"
    n_kv_head: int = 0            # GQA: nº de cabeças de chave/valor (0 = igual a n_head)
    rope_base: float = 10000.0


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

    # ---------------------------------------------- arquitetura "moderna" (v2.x)
    class RMSNorm(nn.Module):
        """Normalização pela raiz da média dos quadrados (mais simples e estável que LayerNorm)."""

        def __init__(self, d: int, eps: float = 1e-6):
            super().__init__()
            self.eps = eps
            self.peso = nn.Parameter(torch.ones(d))

        def forward(self, x):
            x32 = x.float()
            x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
            return (x32 * self.peso.float()).type_as(x)

    def _rope_tabelas(T: int, hd: int, base: float, dev):
        inv = 1.0 / (base ** (torch.arange(0, hd, 2, device=dev).float() / hd))
        ang = torch.outer(torch.arange(T, device=dev).float(), inv)       # (T, hd/2)
        return torch.cos(ang), torch.sin(ang)

    def _rope(x, cos, sin):
        """RoPE: gira pares de dimensões conforme a posição (x: B, H, T, hd)."""
        x1, x2 = x[..., 0::2], x[..., 1::2]
        c, s_ = cos[None, None].to(x.dtype), sin[None, None].to(x.dtype)
        return torch.stack((x1 * c - x2 * s_, x1 * s_ + x2 * c), dim=-1).flatten(-2)

    class BlocoModerno(nn.Module):
        def __init__(self, c: GPTConfig):
            super().__init__()
            self.n_head = c.n_head
            self.n_kv = c.n_kv_head or c.n_head
            assert c.n_head % self.n_kv == 0, "n_head precisa ser múltiplo de n_kv_head"
            self.hd = c.n_embd // c.n_head
            self.norm1 = RMSNorm(c.n_embd)
            self.q = nn.Linear(c.n_embd, c.n_head * self.hd, bias=False)
            self.kv = nn.Linear(c.n_embd, 2 * self.n_kv * self.hd, bias=False)
            self.o = nn.Linear(c.n_head * self.hd, c.n_embd, bias=False)
            self.q_norm, self.k_norm = RMSNorm(self.hd), RMSNorm(self.hd)   # QK-norm
            self.norm2 = RMSNorm(c.n_embd)
            oculto = int(round(c.n_embd * 8 / 3 / 64)) * 64                  # GeGLU ~ mesmo nº de parâmetros
            self.porta = nn.Linear(c.n_embd, oculto, bias=False)
            self.cima = nn.Linear(c.n_embd, oculto, bias=False)
            self.baixo = nn.Linear(oculto, c.n_embd, bias=False)
            self.dropout = c.dropout
            self.drop = nn.Dropout(c.dropout)

        def forward(self, x, cos, sin):
            B, T, C = x.shape
            h = self.norm1(x)
            q = self.q(h).view(B, T, self.n_head, self.hd).transpose(1, 2)
            k, v = self.kv(h).view(B, T, 2, self.n_kv, self.hd).unbind(2)
            k, v = k.transpose(1, 2), v.transpose(1, 2)
            q, k = _rope(self.q_norm(q), cos, sin), _rope(self.k_norm(k), cos, sin)
            if self.n_kv != self.n_head:                                    # GQA
                rep = self.n_head // self.n_kv
                k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                               dropout_p=self.dropout if self.training else 0.0)
            x = x + self.o(y.transpose(1, 2).contiguous().view(B, T, C))
            h = self.norm2(x)
            return x + self.drop(self.baixo(F.gelu(self.porta(h), approximate="tanh") * self.cima(h)))

    class MiniGPT(nn.Module):
        def __init__(self, c: GPTConfig):
            super().__init__()
            self.cfg = c
            self.moderna = c.arquitetura == "moderna"
            self.tok = nn.Embedding(c.vocab_size, c.n_embd)
            self.drop = nn.Dropout(c.dropout)
            if self.moderna:
                self.blocos = nn.ModuleList([BlocoModerno(c) for _ in range(c.n_layer)])
                self.ln = RMSNorm(c.n_embd)
                self._rope_cache = None
            else:
                self.pos = nn.Embedding(c.block_size, c.n_embd)
                self.blocos = nn.ModuleList([Bloco(c) for _ in range(c.n_layer)])
                self.ln = nn.LayerNorm(c.n_embd)
            self.head = nn.Linear(c.n_embd, c.vocab_size, bias=False)
            self.head.weight = self.tok.weight  # weight tying
            self.apply(self._init)
            if self.moderna:   # projeções de saída menores: treino mais estável com muitas camadas
                for b in self.blocos:
                    for w in (b.o.weight, b.baixo.weight):
                        nn.init.normal_(w, mean=0.0, std=0.02 / math.sqrt(2 * c.n_layer))

        def _rope_para(self, T, dev):
            if self._rope_cache is None or self._rope_cache[0].size(0) < T or self._rope_cache[0].device != dev:
                hd = self.cfg.n_embd // self.cfg.n_head
                self._rope_cache = _rope_tabelas(max(T, self.cfg.block_size), hd, self.cfg.rope_base, dev)
            cos, sin = self._rope_cache
            return cos[:T], sin[:T]

        @staticmethod
        def _init(m):
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

        def forward(self, idx, alvo=None):
            B, T = idx.shape
            if self.moderna:
                cos, sin = self._rope_para(T, idx.device)
                x = self.drop(self.tok(idx))
                for b in self.blocos:
                    x = b(x, cos, sin)
            else:
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
                  top_k: int = 40, parar_em=(FIM, USUARIO), penalidade: float = 1.0,
                  sem_repetir: int = 0) -> List[int]:
            """penalidade > 1 desencoraja repetir tokens já gerados; sem_repetir=n proíbe
            repetir um n-grama já gerado ("a soma de a soma de a soma..."). Padrões = 1.0/0
            (desligados) para a avaliação continuar comparável entre versões."""
            self.eval()
            dev = next(self.parameters()).device
            x = torch.tensor([ids[-self.cfg.block_size:]], dtype=torch.long, device=dev)
            novos: List[int] = []
            for _ in range(max_novos):
                logits, _ = self(x[:, -self.cfg.block_size:])
                logits = logits[:, -1, :]
                if penalidade != 1.0 and novos:
                    usados = torch.tensor(sorted(set(novos)), device=dev)
                    lv = logits[0, usados]
                    logits[0, usados] = torch.where(lv > 0, lv / penalidade, lv * penalidade)
                if sem_repetir and len(novos) >= sem_repetir:
                    pref = tuple(novos[-(sem_repetir - 1):]) if sem_repetir > 1 else ()
                    proibidos = {novos[i + sem_repetir - 1] for i in range(len(novos) - sem_repetir + 1)
                                 if tuple(novos[i:i + sem_repetir - 1]) == pref}
                    if proibidos:
                        logits[0, list(proibidos)] = -float("inf")
                logits = logits / max(temperatura, 1e-4)
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


def salvar(modelo, info: Dict, arquivo: Path = None, meia_precisao: bool = False) -> Path:
    arquivo = Path(arquivo or SEED_WEIGHTS)
    arquivo.parent.mkdir(parents=True, exist_ok=True)
    tmp = arquivo.with_suffix(".tmp")
    estado = modelo.state_dict()
    if meia_precisao:   # metade do tamanho em disco; volta a float32 ao carregar
        estado = {k: (v.half() if v.is_floating_point() else v) for k, v in estado.items()}
    tok = getattr(modelo, "tokenizer", BYTE)
    ck = {"config": asdict(modelo.cfg), "modelo": estado, "info": info}
    if tok.to_dict():
        ck["tokenizer"] = tok.to_dict()
    torch.save(ck, tmp)
    tmp.replace(arquivo)
    return arquivo


# ------------------------------------------------ arquivos grandes em partes
# O GitHub não aceita arquivos > 100 MB. Modelos maiores (ex.: HEILO Aurora, ~300 MB)
# vão para o repositório em partes de 90 MB + um manifesto com o sha256; o PC remonta
# o .pt na primeira vez que for usado (e de novo se as partes mudarem).
TAMANHO_PARTE = 90 * 1024 * 1024


def _manifesto_partes(arquivo: Path) -> Path:
    return Path(str(arquivo) + ".partes.json")


def tem_modelo(arquivo: Path) -> bool:
    arquivo = Path(arquivo)
    return arquivo.exists() or _manifesto_partes(arquivo).exists()


def dividir_em_partes(arquivo: Path, tamanho: int = TAMANHO_PARTE) -> Dict:
    import hashlib
    import json as _json
    arquivo = Path(arquivo)
    for velha in arquivo.parent.glob(arquivo.name + ".parte*"):
        velha.unlink()
    h, partes = hashlib.sha256(), []
    with open(arquivo, "rb") as f:
        while True:
            bloco = f.read(tamanho)
            if not bloco:
                break
            h.update(bloco)
            nome = f"{arquivo.name}.parte{len(partes):03d}"
            (arquivo.parent / nome).write_bytes(bloco)
            partes.append(nome)
    man = {"arquivo": arquivo.name, "bytes": arquivo.stat().st_size, "sha256": h.hexdigest(), "partes": partes}
    _manifesto_partes(arquivo).write_text(_json.dumps(man, indent=1), encoding="utf-8")
    return man


def montar_partes(arquivo: Path) -> bool:
    """Remonta o .pt a partir das partes, se houver manifesto e o .pt estiver ausente/desatualizado."""
    import hashlib
    import json as _json
    arquivo = Path(arquivo)
    mp = _manifesto_partes(arquivo)
    if not mp.exists():
        return arquivo.exists()
    man = _json.loads(mp.read_text(encoding="utf-8"))
    marca = Path(str(arquivo) + ".montado")
    if arquivo.exists() and marca.exists() and marca.read_text().strip() == man["sha256"]:
        return True
    h, tmp = hashlib.sha256(), arquivo.with_suffix(".montando")
    with open(tmp, "wb") as out:
        for nome in man["partes"]:
            bloco = (arquivo.parent / nome).read_bytes()
            h.update(bloco)
            out.write(bloco)
    if h.hexdigest() != man["sha256"]:
        tmp.unlink()
        raise ValueError(f"partes de {arquivo.name} corrompidas (sha256 não confere)")
    tmp.replace(arquivo)
    marca.write_text(man["sha256"])
    return True


def carregar(arquivo: Path = None, dispositivo: str = None):
    """Retorna (modelo, info) ou (None, {}) se ainda não existe cérebro treinado."""
    arquivo = Path(arquivo or SEED_WEIGHTS)
    if TORCH_OK and _manifesto_partes(arquivo).exists():
        montar_partes(arquivo)
    if not TORCH_OK or not arquivo.exists():
        return None, {}
    dispositivo = dispositivo or _dispositivo()
    ck = torch.load(arquivo, map_location=dispositivo, weights_only=False)
    modelo = MiniGPT(GPTConfig(**ck["config"])).to(dispositivo)
    modelo.load_state_dict({k: v.float() if v.is_floating_point() else v
                            for k, v in ck["modelo"].items()})
    modelo.tokenizer = tokenizer_from_dict(ck.get("tokenizer"))
    modelo.eval()
    return modelo, ck.get("info", {})


# ------------------------------------------------------------------- treino
def montar_fluxo(dataset: List[Dict], corpus: str = "", repetir_chat: int = 1,
                 semente: int = 0, tok=BYTE) -> List[int]:
    """Transforma dataset + documentos num fluxo único de tokens."""
    rnd = random.Random(semente)
    pedacos: List[List[int]] = []
    for _ in range(max(1, repetir_chat)):
        for ex in dataset:
            pedacos.append(encode_chat(ex["messages"], tok=tok))
    for doc in (corpus or "").split("\n\n"):
        doc = doc.strip()
        if len(doc) > 20:
            pedacos.append([tok.doc] + tok.encode(doc) + [tok.fim])
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
            avaliacao: Optional[List[Dict]] = None, tok=None) -> Dict:
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

    modelo, info = (None, {}) if novo else carregar(arquivo, dispositivo)
    if modelo is None:
        tok = tok or BYTE
        cfg = config or GPTConfig()
        cfg.vocab_size = tok.vocab_size
        modelo = MiniGPT(cfg).to(dispositivo)
        modelo.tokenizer = tok
        info = {"passos_totais": 0, "historico": []}
        log(f"[HEILO Seed] Modelo NOVO (do zero): {modelo.n_params()/1e6:.2f}M parâmetros")
    else:
        log(f"[HEILO Seed] Continuando treino: {info.get('passos_totais', 0)} passos já feitos")
    tok = modelo.tokenizer
    fluxo = montar_fluxo(dataset, corpus, repetir_chat=repetir_chat, semente=semente, tok=tok)
    dados = torch.tensor(fluxo, dtype=torch.long)

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
        ids = encode_chat(ex["messages"], tok=getattr(modelo, "tokenizer", BYTE))[-(T + 1):]
        if len(ids) < 2:
            continue
        x = torch.tensor([ids[:-1]], device=dev)
        y = torch.tensor([ids[1:]], device=dev)
        logits, _ = modelo(x)
        perdas.append(F.cross_entropy(logits[0], y[0]).item())
    modelo.train()
    return sum(perdas) / len(perdas) if perdas else None


@torch.no_grad() if TORCH_OK else (lambda f: f)
def perda_condicional(modelo, pergunta: str, resposta: str) -> Optional[Tuple[float, int]]:
    """Perda (cross-entropy, soma) dos tokens da RESPOSTA + <FIM>, condicionada à pergunta.

    Base da perplexidade: exp(soma / n_tokens). Mede quão bem o modelo prevê
    a resposta de referência — sem depender de amostragem.
    """
    if not TORCH_OK:
        return None
    tok = getattr(modelo, "tokenizer", BYTE)
    prefixo = encode_chat([{"role": "user", "content": pergunta}], tok=tok) + [tok.heilo]
    alvo = tok.encode(resposta) + [tok.fim]
    ids = (prefixo + alvo)[-(modelo.cfg.block_size + 1):]
    n_alvo = min(len(alvo), len(ids) - 1)
    was_training = modelo.training
    modelo.eval()
    dev = next(modelo.parameters()).device
    x = torch.tensor([ids[:-1]], device=dev)
    y = torch.tensor([ids[1:]], device=dev)
    logits, _ = modelo(x)
    perdas = F.cross_entropy(logits[0], y[0], reduction="none")[-n_alvo:]
    if was_training:
        modelo.train()
    return float(perdas.sum().item()), int(n_alvo)


# ---------------------------------------------------------------- conversa
class MiniGPTChat:
    """Interface de conversa em cima do HEILO Seed treinado."""

    def __init__(self, arquivo: Path = None):
        self.modelo, self.info = carregar(arquivo)

    @property
    def pronto(self) -> bool:
        return self.modelo is not None

    def responder(self, messages: List[Dict], temperatura: float = 0.7,
                  max_novos: int = 240, top_k: int = 40, penalidade: float = 1.0,
                  sem_repetir: int = 0) -> str:
        if not self.pronto:
            return ""
        conversa = [m for m in messages if m.get("role") in ("user", "assistant")][-8:]
        bloco = self.modelo.cfg.block_size
        limite = max(bloco // 2, bloco - 64)  # sempre sobra espaço para a resposta
        # monta o contexto de trás para frente, com MENSAGENS INTEIRAS (não corta no meio)
        tok = getattr(self.modelo, "tokenizer", BYTE)
        ids: List[int] = [tok.heilo]
        for m in reversed(conversa):
            pedaco = encode_chat([m], tok=tok)
            if len(ids) + len(pedaco) > limite:
                if len(ids) == 1:   # a última mensagem sozinha não cabe: mantém o fim dela com a tag
                    tag = pedaco[:1]
                    ids = tag + pedaco[1:][-(limite - 2):] + ids
                break
            ids = pedaco + ids
        novos = self.modelo.gerar(ids, max_novos=max_novos, temperatura=temperatura, top_k=top_k,
                                  parar_em=(tok.fim, tok.usuario), penalidade=penalidade,
                                  sem_repetir=sem_repetir)
        return tok.decode(novos).strip()

    def perda_resposta(self, pergunta: str, resposta: str) -> Optional[Tuple[float, int]]:
        """(soma da perda, nº de tokens) SÓ nos tokens da resposta, dado a pergunta."""
        return perda_condicional(self.modelo, pergunta, resposta) if self.pronto else None
