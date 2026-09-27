"""
Tokenizers do HEILO Seed.

- ByteTokenizer: bytes UTF-8 + 4 especiais (v0.x). Não precisa de treino.
- BPETokenizer:  BPE próprio da HEILO, TREINADO pela HEILO no corpus dela
                 (não é o tokenizer do Qwen nem de nenhum outro modelo).
                 Lê palavras em pedaços → o modelo enxerga muito mais texto
                 na mesma janela de contexto.

O tokenizer viaja DENTRO do checkpoint (chave "tokenizer"), então um arquivo
.pt basta para carregar o modelo.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

ESPECIAIS = ["<|usuario|>", "<|heilo|>", "<|fim|>", "<|doc|>", "<|pad|>"]


class ByteTokenizer:
    tipo = "byte"
    usuario, heilo, fim, doc = 256, 257, 258, 259
    pad = 258
    vocab_size = 260

    def encode(self, texto: str) -> List[int]:
        return list((texto or "").encode("utf-8"))

    def decode(self, ids: List[int]) -> str:
        return bytes(i for i in ids if i < 256).decode("utf-8", errors="ignore")

    def to_dict(self) -> Optional[Dict]:
        return None          # checkpoints antigos: ausência = byte


class BPETokenizer:
    tipo = "bpe"

    def __init__(self, tok):
        self.tok = tok
        ids = [tok.token_to_id(t) for t in ESPECIAIS]
        if any(i is None for i in ids):
            raise ValueError("tokenizer BPE sem os tokens especiais da HEILO")
        self.usuario, self.heilo, self.fim, self.doc, self.pad = ids
        self.vocab_size = tok.get_vocab_size()

    def encode(self, texto: str) -> List[int]:
        return self.tok.encode(texto or "", add_special_tokens=False).ids

    def encode_lote(self, textos: List[str]) -> List[List[int]]:
        return [e.ids for e in self.tok.encode_batch(textos, add_special_tokens=False)]

    def decode(self, ids: List[int]) -> str:
        return self.tok.decode([i for i in ids if i not in (self.usuario, self.heilo, self.fim,
                                                            self.doc, self.pad)])

    def to_dict(self) -> Dict:
        return {"tipo": "bpe", "json": self.tok.to_str()}

    @classmethod
    def from_json(cls, texto_json: str) -> "BPETokenizer":
        from tokenizers import Tokenizer
        return cls(Tokenizer.from_str(texto_json))

    @classmethod
    def treinar(cls, textos: Iterable[str], vocab_size: int = 16000,
                min_frequency: int = 2) -> "BPETokenizer":
        """Treina um BPE byte-level do zero (sem vocabulário externo)."""
        from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
        tok = Tokenizer(models.BPE())
        tok.normalizer = normalizers.NFC()
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size, min_frequency=min_frequency, special_tokens=ESPECIAIS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
        tok.train_from_iterator(textos, trainer=trainer)
        return cls(tok)


BYTE = ByteTokenizer()


def from_dict(d: Optional[Dict]):
    if not d:
        return BYTE
    if d.get("tipo") == "bpe":
        return BPETokenizer.from_json(d["json"])
    raise ValueError(f"tokenizer desconhecido: {d.get('tipo')}")
