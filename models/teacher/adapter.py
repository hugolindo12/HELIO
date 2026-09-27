"""
Adaptador do HEILO Teacher.

Genérico para qualquer modelo causal do Hugging Face (+ adaptador LoRA opcional).
O modelo real, a licença e a finalidade ficam em teacher.json — trocar de professor
é trocar esse arquivo, sem mexer no Core.

Dependências (só do Teacher): torch, transformers, peft (se houver LoRA).
Offline: defina HEILO_OFFLINE=1 (ou HF_HUB_OFFLINE=1) para usar só o cache local.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from heilo.models.base import ModelAdapter, ModelCard

TEACHER_DIR = Path(__file__).resolve().parent
TEACHER_META = TEACHER_DIR / "teacher.json"


def load_metadata(path: Optional[Path] = None) -> Dict:
    path = Path(path or TEACHER_META)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


class TeacherAdapter(ModelAdapter):
    key = "teacher"

    def __init__(self, metadata_path: Optional[Path] = None):
        self.meta_path = Path(metadata_path or TEACHER_META)
        self.meta = load_metadata(self.meta_path)
        self._model = None
        self._tok = None
        self._dev = "cpu"
        self._erro: Optional[str] = None

    # -------------------------------------------------------------- metadados
    @property
    def base_model(self) -> str:
        return self.meta.get("base_model", "")

    @property
    def adapter_dir(self) -> Optional[Path]:
        rel = self.meta.get("adapter_dir")
        return (self.meta_path.parent / rel) if rel else None

    def _has_adapter(self) -> bool:
        d = self.adapter_dir
        return bool(d and (d / "adapter_config.json").exists())

    def card(self) -> ModelCard:
        m = self.meta
        return ModelCard(
            name=m.get("name", "HEILO Teacher"),
            role="teacher",
            provider=m.get("provider", "external"),
            base_model=self.base_model,
            adapter=("LoRA" if self._has_adapter() else "nenhum"),
            license=m.get("license", ""),
            purpose=m.get("purpose", "training_teacher"),
            status=m.get("status", "temporary"),
            origin=m.get("origin", ""),
            version=m.get("version", ""),
            extra={"license_url": m.get("license_url", ""), "added": m.get("added", "")},
        )

    # ---------------------------------------------------------- disponibilidade
    def availability(self) -> Tuple[bool, str]:
        if not self.meta or not self.base_model:
            return False, f"metadados do Teacher ausentes/inválidos ({self.meta_path})"
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError:
            return False, "dependências do Teacher ausentes (pip install torch transformers)"
        if self._has_adapter():
            try:
                import peft  # noqa: F401
            except ImportError:
                return False, "adaptador LoRA encontrado, mas 'peft' não está instalado"
        if self._erro:
            return False, f"falha ao carregar o Teacher: {self._erro}"
        return True, "ok"

    # ------------------------------------------------------------------ uso
    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        offline = os.getenv("HEILO_OFFLINE", "0") == "1" or os.getenv("HF_HUB_OFFLINE", "0") == "1"
        self._dev = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self._dev == "cuda" else torch.float32
        tok_src = (str(self.adapter_dir)
                   if self._has_adapter() and (self.adapter_dir / "tokenizer_config.json").exists()
                   else self.base_model)
        self._tok = AutoTokenizer.from_pretrained(tok_src, local_files_only=offline)
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model, dtype=dtype, local_files_only=offline).to(self._dev)
        if self._has_adapter():
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, str(self.adapter_dir)).merge_and_unload()
        model.eval()
        self._model = model

    def generate(self, messages: List[Dict[str, str]], system: Optional[str] = None,
                 temperatura: float = 0.7, max_novos: int = 256, **_) -> str:
        if not self.available:
            return ""
        try:
            self._load()
        except Exception as e:  # sem internet para baixar a base, sem memória...
            self._erro = str(e)
            print(f"[HEILO Teacher] não consegui carregar: {e}")
            return ""
        import torch

        conversa = ([{"role": "system", "content": system}] if system else []) + [
            m for m in messages if m.get("role") in ("user", "assistant")
        ][-10:]
        texto = self._tok.apply_chat_template(conversa, tokenize=False, add_generation_prompt=True)
        entrada = self._tok(texto, return_tensors="pt", add_special_tokens=False)["input_ids"].to(self._dev)
        with torch.no_grad():
            saida = self._model.generate(
                entrada,
                max_new_tokens=max_novos,
                do_sample=temperatura > 0,
                temperature=max(temperatura, 1e-3),
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self._tok.eos_token_id,
            )
        return self._tok.decode(saida[0][entrada.shape[1]:], skip_special_tokens=True).strip()

    def reload(self) -> None:
        self._model = None
        self._tok = None
        self._erro = None
