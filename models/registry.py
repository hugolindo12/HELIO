"""
Monta os adaptadores de modelo por nome.

É o ÚNICO lugar que importa os pacotes de modelo — e faz isso de forma preguiçosa.
Se a pasta models/teacher/ for apagada, o Teacher vira um UnavailableAdapter e o
resto da HEILO segue funcionando.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from heilo.models.base import ModelAdapter, UnavailableAdapter


def build_seed(weights: Optional[Path] = None) -> ModelAdapter:
    try:
        from heilo.models.seed.adapter import SeedAdapter
    except ImportError as e:
        return UnavailableAdapter("seed", "HEILO Seed", f"pacote do Seed ausente: {e}")
    return SeedAdapter(weights=weights)


def build_teacher(enabled: bool = True, metadata_path: Optional[Path] = None) -> ModelAdapter:
    if not enabled:
        return UnavailableAdapter("teacher", "HEILO Teacher",
                                  "desativado (teacher_enabled=false)")
    import os
    if os.getenv("HEILO_TEACHER_BACKEND", "").lower() == "ollama":
        try:
            from heilo.models.teacher.ollama_adapter import OllamaTeacherAdapter
            return OllamaTeacherAdapter()
        except ImportError as e:
            return UnavailableAdapter("teacher", "HEILO Teacher", f"adaptador do Ollama ausente: {e}")
    try:
        from heilo.models.teacher.adapter import TeacherAdapter
    except ImportError as e:
        return UnavailableAdapter("teacher", "HEILO Teacher",
                                  f"não instalado (pasta models/teacher ausente: {e})")
    return TeacherAdapter(metadata_path=metadata_path)
