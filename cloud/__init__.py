"""
HEILO Cloud [PREPARADO — não implementado].

Infraestrutura futura de computação, memória e conhecimento:

    HEILO Local → HEILO Cloud API → { Model Server, Memory, Knowledge, RAG, Database }

O ponto de encaixe já existe: um motor remoto entra no ModelManager como mais um
ModelAdapter (ver CloudModelAdapter abaixo). O Core não muda.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from heilo.models.base import ModelAdapter, ModelCard


class CloudModelAdapter(ModelAdapter):
    """Esqueleto do adaptador para um Model Server na HEILO Cloud. Desativado."""

    key = "cloud"

    def __init__(self, endpoint: str = "", model: str = ""):
        self.endpoint = endpoint
        self.model = model

    def card(self) -> ModelCard:
        return ModelCard(name="HEILO Cloud", role="cloud", provider="heilo-cloud",
                         base_model=self.model, purpose="chat", status="not_implemented",
                         origin=self.endpoint)

    def availability(self) -> Tuple[bool, str]:
        return False, "HEILO Cloud ainda não implementada"

    def generate(self, messages: List[Dict[str, str]], **opts) -> str:
        return ""
