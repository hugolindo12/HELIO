"""Configuração comum dos testes.

Roda ANTES da coleta dos módulos de teste (inclusive do heilo.ui.api, que cria um
Orchestrator ao ser importado): os testes nunca usam o Seed/Teacher reais nem
gravam conversas na memória real do usuário.
"""
import os

os.environ["HEILO_BRAIN_MODE"] = "off"
os.environ["HEILO_MEMORY_LOG"] = "false"

import pytest  # noqa: E402

from heilo.config import config  # noqa: E402

config.brain_mode = "off"
config.memory_log_conversations = False


@pytest.fixture(autouse=True)
def _plataforma_isolada(monkeypatch):
    monkeypatch.setattr(config, "brain_mode", "off", raising=False)
    monkeypatch.setattr(config, "memory_log_conversations", False, raising=False)
