"""
HEILO Independence Test
Explicitly disables HEILO Teacher (Qwen) and runs HEILO Seed in standalone mode.
Verifies that HEILO generates answers without invoking any external teacher or LLM.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Force offline & disable teacher
os.environ["HEILO_TEACHER_ENABLED"] = "false"
os.environ["HEILO_OFFLINE"] = "1"

from heilo.core.model_manager import ModelManager
from heilo.models.seed.gpt import MiniGPTChat


def run_independence_test():
    print("=" * 60)
    print("  HEILO SEED — TESTE DE INDEPENDÊNCIA (TEACHER DESATIVADO)")
    print("=" * 60)

    # 1. Initialize ModelManager with teacher disabled
    manager = ModelManager(mode="seed", teacher_enabled=False)
    status = manager.status()

    print(f"\n[1] Status dos Modelos:")
    print(f"    - Modo ativo: {status.get('mode')}")
    print(f"    - Teacher habilitado: {status.get('teacher_enabled')}")
    print(f"    - Teacher disponível: {manager.teacher_available()}")
    print(f"    - Seed disponível: {manager.is_available('seed')}")

    assert not manager.teacher_available(), "Erro: Teacher ainda consta como disponível!"
    assert manager.is_available("seed"), "Erro: Seed não está disponível!"

    # 2. Test direct inference on Seed via ModelManager
    test_questions = [
        "Oi, quem é você?",
        "O que é uma variável em Python?",
        "Como declarar uma função em Python?",
        "Tudo bem com você?",
    ]

    print(f"\n[2] Executando Inferência Direta no HEILO Seed (100% Autônomo):")
    results = []
    for q in test_questions:
        reply = manager.generate([{"role": "user", "content": q}], mode="seed")
        print(f"\n  Usuário: {q}")
        print(f"  HEILO Seed: {reply.text}")
        print(f"  Motor utilizado: {reply.model} (Erro: '{reply.error}')")
        results.append({
            "question": q,
            "answer": reply.text,
            "model_used": reply.model,
        })
        assert reply.model == "seed", f"Erro: Resposta não veio do Seed (veio de {reply.model})"

    print("\n" + "=" * 60)
    print("  RESULTADO: TESTE DE INDEPENDÊNCIA APROVADO COM SUCESSO!")
    print("  HEILO Seed funciona 100% autônomo sem Qwen!")
    print("=" * 60)
    return results


if __name__ == "__main__":
    run_independence_test()
