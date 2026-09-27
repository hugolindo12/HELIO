"""
HEILO Teacher Dataset Generator
Uses Qwen2.5-0.5B-Instruct as a temporary Teacher to generate high-quality,
concise training examples for HEILO Seed.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Callable
from datetime import datetime, timezone

from heilo.models.teacher.adapter import TeacherAdapter
from heilo.training.records import agora, example_id, write_jsonl, append_jsonl
from heilo.training.validation import validate, MAX_CHARS


CURRICULUM_PROMPTS = [
    # 1. Variáveis e Tipos
    {
        "category": "programming/variables",
        "question": "O que é uma variável em Python?",
        "instruction": "Explique de forma curta e direta o que é uma variável em Python com um exemplo de 1 linha.",
    },
    {
        "category": "programming/variables",
        "question": "Como atribuir um valor a uma variável em Python?",
        "instruction": "Diga de forma curta como atribuir um valor a uma variável em Python usando o sinal de igual.",
    },
    {
        "category": "programming/variables",
        "question": "Quais são os tipos de dados básicos em Python?",
        "instruction": "Cite de forma direta e curta os tipos int, float, str e bool em Python.",
    },
    # 2. Funções
    {
        "category": "programming/functions",
        "question": "Como declarar uma função em Python?",
        "instruction": "Explique de forma curta como declarar uma função usando def e parênteses em Python.",
    },
    {
        "category": "programming/functions",
        "question": "Para que serve o comando return em uma função?",
        "instruction": "Explique de forma curta que return encerra a função e devolve um resultado para quem chamou.",
    },
    {
        "category": "programming/functions",
        "question": "O que são parâmetros de uma função em Python?",
        "instruction": "Explique de forma concisa que parâmetros são variáveis recebidas pela função nos parênteses.",
    },
    # 3. Estruturas de Dados
    {
        "category": "programming/data_structures",
        "question": "Para que serve uma lista em Python?",
        "instruction": "Explique de forma curta que listas guardam múltiplos itens ordenados entre colchetes.",
    },
    {
        "category": "programming/data_structures",
        "question": "O que é um dicionário em Python?",
        "instruction": "Explique de forma curta que dicionários guardam pares de chave e valor entre chaves.",
    },
    {
        "category": "programming/data_structures",
        "question": "Como adicionar um item em uma lista Python?",
        "instruction": "Diga de forma direta que usamos o método append() para adicionar itens ao final da lista.",
    },
    # 4. Controle de Fluxo
    {
        "category": "programming/control_flow",
        "question": "Como funciona o if e else em Python?",
        "instruction": "Explique de forma curta que if testa uma condição e else executa se a condição for falsa.",
    },
    {
        "category": "programming/control_flow",
        "question": "Para que serve o laço for em Python?",
        "instruction": "Explique de forma curta que o laço for percorre sequências como listas, strings ou range.",
    },
    # 5. Boas Práticas & Clean Code
    {
        "category": "programming/clean_code",
        "question": "O que são type hints em Python?",
        "instruction": "Explique de forma concisa que type hints indicam os tipos esperados de argumentos e retorno.",
    },
    {
        "category": "programming/clean_code",
        "question": "Como tratar erros com try e except em Python?",
        "instruction": "Explique de forma curta que o bloco try executa código e o except captura exceções sem travar.",
    },
    # 6. Testes Automatizados com PyTest
    {
        "category": "programming/testing",
        "question": "Como criar um teste unitário com pytest?",
        "instruction": "Explique de forma curta que criamos uma função começando com test_ e usamos assert.",
    },
    {
        "category": "programming/testing",
        "question": "Para que serve a palavra assert em Python?",
        "instruction": "Explique de forma curta que assert verifica se uma condição é verdadeira no teste.",
    },
    # 7. Identidade e Conversação
    {
        "category": "identity/persona",
        "question": "Quem é você?",
        "instruction": "Diga de forma amigável e concisa que você é a HEILO, uma IA parceira de engenharia de software criada pelo Hugo.",
    },
    {
        "category": "identity/mission",
        "question": "Qual é a missão da HEILO?",
        "instruction": "Diga de forma curta que a missão da HEILO é programar, testar e aprender continuamente de forma local.",
    },
    {
        "category": "conversational/smalltalk",
        "question": "Como você está hoje?",
        "instruction": "Responda de forma alegre e prestativa que está ótima e pronta para ajudar.",
    },
]


class TeacherDatasetGenerator:
    """Generates synthetic examples using the Qwen Teacher."""

    def __init__(self, teacher: Optional[TeacherAdapter] = None):
        self.teacher = teacher or TeacherAdapter()

    def generate_example(
        self,
        item: Dict[str, str],
        dataset_version: str = "heilo_v002",
    ) -> Optional[Dict]:
        """Prompts the Teacher and formats the verified example."""
        question = item["question"]
        instruction = item["instruction"]
        category = item["category"]

        system_prompt = (
            "Você é o HEILO Teacher. Responda em Português do Brasil. "
            "Seja claro, didático e MUITO CONCISO (máximo 2 a 3 frases curtas ou exemplo simples de 1-2 linhas). "
            "Não enrole. O modelo aluno tem contexto curto."
        )

        user_content = f"{question}\n\nInstrução: {instruction}"
        messages = [{"role": "user", "content": user_content}]

        raw_answer = self.teacher.generate(
            messages=messages,
            system=system_prompt,
            temperatura=0.3,
            max_novos=160,
        )

        cleaned_answer = raw_answer.strip()
        if not cleaned_answer:
            return None

        # Format full structured training example
        msgs = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": cleaned_answer},
        ]
        ex_id = example_id(msgs)

        example = {
            "id": ex_id,
            "question": question,
            "answer": cleaned_answer,
            "category": category,
            "quality": "pending",
            "origin": "teacher_qwen",
            "dataset_version": dataset_version,
            "created_at": agora(),
            "messages": msgs,
            "meta": {
                "teacher_model": self.teacher.base_model,
                "teacher_version": self.teacher.card().version,
            },
        }

        # Quality validation
        ok, errors = validate(example)
        if ok and 15 <= len(cleaned_answer) <= 300:
            example["quality"] = "approved"
            example["validation_errors"] = []
            return example
        else:
            example["quality"] = "rejected"
            example["validation_errors"] = errors
            return None

    def generate_batch(
        self,
        items: Optional[List[Dict[str, str]]] = None,
        dataset_version: str = "heilo_v002",
        log: Callable[[str], None] = print,
    ) -> List[Dict]:
        """Generates and validates a complete curriculum batch."""
        items = items or CURRICULUM_PROMPTS
        approved = []

        log(f"[TeacherGenerator] Iniciando geração de {len(items)} exemplos com Qwen Teacher...")
        for i, item in enumerate(items, 1):
            ex = self.generate_example(item, dataset_version=dataset_version)
            if ex and ex.get("quality") == "approved":
                approved.append(ex)
                log(f"  [{i}/{len(items)}] OK: {ex['question']} -> {ex['answer'][:60]}...")
            else:
                log(f"  [{i}/{len(items)}] REJEITADO/FALHA: {item['question']}")

        log(f"[TeacherGenerator] Concluído! {len(approved)}/{len(items)} exemplos aprovados.")
        return approved
