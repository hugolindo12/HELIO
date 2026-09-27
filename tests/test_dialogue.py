"""
HEILO Conversational Engine Unit Tests
Validates social intent classification, natural dialogue generation,
and clean separation between small talk and technical code tasks.
"""
import pytest
from heilo.core.dialogue import DialogueClassifier, ConversationalEngine
from heilo.core.orchestrator import Orchestrator


def test_classifier_greetings():
    for phrase in ["oi", "olá", "opa!", "e aí cara", "salve", "bom dia"]:
        assert DialogueClassifier.classify(phrase) == "greeting", f"Failed for: {phrase}"


def test_classifier_wellbeing():
    for phrase in ["tudo bem?", "como você está?", "tudo bom", "como vai?", "tudo joia"]:
        assert DialogueClassifier.classify(phrase) == "wellbeing", f"Failed for: {phrase}"


def test_classifier_identity():
    for phrase in ["quem é você?", "qual seu nome?", "quem te criou?", "qual o seu propósito?"]:
        assert DialogueClassifier.classify(phrase) == "identity", f"Failed for: {phrase}"


def test_classifier_capabilities():
    for phrase in ["o que você sabe fazer?", "como você pode me ajudar?", "quais suas habilidades?"]:
        assert DialogueClassifier.classify(phrase) == "capabilities", f"Failed for: {phrase}"


def test_classifier_gratitude():
    for phrase in ["obrigado!", "valeu cara", "muito obrigado", "show de bola", "mandou bem"]:
        assert DialogueClassifier.classify(phrase) == "gratitude", f"Failed for: {phrase}"


def test_classifier_farewell():
    for phrase in ["tchau", "até mais!", "falou", "até amanhã"]:
        assert DialogueClassifier.classify(phrase) == "farewell", f"Failed for: {phrase}"


def test_classifier_feelings():
    for phrase in ["estou muito cansado hoje", "hoje o dia foi puxado", "muito trabalho por aqui"]:
        assert DialogueClassifier.classify(phrase) == "user_feeling", f"Failed for: {phrase}"


def test_conversational_engine_responses():
    engine = ConversationalEngine()
    resp_greeting = engine.respond("greeting", "oi", user_tz="America/Sao_Paulo")
    assert len(resp_greeting) > 10

    resp_wellbeing = engine.respond("wellbeing", "tudo bem?", user_tz="America/Sao_Paulo")
    assert "tudo" in resp_wellbeing.lower() or "ótimo" in resp_wellbeing.lower() or "excelente" in resp_wellbeing.lower()

    resp_identity = engine.respond("identity", "quem é você?", user_tz="America/Sao_Paulo")
    assert "HEILO" in resp_identity


def test_orchestrator_conversation_routing():
    orch = Orchestrator()
    res = orch.chat("Tudo bem com você?")
    assert res.get("intent") == "conversation"
    assert res.get("sub_intent") == "wellbeing"
    assert res.get("source") == "conversational_brain"
    # Ensure it didn't dump technical code solutions
    assert "apply_discount" not in res.get("content")
    assert "calculate_total" not in res.get("content")


def test_orchestrator_task_with_greeting_prioritizes_task():
    orch = Orchestrator()
    intent = orch._classify_intent("Oi, pesquise como calcular desconto em Python")
    assert intent == "research"
