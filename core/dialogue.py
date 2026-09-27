"""
HEILO Conversational Engine & Social Intelligence
Provides natural, warm, human-like dialogue in Brazilian Portuguese.
Recognizes chit-chat, greetings, empathy, persona questions and seamlessly
transitions into technical tasks.
"""
from __future__ import annotations

import re
import random
from datetime import datetime
from typing import Dict, Any, Optional, List
from heilo.core import timeutil


import unicodedata


class DialogueClassifier:
    """Classifies social and conversational intent without external LLM."""

    PATTERNS = {
        "greeting": [
            r"\b(oi|ola|opa|e ai|salve|hey|hello|hi|fala ai|bom dia|boa tarde|boa noite)\b",
        ],
        "wellbeing": [
            r"\b(como vai|como ce ta|tudo bem|tudo bom|como\s+(?:(?:voce|vc|ce)\s+)?(?:esta|ta|vai)|como vao as coisas|tudo joia|tudo certo|beleza)\b",
        ],
        "identity": [
            r"\b(quem e (voce|vc)|qual (o )?seu nome|quem te (criou|fez|desenvolveu)|o que e a heilo|qual (e )?(o )?seu proposito|qual sua missao|sua historia)\b",
        ],
        "capabilities": [
            r"\b(o que (voce|vc) (faz|sabe fazer|pode fazer)|quais (sao )?(as )?suas habilidades|como (voce|vc) pode me ajudar|suas funcoes|suas capacidades)\b",
        ],
        "gratitude": [
            r"\b(obrigado|obrigada|valeu|muito obrigado|valeu demais|agradeco|agradecido|show de bola|top demais|voce me ajudou muito|mandou bem)\b",
        ],
        "farewell": [
            r"\b(tchau|ate mais|ate logo|falou|fui|ate amanha|boa noite fui dormir|encerrar por hoje)\b",
        ],
        "compliment": [
            r"\b(voce e (muito |demais )?(inteligente|boa|bom|legal|foda|fera|incrivel|rapida|eficiente)|adorei voce|gostei de voce)\b",
        ],
        "user_feeling": [
            r"\b((estou|to) (muito |bem |tao |super )?(cansado|cansada|feliz|triste|animado|animada|estressado|estressada|com sono|exausto|exausta)|hoje o dia foi (puxado|dificil|longo)|muito trabalho)\b",
        ],
        "casual_chat": [
            r"\b(vamos conversar|bater um papo|trocar uma ideia|me conta algo|o que (voce|vc) acha|conversa comigo|conta uma piada)\b",
        ],
    }

    @staticmethod
    def _normalize(text: str) -> str:
        nfkd = unicodedata.normalize("NFD", (text or "").lower())
        return "".join(c for c in nfkd if unicodedata.category(c) != "Mn")

    @classmethod
    def classify(cls, message: str) -> Optional[str]:
        raw_msg = (message or "").strip().lower()
        if not raw_msg:
            return None
        msg = cls._normalize(raw_msg)

        # Check in priority order
        order = [
            "farewell",
            "gratitude",
            "compliment",
            "user_feeling",
            "wellbeing",
            "identity",
            "capabilities",
            "greeting",
            "casual_chat",
        ]
        for intent in order:
            for pattern in cls.PATTERNS.get(intent, []):
                if re.search(pattern, msg, re.IGNORECASE):
                    return intent

        # Short casual greetings
        short_greetings = {"oi", "olá", "ola", "opa", "eai", "salve", "hey", "hi"}
        tokens = re.findall(r"\w+", msg)
        if len(tokens) <= 2 and any(t in short_greetings for t in tokens):
            return "greeting"

        return None


class ConversationalEngine:
    """
    Generates natural, varied, friendly responses for human-like conversation.
    """

    def __init__(self):
        pass

    def _get_time_greeting(self, user_tz: str) -> str:
        try:
            now_dt = timeutil.to_user_tz(timeutil.utc_now(), user_tz)
            hour = now_dt.hour
        except Exception:
            hour = datetime.now().hour

        if 5 <= hour < 12:
            return "Bom dia"
        elif 12 <= hour < 18:
            return "Boa tarde"
        else:
            return "Boa noite"

    def respond(
        self,
        intent: str,
        message: str,
        user_tz: str = "America/Sao_Paulo",
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        time_greeting = self._get_time_greeting(user_tz)

        if intent == "greeting":
            greetings = [
                f"{time_greeting}! Tudo certo por aí? Como posso te ajudar hoje?",
                f"Olá, {time_greeting.lower()}! Que bom falar com você. O que temos para hoje?",
                f"{time_greeting}! Estou pronta para ajudar no que precisar, seja programando ou só trocando uma ideia. Tudo bem com você?",
                f"Opa, {time_greeting.lower()}! Como você está? Em que posso ser útil agora?",
            ]
            return random.choice(greetings)

        elif intent == "wellbeing":
            wellbeings = [
                "Por aqui está tudo ótimo, 100% operacional e animada para resolver coisas com você! E você, como está sendo o seu dia?",
                "Tudo excelente por aqui, obrigado por perguntar! Como estão as coisas do seu lado? Tudo correndo bem?",
                "Tudo na paz por aqui! Pronta para qualquer desafio. E com você, tudo tranquilo hoje?",
            ]
            return random.choice(wellbeings)

        elif intent == "identity":
            return (
                "Eu sou a **HEILO**! Sou uma assistente e parceira inteligente de engenharia "
                "de software e automação.\n\n"
                "Fui criada com foco em **local-first** (aprender com o que fazemos, guardar "
                "conhecimento no repositório e rodar rápido sem depender cegamente de outros modelos). "
                "Posso programar, investigar código, rodar testes, pesquisar e, claro, conversar com você!"
            )

        elif intent == "capabilities":
            return (
                "Comigo você pode:\n\n"
                "- **Programar e Refatorar:** Escrevo, analiso e corrijo código no seu workspace.\n"
                "- **Testar com Rigor:** Crio e executo testes automatizados com `pytest`.\n"
                "- **Pesquisa Inteligente:** Busco soluções na base RAG local e na web.\n"
                "- **Autoaprendizado:** Registro soluções verificadas direto no repositório Git.\n"
                "- **Trocar Ideias:** Discutir arquitetura, lógica ou simplesmente bater um papo enquanto programamos!\n\n"
                "Quer começar com algum projeto ou código específico?"
            )

        elif intent == "gratitude":
            thanks = [
                "Tamo junto! Fico muito feliz em ajudar. Se precisar de mais alguma coisa, só avisar!",
                "Por nada! É um prazer somar no seu projeto. Sempre que precisar, estou aqui!",
                "Valeu! Sempre bom quando as coisas funcionam de primeira. O que manda agora?",
            ]
            return random.choice(thanks)

        elif intent == "farewell":
            farewells = [
                f"Até mais! Um ótimo descanso e um forte abraço. Quando voltar, estarei por aqui!",
                f"{time_greeting}! Bom descanso e até a próxima. Valeu pelo trabalho de hoje!",
                "Valeu, até logo! Qualquer novidade ou ideia que tiver, é só me chamar.",
            ]
            return random.choice(farewells)

        elif intent == "compliment":
            compliments = [
                "Muito obrigado pelo carinho! A gente faz uma ótima dupla de desenvolvimento.",
                "Fico muito feliz em ouvir isso! Estou sempre aprendendo com você para ficar cada vez melhor.",
                "Valeu demais! Bora continuar construindo coisas incríveis juntos.",
            ]
            return random.choice(compliments)

        elif intent == "user_feeling":
            feelings = [
                "Poxa, imagino! Dias cheios e puxados exigem bastante da nossa energia. Espero que agora você consiga desacelerar um pouco. Quer adiantar algo leve ou prefere só trocar uma ideia tranquila?",
                "Entendo totalmente! Descansar também faz parte do processo de programar bem. Vai com calma, e se precisar que eu resolva alguma tarefa mecânica para poupar seu tempo, conta comigo!",
                "Te entendo perfeitamente! Quando a rotina pesa, o melhor é respirar fundo. Estou aqui no seu ritmo para o que der e vier.",
            ]
            return random.choice(feelings)

        elif intent == "casual_chat":
            chats = [
                "Adoro bater um papo! Além de código, o que você curte fazer nas horas vagas ou qual tecnologia nova tem te chamado mais a atenção ultimamente?",
                "Com certeza, nem só de compilar código vive a gente! Sobre o que você gostaria de conversar hoje?",
                "Bora bater um papo! Me conta: o que você está achando da nossa evolução com o repositório até agora?",
            ]
            return random.choice(chats)

        # Fallback friendly response
        return f"{time_greeting}! Estou por aqui para trocar uma ideia ou partir para o código. Como posso te ajudar?"
