"""
HEILO Model Adapter - substitutable LLM interface
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import json
import os
from heilo.config import config, ModelConfig


@dataclass
class Message:
    role: str  # system | user | assistant | tool
    content: str
    name: Optional[str] = None
    tool_call_id: Optional[str] = None


@dataclass
class ModelResponse:
    content: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    raw: Any = None
    usage: Dict[str, int] = field(default_factory=dict)


class BaseModelProvider(ABC):
    @abstractmethod
    def chat(
        self,
        messages: List[Message],
        tools: Optional[List[Dict]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> ModelResponse:
        ...


class StubProvider(BaseModelProvider):
    """
    Deterministic stub for offline testing / demo.
    Advances through realistic flows for CODE and TEST agents.
    """

    def chat(self, messages, tools=None, temperature=None, max_tokens=None) -> ModelResponse:
        history_text = "\n".join(m.content for m in messages if m.content)
        system = next((m.content for m in messages if m.role == "system"), "")
        is_test_agent = "HEILO TEST" in system or "HEILO_TEST" in system
        is_research = "HEILO RESEARCH" in system or "HEILO_RESEARCH" in system
        is_code_agent = "HEILO CODE" in system or "HEILO_CODE" in system or "Task:" in history_text or "TOOL:" in history_text

        if not is_test_agent and not is_research and not is_code_agent:
            return ModelResponse(
                content=(
                    "Olá! Eu sou a **HEILO**, sua plataforma de agentes de IA.\n\n"
                    "Posso ajudar você com:\n"
                    "- **HEILO CODE:** Investigar, editar e corrigir arquivos do workspace com testes.\n"
                    "- **HEILO RESEARCH:** Pesquisar no RAG local e na web.\n"
                    "- **HEILO TEST:** Executar suítes de testes e analisar cobertura.\n"
                    "- **MCP:** Ferramentas de sistema, integração CAD e Blender 3D.\n\n"
                    "Como posso te ajudar hoje?"
                )
            )

        list_count = history_text.count("TOOL_RESULT (list_files)")
        read_count = history_text.count("TOOL_RESULT (read_file)")
        edit_count = history_text.count("TOOL_RESULT (edit_file)") + history_text.count("TOOL_RESULT (write_file)")
        test_count = history_text.count("TOOL_RESULT (run_tests)")
        has_test = test_count > 0
        test_failed = has_test and ("success=False" in history_text or "FAILED" in history_text or "Exit code" in history_text)
        test_passed = has_test and "success=True" in history_text and not test_failed

        if is_test_agent:
            if list_count == 0:
                content = "Vou inspecionar o projeto para mapear o que precisa de testes.\nTOOL: list_files subpath=."
            elif read_count == 0:
                content = "Lendo o código principal.\nTOOL: read_file path=main.py"
            elif read_count == 1:
                content = "Lendo testes existentes.\nTOOL: read_file path=test_main.py"
            elif test_count == 0:
                content = "Executando a suíte de testes.\nTOOL: run_tests command=python -m pytest -q"
            elif test_passed:
                content = (
                    "FINAL: Relatório de testes\n\n"
                    "- test_calculate: PASS\n"
                    "- test_discount: PASS\n"
                    "Cobertura: funções calculate_total e apply_discount cobertas.\n"
                    "Nenhum teste faltando para o módulo principal."
                )
            else:
                content = (
                    "FINAL: Relatório de testes\n\n"
                    "Alguns testes falharam. Verifique apply_discount.\n"
                    "Sugestão: garantir return total * (1 - percent / 100)."
                )
            return ModelResponse(content=content)

        if is_research:
            sk = history_text.count("TOOL_RESULT (search_knowledge)")
            sw = history_text.count("TOOL_RESULT (search_web)")
            sc = history_text.count("TOOL_RESULT (search_code)")
            fu = history_text.count("TOOL_RESULT (fetch_url)")
            # Try to recover the research question
            q = "research"
            for m in messages:
                if m.role == "user" and "Research task:" in (m.content or ""):
                    q = m.content.split("Research task:", 1)[-1].split("\n")[0].strip()[:100]
                    break
            if sk == 0:
                content = (
                    "Vou consultar a base de conhecimento local primeiro.\n"
                    f'TOOL: search_knowledge query="{q}"'
                )
            elif sk > 0 and sw == 0 and "score=" not in history_text and "Python" not in history_text:
                content = (
                    "Base local insuficiente. Buscando na web.\n"
                    'TOOL: search_web query="python calculate discount percentage price final"'
                )
            elif sk > 0 or sw > 0:
                content = (
                    "FINAL: Síntese da pesquisa\n\n"
                    "## Resumo\n"
                    "Para calcular o preço final após um desconto percentual em Python, use:\n\n"
                    "```python\n"
                    "final = total * (1 - percent / 100)\n"
                    "```\n\n"
                    "Erro comum: `total * percent / 100` retorna só o valor do desconto, "
                    "não o preço final.\n\n"
                    "Exemplo: total=60, percent=10 → final=54.0\n\n"
                    "## Fontes\n"
                    "- knowledge/programming/python_discounts.md\n"
                    "- Documentação comum de cálculo comercial"
                )
            else:
                content = 'TOOL: search_knowledge query="research"'
            return ModelResponse(content=content)

        # CODE agent flow
        if list_count == 0:
            content = "Vou começar inspecionando o projeto.\nTOOL: list_files subpath=."
        elif list_count > 0 and read_count == 0:
            content = "Arquivos encontrados. Vou ler o arquivo principal main.py.\nTOOL: read_file path=main.py"
        elif read_count == 1:
            content = (
                "Encontrei a função apply_discount. Há um possível bug na fórmula do desconto. "
                "Vou ler o teste para confirmar.\n"
                "TOOL: read_file path=test_main.py"
            )
        elif read_count >= 2 and edit_count == 0:
            content = (
                "Problema identificado: apply_discount retorna total * percent / 100 em vez de "
                "total * (1 - percent/100). Vou corrigir.\n"
                'TOOL: edit_file path=main.py old_string="return total * percent / 100" new_string="return total * (1 - percent / 100)"'
            )
        elif edit_count > 0 and not has_test:
            content = "Alteração aplicada. Agora vou executar os testes.\nTOOL: run_tests command=python -m pytest -q"
        elif test_passed or (has_test and not test_failed):
            content = (
                "FINAL: Correção concluída com sucesso.\n\n"
                "Problema: a função apply_discount calculava o valor do desconto em vez do preço final.\n"
                "Causa: return total * percent / 100\n"
                "Correção: return total * (1 - percent / 100)\n"
                "Arquivo: main.py\n"
                "Testes: PASS"
            )
        elif test_failed and edit_count < 2:
            content = (
                "Testes falharam. Tentando correção alternativa.\n"
                'TOOL: edit_file path=main.py old_string="return total * percent / 100" new_string="return total * (1 - percent / 100)"'
            )
        elif test_failed and edit_count >= 2 and history_text.count("TOOL_RESULT (run_tests)") < edit_count + 1:
            content = "Nova correção aplicada. Rodando testes novamente.\nTOOL: run_tests command=python -m pytest -q"
        elif test_failed:
            content = (
                "FINAL: Nao foi possivel corrigir automaticamente apos varias tentativas.\n"
                "Verifique apply_discount manualmente."
            )
        else:
            content = "Continuando a análise.\nTOOL: search_code query=def "

        return ModelResponse(content=content)


class OpenAICompatibleProvider(BaseModelProvider):
    """Works with OpenAI, Azure, Ollama, vLLM, LiteLLM, etc."""

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        try:
            from openai import OpenAI
            kwargs = {}
            if cfg.api_key:
                kwargs["api_key"] = cfg.api_key
            if cfg.base_url:
                kwargs["base_url"] = cfg.base_url
            self.client = OpenAI(**kwargs)
            self._available = True
        except Exception:
            self.client = None
            self._available = False

    def chat(self, messages, tools=None, temperature=None, max_tokens=None) -> ModelResponse:
        if not self._available or not self.client:
            return StubProvider().chat(messages, tools, temperature, max_tokens)

        msgs = [{"role": m.role, "content": m.content} for m in messages]
        kwargs = {
            "model": self.cfg.model_name,
            "messages": msgs,
            "temperature": temperature if temperature is not None else self.cfg.temperature,
            "max_tokens": max_tokens or self.cfg.max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        try:
            resp = self.client.chat.completions.create(**kwargs)
            choice = resp.choices[0]
            content = choice.message.content or ""
            tool_calls = []
            if hasattr(choice.message, "tool_calls") and choice.message.tool_calls:
                for tc in choice.message.tool_calls:
                    tool_calls.append({
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": json.loads(tc.function.arguments or "{}"),
                    })
            usage = {}
            if resp.usage:
                usage = {
                    "prompt_tokens": resp.usage.prompt_tokens,
                    "completion_tokens": resp.usage.completion_tokens,
                }
            return ModelResponse(content=content, tool_calls=tool_calls, raw=resp, usage=usage)
        except Exception as e:
            # Fallback to stub on any API error
            print(f"[ModelAdapter] OpenAI error, falling back to stub: {e}")
            return StubProvider().chat(messages, tools, temperature, max_tokens)


class ModelAdapter:
    """
    Single entry point for the rest of the system.
    Switching provider never requires changing agents.
    """

    def __init__(self, cfg: Optional[ModelConfig] = None):
        self.cfg = cfg or config.model
        self._provider = self._build_provider()

    def _build_provider(self) -> BaseModelProvider:
        p = self.cfg.provider.lower()
        if p in ("openai", "ollama", "vllm", "litellm", "azure"):
            return OpenAICompatibleProvider(self.cfg)
        return StubProvider()

    def switch_provider(self, provider: str, model_name: str = None, **kwargs):
        self.cfg.provider = provider
        if model_name:
            self.cfg.model_name = model_name
        for k, v in kwargs.items():
            if hasattr(self.cfg, k):
                setattr(self.cfg, k, v)
        self._provider = self._build_provider()

    def chat(
        self,
        messages: List[Message],
        tools: Optional[List[Dict]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> ModelResponse:
        return self._provider.chat(messages, tools, temperature, max_tokens)

    def simple(self, system: str, user: str) -> str:
        msgs = [Message(role="system", content=system), Message(role="user", content=user)]
        return self.chat(msgs).content
