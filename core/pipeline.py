"""
HEILO Multi-Agent Pipelines
Orchestrates sequential agent handoffs:
  RESEARCH → CODE
  RESEARCH → CODE → TEST
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class PipelineStepResult:
    agent: str
    status: str
    result: Any = None
    steps: List[Dict] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    task_id: Optional[str] = None


@dataclass
class PipelineResult:
    name: str
    status: str  # success | partial | failed | waiting_permission
    steps: List[PipelineStepResult] = field(default_factory=list)
    final_content: str = ""
    permission: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status,
            "final_content": self.final_content[:500] if self.final_content else "",
            "steps": [
                {
                    "agent": s.agent,
                    "status": s.status,
                    "task_id": s.task_id,
                    "tools": len(s.steps),
                    "sources": [str(x)[:100] for x in (s.sources or [])[:3]],
                }
                for s in self.steps
            ],
        }


def build_code_task_from_research(user_message: str, research_text: str) -> str:
    """Turn research findings into a concrete coding task for HEILO CODE."""
    summary = (research_text or "")[:2500]
    return (
        f"Com base na pesquisa abaixo, aplique a solucao no projeto do workspace.\n"
        f"Pedido original do usuario: {user_message}\n\n"
        f"--- Pesquisa (HEILO RESEARCH) ---\n{summary}\n---\n"
        f"Analise o codigo, aplique a correcao recomendada e rode os testes."
    )


def build_test_task_from_code(user_message: str, code_summary: str) -> str:
    """Task for HEILO TEST after CODE has applied changes."""
    summary = (code_summary or "")[:1500]
    return (
        f"Apos alteracoes no codigo, rode a suite de testes e gere um relatorio claro.\n"
        f"Pedido original: {user_message}\n\n"
        f"--- Contexto da etapa CODE ---\n{summary}\n---\n"
        f"Execute os testes (pytest), confirme PASS/FAIL e resuma a cobertura."
    )


def format_numbered_citations(sources: List[str]) -> str:
    if not sources:
        return ""
    lines = ["", "### Referencias"]
    seen = set()
    n = 1
    for s in sources:
        key = str(s)[:200]
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"[{n}] {key}")
        n += 1
        if n > 12:
            break
    return "\n".join(lines)


def format_pipeline_report(
    name: str,
    research_text: str,
    code_text: str,
    test_text: str = "",
    status: str = "success",
) -> str:
    parts = [
        f"**HEILO Pipeline** `{name}`",
        "",
        "### 1. RESEARCH",
        (research_text or "(vazio)")[:1500],
        "",
        "### 2. CODE",
        (code_text or "(vazio)")[:1500],
    ]
    if test_text is not None and name.endswith("test"):
        parts.extend(["", "### 3. TEST", (test_text or "(vazio)")[:1500]])
    parts.extend(["", f"**Status final:** {status}"])
    return "\n".join(parts)
