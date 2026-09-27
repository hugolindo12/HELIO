# HEILO Agents (preparado)

```
HEILO Core → Agent Manager → Code | PC | CAD | SAP | Research
```

| Agente | Situação |
|---|---|
| Code, Test, Research | Já existem (`agents/code`, `agents/test`, `agents/research`). Hoje o Orchestrator roteia para eles direto |
| PC, CAD, SAP | Ainda não criados. Esta etapa é de fundação |

## Observação

Hoje os agentes usam `core/model_adapter.py` (um LLM compatível com OpenAI, ou o stub offline) para tool-calling. Isso é separado do HEILO Seed e do HEILO Teacher.

Um futuro Agent Manager deve receber o motor pelo `ModelManager` em vez de configurar um LLM próprio.
