# HEILO — Conhecimento próprio primeiro

A HEILO não é só um wrapper de modelo de IA.

## Princípio

1. O **repositório** (knowledge/, memory/, verified_solutions/) é a memória e o entendimento da HEILO.
2. O **modelo** (local ou cloud) é opcional: serve para sintetizar, gerar código novo ou quando o repositório não sabe.
3. Soluções **verificadas** (teste PASS) têm prioridade sobre texto genérico.

## Como alimentar a HEILO

- Coloque manuais e notas em `knowledge/programming/`, `knowledge/documentation/`, etc.
- Pipelines que passam em testes gravam em `memory/verified_solutions/`.
- Use RESEARCH + `save_research_note` para persistir achados.

## Configuração

```python
config.local_first = True
config.knowledge_confidence_threshold = 0.35
config.llm_fallback = True   # False = nunca chama modelo externo
```

Com Ollama local:
```python
adapter.switch_provider("ollama", model_name="llama3", base_url="http://localhost:11434/v1")
```
