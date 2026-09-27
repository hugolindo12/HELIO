# HEILO v0.1

Plataforma de agentes de IA com orquestrador, subagente de código, sistema de ferramentas, segurança de workspace, memória e estrutura RAG.

## O que está implementado

- **HEILO Principal (Orchestrator)** – entende intenção, roteia para subagentes, revisa resultados
- **HEILO CODE** – ciclo controlado: entender → planejar → inspecionar → identificar → propor → aplicar → testar → analisar
- **Model Adapter** – interface substituível (OpenAI-compatible + stub offline)
- **Tool System** – list_files, search_code, read/write/edit_file, find_symbol, run_command, run_tests, build, git_diff
- **Workspace Security** – isolamento de path, bloqueio de traversal
- **Permissions** – READ / WRITE / EXECUTE / CRITICAL com confirmação
- **Memory** – conversas + soluções verificadas
- **Knowledge / RAG** – estrutura de pastas + retriever por palavras-chave (pronto para vector DB)
- **Interface** – FastAPI + chat web moderno
- **Task Log** – auditoria completa de cada tarefa

## Execução rápida

```bash
cd /caminho/para/artifacts
pip install -r heilo/requirements.txt

# Demo automatizado (corrige bug no demo_project)
python -m heilo.main demo

# CLI interativo
python -m heilo.main cli

# Interface web
python -m heilo.main server
# abra http://localhost:8000
```

## Configuração de modelo

Por padrão usa o **stub** (offline). Para usar OpenAI:

```bash
export OPENAI_API_KEY=sk-...
```

Ou edite `heilo/config.py` / use `ModelAdapter.switch_provider(...)`.

Para Ollama local:

```python
adapter.switch_provider("ollama", model_name="llama3", base_url="http://localhost:11434/v1")
```

## Estrutura

```
heilo/
├── core/           # orchestrator, model_adapter, task
├── agents/code/    # HEILO CODE + ciclo
├── tools/          # filesystem, execution, git
├── security/       # workspace + permissions
├── memory/
├── knowledge/
├── rag/
├── ui/             # FastAPI + static chat
├── workspace/      # projetos autorizados
└── main.py
```

## Próximos passos naturais

1. Conectar embeddings + Chroma/Qdrant no `rag/`
2. Implementar MCP como camada de ferramentas
3. Adicionar HEILO CAD / HEILO PC como novos agents/
4. Backup/checkpoint automático antes de WRITE
5. Servidor de conhecimento remoto

## Princípio

O **modelo é o motor**. A **HEILO é o sistema** construído ao redor dele (orquestrador + agentes + ferramentas + memória + RAG + permissões + interface).
