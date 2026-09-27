# Arquitetura oficial da HEILO

```
HEILO (plataforma)
├── Core        orquestrador          core/orchestrator.py, core/model_manager.py, core/commands.py
├── Seed        modelo próprio        models/seed/
├── Teacher     professor temporário  models/teacher/   [OPCIONAL]
├── Memory      memória local         memory/manager.py → memory/sessions/
├── Knowledge   conhecimento (RAG)    knowledge/manager.py, rag/
├── Training    dados → Seed          training/, data/
├── Agents      capacidades           agents/ (Code, Test, Research; PC/CAD/SAP futuros)
└── Cloud       infraestrutura        cloud/ (preparado)
```

## Separação obrigatória

| Conceito | Na HEILO | Não é |
|---|---|---|
| MODELO | `ModelAdapter` substituível (Seed, Teacher, Cloud) | a identidade da HEILO |
| PROFESSOR | Teacher: gera exemplos e serve de referência | verdade, nem cérebro principal |
| CONHECIMENTO | arquivos + índice RAG, recuperados na hora | pesos |
| MEMÓRIA | histórico local de conversas | treino |
| TREINAMENTO | passo explícito sobre exemplos **aprovados** | automático |
| PLATAFORMA | Core + componentes | um wrapper de modelo |

## Regras de dependência

- `core/` importa `models/base.py` e `models/registry.py`, e nunca `models/teacher` nem bibliotecas de modelo. O teste `test_core_nao_importa_modelo_externo` verifica isso.
- `models/registry.py` é o único lugar que importa os pacotes de modelo, e faz isso de forma preguiçosa. Se o Teacher não existir, vira um `UnavailableAdapter`.
- `training/` usa o Teacher só pelo `ModelManager`. Sem o Teacher, lança `TeacherRequired` com a mensagem "Esta função requer o HEILO Teacher."

## Fluxos

```
conversa → Memory                                              (nunca altera pesos)
/ensinar → data/taught + data/approved + knowledge/taught      (nunca altera pesos)
/aprender → Memory → data/raw → validação → dataset            (não treina, sem Teacher, sem git)
/gerar → Teacher → data/teacher (não verificado) → /revisar → data/approved
treinar → dataset (só aprovados) → HEILO Seed → data/training/runs.jsonl
```

## Fases

1. Core + Seed + Memory + Knowledge + Teacher. **← estamos aqui**
2. O Teacher gera dados e o Seed evolui (medido em `/metricas`).
3. O Teacher vira opcional, com HEILO Cloud.
4. Final: Core + Seed + Memory + Knowledge + Agents + Cloud, sem Teacher obrigatório.
