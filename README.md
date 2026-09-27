# HEILO

A HEILO é uma **plataforma de IA própria**, formada por orquestrador, modelo próprio, memória, conhecimento, treinamento e agentes.

O modelo é um componente substituível. A HEILO não é o modelo.

```
HEILO ≠ Qwen      HEILO ≠ LoRA      HEILO ≠ modelo externo
```

## Componentes

| Componente | O que é | Onde |
|---|---|---|
| **HEILO Core** | Orquestrador: recebe mensagens, gerencia contexto, escolhe o modelo, acessa memória e conhecimento, controla o treino | `core/` |
| **HEILO Seed** | Modelo próprio experimental da HEILO, **treinado do zero** | `models/seed/` |
| **HEILO Teacher** | Modelo externo **temporário** que ajuda no treino, na geração de exemplos e na avaliação. **Opcional** | `models/teacher/` |
| **HEILO Memory** | Memória persistente e local (histórico, sessões) | `memory/manager.py` |
| **HEILO Knowledge** | Base de conhecimento recuperável (documentos, ensinamentos, RAG) | `knowledge/`, `rag/` |
| **HEILO Training** | Dados, validação, dataset HEILO e treino do Seed | `training/`, `data/` |
| **HEILO Agents** | Agentes especializados. Hoje: Code, Test, Research. Futuro: PC, CAD, SAP | `agents/` |
| **HEILO Cloud** | Infraestrutura futura (preparada, não implementada) | `cloud/` |

```
HEILO Core
   ↓
Model Manager (core/model_manager.py)
   ↓
ModelAdapter (models/base.py)
   ├── HEILO Seed      ← modelo próprio (padrão)
   ├── HEILO Teacher   ← opcional, pode ser desligado ou apagado
   └── futuro modelo HEILO Cloud
```

O Core **não importa** nenhuma biblioteca de modelo (torch, transformers, peft) e não sabe qual modelo externo está por trás do Teacher. O teste `test_core_nao_importa_modelo_externo` garante isso.

## HEILO Seed

- Transformer pequeno (~3,3 M parâmetros), com tokenizer em bytes. Começou com pesos aleatórios.
- Aprende **somente** com o dataset HEILO: exemplos aprovados em `data/approved/`.
- Nenhum peso de modelo externo é copiado para ele.
- É um projeto de longo prazo. Hoje ele responde bem ao que já viu e erra bastante em assuntos novos. Ele evolui com mais dados aprovados e mais treino.
- Veja [models/seed/MODEL_CARD.md](models/seed/MODEL_CARD.md).

## HEILO Teacher: por que existe e por que não é o cérebro

O Teacher existe para **acelerar** a evolução do Seed:

- gera exemplos
- serve de referência para comparação
- ajuda a montar o conhecimento inicial

Ele **não** é o cérebro permanente da HEILO:

- não controla o Core, a memória nem o conhecimento
- não é obrigatório para iniciar a HEILO nem para usá-la offline
- não responde no chat por padrão
- o que ele gera nunca entra no treino sem validação e aprovação, porque **ele também erra**

| Metadado | Valor |
|---|---|
| provider | external |
| base_model | `Qwen/Qwen2.5-0.5B-Instruct` |
| licença | Apache-2.0 |
| adapter | LoRA (opcional) |
| purpose | training_teacher |
| status | temporary |

Os metadados completos estão em `models/teacher/teacher.json`. Veja também [models/teacher/README.md](models/teacher/README.md).

- **Instalar:** `pip install torch transformers peft accelerate`. O modelo base é baixado do Hugging Face na primeira vez.
- **Desativar:** `HEILO_TEACHER_ENABLED=false` no arquivo `heilo/.env`.
- **Remover:** apague a pasta `models/teacher/`. A HEILO continua funcionando com o Seed.
- **Trocar de professor:** edite `models/teacher/teacher.json`. O Core não muda.

## Como a HEILO aprende

```
HEILO Teacher ─┐
/ensinar ──────┼─→ registro → validação → revisão (/revisar) → data/approved → dataset HEILO → treino → HEILO Seed
Memory ────────┘
```

| Pasta | Conteúdo |
|---|---|
| `data/raw/` | Candidatos tirados da memória (não verificados, local) |
| `data/teacher/` | Saídas do Teacher (não verificadas) |
| `data/taught/` | O que você ensinou com `/ensinar` |
| `data/approved/` | **Única fonte do dataset** |
| `data/rejected/` | Rejeitados, com o motivo |
| `data/datasets/` | Versões do dataset e manifesto (quantos exemplos vieram de cada origem) |
| `data/training/` | Registro de cada treino e das comparações Seed × Teacher |

- **Conversar não treina.** Cada troca vai para a memória local (`memory/sessions/`), e só isso.
- **`/ensinar pergunta => resposta`** registra um exemplo supervisionado (já aprovado, porque foi você quem ensinou) e salva o par no conhecimento (`knowledge/taught/`). O conhecimento pode ser recuperado na hora. **Nenhum peso é alterado.**
- **`/aprender`** coleta as trocas da memória, valida, rejeita o que é inválido, deixa o restante pendente de revisão e monta uma nova versão do dataset. **Não treina, não chama o Teacher e não envia nada para o GitHub.**
- **`/revisar`** é onde você aprova, rejeita ou edita os candidatos que vieram da memória e do Teacher.
- **`python -m heilo.main treinar`** treina o Seed com a última versão do dataset. Os exemplos de validação ficam fora do treino e servem para medir o resultado (`perda_validacao`).
- **`/gerar pergunta`** pede ao Teacher um exemplo, que fica como **não verificado**. Se o Teacher estiver ausente, a resposta é: "Esta função requer o HEILO Teacher."

A destilação Teacher → Seed é feita por **exemplos**, não por cópia de pesos. Veja [training/distillation/README.md](training/distillation/README.md).

## Comandos do chat (`python -m heilo.main cli`)

| Comando | Função |
|---|---|
| `/ensinar pergunta => resposta` | Registra um exemplo e um conhecimento |
| `/aprender` | Memória → validação → dataset |
| `/revisar` | Aprovar, rejeitar ou editar candidatos |
| `/gerar pergunta` | O Teacher gera um exemplo não verificado |
| `/comparar pergunta` | Seed × Teacher lado a lado. É uma ferramenta de avaliação e o veredito é seu |
| `/cerebro auto\|seed\|teacher\|off` | Escolhe o motor (`mini` e `lora` continuam funcionando como aliases) |
| `/metricas` | Números reais: Teacher gerou, aprovado, usado no treino; treinos do Seed |
| `/memoria [busca]` | Memória local |
| `/status` | Estado dos modelos, da memória e do conhecimento |

No modo `/cerebro auto`, a HEILO usa **1. o Seed** e **2. o Teacher**, mas o Teacher só entra se `HEILO_TEACHER_IN_CHAT=true`. Se nenhum dos dois estiver disponível, ela cai nas respostas prontas e no conhecimento local.

## Linha de comando

```bash
pip install -r heilo/requirements.txt

python -m heilo.main cli             # conversar
python -m heilo.main aprender        # memória → validação → dataset
python -m heilo.main revisar         # revisar candidatos
python -m heilo.main treinar         # treina uma versão candidata do Seed, avalia e só promove se melhorar
python -m heilo.main versoes         # tabela de versões e métricas
python -m heilo.main gerar --arquivo perguntas.txt   # Teacher gera exemplos (opcional)
python -m heilo.main metricas        # dependência do Teacher × evolução do Seed
python -m heilo.main independencia   # HEILO Independence Test
python -m heilo.main publicar        # commit + push do que é apropriado
python -m heilo.main atualizar       # git pull (ex.: Seed treinado no Colab)
python -m heilo.main server          # interface web (http://localhost:8000)
```

Treino na nuvem (GPU grátis): [training/colab/treinar_heilo.ipynb](training/colab/treinar_heilo.ipynb).

## Offline

Estas partes funcionam **sem internet, sem GitHub, sem Cloud e sem o Teacher**, desde que o PyTorch e os pesos do Seed estejam instalados:

```
HEILO → Core → HEILO Seed → Memory local → Knowledge local
```

- Os embeddings do RAG são locais (LSA). A HEILO só tenta o Ollama em `127.0.0.1`, com timeout de 2 s.
- A pesquisa web (agente Research) e o LLM dos agentes (`OPENAI_API_KEY`) usam a internet **quando configurados**. Sem eles, o sistema cai no modo offline (stub).
- Para comprovar, rode `python -m heilo.main independencia`. O procedimento está em [docs/HEILO_INDEPENDENCE_TEST.md](docs/HEILO_INDEPENDENCE_TEST.md).

## GitHub

O GitHub serve para **código, versionamento, configuração, documentação, dados aprovados e releases do Seed**. Ele não é o cérebro, não é a memória e não é um banco vetorial.

- `memory/sessions/` (conversas), `data/raw/`, `data/datasets/`, `.env` e os adaptadores baixados do Teacher ficam no `.gitignore`.
- Segredos (tokens, API keys) ficam no `heilo/.env`. Veja o modelo em `.env.example`. **Deixe o repositório privado.**

## Configuração (`heilo/.env`)

| Variável | Padrão | Função |
|---|---|---|
| `HEILO_BRAIN_MODE` | `auto` | `auto` / `seed` / `teacher` / `off` |
| `HEILO_TEACHER_ENABLED` | `true` | `false` desliga o Teacher |
| `HEILO_TEACHER_IN_CHAT` | `false` | Permite o Teacher responder no chat |
| `HEILO_MEMORY_LOG` | `true` | Grava as conversas na memória local |
| `HEILO_OFFLINE` | `0` | O Teacher usa só o cache local |
| `OPENAI_API_KEY` | – | LLM dos agentes (opcional) |

Valores inválidos não quebram a HEILO: ela usa o padrão e mostra um aviso.

## Estrutura

```
heilo/
├── core/          orchestrator, model_manager, commands, persona, model_adapter (LLM dos agentes)
├── models/        base.py, registry.py, seed/ (weights/ = ativa, versions/v0.N/), teacher/ [OPCIONAL]
├── memory/        manager.py (memória local), store.py, learning.py
├── knowledge/     manager.py, store.py, taught/, documentos
├── rag/           embeddings, vector_store, retriever
├── training/      pipeline.py, validation.py, records.py, distillation/, colab/
├── data/          raw/ teacher/ taught/ approved/ rejected/ datasets/ training/ eval/ (avaliações congeladas)
├── agents/        code, test, research (PC/CAD/SAP: futuro)
├── cloud/         preparado
├── tools/ security/ mcp/ ui/ workspace/
├── docs/          ARCHITECTURE.md, HEILO_INDEPENDENCE_TEST.md
├── tests/
└── main.py
```

## Como substituir o Teacher no futuro

1. Troque `models/teacher/teacher.json` por outro modelo, ou apague a pasta.
2. Rode `python -m heilo.main independencia`.
3. Acompanhe `/metricas`: o objetivo é que o uso do Teacher caia e que a `perda_validacao` do Seed também caia.

Se amanhã o Qwen desaparecer, a HEILO continua existindo.
