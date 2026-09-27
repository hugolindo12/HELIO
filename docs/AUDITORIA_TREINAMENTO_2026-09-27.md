# Auditoria do treinamento do HEILO Seed (27/09/2026)

**Escopo:** tudo o que foi treinado até 27/09 às ~02h (horário de Brasília) por Claude (Cowork) e pelo Antigravity.

**Método:** leitura de arquivos, checkpoints (`info.historico`), logs, git (`.git/logs`, `git show --stat`), datasets e **testes executados agora**. Nenhum arquivo foi apagado ou alterado durante a auditoria.

**Conflito de interesse declarado:** este relatório foi escrito pelo Claude, que também treinou a v0.3 e escreveu o `heilo_eval_v1`. Por isso a auditoria usa três avaliações:

- o `heilo_eval_v1` (do Claude)
- o benchmark do Antigravity
- uma **sonda nova de 24 perguntas**, escrita antes de ver as respostas dos modelos e conferida: 0 perguntas em comum com qualquer dado de treino

---

## 1. O que realmente foi treinado

| | v0.1 | v0.2 | v0.3 |
|---|---|---|---|
| Quem treinou | Claude (26/09) | Antigravity (27/09) | Claude (27/09) |
| Onde | Nuvem, CPU com 2 vCPU | PC do usuário (o relatório diz i5-1155G7; **não foi possível verificar**) | Nuvem, CPU com 2 vCPU |
| Pai | aleatório (do zero) | v0.1 | v0.1 |
| Arquitetura | GPT byte-level, 4 camadas, 4 cabeças, 256 dim, contexto 256, ~3,3 M parâmetros | igual | igual |
| Dataset | 87 conversas escritas pelo Claude | `heilo_v001`: 92 treino + 11 validação (87 do Claude + 16 do Qwen) | `heilo_v002`: 466 treino + 45 validação (87 + 280 escritas pelo Claude + 145 programáticas; **0 do Qwen**) |
| Passos | 300 + 3000 = 3300 | 4300 no checkpoint (+1000). **Checkpoint intermediário**, ver §7 | +2500, total 5800 |
| Lote / LR | 16 / 3e-4 com cosseno | 16 / 3e-4 | 16 / 3e-4 |
| Tempo | ~3,7 min + ~40 min (datas no histórico do checkpoint) | **Não registrado.** Janela entre o dataset e o checkpoint: ≤ 39 min | 32 min de treino (log `ciclos.log`) |
| Perda final de treino | 0,083 | não registrada no checkpoint | 0,123 (média móvel); 0,003 medida agora em 80 exemplos de treino |
| Checkpoint | `versions/v0.1/heilo_seed_v0.1.pt` = `versions/heilo_seed_v0.1.pt` (hash 65b766…, idênticos) | `versions/v0.2/heilo_seed_v0.2.pt` (60c67c…) | `versions/heilo_seed_v0.3.pt` = `weights/heilo_seed.pt` (22dd9f…) |

Os números do relatório do Antigravity (perda de benchmark 3,6584 → 3,2248) **foram reproduzidos exatamente** com o método dele.

## 2. Aprendeu? Antes × depois (testes executados agora)

### Perda da resposta (menor é melhor)

| Conjunto | v0.1 | v0.2 | v0.3 |
|---|---|---|---|
| Treino (80 exemplos de `heilo_v002`) | 3,508 | 3,530 | **0,003** |
| Validação `heilo_v002` (45) | 2,835 | 3,124 | 2,104 |
| Teste `heilo_eval_v1` (67, Claude) | 4,323 | 4,359 | 1,520 |
| Benchmark do Antigravity (8) | 4,101 | 3,737 | 3,478 |

### Acerto por critério objetivo

| Teste | v0.1 | v0.2 | v0.3 |
|---|---|---|---|
| Resposta **idêntica** à do treino (80 perguntas de treino) | 10% | 10% | **100%** |
| `heilo_eval_v1`: acerto greedy | 9,0% | 6,0% | 26,9% |
| `heilo_eval_v1`: generalização / instruções | 5% / 0% | 5% / 0% | 20% / 17% |
| **Sonda independente (24 perguntas novas)** | 4,2% (1/24) | 0% | 4,2% (1/24, e esse acerto é falso positivo) |
| Sonda: lógica, português, programação, instruções, CNC | 0% | 0% | 0% |
| Consistência (3 amostras) | 0,99 | 0,99 | 1,00 |

### Conclusões

- **A v0.3 decorou o treino.** Ela reproduz 100% das respostas de treino (perda 0,003) e a perda de validação é 2,10, uma diferença de ~700×. Isso é **overfitting forte**.
- O ganho da v0.3 no `heilo_eval_v1` é real, mas vem de perguntas **parecidas** com o treino. O próprio `heilo_eval_v1` puxa para o estilo do treino: a perda no teste (1,52) ficou menor que a da validação (2,10).
- **Em perguntas realmente novas, nenhuma versão funciona.** Exemplos da v0.3: "Qual é maior: 9 ou 12?" → "G91 é modo incremental…"; "2 mais 2 (só o número)" → "20 mais 20 é 4.".
- **A v0.2 não melhorou fora do próprio benchmark.** Ela piorou na validação e no `heilo_eval_v1`, e decorou 12 das 14 respostas do Qwen com que treinou.
- **Melhoria comprovada:** a v0.3 lembra melhor as respostas que viu (e paráfrases próximas), e responde "não tenho acesso em tempo real" nos casos treinados.
- **Não comprovado:** capacidade de generalizar, raciocinar, seguir instruções novas ou conversar em português fora do que foi decorado.

## 3. Overfitting

| | Treino | Validação | Teste independente |
|---|---|---|---|
| v0.3 | perda 0,003, 100% idêntico | perda 2,10 | sonda 0% nos conteúdos reais |
| v0.2 | 12/14 respostas do Qwen idênticas | pior que a v0.1 | 0% |

**Diagnóstico:** overfitting (memorização). É o esperado para um modelo de 3,3 M parâmetros, com tokenizer em bytes e contexto de 256 bytes, treinado **só com ~500 conversas e sem nenhum pré-treino em texto geral**. O modelo não teve dados suficientes para aprender a língua, então aprendeu a "buscar" respostas decoradas.

## 4. Datasets

| Fonte | Exemplos | Problemas encontrados |
|---|---|---|
| `curated_seed` (Claude) | 87 | 2 perguntas duplicadas; 1 exemplo com 266 tokens (> janela de 256); 1 quase igual à avaliação ("bom dia heilo"); **resposta desatualizada e contraditória**: "sua memória fica no GitHub" |
| `curated_v2` (Claude) | 280 | 1 pergunta duplicada; 5 perguntas com a mesma conta da avaliação (5×5, 10×15, 15×10); 87 exemplos de contas decoráveis |
| `ciclo_01` (programático, Claude) | 145 | Contas corretas por construção (0 erros). O verificador de CNC marcou 12 sim/não como errados, mas são **falsos positivos** do verificador |
| `teacher_approved_v0.2` (Qwen/Antigravity) | 16 | 14 com fato não verificado; 2 com > 220 caracteres (> 256 tokens); 2 **idênticos ao benchmark do próprio Antigravity**; erros de conteúdo ("…listas, strings e **rãs**"; "desenvolvida **pela** Hugo"; "Atribua valor… com a sigla \"=.") |

- **Inconsistências entre fontes:** 4 perguntas com respostas diferentes. A mais grave é "onde fica sua memória?", que ensina GitHub **e** memória local.
- **Treino × avaliação:**
  - `heilo_v001` (Antigravity) contém 2 das 8 perguntas do benchmark dele. A avaliação dele está contaminada.
  - `heilo_v002` (Claude) tem 1 pergunta de validação também no treino.
  - A sonda nova não tem nenhuma pergunta em comum com os dados de treino.

## 5. O Qwen

| Etapa | Antigravity | Claude |
|---|---|---|
| Gerou dados? | **Sim**, 18 respostas (`teacher_generator.py`) | **Não.** O Qwen nunca rodou no ambiente do Claude (sem acesso ao Hugging Face). O caminho do Teacher só foi testado com um professor simulado |
| Como? | Perguntas fixas + "Instrução" com a resposta esperada já escrita. O Qwen parafraseou o que o Antigravity ditou | – |
| Validação | Só estrutura e tamanho (15–300 caracteres). Sem checagem de fato | – |
| Usado no treino? | Sim: 14 no treino e 2 na validação da v0.2 | Não |
| O Seed aprendeu? | Decorou 12/14. Nenhum ganho medido fora desses exemplos | – |

- **Dependência escondida:** nenhuma no chat. `core/` não importa o Teacher.
- O Teacher só é importado por:
  - `models/registry.py` (import preguiçoso)
  - `models/teacher/adapter.py`
  - `training/pipeline.py`, `training/ciclos.py` e `training/teacher_generator.py` (treino)
  - `independence.py` e testes
- `HEILO_TEACHER_IN_CHAT` é `false` por padrão, e não existe `.env` no PC.

## 6. Teste sem Qwen (executado agora)

O `python -m heilo.main independencia` roda numa cópia isolada, sem internet e sem git. Resultado nas três rodadas (Teacher habilitado, desligado e com a pasta `models/teacher` apagada):

- inicia ✔
- **conversa pelo Seed** ✔ (`fonte: model_seed`)
- memória ✔
- conhecimento ✔
- `/ensinar` ✔
- `/aprender` ✔
- `/gerar` e `/comparar` respondem "Esta função requer o HEILO Teacher."

**Caminho real:** `HEILO → Core → ModelManager → HEILO Seed → resposta`. Em perguntas "gerais", a HEILO pode responder antes pelo conhecimento local (`source: knowledge`). Se o Seed estiver indisponível, cai no `core/model_adapter.py`, que usa o stub offline ou a OpenAI **se** `OPENAI_API_KEY` existir. Esse caminho não é o Qwen, mas é um motor externo possível.

## 7. Bugs e problemas de código

| # | Onde | Problema | Gravidade |
|---|---|---|---|
| 1 | `training/runner.py` (Antigravity) | Treina **direto nos pesos ativos** (`arquivo=active_weights`, `salvar_cada=500`). Se interromper, o modelo ativo fica num estado intermediário | Alta |
| 2 | v0.2 (Antigravity) | O checkpoint tem `passos_totais=4300` mas **não tem a entrada no histórico**. O `treinar()` só grava o histórico no fim, então isso indica um salvamento intermediário (passo 1000 de um treino de 2000 que não terminou). O `finalize_v02.py` copiou esse estado como v0.2 | Alta |
| 3 | `training/finalize_v02.py` | Valores fixos no código: "1.000 passos", "4.300", fallback `3.6584`. O relatório não reflete medições automáticas | Média |
| 4 | Benchmark do Antigravity | 2/8 perguntas estão no treino; só 8 itens; só perda (sem acerto) | Média |
| 5 | `pipeline.train_seed` / `main treinar` (Claude) | Mesmo risco do #1 quando usado fora do `ciclo` (salva nos pesos ativos durante o treino). O `ciclo` treina numa cópia candidata e é seguro | Média |
| 6 | `training/qualidade.py` (Claude) | Falsos positivos: `fato_cnc` em perguntas sim/não; `codigo_python_ok` reprova código seguido de uma frase; `parece_portugues("Falou! Qualquer coisa…")` = falso | Média |
| 7 | `training/evaluate.py` (Claude) | Critério por palavra é permissivo: "10 vezes 10 é 150" conta como certo; "Ficiono… HEILO Seed…" conta como identidade certa | Média |
| 8 | `heilo_eval_v1` (Claude) | Escrito pelo mesmo autor dos dados; 5 contas em comum com o treino; ~metade é "paráfrase". Mede mais memória do que generalização | Média |
| 9 | `models/seed/gpt.py`, `responder` | Corta o histórico por bytes (`ids[-limite:]`) sem respeitar o limite das mensagens. Em conversas longas o contexto começa no meio de uma mensagem ou de um caractere UTF-8 | Baixa |
| 10 | Manifesto `heilo_v001` | `teacher_examples: 0` apesar de 16 do Qwen (a origem `teacher_qwen` não era contada). Corrigido depois no `pipeline.py` | Baixa |
| 11 | Repositório | `datasets/evaluation/` duplicado na raiz; `.git/index.lock.stale_claude` sobrando; `training/test_independence.py` fora de `tests/` (não roda no pytest) | Baixa |
| 12 | Git (Antigravity) | Commit bd9091c (27/09 01:34) **com push para o GitHub**: 67 arquivos, incluindo 3 checkpoints de 13 MB e `rag/data`. `models/seed/versions/` foi para o `.gitignore` depois, mas os arquivos já rastreados continuam no histórico | Média |

Não foram encontrados:

- imports quebrados
- problemas de tokenizer (vocabulário fixo de 260)
- problemas de GPU/CPU (usa CPU sem CUDA)
- problemas de memória (checkpoints de 13 MB)
- problemas de carregamento: os 3 checkpoints carregaram e geraram texto

## 8. Quem fez o quê

A autoria do Antigravity foi atribuída por exclusão, com base em: horário dos arquivos (03:45–04:34 UTC), ferramenta diferente desta sessão e o que o usuário informou. O git não distingue autores: tudo aparece como `hugolindo12`.

**Claude:**

- Arquitetura: Core, ModelManager, Seed/Teacher, Memory, Knowledge, Training, testes e independência.
- Seed do zero (v0.1).
- 280 exemplos escritos à mão e 145 programáticos.
- `heilo_eval_v1`, avaliador e ciclos automáticos (`training/ciclos.py`, `qualidade.py`, `geradores.py`).
- Treinou a v0.3 e a promoveu a ativa às ~04:55 UTC. **Isso sobrescreveu o modelo ativo do Antigravity**, mas a v0.2 foi preservada.
- Colocou os 16 exemplos do Qwen em quarentena.

**Antigravity:**

- Criou `teacher_generator.py`, `runner.py`, `evaluator.py`, `finalize_v02.py`, `training/test_independence.py`, o benchmark de 8 perguntas e as pastas `versions/v0.1` e `versions/v0.2` com manifest e metrics.
- **Rodou o Qwen de verdade:** 18 gerações, 16 aprovadas.
- Treinou a v0.2 sobre os pesos ativos.
- Mudou o `.gitignore` e fez **commit + push**.
- Não alterou os arquivos de código do Claude (conferido: nenhum arquivo existente foi modificado por ele além do `.gitignore`).

## 9. Conflitos

1. **Os dois partiram do mesmo checkpoint (v0.1)** e produziram versões irmãs (v0.2 e v0.3), não uma sequência.
2. **Os dois escreveram `models/seed/weights/heilo_seed.pt`:** o Antigravity às 04:29 UTC, o Claude às ~04:55 UTC. Não houve escrita simultânea detectada, mas os treinos rodaram ao mesmo tempo (Claude na nuvem 04:10–04:44, Antigravity no PC ~03:50–04:29).
3. **Numeração de dataset colidiu:** os dois criaram `heilo_v001`. O do Claude foi renomeado para `heilo_v002`.
4. **Dois registros de versões:** `versions/*/manifest.json` (Antigravity) e `data/training/versions.json` (Claude).
5. **Risco atual:** se o `treinar_heilo_ciclos.bat` e o runner do Antigravity rodarem juntos, os dois gravam os pesos ativos e o dataset ao mesmo tempo.

## 10. Organização de versões (proposta, nada foi movido)

```
models/seed/
├── versions/
│   ├── v0.1/  heilo_seed_v0.1.pt + manifest.json   (já existe; a cópia solta versions/heilo_seed_v0.1.pt é duplicata idêntica)
│   ├── v0.2/  heilo_seed_v0.2.pt + manifest/metrics/report  (já existe)
│   └── v0.3/  mover versions/heilo_seed_v0.3.pt para cá + manifest + eval
├── weights/heilo_seed.pt   ← cópia da versão ativa ("best")
└── (registro único: data/training/versions.json)
```

## 11. Nota de qualidade

| Área | Status | Motivo |
|---|---|---|
| Treinamento | **PROBLEMA** | Os treinos rodaram de verdade, mas a v0.3 tem overfitting forte e a v0.2 é um checkpoint intermediário, com relatório de valores fixos |
| Dataset | **ATENÇÃO** | Pequeno (~511), quase todo de uma fonte, com contradições, contas decoráveis e 16 exemplos do Qwen não verificados (em quarentena) |
| Avaliação | **ATENÇÃO** | Os dois benchmarks puxam para o próprio treino; critérios permissivos. A sonda independente é o único teste realmente novo, e mostra ~0% |
| Modelo | **PROBLEMA** | Não generaliza: 0% em lógica, português, programação e instruções novas |
| Qwen Teacher | **ATENÇÃO** | Usado uma vez (18 respostas ditadas por instrução, validação fraca). Nenhum ganho mensurável além da memorização. O ciclo com Qwen do Claude **não foi executado** |
| Independência do Qwen | **OK** | Teste executado: chat pelo Seed com o Teacher desligado ou removido; o Core não importa o Teacher |
| Checkpoints | **ATENÇÃO** | Todos preservados e carregáveis (hashes conferidos), mas a v0.2 é intermediária, a organização está misturada e há duplicata |
| Código | **ATENÇÃO** | 12 problemas (§7); nenhum impede a execução |
| Estabilidade | **ATENÇÃO** | Testes: 113 passam no PC (1 falha por falta de permissão do ambiente) e 119 na nuvem. Duas ferramentas escrevendo no mesmo projeto |
| Reprodutibilidade | **ATENÇÃO** | v0.3 reproduzível (dataset v002 salvo, semente fixa, log e registro). v0.2 não (sem log, interrompida, relatório fixo) |

## 12. Recomendação

**Versão atual:** v0.3, ativa. É a melhor em todas as medidas de perda e no `heilo_eval_v1`, **mas decorada**.

**Próximo treino recomendado** (não é "mais ciclos iguais"):

1. **Congelar a avaliação antes de treinar:**
   - juntar a sonda independente com um conjunto novo de ~200 perguntas revisado por humano;
   - usar critérios mais rígidos (resposta = fato correto, não só a palavra);
   - promover versão só com ganho **nesse** conjunto.
2. **Pré-treino de linguagem:** treinar o Seed primeiro em texto geral em português (dezenas a centenas de MB, por exemplo Wikipedia PT) para ele aprender a língua, e só depois ajustar com as conversas. Isso pede GPU (Colab) e provavelmente um modelo maior (20–50 M parâmetros) com tokenizer BPE. É mudança de arquitetura, então só com sua decisão.
3. **Anti-overfitting:** parada antecipada pela perda de validação, menos passos por ciclo e mais dados diversos.
4. **Qwen como Teacher de verdade:**
   - gerar **milhares** de pares variados, não 18;
   - verificar automaticamente tudo o que for verificável;
   - mandar amostras para revisão humana;
   - nunca ditar a resposta no prompt.
5. **Uma só ferramenta treinando por vez.**

**O que NÃO deve ser alterado:**

- os checkpoints v0.1, v0.2 e v0.3;
- os conjuntos de avaliação (congelados);
- a quarentena dos 16 exemplos do Qwen (só `/revisar` libera);
- a separação Core / ModelManager / Seed / Teacher;
- `HEILO_TEACHER_IN_CHAT=false`;
- os arquivos de dados originais de cada agente (são evidência).

---

## Ações tomadas depois da auditoria (27/09, a pedido do usuário)

**Removido** (continua recuperável pelo histórico do git, no commit bd9091c, que já tinha ido para o GitHub):

- checkpoint `versions/v0.2`, com relatório, metrics e config;
- `training/runner.py`, `finalize_v02.py`, `teacher_generator.py`, `evaluator.py` e `test_independence.py` (do Antigravity);
- o dataset `heilo_v001`;
- o benchmark duplicado em `datasets/` e em `data/datasets/evaluation/`;
- `data/review/quarentena.jsonl`;
- caches e outros sobrejantes: `_to_delete/`, `__pycache__`, `.pytest_cache`, as pastas vazias `config/` e `server/`, `heilo_status.txt` e `.git/index.lock.stale_claude`.

**Reorganizado:**

- Versões agora ficam em `models/seed/versions/v0.1/` e `v0.3/`, cada uma com `heilo_seed.pt` e `manifest.json`. A cópia ativa fica em `weights/heilo_seed.pt` (v0.3).
- A duplicata da v0.1 foi eliminada.
- O registro único é `data/training/versions.json` (a v0.2 fica listada como "removida", com o motivo).
- As avaliações congeladas ficam em `data/eval/`: `heilo_eval_v1`, `sonda_independente_v1` e `benchmark_antigravity_v1`.

**Os 16 exemplos do Qwen/Antigravity** saíram de `approved/` e foram para `data/teacher/generated.jsonl` como **não verificados**. Não entram no treino; aparecem no `/revisar`.

**Corrigido no código do Claude:**

- Bug #5: `python -m heilo.main treinar` agora treina numa versão candidata (`versions/v0.N/`), avalia e só promove se melhorar. Nunca treina direto nos pesos ativos.
- Bug #6: falsos positivos do validador (sim/não de CNC, código seguido de frase, "Falou! Qualquer coisa…").
- Bug #9: o contexto do Seed é montado com mensagens inteiras.
- 3 respostas do `curated_seed` que contradiziam a arquitetura: "memória no GitHub", "cada conversa fica salva no repositório e eu treino" e "guardado no GitHub".

**Não alterado:**

- v0.1 e v0.3;
- `heilo_eval_v1` (continua congelado, com os critérios de acerto já corrigidos antes da auditoria);
- `heilo_v002`;
- a memória;
- a separação Core / Seed / Teacher.
