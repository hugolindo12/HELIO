# HEILO Independence Test

**Pergunta:** se o Qwen (HEILO Teacher) desaparecer, a HEILO continua existindo?

**Comando:** `python -m heilo.main independencia`. A versão automatizada está em `tests/test_independence.py`.

## Procedimento

O teste roda numa **cópia** da HEILO, numa pasta temporária. Seus dados reais não são tocados. As condições são:

- sem internet (sockets bloqueados)
- sem git (`PATH` vazio)
- sem `OPENAI_API_KEY`
- Hugging Face em modo offline

| Passo | O que acontece |
|---|---|
| 1. Instalar HEILO | Copia o pacote (sem `.git`, memória, `data/raw`, `datasets`, `_to_delete`) |
| 2. Instalar Seed | Confere `models/seed/weights/heilo_seed.pt` |
| 3–4. Executar e confirmar (**rodada A**) | Teacher habilitado |
| 5–7. Desligar o Teacher (**rodada B**) | `HEILO_TEACHER_ENABLED=false` |
| 8–10. Remover o Teacher (**rodada C**) | Apaga `models/teacher/` e simula `transformers` e `peft` desinstalados |

Em cada rodada o teste verifica:

- iniciar
- configuração
- estado do Seed e do Teacher
- conversa
- memória
- `/ensinar`
- conhecimento
- `/aprender`
- `/comparar`
- `/gerar`

Funções **essenciais** (se falharem, a HEILO não é independente): iniciar, configuração, conversar, memória, ensinar, conhecimento, aprender.

## Resultados reais (27/09/2026)

### Ambiente 1: com PyTorch (container de desenvolvimento)

| Rodada | Essenciais | Conversa | Teacher |
|---|---|---|---|
| A: Teacher habilitado | ✔ todas | **HEILO Seed** respondeu ("Oi! Tudo certo por aí? Em que posso te ajudar hoje?") | Instalado. Sem internet, não conseguiu baixar o Qwen: `/gerar` → "Esta função requer o HEILO Teacher." |
| B: Teacher desligado | ✔ todas | **HEILO Seed** respondeu | "desativado (teacher_enabled=false)" |
| C: Teacher removido | ✔ todas | **HEILO Seed** respondeu | "não instalado (pasta models/teacher ausente)" |

### Ambiente 2: sua pasta no PC (sem PyTorch instalado)

| Rodada | Essenciais | Conversa | Observação |
|---|---|---|---|
| A, B, C | ✔ todas | Respostas prontas (fallback) | O Seed ficou indisponível: "PyTorch não instalado (pip install torch)". O Core não quebrou |

## Funções que dependem do Teacher (e como a HEILO reage)

| Função | Por que depende | Reação sem o Teacher |
|---|---|---|
| `/gerar` e `python -m heilo.main gerar` | É o Teacher quem gera os exemplos | "Esta função requer o HEILO Teacher." com o motivo |
| `/comparar` (lado do Teacher) | Compara Seed × Teacher | O lado do Teacher mostra "Esta função requer o HEILO Teacher." e o lado do Seed funciona normalmente |
| `/cerebro teacher` | Escolhe o Teacher como motor | Recusa e mantém o modo atual |
| Colab, passo 1 | Geração em lote | Pula, e o passo 2 (treino do Seed) funciona |

## Conclusão

**A HEILO continua funcionando sem o Qwen.** Ela inicia, conversa com o Seed (ou com o fallback, se o PyTorch não estiver instalado), usa memória e conhecimento, e executa `/ensinar` e `/aprender`. Só as funções de professor ficam indisponíveis, e cada uma avisa com uma mensagem clara.
