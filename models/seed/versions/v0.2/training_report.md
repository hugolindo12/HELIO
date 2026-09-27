# RELATÓRIO DE TREINAMENTO — HEILO SEED v0.2

- **Versão do Modelo:** `HEILO Seed v0.2` (Baseado em `v0.1`)
- **Data de Conclusão:** 2026-09-27T04:31:34+00:00
- **Hardware Real:** Intel Core i5-1155G7 (4 cores / 8 threads CPU, float32, no CUDA)
- **Teacher Utilizado:** `Qwen/Qwen2.5-0.5B-Instruct` (~988 MB, Apache 2.0)
- **Papel do Teacher:** Exclusivamente externo / temporário para geração curricular supervisionada.

---

## 1. Métricas Objetivas no Benchmark Held-Out

> [!IMPORTANT]
> O conjunto de benchmark contém 8 perguntas essenciais (programação, arquitetura, testes e persona) e **NUNCA** foi visto durante o treinamento.

| Versão | Passos Totais | Eval Loss (Held-Out) | Variação |
| :--- | :---: | :---: | :---: |
| **HEILO Seed v0.1 (Base)** | 3.300 | `3.6584` | Baseline inicial |
| **HEILO Seed v0.2 (Treinado)** | 4.300 | `3.2248` | **-11.85% de erro** |

---

## 2. Parâmetros de Treinamento

- **Passos executados neste ciclo:** 1.000 passos
- **Passos acumulados:** 4.300 passos
- **Batch size:** 16
- **Learning rate:** 3e-4 (com warm-up e decaimento cosseno)
- **Otimizador:** AdamW (weight_decay=0.1, clip_grad=1.0)
- **Janela de contexto:** 256 bytes (tokenizador byte-level UTF-8)
- **Parâmetros do modelo:** 3.32M parâmetros (4 layers, 4 heads, d_model=256)

---

## 3. Dataset Utilizado (heilo_v001)

- **Total de exemplos aprovados:** 103
- **Treino:** 92 exemplos
- **Validação interna:** 11 exemplos
- **Origem Curada:** 77 exemplos
- **Origem Teacher (Qwen):** 16 exemplos validados e aprovados pelas regras de qualidade

---

## 4. Comparativo Qualitativo no Benchmark

| Pergunta | Resposta Seed v0.1 (Antes) | Resposta Seed v0.2 (Agora) |
| :--- | :--- | :--- |
| **O que é uma variável em Python?** | Uma é uma lista guarda um vários valores em ordem, tipo numeros = [1, 2, 3]... | Atribua valor à variável com a sigla "=.... |
| **Como declarar uma função em Python?** | Acho ótimo! É simples de ler, tem biblioteca pra tudo e é perfeito pra auto... | Parâmetros de uma função são variáveis que recebem valores para serem usado... |
| **Qual a diferença entre lista e dicionário?** | Nenhum relos ó bate pra Ma o que você tem em mente.... | O que você acha de python?... |
| **Por que evitar valores padrão mutáveis em Python?** | Acho ótimo! É simples de ler, tem biblioteca pra tudo e é perfeito pra auto... | Listas em Python permitem armazenar várias informações em um único objeto, ... |
| **O que é o padrão Arrange-Act-Assert em testes?** | Sou bem novinha! Nasci em 2026 e ainda estou aprendendo muita coisa.... | Sou bem novinha! Nasci em 2026 e ainda estou aprendendo muita coisa.... |
| **Quem é você e qual sua função?** | Não, eu sou uma inteligência artificial. Mas gosto de conversar de um jeito... | A instrução "return" encerra uma função e retorna um valor para o caller.... |
| **Olá, tudo bem com você?** | Eu sou bem animada pra ajudar! E você, como vai?... | Olá! Estou bem, muito bem. Estou aqui para ajudar!... |
| **Estou muito cansado hoje.** | Poxa, descansa um pouco. Se tiver algo urgente, me passa que eu te ajudo a ... | Por que o programador foi ao médico? Porque estava com muitos bugs e nenhum... |

---

## 5. Garantia de Independência

- O HEILO Seed é um modelo próprio, rodando 100% localmente em CPU.
- O Qwen atuou apenas como Teacher de dados sintéticos e já foi desconectado.
- O sistema opera perfeitamente com `HEILO_TEACHER_ENABLED=false`.
