# Nomes dos modelos da HEILO

Do menor/mais rápido para o maior/mais capaz (tema: o sol, como o nome HEILO/Hélio):

| Linha | Tamanho | Para quê |
|---|---|---|
| **HEILO Faísca** | até ~80 M parâmetros | pequeno e rápido; roda em qualquer PC. É a linha atual. |
| **HEILO Aurora** | ~80–250 M | médio: mais capaz, ainda leve. Próximo passo quando houver mais dados. |
| **HEILO Zênite** | 250 M ou mais | o maior: o sol no ponto mais alto. |

- **HEILO Seed** continua sendo o nome do motor e do projeto de treino do zero (ex.: motor v2.0 = estilo Gemma).
- O nome completo é linha + versão: **HEILO Faísca 1.1**, **HEILO Faísca 2.0**.
- Modelos de código levam "Code": **HEILO Faísca Code 0.2**.
- A linha é definida automaticamente pelo nº de parâmetros (`heilo/models/linhas.py`).

Versões atuais: Faísca 0.1, 0.3 (ativa), 1.0 · Faísca Code 0.1. Em treino: Faísca 1.1, Faísca Code 0.2. Prontas para treinar: Faísca 2.0 e **Aurora 1.0** (~149 M, `training/colab/heilo_aurora.ipynb`, 4 h por dia, 2–3 semanas na T4).
