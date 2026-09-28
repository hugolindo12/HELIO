# Nomes dos modelos da HEILO

Do menor/mais rápido para o maior/mais capaz (tema: o sol, como o nome HEILO/Hélio):

| Linha | Tamanho | Meta | Para quê |
|---|---|---|---|
| **HEILO Faísca** | até ~300 M parâmetros | **200 M** | pequena e rápida; roda em qualquer PC. |
| **HEILO Aurora** | ~300 M a 1 bi | **600 M** | média: mais capaz, ainda roda no PC. |
| **HEILO Zênite** | 1 bi ou mais | **2 bi** | a maior: o sol no ponto mais alto. |

- **HEILO Seed** continua sendo o nome do motor e do projeto de treino do zero (motor v2 = estilo Gemma).
- O nome completo é linha + versão: **HEILO Faísca 2.3**, **HEILO Faísca 3.0**.
- Prévias levam o % do pré-treino: **HEILO Faísca 3.0-p40** = 40% do treino da 3.0.
- Modelos de código levam "Code": **HEILO Faísca Code 0.2**.
- A linha é definida automaticamente pelo nº de parâmetros (`heilo/models/linhas.py`).

Versões: Faísca 0.3 → **2.3 (ativa, 30 M)**. Em preparo: **Faísca 3.0 (200 M)**, tokenizer novo com algarismos
separados, `training/colab/heilo_faisca_200m.ipynb`, ~4 h por dia no Colab.
O notebook `heilo_aurora.ipynb` (~150 M, tokenizer antigo) foi substituído pela Faísca 3.0.
