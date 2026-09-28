# HEILO Seed — versões e ciclos de treino

Avaliação fixa: `heilo_eval_v1` (67 itens que nunca entram no treino).
Versão ativa: **v0.3**

| Versão | Status | Pai | Dataset | Acerto eval_v1 | Sonda independente | Perplexidade | Instruções | Generalização | Comparação |
|---|---|---|---|---|---|---|---|---|---|
| v0.1 | anterior | - | - | 9.0% | 4.2% | 75.403 | 0.0 | 0.05 |  |
| v0.2 | removida | v0.1 | 1 | 6.0% | - | 78.146 | 0.0 | 0.05 | vs v0.1: -2 itens certos (6,0% vs 9,0%) no heilo_eval_v1 |
| v0.3 | ativa | v0.1 | 2 | 26.9% | 4.2% | 4.57 | 0.1667 | 0.2 | vs v0.1: +12 itens certos |
| v1.0 | rejeitada | - | 1 | 16.4% | 4.2% | 7.94 | 0.1667 | 0.1 | vs v0.3: sonda +1, eval -7 itens (generalizou melhor, mas perdeu muito do que sabia: decisão manual) |
| v1.1 | rejeitada | - | 1 | 17.9% | 4.2% | 8.506 | 0.1667 | 0.1 | vs v0.3: sonda +2, eval -6 itens (generalizou melhor, mas perdeu muito do que sabia: decisão manual) |
| v2.0 | rejeitada | - | 1 | 19.4% | 0.0% | 13.294 | 0.1667 | 0.15 | vs v0.3: sonda -1, eval -5 itens (sonda independente piorou) |
| v2.1 | rejeitada | - | 2 | 10.4% | 12.5% | 11.629 | 0.1667 | 0.0 | vs v0.3: sonda +2, eval -11 itens (generalizou melhor, mas perdeu muito do que sabia: decisão manual) |
| v2.2 | rejeitada | - | 1 | 14.9% | 8.3% | 13.694 | 0.1667 | 0.15 | vs v0.3: sonda +1, eval -8 itens (generalizou melhor, mas perdeu muito do que sabia: decisão manual) |
