# HEILO Seed — versões e ciclos de treino

Avaliação fixa: `heilo_eval_v1` (67 itens que nunca entram no treino).
Versão ativa: **v0.3**

| Versão | Status | Pai | Dataset | Acerto eval_v1 | Sonda independente | Perplexidade | Instruções | Generalização | Comparação |
|---|---|---|---|---|---|---|---|---|---|
| v0.1 | anterior | - | - | 9.0% | 4.2% | 75.403 | 0.0 | 0.05 |  |
| v0.2 | removida | v0.1 | 1 | 6.0% | - | 78.146 | 0.0 | 0.05 | vs v0.1: -2 itens certos (6,0% vs 9,0%) no heilo_eval_v1 |
| v0.3 | ativa | v0.1 | 2 | 26.9% | 4.2% | 4.57 | 0.1667 | 0.2 | vs v0.1: +12 itens certos |
