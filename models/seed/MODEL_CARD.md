# HEILO Seed — Model Card

| Campo | Valor |
|---|---|
| Nome | HEILO Seed |
| Papel | Modelo próprio da HEILO (aluno / *student*) |
| Arquitetura | Transformer decoder (estilo GPT), 4 camadas, 4 cabeças, 256 dim, contexto 256 |
| Tokenizer | Bytes UTF-8 (256) + 4 tokens especiais (`<USUARIO>`, `<HEILO>`, `<FIM>`, `<DOC>`) |
| Parâmetros | ~3,3 milhões |
| Inicialização | **Aleatória (treinado do zero)**. Nenhum peso vem de modelo externo |
| Dados | Somente o dataset HEILO (`data/datasets/`), gerado a partir de exemplos **aprovados** |
| Código | `models/seed/gpt.py` |
| Pesos | `models/seed/weights/heilo_seed.pt` |
| Status | Experimental, com evolução gradual |

## Limitações honestas

- Com poucos dados, ele repete bem as respostas que já viu e erra bastante em assuntos novos.
- Ele não tem a capacidade de um modelo grande. O objetivo é evoluir aos poucos com mais dados aprovados e mais treino.
- Exemplos gerados pelo HEILO Teacher só entram no treino depois de validados e aprovados, e o manifesto de cada dataset registra quantos foram usados.
