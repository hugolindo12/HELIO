# Padrão RAG (Retrieval-Augmented Generation)

Fluxo:
1. Pergunta do usuário
2. Retriever busca documentos relevantes (embeddings + vector DB)
3. Contexto reduzido é enviado ao modelo
4. Modelo gera resposta fundamentada

Benefícios: menos alucinação, conhecimento atualizável sem retreinar.
