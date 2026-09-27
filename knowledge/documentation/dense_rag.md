# RAG com vetores densos (HEILO)

## Backends

| Backend | Tipo | Requisito |
|---|---|---|
| `lsa` | denso (SVD sobre TF-IDF) | só NumPy — padrão |
| `ollama` | denso neural | Ollama + modelo embed (`nomic-embed-text`) |
| `sentence-transformers` | denso neural | `pip install sentence-transformers` |
| `tfidf` | esparso normalizado | legado |

## Configuração

```bash
export HEILO_EMBED_BACKEND=auto   # lsa | ollama | st | tfidf
export HEILO_LSA_DIM=64
export HEILO_USE_ST=1             # força ST se instalado
export HEILO_OLLAMA_EMBED_MODEL=nomic-embed-text
export OLLAMA_HOST=http://127.0.0.1:11434
```

## Uso

```python
from heilo.rag.retriever import Retriever
r = Retriever(mode="hybrid")
r.index_knowledge(force=True)
r.search("desconto percentual", top_k=5)
```

API: `POST /api/search` · `POST /api/search/reindex`
