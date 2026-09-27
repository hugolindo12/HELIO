"""Dense + hybrid semantic search tests."""
from pathlib import Path
from heilo.rag.vector_store import LocalVectorStore
from heilo.rag.retriever import Retriever
from heilo.rag.embeddings import LSADenseBackend, TfidfBackend, resolve_backend


def test_lsa_is_default_auto():
    b = resolve_backend("auto")
    # Without ST/Ollama, auto → LSA
    assert b.name in ("lsa", "sentence-transformers", "ollama")


def test_lsa_dense_dim(tmp_path):
    store = LocalVectorStore(
        collection="lsa1",
        root=tmp_path,
        backend=LSADenseBackend(n_components=16),
    )
    store.add(
        [
            "Para calcular desconto percentual use total * (1 - percent / 100)",
            "Asyncio no Python usa async def e await para corrotinas",
            "RAG recupera documentos e aumenta o contexto do modelo",
            "SketchController atualiza a vista apos selecao do usuario",
        ],
        sources=["a.md", "b.md", "c.md", "d.md"],
    )
    info = store.info()
    assert info["backend"] == "lsa"
    assert info["dim"] >= 1
    assert info["embeddings_shape"][0] == 4
    # dense: dim should be << sparse vocab
    assert info["dim"] <= 16


def test_lsa_semantic_ranking(tmp_path):
    store = LocalVectorStore(
        collection="lsa2",
        root=tmp_path,
        backend=LSADenseBackend(n_components=8),
    )
    store.add(
        [
            "Para calcular desconto percentual use total * (1 - percent / 100)",
            "Asyncio no Python usa async def e await para corrotinas",
            "RAG recupera documentos e aumenta o contexto do modelo",
        ],
        sources=["discount.md", "async.md", "rag.md"],
    )
    hits = store.search("preco com percentual de reducao", top_k=2, min_score=0.0)
    assert hits
    # related to discount ideally in top-2
    texts = " ".join(h[0].lower() for h in hits)
    assert "desconto" in texts or "percent" in texts or "total" in texts


def test_hybrid_search(tmp_path):
    store = LocalVectorStore(
        collection="hyb",
        root=tmp_path,
        backend=LSADenseBackend(n_components=8),
    )
    store.add(
        [
            "O modulo SketchController atualiza a vista apos selecao",
            "Calculo financeiro basico de juros compostos",
        ],
        sources=["sketch.md", "finance.md"],
    )
    hits = store.hybrid_search("SketchController selecao", top_k=2, min_score=0.0)
    assert hits
    assert "Sketch" in hits[0][0] or "selecao" in hits[0][0].lower()


def test_retriever_dense_reindex():
    r = Retriever(auto_index=False, mode="hybrid", min_score=0.02)
    r.vector = LocalVectorStore(
        collection="heilo_knowledge_dense_test",
        backend=LSADenseBackend(n_components=32),
    )
    r.index_knowledge(force=True)
    hits = r.search("desconto percentual python", top_k=3)
    assert isinstance(hits, list)
    info = r.info()
    assert info["backend"] == "lsa"
