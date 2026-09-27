"""Funções verificadas do HEILO Faísca Code: passam nos testes e não repetem a prova."""
import numpy as np

from heilo.training import pretrain, seed_code, tarefas_codigo


def test_tarefas_nao_repetem_a_prova_e_passam():
    codigos, rel = tarefas_codigo.exemplos(verificar=False)
    assert rel["exemplos"] >= 250
    prova = {n for n, _, _ in seed_code.PROBLEMAS}
    assert not any(c.split("(")[0].replace("def ", "") in prova for c in codigos)
    # amostra executada de verdade (a lista inteira roda no Colab antes do treino)
    for nomes, params, docs, corpo, testes in tarefas_codigo.TAREFAS[::10]:
        codigo = f'def {nomes[0]}({params}):\n    """{docs[0]}"""\n    {corpo.format()}\n'
        assert seed_code.executar_teste(codigo, testes.format(f=nomes[0])), nomes[0]


def test_mistura_com_peso_fixo():
    a, b = np.zeros(10_000, dtype=np.uint16), np.ones(500, dtype=np.uint16)
    cm = pretrain.CorpusMulti([a, b], [None, 0.3])
    assert np.allclose(cm._pesos(64), [0.7, 0.3])
    assert np.allclose(pretrain.CorpusMulti([a, b])._pesos(64), [10_000 / 10_500, 500 / 10_500])
