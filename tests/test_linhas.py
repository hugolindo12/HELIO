from heilo.models.linhas import linha, nome_modelo


def test_linhas_por_tamanho():
    assert linha(3_300_000) == "Faísca" and linha(30_000_000) == "Faísca"
    assert linha(120_000_000) == "Aurora" and linha(500_000_000) == "Zênite"
    assert nome_modelo(30_000_000, "v1.1") == "HEILO Faísca 1.1"
    assert nome_modelo(30_000_000, "v0.2", codigo=True) == "HEILO Faísca Code 0.2"
