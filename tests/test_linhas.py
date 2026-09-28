from heilo.models.linhas import linha, nome_modelo


def test_linhas_por_tamanho():
    assert linha(3_300_000) == "Faísca" and linha(30_000_000) == "Faísca"
    assert linha(210_000_000) == "Faísca"
    assert linha(600_000_000) == "Aurora" and linha(2_000_000_000) == "Zênite"
    assert nome_modelo(30_000_000, "v1.1") == "HEILO Faísca 1.1"
    assert nome_modelo(30_000_000, "v0.2", codigo=True) == "HEILO Faísca Code 0.2"


def test_faisca_200m_tem_cerca_de_210M_e_e_faisca():
    import pytest
    pytest.importorskip("torch")
    from heilo.models.seed.gpt import MiniGPT
    from heilo.training.faisca_grande import CONFIG_FAISCA_200M, lote_para_gpu
    n = sum(p.numel() for p in MiniGPT(CONFIG_FAISCA_200M).parameters())
    assert 190e6 < n < 230e6
    assert linha(n) == "Faísca"
    for mem in (15, 22, 40):
        L = lote_para_gpu("x", mem)
        assert L["micro"] * L["acumular"] == 32


def test_tokenizer_separa_algarismos(tmp_path):
    import pytest
    pytest.importorskip("tokenizers")
    from heilo.training.faisca_grande import preparar_tokenizer
    textos = ["O ano de 1998 tinha 365 dias. G01 X10.5 M08. Seis vezes oito é 48. " * 40]
    tok = preparar_tokenizer(tmp_path, log=lambda s: None, textos=iter(textos))
    assert (tmp_path / "tokenizer.json").exists()
    ids = tok.encode("6 vezes 8 é 48 e G01")
    pedacos = [tok.tok.id_to_token(i) for i in ids]
    assert "48" not in pedacos and "Ġ48" not in pedacos and "01" not in pedacos
    assert tok.decode(ids) == "6 vezes 8 é 48 e G01"
    # na 2ª vez só carrega o que já existe
    assert preparar_tokenizer(tmp_path, log=lambda s: None, textos=iter([])).vocab_size == tok.vocab_size
