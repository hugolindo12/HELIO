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


def test_crescer_profundidade_mantem_as_respostas(tmp_path):
    """A 200M esticada (mais camadas) responde IGUAL à original antes de treinar."""
    import dataclasses
    import pytest
    torch = pytest.importorskip("torch")
    from heilo.models.seed.gpt import GPTConfig, MiniGPT
    from heilo.training import faisca_grande
    cfg = GPTConfig(block_size=32, n_layer=2, n_head=4, n_embd=64, dropout=0.0, vocab_size=100,
                    arquitetura="moderna", n_kv_head=2)
    torch.manual_seed(0)
    m = MiniGPT(cfg)
    opt = torch.optim.AdamW(m.parameters())
    ck = tmp_path / "pre.pt"
    torch.save({"modelo": m.state_dict(), "opt": opt.state_dict(), "passo": 77, "hist": [],
                "config": cfg.__dict__, "inicio_agenda": 0, "minutos_agenda": 0.0}, ck)
    info = faisca_grande.crescer_profundidade(ck, tmp_path / "pre4.pt", 4, log=lambda s: None)
    st = torch.load(tmp_path / "pre4.pt", weights_only=False)
    g = MiniGPT(GPTConfig(**st["config"]))
    g.load_state_dict(st["modelo"])
    x = torch.randint(0, 100, (2, 16))
    m.eval(); g.eval()
    assert torch.allclose(m(x)[0], g(x)[0], atol=1e-5)
    assert st["passo"] == 77 and st["inicio_agenda"] == 77 and g.cfg.n_layer == 4
    n400 = sum(p.numel() for p in MiniGPT(faisca_grande.CONFIG_FAISCA_400M).parameters())
    assert 350e6 < n400 < 450e6 and linha(n400) == "Faísca"
    assert faisca_grande.lote_para_gpu("L4", 22, n_layer=32)["micro"] == 4


def test_plano_crescimento_16_para_24_e_32():
    from heilo.training import faisca_grande
    p24 = faisca_grande.plano_crescimento(16, 24)
    assert len(p24) == 24 and sum(c for _, c in p24) == 8
    assert [i for i, c in p24 if c] == [1, 3, 5, 7, 9, 11, 13, 15]   # espalhadas, não só no fim
    p32 = faisca_grande.plano_crescimento(16, 32)
    assert len(p32) == 32 and all(p32[2 * i] == (i, False) and p32[2 * i + 1] == (i, True) for i in range(16))
    assert len(faisca_grande.plano_crescimento(24, 32)) == 32
    assert faisca_grande.plano_crescimento(16, 16) == [(i, False) for i in range(16)]


def test_crescer_nao_multiplo_mantem_as_respostas(tmp_path):
    """4 → 6 camadas (como 16 → 24): responde igual antes de treinar; a 300M é Faísca."""
    import pytest
    torch = pytest.importorskip("torch")
    from heilo.models.seed.gpt import GPTConfig, MiniGPT
    from heilo.training import faisca_grande
    cfg = GPTConfig(block_size=32, n_layer=4, n_head=4, n_embd=64, dropout=0.0, vocab_size=100,
                    arquitetura="moderna", n_kv_head=2)
    torch.manual_seed(1)
    m = MiniGPT(cfg)
    ck = tmp_path / "pre.pt"
    torch.save({"modelo": m.state_dict(), "opt": {}, "passo": 5, "hist": [], "config": cfg.__dict__}, ck)
    faisca_grande.crescer_profundidade(ck, tmp_path / "pre6.pt", 6, log=lambda s: None)
    st = torch.load(tmp_path / "pre6.pt", weights_only=False)
    g = MiniGPT(GPTConfig(**st["config"]))
    g.load_state_dict(st["modelo"])
    x = torch.randint(0, 100, (2, 16))
    m.eval(); g.eval()
    assert g.cfg.n_layer == 6 and torch.allclose(m(x)[0], g(x)[0], atol=1e-5)
    n300 = sum(p.numel() for p in MiniGPT(faisca_grande.CONFIG_FAISCA_300M).parameters())
    assert 270e6 < n300 < 330e6 and linha(n300) == "Faísca"
