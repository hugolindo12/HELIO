"""Memória verificada: responde com exemplos APROVADOS, sem confundir perguntas parecidas."""
from heilo.tools.memoria_verificada import MemoriaVerificada
from heilo.training.pipeline import TrainingPipeline
from heilo.training.records import append_jsonl, make_example


def _pipe(tmp_path, pares):
    tp = TrainingPipeline(data_dir=tmp_path)
    append_jsonl(tp.approved_file, [make_example([{"role": "user", "content": q}, {"role": "assistant", "content": a}],
                                                 origin="curated") for q, a in pares])
    return tp


def test_acha_equivalente_e_nao_confunde(tmp_path):
    tp = _pipe(tmp_path, [("qual é a capital da itália?", "A capital da Itália é Roma."),
                          ("qual é a capital da alemanha?", "A capital da Alemanha é Berlim."),
                          ("o que faz o g04?", "O G04 faz uma pausa."),
                          ("Responda só com o código: qual código cancela a compensação?", "G40"),
                          ("o que é o torno?", "Máquina em que a peça gira."), ("o que é a fresadora?", "Máquina em que a ferramenta gira.")])
    m = MemoriaVerificada(tp)
    assert m.buscar("capital da itália")["resposta"].endswith("Roma.")
    assert m.buscar("qual a capital da frança?") is None             # palavra importante diferente
    assert m.buscar("pra que serve o G04")["resposta"] == "O G04 faz uma pausa."
    assert m.buscar("qual código cancela a compensação?") is None     # exemplo de formato não vale
    assert m.buscar("oi") is None


def test_aprendizado_entra_na_hora(tmp_path):
    tp = _pipe(tmp_path, [("o que é o torno?", "Máquina em que a peça gira.")])
    m = MemoriaVerificada(tp, recarregar_seg=0)
    assert m.buscar("qual o nome do meu cachorro?") is None
    tp.register_taught("qual o nome do meu cachorro?", "O nome do seu cachorro é Thor.")
    r = m.buscar("qual é o nome do meu cachorro")
    assert r is None or "Thor" in r["resposta"]
