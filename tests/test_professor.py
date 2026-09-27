"""Modo professor: o Teacher só gera candidatos PENDENTES (nada é aprovado sozinho)."""
from types import SimpleNamespace

from heilo.training.pipeline import TrainingPipeline
from heilo.training.professor import gerar_com_professor


class _FakeTeacher:
    def __init__(self, respostas):
        self.respostas = list(respostas)

    def teacher_available(self):
        return True

    def get(self, _k):
        return SimpleNamespace(card=lambda: SimpleNamespace(to_dict=lambda: {"base_model": "fake", "license": "x"}),
                               availability=lambda: (True, "ok"))

    def generate_with(self, _k, _msgs, **_kw):
        return SimpleNamespace(text=self.respostas.pop(0) if self.respostas else "")


def test_professor_gera_pendentes_e_retoma(tmp_path):
    q = tmp_path / "q.txt"
    q.write_text("# comentário\nconhecimento|qual é a capital da itália?\n"
                 "conhecimento|qual é a capital da alemanha?\n", encoding="utf-8")
    tp = TrainingPipeline(data_dir=tmp_path)
    fake = _FakeTeacher(["A capital da Itália é Roma.", "A capital da Alemanha é Berlim."])
    r = gerar_com_professor(tp, fake, arquivo=q, log=lambda *_: None)
    assert r["pendentes_revisao"] + r["rejeitadas_auto"] == 2
    assert not tp.approved()                       # nada aprovado automaticamente
    r2 = gerar_com_professor(tp, _FakeTeacher([]), arquivo=q, log=lambda *_: None)
    assert r2["ja_feitas"] == 2                    # retoma sem repetir


def test_professor_para_se_o_teacher_nao_responde(tmp_path):
    q = tmp_path / "q.txt"
    q.write_text("\n".join(f"conhecimento|pergunta numero {i} sobre metais?" for i in range(10)), encoding="utf-8")
    tp = TrainingPipeline(data_dir=tmp_path)
    r = gerar_com_professor(tp, _FakeTeacher([]), arquivo=q, log=lambda *_: None)
    assert r["sem_resposta"] == 3 and not tp.pending()
