"""
HEILO Training — como os dados chegam ao HEILO Seed.

    HEILO Teacher ─┐
    /ensinar ──────┼─→ registro → validação → revisão/aprovação → dataset HEILO → treino → HEILO Seed
    Memory ────────┘

Pastas (heilo/data/):
    raw/        candidatos tirados da Memory (não verificados)
    teacher/    respostas geradas pelo Teacher (não verificadas)
    taught/     exemplos ensinados pelo usuário (/ensinar)
    approved/   exemplos aprovados — a ÚNICA fonte do dataset
    rejected/   exemplos rejeitados, com o motivo
    datasets/   versões do dataset HEILO + manifesto (o que entrou, de onde veio)
    training/   registro de cada treino e das comparações Seed × Teacher

Regras:
- Nada entra no treino sem estar em approved/.
- Saída do Teacher nunca é aprovada automaticamente (o Teacher também erra).
- /ensinar e /aprender não alteram pesos. Treinar é sempre um passo explícito.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional

from heilo.config import DATA_DIR, KNOWLEDGE_DIR
from heilo.models.base import TEACHER_REQUIRED_MSG
from heilo.training.records import (
    agora, append_jsonl, example_id, make_example, read_jsonl, write_jsonl,
)
from heilo.training.validation import MAX_CHARS, validate

JANELA_TURNOS = 4           # trocas anteriores usadas como contexto de um exemplo da Memory
FONTES_IGNORADAS = {"knowledge_miss"}


class TeacherRequired(RuntimeError):
    """A função pedida depende do HEILO Teacher, que está desativado/ausente."""

    def __init__(self, detalhe: str = ""):
        super().__init__(TEACHER_REQUIRED_MSG + (f" ({detalhe})" if detalhe else ""))


class TrainingPipeline:
    def __init__(self, data_dir: Optional[Path] = None, models=None, memory=None,
                 knowledge=None, seed_weights: Optional[Path] = None):
        self.root = Path(data_dir or DATA_DIR)
        self.models = models          # ModelManager (opcional: só o Teacher/Seed precisam)
        self.memory = memory          # MemoryManager
        self.knowledge = knowledge    # KnowledgeManager
        self.seed_weights = seed_weights
        for sub in ("raw", "teacher", "taught", "approved", "rejected", "datasets", "training"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ caminhos
    @property
    def raw_file(self) -> Path:
        return self.root / "raw" / "memory_candidates.jsonl"

    @property
    def teacher_file(self) -> Path:
        return self.root / "teacher" / "generated.jsonl"

    @property
    def taught_file(self) -> Path:
        return self.root / "taught" / "taught.jsonl"

    @property
    def approved_file(self) -> Path:
        return self.root / "approved" / "approved.jsonl"

    @property
    def rejected_file(self) -> Path:
        return self.root / "rejected" / "rejected.jsonl"

    @property
    def runs_file(self) -> Path:
        return self.root / "training" / "runs.jsonl"

    @property
    def comparisons_file(self) -> Path:
        return self.root / "training" / "comparisons.jsonl"

    # ------------------------------------------------------------- leitura
    def approved(self) -> List[Dict]:
        """Todos os exemplos aprovados (approved/*.jsonl), normalizados e sem duplicatas."""
        vistos, out = set(), []
        for arq in sorted((self.root / "approved").glob("*.jsonl")):
            for r in read_jsonl(arq):
                if not r.get("messages"):
                    continue
                r.setdefault("id", example_id(r["messages"]))
                r.setdefault("origin", "curated")
                if r["id"] in vistos:
                    continue
                vistos.add(r["id"])
                out.append(r)
        return out

    def rejected(self) -> List[Dict]:
        return read_jsonl(self.rejected_file)

    def _decididos(self) -> set:
        return {r["id"] for r in self.approved()} | {r.get("id") for r in self.rejected()}

    def pending(self) -> List[Dict]:
        """Candidatos (Memory + Teacher) ainda sem decisão."""
        decididos, vistos, out = self._decididos(), set(), []
        for r in read_jsonl(self.raw_file) + read_jsonl(self.teacher_file):
            if r["id"] in decididos or r["id"] in vistos:
                continue
            vistos.add(r["id"])
            out.append(r)
        return out

    # ----------------------------------------------------------- /ensinar
    def register_taught(self, question: str, answer: str) -> Dict:
        """Registra um exemplo supervisionado. NÃO treina e NÃO altera pesos."""
        ex = make_example(
            [{"role": "user", "content": (question or "").strip()},
             {"role": "assistant", "content": (answer or "").strip()}],
            origin="taught", meta={"autor": "usuario", "via": "/ensinar"},
        )
        ja_aprovado = ex["id"] in {r["id"] for r in self.approved()}
        ok, erros = validate(ex)
        ex["validation"] = {"ok": ok, "errors": erros}
        append_jsonl(self.taught_file, [ex])
        if not ok:
            append_jsonl(self.rejected_file, [dict(ex, rejected_at=agora(),
                                                   rejected_by="validacao", reasons=erros)])
            return {"ok": False, "errors": erros, "example": ex}
        if not ja_aprovado:
            # O usuário é a fonte da verdade do que ele mesmo ensina → aprovado.
            append_jsonl(self.approved_file, [dict(ex, approved_at=agora(),
                                                   approved_by="usuario (/ensinar)")])
        knowledge_path = None
        if self.knowledge is not None and not ja_aprovado:
            knowledge_path = str(self.knowledge.add_taught(question, answer))
        return {"ok": True, "errors": [], "example": ex, "ja_existia": ja_aprovado,
                "knowledge": knowledge_path}

    # ------------------------------------------------------ Memory → raw
    def collect_from_memory(self) -> int:
        """Seleciona trocas da Memory como CANDIDATOS (vão para revisão, não para o treino)."""
        if self.memory is None:
            return 0
        conhecidos = self._decididos() | {r["id"] for r in read_jsonl(self.raw_file)}
        novos = []
        for sessao, trocas in self.memory.sessions().items():
            historico: List[Dict] = []
            for t in trocas:
                user, resp = t.get("user", ""), t.get("assistant", "")
                util = (t.get("tipo", "message") == "message"
                        and t.get("fonte") not in FONTES_IGNORADAS
                        and 0 < len(resp) <= MAX_CHARS)
                if util:
                    msgs = historico[-2 * JANELA_TURNOS:] + [
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": resp},
                    ]
                    ex = make_example(msgs, origin="memory", meta={
                        "sessao": sessao, "ts": t.get("ts"), "fonte": t.get("fonte", ""),
                        "modelo": t.get("modelo", ""),
                    })
                    if ex["id"] not in conhecidos:
                        conhecidos.add(ex["id"])
                        novos.append(ex)
                historico += [{"role": "user", "content": user},
                              {"role": "assistant", "content": resp[:MAX_CHARS]}]
        return append_jsonl(self.raw_file, novos) if novos else 0

    # --------------------------------------------------- Teacher → teacher/
    def generate_with_teacher(self, prompts: List[str],
                              log: Callable[[str], None] = lambda *_: None) -> Dict:
        """Pede ao Teacher respostas para perguntas. Resultado: dados NÃO verificados."""
        if self.models is None or not self.models.teacher_available():
            detalhe = ""
            if self.models is not None:
                detalhe = self.models.get("teacher").availability()[1]
            raise TeacherRequired(detalhe)
        card = self.models.get("teacher").card().to_dict()
        conhecidos = self._decididos() | {r["id"] for r in read_jsonl(self.teacher_file)}
        gerados, falhas = [], []
        for p in [p.strip() for p in prompts if p and p.strip()]:
            r = self.models.generate_with("teacher", [{"role": "user", "content": p}])
            if not r.text:
                falhas.append(p)
                continue
            ex = make_example(
                [{"role": "user", "content": p}, {"role": "assistant", "content": r.text}],
                origin="teacher",
                meta={"status": "nao_verificado", "teacher": card.get("name"),
                      "base_model": card.get("base_model"), "adapter": card.get("adapter"),
                      "license": card.get("license")},
            )
            if ex["id"] in conhecidos:
                continue
            conhecidos.add(ex["id"])
            gerados.append(ex)
            log(f"[Teacher] {p[:50]} → {r.text[:60]}")
        append_jsonl(self.teacher_file, gerados)
        return {"gerados": len(gerados), "falhas": falhas}

    # ------------------------------------------------------ validação
    def validate_pending(self) -> Dict:
        """Rejeita automaticamente candidatos com problemas objetivos. Os válidos
        continuam PENDENTES de revisão humana."""
        aprovados = {r["id"] for r in self.approved()}
        rejeitar, validos = [], 0
        for ex in self.pending():
            ok, erros = validate(ex, known_ids=aprovados)
            if ok:
                validos += 1
            else:
                rejeitar.append(dict(ex, rejected_at=agora(), rejected_by="validacao",
                                     reasons=erros))
        append_jsonl(self.rejected_file, rejeitar)
        return {"validos_pendentes": validos, "rejeitados_auto": len(rejeitar)}

    def review(self, ex_id: str, approve: bool, reason: str = "",
               edited_answer: Optional[str] = None) -> Dict:
        """Decisão humana sobre um candidato (/revisar)."""
        ex = next((r for r in self.pending() if r["id"] == ex_id), None)
        if ex is None:
            return {"ok": False, "error": f"candidato {ex_id} não está pendente"}
        if not approve:
            append_jsonl(self.rejected_file, [dict(ex, rejected_at=agora(),
                                                   rejected_by="usuario", reasons=[reason or "rejeitado"])])
            return {"ok": True, "decisao": "rejeitado"}
        final = ex
        if edited_answer and edited_answer.strip():
            msgs = [dict(m) for m in ex["messages"]]
            msgs[-1]["content"] = edited_answer.strip()
            final = make_example(msgs, origin=ex["origin"],
                                 meta=dict(ex.get("meta", {}), editado_de=ex["id"]))
            append_jsonl(self.rejected_file, [dict(ex, rejected_at=agora(), rejected_by="usuario",
                                                   reasons=["substituído por versão editada"])])
        ok, erros = validate(final, known_ids={r["id"] for r in self.approved()})
        if not ok:
            return {"ok": False, "error": "; ".join(erros)}
        append_jsonl(self.approved_file, [dict(final, approved_at=agora(), approved_by="usuario")])
        return {"ok": True, "decisao": "aprovado", "id": final["id"]}

    # -------------------------------------------------------- dataset
    def _versoes(self) -> List[Path]:
        return sorted((self.root / "datasets").glob("heilo_v*.manifest.json"))

    def latest_dataset(self) -> Optional[Dict]:
        v = self._versoes()
        if not v:
            return None
        return json.loads(v[-1].read_text(encoding="utf-8"))

    def build_dataset(self, include_knowledge: bool = False,
                      knowledge_dir: Optional[Path] = None) -> Dict:
        """Gera uma nova versão do dataset HEILO SÓ com exemplos aprovados.

        ~10% vão para validação (escolhidos pelo id, de forma determinística) e
        nunca são vistos no treino — é com eles que medimos o Seed.
        """
        exemplos = self.approved()
        num = len(self._versoes()) + 1
        base = self.root / "datasets" / f"heilo_v{num:03d}"
        treino = [e for e in exemplos if int(e["id"], 16) % 10 != 0]
        valid = [e for e in exemplos if int(e["id"], 16) % 10 == 0]
        if not treino and valid:  # dataset minúsculo: treina com tudo
            treino, valid = valid, []

        def linha(e):
            return {"id": e["id"], "origin": e.get("origin", "curated"), "messages": e["messages"]}

        write_jsonl(base.with_suffix(".jsonl"), map(linha, treino))
        write_jsonl(Path(str(base) + ".val.jsonl"), map(linha, valid))

        corpus_path = None
        if include_knowledge:
            kdir = Path(knowledge_dir or KNOWLEDGE_DIR)
            textos = [p.read_text(encoding="utf-8", errors="replace").strip()
                      for p in sorted(kdir.rglob("*.md"))]
            corpus_path = Path(str(base) + ".corpus.txt")
            corpus_path.write_text("\n\n".join(t for t in textos if t), encoding="utf-8")

        por_origem: Dict[str, int] = {}
        for e in treino:
            por_origem[e.get("origin", "curated")] = por_origem.get(e.get("origin", "curated"), 0) + 1
        manifest = {
            "version": num,
            "created": agora(),
            "train_file": str(base.with_suffix(".jsonl").name),
            "val_file": Path(str(base) + ".val.jsonl").name,
            "corpus_file": corpus_path.name if corpus_path else None,
            "train_examples": len(treino),
            "val_examples": len(valid),
            "por_origem": por_origem,
            "teacher_examples": por_origem.get("teacher", 0),
            "include_knowledge": include_knowledge,
        }
        Path(str(base) + ".manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        return manifest

    # ----------------------------------------------------------- /aprender
    def learn(self) -> Dict:
        """/aprender: coleta → seleciona → valida → dataset. NÃO treina, NÃO chama o
        Teacher e NÃO envia nada para o GitHub."""
        coletados = self.collect_from_memory()
        val = self.validate_pending()
        manifest = self.build_dataset()
        return {
            "coletados_da_memoria": coletados,
            **val,
            "pendentes_revisao": len(self.pending()),
            "aprovados": len(self.approved()),
            "dataset": manifest,
        }

    # ------------------------------------------------------------- treino
    def train_seed(self, passos: int = 2000, novo: bool = False, lote: int = 16,
                   log: Callable[[str], None] = print, **kw) -> Dict:
        """Treina o HEILO Seed com a última versão do dataset (passo explícito)."""
        from heilo.models.seed import SEED_WEIGHTS, gpt

        manifest = self.latest_dataset() or self.build_dataset()
        if manifest["train_examples"] == 0:
            raise RuntimeError("Dataset vazio: aprove exemplos antes (/ensinar, /revisar).")
        pasta = self.root / "datasets"
        val = read_jsonl(pasta / manifest["val_file"])
        corpus = pasta / manifest["corpus_file"] if manifest.get("corpus_file") else None
        pesos = Path(self.seed_weights or SEED_WEIGHTS)
        r = gpt.treinar(passos=passos, lote=lote, novo=novo, arquivo=pesos,
                        dataset_file=pasta / manifest["train_file"], corpus_file=corpus,
                        avaliacao=val, log=log, **kw)
        run = {
            "ts": agora(),
            "modelo": "HEILO Seed",
            "dataset_version": manifest["version"],
            "passos": passos,
            "passos_totais": r["passos_totais"],
            "perda_treino": None if r["perda"] is None else round(r["perda"], 4),
            "perda_validacao": None if r["perda_validacao"] is None else round(r["perda_validacao"], 4),
            "exemplos_treino": manifest["train_examples"],
            "exemplos_validacao": manifest["val_examples"],
            "por_origem": manifest["por_origem"],
            "teacher_examples_used": manifest["teacher_examples"],
        }
        append_jsonl(self.runs_file, [run])
        if self.models is not None:
            self.models.reload()
        return run

    # ----------------------------------------------------- comparação
    def log_comparison(self, prompt: str, outputs: Dict[str, str],
                       verdict: Optional[str] = None) -> Dict:
        """Registra uma comparação Seed × Teacher. O veredito é SEU (ou nenhum)."""
        row = {"ts": agora(), "prompt": prompt, "outputs": outputs, "verdict": verdict}
        append_jsonl(self.comparisons_file, [row])
        return row

    # ---------------------------------------------------------- métricas
    def metrics(self) -> Dict:
        """Somente contagens reais dos arquivos — nenhuma métrica inventada."""
        aprovados = self.approved()
        rejeitados = self.rejected()
        gerados_teacher = read_jsonl(self.teacher_file)
        runs = read_jsonl(self.runs_file)
        comps = read_jsonl(self.comparisons_file)
        vered: Dict[str, int] = {}
        for c in comps:
            if c.get("verdict"):
                vered[c["verdict"]] = vered.get(c["verdict"], 0) + 1

        def conta(rows, origem):
            return sum(1 for r in rows if r.get("origin") == origem)

        return {
            "teacher": {
                "gerados": len(gerados_teacher),
                "aprovados": conta(aprovados, "teacher"),
                "rejeitados": conta(rejeitados, "teacher"),
                "pendentes": conta(self.pending(), "teacher"),
                "usados_no_ultimo_treino": runs[-1]["teacher_examples_used"] if runs else 0,
            },
            "aprovados_por_origem": {o: conta(aprovados, o)
                                     for o in ("curated", "taught", "teacher", "memory")},
            "pendentes_revisao": len(self.pending()),
            "rejeitados": len(rejeitados),
            "treinos_seed": [
                {k: r.get(k) for k in ("ts", "dataset_version", "passos_totais",
                                       "perda_treino", "perda_validacao", "teacher_examples_used")}
                for r in runs[-10:]
            ],
            "comparacoes": {"total": len(comps), "vereditos": vered},
        }
