"""
Ciclos automáticos de treino do HEILO Seed com o HEILO Teacher.

    avaliação da versão ativa ─→ diagnóstico (categorias fracas)
        ─→ geração de dados (Teacher + programáticos)
        ─→ validação/limpeza (training/qualidade.py)
        ─→ dataset HEILO (só aprovados)
        ─→ treino de uma versão CANDIDATA (v0.N)
        ─→ avaliação no conjunto fixo (nunca treinado)
        ─→ promoção SÓ se melhorou de forma mensurável
        ─→ repete enquanto houver ganho

Versões: models/seed/versions/v0.N/heilo_seed.pt + manifest.json (a ativa é
copiada para models/seed/weights/heilo_seed.pt). Registro: data/training/versions.json.
Relatórios: data/training/eval/<versão>.json e data/training/relatorio_ciclos.md.

Uso:  python -m heilo.main ciclo [--max-ciclos 4] [--passos 2000] [--sem-professor]
"""
from __future__ import annotations

import json
import random
import shutil
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

from heilo.config import DATA_DIR
from heilo.models.seed import SEED_DIR, SEED_WEIGHTS
from heilo.training import geradores
from heilo.training.eval_set import EVAL_VERSION, SONDA_VERSION, load_eval, load_sonda
from heilo.training.evaluate import avaliar_seed
from heilo.training.qualidade import avaliar_exemplo
from heilo.training.records import agora, append_jsonl, make_example, read_jsonl

VERSIONS_DIR = SEED_DIR / "versions"
# Ganho mínimo para promover: +1 item certo no conjunto fixo, ou mesma taxa de
# acerto com perplexidade pelo menos 3% menor.
GANHO_MIN_ITENS = 1
GANHO_MIN_PPL = 0.03
# Se a sonda melhora mas o heilo_eval_v1 cai mais que isto, a promoção fica para o usuário.
PERDA_MAX_EVAL = 5


class CiclosSeed:
    def __init__(self, pipeline, models=None, data_dir: Optional[Path] = None,
                 versions_dir: Optional[Path] = None, pesos_ativos: Optional[Path] = None,
                 log: Callable[[str], None] = print, semente: int = 1234):
        self.p = pipeline
        self.models = models
        self.root = Path(data_dir or DATA_DIR)
        self.vdir = Path(versions_dir or VERSIONS_DIR)
        self.ativos = Path(pesos_ativos or SEED_WEIGHTS)
        self.reg_file = self.root / "training" / "versions.json"
        self.eval_dir = self.root / "training" / "eval"
        self.log_file = self.root / "training" / "ciclos.log"
        self._log = log
        self.rnd = random.Random(semente)
        self.itens_eval = load_eval()
        self.itens_sonda = load_sonda()
        self.eval_qs = [i["pergunta"] for i in self.itens_eval]
        for d in (self.vdir, self.eval_dir):
            d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ utilidades
    def log(self, msg: str) -> None:
        linha = f"[{time.strftime('%H:%M:%S')}] {msg}"
        self._log(linha)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(linha + "\n")

    def registro(self) -> Dict:
        if self.reg_file.exists():
            return json.loads(self.reg_file.read_text(encoding="utf-8"))
        return {"ativa": None, "eval": EVAL_VERSION, "versoes": []}

    def _salvar_registro(self, reg: Dict) -> None:
        self.reg_file.parent.mkdir(parents=True, exist_ok=True)
        self.reg_file.write_text(json.dumps(reg, indent=2, ensure_ascii=False), encoding="utf-8")

    def _versao(self, reg: Dict, nome: str) -> Optional[Dict]:
        return next((v for v in reg["versoes"] if v["versao"] == nome), None)

    def pasta_versao(self, nome: str) -> Path:
        d = self.vdir / nome
        d.mkdir(parents=True, exist_ok=True)
        return d

    def arquivo_versao(self, nome: str) -> Path:
        return self.pasta_versao(nome) / "heilo_seed.pt"

    def _escrever_manifesto(self, entrada: Dict) -> None:
        pasta = self.pasta_versao(entrada["versao"])
        (pasta / "manifest.json").write_text(
            json.dumps(entrada, indent=2, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def _num(versao: str):
        maj, men = versao.lstrip("v").split(".")[:2]
        return int(maj), int(men)

    def proxima_versao(self, reg: Dict, major: Optional[int] = None) -> str:
        """Próxima versão na mesma linha da ativa (v0.x → v0.x+1; v1.x → v1.x+1)."""
        if major is None:
            major = self._num(reg["ativa"])[0] if reg.get("ativa") else 0
        mesmos = [self._num(v["versao"])[1] for v in reg["versoes"]
                  if self._num(v["versao"])[0] == major]
        return f"v{major}.{(max(mesmos) + 1) if mesmos else 0}"

    # ------------------------------------------------------------- avaliação
    def avaliar_arquivo(self, pesos: Path, nome: str) -> Dict:
        import torch
        from heilo.models.seed.gpt import MiniGPTChat
        torch.manual_seed(0)
        chat = MiniGPTChat(arquivo=pesos)
        r = avaliar_seed(chat, self.itens_eval)
        r.update({"versao": nome, "eval": EVAL_VERSION, "data": agora(),
                  "tokenizer": getattr(getattr(chat.modelo, "tokenizer", None), "tipo", "byte")})
        if self.itens_sonda:
            torch.manual_seed(0)
            s = avaliar_seed(chat, self.itens_sonda)
            r["sonda"] = {k: s[k] for k in ("itens", "acerto", "acerto_amostrado", "consistencia",
                                            "por_categoria")}
            r["sonda"]["regua"] = SONDA_VERSION
            r["sonda_detalhes"] = s["detalhes"]
        (self.eval_dir / f"{nome}.json").write_text(
            json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
        if r.get("sonda"):
            self.log(f"Sonda independente {nome}: acerto {r['sonda']['acerto']:.3f}")
        self.log(f"Avaliação {nome}: acerto {r['acerto']:.3f} | perplexidade {r['perplexidade']} | "
                 f"instruções {r['seguir_instrucoes']} | generalização {r['generalizacao']} | "
                 f"consistência {r['consistencia']}")
        return r

    @staticmethod
    def resumo(r: Dict) -> Dict:
        return {k: r.get(k) for k in ("acerto", "acerto_amostrado", "consistencia", "perda",
                                      "perplexidade", "seguir_instrucoes", "generalizacao",
                                      "parafrase", "por_categoria", "itens", "sonda",
                                      "perda_por_caractere", "tokenizer")}

    def garantir_base(self) -> Dict:
        """Registra a versão atual (pesos ativos) como a primeira versão, se ainda não houver."""
        reg = self.registro()
        if reg["versoes"]:
            return reg
        if not self.ativos.exists():
            raise RuntimeError(f"Não há pesos do Seed em {self.ativos}. Treine uma vez antes.")
        destino = self.arquivo_versao("v0.1")
        shutil.copy2(self.ativos, destino)
        r = self.avaliar_arquivo(destino, "v0.1")
        entrada = {"versao": "v0.1", "arquivo": "v0.1/heilo_seed.pt", "criada": agora(),
                   "status": "ativa", "pai": None, "ciclo": 0,
                   "dataset_version": None, "origem": "modelo existente (base)",
                   "eval": self.resumo(r)}
        reg["versoes"].append(entrada)
        self._escrever_manifesto(entrada)
        reg["ativa"] = "v0.1"
        self._salvar_registro(reg)
        return reg

    def registrar_versao_externa(self, pesos: Path, nome: str, ciclo: Optional[int], pai: Optional[str],
                                 dataset_version: Optional[int], origem: str,
                                 promover: bool = True, extra: Optional[Dict] = None) -> Dict:
        """Registra uma versão treinada fora do ciclo (ex.: na nuvem), com a mesma avaliação."""
        reg = self.garantir_base()
        destino = self.arquivo_versao(nome)
        shutil.copy2(pesos, destino)
        r = self.avaliar_arquivo(destino, nome)
        dec = self._decidir(reg, nome, destino, r, ciclo, pai, dataset_version, origem, promover)
        if extra:
            reg = self.registro()
            self._versao(reg, nome).update(extra)
            self._salvar_registro(reg)
            self._escrever_manifesto(self._versao(reg, nome))
        return dec

    def promover_manual(self, nome: str, motivo: str) -> Dict:
        """Promove uma versão por decisão humana (registra o motivo)."""
        reg = self.registro()
        v = self._versao(reg, nome)
        if v is None or not v.get("arquivo"):
            raise ValueError(f"versão {nome} não existe")
        ativa = self._versao(reg, reg["ativa"])
        ativa["status"] = "anterior"
        v["status"] = "ativa"
        v["promocao_manual"] = {"data": agora(), "motivo": motivo}
        reg["ativa"] = nome
        shutil.copy2(self.vdir / v["arquivo"], self.ativos)
        self._salvar_registro(reg)
        self._escrever_manifesto(v)
        if self.models is not None:
            self.models.reload()
        self.log(f"PROMOVIDA MANUALMENTE {nome}: {motivo}")
        return v

    # ------------------------------------------------------------ decisão
    @staticmethod
    def _ganho_perda(novo: Dict, antigo: Dict) -> Optional[float]:
        """Ganho relativo de perda, só entre medidas comparáveis (perda por caractere,
        ou perplexidade por token quando os dois usam o mesmo tokenizer)."""
        if novo.get("perda_por_caractere") and antigo.get("perda_por_caractere"):
            return 1 - novo["perda_por_caractere"] / antigo["perda_por_caractere"]
        if (novo.get("tokenizer", "byte") == antigo.get("tokenizer", "byte")
                and novo.get("perplexidade") and antigo.get("perplexidade")):
            return 1 - novo["perplexidade"] / antigo["perplexidade"]
        return None

    def _melhorou(self, novo: Dict, antigo: Dict) -> (bool, str):
        """Promove só com ganho medido. Com sonda independente nas duas versões: a sonda
        não pode piorar; ganho na sonda promove (se o heilo_eval_v1 não cair mais que
        PERDA_MAX_EVAL itens); sem mudança na sonda, decide o heilo_eval_v1."""
        n = novo["itens"]
        d_itens = round((novo["acerto"] - antigo["acerto"]) * n)
        sn, sa = novo.get("sonda"), antigo.get("sonda")
        if sn and sa:
            # A sonda independente mede GENERALIZAÇÃO (perguntas que nunca entram no treino)
            # e manda na decisão; o heilo_eval_v1 (parecido com o treino) é desempate/limite.
            d_sonda = round((sn["acerto"] - sa["acerto"]) * sn["itens"])
            txt = f"sonda {d_sonda:+d}, eval {d_itens:+d} itens"
            if d_sonda < 0:
                return False, f"{txt} (sonda independente piorou)"
            if d_sonda >= GANHO_MIN_ITENS:
                if d_itens < -PERDA_MAX_EVAL:
                    return False, f"{txt} (generalizou melhor, mas perdeu muito do que sabia: decisão manual)"
                return True, txt
            if d_itens >= GANHO_MIN_ITENS:
                return True, txt
            ganho = self._ganho_perda(novo, antigo)
            if d_itens == 0 and ganho is not None and ganho >= GANHO_MIN_PPL:
                return True, f"{txt}, perda por caractere {ganho:.1%} menor"
            return False, f"{txt} (sem ganho mensurável)"
        if d_itens >= GANHO_MIN_ITENS:
            return True, f"+{d_itens} itens certos"
        ganho = self._ganho_perda(novo, antigo)
        if d_itens == 0 and ganho is not None and ganho >= GANHO_MIN_PPL:
            return True, f"mesmo acerto, perda {ganho:.1%} menor"
        return False, f"{d_itens:+d} itens certos (sem ganho mensurável)"

    def _decidir(self, reg, nome, arquivo, r, ciclo, pai, dataset_version, origem, promover=True):
        ativa = self._versao(reg, reg["ativa"])
        if (self.itens_sonda and ativa.get("arquivo") and
                (not ativa["eval"].get("sonda") or not ativa["eval"].get("perda_por_caractere")
                 or ativa["eval"]["sonda"].get("regua") != SONDA_VERSION)):
            # versão antiga sem sonda: mede agora, com o mesmo método, para comparar igual
            ra = self.avaliar_arquivo(self.vdir / ativa["arquivo"], ativa["versao"])
            ativa["eval"] = self.resumo(ra)
        ok, motivo = self._melhorou(self.resumo(r), ativa["eval"])
        entrada = {"versao": nome, "arquivo": f"{nome}/{arquivo.name}", "criada": agora(), "pai": pai,
                   "ciclo": ciclo, "dataset_version": dataset_version, "origem": origem,
                   "eval": self.resumo(r), "comparacao": f"vs {ativa['versao']}: {motivo}"}
        if ok and promover:
            ativa["status"] = "anterior"
            entrada["status"] = "ativa"
            reg["ativa"] = nome
            shutil.copy2(arquivo, self.ativos)
            self.log(f"PROMOVIDA {nome} ({motivo}). Ela é a nova versão ativa.")
        else:
            entrada["status"] = "rejeitada"
            self.log(f"{nome} NÃO promovida ({motivo}). Continua a {reg['ativa']}.")
        reg["versoes"].append(entrada)
        self._salvar_registro(reg)
        self._escrever_manifesto(entrada)
        if self.models is not None:
            self.models.reload()
        return {"promovida": ok and promover, "motivo": motivo, "versao": nome, "eval": r}

    # -------------------------------------------------------- geração de dados
    def _teacher_fn(self):
        if self.models is None or not self.models.teacher_available():
            return None

        def fn(msgs, sistema, temperatura, max_novos):
            r = self.models.generate_with("teacher", msgs, system=sistema,
                                          temperatura=temperatura, max_novos=max_novos)
            return r.text
        return fn

    def gerar_dados(self, ciclo: int, diagnostico: Dict[str, float], base_n: int = 20,
                    usar_professor: bool = True) -> Dict:
        """Gera, valida e separa exemplos novos. Retorna contagens reais."""
        aprovados_ids = {r["id"] for r in self.p.approved()}
        conhecidos = {m["content"].strip().lower()
                      for r in self.p.approved() for m in r["messages"] if m["role"] == "user"}
        card = {}
        teacher = self._teacher_fn() if usar_professor else None
        if teacher:
            card = self.models.get("teacher").card().to_dict()

        def peso(cat):  # categorias mais fracas recebem mais exemplos
            return 0.5 + diagnostico.get(cat, 0.5)

        candidatos: List[Dict] = []   # {q, r, categoria, origem, factual, meta}

        # programáticos (respostas corretas por construção)
        for q, r in geradores.matematica(int(base_n * 2 * peso("matematica")), self.rnd, self.eval_qs):
            candidatos.append(dict(q=q, r=r, categoria="matematica", origem="programatico", factual=False))
        for q, r in geradores.instrucoes(self.eval_qs, self.rnd):
            candidatos.append(dict(q=q, r=r, categoria="instrucao", origem="programatico", factual=False))
        for q, r in geradores.honestidade(self.eval_qs):
            candidatos.append(dict(q=q, r=r, categoria="honestidade", origem="programatico", factual=False))
        for q, r in geradores.identidade(self.eval_qs):
            candidatos.append(dict(q=q, r=r, categoria="identidade", origem="programatico", factual=False))

        # HEILO Teacher
        n_teacher = 0
        if teacher:
            self.log(f"Teacher ({card.get('base_model')}) gerando exemplos...")
            exemplos_treino: Dict[str, List[str]] = {}
            for r in self.p.approved():
                cat = r.get("categoria") or r.get("meta", {}).get("tema")
                if cat:
                    exemplos_treino.setdefault(cat, []).append(r["messages"][0]["content"])
            perguntas: List[tuple] = []
            for cat in ("conversa", "python", "ferramentas"):
                n = int(base_n * peso(cat))
                ex = exemplos_treino.get(cat) or exemplos_treino.get("saudacao", [])[:4] or ["oi, tudo bem?"]
                for q in geradores.perguntas_teacher(teacher, cat, self.rnd.sample(ex, min(4, len(ex))),
                                                     n, self.eval_qs, conhecidos):
                    perguntas.append((q, cat, cat != "conversa", geradores.SISTEMA_RESPOSTA))
            for q in geradores.perguntas_cnc(int(base_n * peso("cnc")), self.rnd, self.eval_qs):
                perguntas.append((q, "cnc", True, geradores.SISTEMA_RESPOSTA))
            for q in geradores.perguntas_raciocinio(int(base_n * 0.5 * peso("matematica")), self.rnd, self.eval_qs):
                perguntas.append((q, "raciocinio", True, geradores.SISTEMA_RACIOCINIO))
            for i, (q, cat, factual, sistema) in enumerate(perguntas, 1):
                resp = teacher([{"role": "user", "content": q}], sistema, 0.6, 96)
                n_teacher += 1
                if i % 10 == 0:
                    self.log(f"  Teacher: {i}/{len(perguntas)} respostas")
                candidatos.append(dict(q=q, r=resp, categoria=cat, origem="teacher", factual=factual))
        elif usar_professor:
            self.log("HEILO Teacher indisponível: ciclo só com dados programáticos.")

        # validação / limpeza
        cont = {"gerados": len(candidatos), "teacher_chamadas": n_teacher,
                "aprovados": 0, "revisao": 0, "rejeitados": 0, "por_origem": {}}
        aprov, rev, rej = [], [], []
        for c in candidatos:
            # programáticos são corretos por construção; o Teacher é sempre verificado
            v = avaliar_exemplo(c["q"], c["r"], c["categoria"], self.eval_qs, conhecidos,
                                factual=c["factual"],
                                verificar_fatos=c["origem"] == "teacher")
            ex = make_example([{"role": "user", "content": c["q"]},
                               {"role": "assistant", "content": (c["r"] or "").strip()}],
                              origin="teacher" if c["origem"] == "teacher" else "curated",
                              meta={"gerador": c["origem"], "ciclo": ciclo,
                                    "professor": card.get("base_model") if c["origem"] == "teacher" else None,
                                    "adapter": card.get("adapter") if c["origem"] == "teacher" else None})
            ex.update({"categoria": c["categoria"], "qualidade": v["qualidade"],
                       "validado": v["status"] == "aprovado", "validacao": v,
                       "lote": f"ciclo_{ciclo}", "data": agora()})
            if ex["id"] in aprovados_ids:
                continue
            o = cont["por_origem"].setdefault(c["origem"], {"aprovados": 0, "revisao": 0, "rejeitados": 0})
            if v["status"] == "aprovado":
                conhecidos.add(c["q"].strip().lower())
                aprovados_ids.add(ex["id"])
                aprov.append(dict(ex, approved_at=agora(), approved_by="validação automática (ciclo)"))
                o["aprovados"] += 1
            elif v["status"] == "revisao":
                rev.append(ex)
                o["revisao"] += 1
            else:
                rej.append(dict(ex, rejected_at=agora(), rejected_by="validação automática",
                                reasons=v["problemas"]))
                o["rejeitados"] += 1
        append_jsonl(self.root / "approved" / f"ciclo_{ciclo:02d}.jsonl", aprov)
        append_jsonl(self.p.teacher_file, [e for e in rev])          # pendentes → /revisar
        append_jsonl(self.p.teacher_file, [e for e in aprov if e["origin"] == "teacher"])
        append_jsonl(self.p.rejected_file, rej)
        cont.update(aprovados=len(aprov), revisao=len(rev), rejeitados=len(rej))
        self.log(f"Dados do ciclo {ciclo}: {cont}")
        return cont

    # ---------------------------------------------------------------- ciclo
    def executar(self, max_ciclos: int = 4, passos: int = 2000, base_n: int = 20,
                 usar_professor: bool = True, parar_sem_ganho: bool = True, **treino_kw) -> Dict:
        inicio = time.time()
        reg = self.garantir_base()
        historico = []
        for ciclo in range(1, max_ciclos + 1):
            ativa = self._versao(reg, reg["ativa"])
            cats = ativa["eval"]["por_categoria"]
            diagnostico = {c: round(1 - (v["acerto"] or 0), 3) for c, v in cats.items()}
            fracas = sorted(diagnostico, key=diagnostico.get, reverse=True)
            self.log(f"=== Ciclo {ciclo} | ativa {reg['ativa']} | categorias mais fracas: "
                     f"{', '.join(f'{c} ({1 - diagnostico[c]:.0%})' for c in fracas[:4])}")
            dados = self.gerar_dados(ciclo, diagnostico, base_n=base_n, usar_professor=usar_professor)
            dec = self.treinar_versao(passos=passos, ciclo=ciclo,
                                      origem="ciclo automático (Teacher + programáticos)", **treino_kw)
            reg = self.registro()
            historico.append({"ciclo": ciclo, "versao": dec["versao"], "promovida": dec["promovida"],
                              "motivo": dec["motivo"], "dados": dados,
                              "treino_segundos": dec["treino_segundos"],
                              "dataset": dec["dataset"], "run": dec["run"]})
            if parar_sem_ganho and not dec["promovida"]:
                self.log("Sem ganho mensurável neste ciclo: parando.")
                break
        total = round(time.time() - inicio)
        rel = {"ciclos": historico, "segundos": total, "ativa": reg["ativa"]}
        self.escrever_relatorio(rel)
        return rel

    def treinar_versao(self, passos: int = 2000, ciclo: Optional[int] = None,
                       origem: str = "treino manual", **treino_kw) -> Dict:
        """Treina uma versão CANDIDATA numa pasta própria (nunca nos pesos ativos),
        avalia no conjunto fixo e só promove se melhorar. Seguro contra interrupção:
        se parar no meio, a versão ativa continua intacta."""
        reg = self.garantir_base()
        ativa = self._versao(reg, reg["ativa"])
        manifest = self.p.build_dataset()
        nome = self.proxima_versao(reg)
        candidato = self.arquivo_versao(nome)
        shutil.copy2(self.vdir / ativa["arquivo"], candidato)
        self.log(f"Treinando {nome} a partir de {ativa['versao']} com dataset v{manifest['version']} "
                 f"({manifest['train_examples']} exemplos, {passos} passos)...")
        t0 = time.time()
        run = self.p.train_seed(passos=passos, pesos=candidato, log=self.log, **treino_kw)
        seg = round(time.time() - t0)
        r = self.avaliar_arquivo(candidato, nome)
        dec = self._decidir(reg, nome, candidato, r, ciclo, ativa["versao"], manifest["version"], origem)
        self._registrar_uso(nome, manifest)
        dec.update(dataset=manifest, run=run, treino_segundos=seg)
        return dec

    def _registrar_uso(self, versao: str, manifest: Dict) -> None:
        """Qual versão do Seed usou quais exemplos (ids do dataset de treino)."""
        ids = [r["id"] for r in read_jsonl(self.root / "datasets" / manifest["train_file"])]
        append_jsonl(self.root / "training" / "uso_dados.jsonl",
                     [{"versao_seed": versao, "dataset_version": manifest["version"],
                       "ids": ids, "data": agora()}])

    # ------------------------------------------------------------- relatório
    def escrever_relatorio(self, rel: Optional[Dict] = None) -> Path:
        reg = self.registro()
        linhas = ["# HEILO Seed — versões e ciclos de treino", "",
                  f"Avaliação fixa: `{EVAL_VERSION}` ({len(self.itens_eval)} itens que nunca entram no treino).",
                  f"Versão ativa: **{reg['ativa']}**", "",
                  "| Versão | Status | Pai | Dataset | Acerto eval_v1 | Sonda independente | Perplexidade | Instruções | Generalização | Comparação |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for v in reg["versoes"]:
            e = v["eval"]
            sonda = e.get("sonda")
            linhas.append(f"| {v['versao']} | {v['status']} | {v.get('pai') or '-'} | "
                          f"{v.get('dataset_version') or '-'} | {e['acerto']:.1%} | "
                          f"{'-' if not sonda else format(sonda['acerto'], '.1%')} | {e['perplexidade']} | "
                          f"{e['seguir_instrucoes']} | {e['generalizacao']} | "
                          f"{v.get('comparacao', '')} |")
        if rel:
            linhas += ["", f"Último processo: {len(rel['ciclos'])} ciclo(s), {rel['segundos'] // 60} min.", ""]
            for c in rel["ciclos"]:
                d = c["dados"]
                linhas.append(f"- Ciclo {c['ciclo']} → {c['versao']}: {'promovida' if c['promovida'] else 'rejeitada'} "
                              f"({c['motivo']}); dados gerados {d['gerados']}, aprovados {d['aprovados']}, "
                              f"revisão {d['revisao']}, rejeitados {d['rejeitados']}; treino {c['treino_segundos'] // 60} min.")
        arq = self.root / "training" / "relatorio_ciclos.md"
        arq.write_text("\n".join(linhas) + "\n", encoding="utf-8")
        return arq
