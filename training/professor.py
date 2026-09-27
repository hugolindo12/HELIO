"""
HEILO Teacher como gerador de dados (modo `professor`).

O Teacher (Qwen, temporário e removível) só RESPONDE perguntas de uma lista.
Cada resposta passa pela validação automática de qualidade; as reprovadas vão
para data/rejected, as outras ficam PENDENTES de revisão humana (Revisar
aprendizados na interface). Nada é aprovado automaticamente e nenhum peso muda.

Pode ser interrompido e rodado de novo: pula as perguntas já respondidas.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from heilo.training import geradores
from heilo.training.eval_set import load_eval, load_sonda
from heilo.training.qualidade import avaliar_exemplo, contaminado
from heilo.training.records import agora, append_jsonl, make_example, read_jsonl

PERGUNTAS_PADRAO = "escritos/perguntas_professor.txt"
FACTUAIS = {"python", "cnc", "ferramentas", "conhecimento", "raciocinio"}


def ler_perguntas(arquivo: Path) -> List[Tuple[str, str]]:
    out = []
    for linha in Path(arquivo).read_text(encoding="utf-8").splitlines():
        linha = linha.strip()
        if not linha or linha.startswith("#"):
            continue
        cat, sep, q = linha.partition("|")
        out.append((cat.strip(), q.strip()) if sep else ("conversa", linha))
    return out


def gerar_com_professor(pipe, models, arquivo: Path = None, limite: int = 0,
                        log: Callable[[str], None] = print) -> Dict:
    if models is None or not models.teacher_available():
        from heilo.training.pipeline import TeacherRequired
        detalhe = models.get("teacher").availability()[1] if models is not None else ""
        raise TeacherRequired(detalhe)
    arquivo = Path(arquivo or pipe.root / PERGUNTAS_PADRAO)
    perguntas = ler_perguntas(arquivo)
    eval_qs = [i["pergunta"] for i in load_eval()]
    sonda_qs = [i["pergunta"] for i in load_sonda()]

    # perguntas que já têm resposta em algum lugar (aprovada, pendente ou rejeitada)
    feitas = set()
    for r in pipe.approved() + read_jsonl(pipe.teacher_file) + read_jsonl(pipe.rejected_file) + pipe._escritos():
        for m in r.get("messages", []):
            if m.get("role") == "user":
                feitas.add(m["content"].strip().lower())
    card = models.get("teacher").card().to_dict()
    cont = {"perguntas": len(perguntas), "ja_feitas": 0, "contaminadas": 0,
            "pendentes_revisao": 0, "rejeitadas_auto": 0, "sem_resposta": 0}
    fila = []
    for cat, q in perguntas:
        if q.lower() in feitas:
            cont["ja_feitas"] += 1
        elif contaminado(q, eval_qs) or contaminado(q, sonda_qs, limiar=0.6):
            cont["contaminadas"] += 1
        else:
            fila.append((cat, q))
    if limite:
        fila = fila[:limite]
    log(f"[Teacher] {card.get('base_model')} vai responder {len(fila)} pergunta(s). "
        f"Já feitas: {cont['ja_feitas']} | fora (parecidas com eval/sonda): {cont['contaminadas']}")
    inicio, seguidas = time.time(), 0
    for i, (cat, q) in enumerate(fila, 1):
        sistema = geradores.SISTEMA_RACIOCINIO if cat == "raciocinio" else geradores.SISTEMA_RESPOSTA
        r = models.generate_with("teacher", [{"role": "user", "content": q}], system=sistema,
                                 temperatura=0.5, max_novos=120)
        resp = (r.text or "").strip()
        if not resp:   # Teacher falhou (não carregou, memória...): não marca a pergunta como feita
            cont["sem_resposta"] += 1
            seguidas = seguidas + 1
            if seguidas >= 3:
                log("[Teacher] 3 respostas vazias seguidas: o Teacher não está funcionando. Parei. "
                    "Verifique com: python -m heilo.main cli → /cerebro")
                break
            continue
        seguidas = 0
        v = avaliar_exemplo(q, resp, cat, eval_qs, set(), factual=cat in FACTUAIS, verificar_fatos=True)
        ex = make_example([{"role": "user", "content": q}, {"role": "assistant", "content": resp}],
                          origin="teacher",
                          meta={"status": "nao_verificado", "gerador": "professor", "categoria": cat,
                                "professor": card.get("base_model"), "license": card.get("license")})
        ex.update({"categoria": cat, "qualidade": v["qualidade"], "validacao": v, "data": agora()})
        if v["status"] == "rejeitado":
            append_jsonl(pipe.rejected_file, [dict(ex, rejected_at=agora(), rejected_by="validação automática",
                                                   reasons=v["problemas"])])
            cont["rejeitadas_auto"] += 1
        else:
            append_jsonl(pipe.teacher_file, [ex])     # PENDENTE: revisão humana
            cont["pendentes_revisao"] += 1
        seg = (time.time() - inicio) / i
        log(f"  {i}/{len(fila)} [{cat}] {q[:45]} → {'ok' if v['status'] != 'rejeitado' else 'rejeitada'}"
            f" | faltam ~{seg * (len(fila) - i) / 60:.0f} min")
    cont["minutos"] = round((time.time() - inicio) / 60, 1)
    return cont
