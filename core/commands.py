"""
Comandos de barra do HEILO Core (usados pelo CLI; reutilizáveis pela UI).

/ensinar pergunta => resposta   registra exemplo + conhecimento (não treina)
/aprender                       Memory → validação → dataset (não treina, não usa Teacher, não faz push)
/revisar                        aprova/rejeita candidatos (Memory e Teacher)
/gerar pergunta                 pede ao Teacher um exemplo NÃO verificado (requer Teacher)
/comparar pergunta              Seed × Teacher lado a lado (ferramenta de avaliação)
/cerebro auto|seed|teacher|off  escolhe o motor (aliases: mini=seed, lora=teacher)
/metricas                       números reais: Teacher, dados aprovados, treinos do Seed
/memoria [busca]                memória local
/status                         modelos, memória e conhecimento
"""
from __future__ import annotations

import json
from typing import Callable, List

from heilo.models.base import TEACHER_REQUIRED_MSG
from heilo.training.pipeline import TeacherRequired

AJUDA = (
    "Comandos: /ensinar pergunta => resposta | /aprender | /revisar | /gerar pergunta | "
    "/comparar pergunta | /cerebro auto|seed|teacher|off | /metricas | /memoria [busca] | /status"
)


def _fmt(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def handle(orch, linha: str, ask: Callable[[str], str] = input) -> str:
    """Executa um comando e devolve o texto para mostrar ao usuário."""
    cmd, _, resto = linha.strip().partition(" ")
    cmd, resto = cmd.lower(), resto.strip()

    if cmd == "/ensinar":
        if "=>" not in resto:
            return "Uso: /ensinar pergunta => resposta"
        pergunta, resposta = (p.strip() for p in resto.split("=>", 1))
        r = orch.ensinar(pergunta, resposta)
        if not r["ok"]:
            return "Não registrei: " + "; ".join(r["errors"])
        extra = " (já existia)" if r.get("ja_existia") else ""
        return (f"Registrado{extra}: exemplo aprovado para o dataset e salvo no conhecimento. "
                "Nenhum peso foi alterado. O Seed aprende no próximo treino "
                "(python -m heilo.main treinar).")

    if cmd == "/aprender":
        r = orch.aprender()
        d = r["dataset"]
        return (f"Coletados da memória: {r['coletados_da_memoria']} | rejeitados na validação: "
                f"{r['rejeitados_auto']} | pendentes de revisão: {r['pendentes_revisao']}\n"
                f"Dataset v{d['version']}: {d['train_examples']} treino + {d['val_examples']} validação "
                f"{d['por_origem']}\n"
                "Nada foi treinado. Revise com /revisar e treine com: python -m heilo.main treinar")

    if cmd == "/revisar":
        return revisar(orch, ask)

    if cmd == "/gerar":
        if not resto:
            return "Uso: /gerar pergunta   (o Teacher responde; o resultado vai para revisão)"
        try:
            r = orch.training.generate_with_teacher([resto])
        except TeacherRequired as e:
            return str(e)
        return (f"Teacher gerou {r['gerados']} exemplo(s) NÃO verificado(s) em data/teacher/. "
                "Use /revisar para aprovar ou rejeitar.")

    if cmd == "/comparar":
        if not resto:
            return "Uso: /comparar pergunta"
        saidas = orch.comparar_modelos(resto)
        linhas = [f"[{k}] {v}" for k, v in saidas.items()]
        linhas.append("(Avaliação: nenhum dos dois é tratado como verdade.)")
        texto = "\n".join(linhas)
        voto = ""
        if TEACHER_REQUIRED_MSG not in saidas.get("teacher", ""):
            voto = ask(texto + "\nQual foi melhor? [s]eed / [t]eacher / [e]mpate / enter = pular: ")
            voto = {"s": "seed", "t": "teacher", "e": "empate"}.get(voto.strip().lower()[:1], "")
        orch.training.log_comparison(resto, saidas, verdict=voto or None)
        return texto if not voto else f"Registrado: {voto}."

    if cmd == "/cerebro":
        if resto:
            return orch.models.set_mode(resto)
        st = orch.models.status()
        return (f"Modo: {st['mode']} | Seed: {st['seed']['reason']} | "
                f"Teacher: {st['teacher']['reason']}")

    if cmd == "/metricas":
        return _fmt(orch.training.metrics())

    if cmd == "/memoria":
        if resto:
            achados = orch.memory_mgr.search(resto)
            return "\n".join(f"- {r['ts']} | {r['user'][:60]} → {r['assistant'][:60]}"
                             for r in achados) or "Nada encontrado na memória."
        return _fmt(orch.memory_mgr.stats())

    if cmd == "/status":
        return _fmt({"models": orch.models.status(), "memory": orch.memory_mgr.stats(),
                     "knowledge": orch.knowledge_mgr.stats()})

    return AJUDA


def revisar(orch, ask: Callable[[str], str] = input, limite: int = 50) -> str:
    """Revisão humana dos candidatos pendentes (Memory e Teacher)."""
    val = orch.training.validate_pending()
    pend: List[dict] = orch.training.pending()[:limite]
    if not pend:
        return f"Nada pendente. (rejeitados automaticamente agora: {val['rejeitados_auto']})"
    aprov = rej = 0
    for ex in pend:
        conversa = "\n".join(f"  {'Você' if m['role'] == 'user' else 'HEILO'}: {m['content']}"
                             for m in ex["messages"][-4:])
        resp = ask(f"\n[{ex['origin']}] {ex['id']}\n{conversa}\n"
                   "[a]provar / [r]ejeitar / [e]ditar resposta / [p]ular / [s]air: ").strip().lower()
        if resp.startswith("s"):
            break
        if resp.startswith("a"):
            aprov += bool(orch.training.review(ex["id"], True)["ok"])
        elif resp.startswith("r"):
            rej += bool(orch.training.review(ex["id"], False, reason="rejeitado na revisão")["ok"])
        elif resp.startswith("e"):
            nova = ask("Resposta correta: ")
            aprov += bool(orch.training.review(ex["id"], True, edited_answer=nova)["ok"])
    return f"Revisão: {aprov} aprovado(s), {rej} rejeitado(s). Pendentes: {len(orch.training.pending())}"
