"""
Rotas da interface de conversa da HEILO (estilo chat).

- Cada conversa da interface é independente: o navegador manda o histórico
  junto com a mensagem e o Core responde só com aquele contexto.
- Comandos de barra (/ensinar, /cerebro, /status...) usam o mesmo código do CLI.
- Ensinar e revisar pela interface alimentam o dataset aprovado. Nada aqui
  altera pesos: o Seed só aprende no próximo treino.
"""
from __future__ import annotations

import asyncio
import json
import threading
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from heilo.core.model_adapter import Message

router = APIRouter()
_lock = threading.Lock()


def _orch():
    from heilo.ui.api import orchestrator
    return orchestrator


class Turno(BaseModel):
    role: str
    content: str


class ConversaRequest(BaseModel):
    message: str
    history: List[Turno] = []


class ComandoRequest(BaseModel):
    texto: str


class EnsinarRequest(BaseModel):
    pergunta: str
    resposta: str


class RevisaoRequest(BaseModel):
    id: str
    aprovar: bool
    resposta: Optional[str] = None


def _responder(message: str, history: List[Turno]) -> Dict[str, Any]:
    orch = _orch()
    with _lock:  # um Core compartilhado: uma conversa por vez
        orch.conversation = [Message(role=t.role, content=t.content)
                             for t in history[-20:] if t.role in ("user", "assistant") and t.content]
        if message.strip().startswith("/"):
            from heilo.core.commands import handle
            if message.strip().split()[0].lower() == "/revisar":
                return {"type": "comando", "content": "Use o botão **Revisar** na barra lateral.",
                        "model": "comando"}
            texto = handle(orch, message, ask=lambda _p: "")
            return {"type": "comando", "content": texto, "model": "comando"}
        return orch.chat(message, somente_conversa=True)


@router.post("/api/conversa")
async def conversa(req: ConversaRequest) -> Dict[str, Any]:
    if not req.message.strip():
        raise HTTPException(400, "mensagem vazia")
    loop = asyncio.get_event_loop()
    try:
        r = await loop.run_in_executor(None, _responder, req.message, req.history)
    except Exception as e:  # a interface nunca fica sem resposta
        return {"type": "erro", "content": f"Erro no HEILO Core: {e}", "model": ""}
    return {k: v for k, v in r.items() if isinstance(v, (str, int, float, bool, type(None), list, dict))}


def _versoes() -> Dict[str, Any]:
    from heilo.config import BASE_DIR
    arq = BASE_DIR / "data" / "training" / "versions.json"
    try:
        d = json.loads(arq.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"ativa": None, "versoes": []}
    vs = []
    for v in d.get("versoes", []):
        ev = v.get("eval") or {}
        so = v.get("sonda") or ev.get("sonda") or {}
        vs.append({"versao": v.get("versao"), "status": v.get("status"),
                   "acerto": ev.get("acerto"), "sonda": so.get("acerto") if isinstance(so, dict) else so,
                   "criada": v.get("criada")})
    return {"ativa": d.get("ativa"), "versoes": vs}


@router.get("/api/heilo")
async def heilo_info() -> Dict[str, Any]:
    orch = _orch()
    st = orch.models.status()
    return {"modo": st["mode"], "seed": st["seed"], "teacher": st["teacher"],
            "teacher_no_chat": st["teacher_in_chat"], **_versoes()}


@router.post("/api/cerebro")
async def cerebro(req: ComandoRequest) -> Dict[str, Any]:
    with _lock:
        return {"texto": _orch().models.set_mode(req.texto)}


@router.post("/api/ensinar")
async def ensinar(req: EnsinarRequest) -> Dict[str, Any]:
    if not req.pergunta.strip() or not req.resposta.strip():
        raise HTTPException(400, "pergunta e resposta são obrigatórias")
    with _lock:
        return _orch().ensinar(req.pergunta.strip(), req.resposta.strip())


@router.get("/api/revisar")
async def revisar_lista(limite: int = 30) -> Dict[str, Any]:
    with _lock:
        tp = _orch().training
        val = tp.validate_pending()
        pend = tp.pending()[:limite]
    return {"rejeitados_auto": val.get("rejeitados_auto", 0),
            "pendentes": [{"id": p["id"], "origem": p.get("origin"), "messages": p["messages"][-4:]}
                          for p in pend]}


@router.post("/api/revisar")
async def revisar(req: RevisaoRequest) -> Dict[str, Any]:
    with _lock:
        tp = _orch().training
        if req.aprovar:
            return tp.review(req.id, True, edited_answer=(req.resposta or None))
        return tp.review(req.id, False, reason="rejeitado na revisão (interface)")
