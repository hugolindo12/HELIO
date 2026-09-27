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
from pathlib import Path
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
    versao: Optional[str] = None   # testar outra versão do Seed sem promover


class ComandoRequest(BaseModel):
    texto: str


class EnsinarRequest(BaseModel):
    pergunta: str
    resposta: str


class RevisaoRequest(BaseModel):
    id: str
    aprovar: bool
    resposta: Optional[str] = None


_testes: Dict[str, Any] = {}   # versão -> SeedAdapter (no máximo 2 em memória)


def _arquivo_versao(versao: str):
    """Checkpoint de uma versão registrada em data/training/versions.json (se estiver no PC)."""
    from heilo.config import BASE_DIR
    from heilo.training.ciclos import VERSIONS_DIR
    try:
        reg = json.loads((BASE_DIR / "data" / "training" / "versions.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    v = next((x for x in reg.get("versoes", []) if x.get("versao") == versao), None)
    if not v or v.get("status") == "removida" or not v.get("arquivo"):
        return None
    base = Path(VERSIONS_DIR).resolve()
    arq = (base / v["arquivo"]).resolve()
    from heilo.models.seed.gpt import tem_modelo
    if base not in arq.parents or not tem_modelo(arq):
        return None
    return arq


def _responder_versao(versao: str, message: str, history: List[Turno]) -> Dict[str, Any]:
    """Conversa com uma versão específica do Seed (teste). Não grava na memória:
    serve para comparar versões antes de promover."""
    from heilo.models.seed.adapter import SeedAdapter
    arq = _arquivo_versao(versao)
    if arq is None:
        return {"type": "erro", "content": f"Versão {versao} não encontrada no PC "
                                           "(rode atualizar_heilo.bat).", "model": ""}
    if versao not in _testes:
        while len(_testes) >= 2:
            _testes.pop(next(iter(_testes)))
        _testes[versao] = SeedAdapter(weights=arq)
    msgs = [{"role": t.role, "content": t.content} for t in history[-20:]
            if t.role in ("user", "assistant") and t.content]
    msgs.append({"role": "user", "content": message})
    texto = _testes[versao].generate(msgs)
    from heilo.models.linhas import nome_modelo
    chat = _testes[versao]._chat
    n = chat.modelo.n_params() if chat is not None and chat.modelo is not None else None
    return {"type": "message", "content": texto or "(sem resposta)", "model": f"{nome_modelo(n, versao)} (teste)",
            "source": "versao_teste"}


def _responder(message: str, history: List[Turno], versao: Optional[str] = None) -> Dict[str, Any]:
    if versao and not message.strip().startswith("/"):
        with _lock:
            return _responder_versao(versao, message, history)
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
            if texto.lstrip().startswith(("{", "[")):
                texto = "```json\n" + texto + "\n```"
            return {"type": "comando", "content": texto, "model": "comando"}
        return orch.chat(message, somente_conversa=True)


@router.post("/api/conversa")
async def conversa(req: ConversaRequest) -> Dict[str, Any]:
    if not req.message.strip():
        raise HTTPException(400, "mensagem vazia")
    loop = asyncio.get_event_loop()
    try:
        r = await loop.run_in_executor(None, _responder, req.message, req.history, req.versao)
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
        vs.append({"versao": v.get("versao"), "status": v.get("status"), "linha": v.get("linha") or "Faísca",
                   "no_pc": bool(v.get("versao") and _arquivo_versao(v["versao"])),
                   "acerto": ev.get("acerto"), "sonda": so.get("acerto") if isinstance(so, dict) else so,
                   "criada": v.get("criada")})
    return {"ativa": d.get("ativa"), "versoes": vs}


def _n_params_ativo(orch):
    try:
        chat = orch.models.get("seed")._load()
        return chat.modelo.n_params() if chat is not None and chat.modelo is not None else None
    except Exception:
        return None


@router.get("/api/heilo")
async def heilo_info() -> Dict[str, Any]:
    orch = _orch()
    st = orch.models.status()
    from heilo.models.linhas import linha
    return {"modo": st["mode"], "seed": st["seed"], "teacher": st["teacher"],
            "linha": linha(_n_params_ativo(orch)),
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


class RevisaoLoteRequest(BaseModel):
    ids: List[str]
    aprovar: bool = True


@router.post("/api/revisar/lote")
async def revisar_lote(req: RevisaoLoteRequest) -> Dict[str, Any]:
    """Aprova (ou rejeita) vários candidatos de uma vez: os que o usuário já leu na tela."""
    ok = falhas = 0
    with _lock:
        tp = _orch().training
        for i in req.ids[:500]:
            r = (tp.review(i, True) if req.aprovar
                 else tp.review(i, False, reason="rejeitado na revisão (interface, em lote)"))
            ok += bool(r.get("ok")); falhas += not r.get("ok")
    return {"ok": True, "feitos": ok, "falhas": falhas}


@router.get("/api/revisar")
async def revisar_lista(limite: int = 300) -> Dict[str, Any]:
    with _lock:
        tp = _orch().training
        val = tp.validate_pending()
        pend = tp.pending()[:limite]
    return {"rejeitados_auto": val.get("rejeitados_auto", 0),
            "pendentes": [{"id": p["id"], "origem": p.get("origin"),
                           "categoria": (p.get("meta") or {}).get("categoria"),
                           "messages": p["messages"][-4:]}
                          for p in pend]}


@router.post("/api/revisar")
async def revisar(req: RevisaoRequest) -> Dict[str, Any]:
    with _lock:
        tp = _orch().training
        if req.aprovar:
            return tp.review(req.id, True, edited_answer=(req.resposta or None))
        return tp.review(req.id, False, reason="rejeitado na revisão (interface)")


@router.get("/api/lembretes/vencidos")
async def lembretes_vencidos() -> Dict[str, Any]:
    """Lembretes cuja hora chegou (a interface consulta a cada 30 s e avisa)."""
    orch = _orch()
    if getattr(orch, "diaadia", None) is None:
        return {"vencidos": []}
    with _lock:
        return {"vencidos": [{"texto": l["texto"], "quando": l["quando"]} for l in orch.diaadia.vencidos()]}
