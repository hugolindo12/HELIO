"""
HEILO FastAPI Interface
"""
import json
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
from pathlib import Path
from heilo.core.orchestrator import Orchestrator
from heilo.config import BASE_DIR

app = FastAPI(title="HEILO", version="0.1.0")
orchestrator = Orchestrator()

static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)


class ChatRequest(BaseModel):
    message: str
    workspace: Optional[str] = None


class PermissionRequest(BaseModel):
    grant_session: bool = False


class TimezoneRequest(BaseModel):
    timezone: str


@app.get("/", response_class=HTMLResponse)
async def index():
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return HTMLResponse("<h1>HEILO</h1><p>UI not found. Place index.html in ui/static/</p>")


@app.post("/api/chat")
async def chat(req: ChatRequest) -> Dict[str, Any]:
    if req.workspace:
        try:
            orchestrator.set_workspace(req.workspace)
        except Exception as e:
            raise HTTPException(400, str(e))
    result = orchestrator.chat(req.message)
    return result


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    if req.workspace:
        try:
            orchestrator.set_workspace(req.workspace)
        except Exception as e:
            raise HTTPException(400, str(e))

    async def event_generator():
        # Start event
        yield f"event: start\ndata: {json.dumps({'message': 'Processando solicitação...'}, ensure_ascii=False)}\n\n"
        await asyncio.sleep(0.01)

        # Intent event
        intent = orchestrator._classify_intent(req.message)
        yield f"event: intent\ndata: {json.dumps({'intent': intent}, ensure_ascii=False)}\n\n"
        await asyncio.sleep(0.01)

        # Chat execution in background thread
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, orchestrator.chat, req.message)

        yield f"event: result\ndata: {json.dumps(result, ensure_ascii=False)}\n\n"
        yield "event: done\ndata: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/checkpoints")
async def list_checkpoints(limit: int = 20):
    return {"checkpoints": orchestrator.checkpoints.list_checkpoints(limit=limit)}


@app.post("/api/checkpoints/rollback")
async def rollback_checkpoint():
    restored = orchestrator.checkpoints.rollback()
    return {"restored": restored, "count": len(restored)}


@app.get("/api/status")
async def status():
    return orchestrator.get_status()


@app.post("/api/approve")
async def approve(req: PermissionRequest):
    return orchestrator.approve_permission(grant_session=req.grant_session)


@app.get("/api/tools")
async def tools():
    return {"tools": [t.__dict__ if hasattr(t, "__dict__") else str(t) for t in orchestrator.tools.list_tools()]}

@app.get("/api/mcp")
async def mcp_servers():
    return {"servers": [s.__dict__ for s in orchestrator.mcp.list_servers()]}


@app.get("/api/timezone")
async def get_timezone():
    status = orchestrator.get_status()
    return status.get("timezone", {})


@app.post("/api/timezone")
async def set_timezone(req: TimezoneRequest):
    return orchestrator.set_user_timezone(req.timezone)


class KnowledgeSyncRequest(BaseModel):
    message: str = ""
    push: bool = False
    remote: str = ""


@app.get("/api/learning")
async def list_learning(limit: int = 20):
    return {"entries": orchestrator.learning.list_entries(limit=limit)}


@app.post("/api/knowledge/sync")
async def knowledge_sync(req: KnowledgeSyncRequest):
    if req.remote:
        orchestrator.knowledge_git.init_repo(remote_url=req.remote)
    result = orchestrator.knowledge_git.sync(
        message=req.message or None,
        push=req.push,
    )
    return result


@app.get("/api/knowledge/status")
async def knowledge_git_status():
    return orchestrator.knowledge_git.status()


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    mode: str = "hybrid"  # hybrid | semantic | keyword


@app.post("/api/search")
async def semantic_search(req: SearchRequest):
    hits = orchestrator.retriever.search(
        req.query, top_k=req.top_k, mode=req.mode
    )
    return {
        "query": req.query,
        "mode": req.mode,
        "count": len(hits),
        "backend": orchestrator.retriever.vector.info().get("backend"),
        "results": hits,
    }


@app.post("/api/search/reindex")
async def reindex_knowledge():
    orchestrator.retriever.index_knowledge(force=True)
    return {"ok": True, "info": orchestrator.retriever.info()}
