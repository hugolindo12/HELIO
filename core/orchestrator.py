"""
HEILO Core — Orchestrator.

Recebe mensagens, gerencia contexto e coordena os componentes:

    Core ─┬─ ModelManager   → HEILO Seed | HEILO Teacher [opcional]   (core/model_manager.py)
          ├─ MemoryManager  → memória local                            (memory/manager.py)
          ├─ KnowledgeManager → conhecimento recuperável (RAG)         (knowledge/manager.py)
          ├─ TrainingPipeline → dados → dataset → treino do Seed       (training/pipeline.py)
          └─ agentes / ferramentas

O Core não importa nenhuma biblioteca de modelo e não conhece o modelo externo
por trás do Teacher.
"""
from pathlib import Path
from typing import Any, Dict, Optional, List
from heilo.core.model_adapter import ModelAdapter, Message
from heilo.core.task import TaskLog, new_task_id
from heilo.core.pipeline import (
    PipelineResult, PipelineStepResult,
    build_code_task_from_research, build_test_task_from_code,
    format_numbered_citations, format_pipeline_report,
)
from heilo.agents.code.agent import CodeAgent
from heilo.agents.test.agent import TestAgent
from heilo.agents.research.agent import ResearchAgent
from heilo.tools.registry import ToolRegistry
from heilo.security.permissions import PermissionManager
from heilo.security.workspace import WorkspaceManager
from heilo.security.checkpoint import CheckpointManager
from heilo.config import config
from heilo.core import timeutil
from heilo.core.knowledge_brain import KnowledgeBrain
from heilo.memory.learning import LearningStore
from heilo.knowledge.git_sync import KnowledgeGitSync
from heilo.memory.store import MemoryStore
from heilo.knowledge.store import KnowledgeStore
from heilo.rag.retriever import Retriever
from heilo.mcp.base import MCPRegistry
from heilo.mcp.servers import SystemInfoMCPServer, HL3DMCPServer, BlenderMCPServer


class Orchestrator:
    """
    HEILO Principal.
    Understands intent, routes to sub-agents, reviews results, talks to user.
    """

    def __init__(
        self,
        model: Optional[ModelAdapter] = None,
        workspace: Optional[WorkspaceManager] = None,
        models=None,
        memory_manager=None,
        knowledge_manager=None,
        training=None,
    ):
        self.model = model or ModelAdapter()
        self.workspace = workspace or WorkspaceManager()
        self.permissions = PermissionManager(
            require_confirmation=config.security.require_confirmation_for_critical
        )
        self.checkpoints = CheckpointManager(self.workspace.root)
        self.mcp = MCPRegistry()
        self._register_mcp_servers()
        self.memory = MemoryStore()
        self.learning = LearningStore()
        self.knowledge_git = KnowledgeGitSync()
        self.knowledge = KnowledgeStore()
        self.retriever = Retriever(self.knowledge)
        self.brain = KnowledgeBrain(
            retriever=self.retriever,
            memory=self.memory,
            knowledge=self.knowledge,
            learning=self.learning,
        )
        from heilo.core.dialogue import ConversationalEngine
        self.dialogue = ConversationalEngine()
        # ---- Componentes da plataforma (injetáveis para testes/cloud) ----
        from heilo.core.model_manager import ModelManager
        from heilo.memory.manager import MemoryManager
        from heilo.knowledge.manager import KnowledgeManager
        from heilo.training.pipeline import TrainingPipeline
        self.models = models or ModelManager.from_config(config)
        # HEILO Dia a Dia: hora, contas, conversões, lembretes, notas e listas (exatos, locais)
        from heilo.tools.diaadia import DiaADia
        import os as _os
        from heilo.config import MEMORY_DIR as _MEM
        self.diaadia = DiaADia(Path(_os.environ.get("HEILO_PESSOAL_DIR") or (_MEM / "pessoal")))
        self.memory_mgr = memory_manager or MemoryManager(
            enabled=getattr(config, "memory_log_conversations", True))
        self.knowledge_mgr = knowledge_manager or KnowledgeManager(
            store=self.knowledge, retriever=self.retriever)
        self.training = training or TrainingPipeline(
            models=self.models, memory=self.memory_mgr, knowledge=self.knowledge_mgr)
        self.tools = self._build_tools()

        agent_kwargs = dict(
            model=self.model,
            tools=self.tools,
            permissions=self.permissions,
            memory_store=self.memory,
            knowledge_store=self.knowledge,
            workspace=self.workspace,
        )
        self.code_agent = CodeAgent(**agent_kwargs)
        self.code_agent.checkpoints = self.checkpoints
        self.test_agent = TestAgent(**agent_kwargs)
        self.research_agent = ResearchAgent(**agent_kwargs)

        self.agents = {
            "code": self.code_agent,
            "HEILO_CODE": self.code_agent,
            "test": self.test_agent,
            "HEILO_TEST": self.test_agent,
            "research": self.research_agent,
            "HEILO_RESEARCH": self.research_agent,
        }

        self.conversation: List[Message] = []
        self.last_task_log: Optional[TaskLog] = None
        self._pending_task_id: Optional[str] = None
        self._pending_agent: Optional[str] = None

    def _register_mcp_servers(self):
        self.mcp.register(SystemInfoMCPServer())
        self.mcp.register(HL3DMCPServer())
        self.mcp.register(BlenderMCPServer())

    def _build_tools(self) -> ToolRegistry:
        from heilo.tools.filesystem import (
            ListFilesTool, ReadFileTool, SearchCodeTool,
            WriteFileTool, EditFileTool, FindSymbolTool,
        )
        from heilo.tools.execution import RunCommandTool, RunTestsTool, BuildProjectTool
        from heilo.tools.git import GetGitDiffTool

        registry = ToolRegistry(self.permissions)
        ws = self.workspace
        cp = self.checkpoints
        for tool in [
            ListFilesTool(ws),
            ReadFileTool(ws),
            SearchCodeTool(ws),
            WriteFileTool(ws, checkpoint_manager=cp),
            EditFileTool(ws, checkpoint_manager=cp),
            FindSymbolTool(ws),
            RunCommandTool(ws),
            RunTestsTool(ws),
            BuildProjectTool(ws),
            GetGitDiffTool(ws),
        ]:
            registry.register(tool)
        # MCP tools
        for mcp_tool in self.mcp.all_tools():
            registry.register(mcp_tool)
        # Research tools
        from heilo.tools.research import (
            SearchKnowledgeTool, SearchWebTool, FetchUrlTool, SaveResearchNoteTool,
        )
        registry.register(SearchKnowledgeTool(retriever=getattr(self, "retriever", None)))
        registry.register(SearchWebTool())
        registry.register(FetchUrlTool())
        registry.register(SaveResearchNoteTool(
            knowledge_store=getattr(self, "knowledge", None),
            retriever=getattr(self, "retriever", None),
        ))
        return registry

    def set_workspace(self, path: str):
        self.workspace.set_workspace(path)
        self.checkpoints = CheckpointManager(self.workspace.root)
        self.tools = self._build_tools()
        for agent in (self.code_agent, self.test_agent, self.research_agent):
            agent.workspace = self.workspace
            agent.tools = self.tools
        self.code_agent.checkpoints = self.checkpoints

    def chat(self, user_message: str, somente_conversa: bool = False) -> Dict[str, Any]:
        """somente_conversa=True (interface de chat): não dispara agentes de código,
        teste ou pesquisa; responde com conhecimento local + HEILO Seed."""
        result = self._chat(user_message, somente_conversa=somente_conversa)
        # HEILO Memory: registra a troca (memória local — NÃO treina nada)
        if isinstance(result, dict) and result.get("content"):
            try:
                self.memory_mgr.record(
                    user_message,
                    result["content"],
                    source=str(result.get("source", "")),
                    kind=str(result.get("type", "message")),
                    model=str(result.get("model", "")),
                )
            except OSError as e:
                print(f"[HEILO Memory] não consegui salvar a conversa: {e}")
        return result

    def _historico_modelo(self) -> List[Dict[str, str]]:
        return [
            {"role": m.role, "content": m.content}
            for m in self.conversation
            if m.role in ("user", "assistant")
        ]

    def _responder_com_modelo(self):
        """Pede ao ModelManager uma resposta. Retorna (texto, modelo) ou ("", "")."""
        try:
            r = self.models.generate(self._historico_modelo())
            return r.text, r.model
        except Exception as e:  # um modelo nunca derruba o Core
            print(f"[HEILO Core] modelo: {e}")
            return "", ""

    # ---- ensino / aprendizado / avaliação ---------------------------
    def ensinar(self, pergunta: str, resposta: str) -> Dict[str, Any]:
        """/ensinar: registra exemplo supervisionado + conhecimento. Não altera pesos."""
        return self.training.register_taught(pergunta, resposta)

    def aprender(self) -> Dict[str, Any]:
        """/aprender: Memory → seleção → validação → dataset. Não treina, não usa o
        Teacher e não envia nada para o GitHub."""
        return self.training.learn()

    def comparar_modelos(self, texto: str) -> Dict[str, str]:
        """Ferramenta de avaliação Seed × Teacher. Nenhum dos dois é tratado como verdade."""
        hist = self._historico_modelo() + [{"role": "user", "content": texto}]
        return self.models.compare(hist)

    # compatibilidade com o nome antigo
    comparar_cerebros = comparar_modelos

    def _chat(self, user_message: str, somente_conversa: bool = False) -> Dict[str, Any]:
        self.conversation.append(Message(role="user", content=user_message))

        if getattr(self, "diaadia", None) is not None:
            ferramenta = self.diaadia.responder(user_message)
            if ferramenta is not None:
                self.conversation.append(Message(role="assistant", content=ferramenta["content"]))
                return ferramenta

        msg_l = user_message.lower().strip()
        if msg_l in ("continue", "continuar", "continue a tarefa anterior.", "continue a tarefa anterior"):
            if self._pending_task_id and self.last_task_log:
                return self._resume_code_task()

        intent = self._classify_intent(user_message)
        if somente_conversa and intent != "conversation":
            intent = "general"
        if intent == "conversation":
            return self._handle_conversational(user_message)
        if intent == "pipeline_research_code_test":
            return self._handle_pipeline_research_code_test(user_message)
        if intent == "pipeline_research_code":
            return self._handle_pipeline_research_code(user_message)
        if intent == "research":
            return self._handle_research_task(user_message)
        if intent == "test":
            return self._handle_test_task(user_message)
        if intent == "code":
            return self._handle_code_task(user_message)
        # Local-first: answer from HEILO knowledge repository when possible
        # Na interface de chat quem responde é o HEILO Seed; o repositório de
        # conhecimento só entra se o modelo não tiver resposta.
        if config.local_first and not somente_conversa:
            local = self._try_local_answer(user_message)
            if local is not None:
                self.conversation.append(Message(role="assistant", content=local["content"]))
                return local

        reply, qual = self._responder_com_modelo()
        source = f"model_{qual}"
        if not reply and somente_conversa:
            local = self._try_local_answer(user_message)
            if local is not None:
                self.conversation.append(Message(role="assistant", content=local["content"]))
                return local
        if not reply:
            reply = self._general_reply(user_message)
            source = "llm_fallback"
        self.conversation.append(Message(role="assistant", content=reply))
        return {
            "type": "message",
            "content": reply,
            "agent": "HEILO",
            "source": source,
            "model": qual,
        }

    def _classify_intent(self, text: str) -> str:
        text_l = text.lower()
        from heilo.core.dialogue import DialogueClassifier
        social = DialogueClassifier.classify(text)
        has_task_action = any(
            k in text_l for k in (
                "corrija", "corrigir", "pesquise", "pesquisar", "rode os testes",
                "rodar teste", "execute", "crie um arquivo", "editar arquivo",
                "altere", "refatore", "analise este projeto", "find_symbol",
                "escreva testes", "criar testes", "consulte a documentacao",
            )
        )
        if social and not has_task_action:
            return "conversation"

        # Full pipeline: research + apply + test
        full_markers = [
            "pesquise corrija e teste", "pesquise, corrija e teste",
            "research fix and test", "research, fix and test",
            "pipeline completo", "research code test",
            "pesquise aplique e teste", "pesquise, aplique e teste",
            "busque corrija e teste", "encontre corrija e teste",
        ]
        has_research = any(k in text_l for k in ("pesquise", "pesquisar", "pesquisa", "research", "busque"))
        has_apply = any(k in text_l for k in ("corrija", "corrigir", "aplique", "aplicar", "fix", "no projeto"))
        has_test = any(k in text_l for k in (
            "teste", "testes", "test", "pytest", "rode os testes", "valide", "validar",
        ))
        if any(k in text_l for k in full_markers) or (has_research and has_apply and has_test):
            return "pipeline_research_code_test"
        # Hybrid: research then apply/fix in project
        pipeline_markers = [
            "pesquise e corrija", "pesquisar e corrigir", "pesquise e aplique",
            "research and fix", "research and apply", "encontre a solução e aplique",
            "encontre a solucao e aplique", "busque a solução e corrija",
            "busque a solucao e corrija", "pesquise a solução e corrija",
            "pesquise a solucao e corrija", "aplique no projeto",
        ]
        if any(k in text_l for k in pipeline_markers):
            return "pipeline_research_code"
        if has_research and has_apply:
            return "pipeline_research_code"
        research_keywords = [
            "pesquisa", "pesquisar", "pesquise", "pesquisando", "research", "buscar informação", "buscar informacao",
            "o que é", "o que e", "como funciona", "documentação", "documentacao",
            "explique", "explorar", "aprender sobre", "procure sobre", "search for",
            "what is", "how does", "look up", "heilo research",
        ]
        if any(k in text_l for k in research_keywords):
            return "research"
        test_keywords = [
            "rode os testes", "rodar teste", "gerar teste", "gerar testes",
            "escreva testes", "criar testes", "cobertura", "pytest",
            "run tests", "write tests", "test coverage", "heilo test",
        ]
        if any(k in text_l for k in test_keywords):
            return "test"
        code_keywords = [
            "código", "code", "bug", "erro", "error", "corrigir", "fix",
            "analise", "analisar", "projeto", "project", "arquivo", "file",
            "função", "class", "teste", "test", "build", "compilar",
            "refator", "implement", "criar arquivo", "editar",
        ]
        if any(k in text_l for k in code_keywords):
            return "code"
        return "general"

    def _handle_conversational(self, user_message: str) -> Dict[str, Any]:
        from heilo.core.dialogue import DialogueClassifier
        sub_intent = DialogueClassifier.classify(user_message) or "greeting"
        reply, qual = self._responder_com_modelo()
        source = f"model_{qual}" if reply else "conversational_brain"
        if not reply:
            tz = self.get_user_timezone()
            reply = self.dialogue.respond(
                intent=sub_intent,
                message=user_message,
                user_tz=tz,
                history=[{"role": m.role, "content": m.content} for m in self.conversation],
            )
        self.conversation.append(Message(role="assistant", content=reply))
        return {
            "type": "message",
            "content": reply,
            "agent": "HEILO",
            "intent": "conversation",
            "sub_intent": sub_intent,
            "source": source,
            "model": qual,
        }

    def _handle_code_task(self, user_message: str) -> Dict[str, Any]:
        task_id = new_task_id()
        task_log = TaskLog(
            task_id=task_id,
            user_message=user_message,
            agent="HEILO_CODE",
            model=self.model.cfg.model_name,
        )
        self.last_task_log = task_log

        # RAG enrichment
        extra_parts = []
        hits = self.retriever.retrieve(user_message, top_k=3)
        if hits:
            extra_parts.append("Relevant knowledge:\n" + "\n---\n".join(hits))
        sols = self.memory.list_solutions(limit=3)
        if sols:
            extra_parts.append(
                "Past verified solutions:\n"
                + "\n".join(f"- {s.get('problem', '')[:120]}" for s in sols)
            )
        extra = "\n\n".join(extra_parts) if extra_parts else None

        result = self.code_agent.run(
            task=user_message,
            task_log=task_log,
            context={"extra_system": extra} if extra else None,
        )

        if result["status"] == "waiting_permission":
            self._pending_task_id = task_id
            return {
                "type": "permission_request",
                "content": (
                    f"O agente HEILO CODE precisa de autorização para: "
                    f"{result['permission'].get('action')} → {result['permission'].get('target')}\n"
                    f"Motivo: {result['permission'].get('reason')}"
                ),
                "permission": result["permission"],
                "task_id": task_id,
                "agent": "HEILO_CODE",
                "steps": result.get("steps", []),
                "resumable": True,
            }

        review = self._review_result(user_message, result)
        task_log.save()
        self.conversation.append(Message(role="assistant", content=review))
        self._pending_task_id = None

        if result["status"] == "success" and self.memory:
            self.memory.maybe_store_solution(user_message, result)
        if result.get("status") in ("success", "aborted", "failed"):
            self._learn_from_result(user_message, result)

        return {
            "type": "task_result",
            "content": review,
            "status": result["status"],
            "agent": "HEILO_CODE",
            "task_id": task_id,
            "steps": result.get("steps", []),
            "task_log": task_log.to_dict(),
        }

    



    def _handle_pipeline_research_code_test(self, user_message: str) -> Dict[str, Any]:
        """
        Pipeline: HEILO RESEARCH → HEILO CODE → HEILO TEST
        """
        pipeline = PipelineResult(name="research_code_test", status="running")
        all_tool_steps: list = []

        # --- 1. RESEARCH ---
        research_out = self._handle_research_task(user_message)
        if research_out.get("type") == "permission_request":
            return {
                "type": "permission_request",
                "content": research_out.get("content"),
                "permission": research_out.get("permission"),
                "task_id": research_out.get("task_id"),
                "agent": "PIPELINE",
                "pipeline": "research_code_test",
                "steps": research_out.get("steps", []),
            }

        research_text = research_out.get("content") or ""
        all_tool_steps.extend(research_out.get("steps") or [])
        pipeline.steps.append(PipelineStepResult(
            agent="HEILO_RESEARCH",
            status=research_out.get("status", "unknown"),
            result=research_text,
            steps=research_out.get("steps", []),
            sources=research_out.get("sources", []),
            task_id=research_out.get("task_id"),
        ))

        if research_out.get("status") not in ("success", None) and research_out.get("type") != "task_result":
            pipeline.status = "partial"
            pipeline.final_content = format_pipeline_report(
                "research_code_test", research_text, "(CODE nao iniciado)", "(TEST nao iniciado)", "partial"
            )
            return {
                "type": "task_result",
                "content": pipeline.final_content,
                "status": "partial",
                "agent": "PIPELINE",
                "pipeline": pipeline.to_dict(),
                "steps": all_tool_steps,
            }

        # --- 2. CODE ---
        code_task = build_code_task_from_research(user_message, research_text)
        code_out = self._handle_code_task(code_task)
        if code_out.get("type") == "permission_request":
            return {
                "type": "permission_request",
                "content": (
                    "Pesquisa OK. HEILO CODE precisa de autorizacao para aplicar a correcao.\n"
                    + (code_out.get("content") or "")
                ),
                "permission": code_out.get("permission"),
                "task_id": code_out.get("task_id"),
                "agent": "PIPELINE",
                "pipeline": "research_code_test",
                "steps": code_out.get("steps", []),
                "research_preview": research_text[:800],
            }

        code_text = code_out.get("content") or ""
        all_tool_steps.extend(code_out.get("steps") or [])
        pipeline.steps.append(PipelineStepResult(
            agent="HEILO_CODE",
            status=code_out.get("status", "unknown"),
            result=code_text,
            steps=code_out.get("steps", []),
            task_id=code_out.get("task_id"),
        ))

        code_ok = code_out.get("status") == "success"
        if not code_ok and code_out.get("type") != "task_result":
            pipeline.status = "partial"
            pipeline.final_content = format_pipeline_report(
                "research_code_test", research_text, code_text, "(TEST nao iniciado)", "partial"
            )
            return {
                "type": "task_result",
                "content": pipeline.final_content,
                "status": "partial",
                "agent": "PIPELINE",
                "pipeline": pipeline.to_dict(),
                "steps": all_tool_steps,
            }

        # --- 3. TEST ---
        test_task = build_test_task_from_code(user_message, code_text)
        test_out = self._handle_test_task(test_task)
        if test_out.get("type") == "permission_request":
            return {
                "type": "permission_request",
                "content": (
                    "CODE aplicado. HEILO TEST precisa de autorizacao.\n"
                    + (test_out.get("content") or "")
                ),
                "permission": test_out.get("permission"),
                "task_id": test_out.get("task_id"),
                "agent": "PIPELINE",
                "pipeline": "research_code_test",
                "steps": test_out.get("steps", []),
            }

        test_text = test_out.get("content") or ""
        all_tool_steps.extend(test_out.get("steps") or [])
        pipeline.steps.append(PipelineStepResult(
            agent="HEILO_TEST",
            status=test_out.get("status", "unknown"),
            result=test_text,
            steps=test_out.get("steps", []),
            task_id=test_out.get("task_id"),
        ))

        test_ok = test_out.get("status") == "success"
        pipeline.status = "success" if (code_ok and test_ok) else "partial"
        pipeline.final_content = format_pipeline_report(
            "research_code_test", research_text, code_text, test_text, pipeline.status
        )

        return {
            "type": "task_result",
            "content": pipeline.final_content,
            "status": pipeline.status,
            "agent": "PIPELINE",
            "pipeline": pipeline.to_dict(),
            "steps": all_tool_steps,
            "task_id": test_out.get("task_id") or code_out.get("task_id"),
        }

    def _handle_pipeline_research_code(self, user_message: str) -> Dict[str, Any]:
        """
        Pipeline: HEILO RESEARCH → HEILO CODE
        1) Research gathers solution knowledge
        2) Code agent applies it in the workspace
        """
        pipeline = PipelineResult(name="research_then_code", status="running")

        # --- Step 1: RESEARCH ---
        research_out = self._handle_research_task(user_message)
        if research_out.get("type") == "permission_request":
            pipeline.status = "waiting_permission"
            pipeline.permission = research_out.get("permission")
            return {
                "type": "permission_request",
                "content": research_out.get("content"),
                "permission": research_out.get("permission"),
                "task_id": research_out.get("task_id"),
                "agent": "PIPELINE",
                "pipeline": "research_then_code",
                "steps": research_out.get("steps", []),
            }

        research_text = research_out.get("content") or research_out.get("result") or ""
        pipeline.steps.append(PipelineStepResult(
            agent="HEILO_RESEARCH",
            status=research_out.get("status", "unknown"),
            result=research_text,
            steps=research_out.get("steps", []),
            sources=research_out.get("sources", []),
            task_id=research_out.get("task_id"),
        ))

        if research_out.get("status") not in ("success", None) and research_out.get("type") != "task_result":
            pipeline.status = "partial"
            pipeline.final_content = (
                "**HEILO Pipeline** (RESEARCH → CODE)\n\n"
                "A pesquisa não concluiu com sucesso; etapa CODE não iniciada.\n\n"
                + research_text
            )
            return {
                "type": "task_result",
                "content": pipeline.final_content,
                "status": "partial",
                "agent": "PIPELINE",
                "pipeline": pipeline.to_dict(),
            }

        # --- Step 2: CODE with research context ---
        code_task = build_code_task_from_research(user_message, research_text)
        code_out = self._handle_code_task(code_task)

        if code_out.get("type") == "permission_request":
            pipeline.status = "waiting_permission"
            return {
                "type": "permission_request",
                "content": (
                    "Pesquisa concluída. HEILO CODE precisa de autorização para aplicar a correção.\n"
                    + (code_out.get("content") or "")
                ),
                "permission": code_out.get("permission"),
                "task_id": code_out.get("task_id"),
                "agent": "PIPELINE",
                "pipeline": "research_then_code",
                "steps": code_out.get("steps", []),
                "research_preview": research_text[:800],
            }

        pipeline.steps.append(PipelineStepResult(
            agent="HEILO_CODE",
            status=code_out.get("status", "unknown"),
            result=code_out.get("content"),
            steps=code_out.get("steps", []),
            task_id=code_out.get("task_id"),
        ))

        ok = code_out.get("status") == "success"
        pipeline.status = "success" if ok else "partial"
        pipeline.final_content = (
            "**HEILO Pipeline** RESEARCH → CODE\n\n"
            "### 1. Pesquisa\n"
            f"{research_text[:1200]}\n\n"
            "### 2. Aplicação no código\n"
            f"{code_out.get('content', '')}\n\n"
            f"Status final: {pipeline.status}"
        )

        return {
            "type": "task_result",
            "content": pipeline.final_content,
            "status": pipeline.status,
            "agent": "PIPELINE",
            "pipeline": pipeline.to_dict(),
            "steps": (research_out.get("steps") or []) + (code_out.get("steps") or []),
            "task_id": code_out.get("task_id"),
        }

    def _handle_research_task(self, user_message: str) -> Dict[str, Any]:
        task_id = new_task_id()
        task_log = TaskLog(
            task_id=task_id,
            user_message=user_message,
            agent="HEILO_RESEARCH",
            model=self.model.cfg.model_name,
        )
        self.last_task_log = task_log
        self._pending_agent = "research"

        hits = self.retriever.retrieve(user_message, top_k=3)
        extra = "Seed context from RAG:\n" + "\n".join(hits) if hits else None

        result = self.research_agent.run(
            task=user_message,
            task_log=task_log,
            context={"extra_system": extra} if extra else None,
        )

        if result["status"] == "waiting_permission":
            self._pending_task_id = task_id
            return {
                "type": "permission_request",
                "content": f"HEILO RESEARCH precisa de autorização: {result.get('permission')}",
                "permission": result.get("permission"),
                "task_id": task_id,
                "agent": "HEILO_RESEARCH",
                "steps": result.get("steps", []),
            }

        raw = result.get("result") or ""
        steps = result.get("steps", [])
        sources = result.get("sources", [])
        summary = "\n".join(
            f"  - {s.get('tool')}: {'OK' if s.get('success') else 'FAIL'}" for s in steps[-12:]
        )
        cites = format_numbered_citations(sources)
        review = f"**HEILO** (via HEILO RESEARCH)\n\n{raw}{cites}\n\n---\nPassos:\n{summary}"
        task_log.save()
        self.conversation.append(Message(role="assistant", content=review))

        # Store successful research summaries into memory
        if result["status"] == "success" and self.memory:
            try:
                self.memory.maybe_store_solution(user_message, result)
            except Exception:
                pass
            self._learn_from_result(user_message, result)

        return {
            "type": "task_result",
            "content": review,
            "status": result["status"],
            "agent": "HEILO_RESEARCH",
            "task_id": task_id,
            "steps": steps,
            "sources": sources,
            "task_log": task_log.to_dict(),
        }

    def _handle_test_task(self, user_message: str) -> Dict[str, Any]:
        task_id = new_task_id()
        task_log = TaskLog(
            task_id=task_id,
            user_message=user_message,
            agent="HEILO_TEST",
            model=self.model.cfg.model_name,
        )
        self.last_task_log = task_log
        self._pending_agent = "test"

        hits = self.retriever.retrieve(user_message, top_k=3)
        extra = "Relevant knowledge:\n" + "\n".join(hits) if hits else None

        result = self.test_agent.run(
            task=user_message,
            task_log=task_log,
            context={"extra_system": extra} if extra else None,
        )

        if result["status"] == "waiting_permission":
            self._pending_task_id = task_id
            return {
                "type": "permission_request",
                "content": f"HEILO TEST precisa de autorização: {result.get('permission')}",
                "permission": result.get("permission"),
                "task_id": task_id,
                "agent": "HEILO_TEST",
                "steps": result.get("steps", []),
            }

        raw = result.get("result") or ""
        steps = result.get("steps", [])
        summary = "\n".join(
            f"  - {s.get('tool')}: {'OK' if s.get('success') else 'FAIL'}" for s in steps[-10:]
        )
        review = (
            f"**HEILO** (via HEILO TEST)\n\n{raw}\n\n---\nPassos:\n{summary}"
        )
        task_log.save()
        self.conversation.append(Message(role="assistant", content=review))
        return {
            "type": "task_result",
            "content": review,
            "status": result["status"],
            "agent": "HEILO_TEST",
            "task_id": task_id,
            "steps": steps,
            "task_log": task_log.to_dict(),
        }

    def _resume_code_task(self) -> Dict[str, Any]:
        if not self._pending_task_id or not self.last_task_log:
            return {"type": "message", "content": "Nenhuma tarefa pendente para retomar.", "agent": "HEILO"}

        result = self.code_agent.resume(self._pending_task_id, self.last_task_log)

        if result["status"] == "waiting_permission":
            return {
                "type": "permission_request",
                "content": (
                    f"Nova autorização necessária: "
                    f"{result['permission'].get('action')} → {result['permission'].get('target')}"
                ),
                "permission": result["permission"],
                "task_id": self._pending_task_id,
                "agent": "HEILO_CODE",
                "steps": result.get("steps", []),
                "resumable": True,
            }

        review = self._review_result(self.last_task_log.user_message, result)
        self.last_task_log.save()
        self.conversation.append(Message(role="assistant", content=review))
        self._pending_task_id = None

        if result["status"] == "success" and self.memory:
            self.memory.maybe_store_solution(self.last_task_log.user_message, result)
        if result.get("status") in ("success", "aborted", "failed"):
            self._learn_from_result(self.last_task_log.user_message, result)

        return {
            "type": "task_result",
            "content": review,
            "status": result["status"],
            "agent": "HEILO_CODE",
            "task_id": self.last_task_log.task_id,
            "steps": result.get("steps", []),
            "task_log": self.last_task_log.to_dict(),
        }

    def _review_result(self, user_message: str, agent_result: Dict) -> str:
        status = agent_result.get("status")
        raw = agent_result.get("result") or ""
        steps = agent_result.get("steps", [])

        summary_steps = "\n".join(
            f"  - [{s.get('phase')}] {s.get('tool')}: {'OK' if s.get('success') else 'FAIL'}"
            for s in steps[-8:]
        )

        if status == "success":
            return (
                f"**HEILO** (via HEILO CODE)\n\n"
                f"Tarefa concluída.\n\n"
                f"{raw}\n\n"
                f"---\nPassos principais:\n{summary_steps}"
            )
        elif status == "aborted":
            return (
                f"**HEILO**\n\n"
                f"A tarefa foi interrompida: {raw}\n\n"
                f"Passos executados:\n{summary_steps}\n\n"
                f"Posso tentar novamente com mais contexto ou outra abordagem."
            )
        else:
            return (
                f"**HEILO**\n\n"
                f"Status: {status}\n\n{raw}\n\n"
                f"Passos:\n{summary_steps}"
            )



    def _learn_from_result(self, problem: str, result: dict, agent: str = ""):
        """Persist errors/corrections into learning + knowledge; optional git commit."""
        if not getattr(config, "auto_learn", True):
            return None
        try:
            entry = self.learning.record_from_task(
                problem=problem,
                result=result,
                project=str(self.workspace.root.name),
                agent=agent or str(result.get("agent") or ""),
            )
        except Exception as e:
            print(f"[Learning] {e}")
            return None
        if entry and getattr(config, "auto_commit_knowledge", False):
            try:
                self.knowledge_git.sync(
                    message=f"heilo learn: {entry.kind} {entry.entry_id}",
                    push=getattr(config, "auto_push_knowledge", False),
                )
            except Exception as e:
                print(f"[KnowledgeGit] {e}")
        return entry

    def _try_local_answer(self, user_message: str):
        """
        Use HEILO's own knowledge brain (repository). Returns response dict or None.
        Does not call external LLM when confidence is high enough.
        """
        try:
            result = self.brain.understand(user_message, top_k=5)
        except Exception as e:
            print(f"[KnowledgeBrain] {e}")
            return None
        threshold = getattr(config, "knowledge_confidence_threshold", 0.35)
        if not result.found or result.confidence < threshold:
            if not getattr(config, "llm_fallback", True):
                return {
                    "type": "message",
                    "content": (
                        "**HEILO** (modo local-only)\n\n"
                        "Nao encontrei conhecimento suficiente no repositorio local.\n"
                        "Adicione documentos em knowledge/ ou ajuste "
                        "`knowledge_confidence_threshold`."
                    ),
                    "agent": "HEILO",
                    "source": "knowledge_miss",
                    "confidence": result.confidence,
                }
            return None
        cites = ""
        if result.sources:
            cites = "\n\n### Fontes do repositorio HEILO\n" + "\n".join(
                f"- {s}" for s in result.sources[:8]
            )
        return {
            "type": "message",
            "content": result.answer + cites,
            "agent": "HEILO",
            "source": result.mode,
            "confidence": result.confidence,
            "sources": result.sources,
        }

    def _general_reply(self, user_message: str) -> str:
        msg_l = (user_message or "").strip().lower()
        greetings = ("oi", "ola", "olá", "hello", "hi", "opa", "e ai", "e aí", "hey", "bom dia", "boa tarde", "boa noite", "tudo bem", "tudo bom", "como vai")
        if any(msg_l == g or msg_l.startswith(g + " ") or msg_l.startswith(g + "!") or msg_l.startswith(g + ",") for g in greetings):
            return (
                "Olá! Eu sou a **HEILO**, sua plataforma de agentes de IA.\n\n"
                "Estou à disposição para ajudar você com:\n"
                "- **HEILO CODE:** Investigar, editar e corrigir arquivos com testes automatizados.\n"
                "- **HEILO RESEARCH:** Pesquisar na base local RAG e na web.\n"
                "- **HEILO TEST:** Executar suítes de testes e analisar cobertura.\n"
                "- **MCP Tools:** Informações do sistema, CAD e Blender 3D.\n\n"
                "Como posso te ajudar hoje?"
            )

        # Enrich with RAG even for general chat
        hits = self.retriever.retrieve(user_message, top_k=2)
        extra = ""
        if hits:
            extra = "\n\nContext from knowledge base:\n" + "\n".join(hits)

        system = (
            "You are HEILO, the main orchestrator of an AI agent platform. "
            "You can route programming tasks to HEILO CODE. "
            "Be concise, helpful and clear. Reply in the same language as the user."
            + extra
        )
        return self.model.simple(system, user_message)

    def approve_permission(self, grant_session: bool = False) -> Dict[str, Any]:
        pending = self.permissions.get_pending()
        if not pending:
            # Still try to resume if we have a pending task
            if self._pending_task_id:
                return self._resume_code_task()
            return {"type": "message", "content": "Nenhuma permissão pendente."}

        req = pending[0]
        self.permissions.approve(req, grant_session=grant_session)

        # Auto-resume after approval
        if self._pending_task_id:
            return self._resume_code_task()

        return {
            "type": "message",
            "content": f"Permissão concedida para {req.action} ({req.target}).",
        }

    def get_user_timezone(self) -> str:
        try:
            return self.memory.get_timezone(default=config.user_timezone)
        except Exception:
            return config.user_timezone

    def set_user_timezone(self, tz_name: str) -> Dict[str, Any]:
        # Validate
        try:
            timeutil.get_zone(tz_name)
        except Exception:
            return {"ok": False, "error": f"Invalid timezone: {tz_name}"}
        self.memory.set_preference("timezone", tz_name)
        config.user_timezone = tz_name
        return {
            "ok": True,
            "timezone": tz_name,
            "now_local": timeutil.format_user(timeutil.utc_now(), tz_name),
            "now_utc": timeutil.utc_now_iso(),
        }

    def get_status(self) -> Dict[str, Any]:
        tz = self.get_user_timezone()
        return {
            "workspace": self.workspace.get_info(),
            "model": {
                "provider": self.model.cfg.provider,
                "model_name": self.model.cfg.model_name,
            },
            "agents": list(self.agents.keys()),
            "tools": self.tools.list_names(),
            "pending_permissions": len(self.permissions.get_pending()),
            "last_task": self.last_task_log.task_id if self.last_task_log else None,
            "pending_task": self._pending_task_id,
            "checkpoints": self.checkpoints.list_checkpoints(limit=5),
            "rag": self.retriever.info(),
            "mcp_servers": [s.__dict__ if hasattr(s, "__dict__") else str(s) for s in self.mcp.list_servers()],
            "models": self.models.status(),
            "memory": self.memory_mgr.stats(),
            "local_first": getattr(config, "local_first", True),
            "llm_fallback": getattr(config, "llm_fallback", True),
            "knowledge_threshold": getattr(config, "knowledge_confidence_threshold", 0.35),
            "timezone": {
                "user": tz,
                "now_local": timeutil.format_user(timeutil.utc_now(), tz),
                "now_utc": timeutil.utc_now_iso(),
                "common": timeutil.list_common_timezones(),
            },
        }
