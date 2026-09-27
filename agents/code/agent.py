"""
HEILO Code Agent
Specialized programming sub-agent with controlled autonomous cycle,
resumable state, native tool-calling and automatic checkpoint/rollback.
"""
from typing import Any, Dict, Optional, List
import re
import time
from heilo.agents.base import BaseAgent
from heilo.agents.code.cycle import CyclePhase, next_phase
from heilo.core.model_adapter import Message
from heilo.core.task import TaskLog
from heilo.core.agent_state import AgentState, AgentStateStore
from heilo.config import config
from heilo.security.workspace import WorkspaceManager
from heilo.security.checkpoint import CheckpointManager


class CodeAgent(BaseAgent):
    name = "HEILO_CODE"
    description = (
        "Specialized agent for software development: analyze projects, find bugs, "
        "edit code, run tests and fix problems safely inside a workspace."
    )
    instructions = """
You are HEILO CODE. Follow this strict cycle:
1. UNDERSTAND the user request
2. PLAN the investigation steps
3. INSPECT (list files, search, read)
4. IDENTIFY the problem
5. PROPOSE a concrete fix
6. APPLY the change (only after permission if required)
7. TEST
8. ANALYZE the result
9. If failed → FIX and retry (up to max attempts)
10. FINISH with a clear report

Rules:
- Never leave the authorized workspace.
- Prefer minimal, targeted changes.
- Always explain what you are doing.
- When the model supports native tool calling, use tools directly.
- Otherwise use the format: TOOL: name key="value"
- When you have enough information to answer, write a final report starting with FINAL:
"""

    def __init__(self, *args, workspace: WorkspaceManager = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.workspace = workspace
        self.max_attempts = config.limits.max_attempts
        self.max_commands = config.limits.max_commands
        self.timeout = config.limits.timeout_seconds
        self.state_store = AgentStateStore()
        self.checkpoints: Optional[CheckpointManager] = None
        if workspace:
            self.checkpoints = CheckpointManager(workspace.root)

    def run(
        self,
        task: str,
        task_log: TaskLog,
        context: Optional[Dict] = None,
        resume_task_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute or resume the agent cycle.
        If resume_task_id is given, load saved state and continue.
        """
        start = time.time()

        # --- Resume or start fresh ---
        if resume_task_id:
            state = self.state_store.load(resume_task_id)
            if state is None:
                return {"status": "failed", "result": f"No saved state for task {resume_task_id}"}
            phase = CyclePhase[state.phase] if state.phase in CyclePhase.__members__ else CyclePhase.UNDERSTAND
            attempt = state.attempt
            history = state.to_messages()
            steps = list(state.steps)
            # Clear waiting status
            state.status = "running"
            state.pending_permission = None
        else:
            phase = CyclePhase.UNDERSTAND
            attempt = 0
            history: List[Message] = [
                Message(role="system", content=self._system_prompt()),
            ]
            if context and context.get("extra_system"):
                history.append(Message(role="system", content=context["extra_system"]))
            history.append(Message(
                role="user",
                content=(
                    f"Task: {task}\n"
                    f"Workspace: {self.workspace.root if self.workspace else 'default'}\n"
                    "Start the cycle. Begin by understanding and inspecting the project."
                ),
            ))
            steps = []
            state = AgentState(
                task_id=task_log.task_id,
                agent_name=self.name,
                task=task,
                phase=phase.name,
                attempt=0,
                status="running",
            )

        if self.checkpoints:
            self.checkpoints.start(task_log.task_id)

        final_result = None
        status = "running"
        tools_schema = self._tools_schema()

        while True:
            if time.time() - start > self.timeout:
                status = "aborted"
                final_result = "Timeout exceeded"
                break
            if attempt >= self.max_attempts and phase in (
                CyclePhase.FIX, CyclePhase.APPLY, CyclePhase.TEST
            ):
                status = "aborted"
                final_result = f"Max attempts ({self.max_attempts}) reached"
                if self.checkpoints:
                    restored = self.checkpoints.rollback()
                    final_result += f" | Rollback: {restored}"
                break

            # Ask model (with native tools when provider supports it)
            response = self.model.chat(history, tools=tools_schema)
            content = response.content or ""
            history.append(Message(role="assistant", content=content))

            # Collect tool calls: prefer native, fallback to text protocol
            tool_calls = []
            if response.tool_calls:
                for tc in response.tool_calls:
                    tool_calls.append({
                        "name": tc.get("name", ""),
                        "args": tc.get("arguments", {}),
                        "id": tc.get("id"),
                    })
            else:
                tool_calls = self._parse_tool_calls(content)

            if tool_calls:
                for tc in tool_calls:
                    name = tc["name"]
                    args = tc["args"]
                    result = self.tools.execute(name, **args)

                    # Permission gate → save state and pause
                    if result.error in ("PERMISSION_REQUIRED", "CRITICAL_COMMAND"):
                        perm = result.metadata.get("permission_request", {})
                        state.phase = phase.name
                        state.attempt = attempt
                        state.status = "waiting_permission"
                        state.pending_permission = perm
                        state.set_history(history)
                        state.steps = steps
                        self.state_store.save(state)
                        task_log.status = "waiting_permission"
                        task_log.metadata["permission"] = perm
                        return {
                            "status": "waiting_permission",
                            "result": None,
                            "permission": perm,
                            "steps": steps,
                            "phase": phase.name,
                            "task_id": task_log.task_id,
                            "resumable": True,
                        }

                    summary = str(result.output)[:600] if result.success else (result.error or "error")
                    task_log.add_tool(name, args, result.success, summary)

                    if name in ("read_file", "list_files", "search_code") and result.success:
                        if name == "read_file":
                            task_log.files_analyzed.append(args.get("path", ""))
                        elif name == "list_files" and isinstance(result.output, list):
                            task_log.files_analyzed.extend(result.output[:20])

                    steps.append({
                        "phase": phase.name,
                        "tool": name,
                        "args": {k: str(v)[:100] for k, v in args.items()},
                        "success": result.success,
                        "output_preview": summary[:300],
                    })

                    # Feed result back
                    history.append(Message(
                        role="user",
                        content=f"TOOL_RESULT ({name}): success={result.success}\n{summary}",
                    ))

                    # Anti-loop check: detect repeated identical calls
                    curr_args_summary = {k: str(v)[:100] for k, v in args.items()}
                    if len(steps) >= 3:
                        last_calls = steps[-3:]
                        if all(s.get("tool") == name and s.get("args") == curr_args_summary for s in last_calls):
                            history.append(Message(
                                role="user",
                                content=(
                                    f"AVISO DO SISTEMA: A ferramenta '{name}' já foi executada 3 vezes consecutivas com os mesmos argumentos. "
                                    "Não repita a mesma chamada. Avance para a próxima fase do ciclo (leia os arquivos inspecionados ou proponha a correção)."
                                ),
                            ))

                    # Phase transitions
                    if name in ("list_files", "search_code", "read_file", "find_symbol"):
                        if phase in (CyclePhase.UNDERSTAND, CyclePhase.PLAN):
                            phase = CyclePhase.INSPECT
                        elif phase == CyclePhase.INSPECT:
                            phase = CyclePhase.IDENTIFY
                    elif name in ("write_file", "edit_file"):
                        phase = CyclePhase.APPLY
                        attempt += 1
                        task_log.changes.append({"tool": name, "args": args})
                    elif name in ("run_tests", "run_command", "build_project"):
                        phase = CyclePhase.TEST
                        task_log.tests.append({
                            "command": args.get("command", name),
                            "success": result.success,
                            "output": summary,
                        })
                        if not result.success and self.checkpoints:
                            # Automatic rollback on test failure
                            restored = self.checkpoints.rollback()
                            history.append(Message(
                                role="user",
                                content=(
                                    f"Tests failed. Automatic rollback performed: {restored}. "
                                    "Propose a different fix."
                                ),
                            ))
                        phase = next_phase(
                            phase,
                            test_passed=result.success,
                            attempts=attempt,
                            max_attempts=self.max_attempts,
                        )

            # FINAL report
            if "FINAL:" in content.upper() or content.strip().startswith("FINAL"):
                final_result = content
                status = "success"
                phase = CyclePhase.FINISH
                if self.checkpoints:
                    self.checkpoints.commit()
                break

            if not tool_calls and phase == CyclePhase.UNDERSTAND:
                history.append(Message(
                    role="user",
                    content="Continue. Use the list_files tool now to inspect the project.",
                ))
                phase = CyclePhase.PLAN

            if len(steps) > 40:
                status = "aborted"
                final_result = "Too many steps – circuit breaker"
                if self.checkpoints:
                    self.checkpoints.rollback()
                break

            if phase in (CyclePhase.FINISH, CyclePhase.ABORT):
                break

            # Persist intermediate state (crash recovery)
            state.phase = phase.name
            state.attempt = attempt
            state.set_history(history)
            state.steps = steps
            self.state_store.save(state)

        # Cleanup
        if status in ("success", "aborted", "failed"):
            self.state_store.delete(task_log.task_id)

        task_log.finish(status, final_result)
        return {
            "status": status,
            "result": final_result,
            "steps": steps,
            "phase": phase.name,
            "attempts": attempt,
            "task_log": task_log.to_dict(),
        }

    def resume(self, task_id: str, task_log: TaskLog) -> Dict[str, Any]:
        """Resume a paused task after permission was granted."""
        return self.run(task="", task_log=task_log, resume_task_id=task_id)

    def _tools_schema(self) -> List[Dict]:
        """OpenAI-compatible tools schema for native tool calling."""
        schemas = []
        for t in self.tools.list_tools():
            schemas.append({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters if t.parameters else {"type": "object", "properties": {}},
                },
            })
        return schemas

    def _parse_tool_calls(self, text: str) -> List[Dict]:
        """Fallback text protocol for stub / models without native tools."""
        calls = []
        for line in text.splitlines():
            line = line.strip()
            m = re.match(r"TOOL:\s*(\w+)\s*(.*)", line, re.IGNORECASE)
            if not m:
                continue
            name = m.group(1).lower()
            rest = m.group(2).strip()
            args = {}
            pattern = re.compile(
                r"""(\w+)=("([^"]*)"|'([^']*)'|(\S+(?:\s+(?!\w+=)\S+)*))"""
            )
            for match in pattern.finditer(rest):
                k = match.group(1)
                if match.group(3) is not None:
                    v = match.group(3)
                elif match.group(4) is not None:
                    v = match.group(4)
                else:
                    v = (match.group(5) or "").strip()
                args[k] = v
            if not args and rest:
                if name == "list_files":
                    args["subpath"] = rest or "."
                elif name in ("read_file", "write_file"):
                    args["path"] = rest
            calls.append({"name": name, "args": args})
        return calls
