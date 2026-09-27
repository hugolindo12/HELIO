"""
HEILO TEST Agent
Specialized in discovering code, generating tests, running them and reporting coverage gaps.
"""
from typing import Any, Dict, Optional, List
import re
import time
from heilo.agents.base import BaseAgent
from heilo.core.model_adapter import Message
from heilo.core.task import TaskLog
from heilo.config import config
from heilo.security.workspace import WorkspaceManager


class TestAgent(BaseAgent):
    name = "HEILO_TEST"
    description = (
        "Specialized agent for software testing: discover code under test, "
        "generate pytest tests, run them and report results."
    )
    instructions = """
You are HEILO TEST. Your job:
1. Inspect the project structure
2. Identify modules/functions that need tests
3. Read existing tests if any
4. Generate or improve pytest tests
5. Run the tests
6. Report clearly what passed, failed and what is missing

Rules:
- Prefer pytest style
- Keep tests focused and minimal
- Never leave the workspace
- Use tools: TOOL: name key="value"
- End with FINAL: report
"""

    def __init__(self, *args, workspace: WorkspaceManager = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.workspace = workspace
        self.timeout = config.limits.timeout_seconds
        self.max_steps = 25

    def run(self, task: str, task_log: TaskLog, context: Optional[Dict] = None) -> Dict[str, Any]:
        start = time.time()
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
                "Inspect the project, ensure tests exist and pass, generate missing ones if needed."
            ),
        ))

        steps: List[Dict] = []
        final_result = None
        status = "running"
        tools_schema = self._tools_schema()

        while True:
            if time.time() - start > self.timeout:
                status = "aborted"
                final_result = "Timeout"
                break
            if len(steps) > self.max_steps:
                status = "aborted"
                final_result = "Too many steps"
                break

            response = self.model.chat(history, tools=tools_schema)
            content = response.content or ""
            history.append(Message(role="assistant", content=content))

            tool_calls = []
            if response.tool_calls:
                for tc in response.tool_calls:
                    tool_calls.append({"name": tc.get("name", ""), "args": tc.get("arguments", {})})
            else:
                tool_calls = self._parse_tool_calls(content)

            if tool_calls:
                for tc in tool_calls:
                    name = tc["name"]
                    args = tc["args"]
                    result = self.tools.execute(name, **args)

                    if result.error in ("PERMISSION_REQUIRED", "CRITICAL_COMMAND"):
                        task_log.status = "waiting_permission"
                        return {
                            "status": "waiting_permission",
                            "result": None,
                            "permission": result.metadata.get("permission_request"),
                            "steps": steps,
                            "agent": self.name,
                        }

                    summary = str(result.output)[:500] if result.success else (result.error or "error")
                    task_log.add_tool(name, args, result.success, summary)
                    steps.append({
                        "tool": name,
                        "args": {k: str(v)[:80] for k, v in args.items()},
                        "success": result.success,
                        "output_preview": summary[:250],
                    })
                    history.append(Message(
                        role="user",
                        content=f"TOOL_RESULT ({name}): success={result.success}\n{summary}",
                    ))

            if "FINAL:" in content.upper() or content.strip().startswith("FINAL"):
                final_result = content
                status = "success"
                break

            if not tool_calls and len(steps) == 0:
                history.append(Message(
                    role="user",
                    content="Start by listing project files with list_files.",
                ))

        task_log.finish(status, final_result)
        return {
            "status": status,
            "result": final_result,
            "steps": steps,
            "task_log": task_log.to_dict(),
            "agent": self.name,
        }

    def _tools_schema(self) -> List[Dict]:
        schemas = []
        for t in self.tools.list_tools():
            schemas.append({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters or {"type": "object", "properties": {}},
                },
            })
        return schemas

    def _parse_tool_calls(self, text: str) -> List[Dict]:
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
