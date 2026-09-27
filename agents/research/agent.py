"""
HEILO RESEARCH Agent
Specialized in gathering, synthesizing and optionally storing technical knowledge.
"""
from typing import Any, Dict, Optional, List
import re
import time
from heilo.agents.base import BaseAgent
from heilo.core.model_adapter import Message
from heilo.core.task import TaskLog
from heilo.config import config


class ResearchAgent(BaseAgent):
    name = "HEILO_RESEARCH"
    description = (
        "Specialized research agent: searches local knowledge, project code, "
        "and the public web; synthesizes findings and can save verified notes."
    )
    instructions = """
You are HEILO RESEARCH. Your job:
1. Understand the research question
2. Search local knowledge base first (search_knowledge)
3. Search project code if relevant (search_code, list_files, read_file)
4. Search the web when local knowledge is insufficient (search_web)
5. Optionally fetch important pages (fetch_url)
6. Synthesize a clear answer with sources
7. Optionally save important findings (save_research_note) when verified

Rules:
- Prefer local knowledge before the web
- Always cite sources (file paths or URLs)
- Be concise and structured
- Use tools: TOOL: name key="value"
- End with FINAL: report including sources
"""

    def __init__(self, *args, workspace=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.workspace = workspace
        self.timeout = min(config.limits.timeout_seconds, 180)
        self.max_steps = 20

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
                f"Research task: {task}\n"
                f"Workspace: {self.workspace.root if self.workspace else 'n/a'}\n"
                "Start by searching the local knowledge base, then expand if needed."
            ),
        ))

        steps: List[Dict] = []
        sources: List[str] = []
        final_result = None
        status = "running"
        tools_schema = self._tools_schema()

        while True:
            if time.time() - start > self.timeout:
                status = "aborted"
                final_result = "Timeout during research"
                break
            if len(steps) > self.max_steps:
                status = "aborted"
                final_result = "Too many research steps"
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

                    summary = str(result.output)[:800] if result.success else (result.error or "error")
                    task_log.add_tool(name, args, result.success, summary)
                    steps.append({
                        "tool": name,
                        "args": {k: str(v)[:80] for k, v in args.items()},
                        "success": result.success,
                        "output_preview": summary[:300],
                    })

                    # Collect sources
                    if result.success:
                        if name == "search_knowledge" and isinstance(result.output, list):
                            for h in result.output[:5]:
                                sources.append(str(h)[:120])
                        elif name == "search_web" and isinstance(result.output, list):
                            for item in result.output:
                                if isinstance(item, dict) and item.get("url"):
                                    sources.append(item["url"])
                        elif name == "fetch_url":
                            sources.append(args.get("url", ""))
                        elif name in ("read_file", "search_code"):
                            sources.append(args.get("path", args.get("query", "code")))

                    history.append(Message(
                        role="user",
                        content=f"TOOL_RESULT ({name}): success={result.success}\n{summary}",
                    ))

            if "FINAL:" in content.upper() or content.strip().startswith("FINAL"):
                final_result = content
                if sources:
                    final_result += "\n\nSources collected:\n" + "\n".join(f"- {s}" for s in sources[:12])
                status = "success"
                break

            if not tool_calls and len(steps) == 0:
                history.append(Message(
                    role="user",
                    content='Start with: TOOL: search_knowledge query="' + task[:80] + '"',
                ))

        task_log.finish(status, final_result)
        task_log.metadata["sources"] = sources[:20]
        return {
            "status": status,
            "result": final_result,
            "steps": steps,
            "sources": sources[:20],
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
                if name in ("search_knowledge", "search_web", "search_code"):
                    args["query"] = rest
                elif name == "fetch_url":
                    args["url"] = rest
            calls.append({"name": name, "args": args})
        return calls
