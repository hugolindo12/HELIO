"""O status final do HEILO CODE deve refletir o resultado real dos testes."""
from unittest.mock import MagicMock

from heilo.agents.code.agent import CodeAgent
from heilo.core.model_adapter import ModelResponse
from heilo.core.task import TaskLog
from heilo.tools.base import ToolResult


def _run_agent(test_success: bool, tmp_path, monkeypatch):
    model = MagicMock()
    model.chat.side_effect = [
        ModelResponse(content="Rodando testes.\nTOOL: run_tests command=pytest"),
        ModelResponse(content="FINAL: relatório"),
    ]
    tools = MagicMock()
    tools.list_tools.return_value = []
    tools.execute.return_value = ToolResult(success=test_success, output="saida")
    agent = CodeAgent(model, tools, MagicMock())
    agent.state_store = MagicMock()
    log = TaskLog(task_id="t1", user_message="corrija", agent="HEILO_CODE", model="fake")
    return agent.run(task="corrija", task_log=log)


def test_final_after_failed_tests_is_failed(tmp_path, monkeypatch):
    assert _run_agent(False, tmp_path, monkeypatch)["status"] == "failed"


def test_final_after_passing_tests_is_success(tmp_path, monkeypatch):
    assert _run_agent(True, tmp_path, monkeypatch)["status"] == "success"
