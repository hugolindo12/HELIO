"""Unit tests for heilo.core.pipeline"""
from heilo.core.pipeline import (
    PipelineResult,
    PipelineStepResult,
    build_code_task_from_research,
    build_test_task_from_code,
    format_numbered_citations,
    format_pipeline_report,
)


class TestBuildTasks:
    def test_build_code_task_contains_research(self):
        task = build_code_task_from_research(
            "corrija desconto",
            "use total * (1 - percent/100)",
        )
        assert "RESEARCH" in task or "Pesquisa" in task or "pesquisa" in task.lower()
        assert "total * (1 - percent/100)" in task
        assert "corrija desconto" in task

    def test_build_code_task_truncates_long_research(self):
        long = "X" * 5000
        task = build_code_task_from_research("q", long)
        assert len(task) < 5000 + 500

    def test_build_test_task(self):
        task = build_test_task_from_code("rode testes", "CODE: fixed apply_discount")
        assert "pytest" in task.lower() or "testes" in task.lower()
        assert "apply_discount" in task or "CODE" in task


class TestCitations:
    def test_empty_sources(self):
        assert format_numbered_citations([]) == ""

    def test_numbered_and_dedup(self):
        src = ["a.md", "b.md", "a.md", "c.md"]
        out = format_numbered_citations(src)
        assert "[1]" in out
        assert "[2]" in out
        assert "[3]" in out
        assert out.count("[1]") == 1
        assert "a.md" in out

    def test_limit_12(self):
        src = [f"file{i}.md" for i in range(20)]
        out = format_numbered_citations(src)
        assert "[12]" in out
        assert "[13]" not in out


class TestPipelineResult:
    def test_to_dict(self):
        pr = PipelineResult(name="research_code_test", status="success")
        pr.steps.append(PipelineStepResult(
            agent="HEILO_RESEARCH",
            status="success",
            steps=[{"tool": "search_knowledge"}],
            sources=["doc.md"],
            task_id="abc",
        ))
        d = pr.to_dict()
        assert d["name"] == "research_code_test"
        assert d["status"] == "success"
        assert d["steps"][0]["agent"] == "HEILO_RESEARCH"
        assert d["steps"][0]["tools"] == 1
        assert d["steps"][0]["task_id"] == "abc"

    def test_format_pipeline_report_two_steps(self):
        report = format_pipeline_report(
            "research_code",
            "research body",
            "code body",
            status="success",
        )
        assert "RESEARCH" in report
        assert "CODE" in report
        assert "research body" in report
        assert "success" in report

    def test_format_pipeline_report_three_steps(self):
        report = format_pipeline_report(
            "research_code_test",
            "R",
            "C",
            "T",
            status="partial",
        )
        assert "### 3. TEST" in report
        assert "T" in report
        assert "partial" in report
