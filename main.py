"""
HEILO - Entry point
"""
import argparse
import sys
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from heilo.core.orchestrator import Orchestrator
from heilo.config import config, WORKSPACE_ROOT


def run_cli():
    print("=" * 50)
    print("  HEILO v0.3 – CODE + TEST + RESEARCH + PIPELINES")
    print("=" * 50)
    orch = Orchestrator()
    print(f"Workspace: {orch.workspace.root}")
    print(f"Model: {orch.model.cfg.provider}/{orch.model.cfg.model_name}")
    print("Digite 'sair' para encerrar.\n")
    while True:
        try:
            user = input("Voce: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nEncerrando.")
            break
        if not user:
            continue
        if user.lower() in ("sair", "exit", "quit"):
            break
        result = orch.chat(user)
        print(f"\n[{result.get('agent', 'HEILO')}]")
        print(result.get("content", result))
        if result.get("type") == "permission_request":
            ans = input("\nAutorizar? [s/N]: ").strip().lower()
            if ans in ("s", "sim", "y", "yes"):
                print(orch.approve_permission(grant_session=True).get("content"))
        print()


def run_server(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    from heilo.ui.api import app
    print(f"HEILO UI -> http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)


def _reset_demo_project(demo: Path, buggy: bool = True):
    demo.mkdir(parents=True, exist_ok=True)
    heilo_dir = demo / ".heilo"
    if heilo_dir.exists():
        shutil.rmtree(heilo_dir, ignore_errors=True)
    discount_line = (
        "    return total * percent / 100"
        if buggy
        else "    return total * (1 - percent / 100)"
    )
    lines = [
        '"""Demo project."""',
        "",
        "def calculate_total(prices):",
        "    total = 0",
        "    for p in prices:",
        "        total += p",
        "    return total",
        "",
        "def apply_discount(total, percent):",
        discount_line,
        "",
        'if __name__ == "__main__":',
        "    prices = [10, 20, 30]",
        "    total = calculate_total(prices)",
        "    final = apply_discount(total, 10)",
        '    print(f"Total: {total}, After discount: {final}")',
        '    assert abs(final - 54.0) < 0.01, f"Expected 54.0, got {final}"',
        "",
    ]
    (demo / "main.py").write_text("\n".join(lines), encoding="utf-8")
    test_lines = [
        "from main import calculate_total, apply_discount",
        "",
        "def test_calculate():",
        "    assert calculate_total([10, 20, 30]) == 60",
        "",
        "def test_discount():",
        "    assert abs(apply_discount(60, 10) - 54.0) < 0.01",
        "",
    ]
    (demo / "test_main.py").write_text("\n".join(test_lines), encoding="utf-8")


def run_demo():
    print("=== HEILO Demo: CODE ===\n")
    config.security.require_confirmation_for_critical = False
    orch = Orchestrator()
    demo = WORKSPACE_ROOT / "demo_project"
    _reset_demo_project(demo, buggy=True)
    orch.set_workspace(str(demo))
    result = orch.chat("Analise este projeto e encontre erros. Corrija se possivel.")
    print(f"[{result.get('agent')}] status={result.get('status')}")
    print((result.get("content") or "")[:2000])
    print("=== done ===")
    return result


def run_demo_test():
    print("=== HEILO Demo: TEST ===\n")
    config.security.require_confirmation_for_critical = False
    orch = Orchestrator()
    demo = WORKSPACE_ROOT / "demo_project"
    _reset_demo_project(demo, buggy=False)
    orch.set_workspace(str(demo))
    result = orch.chat("Rode os testes e gere um relatorio.")
    print(f"[{result.get('agent')}] status={result.get('status')}")
    print((result.get("content") or "")[:2000])
    print("=== done ===")
    return result


def run_demo_research():
    print("=== HEILO Demo: RESEARCH ===\n")
    config.security.require_confirmation_for_critical = False
    orch = Orchestrator()
    result = orch.chat("Pesquise como calcular desconto percentual em Python corretamente.")
    print(f"[{result.get('agent')}] status={result.get('status')}")
    print((result.get("content") or "")[:2500])
    print("=== done ===")
    return result


def run_demo_pipeline():
    print("=== HEILO Demo: RESEARCH -> CODE ===\n")
    config.security.require_confirmation_for_critical = False
    orch = Orchestrator()
    demo = WORKSPACE_ROOT / "demo_project"
    _reset_demo_project(demo, buggy=True)
    orch.set_workspace(str(demo))
    msg = "Pesquise a solucao para calculo de desconto em Python e aplique no projeto."
    print(f"Voce: {msg}\nintent: {orch._classify_intent(msg)}\n")
    result = orch.chat(msg)
    print(f"[{result.get('agent')}] status={result.get('status')}")
    print((result.get("content") or "")[:3000])
    print("=== done ===")
    return result


def run_demo_pipeline_full():
    print("=== HEILO Demo: RESEARCH -> CODE -> TEST ===\n")
    config.security.require_confirmation_for_critical = False
    orch = Orchestrator()
    demo = WORKSPACE_ROOT / "demo_project"
    _reset_demo_project(demo, buggy=True)
    orch.set_workspace(str(demo))
    msg = "Pesquise a solucao de desconto em Python, aplique no projeto e rode os testes."
    print(f"Voce: {msg}\nintent: {orch._classify_intent(msg)}\n")
    result = orch.chat(msg)
    print(f"[{result.get('agent')}] status={result.get('status')}")
    print((result.get("content") or "")[:4000])
    if result.get("pipeline"):
        print("\nPipeline meta:", result["pipeline"])
    print("\n--- main.py ---")
    print((demo / "main.py").read_text(encoding="utf-8"))
    print("=== done ===")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HEILO Platform")
    parser.add_argument(
        "mode",
        nargs="?",
        default="cli",
        choices=[
            "cli", "server", "demo", "demo_test", "demo_research",
            "demo_pipeline", "demo_pipeline_full",
        ],
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    {
        "cli": run_cli,
        "server": lambda: run_server(args.host, args.port),
        "demo": run_demo,
        "demo_test": run_demo_test,
        "demo_research": run_demo_research,
        "demo_pipeline": run_demo_pipeline,
        "demo_pipeline_full": run_demo_pipeline_full,
    }[args.mode]()
