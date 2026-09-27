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
    from heilo.core import commands
    print("=" * 50)
    print("  HEILO – Core + Seed + Memory + Knowledge (+ Teacher opcional)")
    print("=" * 50)
    orch = Orchestrator()
    print(f"Workspace: {orch.workspace.root}")
    print(commands.handle(orch, "/cerebro"))
    print(commands.AJUDA)
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
        if user.startswith("/"):
            print(commands.handle(orch, user) + "\n")
            continue
        result = orch.chat(user)
        print(f"\n[{result.get('agent', 'HEILO')}]")
        print(result.get("content", result))
        if result.get("type") == "permission_request":
            ans = input("\nAutorizar? [s/N]: ").strip().lower()
            if ans in ("s", "sim", "y", "yes"):
                print(orch.approve_permission(grant_session=True).get("content"))
        print()


def _pipeline():
    """TrainingPipeline sem subir o Orchestrator inteiro."""
    from heilo.config import config
    from heilo.core.model_manager import ModelManager
    from heilo.knowledge.manager import KnowledgeManager
    from heilo.memory.manager import MemoryManager
    from heilo.training.pipeline import TrainingPipeline
    return TrainingPipeline(models=ModelManager.from_config(config), memory=MemoryManager(),
                            knowledge=KnowledgeManager())


def run_aprender():
    """Memory → seleção → validação → dataset. Não treina e não envia nada ao GitHub."""
    import json
    print(json.dumps(_pipeline().learn(), ensure_ascii=False, indent=2))


def run_dataset():
    import json
    print(json.dumps(_pipeline().build_dataset(), ensure_ascii=False, indent=2))


def run_treinar(passos: int = 2000, novo: bool = False):
    """Treina uma versão nova do HEILO Seed numa pasta própria, avalia e só promove se melhorar.
    (--novo recomeça do zero: só use se quiser descartar o aprendizado atual.)"""
    import json
    from heilo.training.ciclos import CiclosSeed
    from heilo.training.pipeline import TrainingPipeline
    pipe = TrainingPipeline()
    if novo:
        print(json.dumps(pipe.train_seed(passos=passos, novo=True), ensure_ascii=False, indent=2))
        return
    dec = CiclosSeed(pipe).treinar_versao(passos=passos, origem="python -m heilo.main treinar")
    print(f"{dec['versao']}: {'PROMOVIDA' if dec['promovida'] else 'não promovida'} ({dec['motivo']})")


def run_revisar():
    from heilo.core import commands
    print(commands.revisar(Orchestrator()))


def run_gerar(arquivo: str = ""):
    """Teacher gera exemplos (NÃO verificados) para as perguntas de um arquivo, 1 por linha."""
    from heilo.training.pipeline import TeacherRequired
    if not arquivo:
        print("Uso: python -m heilo.main gerar --arquivo perguntas.txt")
        return
    perguntas = Path(arquivo).read_text(encoding="utf-8").splitlines()
    try:
        print(_pipeline().generate_with_teacher(perguntas, log=print))
    except TeacherRequired as e:
        print(e)


def run_professor(arquivo: str = "", limite: int = 0):
    """HEILO Teacher responde a lista de perguntas → validação → PENDENTE de revisão."""
    from heilo.config import config
    from heilo.core.model_manager import ModelManager
    from heilo.training.pipeline import TeacherRequired, TrainingPipeline
    from heilo.training.professor import gerar_com_professor
    models = ModelManager.from_config(config)
    try:
        r = gerar_com_professor(TrainingPipeline(models=models), models,
                                arquivo=Path(arquivo) if arquivo else None, limite=limite)
    except TeacherRequired as e:
        print(e)
        return
    print(r)
    print("Pronto. Revise na interface (abrir_heilo.bat → Revisar aprendizados). "
          "Nada entra no treino sem a sua aprovação.")


def run_auto_treino(passos: int = 0, forcar: bool = False):
    """Treino automático (sem ninguém olhando): só roda se houver exemplos APROVADOS novos
    desde a última vez e no máximo 1× a cada 12 h. Treina uma CANDIDATA a partir da versão
    ativa, avalia na mesma régua e só promove se melhorar. Nunca aprende com conversas não
    revisadas."""
    import hashlib
    import json
    import time
    from heilo.config import DATA_DIR
    from heilo.training.ciclos import CiclosSeed
    from heilo.training.pipeline import TrainingPipeline
    from heilo.training.records import agora
    estado_arq = DATA_DIR / "training" / "auto_treino.json"
    estado = json.loads(estado_arq.read_text(encoding="utf-8")) if estado_arq.exists() else {}
    pipe = TrainingPipeline()
    ids = sorted(r["id"] for r in pipe.approved())
    assinatura = hashlib.sha256("\n".join(ids).encode()).hexdigest()[:16]
    if not forcar and estado.get("assinatura") == assinatura:
        print(f"Nada novo para aprender ({len(ids)} exemplos aprovados, iguais ao último treino).")
        return
    if not forcar and time.time() - estado.get("quando_ts", 0) < 12 * 3600:
        print("Já treinei nas últimas 12 horas. Deixo para depois.")
        return
    ciclos = CiclosSeed(pipe)
    reg = ciclos.registro()
    ativa = next(v for v in reg["versoes"] if v["versao"] == reg["ativa"])
    if not passos:   # modelo pequeno (v0.x) treina mais passos; grande (30 M+) menos, para caber na CPU
        passos = 1500 if "v0." in ativa["versao"] else 400
    print(f"Aprendendo com {len(ids)} exemplos aprovados (novos desde o último treino). "
          f"Base: {ativa['versao']}, {passos} passos. Pode demorar; o PC fica mais lento.")
    dec = ciclos.treinar_versao(passos=passos, origem="auto-treino (exemplos aprovados novos)")
    registro = {"quando": agora(), "quando_ts": time.time(), "assinatura": assinatura,
                "exemplos": len(ids), "versao": dec["versao"], "promovida": dec["promovida"],
                "motivo": dec["motivo"]}
    estado_arq.write_text(json.dumps(registro, ensure_ascii=False, indent=1), encoding="utf-8")
    with open(DATA_DIR / "training" / "auto_treino.log", "a", encoding="utf-8") as f:
        f.write(json.dumps(registro, ensure_ascii=False) + "\n")
    print(f"{dec['versao']}: {'PROMOVIDA — virou a HEILO ativa' if dec['promovida'] else 'não promovida'} "
          f"({dec['motivo']})")


def run_regua():
    """Recalcula a sonda de todas as versões com a régua atual (usa as respostas salvas)."""
    from heilo.config import DATA_DIR
    from heilo.training.eval_set import reavaliar_sonda_salva
    reavaliar_sonda_salva(DATA_DIR / "training" / "eval", DATA_DIR / "training" / "versions.json")


def run_metricas():
    import json
    print(json.dumps(_pipeline().metrics(), ensure_ascii=False, indent=2))


def run_publicar():
    """Versiona no Git o que é apropriado (conhecimento, dados aprovados, métricas) e faz push.
    Memória de conversas NÃO vai para o GitHub."""
    from heilo.knowledge.git_sync import KnowledgeGitSync
    r = KnowledgeGitSync().sync(message="heilo: conhecimento e dados aprovados", push=True)
    print(r.get("push") or r.get("commit"))


def run_independencia():
    """HEILO Independence Test: cópia da HEILO sem internet, sem git e sem o Teacher."""
    from heilo.independence import resumo, run_independence_test
    print("\n".join(resumo(run_independence_test())))


def run_ciclo(max_ciclos: int = 4, passos: int = 2500, base_n: int = 20,
              sem_professor: bool = False):
    """Ciclos automáticos: Teacher gera dados → validação → treino → avaliação → promoção."""
    from heilo.config import config
    from heilo.core.model_manager import ModelManager
    from heilo.training.ciclos import CiclosSeed
    from heilo.training.pipeline import TrainingPipeline
    models = ModelManager.from_config(config)
    pipe = TrainingPipeline(models=models)
    ciclos = CiclosSeed(pipe, models=models)
    rel = ciclos.executar(max_ciclos=max_ciclos, passos=passos, base_n=base_n,
                          usar_professor=not sem_professor)
    print(ciclos.escrever_relatorio(rel).read_text(encoding="utf-8"))


def run_versoes():
    from heilo.training.ciclos import CiclosSeed
    from heilo.training.pipeline import TrainingPipeline
    print(CiclosSeed(TrainingPipeline(), log=lambda *_: None).escrever_relatorio().read_text(encoding="utf-8"))


def run_promover(versao: str, motivo: str):
    """Promove manualmente uma versão já avaliada (fica registrado quem/por quê)."""
    from heilo.training.ciclos import CiclosSeed
    from heilo.training.pipeline import TrainingPipeline
    if not versao:
        print("Uso: python -m heilo.main promover --versao v1.0 --motivo \"...\"")
        return
    v = CiclosSeed(TrainingPipeline()).promover_manual(versao, motivo or "decisão do usuário")
    print(f"{v['versao']} agora é a versão ativa.")


def run_atualizar():
    """Baixa do GitHub código/dados/pesos publicados (ex.: Seed treinado no Colab)."""
    from heilo.knowledge.git_sync import KnowledgeGitSync
    print(KnowledgeGitSync()._run(["git", "pull", "--rebase", "--autostash"], timeout=300))


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
            "dataset", "treinar", "aprender", "atualizar",
            "revisar", "gerar", "professor", "regua", "auto_treino", "metricas", "publicar", "independencia", "ciclo", "versoes", "promover",
        ],
    )
    parser.add_argument("--passos", type=int, default=2000, help="passos de treino do HEILO Seed")
    parser.add_argument("--novo", action="store_true", help="recomeça o HEILO Seed do zero")
    parser.add_argument("--arquivo", default="", help="perguntas para o Teacher (modo gerar)")
    parser.add_argument("--max-ciclos", type=int, default=4, help="ciclos de treino (modo ciclo)")
    parser.add_argument("--base-n", type=int, default=20, help="exemplos por categoria por ciclo")
    parser.add_argument("--sem-professor", action="store_true", help="ciclo sem o Teacher")
    parser.add_argument("--limite", type=int, default=0, help="máximo de perguntas (modo professor)")
    parser.add_argument("--forcar", action="store_true", help="auto_treino mesmo sem exemplos novos")
    parser.add_argument("--versao", default="", help="versão (modo promover)")
    parser.add_argument("--motivo", default="", help="motivo da promoção manual")
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
        "dataset": run_dataset,
        "treinar": lambda: run_treinar(args.passos, args.novo),
        "aprender": run_aprender,
        "atualizar": run_atualizar,
        "revisar": run_revisar,
        "gerar": lambda: run_gerar(args.arquivo),
        "professor": lambda: run_professor(args.arquivo, args.limite),
        "regua": run_regua,
        "auto_treino": lambda: run_auto_treino(args.passos if args.passos != 2000 else 0, args.forcar),
        "metricas": run_metricas,
        "publicar": run_publicar,
        "independencia": run_independencia,
        "ciclo": lambda: run_ciclo(args.max_ciclos, args.passos if args.passos != 2000 else 2500,
                                   args.base_n, args.sem_professor),
        "versoes": run_versoes,
        "promover": lambda: run_promover(args.versao, args.motivo),
    }[args.mode]()
