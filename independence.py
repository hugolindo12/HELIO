"""
HEILO Independence Test — a HEILO continua existindo sem o Teacher (Qwen)?

Executa numa CÓPIA da HEILO (pasta temporária), sem internet, sem git e sem
chaves de API, em três rodadas:

    A) Teacher habilitado (instalação normal)
    B) Teacher desligado  (HEILO_TEACHER_ENABLED=false)
    C) Teacher REMOVIDO   (pasta models/teacher apagada + transformers/peft ausentes)

Em cada rodada verifica: iniciar, configuração, Seed, conversa, memória,
conhecimento, /ensinar, /aprender e funções que exigem o Teacher.

Uso:  python -m heilo.main independencia
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

PKG_DIR = Path(__file__).resolve().parent

_IGNORAR = shutil.ignore_patterns(
    ".git", "__pycache__", ".pytest_cache", "_to_delete", "logs", "sessions",
    "raw", "datasets", "web_cache", "*.tmp", ".env",
)

# Roda dentro do processo filho. Bloqueia internet antes de importar a HEILO.
_SONDA = r'''
import json, socket, sys, os
def _sem_rede(*a, **k):
    raise OSError("sem internet (HEILO Independence Test)")
socket.socket.connect = _sem_rede
socket.create_connection = _sem_rede
if os.environ.get("HEILO_SIMULAR_SEM_QWEN") == "1":
    for m in ("transformers", "peft"):
        sys.modules[m] = None
res = {}
def passo(nome, fn):
    try:
        res[nome] = {"ok": True, "detalhe": fn()}
    except Exception as e:
        res[nome] = {"ok": False, "detalhe": f"{type(e).__name__}: {e}"}
try:
    from heilo.core.orchestrator import Orchestrator
    from heilo.core import commands
    from heilo.config import config
    o = Orchestrator()
    res["iniciar"] = {"ok": True, "detalhe": "Orchestrator criado"}
except Exception as e:
    res["iniciar"] = {"ok": False, "detalhe": f"{type(e).__name__}: {e}"}
    print(json.dumps(res, ensure_ascii=False)); sys.exit(0)
passo("configuracao", lambda: {"brain_mode": config.brain_mode,
                               "teacher_enabled": config.teacher_enabled})
passo("seed", lambda: o.models.status()["seed"]["reason"])
passo("teacher", lambda: o.models.status()["teacher"]["reason"])
def _conversa():
    r = o.chat("oi")
    return {"fonte": r.get("source"), "modelo": r.get("model"), "resposta": r["content"][:80]}
passo("conversar", _conversa)
passo("memoria", lambda: o.memory_mgr.stats()["trocas"])
passo("ensinar", lambda: commands.handle(o, "/ensinar qual meu nome? => Seu nome é Hugo."))
passo("conhecimento", lambda: [h["source"] for h in o.knowledge_mgr.search("qual meu nome", top_k=3)])
passo("aprender", lambda: {k: v for k, v in o.aprender().items() if k != "dataset"})
passo("comparar", lambda: o.models.compare([{"role": "user", "content": "oi"}])["teacher"][:80])
passo("gerar_com_teacher", lambda: commands.handle(o, "/gerar o que é python?")[:120])
print("@@RESULTADO@@" + json.dumps(res, ensure_ascii=False))
'''


def _rodar(copia: Path, env_extra: Dict[str, str]) -> Dict:
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("OPENAI", "HEILO_", "HF_"))}
    vazio = copia / "_sem_path"
    vazio.mkdir(exist_ok=True)
    env.update({
        "PATH": str(vazio),              # sem git (e sem nada mais) no PATH
        "PYTHONPATH": str(copia.parent),
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HEILO_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
        "HEILO_MEMORY_LOG": "true",
        **env_extra,
    })
    p = subprocess.run([sys.executable, "-c", _SONDA], cwd=str(copia.parent), env=env,
                       capture_output=True, text=True, encoding="utf-8", timeout=600)
    saida = p.stdout.split("@@RESULTADO@@")
    if len(saida) < 2:
        return {"iniciar": {"ok": False, "detalhe": (p.stderr or p.stdout)[-800:]}}
    return json.loads(saida[-1])


def run_independence_test(destino: Path = None) -> Dict:
    base = Path(destino or tempfile.mkdtemp(prefix="heilo_independencia_"))
    copia = base / "heilo"
    if copia.exists():
        shutil.rmtree(copia)
    shutil.copytree(PKG_DIR, copia, ignore=_IGNORAR)

    relatorio: Dict = {"copia": str(copia),
                       "seed_pesos_presentes": (copia / "models/seed/weights/heilo_seed.pt").exists()}
    relatorio["A_teacher_habilitado"] = _rodar(copia, {"HEILO_TEACHER_ENABLED": "true"})
    relatorio["B_teacher_desligado"] = _rodar(copia, {"HEILO_TEACHER_ENABLED": "false"})
    shutil.rmtree(copia / "models" / "teacher", ignore_errors=True)
    relatorio["C_teacher_removido"] = _rodar(copia, {"HEILO_TEACHER_ENABLED": "true",
                                                     "HEILO_SIMULAR_SEM_QWEN": "1"})
    return relatorio


ESSENCIAIS = ("iniciar", "configuracao", "conversar", "memoria", "ensinar",
              "conhecimento", "aprender")


def resumo(rel: Dict) -> List[str]:
    linhas = [f"Cópia testada: {rel['copia']}",
              f"Pesos do Seed presentes: {rel['seed_pesos_presentes']}"]
    for rodada in ("A_teacher_habilitado", "B_teacher_desligado", "C_teacher_removido"):
        r = rel[rodada]
        falhas = [k for k in ESSENCIAIS if not r.get(k, {}).get("ok")]
        linhas.append(f"\n{rodada}: {'OK' if not falhas else 'FALHOU em ' + ', '.join(falhas)}")
        for k, v in r.items():
            linhas.append(f"  {'✔' if v.get('ok') else '✘'} {k}: {v.get('detalhe')}")
    return linhas


if __name__ == "__main__":
    print("\n".join(resumo(run_independence_test())))
