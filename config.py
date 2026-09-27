"""
HEILO Configuration
"""
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import os

BASE_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = BASE_DIR / "workspace"
KNOWLEDGE_DIR = BASE_DIR / "knowledge"
MEMORY_DIR = BASE_DIR / "memory"
LOGS_DIR = BASE_DIR / "logs"
# HEILO Training: raw/ teacher/ taught/ approved/ rejected/ datasets/ training/
DATA_DIR = BASE_DIR / "data"
# Componentes de modelo substituíveis: models/seed (próprio) e models/teacher (opcional)
MODELS_DIR = BASE_DIR / "models"

# Create essential dirs
for d in [WORKSPACE_ROOT, KNOWLEDGE_DIR, MEMORY_DIR, LOGS_DIR, DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def _load_dotenv(path: Path) -> None:
    """Lê heilo/.env (KEY=VALUE) sem sobrescrever variáveis já definidas.

    Segredos (tokens, API keys) ficam no .env, que está no .gitignore — nunca no código.
    """
    if not path.exists():
        return
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    except OSError:
        pass


_load_dotenv(BASE_DIR / ".env")

_TRUE = {"1", "true", "yes", "sim", "on"}
_FALSE = {"0", "false", "no", "nao", "não", "off"}


def env_bool(name: str, default: bool) -> bool:
    """Lê um booleano do ambiente. Valor inválido → default (com aviso)."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    v = raw.strip().lower()
    if v in _TRUE:
        return True
    if v in _FALSE:
        return False
    print(f"[HEILO config] {name}={raw!r} inválido; usando {default}.")
    return default


# Modos do cérebro de conversa. "mini"/"lora" são aliases antigos (compatibilidade).
BRAIN_MODES = ("auto", "seed", "teacher", "off")
BRAIN_MODE_ALIASES = {"mini": "seed", "lora": "teacher"}


def normalize_brain_mode(mode: Optional[str]) -> str:
    """Normaliza o modo do cérebro. Valor inválido → 'auto' (com aviso), nunca quebra."""
    m = (mode or "auto").strip().lower()
    m = BRAIN_MODE_ALIASES.get(m, m)
    if m not in BRAIN_MODES:
        print(f"[HEILO config] brain_mode={mode!r} inválido; usando 'auto'. "
              f"Opções: {', '.join(BRAIN_MODES)}.")
        return "auto"
    return m


@dataclass
class ModelConfig:
    provider: str = "openai"  # openai | ollama | anthropic | stub
    model_name: str = "gpt-4o-mini"
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    base_url: Optional[str] = None  # for local/ollama
    temperature: float = 0.2
    max_tokens: int = 4096


@dataclass
class AgentLimits:
    max_attempts: int = 5
    max_commands: int = 30
    timeout_seconds: int = 300
    max_file_reads: int = 50
    max_file_writes: int = 20


@dataclass
class SecurityConfig:
    default_workspace: Path = field(default_factory=lambda: WORKSPACE_ROOT / "demo_project")
    allow_absolute_paths: bool = False
    require_confirmation_for_critical: bool = True
    blocked_patterns: list = field(default_factory=lambda: [
        r"\.\.", r"/etc", r"/usr", r"/bin", r"/sbin", r"C:\\Windows", r"System32"
    ])


@dataclass
class HeiloConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    limits: AgentLimits = field(default_factory=AgentLimits)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    log_level: str = "INFO"
    enable_rag: bool = True
    enable_memory: bool = True
    user_timezone: str = "America/Sao_Paulo"
    # Prefer HEILO's own knowledge over external LLM
    local_first: bool = True
    knowledge_confidence_threshold: float = 0.35
    # model is optional fallback when local knowledge is insufficient
    llm_fallback: bool = True
    # Learn from errors/corrections at runtime
    auto_learn: bool = True
    # After learning, auto git-commit knowledge (no push unless enabled)
    auto_commit_knowledge: bool = False
    auto_push_knowledge: bool = False
    knowledge_git_remote: str = ""  # e.g. https://github.com/user/heilo-knowledge.git
    # ---- Modelos (HEILO Core → Model Manager → Model Adapters) -------------
    # auto | seed | teacher | off   (aliases antigos: mini=seed, lora=teacher)
    brain_mode: str = field(default_factory=lambda: normalize_brain_mode(
        os.getenv("HEILO_BRAIN_MODE", "auto")))
    # HEILO Teacher (modelo externo temporário). false = HEILO funciona só com o Seed.
    teacher_enabled: bool = field(default_factory=lambda: env_bool("HEILO_TEACHER_ENABLED", True))
    # O Teacher só responde no chat se isto for true (ou com /cerebro teacher).
    # Por padrão ele é professor: gera dados e é comparado, não é o cérebro principal.
    teacher_in_chat: bool = field(default_factory=lambda: env_bool("HEILO_TEACHER_IN_CHAT", False))
    # ---- HEILO Memory --------------------------------------------------------
    # Grava cada troca do chat na memória local (memory/sessions/). Não treina nada.
    memory_log_conversations: bool = field(default_factory=lambda: env_bool(
        "HEILO_MEMORY_LOG", True))

    # Compatibilidade com o nome antigo (brain_log_conversations)
    @property
    def brain_log_conversations(self) -> bool:
        return self.memory_log_conversations

    @brain_log_conversations.setter
    def brain_log_conversations(self, value: bool) -> None:
        self.memory_log_conversations = bool(value)




# Global config instance (can be overridden)
config = HeiloConfig()
