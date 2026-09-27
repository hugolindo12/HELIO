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

# Create essential dirs
for d in [WORKSPACE_ROOT, KNOWLEDGE_DIR, MEMORY_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


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




# Global config instance (can be overridden)
config = HeiloConfig()
