"""HEILO Seed — modelo próprio experimental da HEILO, treinado do zero."""
from pathlib import Path

SEED_DIR = Path(__file__).resolve().parent
SEED_WEIGHTS = SEED_DIR / "weights" / "heilo_seed.pt"
