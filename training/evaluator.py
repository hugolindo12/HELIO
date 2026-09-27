"""
HEILO Seed Model Evaluator
Computes objective cross-entropy evaluation loss and qualitative benchmark scores
on held-out evaluation datasets.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import torch
import torch.nn.functional as F

from heilo.models.seed.gpt import (
    MiniGPT, GPTConfig, carregar, encode_chat, decode, HEILO
)
from heilo.training.records import read_jsonl

DEFAULT_EVAL_PATH = Path(__file__).resolve().parent.parent / "data" / "datasets" / "evaluation" / "benchmark_eval.jsonl"


class ModelEvaluator:
    """Evaluates HEILO Seed versions without external dependencies."""

    def __init__(self, eval_file: Optional[Path] = None):
        self.eval_file = Path(eval_file or DEFAULT_EVAL_PATH)

    def load_examples(self) -> List[Dict]:
        if not self.eval_file.exists():
            return []
        return read_jsonl(self.eval_file)

    def evaluate_loss(self, model: MiniGPT, examples: List[Dict]) -> Optional[float]:
        """Calculates exact token cross-entropy loss on evaluation examples."""
        if not examples or model is None:
            return None
        model.eval()
        dev = next(model.parameters()).device
        T = model.cfg.block_size
        losses = []

        with torch.no_grad():
            for ex in examples:
                ids = encode_chat(ex["messages"])[-(T + 1):]
                if len(ids) < 2:
                    continue
                x = torch.tensor([ids[:-1]], device=dev)
                y = torch.tensor([ids[1:]], device=dev)
                logits, _ = model(x)
                loss = F.cross_entropy(logits[0], y[0]).item()
                losses.append(loss)

        return sum(losses) / len(losses) if losses else None

    def evaluate_checkpoint(self, checkpoint_path: Path) -> Dict[str, Any]:
        """Runs full evaluation (loss + qualitative generation) on a checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            return {"error": f"Checkpoint not found: {checkpoint_path}"}

        model, info = carregar(checkpoint_path)
        if model is None:
            return {"error": "Failed to load model from checkpoint"}

        examples = self.load_examples()
        eval_loss = self.evaluate_loss(model, examples)

        # Qualitative generation test
        sample_results = []
        dev = next(model.parameters()).device
        bloco = model.cfg.block_size
        limite = max(bloco // 2, bloco - 64)

        for ex in examples:
            user_msg = ex["question"]
            conv = [{"role": "user", "content": user_msg}]
            ids = encode_chat(conv) + [HEILO]
            ids = ids[-limite:]
            gen_tokens = model.gerar(ids, max_novos=150, temperatura=0.4, top_k=20)
            gen_text = decode(gen_tokens).strip()

            sample_results.append({
                "id": ex.get("id"),
                "category": ex.get("category"),
                "question": user_msg,
                "target": ex.get("target"),
                "seed_output": gen_text,
            })

        return {
            "checkpoint": str(checkpoint_path.name),
            "eval_examples_count": len(examples),
            "eval_loss": round(eval_loss, 4) if eval_loss is not None else None,
            "steps_trained": info.get("passos_totais", 0),
            "samples": sample_results,
        }
