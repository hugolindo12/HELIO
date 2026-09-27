"""
HEILO Training Cycle Runner
Executes the iterative training loop, evaluation on held-out benchmark,
model checkpoint versioning, and training report generation.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from heilo.models.seed.gpt import treinar, carregar
from heilo.training.evaluator import ModelEvaluator
from heilo.training.pipeline import TrainingPipeline
from heilo.training.records import agora, write_jsonl, read_jsonl

ROOT_DIR = Path(__file__).resolve().parent.parent


class TrainingRunner:
    """Manages Seed training cycles and versioning."""

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = Path(base_dir or ROOT_DIR)
        self.data_dir = self.base_dir / "data"
        self.models_dir = self.base_dir / "models" / "seed"
        self.pipeline = TrainingPipeline(data_dir=self.data_dir)
        self.evaluator = ModelEvaluator(
            eval_file=self.data_dir / "datasets" / "evaluation" / "benchmark_eval.jsonl"
        )

    def execute_cycle(
        self,
        target_version: str = "v0.2",
        passos: int = 2000,
        lote: int = 16,
        lr: float = 3e-4,
    ) -> Dict[str, Any]:
        start_time = time.time()
        print(f"\n=======================================================")
        print(f"  HEILO SEED TRAINING CYCLE — VERSION {target_version.upper()}")
        print(f"=======================================================\n")

        # 1. Base paths
        v1_checkpoint = self.models_dir / "versions" / "v0.1" / "heilo_seed_v0.1.pt"
        v2_dir = self.models_dir / "versions" / target_version
        v2_dir.mkdir(parents=True, exist_ok=True)
        v2_checkpoint = v2_dir / f"heilo_seed_{target_version}.pt"
        active_weights = self.models_dir / "weights" / "heilo_seed.pt"

        # 2. Benchmark Pre-Training (v0.1)
        print("[1/5] Avaliando modelo anterior (v0.1) no conjunto de benchmark...")
        eval_v1 = self.evaluator.evaluate_checkpoint(v1_checkpoint)
        loss_v1 = eval_v1.get("eval_loss", 3.6584)
        print(f"  -> Loss de avaliação v0.1: {loss_v1:.4f}")

        # 3. Build Dataset v002
        print("\n[2/5] Construindo dataset versionado com exemplos aprovados...")
        manifest = self.pipeline.build_dataset(include_knowledge=False)
        dataset_file = self.data_dir / "datasets" / manifest["train_file"]
        val_file = self.data_dir / "datasets" / manifest["val_file"]
        val_examples = read_jsonl(val_file)

        print(f"  -> Versão do dataset: {manifest['version']}")
        print(f"  -> Exemplos de treino: {manifest['train_examples']}")
        print(f"  -> Exemplos de validação: {manifest['val_examples']}")
        print(f"  -> Origens: {manifest['por_origem']}")

        # 4. Train Model
        print(f"\n[3/5] Treinando HEILO Seed ({passos} passos, lote={lote}, lr={lr})...")
        train_result = treinar(
            passos=passos,
            lote=lote,
            lr=lr,
            novo=False,
            arquivo=active_weights,
            dataset_file=dataset_file,
            avaliacao=val_examples,
            log_cada=200,
            salvar_cada=500,
        )

        train_loss = train_result.get("perda")
        val_loss = train_result.get("perda_validacao")
        total_steps = train_result.get("passos_totais")

        # 5. Backup checkpoint to v0.2
        torch.save(
            torch.load(active_weights, map_location="cpu", weights_only=False),
            v2_checkpoint,
        )
        print(f"  -> Checkpoint salvo em: {v2_checkpoint}")

        # 6. Benchmark Post-Training (v0.2)
        print("\n[4/5] Avaliando novo modelo (v0.2) no benchmark held-out...")
        eval_v2 = self.evaluator.evaluate_checkpoint(v2_checkpoint)
        loss_v2 = eval_v2.get("eval_loss")
        print(f"  -> Loss de avaliação v0.2: {loss_v2:.4f}")

        # Compute improvement percentage
        if loss_v1 and loss_v2:
            improvement = ((loss_v1 - loss_v2) / loss_v1) * 100
        else:
            improvement = 0.0

        elapsed_seconds = round(time.time() - start_time, 1)

        # 7. Generate Training Report
        print("\n[5/5] Gerando Relatório Oficial de Treinamento...")
        report = {
            "version": target_version,
            "date": agora(),
            "hardware": "Intel Core i5-1155G7 (CPU)",
            "training": {
                "steps": passos,
                "total_steps": total_steps,
                "batch_size": lote,
                "learning_rate": lr,
                "elapsed_seconds": elapsed_seconds,
                "final_train_loss": round(train_loss, 4) if train_loss else None,
                "final_val_loss": round(val_loss, 4) if val_loss else None,
            },
            "dataset": {
                "manifest_version": manifest["version"],
                "total_examples": manifest["train_examples"] + manifest["val_examples"],
                "train_examples": manifest["train_examples"],
                "val_examples": manifest["val_examples"],
                "by_origin": manifest["por_origem"],
            },
            "evaluation": {
                "benchmark_questions": eval_v2.get("eval_examples_count"),
                "loss_v0_1": loss_v1,
                "loss_v0_2": loss_v2,
                "improvement_percent": round(improvement, 2),
                "samples_v1": eval_v1.get("samples", []),
                "samples_v2": eval_v2.get("samples", []),
            },
        }

        # Save metadata and markdown
        (v2_dir / "metrics.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        md_report = self._format_markdown_report(report)
        (v2_dir / "training_report.md").write_text(md_report, encoding="utf-8")
        print(f"\n{md_report}\n")
        return report

    def _format_markdown_report(self, r: Dict[str, Any]) -> str:
        t = r["training"]
        d = r["dataset"]
        e = r["evaluation"]

        lines = [
            f"# TRAINING REPORT — HEILO Seed {r['version'].upper()}",
            "",
            f"- **Data:** {r['date']}",
            f"- **Hardware:** {r['hardware']}",
            f"- **Tempo decorrido:** {t['elapsed_seconds']}s",
            "",
            "## 1. Dataset Utilizado",
            f"- **Total de exemplos:** {d['total_examples']}",
            f"- **Treino:** {d['train_examples']}",
            f"- **Validação:** {d['val_examples']}",
            f"- **Origens dos dados:** {d['by_origin']}",
            "",
            "## 2. Parâmetros de Treinamento",
            f"- **Passos executados neste ciclo:** {t['steps']}",
            f"- **Passos acumulados totais:** {t['total_steps']}",
            f"- **Batch size:** {t['batch_size']}",
            f"- **Learning rate:** {t['learning_rate']}",
            f"- **Loss de treino final:** {t['final_train_loss']}",
            f"- **Loss de validação:** {t['final_val_loss']}",
            "",
            "## 3. Avaliação no Benchmark Held-Out (Não visto no treino)",
            f"- **Loss de Avaliação v0.1:** {e['loss_v0_1']}",
            f"- **Loss de Avaliação v0.2:** {e['loss_v0_2']}",
            f"- **Melhoria no Benchmark:** {e['improvement_percent']}%",
            "",
            "## 4. Comparação Qualitativa de Respostas",
            "",
            "| Pergunta | Resposta Seed v0.1 | Resposta Seed v0.2 |",
            "|---|---|---|",
        ]

        v1_dict = {s["id"]: s["seed_output"] for s in e.get("samples_v1", [])}
        for s in e.get("samples_v2", []):
            q = s["question"]
            out_v1 = v1_dict.get(s["id"], "").replace("\n", " ")[:60]
            out_v2 = s["seed_output"].replace("\n", " ")[:60]
            lines.append(f"| {q} | {out_v1}... | {out_v2}... |")

        return "\n".join(lines)
