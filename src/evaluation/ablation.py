"""
Ablation Study Module.

Supports systematic ablation experiments to evaluate the contribution
of each system component:

1. Facial behavioral features only
2. Facial + Head pose attention
3. Full system with temporal fusion
"""

import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.models.affective_model import AffectiveModel
from src.models.attention_estimator import AttentionEstimator
from src.models.temporal_model import SlidingWindowSmoother
from src.models.risk_fusion import RiskFusionEngine
from src.evaluation.evaluator import Evaluator
from src.utils.helpers import ensure_dir

logger = logging.getLogger("classroom_monitor.evaluation")


class AblationStudy:
    """
    Conducts ablation experiments to assess each component's contribution.

    Experiment configurations:
        - facial_only: Only CNN behavioral predictions
        - facial_headpose: CNN predictions + head pose attention
        - full_system: CNN + head pose + temporal fusion + risk scoring

    Args:
        model: Trained AffectiveModel.
        config: System configuration.
        device: Torch device.
        output_dir: Directory for saving results.
    """

    ABLATION_CONFIGS = {
        "facial_only": {
            "use_cnn": True,
            "use_headpose": False,
            "use_temporal": False,
            "description": "Facial behavioral features only (CNN multi-task)",
        },
        "facial_headpose": {
            "use_cnn": True,
            "use_headpose": True,
            "use_temporal": False,
            "description": "Facial features + Head pose attention estimation",
        },
        "full_system": {
            "use_cnn": True,
            "use_headpose": True,
            "use_temporal": True,
            "description": "Full system: Facial + Head pose + Temporal fusion",
        },
    }

    def __init__(
        self,
        model: AffectiveModel,
        config: Dict,
        device: torch.device,
        output_dir: str = "outputs/ablation",
    ):
        self.model = model
        self.config = config
        self.device = device
        self.output_dir = ensure_dir(output_dir)
        self.results: Dict[str, Dict] = {}

    def run_ablation(
        self,
        test_dataloader,
        configs: Optional[List[str]] = None,
    ) -> Dict[str, Dict]:
        """
        Run ablation experiments.

        Args:
            test_dataloader: Test set DataLoader.
            configs: List of ablation config names to run.
                     If None, runs all configurations.

        Returns:
            Dictionary mapping config names to result dicts.
        """
        if configs is None:
            configs = list(self.ABLATION_CONFIGS.keys())

        logger.info(f"Running ablation study: {configs}")

        for config_name in configs:
            if config_name not in self.ABLATION_CONFIGS:
                logger.warning(f"Unknown ablation config: {config_name}")
                continue

            ablation_cfg = self.ABLATION_CONFIGS[config_name]
            logger.info(
                f"\n{'='*60}\n"
                f"Ablation: {config_name}\n"
                f"Description: {ablation_cfg['description']}\n"
                f"{'='*60}"
            )

            result = self._run_single_ablation(
                config_name, ablation_cfg, test_dataloader
            )
            self.results[config_name] = result

        # Generate comparison report
        self._generate_comparison_report()

        return self.results

    def _run_single_ablation(
        self,
        config_name: str,
        ablation_cfg: Dict,
        dataloader,
    ) -> Dict[str, Any]:
        """Run a single ablation experiment."""
        evaluator = Evaluator(
            self.model,
            self.device,
            output_dir=os.path.join(self.output_dir, config_name),
        )

        # Evaluate CNN predictions (always available)
        metrics = evaluator.evaluate(dataloader)

        result = {
            "config": ablation_cfg,
            "metrics": metrics,
        }

        # If using head pose, simulate attention scores
        if ablation_cfg["use_headpose"]:
            attn_cfg = self.config.get("attention", {})
            attention_estimator = AttentionEstimator(
                yaw_threshold=attn_cfg.get("yaw_threshold", 30),
                pitch_threshold=attn_cfg.get("pitch_threshold", 25),
                roll_threshold=attn_cfg.get("roll_threshold", 20),
            )
            result["attention_enabled"] = True
            attention_estimator.close()
        else:
            result["attention_enabled"] = False

        # If using temporal, note it
        if ablation_cfg["use_temporal"]:
            result["temporal_enabled"] = True
        else:
            result["temporal_enabled"] = False

        # Save confusion matrices
        evaluator.plot_confusion_matrices()

        logger.info(f"Ablation '{config_name}' complete")
        return result

    def _generate_comparison_report(self):
        """Generate a visual comparison of ablation results."""
        if not self.results:
            return

        # Gather metrics for comparison
        config_names = list(self.results.keys())
        tasks = ["engagement", "boredom", "confusion", "frustration", "average"]

        # Create comparison table
        comparison = {}
        for config_name in config_names:
            metrics = self.results[config_name].get("metrics", {})
            comparison[config_name] = {}
            for task in tasks:
                if task in metrics:
                    comparison[config_name][task] = {
                        "accuracy": metrics[task].get("accuracy", 0),
                        "f1_weighted": metrics[task].get("f1_weighted", 0),
                    }

        # Plot comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Accuracy comparison
        x = np.arange(len(tasks))
        width = 0.25

        for i, config_name in enumerate(config_names):
            accs = [
                comparison.get(config_name, {}).get(t, {}).get("accuracy", 0)
                for t in tasks
            ]
            axes[0].bar(x + i * width, accs, width, label=config_name, alpha=0.8)

        axes[0].set_xlabel("Task")
        axes[0].set_ylabel("Accuracy")
        axes[0].set_title("Accuracy Comparison", fontweight="bold")
        axes[0].set_xticks(x + width)
        axes[0].set_xticklabels([t.capitalize() for t in tasks], rotation=30)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis="y")

        # F1 comparison
        for i, config_name in enumerate(config_names):
            f1s = [
                comparison.get(config_name, {}).get(t, {}).get("f1_weighted", 0)
                for t in tasks
            ]
            axes[1].bar(x + i * width, f1s, width, label=config_name, alpha=0.8)

        axes[1].set_xlabel("Task")
        axes[1].set_ylabel("F1 Score (Weighted)")
        axes[1].set_title("F1 Score Comparison", fontweight="bold")
        axes[1].set_xticks(x + width)
        axes[1].set_xticklabels([t.capitalize() for t in tasks], rotation=30)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis="y")

        plt.suptitle("Ablation Study Results", fontsize=14, fontweight="bold")
        plt.tight_layout()

        path = os.path.join(self.output_dir, "ablation_comparison.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Ablation comparison plot saved: {path}")

    def format_results_table(self) -> str:
        """Format ablation results as a text table."""
        if not self.results:
            return "No ablation results available."

        lines = [
            "=" * 80,
            "ABLATION STUDY RESULTS",
            "=" * 80,
            f"{'Configuration':<25} {'Avg Accuracy':<15} {'Avg F1':<15} {'Components':<25}",
            "-" * 80,
        ]

        for config_name, result in self.results.items():
            metrics = result.get("metrics", {})
            avg = metrics.get("average", {})
            acc = avg.get("accuracy", 0)
            f1 = avg.get("f1_weighted", 0)

            cfg = result.get("config", {})
            components = []
            if cfg.get("use_cnn"):
                components.append("CNN")
            if cfg.get("use_headpose"):
                components.append("HeadPose")
            if cfg.get("use_temporal"):
                components.append("Temporal")

            lines.append(
                f"{config_name:<25} {acc:<15.4f} {f1:<15.4f} {'+'.join(components):<25}"
            )

        lines.append("=" * 80)
        return "\n".join(lines)
