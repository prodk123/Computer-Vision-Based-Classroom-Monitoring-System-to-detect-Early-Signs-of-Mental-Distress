"""
Evaluation Module.

Comprehensive evaluation including:
- Per-task metrics computation
- Confusion matrix visualization
- Temporal case study visualization
- Training history plots
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.utils.data import DataLoader

from src.models.affective_model import AffectiveModel
from src.training.metrics import MetricsCalculator
from src.utils.helpers import ensure_dir

logger = logging.getLogger("classroom_monitor.evaluation")


class Evaluator:
    """
    Comprehensive model evaluation with visualization.

    Args:
        model: Trained AffectiveModel.
        device: Torch device.
        output_dir: Directory for saving evaluation outputs.
    """

    TASKS = ["engagement", "boredom", "confusion", "frustration"]
    LEVEL_NAMES = ["Very Low", "Low", "High", "Very High"]

    def __init__(
        self,
        model: AffectiveModel,
        device: torch.device,
        output_dir: str = "outputs",
    ):
        self.model = model.to(device)
        self.device = device
        self.output_dir = ensure_dir(output_dir)
        self.metrics_calc = MetricsCalculator()

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, Any]:
        """
        Run full evaluation on a dataset.

        Args:
            dataloader: DataLoader for evaluation.

        Returns:
            Dictionary of all evaluation metrics.
        """
        self.model.eval()
        self.metrics_calc.reset()

        for images, labels in dataloader:
            images = images.to(self.device)
            targets = {k: v.to(self.device) for k, v in labels.items()}

            predictions = self.model(images)
            self.metrics_calc.update(predictions, targets)

        metrics = self.metrics_calc.compute_metrics()
        logger.info(self.metrics_calc.format_summary())

        return metrics

    def plot_confusion_matrices(
        self,
        save: bool = True,
        figsize: Tuple[int, int] = (16, 4),
    ) -> Optional[plt.Figure]:
        """
        Plot confusion matrices for all tasks.

        Args:
            save: Whether to save the figure.
            figsize: Figure size.

        Returns:
            Matplotlib figure, or None if data is unavailable.
        """
        matrices = self.metrics_calc.get_confusion_matrices()

        if not matrices:
            logger.warning("No confusion matrix data available. Run evaluate() first.")
            return None

        n_tasks = len(matrices)
        fig, axes = plt.subplots(1, n_tasks, figsize=figsize)

        if n_tasks == 1:
            axes = [axes]

        for ax, (task, cm) in zip(axes, matrices.items()):
            # Normalize
            cm_normalized = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

            sns.heatmap(
                cm_normalized,
                annot=True,
                fmt=".2f",
                cmap="Blues",
                xticklabels=self.LEVEL_NAMES,
                yticklabels=self.LEVEL_NAMES,
                ax=ax,
                vmin=0, vmax=1,
            )
            ax.set_title(f"{task.capitalize()}", fontsize=12, fontweight="bold")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")

        plt.suptitle("Confusion Matrices (Normalized)", fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save:
            path = os.path.join(self.output_dir, "confusion_matrices.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            logger.info(f"Confusion matrices saved: {path}")

        return fig

    def plot_training_history(
        self,
        history: Dict[str, list],
        save: bool = True,
    ) -> plt.Figure:
        """
        Plot training history curves.

        Args:
            history: Dictionary from Trainer.train() containing
                     train/val loss, accuracy, F1, learning rate.
            save: Whether to save the figure.

        Returns:
            Matplotlib figure.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        epochs = list(range(1, len(history["train_loss"]) + 1))

        # Loss
        axes[0, 0].plot(epochs, history["train_loss"], "b-", label="Train", linewidth=2)
        axes[0, 0].plot(epochs, history["val_loss"], "r-", label="Val", linewidth=2)
        axes[0, 0].set_title("Loss", fontweight="bold")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Accuracy
        axes[0, 1].plot(epochs, history["train_acc"], "b-", label="Train", linewidth=2)
        axes[0, 1].plot(epochs, history["val_acc"], "r-", label="Val", linewidth=2)
        axes[0, 1].set_title("Accuracy", fontweight="bold")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # F1
        axes[1, 0].plot(epochs, history["train_f1"], "b-", label="Train", linewidth=2)
        axes[1, 0].plot(epochs, history["val_f1"], "r-", label="Val", linewidth=2)
        axes[1, 0].set_title("F1 Score (Weighted)", fontweight="bold")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Learning Rate
        axes[1, 1].plot(epochs, history["lr"], "g-", linewidth=2)
        axes[1, 1].set_title("Learning Rate", fontweight="bold")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_yscale("log")
        axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle("Training History", fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save:
            path = os.path.join(self.output_dir, "training_history.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            logger.info(f"Training history plot saved: {path}")

        return fig

    def plot_temporal_case_study(
        self,
        student_states: List[Dict],
        save: bool = True,
        title: str = "Temporal Behavioral Case Study",
    ) -> plt.Figure:
        """
        Visualize temporal behavioral patterns for a single student.

        Args:
            student_states: List of student state dictionaries over time.
            save: Whether to save the figure.
            title: Plot title.

        Returns:
            Matplotlib figure.
        """
        if not student_states:
            logger.warning("No student states provided for case study")
            return None

        timestamps = list(range(len(student_states)))

        # Extract signals
        engagement = [s["predictions"].get("engagement", 0) for s in student_states]
        boredom = [s["predictions"].get("boredom", 0) for s in student_states]
        confusion = [s["predictions"].get("confusion", 0) for s in student_states]
        frustration = [s["predictions"].get("frustration", 0) for s in student_states]

        attention = [
            s["attention"].get("attention_score", 0.5) or 0.5
            for s in student_states
        ]
        risk_scores = [s["risk"].get("risk_score", 0) for s in student_states]

        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

        # Panel 1: Behavioral predictions
        axes[0].plot(timestamps, engagement, "-o", label="Engagement", markersize=2, linewidth=1.5)
        axes[0].plot(timestamps, boredom, "-s", label="Boredom", markersize=2, linewidth=1.5)
        axes[0].plot(timestamps, confusion, "-^", label="Confusion", markersize=2, linewidth=1.5)
        axes[0].plot(timestamps, frustration, "-d", label="Frustration", markersize=2, linewidth=1.5)
        axes[0].set_ylabel("Level (0–3)")
        axes[0].set_title("Behavioral Predictions Over Time", fontweight="bold")
        axes[0].legend(loc="upper right")
        axes[0].set_ylim(-0.2, 3.2)
        axes[0].grid(True, alpha=0.3)

        # Panel 2: Attention score
        axes[1].fill_between(timestamps, attention, alpha=0.3, color="blue")
        axes[1].plot(timestamps, attention, "b-", linewidth=1.5, label="Attention Score")
        axes[1].axhline(y=0.5, color="r", linestyle="--", alpha=0.5, label="Off-task threshold")
        axes[1].set_ylabel("Attention Score")
        axes[1].set_title("Attention Level", fontweight="bold")
        axes[1].set_ylim(-0.05, 1.05)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Panel 3: Risk score
        axes[2].fill_between(timestamps, risk_scores, alpha=0.3, color="red")
        axes[2].plot(timestamps, risk_scores, "r-", linewidth=1.5, label="Risk Score")
        axes[2].axhline(y=0.65, color="orange", linestyle="--", alpha=0.7, label="Alert threshold")
        axes[2].set_ylabel("Risk Score")
        axes[2].set_xlabel("Frame")
        axes[2].set_title("Risk Indicator Score", fontweight="bold")
        axes[2].set_ylim(-0.05, 1.05)
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        # Highlight alert regions
        for i, state in enumerate(student_states):
            if state["risk"].get("alert_active"):
                for ax in axes:
                    ax.axvspan(i - 0.5, i + 0.5, alpha=0.1, color="red")

        plt.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save:
            path = os.path.join(self.output_dir, "temporal_case_study.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            logger.info(f"Temporal case study saved: {path}")

        return fig
