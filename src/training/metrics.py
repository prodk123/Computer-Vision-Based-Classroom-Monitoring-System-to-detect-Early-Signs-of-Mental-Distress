"""
Metrics Calculation Module.

Computes accuracy, F1 scores, confusion matrices, and other evaluation
metrics for multi-task behavioral classification.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger("classroom_monitor.training")


class MetricsCalculator:
    """
    Computes and tracks evaluation metrics across multiple tasks.

    Designed for DAiSEE-style multi-task classification with
    4 behavioral dimensions, each having 4 ordinal levels.
    """

    TASKS = ["engagement", "boredom", "confusion", "frustration"]
    LEVEL_NAMES = ["Very Low", "Low", "High", "Very High"]

    def __init__(self):
        self._predictions: Dict[str, List] = {t: [] for t in self.TASKS}
        self._targets: Dict[str, List] = {t: [] for t in self.TASKS}

    def reset(self):
        """Clear accumulated predictions and targets."""
        self._predictions = {t: [] for t in self.TASKS}
        self._targets = {t: [] for t in self.TASKS}

    def update(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ):
        """
        Accumulate batch predictions and targets.

        Args:
            predictions: Dict of logits (B, C) or class indices (B,).
            targets: Dict of target labels (B,).
        """
        for task in self.TASKS:
            if task not in predictions or task not in targets:
                continue

            preds = predictions[task]
            tgts = targets[task]

            # Convert logits to predictions if needed
            if preds.dim() > 1:
                preds = torch.argmax(preds, dim=-1)

            self._predictions[task].extend(preds.cpu().numpy().tolist())
            self._targets[task].extend(tgts.cpu().numpy().tolist())

    def compute_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Compute all metrics for accumulated predictions.

        Returns:
            Nested dictionary: {task_name: {metric_name: value}}.
        """
        results = {}

        for task in self.TASKS:
            preds = self._predictions[task]
            tgts = self._targets[task]

            if not preds or not tgts:
                continue

            preds_arr = np.array(preds)
            tgts_arr = np.array(tgts)

            results[task] = {
                "accuracy": accuracy_score(tgts_arr, preds_arr),
                "f1_weighted": f1_score(
                    tgts_arr, preds_arr, average="weighted", zero_division=0
                ),
                "f1_macro": f1_score(
                    tgts_arr, preds_arr, average="macro", zero_division=0
                ),
                "precision_weighted": precision_score(
                    tgts_arr, preds_arr, average="weighted", zero_division=0
                ),
                "recall_weighted": recall_score(
                    tgts_arr, preds_arr, average="weighted", zero_division=0
                ),
                "num_samples": len(tgts),
            }

        # Compute average across tasks
        if results:
            avg_metrics = {}
            metric_keys = ["accuracy", "f1_weighted", "f1_macro"]
            for metric in metric_keys:
                values = [results[t][metric] for t in results if metric in results[t]]
                avg_metrics[metric] = np.mean(values) if values else 0.0
            results["average"] = avg_metrics

        return results

    def get_confusion_matrices(self) -> Dict[str, np.ndarray]:
        """
        Compute confusion matrices for all tasks.

        Returns:
            Dictionary mapping task names to confusion matrix arrays.
        """
        matrices = {}
        for task in self.TASKS:
            preds = self._predictions[task]
            tgts = self._targets[task]

            if not preds or not tgts:
                continue

            labels = list(range(4))  # 0, 1, 2, 3
            cm = confusion_matrix(tgts, preds, labels=labels)
            matrices[task] = cm

        return matrices

    def get_classification_reports(self) -> Dict[str, str]:
        """
        Get detailed classification reports for all tasks.

        Returns:
            Dictionary mapping task names to report strings.
        """
        reports = {}
        for task in self.TASKS:
            preds = self._predictions[task]
            tgts = self._targets[task]

            if not preds or not tgts:
                continue

            report = classification_report(
                tgts,
                preds,
                labels=[0, 1, 2, 3],
                target_names=self.LEVEL_NAMES,
                zero_division=0,
            )
            reports[task] = report

        return reports

    def format_summary(self) -> str:
        """
        Format metrics as a human-readable summary string.
        """
        metrics = self.compute_metrics()
        lines = ["=" * 60, "Evaluation Results", "=" * 60]

        for task in self.TASKS:
            if task not in metrics:
                continue
            m = metrics[task]
            lines.append(
                f"\n{task.upper()}:"
                f"\n  Accuracy:       {m['accuracy']:.4f}"
                f"\n  F1 (weighted):  {m['f1_weighted']:.4f}"
                f"\n  F1 (macro):     {m['f1_macro']:.4f}"
                f"\n  Precision:      {m['precision_weighted']:.4f}"
                f"\n  Recall:         {m['recall_weighted']:.4f}"
                f"\n  Samples:        {m['num_samples']}"
            )

        if "average" in metrics:
            avg = metrics["average"]
            lines.append(
                f"\nAVERAGE ACROSS TASKS:"
                f"\n  Accuracy:       {avg['accuracy']:.4f}"
                f"\n  F1 (weighted):  {avg['f1_weighted']:.4f}"
                f"\n  F1 (macro):     {avg['f1_macro']:.4f}"
            )

        lines.append("=" * 60)
        return "\n".join(lines)
