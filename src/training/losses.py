"""
Custom Loss Functions for Multi-Task Learning.

Includes weighted multi-task cross-entropy loss with optional
label smoothing and focal loss variants.
"""

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("classroom_monitor.training")


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss combining weighted cross-entropy losses across
    multiple behavioral classification tasks.

    L_total = Σ λ_k · L_CE^(k)

    Args:
        task_weights: Dictionary mapping task names to loss weights (λ_k).
        label_smoothing: Label smoothing factor (0 = no smoothing).
        use_focal: Whether to use focal loss instead of standard CE.
        focal_gamma: Gamma parameter for focal loss.
    """

    TASKS = ["engagement", "boredom", "confusion", "frustration"]

    def __init__(
        self,
        task_weights: Optional[Dict[str, float]] = None,
        label_smoothing: float = 0.0,
        use_focal: bool = False,
        focal_gamma: float = 2.0,
        class_weights: Optional[Dict[str, torch.Tensor]] = None,
    ):
        super().__init__()

        if task_weights is None:
            task_weights = {task: 1.0 for task in self.TASKS}

        self.task_weights = task_weights
        self.label_smoothing = label_smoothing
        self.use_focal = use_focal
        self.focal_gamma = focal_gamma

        # Per-task class weights for handling imbalance
        self.class_weights = class_weights or {}

        # Create loss functions per task
        self.criteria = nn.ModuleDict()
        for task in self.TASKS:
            cw = self.class_weights.get(task, None)
            self.criteria[task] = nn.CrossEntropyLoss(
                weight=cw,
                label_smoothing=label_smoothing,
            )

    def focal_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Focal Loss for handling class imbalance.

        FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

        Args:
            logits: Model output logits (B, C).
            targets: Ground truth labels (B,).
            gamma: Focusing parameter.
            weight: Optional class weights.

        Returns:
            Scalar focal loss.
        """
        ce_loss = F.cross_entropy(logits, targets, weight=weight, reduction="none")
        pt = torch.exp(-ce_loss)
        focal = ((1 - pt) ** gamma) * ce_loss
        return focal.mean()

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss.

        Args:
            predictions: Dict mapping task names to logits (B, C).
            targets: Dict mapping task names to labels (B,).

        Returns:
            Dictionary with:
                - 'total': Total weighted loss (scalar)
                - Per-task losses: 'engagement_loss', 'boredom_loss', etc.
        """
        total_loss = torch.tensor(0.0, device=next(iter(predictions.values())).device)
        losses = {}

        for task in self.TASKS:
            if task not in predictions or task not in targets:
                continue

            logits = predictions[task]
            target = targets[task]

            if self.use_focal:
                cw = self.class_weights.get(task, None)
                task_loss = self.focal_loss(logits, target, self.focal_gamma, cw)
            else:
                task_loss = self.criteria[task](logits, target)

            weight = self.task_weights.get(task, 1.0)
            weighted_loss = weight * task_loss

            losses[f"{task}_loss"] = task_loss.detach()
            total_loss = total_loss + weighted_loss

        losses["total"] = total_loss

        return losses
