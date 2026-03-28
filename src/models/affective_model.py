"""
Affective Behavioral Model (Branch 1).

Multi-task CNN using a pretrained backbone (ResNet18 or EfficientNet-B0)
with separate classification heads for engagement, boredom, confusion,
and frustration prediction.

Produces both per-task class predictions and a shared embedding vector
for downstream temporal modeling.
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torchvision.models as models

logger = logging.getLogger("classroom_monitor.models")


class TaskHead(nn.Module):
    """
    Single classification head for one behavioral dimension.

    Args:
        in_features: Input embedding dimension.
        num_classes: Number of output classes (typically 4 for DAiSEE).
        dropout: Dropout probability.
    """

    def __init__(self, in_features: int, num_classes: int = 4, dropout: float = 0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class AffectiveModel(nn.Module):
    """
    Multi-task affective behavioral classification model.

    Architecture:
        Backbone (ResNet18/EfficientNet-B0)
        → Global Average Pooling
        → Shared Embedding Layer
        → 4 Task-Specific Classification Heads

    Args:
        backbone_name: Name of the pretrained backbone ("resnet18" or "efficientnet_b0").
        pretrained: Whether to use ImageNet-pretrained weights.
        embedding_dim: Dimension of the shared embedding layer.
        num_classes: Dict mapping task names to number of classes.
        dropout: Dropout probability for task heads.
        freeze_backbone: Whether to freeze backbone weights initially.
    """

    TASKS = ["engagement", "boredom", "confusion", "frustration"]

    def __init__(
        self,
        backbone_name: str = "resnet18",
        pretrained: bool = True,
        embedding_dim: int = 256,
        num_classes: Optional[Dict[str, int]] = None,
        dropout: float = 0.3,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        if num_classes is None:
            num_classes = {task: 4 for task in self.TASKS}

        self.backbone_name = backbone_name
        self.embedding_dim = embedding_dim

        # --- Build Backbone ---
        if backbone_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            base_model = models.resnet18(weights=weights)
            backbone_out_features = base_model.fc.in_features
            # Remove the final FC layer
            self.backbone = nn.Sequential(*list(base_model.children())[:-1])

        elif backbone_name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            base_model = models.efficientnet_b0(weights=weights)
            backbone_out_features = base_model.classifier[1].in_features
            # Remove the classifier
            self.backbone = nn.Sequential(
                base_model.features,
                base_model.avgpool,
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # --- Shared Embedding Layer ---
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(backbone_out_features, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
        )

        # --- Task-Specific Heads ---
        self.task_heads = nn.ModuleDict({
            task: TaskHead(embedding_dim, num_classes.get(task, 4), dropout)
            for task in self.TASKS
        })

        # Freeze backbone if requested
        if freeze_backbone:
            self.freeze_backbone()

        logger.info(
            f"AffectiveModel initialized: backbone={backbone_name}, "
            f"embedding_dim={embedding_dim}, tasks={list(num_classes.keys())}"
        )

    def freeze_backbone(self):
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone parameters frozen")

    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info("Backbone parameters unfrozen")

    def forward(
        self,
        x: torch.Tensor,
        return_embedding: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, 3, H, W).
            return_embedding: If True, include the embedding in output dict.

        Returns:
            Dictionary with keys for each task ('engagement', 'boredom',
            'confusion', 'frustration') mapped to logits of shape (B, C).
            Optionally includes 'embedding' of shape (B, embedding_dim).
        """
        # Backbone feature extraction
        features = self.backbone(x)

        # Shared embedding
        embedding = self.embedding(features)

        # Task predictions
        outputs = {}
        for task_name, head in self.task_heads.items():
            outputs[task_name] = head(embedding)

        if return_embedding:
            outputs["embedding"] = embedding

        return outputs

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract only the embedding vector (for temporal modeling).

        Args:
            x: Input tensor of shape (B, 3, H, W).

        Returns:
            Embedding tensor of shape (B, embedding_dim).
        """
        features = self.backbone(x)
        embedding = self.embedding(features)
        return embedding

    def get_predictions(
        self, x: torch.Tensor
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get both class predictions and probabilities.

        Args:
            x: Input tensor of shape (B, 3, H, W).

        Returns:
            Dictionary mapping task names to (predicted_class, probabilities).
        """
        outputs = self.forward(x)
        predictions = {}

        for task_name in self.TASKS:
            logits = outputs[task_name]
            probs = torch.softmax(logits, dim=-1)
            pred_class = torch.argmax(probs, dim=-1)
            predictions[task_name] = (pred_class, probs)

        return predictions
