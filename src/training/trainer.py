"""
Training Pipeline.

Implements the full training loop for the multi-task affective model
with support for learning rate scheduling, gradient clipping,
early stopping, checkpoint saving, and backbone freeze/unfreeze.
"""

import logging
import os
import time
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
from tqdm import tqdm

from src.models.affective_model import AffectiveModel
from src.training.losses import MultiTaskLoss
from src.training.metrics import MetricsCalculator
from src.utils.helpers import AverageMeter, ensure_dir

logger = logging.getLogger("classroom_monitor.training")


class DAiSEEFrameDataset(Dataset):
    """
    PyTorch Dataset for DAiSEE frames.

    Loads face-cropped images and returns them with multi-task labels.

    Args:
        dataframe: Pandas DataFrame with columns 'FramePath',
                   'Engagement', 'Boredom', 'Confusion', 'Frustration'.
        transform: Optional torchvision transforms to apply.
    """

    TASKS = ["Engagement", "Boredom", "Confusion", "Frustration"]

    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.dropna(subset=self.TASKS).reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        import cv2
        from PIL import Image

        row = self.dataframe.iloc[idx]
        img_path = row["FramePath"]

        # Load image
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        # Labels (lowercase keys for model compatibility)
        labels = {
            "engagement": torch.tensor(int(row["Engagement"]), dtype=torch.long),
            "boredom": torch.tensor(int(row["Boredom"]), dtype=torch.long),
            "confusion": torch.tensor(int(row["Confusion"]), dtype=torch.long),
            "frustration": torch.tensor(int(row["Frustration"]), dtype=torch.long),
        }

        return image, labels


class Trainer:
    """
    Training manager for the multi-task affective model.

    Handles the complete training lifecycle including:
    - Training and validation loops
    - Learning rate scheduling
    - Gradient clipping
    - Early stopping
    - Checkpoint saving/loading
    - Backbone freeze/unfreeze scheduling

    Args:
        model: AffectiveModel instance.
        config: Training configuration dictionary.
        device: Torch device for computation.
    """

    def __init__(
        self,
        model: AffectiveModel,
        config: Dict[str, Any],
        device: torch.device,
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device

        # Training config
        train_cfg = config.get("training", {})
        self.epochs = train_cfg.get("epochs", 50)
        self.lr = train_cfg.get("learning_rate", 0.001)
        self.weight_decay = train_cfg.get("weight_decay", 0.0001)
        self.gradient_clip = train_cfg.get("gradient_clip", 1.0)
        self.patience = train_cfg.get("patience", 7)
        self.save_every = train_cfg.get("save_every", 5)

        model_cfg = config.get("model", {})
        self.freeze_backbone_epochs = model_cfg.get("freeze_backbone_epochs", 5)

        # Loss function
        task_weights = train_cfg.get("task_weights", {})
        self.criterion = MultiTaskLoss(task_weights=task_weights)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # Scheduler
        scheduler_name = train_cfg.get("scheduler", "cosine")
        if scheduler_name == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=self.epochs
            )
        elif scheduler_name == "step":
            self.scheduler = StepLR(
                self.optimizer,
                step_size=train_cfg.get("step_size", 10),
                gamma=train_cfg.get("gamma", 0.1),
            )
        elif scheduler_name == "plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode="min", patience=3, factor=0.5
            )
        else:
            self.scheduler = None

        # Metrics
        self.metrics_calc = MetricsCalculator()

        # Checkpoint directory
        self.checkpoint_dir = ensure_dir(
            config.get("paths", {}).get("checkpoints", "checkpoints")
        )

        # Training state
        self.best_val_loss = float("inf")
        self.best_val_f1 = 0.0
        self.epochs_without_improvement = 0
        self.current_epoch = 0
        self.train_history = []

    def get_transforms(self, is_training: bool = True):
        """Get image transforms for training or evaluation."""
        from torchvision import transforms

        if is_training:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.1
                ),
                transforms.RandomGrayscale(p=0.05),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
        else:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Run a single training epoch.

        Args:
            dataloader: Training DataLoader.

        Returns:
            Dictionary of average training metrics.
        """
        self.model.train()
        loss_meter = AverageMeter("train_loss")
        task_loss_meters = {
            f"{task}_loss": AverageMeter(f"{task}_loss")
            for task in AffectiveModel.TASKS
        }
        self.metrics_calc.reset()

        pbar = tqdm(dataloader, desc=f"Epoch {self.current_epoch} [Train]")
        for images, labels in pbar:
            images = images.to(self.device)
            targets = {k: v.to(self.device) for k, v in labels.items()}

            # Forward pass
            predictions = self.model(images)
            losses = self.criterion(predictions, targets)
            total_loss = losses["total"]

            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()

            # Gradient clipping
            if self.gradient_clip > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip
                )

            self.optimizer.step()

            # Update meters
            batch_size = images.size(0)
            loss_meter.update(total_loss.item(), batch_size)
            for task in AffectiveModel.TASKS:
                key = f"{task}_loss"
                if key in losses:
                    task_loss_meters[key].update(losses[key].item(), batch_size)

            # Update metrics
            self.metrics_calc.update(predictions, targets)

            pbar.set_postfix(loss=f"{loss_meter.avg:.4f}")

        metrics = self.metrics_calc.compute_metrics()
        results = {"loss": loss_meter.avg}
        for key, meter in task_loss_meters.items():
            results[key] = meter.avg
        if "average" in metrics:
            results["accuracy"] = metrics["average"].get("accuracy", 0.0)
            results["f1_weighted"] = metrics["average"].get("f1_weighted", 0.0)

        return results

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Run validation.

        Args:
            dataloader: Validation DataLoader.

        Returns:
            Dictionary of validation metrics.
        """
        self.model.eval()
        loss_meter = AverageMeter("val_loss")
        self.metrics_calc.reset()

        for images, labels in tqdm(dataloader, desc=f"Epoch {self.current_epoch} [Val]"):
            images = images.to(self.device)
            targets = {k: v.to(self.device) for k, v in labels.items()}

            predictions = self.model(images)
            losses = self.criterion(predictions, targets)

            loss_meter.update(losses["total"].item(), images.size(0))
            self.metrics_calc.update(predictions, targets)

        metrics = self.metrics_calc.compute_metrics()
        results = {"loss": loss_meter.avg}
        if "average" in metrics:
            results["accuracy"] = metrics["average"].get("accuracy", 0.0)
            results["f1_weighted"] = metrics["average"].get("f1_weighted", 0.0)
            results["f1_macro"] = metrics["average"].get("f1_macro", 0.0)

        return results

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Dict[str, list]:
        """
        Full training loop with validation, scheduling, and checkpointing.

        Args:
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.

        Returns:
            Dictionary of training history lists.
        """
        logger.info(f"Starting training for {self.epochs} epochs")
        logger.info(f"Device: {self.device}")

        # Freeze backbone initially if configured, or unfreeze if resuming past threshold
        if self.freeze_backbone_epochs > 0:
            if self.current_epoch < self.freeze_backbone_epochs:
                self.model.freeze_backbone()
                logger.info(
                    f"Backbone frozen for first {self.freeze_backbone_epochs} epochs"
                )
            else:
                self.model.unfreeze_backbone()
                logger.info("Resuming training with unfrozen backbone")

        history = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [],
            "train_f1": [], "val_f1": [],
            "lr": [],
        }

        start_epoch = self.current_epoch
        for epoch in range(start_epoch, self.epochs):
            self.current_epoch = epoch + 1
            epoch_start = time.time()

            # Unfreeze backbone after specified epochs
            if epoch == self.freeze_backbone_epochs and self.freeze_backbone_epochs > 0:
                self.model.unfreeze_backbone()
                logger.info("Backbone unfrozen — fine-tuning all layers")

            # Train
            train_results = self.train_epoch(train_loader)

            # Validate
            val_results = self.validate(val_loader)

            # Learning rate scheduling
            current_lr = self.optimizer.param_groups[0]["lr"]
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_results["loss"])
            elif self.scheduler is not None:
                self.scheduler.step()

            # Record history
            history["train_loss"].append(train_results["loss"])
            history["val_loss"].append(val_results["loss"])
            history["train_acc"].append(train_results.get("accuracy", 0.0))
            history["val_acc"].append(val_results.get("accuracy", 0.0))
            history["train_f1"].append(train_results.get("f1_weighted", 0.0))
            history["val_f1"].append(val_results.get("f1_weighted", 0.0))
            history["lr"].append(current_lr)

            elapsed = time.time() - epoch_start

            logger.info(
                f"Epoch {self.current_epoch}/{self.epochs} "
                f"({elapsed:.1f}s) — "
                f"Train Loss: {train_results['loss']:.4f}, "
                f"Val Loss: {val_results['loss']:.4f}, "
                f"Val Acc: {val_results.get('accuracy', 0):.4f}, "
                f"Val F1: {val_results.get('f1_weighted', 0):.4f}, "
                f"LR: {current_lr:.6f}"
            )

            # Checkpointing — save best model
            val_f1 = val_results.get("f1_weighted", 0.0)
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.best_val_loss = val_results["loss"]
                self.epochs_without_improvement = 0
                self.save_checkpoint("best_model.pth", val_results)
                logger.info(f"  ★ New best model (F1: {val_f1:.4f})")
            else:
                self.epochs_without_improvement += 1

            # Periodic checkpoint
            if self.current_epoch % self.save_every == 0:
                self.save_checkpoint(
                    f"checkpoint_epoch_{self.current_epoch}.pth", val_results
                )

            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                logger.info(
                    f"Early stopping at epoch {self.current_epoch} "
                    f"(no improvement for {self.patience} epochs)"
                )
                break

        # Save final checkpoint
        self.save_checkpoint("final_model.pth", val_results)
        logger.info(
            f"Training complete. Best Val F1: {self.best_val_f1:.4f}"
        )

        self.train_history = history
        return history

    def save_checkpoint(
        self,
        filename: str,
        val_metrics: Optional[Dict] = None,
    ):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_f1": self.best_val_f1,
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }
        if val_metrics:
            checkpoint["val_metrics"] = val_metrics
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, path)
        logger.debug(f"Checkpoint saved: {path}")

    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint.get("epoch", 0)
        self.best_val_f1 = checkpoint.get("best_val_f1", 0.0)
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        logger.info(f"Checkpoint loaded: {filepath} (epoch {self.current_epoch})")
