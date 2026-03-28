"""
CLI Training Script.

Trains the multi-task affective behavioral model on DAiSEE dataset.

Usage:
    python scripts/train.py --config config/config.yaml
    python scripts/train.py --config config/config.yaml --backbone resnet18 --epochs 50
    python scripts/train.py --config config/config.yaml --resume checkpoints/checkpoint_epoch_10.pth
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.models.affective_model import AffectiveModel
from src.training.trainer import Trainer, DAiSEEFrameDataset
from src.evaluation.evaluator import Evaluator
from src.utils.helpers import load_config, set_seed, get_device, count_parameters
from src.utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the multi-task affective model"
    )
    parser.add_argument(
        "--config", type=str, default="config/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--backbone", type=str, default=None,
        help="Override backbone model (resnet18 or efficientnet_b0)",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override number of training epochs",
    )
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Override learning rate",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume training from",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    # Apply CLI overrides
    if args.backbone:
        config["model"]["backbone"] = args.backbone
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.lr:
        config["training"]["learning_rate"] = args.lr

    logger = setup_logger(
        name="classroom_monitor",
        log_dir=config["paths"]["logs"],
        level=config["logging"]["level"],
    )

    set_seed(config["dataset"]["random_seed"])
    device = get_device(config.get("inference", {}).get("device", "auto"))
    logger.info(f"Using device: {device}")

    # ---- Load Data ----
    splits_dir = config["paths"]["splits_dir"]

    train_df = pd.read_csv(os.path.join(splits_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(splits_dir, "val.csv"))
    test_df = pd.read_csv(os.path.join(splits_dir, "test.csv"))

    logger.info(
        f"Dataset sizes — Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}"
    )

    # ---- Create Model ----
    model_cfg = config["model"]
    model = AffectiveModel(
        backbone_name=model_cfg["backbone"],
        pretrained=model_cfg["pretrained"],
        embedding_dim=model_cfg["embedding_dim"],
        num_classes={
            task: model_cfg["num_classes"][task]
            for task in AffectiveModel.TASKS
        },
        dropout=model_cfg["dropout"],
    )

    params = count_parameters(model)
    logger.info(
        f"Model parameters — Total: {params['total']:,}, "
        f"Trainable: {params['trainable']:,}, Frozen: {params['frozen']:,}"
    )

    # ---- Create Trainer ----
    trainer = Trainer(model=model, config=config, device=device)

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # ---- Create DataLoaders ----
    train_dataset = DAiSEEFrameDataset(train_df, transform=trainer.get_transforms(True))
    val_dataset = DAiSEEFrameDataset(val_df, transform=trainer.get_transforms(False))
    test_dataset = DAiSEEFrameDataset(test_df, transform=trainer.get_transforms(False))

    dataset_cfg = config["dataset"]
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=dataset_cfg["num_workers"],
        pin_memory=dataset_cfg["pin_memory"],
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=dataset_cfg["num_workers"],
        pin_memory=dataset_cfg["pin_memory"],
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=dataset_cfg["num_workers"],
        pin_memory=dataset_cfg["pin_memory"],
    )

    # ---- Train ----
    logger.info("Starting training...")
    history = trainer.train(train_loader, val_loader)

    # ---- Evaluate on Test Set ----
    logger.info("Evaluating on test set...")
    evaluator = Evaluator(
        model=model,
        device=device,
        output_dir=config["paths"]["outputs"],
    )

    # Load best model
    best_model_path = os.path.join(config["paths"]["checkpoints"], "best_model.pth")
    if os.path.exists(best_model_path):
        trainer.load_checkpoint(best_model_path)

    test_metrics = evaluator.evaluate(test_loader)
    evaluator.plot_confusion_matrices()
    evaluator.plot_training_history(history)

    logger.info("Training and evaluation complete!")
    logger.info(f"Best model saved at: {best_model_path}")


if __name__ == "__main__":
    main()
