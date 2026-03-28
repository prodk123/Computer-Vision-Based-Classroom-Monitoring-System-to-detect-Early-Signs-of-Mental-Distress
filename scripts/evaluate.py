"""
CLI Evaluation Script.

Evaluates a trained model on test data and optionally runs ablation studies.

Usage:
    python scripts/evaluate.py --config config/config.yaml --checkpoint checkpoints/best_model.pth
    python scripts/evaluate.py --config config/config.yaml --checkpoint checkpoints/best_model.pth --ablation
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.models.affective_model import AffectiveModel
from src.training.trainer import DAiSEEFrameDataset, Trainer
from src.evaluation.evaluator import Evaluator
from src.evaluation.ablation import AblationStudy
from src.utils.helpers import load_config, set_seed, get_device
from src.utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument(
        "--config", type=str, default="config/config.yaml",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--ablation", action="store_true",
        help="Run ablation study",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Override output directory",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    logger = setup_logger(
        name="classroom_monitor",
        log_dir=config["paths"]["logs"],
    )

    set_seed(config["dataset"]["random_seed"])
    device = get_device(config.get("inference", {}).get("device", "auto"))
    output_dir = args.output_dir or config["paths"]["outputs"]

    # Load model
    model_cfg = config["model"]
    model = AffectiveModel(
        backbone_name=model_cfg["backbone"],
        pretrained=False,
        embedding_dim=model_cfg["embedding_dim"],
        dropout=model_cfg["dropout"],
    )

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    logger.info(f"Model loaded from {args.checkpoint}")

    # Load test data
    test_df = pd.read_csv(os.path.join(config["paths"]["splits_dir"], "test.csv"))

    # Create dummy trainer to get transforms
    temp_trainer = Trainer(model=model, config=config, device=device)
    test_dataset = DAiSEEFrameDataset(
        test_df, transform=temp_trainer.get_transforms(False)
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["dataset"]["num_workers"],
    )

    # Evaluate
    evaluator = Evaluator(model, device, output_dir)
    metrics = evaluator.evaluate(test_loader)
    evaluator.plot_confusion_matrices()

    # Print classification reports
    reports = evaluator.metrics_calc.get_classification_reports()
    for task, report in reports.items():
        logger.info(f"\n{task.upper()} Classification Report:\n{report}")

    # Ablation study
    if args.ablation:
        logger.info("\nRunning ablation study...")
        ablation = AblationStudy(
            model=model,
            config=config,
            device=device,
            output_dir=os.path.join(output_dir, "ablation"),
        )
        ablation.run_ablation(test_loader)
        logger.info(ablation.format_results_table())

    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
