"""
CLI Preprocessing Script.

Extracts frames from DAiSEE videos, detects and crops faces,
and builds the dataset DataFrame with train/val/test splits.

Usage:
    python scripts/preprocess.py --config config/config.yaml
    python scripts/preprocess.py --config config/config.yaml --steps extract crop build split
"""

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.helpers import load_config, set_seed
from src.utils.logger import setup_logger
from src.preprocessing.frame_extractor import FrameExtractor
from src.preprocessing.face_detector import FaceDetector
from src.preprocessing.dataset_builder import DatasetBuilder


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess DAiSEE dataset for classroom monitoring"
    )
    parser.add_argument(
        "--config", type=str, default="config/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--steps", nargs="+",
        default=["extract", "crop", "build", "split"],
        choices=["extract", "crop", "build", "split"],
        help="Preprocessing steps to run",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    logger = setup_logger(
        name="classroom_monitor",
        log_dir=config["paths"]["logs"],
        level=config["logging"]["level"],
    )

    set_seed(config["dataset"]["random_seed"])

    paths = config["paths"]
    preproc = config["preprocessing"]

    raw_data_dir = paths["raw_data"]
    processed_dir = paths["processed_data"]
    splits_dir = paths["splits_dir"]

    # Step 1: Extract frames from videos
    if "extract" in args.steps:
        logger.info("=" * 60)
        logger.info("STEP 1: Extracting frames from videos")
        logger.info("=" * 60)

        extractor = FrameExtractor(
            target_fps=preproc["target_fps"],
            output_size=None,  # Don't resize during extraction
        )

        dataset_dir = os.path.join(raw_data_dir, "DataSet")
        if os.path.exists(dataset_dir):
            frames_dir = os.path.join(processed_dir, "frames")
            extractor.extract_from_directory(dataset_dir, frames_dir)
        else:
            logger.warning(
                f"DAiSEE DataSet directory not found: {dataset_dir}\n"
                "Please download the DAiSEE dataset and place it at:\n"
                f"  {raw_data_dir}/DataSet/{{Train,Validation,Test}}/..."
            )

    # Step 2: Detect and crop faces
    if "crop" in args.steps:
        logger.info("=" * 60)
        logger.info("STEP 2: Detecting and cropping faces")
        logger.info("=" * 60)

        detector = FaceDetector(
            min_confidence=preproc["face_min_confidence"],
            crop_size=tuple(preproc["face_crop_size"]),
            padding=preproc["face_padding"],
        )

        frames_dir = os.path.join(processed_dir, "frames")
        faces_dir = os.path.join(processed_dir, "faces")

        if os.path.exists(frames_dir):
            import cv2
            from tqdm import tqdm

            frame_files = []
            for root, _, files in os.walk(frames_dir):
                for f in files:
                    if f.lower().endswith((".jpg", ".png", ".jpeg")):
                        frame_files.append(os.path.join(root, f))

            logger.info(f"Processing {len(frame_files)} frames for face detection")

            for frame_path in tqdm(frame_files, desc="Cropping faces"):
                rel_path = os.path.relpath(frame_path, frames_dir)
                output_subdir = os.path.join(
                    faces_dir, os.path.dirname(rel_path)
                )
                prefix = os.path.splitext(os.path.basename(frame_path))[0]
                detector.crop_and_save(frame_path, output_subdir, prefix)

            detector.close()
        else:
            logger.warning(f"Frames directory not found: {frames_dir}")

    # Step 3: Build dataset DataFrame
    if "build" in args.steps:
        logger.info("=" * 60)
        logger.info("STEP 3: Building dataset DataFrame")
        logger.info("=" * 60)

        builder = DatasetBuilder(
            raw_data_dir=raw_data_dir,
            processed_dir=os.path.join(processed_dir, "faces"),
            splits_dir=splits_dir,
        )

        labels_df = builder.load_daisee_labels()
        df = builder.build_frame_dataframe(labels_df)

        if len(df) > 0:
            dist = builder.get_label_distribution(df)
            logger.info(f"Label distribution:\n{dist}")
        else:
            logger.warning("No frames with valid labels found")

    # Step 4: Create train/val/test splits
    if "split" in args.steps:
        logger.info("=" * 60)
        logger.info("STEP 4: Creating data splits")
        logger.info("=" * 60)

        builder = DatasetBuilder(
            raw_data_dir=raw_data_dir,
            processed_dir=os.path.join(processed_dir, "faces"),
            splits_dir=splits_dir,
        )

        labels_df = builder.load_daisee_labels()
        df = builder.build_frame_dataframe(labels_df)

        if len(df) > 0:
            dataset_cfg = config["dataset"]
            train_df, val_df, test_df = builder.create_splits(
                df,
                train_ratio=dataset_cfg["train_ratio"],
                val_ratio=dataset_cfg["val_ratio"],
                test_ratio=dataset_cfg["test_ratio"],
                random_seed=dataset_cfg["random_seed"],
            )
            builder.save_splits(train_df, val_df, test_df)
        else:
            logger.error("Cannot create splits — no valid data found")

    logger.info("Preprocessing complete!")


if __name__ == "__main__":
    main()
