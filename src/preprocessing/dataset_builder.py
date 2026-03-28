"""
Dataset Builder Module.

Builds structured DataFrames mapping extracted frames to DAiSEE labels.
Creates train/validation/test splits preserving user-level separation.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

logger = logging.getLogger("classroom_monitor.preprocessing")


class DatasetBuilder:
    """
    Constructs a dataset DataFrame from DAiSEE directory structure and labels.

    DAiSEE structure:
        DAiSEE/
        ├── DataSet/
        │   ├── Train/
        │   │   └── {UserID}/
        │   │       └── {ClipID}.avi
        │   ├── Validation/
        │   └── Test/
        └── Labels/
            ├── TrainLabels.csv
            ├── ValidationLabels.csv
            └── TestLabels.csv

    Label CSV columns: ClipID, Boredom, Engagement, Confusion, Frustration

    Args:
        raw_data_dir: Path to raw DAiSEE dataset root.
        processed_dir: Path to directory containing extracted/cropped frames.
        splits_dir: Path to directory for saving split CSVs.
    """

    # DAiSEE label columns
    LABEL_COLUMNS = ["Engagement", "Boredom", "Confusion", "Frustration"]

    def __init__(
        self,
        raw_data_dir: str,
        processed_dir: str,
        splits_dir: str = "data/splits",
    ):
        self.raw_data_dir = raw_data_dir
        self.processed_dir = processed_dir
        self.splits_dir = splits_dir

    def load_daisee_labels(self) -> pd.DataFrame:
        """
        Load and concatenate DAiSEE label CSV files.

        Returns:
            DataFrame with columns: ClipID, Engagement, Boredom, Confusion,
            Frustration, Split.
        """
        labels_dir = os.path.join(self.raw_data_dir, "Labels")

        splits_map = {
            "TrainLabels.csv": "train",
            "ValidationLabels.csv": "validation",
            "TestLabels.csv": "test",
        }

        all_labels = []
        for filename, split_name in splits_map.items():
            label_path = os.path.join(labels_dir, filename)
            if os.path.exists(label_path):
                df = pd.read_csv(label_path)
                df["OriginalSplit"] = split_name
                all_labels.append(df)
                logger.info(
                    f"Loaded {len(df)} labels from {filename} ({split_name})"
                )
            else:
                logger.warning(f"Label file not found: {label_path}")

        if not all_labels:
            raise FileNotFoundError(
                f"No label files found in {labels_dir}. "
                "Please ensure DAiSEE Labels directory exists."
            )

        combined = pd.concat(all_labels, ignore_index=True)
        combined.columns = combined.columns.str.strip()

        # Clean ClipID — remove extension if present
        if "ClipID" in combined.columns:
            combined["ClipID"] = combined["ClipID"].apply(
                lambda x: str(x).replace(".avi", "").replace(".mp4", "").strip()
            )

        logger.info(f"Total label entries: {len(combined)}")
        return combined

    def build_frame_dataframe(
        self,
        labels_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Build a DataFrame mapping each extracted frame to its video label.

        Args:
            labels_df: Optional pre-loaded labels DataFrame. If None, loads from disk.

        Returns:
            DataFrame with columns: FramePath, VideoID, UserID, ClipID,
            Engagement, Boredom, Confusion, Frustration.
        """
        if labels_df is None:
            labels_df = self.load_daisee_labels()

        # Build a lookup from ClipID to labels
        label_lookup = {}
        for _, row in labels_df.iterrows():
            clip_id = str(row["ClipID"])
            label_lookup[clip_id] = {
                col: int(row[col]) for col in self.LABEL_COLUMNS if col in row
            }

        # Scan processed directory for frames
        records = []
        if not os.path.exists(self.processed_dir):
            logger.warning(
                f"Processed data directory not found: {self.processed_dir}. "
                "Run frame extraction first."
            )
            return pd.DataFrame()

        for root, dirs, files in os.walk(self.processed_dir):
            for fname in sorted(files):
                if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
                    continue

                frame_path = os.path.join(root, fname)

                # Extract video/clip information from path
                rel_path = os.path.relpath(frame_path, self.processed_dir)
                parts = Path(rel_path).parts

                # Try to extract UserID and ClipID from directory structure
                # Expected structure: {split}/{UserID}/{ClipID}/frame_XXXXX.jpg
                user_id = parts[1] if len(parts) > 2 else "unknown"
                clip_id = parts[2] if len(parts) > 3 else parts[0] if parts else "unknown"

                # Also try matching by the video name part of the frame filename
                video_name = fname.split("_frame_")[0] if "_frame_" in fname else clip_id

                # Look up labels
                labels = label_lookup.get(
                    video_name,
                    label_lookup.get(clip_id, None),
                )

                record = {
                    "FramePath": frame_path,
                    "VideoID": video_name,
                    "UserID": user_id,
                    "ClipID": clip_id,
                }

                if labels is not None:
                    record.update(labels)
                else:
                    # Default to zeros if label not found
                    for col in self.LABEL_COLUMNS:
                        record[col] = -1  # -1 indicates missing label

                records.append(record)

        df = pd.DataFrame(records)

        # Filter out frames with missing labels
        valid_mask = df[self.LABEL_COLUMNS].min(axis=1) >= 0
        n_missing = (~valid_mask).sum()
        if n_missing > 0:
            logger.warning(
                f"{n_missing} frames have missing labels and will be excluded"
            )
        df = df[valid_mask].reset_index(drop=True)

        logger.info(
            f"Built frame DataFrame with {len(df)} samples "
            f"from {df['VideoID'].nunique()} videos"
        )

        return df

    def create_splits(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42,
        group_column: str = "UserID",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create train/validation/test splits with user-level separation.

        Ensures no user appears in multiple splits to prevent data leakage.

        Args:
            df: Full dataset DataFrame.
            train_ratio: Fraction for training.
            val_ratio: Fraction for validation.
            test_ratio: Fraction for testing.
            random_seed: Random seed for reproducibility.
            group_column: Column to group by for split separation.

        Returns:
            Tuple of (train_df, val_df, test_df).
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Split ratios must sum to 1.0"

        # First split: separate test set
        gss1 = GroupShuffleSplit(
            n_splits=1,
            test_size=test_ratio,
            random_state=random_seed,
        )

        groups = df[group_column].values
        train_val_idx, test_idx = next(gss1.split(df, groups=groups))

        test_df = df.iloc[test_idx].reset_index(drop=True)
        train_val_df = df.iloc[train_val_idx]

        # Second split: separate validation from training
        relative_val_ratio = val_ratio / (train_ratio + val_ratio)
        gss2 = GroupShuffleSplit(
            n_splits=1,
            test_size=relative_val_ratio,
            random_state=random_seed,
        )

        train_val_groups = train_val_df[group_column].values
        train_idx, val_idx = next(gss2.split(train_val_df, groups=train_val_groups))

        train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
        val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

        logger.info(
            f"Split sizes — Train: {len(train_df)}, "
            f"Val: {len(val_df)}, Test: {len(test_df)}"
        )
        logger.info(
            f"Unique users — Train: {train_df[group_column].nunique()}, "
            f"Val: {val_df[group_column].nunique()}, "
            f"Test: {test_df[group_column].nunique()}"
        )

        return train_df, val_df, test_df

    def save_splits(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> Dict[str, str]:
        """
        Save split DataFrames as CSV files.

        Returns:
            Dictionary mapping split names to file paths.
        """
        os.makedirs(self.splits_dir, exist_ok=True)

        paths = {}
        for name, split_df in [
            ("train", train_df),
            ("val", val_df),
            ("test", test_df),
        ]:
            path = os.path.join(self.splits_dir, f"{name}.csv")
            split_df.to_csv(path, index=False)
            paths[name] = path
            logger.info(f"Saved {name} split ({len(split_df)} samples) to {path}")

        return paths

    def load_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load previously saved split CSVs."""
        splits = []
        for name in ["train", "val", "test"]:
            path = os.path.join(self.splits_dir, f"{name}.csv")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Split file not found: {path}")
            splits.append(pd.read_csv(path))
            logger.info(f"Loaded {name} split: {len(splits[-1])} samples")

        return tuple(splits)

    def get_label_distribution(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute label distribution statistics.

        Args:
            df: Dataset DataFrame.

        Returns:
            DataFrame with value counts for each label column.
        """
        stats = {}
        for col in self.LABEL_COLUMNS:
            if col in df.columns:
                stats[col] = df[col].value_counts().sort_index().to_dict()

        return pd.DataFrame(stats).fillna(0).astype(int)
